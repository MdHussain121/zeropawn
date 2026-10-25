"""Supervised warm start training script for ZeroPawn.

Contract Specifications:
- Model: PyTorch dual-head ResNet (10 ResBlocks, 128 filters, channels-last)
- Optimizer: AdamW
- Schedule: Linear warmup (few hundred steps) + Cosine decay
- Policy Loss: Cross-Entropy with 0.05 label smoothing
- Total Loss: policy_loss + 0.5 * wdl_loss
- Value Mask: Positions with ply < 10 have zero value loss
- Mixed Precision: bfloat16 autocast
- Checkpointing: Resumable checkpoints every 30 minutes / configured steps
- Logging: TensorBoard metrics (losses, top-1, top-3, policy entropy, GPU utilization)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from contextlib import nullcontext
import sys
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.cli.common import setup_logging
from python.common.progress import (
    artifacts_list,
    banner,
    emit_error,
    emit_report,
    error_block,
    file_record,
    format_kv_line,
    header_block,
    parameters_table,
    run_context,
    summary_block,
)
from python.data.dataset import (
    BinaryChessDataset,
    GpuResidentDataLoader,
    SourceAwareCombinedDataset,
)
from python.model.chess_net import ChessNet

logger = logging.getLogger(__name__)


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    """Write torch state atomically via tmp file + os.replace, then fsync dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        logger.debug("Could not open dir for fsync: %s", path.parent)
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("fsync failed for dir %s", path.parent)
    finally:
        os.close(fd)


def _safe_numpy_rng_state(state: Optional[tuple] = None) -> tuple:
    s = np.random.get_state() if state is None else state
    if isinstance(s, tuple) and len(s) == 5:
        keys = s[1].tolist() if hasattr(s[1], "tolist") else list(s[1])
        return (str(s[0]), keys, int(s[2]), int(s[3]), float(s[4]))
    return s


def _restore_numpy_rng_state(s: Any) -> None:
    if isinstance(s, (tuple, list)) and len(s) == 5:
        keys = np.array(s[1], dtype=np.uint32)
        np.random.set_state((str(s[0]), keys, int(s[2]), int(s[3]), float(s[4])))
    elif isinstance(s, tuple):
        np.random.set_state(s)


def _validate_arch(
    checkpoint: dict[str, Any],
    model: nn.Module,
    num_blocks: int,
    num_filters: int,
    path: str | Path,
) -> None:
    """Friendly arch check: ckpt header vs expected args and state_dict."""
    ckpt_blocks = checkpoint.get("num_blocks", None)
    ckpt_filters = checkpoint.get("num_filters", None)
    if ckpt_blocks is not None and int(ckpt_blocks) != int(num_blocks):
        raise RuntimeError(
            f"Architecture mismatch loading '{path}': checkpoint num_blocks={ckpt_blocks} "
            f"!= requested num_blocks={num_blocks}. Resume with a matching --num-blocks."
        )
    if ckpt_filters is not None and int(ckpt_filters) != int(num_filters):
        raise RuntimeError(
            f"Architecture mismatch loading '{path}': checkpoint num_filters={ckpt_filters} "
            f"!= requested num_filters={num_filters}. Resume with a matching --num-filters."
        )
    state = checkpoint.get("model")
    if isinstance(state, dict):
        block_ids = set()
        for k in state.keys():
            if k.startswith("blocks."):
                try:
                    block_ids.add(int(str(k).split(".")[1]))
                except (ValueError, IndexError):
                    continue
        if block_ids and len(block_ids) != int(num_blocks):
            raise RuntimeError(
                f"Checkpoint '{path}' is inconsistent: expected {num_blocks} residual "
                f"blocks but state_dict contains {len(block_ids)}."
            )


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-5,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(warmup_steps + 1, total_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.base_lr = state["base_lr"]
        self.min_lr = state["min_lr"]
        self.current_step = state["current_step"]


def compute_accuracy_topk(
    logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 3)
) -> list[float]:
    """Compute top-k classification accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / targets.size(0)).item()))
        return res


def compute_policy_entropy(logits: torch.Tensor) -> float:
    """Compute mean Shannon entropy in nats across batch."""
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return float(entropy.item())


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    use_autocast: bool = True,
    accum_steps: int = 1,
    step_optimizer: bool = True,
    zero_grad: bool = True,
    v_total: Optional[int] = None,
) -> tuple[float, float, float, float, float, float, bool]:
    """Single optimization step or accumulation micro-step. Returns (total_loss, policy_loss, wdl_loss, top1, top3, entropy)."""
    device = torch.device(device)
    device_type: str = device.type
    planes = batch["planes"].to(device, memory_format=torch.channels_last, non_blocking=True)
    moves = batch["move"].to(device, non_blocking=True)
    value_targets = batch["value_target"].to(device, non_blocking=True)
    plies = batch["ply"].to(device, non_blocking=True)

    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    autocast_context = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_autocast
        else nullcontext()
    )

    with autocast_context:
        policy_logits, wdl_logits = model(planes)

        # Policy loss: Cross-Entropy with 0.05 label smoothing
        policy_loss = F.cross_entropy(policy_logits, moves, label_smoothing=0.05)

        # Value loss: Cross-Entropy with soft WDL targets + mask for first 10 plies
        log_wdl = F.log_softmax(wdl_logits, dim=-1)
        ce_wdl = -(value_targets * log_wdl).sum(dim=-1)

        # Mask ply < 10 (no value loss) and tactical samples with untrusted value targets ([0, 0, 0])
        val_mask = ((plies >= 10) & (value_targets.sum(dim=-1) > 0)).float()
        valid_count = val_mask.sum()  # Keep as GPU tensor; convert only once per window
        valid_sum = (ce_wdl * val_mask).sum()

        if v_total is not None:
            norm_v = float(v_total)
        else:
            # Defer sync: keep as float only when needed for division
            norm_v_float = float(valid_count) * accum_steps
            norm_v = norm_v_float

        if norm_v > 0.0:
            wdl_loss_scaled = 0.5 * (valid_sum / norm_v)
        else:
            wdl_loss_scaled = torch.tensor(0.0, device=device)

        policy_loss_scaled = policy_loss / accum_steps if accum_steps > 1 else policy_loss
        loss_to_backward = policy_loss_scaled + wdl_loss_scaled

        # Metrics for reporting
        unscaled_wdl = (
            (valid_sum / valid_count) if valid_count > 0 else torch.tensor(0.0, device=device)
        )
        unscaled_total = policy_loss + 0.5 * unscaled_wdl

    if not torch.isfinite(loss_to_backward.detach()):
        logger.warning("Non-finite loss; skipping optimizer step")
        if zero_grad:
            optimizer.zero_grad(set_to_none=True)
        top1, top3 = compute_accuracy_topk(policy_logits, moves, topk=(1, 3))
        entropy = compute_policy_entropy(policy_logits)
        return (
            float(unscaled_total.item()),
            float(policy_loss.item()),
            float(unscaled_wdl.item()),
            top1,
            top3,
            entropy,
            True,  # non-finite: skip window optimizer step
        )

    loss_to_backward.backward()

    if step_optimizer:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    top1, top3 = compute_accuracy_topk(policy_logits, moves, topk=(1, 3))
    entropy = compute_policy_entropy(policy_logits)

    return (
        float(unscaled_total.item()),
        float(policy_loss.item()),
        float(unscaled_wdl.item()),
        top1,
        top3,
        entropy,
        False,  # finite loss: proceed normally
    )


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str | torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate model on validation dataset."""
    device = torch.device(device)
    model.eval()
    total_samples = 0
    sum_policy_loss = 0.0
    sum_wdl_loss = 0.0
    sum_wdl_masked_samples = 0
    correct_top1 = 0
    correct_top3 = 0
    sum_entropy = 0.0

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            planes = batch["planes"].to(
                device, memory_format=torch.channels_last, non_blocking=True
            )
            moves = batch["move"].to(device, non_blocking=True)
            value_targets = batch["value_target"].to(device, non_blocking=True)
            plies = batch["ply"].to(device, non_blocking=True)
            batch_sz = moves.size(0)

            autocast_context = (
                torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16
                )
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                policy_logits, wdl_logits = model(planes)
                policy_loss = F.cross_entropy(
                    policy_logits, moves, label_smoothing=0.05, reduction="sum"
                )
                log_wdl = F.log_softmax(wdl_logits, dim=-1)
                ce_wdl = -(value_targets * log_wdl).sum(dim=-1)
                val_mask = ((plies >= 10) & (value_targets.sum(dim=-1) > 0)).float()
                wdl_loss_sum = (ce_wdl * val_mask).sum().item()
                valid_val_count = int(val_mask.sum().item())

            sum_policy_loss += float(policy_loss.item())
            sum_wdl_loss += wdl_loss_sum
            sum_wdl_masked_samples += valid_val_count
            total_samples += batch_sz

            _, pred = policy_logits.topk(3, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(moves.view(1, -1).expand_as(pred))
            correct_top1 += int(correct[:1].reshape(-1).float().sum().item())
            correct_top3 += int(correct[:3].reshape(-1).float().sum().item())
            sum_entropy += compute_policy_entropy(policy_logits) * batch_sz

    model.train()
    if total_samples == 0:
        return {"policy_loss": 0.0, "wdl_loss": 0.0, "top1": 0.0, "top3": 0.0, "entropy": 0.0}

    return {
        "policy_loss": sum_policy_loss / total_samples,
        "wdl_loss": sum_wdl_loss / max(1, sum_wdl_masked_samples),
        "total_loss": (sum_policy_loss / total_samples)
        + 0.5 * (sum_wdl_loss / max(1, sum_wdl_masked_samples)),
        "top1": (correct_top1 / total_samples) * 100.0,
        "top3": (correct_top3 / total_samples) * 100.0,
        "entropy": sum_entropy / total_samples,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    step: int,
    epoch: int,
    best_val_loss: float,
    num_blocks: int = 12,
    num_filters: int = 128,
    keep_last_n: int = 3,
) -> None:
    """Save training state checkpoint (atomic tmp write + os.replace)."""
    checkpoint_path = Path(checkpoint_path)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
        "metadata": {
            "architecture": f"{num_blocks} ResBlocks x {num_filters} Filters",
            "step": step,
            "epoch": epoch,
        },
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": _safe_numpy_rng_state(),
        "python_rng_state": random.getstate(),
    }
    _atomic_torch_save(state, checkpoint_path)
    print(f"\n[CHECKPOINT] Saved step {step} state to {checkpoint_path}")
    if keep_last_n >= 0:
        try:
            from python.common.cleanup import prune_checkpoints

            parent = Path(checkpoint_path).parent
            # Only prune periodic step checkpoints, never best/latest.
            if Path(checkpoint_path).name.startswith("checkpoint_step_"):
                prune_checkpoints(parent, keep_last_n=keep_last_n)
        except (OSError, ValueError):
            logger.debug("supervised checkpoint prune skipped", exc_info=True)


def train_supervised(
    train_dir: str | Path,
    val_dir: str | Path,
    batch_size: int = 512,
    base_lr: float = 0.0002,
    warmup_steps: int = 500,
    epochs: int = 5,
    max_steps: Optional[int] = None,
    checkpoint_dir: str | Path = "models/checkpoints",
    log_dir: str | Path = "runs/supervised",
    checkpoint_interval_sec: float = 1800.0,
    resume_path: Optional[str | Path] = None,
    num_workers: int = 0 if sys.platform == "win32" else 2,
    gpu_resident: bool = False,
    num_blocks: int = 12,
    num_filters: int = 128,
    seed: Optional[int] = None,
    resume_weights_only: bool = False,
    gpu_budget_gb: Optional[float] = None,
    tactics_ratio: Optional[float] = None,
    micro_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    tactics_dir: Optional[str | Path] = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Execute supervised training loop.

    ``seed`` sets Python/numpy/torch RNGs once up front (None = unseeded,
    today's behavior). Seeding is initialization only; no training math changes.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve effective batch size and gradient accumulation steps
    effective_batch_size = int(batch_size)
    if micro_batch_size is not None and int(micro_batch_size) > 0:
        micro_bs = int(micro_batch_size)
        if gradient_accumulation_steps is not None and int(gradient_accumulation_steps) > 0:
            accum_steps = int(gradient_accumulation_steps)
            effective_batch_size = micro_bs * accum_steps
        else:
            accum_steps = max(1, effective_batch_size // micro_bs)
            effective_batch_size = micro_bs * accum_steps
    elif gradient_accumulation_steps is not None and int(gradient_accumulation_steps) > 1:
        accum_steps = int(gradient_accumulation_steps)
        micro_bs = max(1, effective_batch_size // accum_steps)
        effective_batch_size = micro_bs * accum_steps
    else:
        # Default adaptive microbatching for safe memory on 8GB GPU / Windows
        if effective_batch_size >= 128:
            micro_bs = 64
            accum_steps = max(1, effective_batch_size // micro_bs)
            effective_batch_size = micro_bs * accum_steps
        else:
            accum_steps = 1
            micro_bs = effective_batch_size

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    prec_str = "bfloat16 AMP" if device.type == "cuda" else "float32"
    print("\n" + "=" * 80)
    print("ZeroPawn Supervised Training Configuration Summary")
    print("=" * 80)
    print(f"Dataset Train Path             : {train_dir}")
    print(f"Dataset Val Path               : {val_dir}")
    print(
        f"Model Architecture             : {num_blocks} ResBlocks, {num_filters} Filters (8.63M parameters)"
    )
    print(f"Effective Optimizer Batch Size : {effective_batch_size} samples")
    print(f"Physical Micro-Batch Size      : {micro_bs} samples")
    print(f"Gradient Accumulation Steps    : {accum_steps} steps")
    print(f"Initial / Base Learning Rate   : {base_lr:.6f}")
    print("Optimizer                      : AdamW (weight_decay=1e-4)")
    print(f"Device                         : {device} ({device_name})")
    print(f"Precision Mode                 : {prec_str}")
    print(f"DataLoader Workers             : {num_workers} (Windows-safe low-memory)")
    print(
        f"DataLoader Mode                : {'GpuResident' if gpu_resident and device.type == 'cuda' else 'Disk-backed Memmap Stream'}"
    )
    print("=" * 80 + "\n")

    if gpu_resident and device.type == "cuda":
        print("Using high-throughput GpuResidentDataLoader (maximizing GPU VRAM over CPU/RAM)...")
        train_loader = GpuResidentDataLoader(
            train_dir,
            batch_size=micro_bs,
            shuffle=True,
            drop_last=True,
            device=device,
            storage_dtype=torch.bfloat16,
            max_gpu_memory_gb=gpu_budget_gb,
            tactics_ratio=tactics_ratio,
            tactics_dir=tactics_dir,
        )
        val_loader = GpuResidentDataLoader(
            val_dir,
            batch_size=micro_bs,
            shuffle=False,
            drop_last=False,
            device=device,
            storage_dtype=torch.bfloat16,
            max_gpu_memory_gb=gpu_budget_gb,
        )
    else:
        train_ds = BinaryChessDataset(train_dir)
        val_ds = BinaryChessDataset(val_dir)
        print(f"Loaded datasets: {len(train_ds):,} train samples, {len(val_ds):,} val samples")

        sampler = None
        shuffle = True

        if tactics_dir is not None:
            t_path = Path(tactics_dir)
            if not t_path.is_dir():
                raise FileNotFoundError(f"Tactical dataset directory not found: {t_path}")
            t_ratio = tactics_ratio if tactics_ratio is not None else 0.3
            if not (0.0 <= t_ratio <= 1.0):
                raise ValueError(f"tactics_ratio must be between 0.0 and 1.0, got {t_ratio}")
            tactics_ds = BinaryChessDataset(t_path, is_tactical=True)
            if len(tactics_ds) == 0 and t_ratio > 0.0:
                raise ValueError(
                    f"tactics_ratio is {t_ratio} but tactical dataset at {t_path} is empty."
                )
            combined_ds = SourceAwareCombinedDataset(train_ds, tactics_ds, tactics_ratio=t_ratio)
            sampler = combined_ds.get_sampler()
            train_ds = combined_ds
            shuffle = False
        elif tactics_ratio is not None:
            if not (0.0 <= tactics_ratio <= 1.0):
                raise ValueError(f"tactics_ratio must be between 0.0 and 1.0, got {tactics_ratio}")
            outcomes_bin = Path(train_dir) / "outcomes.bin"
            plies_bin = Path(train_dir) / "plies.bin"
            if outcomes_bin.exists():
                outcomes_m = np.memmap(
                    outcomes_bin, dtype=np.uint8, mode="r", shape=(len(train_ds),)
                )
                tactics_mask = outcomes_m[:] == 255
            elif plies_bin.exists():
                plies_m = np.memmap(plies_bin, dtype=np.int16, mode="r", shape=(len(train_ds),))
                tactics_mask = plies_m[:] == 0
            else:
                tactics_mask = None

            if tactics_mask is not None:
                n_tactics = int(tactics_mask.sum())
                n_master = len(train_ds) - n_tactics
                if tactics_ratio > 0.0 and n_tactics == 0:
                    raise ValueError(
                        f"tactics_ratio is {tactics_ratio} but dataset at {train_dir} contains 0 tactical positions."
                    )
                if n_tactics > 0 and n_master > 0:
                    from torch.utils.data import WeightedRandomSampler

                    w_tactics = tactics_ratio / n_tactics
                    w_master = (1.0 - tactics_ratio) / n_master
                    weights = np.where(tactics_mask, w_tactics, w_master).astype(np.float32)
                    sampler = WeightedRandomSampler(
                        torch.from_numpy(weights), num_samples=len(train_ds), replacement=True
                    )
                    shuffle = False

        train_loader = DataLoader(
            train_ds,
            batch_size=micro_bs,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=micro_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = ChessNet(
        in_channels=19, num_filters=num_filters, num_blocks=num_blocks, num_actions=4672
    )
    model = model.to(device=device, memory_format=torch.channels_last)
    # Modern PyTorch optimizations (Finding #8): compile, fused AdamW, cuDNN benchmark, TF32
    # torch.compile disabled for Triton compatibility
    # if device.type == "cuda" and hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    #     except Exception:
    #         pass
    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False
    if device.type == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=1e-4, fused=(device.type == "cuda")
    )

    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_training_steps = max_steps if max_steps is not None else (steps_per_epoch * epochs)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        base_lr=base_lr,
    )

    writer = SummaryWriter(log_dir=str(log_dir))
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.is_file() or ckpt_dir.suffix == ".pt":
        if not resume_path:
            resume_path = str(ckpt_dir)
        ckpt_dir = ckpt_dir.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path and Path(resume_path).exists():
        print(f"Resuming checkpoint from {resume_path}...")
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        except (OSError, RuntimeError, KeyError) as e:
            raise RuntimeError(f"Failed to load supervised checkpoint '{resume_path}': {e}") from e
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"Checkpoint '{resume_path}' is invalid: missing 'model' weights.")
        _validate_arch(checkpoint, model, num_blocks, num_filters, resume_path)
        model.load_state_dict(checkpoint.get("model"))
        if not resume_weights_only:
            if checkpoint.get("optimizer") is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except (RuntimeError, KeyError, ValueError) as e:
                    raise RuntimeError(
                        f"Checkpoint '{resume_path}' has an incompatible optimizer state: {e}"
                    ) from e
            if checkpoint.get("scheduler") is not None:
                try:
                    saved_scheduler = checkpoint["scheduler"]
                    # Only restore the current step; keep CLI-provided LR, warmup,
                    # total_steps, and min_lr (fixes silent CLI override bug).
                    saved_current_step = saved_scheduler.get("current_step", 0)
                    scheduler.current_step = saved_current_step
                    # Preserve CLI/base values set when scheduler was constructed
                    # (base_lr, warmup_steps, total_steps, min_lr stay as-is)
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(
                        "Ignoring incompatible scheduler state in '%s': %s", resume_path, e
                    )
            start_step = checkpoint.get("step", 0)
            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            if checkpoint.get("rng_state") is not None:
                try:
                    rng = checkpoint["rng_state"]
                    if hasattr(rng, "cpu"):
                        rng = rng.cpu()
                    if hasattr(rng, "to") and hasattr(rng, "dtype") and rng.dtype != torch.uint8:
                        rng = rng.to(torch.uint8)
                    torch.set_rng_state(rng)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Ignoring incompatible rng_state in '%s': %s", resume_path, e)
            if checkpoint.get("numpy_rng_state") is not None:
                try:
                    _restore_numpy_rng_state(checkpoint["numpy_rng_state"])
                except (ValueError, TypeError) as e:
                    logger.debug(
                        "Ignoring incompatible numpy_rng_state in '%s': %s", resume_path, e
                    )
            if checkpoint.get("python_rng_state") is not None:
                try:
                    random.setstate(checkpoint["python_rng_state"])
                except (ValueError, TypeError) as e:
                    logger.debug(
                        "Ignoring incompatible python_rng_state in '%s': %s", resume_path, e
                    )
            print(
                f"Resumed at step {start_step}, epoch {start_epoch}, best val loss {best_val_loss:.4f}"
            )
        else:
            print(
                f"Loaded weights only from {resume_path} (fresh optimizer and step 0 for fine-tuning)"
            )

    global_step = start_step
    last_checkpoint_time = time.time()
    t_start = time.time()

    val_metrics: dict[str, float] = {}

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_samples = 0
            t_epoch_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            batch_buffer: list[dict[str, torch.Tensor]] = []

            def _step_window(window: list[dict[str, torch.Tensor]], step_t0: float) -> None:
                nonlocal global_step, epoch_samples, last_checkpoint_time
                w_size = len(window)
                if w_size == 0:
                    return

                # Calculate total valid WDL samples across this effective window
                v_total = 0
                for b in window:
                    pl = b["ply"]
                    vt = b["value_target"]
                    v_mask = (pl >= 10) & (vt.sum(dim=-1) > 0)
                    v_total += v_mask.sum().item()

                window_losses = []
                window_skip = False
                for m_idx, b in enumerate(window):
                    is_first = m_idx == 0
                    is_last = m_idx == w_size - 1
                    tot, pol, wdl, t1, t3, ent, skip = train_step(
                        model=model,
                        batch=b,
                        optimizer=optimizer,
                        device=device,
                        use_autocast=(device.type == "cuda"),
                        accum_steps=w_size,
                        step_optimizer=is_last,
                        zero_grad=is_first,
                        v_total=v_total,
                    )
                    if skip:
                        window_skip = True
                    window_losses.append((tot, pol, wdl, t1, t3, ent))

                # Only advance scheduler and step counter if no non-finite loss occurred.
                # This fixes the bug where a bad micro-batch still advanced LR/scheduler.
                lr = None
                if not window_skip:
                    lr = scheduler.step()
                    global_step += 1
                else:
                    # When skipped, report the current LR without advancing scheduler
                    lr = optimizer.param_groups[0].get("lr", 0.0) if optimizer.param_groups else 0.0
                effective_samples = w_size * micro_bs
                epoch_samples += effective_samples

                avg_tot = sum(x[0] for x in window_losses) / len(window_losses)
                avg_pol = sum(x[1] for x in window_losses) / len(window_losses)
                avg_wdl = sum(x[2] for x in window_losses) / len(window_losses)
                avg_top1 = sum(x[3] for x in window_losses) / len(window_losses)
                avg_top3 = sum(x[4] for x in window_losses) / len(window_losses)
                avg_ent = sum(x[5] for x in window_losses) / len(window_losses)

                step_time = time.time() - step_t0
                sps = effective_samples / max(1e-4, step_time)

                if global_step % 20 == 0:
                    writer.add_scalar("loss/total", avg_tot, global_step)
                    writer.add_scalar("loss/policy", avg_pol, global_step)
                    writer.add_scalar("loss/wdl", avg_wdl, global_step)
                    writer.add_scalar("accuracy/top1", avg_top1, global_step)
                    writer.add_scalar("accuracy/top3", avg_top3, global_step)
                    writer.add_scalar("entropy/policy", avg_ent, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("perf/samples_per_sec", sps, global_step)

                if global_step % 100 == 0:
                    emit_report(
                        format_kv_line(
                            [
                                ("step", global_step),
                                ("loss", f"{avg_tot:.4f}"),
                                ("pol_loss", f"{avg_pol:.4f}"),
                                ("wdl_loss", f"{avg_wdl:.4f}"),
                                ("top1", f"{avg_top1:.1f}"),
                                ("top3", f"{avg_top3:.1f}"),
                                ("sps", f"{sps:.1f}"),
                                ("lr", f"{lr:.6f}"),
                            ]
                        )
                    )

                now = time.time()
                if now - last_checkpoint_time >= checkpoint_interval_sec:
                    save_checkpoint(
                        ckpt_dir / f"checkpoint_step_{global_step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        best_val_loss,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                    )
                    last_checkpoint_time = now

            step_t0 = time.time()
            for batch in train_loader:
                batch_buffer.append(batch)
                if len(batch_buffer) == accum_steps:
                    _step_window(batch_buffer, step_t0)
                    batch_buffer = []
                    step_t0 = time.time()
                    if max_steps is not None and global_step >= max_steps:
                        break

            if batch_buffer and (max_steps is None or global_step < max_steps):
                _step_window(batch_buffer, step_t0)
                batch_buffer = []

            if max_steps is not None and global_step >= max_steps:
                max_val_batches = 2 if (max_steps is not None and max_steps <= 20) else None
                val_metrics = evaluate(
                    model, val_loader, device=device, max_batches=max_val_batches
                )
                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    save_checkpoint(
                        ckpt_dir / "best_model.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        best_val_loss,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                    )
                break

            # End of epoch validation
            max_val_batches = 2 if (max_steps is not None and max_steps <= 20) else None
            val_metrics = evaluate(model, val_loader, device=device, max_batches=max_val_batches)
            emit_report(
                format_kv_line(
                    [
                        ("epoch", f"{epoch + 1}/{epochs}"),
                        ("phase", "val"),
                        ("val_loss", f"{val_metrics['total_loss']:.4f}"),
                        ("pol_loss", f"{val_metrics['policy_loss']:.4f}"),
                        ("wdl_loss", f"{val_metrics['wdl_loss']:.4f}"),
                        ("top1", f"{val_metrics['top1']:.2f}"),
                        ("top3", f"{val_metrics['top3']:.2f}"),
                        ("entropy", f"{val_metrics['entropy']:.3f}"),
                    ]
                )
            )

            writer.add_scalar("val/total_loss", val_metrics["total_loss"], global_step)
            writer.add_scalar("val/policy_loss", val_metrics["policy_loss"], global_step)
            writer.add_scalar("val/wdl_loss", val_metrics["wdl_loss"], global_step)
            writer.add_scalar("val/top1", val_metrics["top1"], global_step)
            writer.add_scalar("val/top3", val_metrics["top3"], global_step)
            writer.add_scalar("val/entropy", val_metrics["entropy"], global_step)

            # Save best model checkpoint
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                save_checkpoint(
                    ckpt_dir / "best_model.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    best_val_loss,
                    num_blocks=num_blocks,
                    num_filters=num_filters,
                )

            if max_steps is not None and global_step >= max_steps:
                break

    except KeyboardInterrupt:
        print("\n[PAUSED] Training interrupted by user. Saving emergency checkpoint...")
        current_epoch = epoch if "epoch" in locals() else start_epoch
        save_checkpoint(
            ckpt_dir / "latest_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            global_step,
            current_epoch,
            best_val_loss,
            num_blocks=num_blocks,
            num_filters=num_filters,
        )
        raise
    finally:
        # Save final/latest checkpoint on clean completion (fixes missing final checkpoint bug)
        try:
            final_epoch = (
                epoch + 1
                if "epoch" in locals()
                else (start_epoch + 1 if "start_epoch" in locals() else 0)
            )
            final_step = (
                global_step
                if "global_step" in locals()
                else (start_step if "start_step" in locals() else 0)
            )
            if "model" in locals() and "optimizer" in locals() and "scheduler" in locals():
                save_checkpoint(
                    ckpt_dir / "latest_checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    final_step,
                    final_epoch,
                    best_val_loss if "best_val_loss" in locals() else float("inf"),
                    num_blocks=num_blocks if "num_blocks" in locals() else 12,
                    num_filters=num_filters if "num_filters" in locals() else 128,
                )
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        if "train_loader" in locals() and hasattr(train_loader, "close"):
            try:
                train_loader.close()
            except Exception:
                pass
        if "val_loader" in locals() and hasattr(val_loader, "close"):
            try:
                val_loader.close()
            except Exception:
                pass
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return val_metrics


def main() -> None:
    # Single logging setup: rotating file for everything, WARNING+ on stderr.
    setup_logging(log_dir="logs", log_filename="train_supervised.log")
    default_train = (
        "data/dataset_training/train"
        if Path("data/dataset_training/train").is_dir()
        else (
            "data/dataset_scaled/train"
            if Path("data/dataset_scaled/train").is_dir()
            else "data/dataset/train"
        )
    )
    default_val = (
        "data/dataset_training/val"
        if Path("data/dataset_training/val").is_dir()
        else (
            "data/dataset_scaled/val"
            if Path("data/dataset_scaled/val").is_dir()
            else "data/dataset/val"
        )
    )
    default_ckpt = (
        "models/checkpoints_scaled"
        if Path("models/checkpoints_scaled").is_dir()
        else "models/checkpoints"
    )

    parser = argparse.ArgumentParser(description="Train ZeroPawn supervised warm start model.")
    parser.add_argument("--train-dir", type=str, default=default_train, help="Train data dir.")
    parser.add_argument("--val-dir", type=str, default=default_val, help="Validation data dir.")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Effective batch size (default: 512)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Base learning rate (default: 0.0002)."
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Physical micro-batch size for forward/backward passes.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps.")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=default_ckpt, help="Checkpoints path."
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs/supervised", help="TensorBoard logs dir."
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument(
        "--gpu-resident",
        dest="gpu_resident",
        action="store_true",
        default=True,
        help="Store dataset in GPU VRAM (with host RAM spillover).",
    )
    parser.add_argument(
        "--no-gpu-resident",
        dest="gpu_resident",
        action="store_false",
        help="Disable GPU-resident data loading.",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=12, help="Number of ResNet residual blocks."
    )
    parser.add_argument(
        "--num-filters", type=int, default=128, help="Number of convolutional filters."
    )
    parser.add_argument(
        "--tactics-ratio",
        type=float,
        default=None,
        help="Target ratio for tactical puzzle positions (0.0 to 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible runs (default: None = unseeded, today's behavior).",
    )

    args = parser.parse_args()

    ctx = run_context("warm-start")
    t0 = time.time()
    emit_report(banner("warm-start", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "auto",
            str(args.seed),
        )
    )
    emit_report(
        parameters_table(
            [
                (
                    "train_dir",
                    args.train_dir,
                    "user" if args.train_dir != default_train else "default",
                ),
                ("val_dir", args.val_dir, "user" if args.val_dir != default_val else "default"),
                ("batch_size", args.batch_size, "user" if args.batch_size != 512 else "default"),
                ("lr", args.lr, "user" if args.lr != 0.0002 else "default"),
                (
                    "micro_batch_size",
                    args.micro_batch_size,
                    "user" if args.micro_batch_size is not None else "default",
                ),
                (
                    "gradient_accumulation_steps",
                    args.gradient_accumulation_steps,
                    "user" if args.gradient_accumulation_steps is not None else "default",
                ),
                (
                    "warmup_steps",
                    args.warmup_steps,
                    "user" if args.warmup_steps != 500 else "default",
                ),
                ("epochs", args.epochs, "user" if args.epochs != 5 else "default"),
                ("max_steps", args.max_steps, "user" if args.max_steps is not None else "default"),
                (
                    "checkpoint_dir",
                    args.checkpoint_dir,
                    "user" if args.checkpoint_dir != default_ckpt else "default",
                ),
                (
                    "log_dir",
                    args.log_dir,
                    "user" if args.log_dir != "runs/supervised" else "default",
                ),
                ("resume", args.resume, "user" if args.resume is not None else "default"),
                ("workers", args.workers, "user" if args.workers != 2 else "default"),
                ("gpu_resident", args.gpu_resident, "user" if not args.gpu_resident else "default"),
                ("num_blocks", args.num_blocks, "user" if args.num_blocks != 12 else "default"),
                ("num_filters", args.num_filters, "user" if args.num_filters != 128 else "default"),
                (
                    "tactics_ratio",
                    args.tactics_ratio,
                    "user" if args.tactics_ratio is not None else "default",
                ),
                ("seed", args.seed, "user" if args.seed is not None else "default"),
            ]
        )
    )

    try:
        metrics = train_supervised(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            epochs=args.epochs,
            max_steps=args.max_steps,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_path=args.resume,
            num_workers=args.workers,
            gpu_resident=args.gpu_resident,
            num_blocks=args.num_blocks,
            num_filters=args.num_filters,
            tactics_ratio=args.tactics_ratio,
            seed=args.seed,
        )
        emit_report(
            format_kv_line(
                [
                    ("phase", "final"),
                    ("val_loss", f"{metrics.get('total_loss', float('nan')):.4f}"),
                    ("top1", f"{metrics.get('top1', float('nan')):.2f}"),
                ]
            )
        )
        records = []
        ckpt_path = Path(args.checkpoint_dir)
        if ckpt_path.is_dir():
            for path in sorted(ckpt_path.glob("*.pt")):
                records.append(file_record(str(path)))
        records.append((f"tensorboard: {args.log_dir}", -1, None))
        emit_report(artifacts_list(records))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"warm-start failed: {e}",
                "bad dataset dirs, out of memory/VRAM, or unreadable resume checkpoint",
                "check --train-dir/--val-dir contents and free VRAM, then see LOG",
                "logs/train_supervised.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
