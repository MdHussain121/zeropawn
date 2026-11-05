"""Supervised warm start training script for ZeroPawn.

Contract Specifications:
- Model: PyTorch dual-head ResNet (12 ResBlocks, 128 filters, channels-last)
- Optimizer: AdamW
- Schedule: Linear warmup + Cosine decay (single formula, see WarmupCosineScheduler)
- Policy Loss: legal-masked CE with 0.05 label smoothing over legal actions
  (full-vocabulary fallback only for schema-1 datasets, logged once)
- Total Loss: policy_loss + 0.5 * wdl_loss (WDL masked to ply >= 10 with a
  non-zero target)
- Mixed Precision: bfloat16 autocast
- Checkpointing: Resumable checkpoints every 30 minutes / configured steps;
  every save carries optimizer/checkpoint type stamps and run provenance
- Telemetry: TensorBoard metrics + durable run_manifest.json + telemetry.jsonl
  next to the checkpoints (audit SLQ-02)
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from python.constants import NUM_ACTIONS, NUM_PLANES
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
    seed_worker,
)
from python.model.chess_net import ChessNet

logger = logging.getLogger(__name__)


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    """Write torch state atomically: tmp file, fsync file, replace, fsync dir.

    Every artifact writer in the project funnels through this primitive. The
    temp file's own fsync is the step that makes the "complete candidate" real
    before the rename (audit CKP-03 requirement 5), so an interrupted write can
    never publish a partially-flushed rename target.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    try:
        fd = os.open(str(tmp_path), os.O_RDONLY)
    except OSError:
        logger.debug("Could not open tmp checkpoint for fsync: %s", tmp_path)
    else:
        try:
            os.fsync(fd)
        except OSError:
            logger.debug("fsync failed for %s", tmp_path)
        finally:
            os.close(fd)
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

    def step(self, step_num: Optional[int] = None) -> float:
        """Advance the schedule one step and apply the resulting LR.

        With ``step_num`` the LR is computed as a pure function of that step
        (identical to :meth:`get_lr`) and the internal counter jumps there. The
        RL trainer uses this to drive the schedule from its own checkpointed
        step counter; callers without an argument advance by one. Both paths
        delegate to :meth:`get_lr` — the single definition of the schedule
        (audit SLQ-11: the previous no-arg warmup formula
        ``base_lr · k / warmup_steps`` diverged from ``get_lr``'s
        min_lr-anchored warmup; they can no longer disagree).
        """
        if step_num is not None:
            self.current_step = int(step_num)
        else:
            self.current_step += 1
        lr = self.get_lr(self.current_step)

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

    def get_lr(self, step: int) -> float:
        """Return the learning rate at a given step without advancing the scheduler.

        Single source of truth for the schedule (audit SLQ-11): min_lr-anchored
        linear warmup for ``step < warmup_steps``, then cosine decay toward
        ``min_lr``. Both :meth:`step` paths delegate here.
        """
        if step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (step / max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def load_state_dict(self, state: dict, override_lr: bool = False) -> None:
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        if not override_lr:
            self.base_lr = state["base_lr"]
            self.min_lr = state["min_lr"]
        # Legacy checkpoints (pre shared-scheduler) may not carry current_step.
        self.current_step = state.get("current_step", self.current_step)


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


_LABEL_SMOOTHING = 0.05
_warned_unmasked = False


def legal_masked_policy_loss(
    logits: torch.Tensor,
    moves: torch.Tensor,
    legal_mask: Optional[torch.Tensor],
    label_smoothing: float = _LABEL_SMOOTHING,
) -> torch.Tensor:
    """Policy cross-entropy with label smoothing restricted to LEGAL actions (SL-01).

    With mask: the target distribution is ``(1-s)`` on the played move plus
    ``s`` spread uniformly over the position's legal actions, so no probability
    mass is ever spent teaching the model that impossible moves are plausible.
    Without mask: legacy full-vocabulary smoothing (the pre-remediation
    behavior), loudly logged once — capability is opt-in by dataset schema,
    never silently skipped.

    Illegal training labels are a data-integrity error and raise.
    """
    global _warned_unmasked
    if legal_mask is None:
        if not _warned_unmasked:
            logger.warning(
                "Batch has NO legal-mask stream: falling back to full-vocabulary "
                "label smoothing over all 4672 actions (pre-remediation behavior). "
                "Rebuild the dataset with a schema-2 builder to enable legal-move "
                "masking (SL-01)."
            )
            _warned_unmasked = True
        return F.cross_entropy(logits, moves, label_smoothing=label_smoothing)

    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    legal_counts = legal_mask.sum(dim=1)
    if int((legal_counts == 0).sum().item()):
        bad = int((legal_counts == 0).sum().item())
        raise ValueError(f"legal-mask stream contains {bad} position(s) with zero legal actions")

    # Data-integrity gate: the stored label must be a legal action.
    label_legal = legal_mask.gather(1, moves.unsqueeze(1)).squeeze(1)
    illegal = (~label_legal).sum().item()
    if illegal:
        idx = torch.nonzero(~label_legal, as_tuple=True)[0][:5].tolist()
        raise ValueError(
            f"Illegal training label(s): {illegal} move(s) in this batch are not in "
            f"the position's legal-action set (first indices {idx}). This is a "
            "data-integrity error, not something to smooth over."
        )

    log_probs = F.log_softmax(logits.float(), dim=-1)
    natural = -log_probs.gather(1, moves.unsqueeze(1)).squeeze(1)
    # Mean log-probability over legal actions only (per row).
    legal_mean = -(log_probs * legal_mask).sum(dim=1) / legal_counts
    s = float(label_smoothing)
    return ((1.0 - s) * natural + s * legal_mean).mean()


def _grad_norms(model: nn.Module, device: torch.device) -> tuple[float, float]:
    """Pre-clip L2 grad norms (total, value head) over parameters carrying grads.

    The value head is identified by the ChessNet parameter-name contract
    (``value_conv``/``value_bn``/``value_fc1``/``value_fc2``). Used for the
    value-head gradient diagnostics that prove WDL supervision reaches the
    value head (audit SLQ-01/SLQ-02 telemetry requirement).
    """
    total = torch.zeros((), device=device)
    value = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().pow(2).sum()
        total = total + g
        if "value" in name:
            value = value + g
    return float(total.sqrt().item()), float(value.sqrt().item())


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
) -> tuple[float, float, float, float, float, float, bool, Optional[dict[str, float]]]:
    """Single optimization step or accumulation micro-step.

    Returns ``(total_loss, policy_loss, wdl_loss, top1, top3, entropy, skip,
    grad_diag)`` where ``grad_diag`` carries the pre-clip total and value-head
    gradient norms on optimizer-step micro-batches (None otherwise).
    """
    device = torch.device(device)
    device_type: str = device.type
    planes = batch["planes"].to(device, memory_format=torch.channels_last, non_blocking=True)
    moves = batch["move"].to(device, non_blocking=True)
    value_targets = batch["value_target"].to(device, non_blocking=True)
    plies = batch["ply"].to(device, non_blocking=True)
    # The legal mask must ride with the batch: the disk-backed loader yields
    # CPU tensors while the model runs on CUDA (the GPU-resident loader
    # yields device tensors, so this is a no-op there).
    legal_mask = batch.get("legal_mask")
    if legal_mask is not None:
        legal_mask = legal_mask.to(device, non_blocking=True)

    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    autocast_context = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_autocast
        else nullcontext()
    )

    with autocast_context:
        policy_logits, wdl_logits = model(planes)

        # Policy loss: legal-move-masked CE with smoothing over legal actions
        # only (SL-01); falls back to full-vocabulary smoothing when the batch
        # carries no mask stream (logged once).
        policy_loss = legal_masked_policy_loss(
            policy_logits, moves, legal_mask, label_smoothing=0.05
        )

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
            None,
        )

    loss_to_backward.backward()

    grad_diag: Optional[dict[str, float]] = None
    if step_optimizer:
        total_norm, value_norm = _grad_norms(model, device)
        grad_diag = {"total_grad_norm": total_norm, "value_grad_norm": value_norm}
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
        grad_diag,
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
            batch_legal_mask = batch.get("legal_mask")
            if batch_legal_mask is not None:
                batch_legal_mask = batch_legal_mask.to(device, non_blocking=True)
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
                # Evaluation keeps the same objective as training (masked when
                # the batch carries a mask stream), summed per-sample for
                # exact averaging over the split.
                per_sample_policy = legal_masked_policy_loss(
                    policy_logits, moves, batch_legal_mask, label_smoothing=0.05
                )
                policy_loss = per_sample_policy * batch_sz
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
    run_metadata: Optional[dict[str, Any]] = None,
    provenance: Optional[dict[str, Any]] = None,
    artifact_ticket: Optional[Any] = None,
) -> None:
    """Save training state checkpoint (atomic tmp write + os.replace).

    Every save carries the optimizer/checkpoint type stamps (audit SLQ-17), the
    run's provenance metadata (audit SLQ-02: run identity, dataset receipts, git
    commit, seed, and configuration travel inside the artifact), the formal
    schema version (CKP-10), the run-status ledger (CKP-01), and CUDA RNG state
    when training on CUDA (CKP-12). The destination is validated against the
    production-write policy and held under a cross-process lock for the update.
    """
    from python.common.artifacts import artifact_lock, assert_safe_artifact_destination
    from python.model.checkpoint_schema import CURRENT_SUPERVISED_FORMAT, build_provenance

    checkpoint_path = assert_safe_artifact_destination(
        checkpoint_path,
        ticket=artifact_ticket,
        arch=(int(num_blocks), int(num_filters)),
        what="supervised checkpoint",
    )
    ledger = dict(run_metadata) if run_metadata else {}
    provenance_payload = (
        dict(provenance)
        if provenance
        else build_provenance(
            run_id=ledger.get("run_id"),
            run_type=ledger.get("run_type", "supervised"),
            config=ledger.get("config"),
            model_arch=(int(num_blocks), int(num_filters)),
            param_count=int(sum(p.numel() for p in model.parameters())),
            dataset=ledger.get("dataset"),
            parent_checkpoint=ledger.get("parent_checkpoint"),
            optimizer_config={"type": type(optimizer).__name__},
            scheduler_config=scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
        )
    )
    cuda_rng: Optional[list] = None
    if torch.cuda.is_available():
        try:
            cuda_rng = list(torch.cuda.get_rng_state_all())
        except Exception:  # pragma: no cover - defensive
            logger.debug("CUDA RNG capture failed", exc_info=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
        "checkpoint_type": "supervised",
        "checkpoint_format": CURRENT_SUPERVISED_FORMAT,
        "_optimizer_type": "AdamW",
        "run_metadata": ledger,
        "provenance": provenance_payload,
        "metadata": {
            "architecture": f"{num_blocks} ResBlocks x {num_filters} Filters",
            "step": step,
            "epoch": epoch,
        },
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": _safe_numpy_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": cuda_rng,
    }
    with artifact_lock(checkpoint_path, timeout=0.0, owner=f"supervised-save step {step}"):
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


class _RunTelemetry:
    """Durable per-run telemetry: run manifest + append-only JSONL stream.

    Audit SLQ-02/SLQ-15 remediation: every supervised run persists enough
    information next to its checkpoints to reproduce and interpret it —
    run identity, git commit, dataset receipts, seed, full configuration,
    schedule geometry, and per-step/per-epoch metrics. The JSONL stream is
    flushed per line so interrupted runs retain their telemetry, and each
    line carries the run_id so concurrent or resumed runs sharing a
    checkpoint directory remain attributable.
    """

    def __init__(self, ckpt_dir: Path, manifest: dict[str, Any]) -> None:
        import uuid
        from datetime import datetime, timezone

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.ckpt_dir / "run_manifest.json"
        self.telemetry_path = self.ckpt_dir / "telemetry.jsonl"
        self.run_id = manifest.get("run_id") or (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        manifest["run_id"] = self.run_id
        # Rotate any previous run's manifest so it is never silently lost.
        if self.manifest_path.exists():
            prior_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            try:
                self.manifest_path.replace(self.ckpt_dir / f"run_manifest_{prior_stamp}.json")
            except OSError:
                logger.debug("manifest rotation failed", exc_info=True)
        self.manifest = manifest
        self._write_manifest()
        self._fh = open(self.telemetry_path, "a", encoding="utf-8")

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(self.manifest_path)

    def log(self, record: dict[str, Any]) -> None:
        record = {"run_id": self.run_id, "ts": time.time(), **record}
        try:
            self._fh.write(json.dumps(record, sort_keys=True) + "\n")
            self._fh.flush()
        except (OSError, ValueError, TypeError):
            logger.debug("telemetry write failed", exc_info=True)

    def checkpoint_event(self, path: Path, reason: str, step: int) -> None:
        self.log({"event": "checkpoint_saved", "path": str(path), "reason": reason, "step": step})

    def finalize(self, status: str, extra: Optional[dict[str, Any]] = None) -> None:
        self.manifest["status"] = status
        self.manifest["finished_at"] = time.time()
        if extra:
            self.manifest.update(extra)
        self._write_manifest()
        try:
            self.log({"event": "run_finished", "status": status})
            self._fh.close()
        except (OSError, ValueError):
            logger.debug("telemetry finalize failed", exc_info=True)


def _build_run_metadata(
    train_dir: str | Path,
    val_dir: str | Path,
    gate_receipts: dict[str, Any],
    seed: Optional[int],
    num_blocks: int,
    num_filters: int,
    batch_size: int,
    micro_bs: int,
    accum_steps: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    epochs: int,
    max_steps: Optional[int],
    gpu_resident: bool,
    tactics_ratio: Optional[float],
    tactics_dir: Optional[str | Path],
    data_role: Optional[str],
    allow_uncertified_data: bool,
    verify_identity: str,
    checkpoint_dir: str | Path,
    log_dir: str | Path,
) -> dict[str, Any]:
    """Assemble the run manifest written next to the checkpoints (audit SLQ-02)."""
    import platform as _platform

    from python.data.dataset_contract import detect_git_commit

    def _receipt_summary(label: str) -> dict[str, Any]:
        info = gate_receipts.get(label, {})
        receipt = info.get("receipt")
        files_digest = None
        if isinstance(receipt, dict) and "files" in receipt:
            files_str = json.dumps(receipt["files"], sort_keys=True)
            files_digest = f"sha256:{hashlib.sha256(files_str.encode()).hexdigest()[:16]}"
        if not isinstance(receipt, dict):
            return {
                "data_dir": info.get("data_dir"),
                "certified": info.get("certified", False),
                "receipt_id": None,
                "files_digest": None,
            }
        return {
            "data_dir": info.get("data_dir"),
            "certified": info.get("certified", False),
            "receipt_id": (
                f"sha256:{hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:16]}"
            ),
            "files_digest": files_digest,
            "validated_at": receipt.get("validated_at"),
            "receipt_overall": receipt.get("overall"),
            "role": receipt.get("role"),
            "dataset_build_dir": receipt.get("dataset_dir"),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "entry": "python.train.train_supervised",
        "created_at": time.time(),
        "git_commit": detect_git_commit(),
        "seed": seed,
        "dataset": {
            "train": _receipt_summary("train"),
            "val": _receipt_summary("val"),
        },
        "model": {
            "num_blocks": num_blocks,
            "num_filters": num_filters,
            "in_channels": 19,
            "num_actions": NUM_ACTIONS,
            "param_count": None,  # filled after model construction
        },
        "optimizer": {
            "type": "AdamW",
            "base_lr": base_lr,
            "weight_decay": 1e-4,
            "fused": device.type == "cuda",
        },
        "schedule": {
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "min_lr": 1e-5,
        },
        "config": {
            "batch_size": batch_size,
            "micro_batch_size": micro_bs,
            "gradient_accumulation_steps": accum_steps,
            "epochs": epochs,
            "max_steps": max_steps,
            "gpu_resident": gpu_resident,
            "tactics_ratio": tactics_ratio,
            "tactics_dir": str(tactics_dir) if tactics_dir else None,
            "data_role": data_role,
            "allow_uncertified_data": allow_uncertified_data,
            "verify_identity": verify_identity,
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
        },
        "host": {
            "platform": _platform.platform(),
            "python": _platform.python_version(),
            # str(): torch.__version__ is a TorchVersion instance, which the
            # weights_only unpickler rejects on reload (SLQ-13 contract).
            "torch": str(torch.__version__),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        },
    }


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
    data_role: Optional[str] = None,
    allow_uncertified_data: bool = False,
    verify_identity: str = "full",
    override_schedule: bool = False,
    checkpoint_identity: str = "auto",
    artifact_ticket: Optional[Any] = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Execute supervised training loop.

    ``seed`` sets Python/numpy/torch RNGs once up front (None = unseeded,
    today's behavior). Seeding is initialization only; no training math changes.
    """
    # ── Consumption gate (audit F-04/F-01): refuse invalid, stale, or
    #    uncertified datasets BEFORE any loader/optimizer work. Each dataset
    #    must satisfy its declared role (a WDL-supervised game dataset whose
    #    every value label is masked fails here), and the loaded bytes must
    #    match a passing validation receipt unless the caller explicitly opts
    #    out with allow_uncertified_data.
    from python.data.dataset_validation import require_validated_dataset

    gate_receipts: dict[str, Any] = {}
    for label, split_dir in (("train", train_dir), ("val", val_dir)):
        gate = require_validated_dataset(
            Path(split_dir),
            expected_role=data_role,
            allow_uncertified=allow_uncertified_data,
            deep=(verify_identity == "full"),
            consumer=f"supervised-trainer[{label}]",
        )
        role_name = gate.role.name if gate.role else "unclassified"
        print(
            f"[data-gate] {label}: role={role_name} certified={gate.certified} "
            f"masked_fraction={gate.masked_fraction:.4f} samples={gate.num_samples:,} "
            f"identity={verify_identity}"
        )
        for warning in gate.warnings:
            print(f"[data-gate] {warning}")
        # Audit SLQ-02/SLQ-15: bind this run to the dataset identity the gate
        # verified, so the checkpoint's training inputs are reconstructable.
        gate_receipts[label] = {
            "data_dir": str(split_dir),
            "certified": gate.certified,
            "role": role_name,
            "masked_fraction": gate.masked_fraction,
            "num_samples": gate.num_samples,
            "receipt": gate.receipt,
        }

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # SA5-03: the configured seed must reach the actual batch samplers, not
    # just the process-global RNGs. This dedicated generator drives the train
    # DataLoader's shuffle / WeightedRandomSampler and the worker base seeds;
    # the training loop reseeds it to seed + epoch at the top of every epoch,
    # making batch order a pure function of (seed, epoch) that a resumed run
    # reproduces from the checkpointed epoch counter without extra state.
    loader_generator: Optional[torch.Generator] = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed))
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

    # Audit SLQ-18: one tactics-ratio default across loader paths. The
    # disk-backed path historically sampled at 0.3 when --tactics-dir was
    # given without --tactics-ratio; the GPU path silently used natural
    # composition. Both now resolve identically before any loader is built.
    if tactics_dir is not None and tactics_ratio is None:
        tactics_ratio = 0.3

    # SA5-03: captured here (not re-derived with isinstance in the loop) so
    # test doubles that substitute the loader classes stay valid.
    using_gpu_resident_loader = gpu_resident and device.type == "cuda"
    if using_gpu_resident_loader:
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
            seed=seed,
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
            # tactics_ratio was resolved above (0.3 default when --tactics-dir
            # is set without an explicit ratio — audit SLQ-18).
            t_ratio = tactics_ratio
            if not (0.0 <= t_ratio <= 1.0):
                raise ValueError(f"tactics_ratio must be between 0.0 and 1.0, got {t_ratio}")
            tactics_ds = BinaryChessDataset(t_path, is_tactical=True)
            if len(tactics_ds) == 0 and t_ratio > 0.0:
                raise ValueError(
                    f"tactics_ratio is {t_ratio} but tactical dataset at {t_path} is empty."
                )
            combined_ds = SourceAwareCombinedDataset(train_ds, tactics_ds, tactics_ratio=t_ratio)
            # Epoch length under ratio sampling stays the historical master-row
            # count (audit SLQ-19: __len__ now reports the combined total, so
            # the sampler's num_samples must be passed explicitly).
            sampler = combined_ds.get_sampler(
                num_samples=combined_ds.n_master, generator=loader_generator
            )
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
                        torch.from_numpy(weights),
                        num_samples=len(train_ds),
                        replacement=True,
                        generator=loader_generator,
                    )
                    shuffle = False

        train_loader = DataLoader(
            train_ds,
            batch_size=micro_bs,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            # SA5-03: the seeded generator (None when the run is unseeded)
            # drives both the shuffle permutation and the per-worker base
            # seeds; seed_worker propagates those base seeds into each
            # worker's numpy/python RNGs.
            generator=loader_generator,
            worker_init_fn=seed_worker,
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
        in_channels=NUM_PLANES,
        num_filters=num_filters,
        num_blocks=num_blocks,
        num_actions=NUM_ACTIONS,
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

    # Durable run telemetry (audit SLQ-02/SLQ-15): manifest + JSONL stream
    # written next to the checkpoints, flushed per line.
    run_metadata = _build_run_metadata(
        train_dir=train_dir,
        val_dir=val_dir,
        gate_receipts=gate_receipts,
        seed=seed,
        num_blocks=num_blocks,
        num_filters=num_filters,
        batch_size=batch_size,
        micro_bs=micro_bs,
        accum_steps=accum_steps,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        epochs=epochs,
        max_steps=max_steps,
        gpu_resident=gpu_resident,
        tactics_ratio=tactics_ratio,
        tactics_dir=tactics_dir,
        data_role=data_role,
        allow_uncertified_data=allow_uncertified_data,
        verify_identity=verify_identity,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )
    run_metadata["model"]["param_count"] = sum(p.numel() for p in model.parameters())
    run_metadata.setdefault("run_type", "supervised")
    telemetry = _RunTelemetry(ckpt_dir, run_metadata)
    telemetry.log(
        {
            "event": "run_started",
            "resume_path": str(resume_path) if resume_path else None,
            "resume_weights_only": resume_weights_only,
            "device": str(device),
        }
    )

    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path and Path(resume_path).exists():
        mode_label = "weights-only warm start" if resume_weights_only else "full resume"
        print(f"[MODE] {mode_label} from {resume_path}...")
        logger.info("[MODE] supervised %s from %s", mode_label, resume_path)
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        except (OSError, RuntimeError, KeyError) as e:
            raise RuntimeError(f"Failed to load supervised checkpoint '{resume_path}': {e}") from e
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(f"Checkpoint '{resume_path}' is invalid: missing 'model' weights.")
        _validate_arch(checkpoint, model, num_blocks, num_filters, resume_path)

        # Identity (phase J) and schema (phase I) gates before any state is
        # adopted. "auto" verifies registered artifacts and rejects hash drift,
        # quarantine, and disqualification; unregistered files load by explicit
        # path with a logged note.
        if checkpoint_identity != "off":
            from python.model.checkpoint_registry import verify_checkpoint_identity

            verdict = verify_checkpoint_identity(Path(resume_path), repo_root=_repo_root)
            if verdict["status"] in ("quarantined", "disqualified"):
                raise RuntimeError(
                    f"Checkpoint '{resume_path}' is {verdict['status']} in the checkpoint "
                    f"registry ({verdict.get('detail', 'no detail')}). Refusing to load it."
                )
            if verdict["status"] == "hash_mismatch":
                raise RuntimeError(
                    f"Checkpoint '{resume_path}' hash {str(verdict['sha256'])[:16]}... does not "
                    f"match its registry record ({str(verdict.get('expected'))[:16]}...); the "
                    "file changed after verification. Refusing to load."
                )
            if verdict["status"] == "unregistered" and checkpoint_identity == "required":
                raise RuntimeError(
                    f"Checkpoint '{resume_path}' is not registered in "
                    "docs/checkpoint_registry.json (identity verification is required)."
                )
            logger.info("[IDENTITY] %s", verdict)

        from python.model.checkpoint_schema import (
            RUN_STATE_FAILED,
            describe_run_status,
            validate_checkpoint_schema,
            validate_run_metadata,
        )

        verdict_schema = validate_checkpoint_schema(
            checkpoint,
            path=resume_path,
            expected_family=None if resume_weights_only else "supervised",
            allow_legacy=True,
            require_full_state=not resume_weights_only,
        )
        for warning in verdict_schema.warnings:
            logger.warning("[SCHEMA] %s: %s", Path(resume_path).name, warning)
        predecessor_ledger = (
            checkpoint.get("run_metadata")
            if isinstance(checkpoint.get("run_metadata"), dict)
            else None
        )
        for warning in validate_run_metadata(
            predecessor_ledger, path=resume_path, required=not verdict_schema.is_legacy
        ):
            logger.warning("[RUN-STATUS] %s: %s", Path(resume_path).name, warning)
        logger.info(
            "[RUN-STATUS] %s predecessor: %s",
            Path(resume_path).name,
            describe_run_status(predecessor_ledger),
        )
        if predecessor_ledger and predecessor_ledger.get("run_state") == RUN_STATE_FAILED:
            logger.warning(
                "Checkpoint '%s' records a FAILED run (%s); the new run's ledger preserves "
                "that predecessor status.",
                resume_path,
                predecessor_ledger.get("failure_reason"),
            )
        from python.model.checkpoint_schema import checkpoint_identity as _ckpt_identity

        run_metadata["parent_checkpoint"] = _ckpt_identity(resume_path, hash_file=True)
        run_metadata["predecessor_run"] = predecessor_ledger
        run_metadata["run_type"] = (
            "supervised-warm-start" if resume_weights_only else "supervised-resume"
        )

        # Dataset identity compatibility check (DI-04 / P1-5)
        saved_dataset = None
        if predecessor_ledger and isinstance(predecessor_ledger.get("dataset"), dict):
            saved_dataset = predecessor_ledger.get("dataset")
        elif isinstance(checkpoint.get("provenance"), dict) and isinstance(
            checkpoint["provenance"].get("dataset"), dict
        ):
            saved_dataset = checkpoint["provenance"].get("dataset")

        if saved_dataset and run_metadata.get("dataset"):
            saved_train = saved_dataset.get("train", {}) if isinstance(saved_dataset, dict) else {}
            curr_train = run_metadata["dataset"].get("train", {})
            saved_receipt_id = saved_train.get("receipt_id")
            curr_receipt_id = curr_train.get("receipt_id")
            if saved_receipt_id and curr_receipt_id and saved_receipt_id != curr_receipt_id:
                logger.warning(
                    "[DATASET-MISMATCH] Resumed checkpoint '%s' was trained on dataset receipt %s, "
                    "but current run is gated against dataset receipt %s. Training data differs!",
                    Path(resume_path).name,
                    saved_receipt_id,
                    curr_receipt_id,
                )

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
                    saved_current_step = saved_scheduler.get("current_step", 0)
                    # Audit SLQ-11: a resumed run continues the ORIGINAL
                    # schedule. Restore the checkpoint's schedule geometry
                    # (warmup/total/base/min) instead of silently reshaping the
                    # remaining schedule from the current CLI values; adopting
                    # the CLI geometry is an explicit --override-schedule act.
                    geometry = ("warmup_steps", "total_steps", "base_lr", "min_lr")
                    stored = {k: saved_scheduler.get(k) for k in geometry}
                    if override_schedule or any(v is None for v in stored.values()):
                        if override_schedule:
                            print(
                                "[SCHEDULE] --override-schedule given: adopting CLI schedule "
                                f"geometry (warmup={scheduler.warmup_steps}, "
                                f"total={scheduler.total_steps}, base_lr={scheduler.base_lr}); "
                                "continuing at the checkpoint's step position."
                            )
                        scheduler.current_step = saved_current_step
                    else:
                        cli_geometry = {
                            "warmup_steps": scheduler.warmup_steps,
                            "total_steps": scheduler.total_steps,
                            "base_lr": scheduler.base_lr,
                            "min_lr": scheduler.min_lr,
                        }
                        diffs = {
                            k: f"cli={cli_geometry[k]} ckpt={stored[k]}"
                            for k in geometry
                            if stored[k] != cli_geometry[k]
                        }
                        if diffs:
                            print(
                                "[SCHEDULE] Resuming with the CHECKPOINT's schedule geometry so "
                                "the original schedule is not reshaped (audit SLQ-11). "
                                f"CLI values differ: {diffs}. Pass --override-schedule to "
                                "adopt the CLI geometry deliberately."
                            )
                        scheduler.warmup_steps = stored["warmup_steps"]
                        scheduler.total_steps = stored["total_steps"]
                        scheduler.base_lr = stored["base_lr"]
                        scheduler.min_lr = stored["min_lr"]
                        scheduler.current_step = saved_current_step
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
                    logger.warning(
                        "Checkpoint '%s' torch RNG state could not be restored (%s); continuing "
                        "with fresh torch RNG (RNG continuity is NOT claimed).",
                        resume_path,
                        e,
                    )
            if checkpoint.get("numpy_rng_state") is not None:
                try:
                    _restore_numpy_rng_state(checkpoint["numpy_rng_state"])
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Checkpoint '%s' numpy RNG state could not be restored (%s); continuing "
                        "with fresh numpy RNG.",
                        resume_path,
                        e,
                    )
            if checkpoint.get("python_rng_state") is not None:
                try:
                    random.setstate(checkpoint["python_rng_state"])
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Checkpoint '%s' python RNG state could not be restored (%s); continuing "
                        "with fresh python RNG.",
                        resume_path,
                        e,
                    )
            # CUDA RNG (CKP-12): restored only when the captured device topology
            # matches this run; otherwise the difference is stated explicitly.
            saved_cuda_rng = checkpoint.get("cuda_rng_state")
            if saved_cuda_rng is not None:
                if device.type != "cuda" or not torch.cuda.is_available():
                    logger.info(
                        "Checkpoint '%s' carries CUDA RNG state but this run is CPU-only; "
                        "not restored.",
                        resume_path,
                    )
                else:
                    try:
                        states = [torch.as_tensor(s, dtype=torch.uint8) for s in saved_cuda_rng]
                        if len(states) != torch.cuda.device_count():
                            logger.warning(
                                "Checkpoint '%s' captured CUDA RNG state for %d device(s) but %d "
                                "are present; CUDA RNG NOT restored.",
                                resume_path,
                                len(states),
                                torch.cuda.device_count(),
                            )
                        else:
                            torch.cuda.set_rng_state_all(states)
                            logger.info(
                                "Restored CUDA RNG state for %d device(s) from '%s'",
                                len(states),
                                resume_path,
                            )
                    except (RuntimeError, ValueError, TypeError) as e:
                        logger.warning(
                            "Checkpoint '%s' CUDA RNG state could not be restored (%s).",
                            resume_path,
                            e,
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

    # The run is now unmistakably RUNNING inside every artifact it writes
    # (CKP-01): a checkpoint taken from here on can never be read as CREATED.
    from python.model.checkpoint_schema import (
        advance_run_state_inplace,
        build_provenance,
        checkpoint_identity as _ckpt_identity,
    )

    advance_run_state_inplace(run_metadata, "running", step=global_step, epoch=start_epoch)
    provenance = build_provenance(
        run_id=run_metadata.get("run_id"),
        run_type=str(run_metadata.get("run_type", "supervised")),
        config=run_metadata.get("config"),
        model_arch=(int(num_blocks), int(num_filters)),
        param_count=run_metadata.get("model", {}).get("param_count"),
        dataset=run_metadata.get("dataset"),
        parent_checkpoint=run_metadata.get("parent_checkpoint"),
        optimizer_config={"type": type(optimizer).__name__},
        scheduler_config=scheduler.state_dict(),
    )

    val_metrics: dict[str, float] = {}
    _run_failure: Optional[BaseException] = None

    try:
        for epoch in range(start_epoch, epochs):
            # SA5-03: align the batch-sampling stream with the epoch. Batch
            # order for epoch e is a pure function of (seed, e), so a resumed
            # run (start_epoch restored from the checkpoint) replays the exact
            # order an uninterrupted run used for the same epochs — no extra
            # sampler state needs to travel in the checkpoint. When seed is
            # None both calls are inert and the historical unseeded behavior
            # is preserved.
            if loader_generator is not None:
                loader_generator.manual_seed(int(seed) + int(epoch))
            if using_gpu_resident_loader:
                train_loader.set_epoch(epoch)
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
                window_grad_diag: Optional[dict[str, float]] = None
                window_skip = False
                for m_idx, b in enumerate(window):
                    is_first = m_idx == 0
                    is_last = m_idx == w_size - 1
                    tot, pol, wdl, t1, t3, ent, skip, grad_diag = train_step(
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
                    if not skip:
                        window_losses.append((tot, pol, wdl, t1, t3, ent))
                    if grad_diag is not None:
                        window_grad_diag = grad_diag
                    if skip:
                        window_skip = True
                        # Audit SLQ-16: a non-finite micro-batch aborts the rest
                        # of the window, clears the partially accumulated
                        # gradients immediately, and this window performs no
                        # optimizer step. The next window starts from a clean
                        # gradient state.
                        optimizer.zero_grad(set_to_none=True)
                        break

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

                if window_losses:
                    avg_tot = sum(x[0] for x in window_losses) / len(window_losses)
                    avg_pol = sum(x[1] for x in window_losses) / len(window_losses)
                    avg_wdl = sum(x[2] for x in window_losses) / len(window_losses)
                    avg_top1 = sum(x[3] for x in window_losses) / len(window_losses)
                    avg_top3 = sum(x[4] for x in window_losses) / len(window_losses)
                    avg_ent = sum(x[5] for x in window_losses) / len(window_losses)
                else:
                    avg_tot = avg_pol = avg_wdl = avg_top1 = avg_top3 = avg_ent = 0.0

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

                # Durable telemetry (audit SLQ-02): every optimizer window,
                # including the value-head gradient diagnostics proving WDL
                # supervision reaches the value head. Short runs and interrupted
                # runs keep a complete step-level record.
                if True:
                    telemetry.log(
                        {
                            "event": "train_step",
                            "step": global_step,
                            "epoch": epoch,
                            "loss": avg_tot,
                            "policy_loss": avg_pol,
                            "wdl_loss": avg_wdl,
                            "top1": avg_top1,
                            "top3": avg_top3,
                            "entropy": avg_ent,
                            "lr": lr,
                            "samples_per_sec": sps,
                            "window_valid_wdl": v_total,
                            "window_size": w_size,
                            "window_skipped": window_skip,
                            "grad": window_grad_diag,
                        }
                    )

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
                        run_metadata=run_metadata,
                        provenance=provenance,
                        artifact_ticket=artifact_ticket,
                    )
                    telemetry.checkpoint_event(
                        ckpt_dir / f"checkpoint_step_{global_step}.pt", "periodic", global_step
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
                        run_metadata=run_metadata,
                        provenance=provenance,
                        artifact_ticket=artifact_ticket,
                    )
                    telemetry.checkpoint_event(ckpt_dir / "best_model.pt", "best", global_step)
                telemetry.log(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "step": global_step,
                        "val": val_metrics,
                        "epoch_samples": epoch_samples,
                        "max_steps_reached": True,
                    }
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
            writer.flush()

            # Durable per-epoch telemetry (audit SLQ-02) — val metrics are
            # recomputable from the saved checkpoint + the receipt-identified
            # val split recorded in the manifest.
            telemetry.log(
                {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "step": global_step,
                    "val": val_metrics,
                    "epoch_samples": epoch_samples,
                }
            )

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
                    run_metadata=run_metadata,
                    provenance=provenance,
                    artifact_ticket=artifact_ticket,
                )
                telemetry.checkpoint_event(ckpt_dir / "best_model.pt", "best", global_step)

            if max_steps is not None and global_step >= max_steps:
                break

    except KeyboardInterrupt:
        print("\n[PAUSED] Training interrupted by user. Saving emergency checkpoint...")
        current_epoch = epoch if "epoch" in locals() else start_epoch
        # The interrupted status is stamped BEFORE the emergency save so the
        # artifact itself carries "interrupted" (CKP-01 §6.4) instead of the
        # "running" state it would inherit from the periodic save path.
        advance_run_state_inplace(
            run_metadata,
            "interrupted",
            step=int(global_step),
            epoch=int(current_epoch),
            reason="KeyboardInterrupt",
        )
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
            run_metadata=run_metadata,
            provenance=provenance,
            artifact_ticket=artifact_ticket,
        )
        telemetry.checkpoint_event(ckpt_dir / "latest_checkpoint.pt", "interrupted", global_step)
        raise
    except BaseException as exc:  # noqa: BLE001 - the ledger must record any failure
        advance_run_state_inplace(
            run_metadata,
            "failed",
            step=int(locals().get("global_step", start_step) or 0),
            reason=f"{type(exc).__name__}: {exc}",
        )
        try:
            telemetry.log({"event": "run_failed", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            logger.debug("telemetry failure log failed", exc_info=True)
        raise
    finally:
        # Save final/latest checkpoint on clean completion (fixes missing final
        # checkpoint bug). A failure here is NEVER silent (finding CKP-07): it is
        # logged, recorded as an unsuccessful persistence event, reflected in the
        # run-status ledger, and re-raised so the run cannot look successful.
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
        save_error: Optional[BaseException] = None
        try:
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
                    run_metadata=run_metadata,
                    provenance=provenance if "provenance" in locals() else None,
                    artifact_ticket=artifact_ticket,
                )
                telemetry.checkpoint_event(ckpt_dir / "latest_checkpoint.pt", "final", final_step)
        except BaseException as exc:  # noqa: BLE001 - must not be swallowed
            save_error = exc
            logger.exception("Final supervised checkpoint save FAILED: %s", exc)
            try:
                telemetry.log(
                    {
                        "event": "checkpoint_saved_failed",
                        "reason": "final",
                        "path": str(ckpt_dir / "latest_checkpoint.pt"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                logger.debug("telemetry failure record failed", exc_info=True)

        in_flight = sys.exc_info()[1]
        if save_error is not None:
            final_status = "failed"
            failure_reason = (
                f"final checkpoint save failed: {type(save_error).__name__}: {save_error}"
            )
        elif isinstance(in_flight, KeyboardInterrupt):
            final_status = "interrupted"
            failure_reason = None
        elif in_flight is not None:
            final_status = "failed"
            failure_reason = f"{type(in_flight).__name__}: {in_flight}"
        else:
            final_status = "completed"
            failure_reason = None
        try:
            advance_run_state_inplace(
                run_metadata,
                final_status,
                step=int(final_step or 0),
                epoch=int(final_epoch or 0),
                reason=failure_reason,
            )
        except Exception:  # pragma: no cover - ledger already terminal
            logger.debug("could not advance run ledger to %s", final_status, exc_info=True)
        try:
            telemetry.finalize(
                final_status,
                extra={
                    "final_step": final_step,
                    "best_val_loss": best_val_loss if "best_val_loss" in locals() else None,
                    "val_metrics": val_metrics if "val_metrics" in locals() else None,
                    "final_save_error": (
                        None if save_error is None else f"{type(save_error).__name__}: {save_error}"
                    ),
                },
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

        # A run whose required final persistence failed must not return success
        # to its caller (CKP-07). When an exception is already propagating it
        # keeps priority (it is at least as severe); otherwise the save failure
        # is raised here with the recovery diagnostic.
        if save_error is not None and sys.exc_info()[1] is None:
            raise RuntimeError(
                f"Training finished, but the final checkpoint could not be saved to "
                f"'{ckpt_dir / 'latest_checkpoint.pt'}': {type(save_error).__name__}: "
                f"{save_error}. The previous valid checkpoint (if any) was left untouched; "
                "check free disk space, directory permissions, and for another process "
                "holding the artifact lock, then re-run."
            ) from save_error

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
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load only the model weights from --resume (fresh optimizer, step 0) for "
        "fine-tuning. Default: full resume (optimizer/scheduler/step/RNG state).",
    )
    parser.add_argument(
        "--override-schedule",
        action="store_true",
        help="On resume, adopt the CLI's LR/warmup/total-steps geometry instead of the "
        "checkpoint's original schedule (logged loudly). Default: continue the original "
        "schedule at the checkpoint's step position (audit SLQ-11).",
    )
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
    parser.add_argument(
        "--data-role",
        choices=("games_wdl", "policy_tactics", "endgame_syzygy", "unified"),
        default=None,
        help="Explicit dataset role for train/val (default: resolved from dataset metadata "
        "declarations; unclassified datasets are refused).",
    )
    parser.add_argument(
        "--allow-uncertified-data",
        action="store_true",
        help="Consume datasets without a valid validation receipt (loud UNCERTIFIED warning; "
        "role/label invariants still apply). Default: refuse.",
    )
    parser.add_argument(
        "--verify-identity",
        choices=("full", "stat"),
        default="full",
        help="full: re-hash every dataset file against the validation receipt before training; "
        "stat: cheaper size/mtime staleness check.",
    )

    args = parser.parse_args()

    if args.train_dir == default_train:
        logger.warning(
            "[DATASET-FALLBACK] No explicit --train-dir provided; using auto-detected path: %s. "
            "Pass --train-dir explicitly to avoid retargeting runs across dataset builds (DI-30).",
            args.train_dir,
        )
    if args.val_dir == default_val:
        logger.warning(
            "[DATASET-FALLBACK] No explicit --val-dir provided; using auto-detected path: %s. "
            "Pass --val-dir explicitly to avoid retargeting runs across dataset builds (DI-30).",
            args.val_dir,
        )

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
            data_role=args.data_role,
            allow_uncertified_data=args.allow_uncertified_data,
            verify_identity=args.verify_identity,
            seed=args.seed,
            resume_weights_only=args.resume_weights_only,
            override_schedule=args.override_schedule,
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
