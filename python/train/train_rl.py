"""Reinforcement Learning Trainer for Self-Play ZeroPawn.

Contract Specifications:
- Continuous training loop sampling from the memory-mapped replay buffer.
- Policy cross-entropy loss applied ONLY to positions with is_full_search == True.
- WDL cross-entropy loss applied to ALL positions.
- SGD with momentum 0.9 and weight decay 1e-4.
- Learning rate schedule: Warmup (500 steps) + Cosine Annealing (lr_0 = 0.01 -> 1e-4; warm-start recommendation: base_lr=0.0005 -> min_lr=1e-5).
- Pushes new weights to the inference server every N steps (configurable, default 200).
- Checkpoints every 30 minutes, fully resumable (warm-start recommended LR: 0.0005 -> 1e-5).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import queue
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from python.selfplay.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def _atomic_torch_save(state: Dict[str, Any], path: Path) -> None:
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


def _validate_arch(ckpt: Dict[str, Any], model: nn.Module, path: Union[str, Path]) -> None:
    """Friendly arch check: ckpt num_blocks/num_filters vs model and state_dict."""
    model_blocks = getattr(model, "num_blocks", None)
    model_filters = getattr(model, "num_filters", None)
    ckpt_blocks = ckpt.get("num_blocks", None)
    ckpt_filters = ckpt.get("num_filters", None)
    if ckpt_blocks is not None and model_blocks is not None and ckpt_blocks != model_blocks:
        raise RuntimeError(
            f"Architecture mismatch loading '{path}': checkpoint num_blocks={ckpt_blocks} "
            f"!= model num_blocks={model_blocks}. Train with a matching --num-blocks."
        )
    if ckpt_filters is not None and model_filters is not None and ckpt_filters != model_filters:
        raise RuntimeError(
            f"Architecture mismatch loading '{path}': checkpoint num_filters={ckpt_filters} "
            f"!= model num_filters={model_filters}. Train with a matching --num-filters."
        )
    state = ckpt.get("model")
    if isinstance(state, dict):
        block_ids = set()
        for k in state.keys():
            if k.startswith("blocks."):
                try:
                    block_ids.add(int(str(k).split(".")[1]))
                except (ValueError, IndexError):
                    continue
        if block_ids and ckpt_blocks is not None and len(block_ids) != int(ckpt_blocks):
            raise RuntimeError(
                f"Checkpoint '{path}' is inconsistent: header num_blocks={ckpt_blocks} "
                f"but state_dict contains {len(block_ids)} residual blocks."
            )
        stem = state.get("stem_conv.weight")
        if stem is not None and hasattr(stem, "shape") and ckpt_filters is not None:
            try:
                if int(stem.shape[0]) != int(ckpt_filters):  # type: ignore[index]
                    raise RuntimeError(
                        f"Checkpoint '{path}' is inconsistent: header num_filters={ckpt_filters} "
                        f"but stem width is {int(stem.shape[0])}."  # type: ignore[index]
                    )
            except (ValueError, TypeError, IndexError) as e:
                logger.debug("Skipping stem width validation for '%s': %s", path, e)


class WarmupCosineDecayScheduler:
    """Linear warmup followed by cosine annealing learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 500,
        total_steps: int = 100_000,
        base_lr: float = 0.01,
        min_lr: float = 1e-4,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (step / max(1, self.warmup_steps))
        else:
            progress = min(
                1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def step(self, step: int) -> float:
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: Dict[str, Any], override_lr: bool = False) -> None:
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        if not override_lr:
            self.base_lr = state["base_lr"]
            self.min_lr = state["min_lr"]


class RLTrainer:
    """Continuous RL trainer for AlphaZero self-play."""

    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        device: Optional[Union[str, torch.device]] = None,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        total_steps: int = 100_000,
        min_lr: float = 1e-4,
        batch_size: int = 256,
        push_freq: int = 200,
        checkpoint_interval_sec: float = 1800.0,  # 30 minutes
        checkpoint_dir: Union[str, Path] = "models/checkpoints_rl",
        sync_weights_path: Union[str, Path] = "models/checkpoints_rl/live_weights.pt",
        reload_queue: Optional[Any] = None,
        direct_server: Optional[Any] = None,
        use_bfloat16: bool = True,
        gpu_cache_capacity: Optional[int] = None,
        policy_target_temp: float = 0.5,
        entropy_bonus: float = 0.001,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.device_type: str = self.device.type

        self.model = model.to(self.device)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.push_freq = push_freq
        self.checkpoint_interval_sec = checkpoint_interval_sec
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sync_weights_path = Path(sync_weights_path)
        self.sync_weights_path.parent.mkdir(parents=True, exist_ok=True)

        self.reload_queue = reload_queue
        self.direct_server = direct_server
        self.use_bfloat16 = use_bfloat16 and self.device.type == "cuda"

        if gpu_cache_capacity is not None:
            self.gpu_cache_capacity = int(gpu_cache_capacity)
        elif self.device.type == "cuda" and torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (
                    1024**3
                )
                if total_vram_gb >= 11.0:
                    self.gpu_cache_capacity = 250_000
                elif total_vram_gb >= 7.5:
                    self.gpu_cache_capacity = 160_000
                elif total_vram_gb >= 5.5:
                    self.gpu_cache_capacity = 100_000
                else:
                    self.gpu_cache_capacity = 40_000
            except Exception:
                self.gpu_cache_capacity = 160_000
        else:
            self.gpu_cache_capacity = 100_000

        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.policy_target_temp = float(policy_target_temp)
        self.entropy_bonus = float(entropy_bonus)

        # Optimizer: SGD with momentum 0.9 and weight decay 1e-4
        self.optimizer = SGD(
            self.model.parameters(),
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Learning rate schedule
        self.scheduler = WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=min_lr,
        )

        self.step_num = 0
        self.last_checkpoint_time = time.time()
        self.last_push_step = 0

        # Dedicated non-default CUDA stream for training pipeline
        if self.device.type == "cuda":
            try:
                self.train_stream = torch.cuda.Stream(device=self.device, priority=0)
            except Exception:
                self.train_stream = torch.cuda.Stream(device=self.device)
        else:
            self.train_stream = None

        # Enable Tier 1 GPU cache on replay buffer if running on CUDA and available
        if self.device.type == "cuda" and hasattr(self.replay_buffer, "enable_gpu_cache"):
            if getattr(self.replay_buffer, "gpu_cache", None) is None:
                gpu_cap = min(
                    self.gpu_cache_capacity,
                    getattr(self.replay_buffer, "capacity", self.gpu_cache_capacity),
                )
                self.replay_buffer.enable_gpu_cache(capacity=gpu_cap, device=self.device)

    def train_step(self, batch_dict: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Execute one training step on a sampled or provided batch (with batch_dict provided externally, tactical injection only when sampling internally is supported)."""
        self.model.train()

        actual_mix = 0.0
        actual_tactical_mix = 0.0
        if batch_dict is None:
            # 1. Sample mini-batch from replay buffer (mixing human data via taper schedule)
            if hasattr(self.replay_buffer, "sample_batch_tensors"):
                batch_dict, actual_mix = self.replay_buffer.sample_batch_tensors(
                    batch_size=self.batch_size,
                    step=self.step_num,
                    device=self.device,
                )
            else:
                batch_dict, actual_mix = self.replay_buffer.sample_batch(
                    batch_size=self.batch_size,
                    step=self.step_num,
                )
            actual_tactical_mix = getattr(self.replay_buffer, "last_tactical_mix", 0.0)
        else:
            actual_tactical_mix = float(batch_dict.get("actual_tactical_mix", 0.0))

        # Use synchronous H2D transfer when dedicated train_stream is active
        # (eliminates cross-stream race between default-stream copies and train_stream reads).
        sync_transfer = torch.cuda.is_available() and self.train_stream is not None
        if isinstance(batch_dict["planes"], torch.Tensor):
            states_t = batch_dict["planes"].to(self.device, non_blocking=not sync_transfer)
            visits_t = batch_dict["visits"].to(self.device, non_blocking=not sync_transfer)
            outcomes_t = batch_dict["outcomes"].to(self.device, non_blocking=not sync_transfer)
            is_full_t = batch_dict["is_full_search"].to(self.device, non_blocking=not sync_transfer)
        else:
            states_t = torch.from_numpy(batch_dict["planes"]).to(
                self.device, non_blocking=not sync_transfer
            )
            visits_t = torch.from_numpy(batch_dict["visits"]).to(
                self.device, non_blocking=not sync_transfer
            )
            outcomes_t = torch.from_numpy(batch_dict["outcomes"]).to(
                self.device, non_blocking=not sync_transfer
            )
            is_full_t = torch.from_numpy(batch_dict["is_full_search"]).to(
                self.device, non_blocking=not sync_transfer
            )

        # 2. Update learning rate
        current_lr = self.scheduler.step(self.step_num)
        self.optimizer.zero_grad(set_to_none=True)

        stream_ctx = (
            torch.cuda.stream(self.train_stream)
            if self.train_stream is not None
            else contextlib.nullcontext()
        )
        with stream_ctx:
            # 3. Forward pass
            if self.use_bfloat16:
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    p_logits, w_logits = self.model(states_t)
            else:
                p_logits, w_logits = self.model(states_t)

            # 4. Policy Loss: Cross-entropy ONLY on full-search positions.
            # Standard AlphaZero cross-entropy over the action space:
            # - For one-hot human targets, this matches supervised cross-entropy.
            # - For MCTS visit targets, this matches AlphaZero (pi * log p), driving p -> pi
            #   while penalizing unvisited and illegal actions.
            full_mask = is_full_t
            if full_mask.any():
                full_p_logits = p_logits[full_mask].float()
                full_visits = visits_t[full_mask].float()
                with torch.no_grad():
                    p_target_norm = full_visits / full_visits.sum(dim=-1, keepdim=True).clamp_min(
                        1e-8
                    )
                    target_entropy_t = (
                        -(p_target_norm * (p_target_norm + 1e-12).log()).sum(dim=-1).mean()
                    )
                log_probs = F.log_softmax(full_p_logits, dim=-1)
                if self.policy_target_temp != 1.0:
                    tau = self.policy_target_temp
                    sharp = full_visits.clamp_min(0) ** (1.0 / tau)
                    target = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                else:
                    target = full_visits
                policy_loss = -(target * log_probs).sum(dim=-1).mean()
            else:
                policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                target_entropy_t = torch.tensor(0.0, device=self.device)

            # 5. Value Loss: Cross-entropy on ALL positions (WDL head)
            # ignore_index=255 guards against OUTCOME_UNAVAILABLE in tactical/human data;
            # if an entire batch has outcome=255, guard against 0/0 NaN in PyTorch reduction.
            # Handle both class probability targets (N, 3) and class index targets (N,) or (N, 1).
            if (
                outcomes_t.is_floating_point()
                and outcomes_t.ndim >= 2
                and outcomes_t.shape[-1] == 3
            ):
                wdl_loss = F.cross_entropy(w_logits.float(), outcomes_t)
            else:
                outcomes_idx = outcomes_t.long()
                if outcomes_idx.ndim > 1 and outcomes_idx.shape[-1] == 1:
                    outcomes_idx = outcomes_idx.squeeze(-1)
                valid_outcomes = outcomes_idx != 255
                if valid_outcomes.any():
                    wdl_loss = F.cross_entropy(w_logits.float(), outcomes_idx, ignore_index=255)
                else:
                    wdl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # 6. Combined total loss with small entropy regularization (prevents sharp collapse)
            # Entropy bonus: maximize entropy of the policy head on full-search positions.
            entropy_reg = 0.0
            if full_mask.any():
                probs_full = F.softmax(full_p_logits, dim=-1)
                entropy_reg = -(probs_full * torch.log(probs_full + 1e-12)).sum(dim=-1).mean()
            total_loss = policy_loss + 0.5 * wdl_loss - self.entropy_bonus * entropy_reg

            # 7. Backward pass and optimizer step
            if not torch.isfinite(total_loss.detach()):
                logger.warning(
                    "Non-finite total_loss at step %d; skipping optimizer step", self.step_num
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.step_num += 1
                return {
                    "total_loss": float(total_loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "wdl_loss": float(wdl_loss.item()),
                    "policy_entropy": float("nan"),
                    "target_entropy": float("nan"),
                    "actual_human_mix": actual_mix,
                    "actual_tactical_mix": actual_tactical_mix,
                    "lr": current_lr,
                    "step": self.step_num,
                }
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 8. Compute policy entropy metric
            with torch.no_grad():
                probs = F.softmax(p_logits.float(), dim=-1)
                policy_entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()

            # Bottleneck F elimination: batch scalar transfers in a single host-device sync
            metrics_t = (
                torch.stack(
                    [
                        total_loss.detach(),
                        policy_loss.detach(),
                        wdl_loss.detach(),
                        policy_entropy_t.detach(),
                        target_entropy_t.detach(),
                    ]
                )
                .cpu()
                .tolist()
            )

        self.step_num += 1

        # 9. Weight push to inference server every N steps
        if self.step_num % self.push_freq == 0:
            self.push_weights()

        # 10. Checkpoint every 30 minutes
        now = time.time()
        if now - self.last_checkpoint_time >= self.checkpoint_interval_sec:
            self.save_checkpoint()
            if hasattr(self.replay_buffer, "flush_to_disk"):
                self.replay_buffer.flush_to_disk()
            self.last_checkpoint_time = now

        # Periodically refresh human cache with fresh random samples
        if self.step_num > 0 and self.step_num % 10_000 == 0:
            if hasattr(self.replay_buffer, "refresh_human_cache"):
                self.replay_buffer.refresh_human_cache()

        return {
            "total_loss": float(metrics_t[0]),
            "policy_loss": float(metrics_t[1]),
            "wdl_loss": float(metrics_t[2]),
            "policy_entropy": float(metrics_t[3]),
            "target_entropy": float(metrics_t[4]),
            "actual_human_mix": actual_mix,
            "actual_tactical_mix": actual_tactical_mix,
            "lr": current_lr,
            "step": self.step_num,
        }

    def push_weights(self) -> None:
        """Push current model weights to the inference server."""
        num_blocks = getattr(self.model, "num_blocks", 12)
        num_filters = getattr(self.model, "num_filters", 128)

        # Record synchronization event on training stream
        sync_event = None
        if self.train_stream is not None:
            sync_event = self.train_stream.record_event()

        # Direct in-memory synchronization if direct_server is connected
        if self.direct_server is not None:
            if hasattr(self.direct_server, "load_state_dict_weights"):
                self.direct_server.load_state_dict_weights(
                    self.model.state_dict(), sync_event=sync_event
                )
            else:
                self.direct_server.load_checkpoint(
                    str(self.sync_weights_path), sync_event=sync_event
                )

        payload = {
            "model": self.model.state_dict(),
            "step": self.step_num,
            "num_blocks": num_blocks,
            "num_filters": num_filters,
        }
        # Atomic same-dir write (no TOCTOU exists-check before replace).
        tmp_path = self.sync_weights_path.with_name(self.sync_weights_path.name + ".tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self.sync_weights_path)
        except OSError as e:
            logger.warning("Failed to push weights to %s: %s", self.sync_weights_path, e)
            try:
                tmp_path.unlink()
            except OSError:
                logger.debug("Failed to clean tmp weights file %s", tmp_path)
            return

        # Notify inference server
        if self.reload_queue is not None:
            try:
                self.reload_queue.put(str(self.sync_weights_path), timeout=5)
            except TypeError:
                # Queue-like object without timeout support.
                try:
                    self.reload_queue.put(str(self.sync_weights_path))
                except (OSError, RuntimeError) as e:
                    logger.debug("reload_queue notification failed: %s", e)
            except queue.Full:
                logger.warning("reload_queue full; skipping notify for step %d", self.step_num)

        self.last_push_step = self.step_num
        logger.info(f"Pushed weights to inference server at step {self.step_num}")
        return sync_event

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None, keep_last_n: int = 3) -> str:
        """Save fully resumable checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_step_{self.step_num}.pt"
        else:
            path = Path(path)

        num_blocks = getattr(self.model, "num_blocks", 12)
        num_filters = getattr(self.model, "num_filters", 128)

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step_num,
            "num_blocks": num_blocks,
            "num_filters": num_filters,
            "buffer_size": len(self.replay_buffer),
            "buffer_write_head": getattr(self.replay_buffer, "write_head", 0),
            "tactical_pool_size": getattr(self.replay_buffer, "tactical_pool_size", 0),
            "policy_target_temp": self.policy_target_temp,
            "entropy_bonus": self.entropy_bonus,
            "last_push_step": self.last_push_step,
            "model_version": getattr(self.direct_server, "model_version", 1),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": _safe_numpy_rng_state(),
            "python_rng_state": random.getstate(),
        }

        # Single write, then copy to latest via replace (not 2x torch.save).
        _atomic_torch_save(ckpt, path)
        # Also maintain a symlink/copy to latest_checkpoint.pt
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        latest_tmp = latest_path.with_name(latest_path.name + ".tmp")
        try:
            with open(path, "rb") as fsrc, open(latest_tmp, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            os.replace(latest_tmp, latest_path)
        except OSError as e:
            logger.warning("Failed to update %s: %s", latest_path, e)
            try:
                latest_tmp.unlink()
            except OSError:
                logger.debug("Failed to clean tmp checkpoint file %s", latest_tmp)
        logger.info(f"Saved resumable checkpoint to {path}")
        if keep_last_n >= 0:
            try:
                from python.common.cleanup import prune_checkpoints

                pruned = prune_checkpoints(self.checkpoint_dir, keep_last_n=keep_last_n)
                if pruned["deleted"]:
                    logger.info(
                        "Pruned %d old checkpoint(s), freed %.1f MB",
                        pruned["deleted"],
                        pruned["freed_bytes"] / (1024 * 1024),
                    )
            except (OSError, ValueError) as e:
                logger.debug("checkpoint prune skipped: %s", e)
        return str(path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        scheduler_override: bool = True,
    ) -> int:
        """Resume training from a checkpoint."""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, KeyError) as e:
            raise RuntimeError(f"Failed to load RL checkpoint '{path}': {e}") from e
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Checkpoint '{path}' is invalid: missing 'model' weights.")
        _validate_arch(ckpt, self.model, path)
        self.model.load_state_dict(ckpt.get("model"))
        if ckpt.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (RuntimeError, KeyError, ValueError) as e:
                raise RuntimeError(
                    f"Checkpoint '{path}' has an incompatible optimizer state: {e}"
                ) from e
        if ckpt.get("scheduler") is not None and hasattr(self.scheduler, "load_state_dict"):
            try:
                saved_sched = ckpt["scheduler"]
                if scheduler_override:
                    saved_base_lr = saved_sched.get("base_lr")
                    saved_min_lr = saved_sched.get("min_lr")
                    cur_base_lr = getattr(self.scheduler, "base_lr", getattr(self, "base_lr", None))
                    cur_min_lr = getattr(self.scheduler, "min_lr", getattr(self, "min_lr", None))
                    if (
                        saved_base_lr is not None
                        and cur_base_lr is not None
                        and saved_base_lr != cur_base_lr
                    ) or (
                        saved_min_lr is not None
                        and cur_min_lr is not None
                        and saved_min_lr != cur_min_lr
                    ):
                        logger.warning(
                            "Ignoring saved scheduler lr (base_lr=%s, min_lr=%s); "
                            "explicit config takes precedence (base_lr=%s, min_lr=%s).",
                            saved_base_lr,
                            saved_min_lr,
                            cur_base_lr,
                            cur_min_lr,
                        )
                    self.scheduler.load_state_dict(saved_sched, override_lr=True)
                    if cur_base_lr is not None:
                        self.scheduler.base_lr = cur_base_lr
                    if cur_min_lr is not None:
                        self.scheduler.min_lr = cur_min_lr
                else:
                    self.scheduler.load_state_dict(saved_sched)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("Ignoring incompatible scheduler state in '%s': %s", path, e)
        self.step_num = ckpt.get("step", 0)
        self.last_push_step = ckpt.get("last_push_step", self.step_num)
        if ckpt.get("policy_target_temp") is not None:
            saved_temp = float(ckpt["policy_target_temp"])
            if saved_temp != self.policy_target_temp:
                logger.info(
                    "Overriding saved policy_target_temp (%s) with explicit config (%s)",
                    saved_temp,
                    self.policy_target_temp,
                )
        if ckpt.get("entropy_bonus") is not None:
            saved_entropy_bonus = float(ckpt["entropy_bonus"])
            if saved_entropy_bonus != self.entropy_bonus:
                logger.info(
                    "Overriding saved entropy_bonus (%s) with explicit config (%s)",
                    saved_entropy_bonus,
                    self.entropy_bonus,
                )
        if scheduler_override:
            self.scheduler.step(self.step_num)
        if "model_version" in ckpt and self.direct_server is not None:
            try:
                self.direct_server.model_version = int(ckpt["model_version"])
            except Exception:
                pass
        if ckpt.get("buffer_write_head") is not None and hasattr(self.replay_buffer, "write_head"):
            try:
                head = int(ckpt["buffer_write_head"])
                cap = getattr(self.replay_buffer, "capacity", None)
                if cap and cap > 0:
                    head = head % cap
                self.replay_buffer.write_head = head
            except (ValueError, TypeError) as e:
                logger.debug("Ignoring invalid buffer_write_head in '%s': %s", path, e)
        if ckpt.get("rng_state") is not None:
            try:
                rng_s = ckpt["rng_state"]
                if hasattr(rng_s, "cpu"):
                    rng_s = rng_s.cpu()
                torch.set_rng_state(rng_s)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug("Ignoring incompatible rng_state in '%s': %s", path, e)
        if ckpt.get("numpy_rng_state") is not None:
            try:
                _restore_numpy_rng_state(ckpt["numpy_rng_state"])
            except (ValueError, TypeError) as e:
                logger.debug("Ignoring incompatible numpy_rng_state in '%s': %s", path, e)
        if ckpt.get("python_rng_state") is not None:
            try:
                random.setstate(ckpt["python_rng_state"])
            except (ValueError, TypeError) as e:
                logger.debug("Ignoring incompatible python_rng_state in '%s': %s", path, e)
        logger.info(f"Resumed training from {path} at step {self.step_num}")
        return self.step_num
