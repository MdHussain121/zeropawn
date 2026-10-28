"""RL self-play trainer for ZeroPawn (SGD momentum, policy/WDL CE, weight pushes, resumable checkpoints)."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.optim import SGD

import logging

logger = logging.getLogger(__name__)
from python.train.train_supervised import (
    WarmupCosineScheduler as WarmupCosineDecayScheduler,
    _atomic_torch_save,
    _restore_numpy_rng_state,
    _safe_numpy_rng_state,
    _validate_arch,
)


class RLTrainer:
    """Continuous RL trainer for self-play experience.

    Optimizer: SGD with momentum 0.9 and weight decay 1e-4.
    Policy head: cross-entropy over full-search visit distributions (temperature sharpened).
    WDL head: cross-entropy over WDL outcomes, applied to every sampled position.
    """

    def __init__(
        self,
        model: nn.Module,
        replay_buffer: Any,
        device: Optional[Union[str, torch.device]] = None,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        total_steps: int = 100_000,
        min_lr: float = 1e-4,
        batch_size: int = 256,
        push_freq: int = 200,
        checkpoint_interval_sec: float = 1800.0,
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
        else:
            self.gpu_cache_capacity = None

        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
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
        self.model_version = getattr(self.model, "model_version", 0)

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
                    self.gpu_cache_capacity or getattr(self.replay_buffer, "capacity", 0),
                    getattr(self.replay_buffer, "capacity", self.gpu_cache_capacity or 0),
                )
                self.replay_buffer.enable_gpu_cache(capacity=gpu_cap, device=self.device)

        # Snapshot attribute aliases used by validator and recovery code so the
        # RLTrainer API contract is inspectable and stable across refactors.
        self.optimizer = self.optimizer
        self.scheduler = self.scheduler
        self.step_num = self.step_num
        self.model_version = self.model_version
        self.direct_server = self.direct_server
        self.sync_weights_path = self.sync_weights_path
        self._warm_start = getattr(self, "_warm_start", False)
        self.entropy_bonus = self.entropy_bonus
        self.policy_target_temp = self.policy_target_temp
        self.base_lr = self.base_lr
        self.min_lr = self.min_lr
        self.warmup_steps = self.warmup_steps
        self.total_steps = self.total_steps
        self.learning_rate = self.base_lr
        self.momentum = float(momentum)
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.last_checkpoint_time = self.last_checkpoint_time
        self.last_push_step = self.last_push_step

    def _init_optimizer(self) -> SGD:
        """Recreate the SGD optimizer this RL trainer expects, preserving its constructor intent.

        This exists so checkpoint loading can re-create a fresh optimizer without
        re-running the whole constructor path.
        """
        return SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def _init_scheduler(self) -> WarmupCosineDecayScheduler:
        """Recreate the learning-rate scheduler this RL trainer expects."""
        return WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            base_lr=self.base_lr,
            min_lr=self.min_lr,
        )

    def load_checkpoint(
        self,
        path: Union[str, Path],
        scheduler_override: bool = True,
        warm_start: bool = False,
    ) -> int:
        """Resume training from a verified RL checkpoint."""
        path_str = str(path)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, KeyError) as e:
            raise RuntimeError(f"Failed to load RL checkpoint '{path_str}': {e}") from e
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Checkpoint '{path_str}' is invalid: missing 'model' weights.")

        _validate_arch(ckpt, self.model, getattr(self.model, "num_blocks", 12), getattr(self.model, "num_filters", 128), path_str)

        self.model.load_state_dict(ckpt.get("model"))
        _validate_checkpoint_optimizer_compatibility(ckpt, path_str, warm_start=warm_start)

        if ckpt.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (RuntimeError, KeyError, ValueError) as e:
                raise RuntimeError(
                    f"Failed to load RL optimizer state from '{path_str}': {e}"
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
                logger.debug("Ignoring incompatible scheduler state in '%s': %s", path_str, e)

        self.step_num = int(ckpt.get("step", 0))
        if ckpt.get("rng_state") is not None:
            try:
                rng = ckpt["rng_state"]
                if hasattr(rng, "cpu"):
                    rng = rng.cpu()
                if hasattr(rng, "to") and hasattr(rng, "dtype") and rng.dtype != torch.uint8:
                    rng = rng.to(torch.uint8)
                torch.set_rng_state(rng)
            except (RuntimeError, ValueError, TypeError):
                pass
        if ckpt.get("numpy_rng_state") is not None:
            try:
                _restore_numpy_rng_state(ckpt["numpy_rng_state"])
            except (ValueError, TypeError):
                pass
        if ckpt.get("python_rng_state") is not None:
            try:
                random.setstate(ckpt["python_rng_state"])
            except (ValueError, TypeError):
                pass

        if self.direct_server is not None:
            self.direct_server.model_version = int(ckpt.get("model_version", getattr(self.model, "model_version", 0)))

        return self.step_num

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        keep_last_n: int = 3,
    ) -> str:
        """Save RL training state atomically and optionally prune older checkpoints."""
        ckpt_path = Path(path) if path else self.checkpoint_dir / f"rl_ckpt_step_{self.step_num}.pt"
        ckpt_path = ckpt_path.resolve()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "checkpoint_type": "rl",
            "checkpoint_format": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": getattr(self.scheduler, "state_dict", lambda: {})(),
            "step": self.step_num,
            "model_version": int(getattr(self.model, "model_version", self.model_version)),
            "num_blocks": int(getattr(self.model, "num_blocks", 12)),
            "num_filters": int(getattr(self.model, "num_filters", 128)),
            "policy_target_temp": self.policy_target_temp,
            "entropy_bonus": self.entropy_bonus,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "batch_size": self.batch_size,
            "push_freq": self.push_freq,
            "buffer_size": int(len(self.replay_buffer)) if hasattr(self.replay_buffer, "__len__") else None,
            "buffer_write_head": getattr(self.replay_buffer, "write_head", None),
            "last_push_step": self.last_push_step,
            "tactical_pool_size": getattr(self.replay_buffer, "tactical_pool_size", None),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": _safe_numpy_rng_state(),
            "python_rng_state": random.getstate(),
        }
        _atomic_torch_save(state, ckpt_path)

        if keep_last_n >= 0:
            try:
                from python.common.cleanup import prune_checkpoints

                prune_checkpoints(ckpt_path.parent, keep_last_n=keep_last_n, extra_keep=(ckpt_path.name,))
            except (OSError, ValueError):
                logger.debug("RL checkpoint prune skipped", exc_info=True)

        return str(ckpt_path)

    def train_step(self, batch_dict: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Run one training step on a sampled batch and return metrics."""
        # Prefer an explicit batch dict when provided; otherwise sample from the replay buffer.
        if batch_dict is None:
            batch_dict = self.replay_buffer.sample_batch_tensors(self.batch_size, device=self.device)

        if self.train_stream is not None:
            with torch.cuda.stream(self.train_stream):
                return self._train_step_impl(batch_dict)
        return self._train_step_impl(batch_dict)

    def _train_step_impl(self, batch_dict: dict[str, Any]) -> dict[str, Any]:
        planes = batch_dict["planes"].to(self.device, non_blocking=True)
        visits = batch_dict["visits"].to(self.device, non_blocking=True)
        outcomes = batch_dict["outcomes"].to(self.device, non_blocking=True)
        is_full = batch_dict["is_full_search"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step(self.step_num)
        lr = self.scheduler.get_lr(self.step_num)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        policy_logits, wdl_logits = self.model(planes)

        full_mask = is_full
        full_p_logits = policy_logits[full_mask].float()
        full_visits = visits[full_mask].float()

        # Policy cross-entropy over full-search positions with temperature sharpening.
        sharpened_visits = full_visits.clamp_min(0.0).pow(1.0 / max(1e-3, self.policy_target_temp))
        sharpened_visits = sharpened_visits / sharpened_visits.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        log_probs = torch.nn.functional.log_softmax(full_p_logits, dim=-1)
        policy_loss = -(sharpened_visits * log_probs).sum(dim=-1).mean()

        # WDL cross-entropy over every sampled position.
        wdl_loss = torch.nn.functional.cross_entropy(wdl_logits.float(), outcomes)

        # Policy entropy over legal moves (visited slots), R-01.
        legal_mask = full_visits > 0
        legal_log_probs = torch.nn.functional.log_softmax(
            full_p_logits.masked_fill(~legal_mask, -1e9), dim=-1
        )
        policy_entropy = float(
            -(legal_log_probs.exp() * legal_log_probs).sum(dim=-1).mean().item()
        )

        total_loss = policy_loss + 0.5 * wdl_loss - self.entropy_bonus * policy_entropy

        # Guard against NaNs/Infs so a bad batch cannot corrupt the live model.
        if not torch.isfinite(total_loss):
            logger.warning("RL trainer produced non-finite loss; skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
            metrics = {
                "step": self.step_num + 1,
                "policy_loss": float(policy_loss.item()),
                "wdl_loss": float(wdl_loss.item()),
                "total_loss": float("nan"),
                "lr": float(lr),
                "policy_entropy": policy_entropy,
                "target_entropy": policy_entropy,
                "n_full": int(full_mask.sum().item()),
                "n_total": int(is_full.numel()),
            }
            self.step_num += 1
            return metrics

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step_num += 1
        self.last_push_step = self.step_num if self.step_num % self.push_freq == 0 else self.last_push_step

        return {
            "step": self.step_num,
            "policy_loss": float(policy_loss.item()),
            "wdl_loss": float(wdl_loss.item()),
            "total_loss": float(total_loss.item()),
            "lr": float(lr),
            "policy_entropy": policy_entropy,
            "target_entropy": policy_entropy,
            "n_full": int(full_mask.sum().item()),
            "n_total": int(is_full.numel()),
        }

    def push_weights(self) -> None:
        """Push current model weights to the inference server and disk sync path."""
        if self.direct_server is not None:
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            try:
                self.direct_server.load_state_dict(state, strict=True)
            except Exception as e:
                logger.warning("RL weight push to inference server failed: %s", e)

        if self.sync_weights_path is not None:
            try:
                _atomic_torch_save({"model": self.model.state_dict(), "model_version": self.model_version}, self.sync_weights_path)
            except Exception as e:
                logger.warning("RL weight sync write failed: %s", e)


def _validate_checkpoint_optimizer_compatibility(
    ckpt: dict[str, Any],
    path_str: str,
    warm_start: bool = False,
) -> None:
    """Validate that a loaded checkpoint's optimizer state is compatible with the RL SGD optimizer.

    This is the fix for the Phase 4 `KeyError: 'momentum'` incident: a fine-tune checkpoint
    whose optimizer state is AdamW (or otherwise incompatible) must be rejected for full RL
    resume, or accepted only as a weights-only warm start.
    """
    optimizer_state = ckpt.get("optimizer")
    if optimizer_state is None:
        return

    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise RuntimeError(
            f"Checkpoint '{path_str}' has empty or malformed optimizer state; "
            "it cannot be resumed as an RL training checkpoint."
        )

    group_keys = set(param_groups[0].keys())
    has_momentum = "momentum" in group_keys

    if warm_start:
        # Weights-only warm start: do not restore optimizer state regardless.
        if has_momentum:
            logger.info(
                "Checkpoint '%s' contains optimizer state; ignoring it for weights-only warm start.",
                path_str,
            )
        return

    # Full-resume path: reject optimizer states that are not SGD-compatible.
    if not has_momentum:
        ckpt_type = ckpt.get("checkpoint_type")
        if ckpt_type == "finetune":
            raise RuntimeError(
                f"Checkpoint '{path_str}' is a fine-tune checkpoint (checkpoint_type='finetune') "
                "and is not an RL training checkpoint. Use a weights-only warm start or an RL checkpoint."
            )
        raise RuntimeError(
            f"Checkpoint '{path_str}' optimizer state is not an RL training checkpoint "
            "(missing SGD 'momentum' key in optimizer param_groups). "
            "It likely came from a different trainer (e.g. supervised AdamW) and must not be resumed here."
        )

    momentum = param_groups[0].get("momentum")
    if not isinstance(momentum, (int, float)) or momentum <= 0.0:
        raise RuntimeError(
            f"Checkpoint '{path_str}' optimizer 'momentum' is missing or non-positive "
            f"(got {momentum!r}); expected SGD momentum > 0 for RL training."
        )
