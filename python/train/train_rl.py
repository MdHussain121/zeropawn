"""RL self-play trainer for ZeroPawn (SGD momentum, policy/WDL CE, weight pushes, resumable checkpoints)."""

from __future__ import annotations

import logging
import os
import pickle
import queue
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

logger = logging.getLogger(__name__)
from python.train.train_supervised import (
    WarmupCosineScheduler as WarmupCosineDecayScheduler,
    _atomic_torch_save,
    _restore_numpy_rng_state,
    _safe_numpy_rng_state,
    _validate_arch,
)

# Finite stand-in for -inf when masking illegal policy logits. Using -inf directly
# makes exp(log_softmax) == 0 and 0 * -inf == NaN in the entropy expression.
_MASKED_LOGIT = -1e9


class WeightPushError(RuntimeError):
    """Raised when the learner cannot publish new weights to the inference server.

    Fail-closed by design (Stage A §4): a push that fails silently would let
    actors keep sampling stale weights while the learner advances, so the
    failure must be observable to the caller.
    """


def _masked_policy_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy (nats) of softmax(logits) restricted to ``mask``.

    ``logits`` is [B, A] and ``mask`` is a bool [B, A] legal-action mask. Illegal
    actions are pushed to ``_MASKED_LOGIT`` so they carry exactly zero probability and
    contribute nothing to the entropy, keeping every intermediate finite (hence
    differentiable).
    """
    log_probs = F.log_softmax(logits.masked_fill(~mask, _MASKED_LOGIT), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1).mean()


def _legal_move_mask(visits: torch.Tensor) -> torch.Tensor:
    """Derive a legal-move mask from MCTS visit counts.

    MCTS only ever visits legal moves, so ``visits > 0`` identifies a subset of the
    legal actions for a position. Rows with no visits at all keep a single slot so
    that ``log_softmax`` stays finite (such rows carry a zero policy target anyway).
    """
    mask = visits > 0
    empty = ~mask.any(dim=-1)
    if bool(empty.any()):
        mask = mask.clone()
        mask[empty, 0] = True
    return mask


def classify_optimizer_state(ckpt: Dict[str, Any]) -> str:
    """Classify a checkpoint's optimizer state without loading it into an optimizer.

    Returns one of ``"SGD"``, ``"AdamW"`` (any Adam-family state), ``"empty"``
    (optimizer present but no per-parameter state yet — always safe), or
    ``"unknown"``. A stamped ``_optimizer_type`` wins; otherwise the per-param
    state keys decide (``exp_avg``/``exp_avg_sq`` are Adam-family,
    ``momentum_buffer`` is SGD). ``None``/absent optimizer state is ``"empty"``.
    """
    opt_state = ckpt.get("optimizer")
    if opt_state is None:
        return "empty"
    stamped = ckpt.get("_optimizer_type")
    if isinstance(stamped, str) and stamped:
        return stamped
    state = opt_state.get("state") if isinstance(opt_state, dict) else None
    if not isinstance(state, dict) or not state:
        return "empty"
    saw_any_keys = False
    for per_param in state.values():
        if isinstance(per_param, dict):
            keys = set(per_param.keys())
            if keys & {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                return "AdamW"
            if "momentum_buffer" in keys:
                return "SGD"
            saw_any_keys = saw_any_keys or bool(keys)
    return "unknown" if saw_any_keys else "empty"


def inspect_rl_checkpoint(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read an RL checkpoint's metadata cheaply, or return None if unusable.

    Used by auto-resume selection: a file that cannot be opened, lacks model
    weights, or fails the architecture header is never a resume candidate.
    The heavy state is memory-mapped so only metadata tensors are materialized.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # older torch without mmap
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, KeyError, ValueError, EOFError, pickle.UnpicklingError) as e:
        logger.warning("Skipping unreadable checkpoint %s: %s", p, e)
        return None
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        logger.warning("Skipping checkpoint %s: no 'model' weights", p)
        return None
    blocks = ckpt.get("num_blocks")
    filters = ckpt.get("num_filters")
    if blocks is not None and filters is not None and (int(blocks), int(filters)) != (12, 128):
        logger.warning(
            "Skipping checkpoint %s: architecture %sx%s does not match the project 12x128",
            p,
            blocks,
            filters,
        )
        return None
    return {
        "path": str(p),
        "step": int(ckpt.get("step", 0) or 0),
        "optimizer_class": classify_optimizer_state(ckpt),
        "model_version": ckpt.get("model_version"),
        "buffer_write_head": ckpt.get("buffer_write_head"),
        "last_push_step": ckpt.get("last_push_step"),
        "mtime": p.stat().st_mtime,
    }
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
        # Weight-push telemetry: every push attempt is counted so a failing
        # learner-to-actor loop is visible in metrics, not just in logs.
        self.push_success_count = 0
        self.push_failure_count = 0
        # Provenance for warm starts (set by load_checkpoint): the supervised
        # checkpoint this run started from is metadata, NOT RL progress —
        # step_num must only ever count this run's own SGD updates.
        self.inherited_step_num = 0
        self.warm_start_path: Optional[str] = None
        self.warm_start_step: Optional[int] = None

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

    # Keys torch re-derives on load or shares across optimizer families.
    _COMMON_OPTIMIZER_KEYS = frozenset(
        {
            "lr",
            "weight_decay",
            "maximize",
            "foreach",
            "fused",
            "differentiable",
            "capturable",
            "nesterov",
            "initial_lr",
        }
    )

    def _optimizer_state_compatible(
        self,
        opt_state: Dict[str, Any],
        saved_type: Optional[str],
        path: Union[str, Path],
    ) -> bool:
        """Return True only when optimizer state may be restored without poisoning this trainer.

        Restoring a foreign optimizer's state_dict (e.g. supervised AdamW state into the
        RL SGD optimizer) silently replaces SGD's hyperparameters and crashes on the
        first step with ``KeyError: 'momentum'``. Per the optimizer-isolation decision,
        model weights are always restored; foreign optimizer state is discarded.
        """
        current_type = type(self.optimizer).__name__
        if saved_type is not None:
            if saved_type == current_type:
                return True
            logger.warning(
                "Checkpoint optimizer (%s) differs from configured optimizer (%s); "
                "restoring model weights only (optimizer state discarded): %s",
                saved_type,
                current_type,
                path,
            )
            return False
        current_keys: set = set()
        for group in self.optimizer.param_groups:
            current_keys.update(group.keys())
        for group in opt_state.get("param_groups") or []:
            extra = (
                {k for k in group if k not in current_keys}
                - self._COMMON_OPTIMIZER_KEYS
                - {"params"}
            )
            if extra:
                logger.warning(
                    "Checkpoint optimizer state (unexpected keys %s) does not match "
                    "configured optimizer (%s); restoring model weights only "
                    "(optimizer state discarded): %s",
                    sorted(extra),
                    current_type,
                    path,
                )
                return False
        return True

    def load_checkpoint(
        self,
        path: Union[str, Path],
        scheduler_override: bool = True,
        reset_step: bool = False,
        warm_start: bool = False,
    ) -> int:
        """Resume training from a checkpoint.

        ``reset_step=True`` (Stage A alias: ``warm_start=True``) marks a *warm
        start*: model weights are adopted but the checkpoint's step counter,
        push counter, and LR-schedule position belong to a foreign
        (supervised) run, so RL progress restarts at 0 and the inherited
        counter is kept only as provenance metadata. Resuming this run's own
        checkpoint uses the default.
        """
        path_str = str(path)
        reset_step = bool(reset_step or warm_start)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, KeyError) as e:
            raise RuntimeError(f"Failed to load RL checkpoint '{path_str}': {e}") from e
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Checkpoint '{path_str}' is invalid: missing 'model' weights.")

        _validate_arch(ckpt, self.model, getattr(self.model, "num_blocks", 12), getattr(self.model, "num_filters", 128), path_str)

        self.model.load_state_dict(ckpt.get("model"))
        _validate_checkpoint_optimizer_compatibility(ckpt, path_str, warm_start=reset_step)

        if reset_step:
            # Weights-only warm start (audit R-09): never adopt foreign
            # optimizer/scheduler state, and restart the RL schedule from
            # step 0 — the checkpointed step belongs to the source trainer
            # and is kept only as provenance.
            inherited_step = int(ckpt.get("step", 0) or 0)
            self.step_num = 0
            self.last_push_step = 0
            self.inherited_step_num = inherited_step
            self.warm_start_path = path_str
            self.warm_start_step = inherited_step
            self.scheduler.step(0)
            logger.info(
                "Warm start from '%s': weights loaded; optimizer/scheduler state "
                "ignored; RL step counter reset to 0 (inherited source step %d).",
                path_str,
                inherited_step,
            )
            return self.step_num

        opt_state = ckpt.get("optimizer")
        if opt_state is not None:
            if self._optimizer_state_compatible(opt_state, ckpt.get("_optimizer_type"), path_str):
                try:
                    self.optimizer.load_state_dict(opt_state)
                except (RuntimeError, KeyError, ValueError) as e:
                    raise RuntimeError(
                        f"Checkpoint '{path_str}' has an incompatible optimizer state: {e}"
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
        self.last_push_step = int(ckpt.get("last_push_step", self.step_num))

        # Drive the shared scheduler to the resumed step so the optimizer LR and the
        # scheduler counter both match the trainer's checkpointed position.
        if scheduler_override:
            self.scheduler.step(self.step_num)

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

        if ckpt.get("buffer_write_head") is not None and hasattr(self.replay_buffer, "write_head"):
            try:
                head = int(ckpt["buffer_write_head"])
                cap = getattr(self.replay_buffer, "capacity", None)
                if cap and cap > 0:
                    head = head % cap
                self.replay_buffer.write_head = head
            except (ValueError, TypeError):
                logger.debug("Ignoring invalid buffer_write_head in '%s'", path_str)

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

        if "model_version" in ckpt and self.direct_server is not None:
            try:
                self.direct_server.model_version = int(ckpt["model_version"])
            except (TypeError, ValueError):
                pass

        logger.info("Resumed training from %s at step %d", path_str, self.step_num)
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
            "_optimizer_type": type(self.optimizer).__name__,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": getattr(self.scheduler, "state_dict", lambda: {})(),
            "step": self.step_num,
            "model_version": int(getattr(self.direct_server, "model_version", getattr(self.model, "model_version", self.model_version))),
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

        # Single write, then byte-copy to latest_checkpoint.pt (candidate-selection
        # and crash-recovery paths resolve this stable name).
        latest_path = ckpt_path.parent / "latest_checkpoint.pt"
        latest_tmp = latest_path.with_name(latest_path.name + ".tmp")
        try:
            with open(ckpt_path, "rb") as fsrc, open(latest_tmp, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            os.replace(latest_tmp, latest_path)
        except OSError as e:
            logger.warning("Failed to update %s: %s", latest_path, e)
            try:
                latest_tmp.unlink()
            except OSError:
                logger.debug("Failed to clean tmp checkpoint file %s", latest_tmp)

        if keep_last_n >= 0:
            try:
                from python.common.cleanup import prune_checkpoints

                prune_checkpoints(ckpt_path.parent, keep_last_n=keep_last_n, extra_keep=(ckpt_path.name,))
            except (OSError, ValueError):
                logger.debug("RL checkpoint prune skipped", exc_info=True)

        return str(ckpt_path)

    def train_step(self, batch_dict: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Run one training step on a sampled batch and return metrics."""
        self.model.train()
        # Prefer an explicit batch dict when provided; otherwise sample from the replay buffer.
        if batch_dict is None:
            # sample_batch_tensors returns (batch_dict, human_mix).
            batch_dict, _ = self.replay_buffer.sample_batch_tensors(
                batch_size=self.batch_size,
                step=self.step_num,
                device=self.device,
            )

        if self.train_stream is not None:
            with torch.cuda.stream(self.train_stream):
                return self._train_step_impl(batch_dict)
        return self._train_step_impl(batch_dict)

    def _train_step_impl(self, batch_dict: dict[str, Any]) -> dict[str, Any]:
        planes = batch_dict["planes"].to(self.device, non_blocking=True)
        visits = batch_dict["visits"].to(self.device, non_blocking=True)
        outcomes = batch_dict["outcomes"].to(self.device, non_blocking=True)
        is_full = batch_dict["is_full_search"].to(self.device, non_blocking=True)

        # LR is a pure function of the checkpointed trainer step (not of the shared
        # scheduler's internal advance counter); the explicit-step call also keeps
        # the shared counter in sync so scheduler state stays consistent across save/resume.
        self.scheduler.step(self.step_num)
        lr = self.scheduler.get_lr(self.step_num)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        # bf16 autocast covers the forward pass only (matching the inference server
        # and the verified pre-regression configuration); losses are computed in fp32.
        with torch.autocast(
            device_type=self.device_type, dtype=torch.bfloat16, enabled=self.use_bfloat16
        ):
            policy_logits, wdl_logits = self.model(planes)

        # Policy cross-entropy over full-search positions only, with visit-count
        # temperature sharpening. A batch without full-search rows must yield a
        # zero (not NaN) policy loss so the optimizer step is not skipped.
        full_mask = is_full
        entropy_reg: torch.Tensor
        if full_mask.any():
            full_p_logits = policy_logits[full_mask].float()
            full_visits = visits[full_mask].float()

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
                sharpened = full_visits.clamp_min(0) ** (1.0 / tau)
                target = sharpened / sharpened.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                # temp=1.0 keeps the historical raw-visit target (no normalization).
                target = full_visits
            policy_loss = -(target * log_probs).sum(dim=-1).mean()

            # Entropy regularizer over LEGAL moves only (visited slots), differentiable
            # so the bonus actually contributes gradients (R-01 + R-02 semantics).
            legal_mask = _legal_move_mask(full_visits)
            entropy_reg = _masked_policy_entropy(full_p_logits, legal_mask)
        else:
            policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            target_entropy_t = torch.tensor(0.0, device=self.device)
            entropy_reg = torch.tensor(0.0, device=self.device)

        # WDL cross-entropy over every sampled position. Handles both class
        # probability targets (N, 3) and class index targets (N,) or (N, 1);
        # ignore_index=255 guards OUTCOME_UNAVAILABLE rows in tactical/human data.
        if (
            outcomes.is_floating_point()
            and outcomes.ndim >= 2
            and outcomes.shape[-1] == 3
        ):
            wdl_loss = F.cross_entropy(wdl_logits.float(), outcomes)
        else:
            outcomes_idx = outcomes.long()
            if outcomes_idx.ndim > 1 and outcomes_idx.shape[-1] == 1:
                outcomes_idx = outcomes_idx.squeeze(-1)
            if (outcomes_idx != 255).any():
                wdl_loss = F.cross_entropy(
                    wdl_logits.float(), outcomes_idx, ignore_index=255
                )
            else:
                wdl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = policy_loss + 0.5 * wdl_loss - self.entropy_bonus * entropy_reg

        # Guard against NaNs/Infs so a bad batch cannot corrupt the live model.
        if not torch.isfinite(total_loss):
            logger.warning("RL trainer produced non-finite loss; skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
            self.step_num += 1
            metrics = {
                "step": self.step_num,
                "policy_loss": float(policy_loss.item()),
                "wdl_loss": float(wdl_loss.item()),
                "total_loss": float("nan"),
                "lr": float(lr),
                "policy_entropy": float("nan"),
                "target_entropy": float("nan"),
                "n_full": int(full_mask.sum().item()),
                "n_total": int(is_full.numel()),
            }
            return metrics

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Reported entropy is a logging metric: the masked legal-move entropy when
        # full-search rows exist, otherwise the unmasked value (no legal-move signal).
        with torch.no_grad():
            if full_mask.any():
                entropy_metric_t = entropy_reg.detach()
            else:
                probs = F.softmax(policy_logits.float(), dim=-1)
                entropy_metric_t = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()

        # Single host-device sync for all scalar metrics.
        metrics_t = torch.stack(
            [
                total_loss.detach(),
                policy_loss.detach(),
                wdl_loss.detach(),
                entropy_metric_t.detach(),
                target_entropy_t.detach(),
                full_mask.sum().detach().float(),
            ]
        ).cpu()
        n_total = int(is_full.numel())

        self.step_num += 1
        self.last_push_step = self.step_num if self.step_num % self.push_freq == 0 else self.last_push_step

        # Weight push to the inference server every N steps.
        if self.step_num % self.push_freq == 0:
            self.push_weights()

        # Periodic checkpoint (default every 30 minutes).
        now = time.time()
        if now - self.last_checkpoint_time >= self.checkpoint_interval_sec:
            self.save_checkpoint()
            if hasattr(self.replay_buffer, "flush_to_disk"):
                self.replay_buffer.flush_to_disk()
            self.last_checkpoint_time = now

        m = metrics_t.tolist()
        return {
            "step": self.step_num,
            "policy_loss": float(m[1]),
            "wdl_loss": float(m[2]),
            "total_loss": float(m[0]),
            "lr": float(lr),
            "policy_entropy": float(m[3]),
            "target_entropy": float(m[4]),
            "n_full": int(m[5]),
            "n_total": n_total,
        }

    def push_weights(self) -> Any:
        """Push current model weights to the inference server and disk sync path.

        Returns the CUDA event recorded on the training stream (so callers and the
        server can order the weight swap behind in-flight training kernels), or None
        on CPU / when no dedicated training stream exists.

        The server's weight-loading contract is to return the new model version on
        a successful swap and the *unchanged* version on failure. A push whose
        version does not advance is therefore a failed push, not a success.

        Raises:
            WeightPushError: when the connected inference server fails to adopt
                the new weights or exposes no weight-loading API. The trainer's
                ``model_version`` mirrors the server and advances only after a
                successful push, so versions and served weights never diverge.
        """
        num_blocks = getattr(self.model, "num_blocks", 12)
        num_filters = getattr(self.model, "num_filters", 128)

        # Record synchronization event on the training stream.
        sync_event = None
        if self.train_stream is not None:
            sync_event = self.train_stream.record_event()

        # Direct in-memory synchronization if an inference server is connected.
        if self.direct_server is not None:
            state = {k: v.detach() for k, v in self.model.state_dict().items()}
            prev_version = int(getattr(self.direct_server, "model_version", 0) or 0)
            try:
                if hasattr(self.direct_server, "load_state_dict_weights"):
                    new_version = self.direct_server.load_state_dict_weights(
                        state, sync_event=sync_event
                    )
                elif hasattr(self.direct_server, "load_checkpoint"):
                    new_version = self.direct_server.load_checkpoint(
                        str(self.sync_weights_path), sync_event=sync_event
                    )
                else:
                    raise WeightPushError(
                        "direct server exposes no weight-loading API "
                        "(expected load_state_dict_weights or load_checkpoint)"
                    )
            except WeightPushError:
                self.push_failure_count += 1
                raise
            except Exception as e:
                self.push_failure_count += 1
                logger.error(
                    "RL weight push failed at step %d (%d failures so far): %s",
                    self.step_num,
                    self.push_failure_count,
                    e,
                    exc_info=True,
                )
                raise WeightPushError(
                    f"inference server rejected weight push at step {self.step_num}: {e}"
                ) from e

            try:
                new_version = int(new_version)
            except (TypeError, ValueError):
                new_version = prev_version
            if new_version <= prev_version:
                self.push_failure_count += 1
                raise WeightPushError(
                    f"inference server did not adopt new weights "
                    f"(model_version {prev_version} -> {new_version}) at step {self.step_num}"
                )
            self.push_success_count += 1
            self.model_version = new_version
            logger.info(
                "Pushed weights to inference server: model_version %d -> %d "
                "(step %d, pushes ok=%d failed=%d)",
                prev_version,
                new_version,
                self.step_num,
                self.push_success_count,
                self.push_failure_count,
            )

        if self.sync_weights_path is not None:
            payload = {
                "model": self.model.state_dict(),
                "model_version": self.model_version,
                "step": self.step_num,
                "num_blocks": num_blocks,
                "num_filters": num_filters,
            }
            try:
                _atomic_torch_save(payload, self.sync_weights_path)
            except Exception as e:
                logger.error(
                    "RL weight sync write to %s failed: %s",
                    self.sync_weights_path,
                    e,
                    exc_info=True,
                )

        return sync_event


def _validate_checkpoint_optimizer_compatibility(
    ckpt: dict[str, Any],
    path_str: str,
    warm_start: bool = False,
) -> None:
    """Validate that a loaded checkpoint's optimizer state is compatible with the RL SGD optimizer.

    This is the fix for the Phase 4 `KeyError: 'momentum'` incident: a fine-tune checkpoint
    whose optimizer state is AdamW (or otherwise incompatible) must be rejected for full RL
    resume, or accepted only as a weights-only warm start. Checkpoints explicitly tagged as
    something other than "rl" (e.g. finetune, rnd) are rejected on every path: their
    provenance is not an RL training checkpoint even when their optimizer shape looks
    SGD-compatible.
    """
    ckpt_type = ckpt.get("checkpoint_type")
    if ckpt_type is not None and ckpt_type != "rl":
        if ckpt_type == "finetune":
            raise RuntimeError(
                f"Checkpoint '{path_str}' is a fine-tune checkpoint (checkpoint_type='finetune') "
                "and is not an RL training checkpoint. Use an RL checkpoint for resume."
            )
        raise RuntimeError(
            f"Checkpoint '{path_str}' has checkpoint_type={ckpt_type!r} and is "
            "not an RL training checkpoint."
        )

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
