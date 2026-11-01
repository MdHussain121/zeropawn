"""RL self-play trainer for ZeroPawn (SGD momentum, policy/WDL CE, weight pushes, resumable checkpoints)."""

from __future__ import annotations

import logging
import os
import pickle
import random
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

logger = logging.getLogger(__name__)
from python.common.artifacts import (
    ProductionWriteTicket,
    artifact_lock,
    assert_safe_artifact_destination,
    resolve_artifact_path,
)
from python.model.checkpoint_schema import (
    CURRENT_RL_FORMAT,
    RUN_STATE_FAILED,
    build_provenance,
    build_run_metadata,
    checkpoint_identity,
    describe_run_status,
    stamp_run_state,
    validate_checkpoint_schema,
    validate_run_metadata,
)
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


class BufferCheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint's buffer snapshot contradicts the live buffer.

    Audit SR-04: a checkpoint from a different buffer directory, capacity,
    schema, generation, or with pointer/count arithmetic that does not
    correspond to appends must never pair silently with whatever buffer the
    launcher happens to point at. The live buffer state (persisted atomically
    after every insert) is authoritative and is NEVER rolled backward from a
    checkpoint; the operator can permit the pairing explicitly with
    ``allow_buffer_mismatch=True`` (loudly logged).
    """


class BufferRestoreError(RuntimeError):
    """Raised when a checkpoint's replay-buffer snapshot cannot be reconciled.

    Fail-closed by design (finding CKP-05): rewinding a ring buffer's head over
    newer metadata, or restoring a head without a matching size, silently mixes
    policy generations. When the checkpoint and the live buffer cannot be
    proven compatible the resume is refused with a diagnostic instead of
    inventing a state.
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
    ledger = ckpt.get("run_metadata")
    return {
        "path": str(p),
        "step": int(ckpt.get("step", 0) or 0),
        "optimizer_class": classify_optimizer_state(ckpt),
        "model_version": ckpt.get("model_version"),
        # Checkpoint type stamp (audit SR-03): "rl" for trainer checkpoints;
        # None for pre-stamp legacy files; other values (e.g. "finetune") are
        # never eligible for RL resume.
        "checkpoint_type": ckpt.get("checkpoint_type"),
        # Buffer association (audit SR-04): the full snapshot when present,
        # the legacy write-head/size pair otherwise.
        "buffer_state": ckpt.get("buffer_state"),
        "buffer_size": ckpt.get("buffer_size"),
        "buffer_write_head": ckpt.get("buffer_write_head"),
        "last_push_step": ckpt.get("last_push_step"),
        "mtime": p.stat().st_mtime,
        # Resume-selection inputs (findings CKP-01/CKP-08): a weights-only
        # container in the RL directory must never be picked as a resume point,
        # and a checkpoint that records a failed run must not silently look
        # like a clean one.
        "checkpoint_type": ckpt.get("checkpoint_type"),
        "optimizer_type_stamp": ckpt.get("_optimizer_type"),
        "run_state": ledger.get("run_state") if isinstance(ledger, dict) else None,
        "run_id": ledger.get("run_id") if isinstance(ledger, dict) else None,
    }


def is_resumable_candidate(info: Dict[str, Any]) -> tuple[bool, str]:
    """Decide whether ``inspect_rl_checkpoint`` output may be an auto-resume point.

    A candidate must either carry SGD optimizer state, or *declare itself* an
    RL training checkpoint (a first-save RL checkpoint legitimately has no
    per-parameter optimizer state yet). A weights-only export/container — no
    ``checkpoint_type``, no ``_optimizer_type`` — is never resumable, no matter
    where it is copied (findings CKP-08/CKP-09).
    """
    opt = info.get("optimizer_class")
    if opt == "SGD":
        return True, "SGD optimizer state"
    if opt == "empty":
        declared = info.get("checkpoint_type")
        stamped = info.get("optimizer_type_stamp")
        if declared == "rl" or stamped == "SGD":
            return True, "declared RL checkpoint without per-parameter optimizer state yet"
        return (
            False,
            "no optimizer state and not declared as an RL checkpoint "
            "(weights-only container; must not become a resume point)",
        )
    return False, f"optimizer state is {opt} (foreign/poisoned — not resumable by an SGD trainer)"
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
        checkpoint_dir: Optional[Union[str, Path]] = None,
        sync_weights_path: Optional[Union[str, Path]] = None,
        reload_queue: Optional[Any] = None,
        direct_server: Optional[Any] = None,
        use_bfloat16: bool = True,
        gpu_cache_capacity: Optional[int] = None,
        policy_target_temp: float = 0.5,
        entropy_bonus: float = 0.001,
        artifact_ticket: Optional[ProductionWriteTicket] = None,
        label: str = "rl-trainer",
    ) -> None:
        """Construct the trainer.

        ``checkpoint_dir``/``sync_weights_path`` are artifacts paths and are
        **never** resolved against the process working directory (finding
        CKP-04: CWD-relative defaults let a 2x32 test model overwrite the
        production ``live_weights.pt`` on 2026-10-30). Relative paths resolve
        against the repository root; when omitted entirely the trainer writes
        into a session temporary directory and says so loudly — production
        callers must name their output tree explicitly, which also mints the
        :class:`ProductionWriteTicket` that protected destinations require.
        """
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
        if checkpoint_dir is None:
            session_dir = Path(tempfile.mkdtemp(prefix=f"zeropawn_{label}_"))
            logger.warning(
                "RLTrainer constructed without an explicit checkpoint_dir; writing to session "
                "directory %s. Production training must pass an explicit checkpoint directory.",
                session_dir,
            )
            checkpoint_dir = session_dir
        self.checkpoint_dir = resolve_artifact_path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if sync_weights_path is None:
            sync_weights_path = self.checkpoint_dir / "live_weights.pt"
        self.sync_weights_path = resolve_artifact_path(sync_weights_path)
        self.sync_weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_ticket = artifact_ticket

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
        # Run provenance written into every checkpoint and by the orchestrator
        # (run_id, warm-start record, run_status, termination time, ...).
        self.run_metadata: Dict[str, Any] = {}
        # Sampling provenance of the most recent batch (set in train_step).
        self._last_sampling_stats: Dict[str, Any] = {
            "sample_source": "none",
            "sampled_selfplay_rows": 0,
            "sampled_rows_with_game_id": 0,
            "sampled_unique_games": 0,
            "staleness_fallbacks": 0,
        }

        # Run-status ledger (finding CKP-01): every checkpoint this trainer
        # writes embeds this dict, so a structurally valid checkpoint from a
        # failed run can never masquerade as a completed one. The orchestrator
        # (or load_checkpoint for a resume) replaces/extends it.
        self.run_metadata: Dict[str, Any] = build_run_metadata(
            run_id=f"rl-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
            run_type="rl",
            seed=None,
            config={
                "base_lr": float(base_lr),
                "min_lr": float(min_lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "warmup_steps": int(warmup_steps),
                "total_steps": int(total_steps),
                "batch_size": int(batch_size),
                "push_freq": int(push_freq),
                "policy_target_temp": float(policy_target_temp),
                "entropy_bonus": float(entropy_bonus),
            },
        )
        self.provenance: Dict[str, Any] = {}

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
        allow_buffer_mismatch: bool = False,
        reset_buffer: bool = False,
        verify_identity: str = "auto",
    ) -> int:
        """Resume training from a checkpoint.

        ``reset_step=True`` (Stage A alias: ``warm_start=True``) marks a *warm
        start*: model weights are adopted but the checkpoint's step counter,
        push counter, and LR-schedule position belong to a foreign
        (supervised) run, so RL progress restarts at 0 and the inherited
        counter is kept only as provenance metadata. Resuming this run's own
        checkpoint uses the default.

        Buffer reconciliation is a two-stage contract (SR-04 identity + CKP-05
        triple). Validation performed before any state is touched (findings
        CKP-01/05/06/10 and roadmap phases I/J/P):

        * the schema contract (``validate_checkpoint_schema``) — current format
          fully validated, legacy artifacts accepted explicitly with a warning,
          newer/unknown formats rejected;
        * the run-status ledger is surfaced (``describe_run_status``) so a
          failed run's checkpoint cannot be read as a completed one;
        * registry identity is verified against the exact file being loaded
          (``verify_identity``: "auto" checks registered artifacts and rejects
          hash drift, quarantine, and disqualification; "required" also refuses
          unregistered files; "off" skips);
        * stage 1 (audit SR-04): the checkpoint's stored buffer snapshot
          (``buffer_state``, or the legacy ``buffer_size``/``buffer_write_head``
          pair) must be consistent with the LIVE buffer — ``match`` or a
          verifiably appended ``ahead`` state. Any other combination raises
          :class:`BufferCheckpointMismatchError` unless the operator
          acknowledges it explicitly (``allow_buffer_mismatch=True``, or the
          stronger ``reset_buffer=True``; both loudly logged). The live buffer
          pointer is NEVER overwritten from the checkpoint;
        * stage 2 (finding CKP-05, full resume only): the snapshot triple and
          generation are reconciled with the live buffer as a unit
          (``reset_buffer`` acknowledges a deliberate reset);
          ``allow_buffer_mismatch=True`` also downgrades stage-2 refusals to
          loud, recorded overrides so one acknowledgement cannot be silently
          undone by the other stage.
        """
        path_str = str(path)
        reset_step = bool(reset_step or warm_start)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, KeyError) as e:
            raise RuntimeError(f"Failed to load RL checkpoint '{path_str}': {e}") from e
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Checkpoint '{path_str}' is invalid: missing 'model' weights.")

        verification = self._verify_checkpoint_identity(path, verify_identity)
        if verification is not None:
            logger.info("[IDENTITY] %s", verification)

        # Domain-level gates first (architecture, then optimizer family): these
        # produce the specific diagnostics users act on. The schema contract is
        # enforced immediately afterwards, before ANY state is adopted.
        _validate_arch(ckpt, self.model, getattr(self.model, "num_blocks", 12), getattr(self.model, "num_filters", 128), path_str)
        _validate_checkpoint_optimizer_compatibility(ckpt, path_str, warm_start=reset_step)

        verdict = validate_checkpoint_schema(
            ckpt,
            path=path_str,
            expected_family=None if reset_step else "rl",
            allow_legacy=True,
            require_full_state=not reset_step,
        )
        for warning in verdict.warnings:
            logger.warning("[SCHEMA] %s: %s", Path(path_str).name, warning)
        ledger = ckpt.get("run_metadata")
        for warning in validate_run_metadata(
            ledger, path=path_str, required=not verdict.is_legacy
        ):
            logger.warning("[RUN-STATUS] %s: %s", Path(path_str).name, warning)
        logger.info(
            "[RUN-STATUS] %s predecessor: %s", Path(path_str).name, describe_run_status(ledger)
        )
        if isinstance(ledger, dict) and ledger.get("run_state") == RUN_STATE_FAILED:
            logger.warning(
                "Checkpoint '%s' records a FAILED run (%s). Resuming is allowed, but the "
                "failure is preserved in the new run's ledger as predecessor status.",
                path_str,
                ledger.get("failure_reason"),
            )

        self.model.load_state_dict(ckpt.get("model"))

        # Continuation lineage: a resumed run is a NEW run whose ledger names the
        # predecessor. Terminal predecessors are never re-opened (CKP-01 §6.4).
        self._resume_parent_identity = checkpoint_identity(path, hash_file=True)
        predecessor_ledger = dict(ledger) if isinstance(ledger, dict) else None
        self.run_metadata = build_run_metadata(
            run_type="rl-resume" if not reset_step else "rl-warm-start",
            seed=self.run_metadata.get("seed"),
            config=self.run_metadata.get("config"),
            parent_checkpoint=self._resume_parent_identity,
            extra=(
                {"predecessor_run": predecessor_ledger}
                if predecessor_ledger
                else {}
            ),
        )
        if reset_step:
            self.run_metadata["warm_start"] = path_str
        self.provenance = {}

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

        # ── Replay-buffer ↔ checkpoint reconciliation (SR-04 + CKP-05) ──────
        # The live buffer's own meta.json (atomic fsync + replace after every
        # insert) is the authoritative state; the pointer is NEVER restored
        # from the checkpoint. Reconciliation is a two-stage contract:
        #
        # full resume — the CKP-05 reconciliation owns the flow: identity-class
        #   contradictions (a different buffer directory) raise
        #   :class:`BufferCheckpointMismatchError` (audit SR-04); state
        #   contradictions (older/emptier live buffer, foreign save generation)
        #   raise :class:`BufferRestoreError`. On adoption, the SR-04
        #   append-arithmetic assertion runs, so a live ring that is "newer" by
        #   count but not explainable as appends is still refused.
        # warm start — counters reset and nothing is adopted, but the SR-04
        #   pairing assertion still applies to snapshots that carry one.
        if hasattr(self.replay_buffer, "write_head"):
            snapshot = ckpt.get("buffer_state")
            if snapshot is None and (
                ckpt.get("buffer_size") is not None or ckpt.get("buffer_write_head") is not None
            ):
                snapshot = {
                    "size": ckpt.get("buffer_size"),
                    "write_head": ckpt.get("buffer_write_head"),
                }
            has_snapshot_apis = callable(
                getattr(self.replay_buffer, "buffer_state", None)
            ) or callable(getattr(self.replay_buffer, "checkpoint_consistency", None))
            if not has_snapshot_apis:
                # Buffer object without the snapshot APIs (minimal test
                # doubles): fall back to the legacy write-head restore for
                # compatibility.
                if ckpt.get("buffer_write_head") is not None:
                    try:
                        head = int(ckpt["buffer_write_head"])
                        cap = getattr(self.replay_buffer, "capacity", None)
                        if cap and cap > 0:
                            head = head % cap
                        self.replay_buffer.write_head = head
                    except (ValueError, TypeError):
                        logger.debug("Ignoring invalid buffer_write_head in '%s'", path_str)
            elif snapshot is None:
                logger.info(
                    "[BUFFER] checkpoint '%s' carries no buffer snapshot (legacy format); "
                    "live buffer state kept as-is",
                    path_str,
                )
            elif reset_step:
                self._assert_buffer_pairing(
                    snapshot,
                    path_str,
                    allow_override=allow_buffer_mismatch or reset_buffer,
                )
            else:
                self._restore_buffer_state(
                    ckpt,
                    path_str,
                    reset_buffer=reset_buffer,
                    allow_mismatch=allow_buffer_mismatch,
                )

        if ckpt.get("rng_state") is not None:
            try:
                rng = ckpt["rng_state"]
                if hasattr(rng, "cpu"):
                    rng = rng.cpu()
                if hasattr(rng, "to") and hasattr(rng, "dtype") and rng.dtype != torch.uint8:
                    rng = rng.to(torch.uint8)
                torch.set_rng_state(rng)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(
                    "Checkpoint '%s' torch RNG state could not be restored (%s); this run "
                    "continues with fresh torch RNG (bitwise RNG continuity is NOT claimed).",
                    path_str,
                    e,
                )
        if ckpt.get("numpy_rng_state") is not None:
            try:
                _restore_numpy_rng_state(ckpt["numpy_rng_state"])
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Checkpoint '%s' numpy RNG state could not be restored (%s); continuing "
                    "with fresh numpy RNG.",
                    path_str,
                    e,
                )
        if ckpt.get("python_rng_state") is not None:
            try:
                random.setstate(ckpt["python_rng_state"])
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Checkpoint '%s' python RNG state could not be restored (%s); continuing "
                    "with fresh python RNG.",
                    path_str,
                    e,
                )
        self._restore_cuda_rng_state(ckpt.get("cuda_rng_state"), path_str)

        if "model_version" in ckpt and self.direct_server is not None:
            try:
                self.direct_server.model_version = int(ckpt["model_version"])
            except (TypeError, ValueError):
                pass

        logger.info("Resumed training from %s at step %d", path_str, self.step_num)
        return self.step_num

    # ── Identity verification (Phase J) ──────────────────────────────────────

    def _verify_checkpoint_identity(self, path: Union[str, Path], mode: str) -> Optional[str]:
        """Verify the exact file being loaded against the checkpoint registry.

        ``mode``: "off" skips; "required" also refuses unregistered files;
        "auto" (default) verifies registered artifacts and logs when a file is
        unregistered. Hash mismatches, quarantined, and disqualified entries are
        always refused — a file that changed after verification, or a
        quarantined snapshot, must never be resumed silently.
        """
        if mode == "off":
            return None
        from python.model.checkpoint_registry import verify_checkpoint_identity

        # The registry is discovered from the artifact's own tree (a worktree may
        # hold checkpoints whose governance record lives in the same tree).
        verdict = verify_checkpoint_identity(Path(path))
        if verdict["status"] == "unregistered":
            if mode == "required":
                raise RuntimeError(
                    f"Checkpoint '{path}' is not registered in docs/checkpoint_registry.json "
                    "(identity verification is required). Register it or pass "
                    "verify_identity='auto'."
                )
            return f"unregistered artifact (sha256 {verdict['sha256'][:16]}...); loading by explicit path"
        if verdict["status"] in ("quarantined", "disqualified"):
            raise RuntimeError(
                f"Checkpoint '{path}' is {verdict['status']} in the checkpoint registry "
                f"({verdict.get('detail', 'no detail')}). Refusing to load it."
            )
        if verdict["status"] == "hash_mismatch":
            raise RuntimeError(
                f"Checkpoint '{path}' hash {verdict['sha256'][:16]}... does not match its "
                f"registry record ({str(verdict.get('expected'))[:16]}...). The file changed "
                "after verification; provenance is broken. Refusing to load."
            )
        return (
            f"registry {verdict['status']} (entry {verdict.get('entry_id')}, "
            f"sha256 {verdict['sha256'][:16]}...)"
        )

    # ── Replay-buffer restoration contract (finding CKP-05) ──────────────────

    def _restore_buffer_state(
        self,
        ckpt: Dict[str, Any],
        path_str: str,
        *,
        reset_buffer: bool = False,
        allow_mismatch: bool = False,
    ) -> None:
        """Reconcile the checkpoint's buffer snapshot with the live buffer.

        Contract (explicit, fail-closed, recorded in run metadata):

        * ``reset_buffer`` — the operator already acknowledged losing buffer
          contents (``--reset-buffer``): the live counters are left untouched
          (an empty buffer stays empty; no records are invented), and the
          checkpoint's snapshot is kept as provenance only.
        * ``allow_mismatch`` — the operator already overrode the SR-04
          identity assertion (``--allow-buffer-mismatch``): the CKP-05
          refusals are downgraded to loud, recorded overrides with the same
          live-state-kept semantics, so one acknowledgement cannot be silently
          undone by the other stage.
        * incompatible generations — checkpoint and live buffer both carry a
          generation id and they differ: refused.
        * live buffer older or emptier than the checkpoint's snapshot: refused
          (restoring would silently discard records the checkpoint expects).
        * live buffer same generation (or unknown legacy markers) and equal-or-
          newer: the **live** triple (write_head, size, total_added) is adopted
          as a unit, so the ring stays internally consistent and no consumer can
          observe a rewound head over a newer size.
        """
        live = None
        if hasattr(self.replay_buffer, "buffer_state"):
            try:
                live = dict(self.replay_buffer.buffer_state())
            except Exception:  # pragma: no cover - defensive
                live = None
        if live is None:
            live = {
                "write_head": int(getattr(self.replay_buffer, "write_head", 0) or 0),
                "size": int(len(self.replay_buffer)) if hasattr(self.replay_buffer, "__len__") else 0,
                "total_added": int(getattr(self.replay_buffer, "total_added", 0) or 0),
                "capacity": int(getattr(self.replay_buffer, "capacity", 0) or 0),
                "generation": getattr(self.replay_buffer, "buffer_generation", None),
            }

        saved_state = ckpt.get("buffer_state")
        if not isinstance(saved_state, dict):
            # Legacy checkpoint: only buffer_write_head/buffer_size exist.
            saved_head = ckpt.get("buffer_write_head")
            saved_size = ckpt.get("buffer_size")
            if saved_head is None and saved_size is None:
                logger.info(
                    "[BUFFER] Checkpoint '%s' carries no buffer snapshot; live buffer state is "
                    "unchanged (%s).",
                    path_str,
                    live,
                )
                self.run_metadata["buffer_resume_decision"] = {
                    "decision": "checkpoint_has_no_buffer_state",
                    "live": live,
                }
                return
            saved_state = {
                "write_head": int(saved_head or 0),
                "size": int(saved_size or 0),
                "total_added": int(saved_size or 0),  # legacy: no total_added was recorded
                "capacity": None,
                "generation": None,
                "legacy": True,
            }
        saved = {
            "write_head": int(saved_state.get("write_head") or 0),
            "size": int(saved_state.get("size") or 0),
            "total_added": int(saved_state.get("total_added") or saved_state.get("size") or 0),
            "capacity": saved_state.get("capacity"),
            "generation": saved_state.get("generation"),
            "legacy": bool(saved_state.get("legacy", False)),
        }

        decision: Dict[str, Any] = {
            "checkpoint": saved,
            "live": live,
            "policy": "live-if-newer (same generation)",
        }

        if reset_buffer:
            decision["decision"] = "reset_buffer_explicit"
            logger.info(
                "[BUFFER] --reset-buffer given: live buffer starts fresh (size=%s) and the "
                "checkpoint's snapshot %s is retained as provenance only. No records are "
                "invented from the checkpoint.",
                live["size"],
                {k: saved[k] for k in ("write_head", "size", "total_added")},
            )
            self.run_metadata["buffer_resume_decision"] = decision
            return

        if allow_mismatch:
            decision["decision"] = "allow_buffer_mismatch_explicit"
            logger.warning(
                "[BUFFER OVERRIDE] --allow-buffer-mismatch given: the checkpoint's snapshot %s "
                "is NOT reconciled against the live buffer (size=%s, head=%s, total_added=%s); "
                "the live state is kept as-is and its save generation is not asserted. This "
                "pairing was explicitly acknowledged.",
                {k: saved[k] for k in ("write_head", "size", "total_added")},
                live["size"],
                live["write_head"],
                live["total_added"],
            )
            self.run_metadata["buffer_resume_decision"] = decision
            return

        # SR-04 identity class: a snapshot from a DIFFERENT buffer directory is
        # a pairing error, not a state contradiction — it refuses with
        # :class:`BufferCheckpointMismatchError` before any triple logic runs.
        saved_buffer_id = saved_state.get("buffer_id") if isinstance(saved_state, dict) else None
        live_buffer_id = getattr(self.replay_buffer, "buffer_id", None)
        if (
            saved_buffer_id is not None
            and live_buffer_id is not None
            and str(saved_buffer_id) != str(live_buffer_id)
        ):
            decision["decision"] = "refused_different_buffer_directory"
            self.run_metadata["buffer_resume_decision"] = decision
            raise BufferCheckpointMismatchError(
                f"Checkpoint '{path_str}' was saved against replay buffer {saved_buffer_id} "
                f"but the live buffer is {live_buffer_id} (a different directory). Pairing "
                "them would mix unrelated self-play data. The live buffer state (size="
                f"{live['size']}, write_head={live['write_head']}) is authoritative; pass "
                "--allow-buffer-mismatch or --reset-buffer to acknowledge explicitly."
            )

        saved_gen = saved.get("generation")
        live_gen = live.get("generation")
        if saved_gen and live_gen and saved_gen != live_gen:
            decision["decision"] = "refused_generation_mismatch"
            self.run_metadata["buffer_resume_decision"] = decision
            raise BufferRestoreError(
                f"Refusing to resume '{path_str}': the checkpoint was saved against replay-buffer "
                f"generation {saved_gen} but the live buffer is generation {live_gen}. Splicing "
                "two generations would mix unrelated self-play data. Pass --reset-buffer to "
                "acknowledge starting from the live buffer as-is, or restore the matching buffer."
            )

        if live["size"] == 0 and live["total_added"] == 0 and (
            saved["size"] > 0 or saved["total_added"] > 0
        ):
            decision["decision"] = "refused_live_buffer_empty"
            self.run_metadata["buffer_resume_decision"] = decision
            raise BufferRestoreError(
                f"Refusing to resume '{path_str}': the checkpoint expects "
                f"{saved['size']:,} buffered records but the live replay buffer is empty. "
                "Restoring the checkpoint pointer into an empty buffer would fabricate a "
                "state that does not exist. Pass --reset-buffer to continue from the empty "
                "buffer knowingly (the checkpoint's records are then unrecoverable)."
            )

        if live["total_added"] < saved["total_added"]:
            decision["decision"] = "refused_live_buffer_older"
            self.run_metadata["buffer_resume_decision"] = decision
            raise BufferRestoreError(
                f"Refusing to resume '{path_str}': the checkpoint snapshot holds "
                f"{saved['total_added']:,} records but the live buffer only has "
                f"{live['total_added']:,}. The live buffer is older than the checkpoint "
                "snapshot (records were lost or replaced). Pass --reset-buffer to acknowledge."
            )

        # Equal-or-newer live state, compatible generation: adopt live as a unit.
        adopted = {
            "write_head": live["write_head"],
            "size": live["size"],
            "total_added": live["total_added"],
        }
        cap = live.get("capacity") or 0
        if cap > 0:
            if not (0 <= adopted["write_head"] < cap):
                raise BufferRestoreError(
                    f"Live replay buffer head {adopted['write_head']} is outside [0, {cap}); "
                    "refusing to resume on an inconsistent ring."
                )
            if not (0 <= adopted["size"] <= cap):
                raise BufferRestoreError(
                    f"Live replay buffer size {adopted['size']} is outside [0, {cap}]; "
                    "refusing to resume on an inconsistent ring."
                )
            if adopted["size"] < cap and adopted["write_head"] != adopted["size"]:
                decision["warning"] = (
                    f"non-wrapped ring with head {adopted['write_head']} != size "
                    f"{adopted['size']} (legacy metadata); adopting the live triple as-is"
                )
                logger.warning("[BUFFER] %s", decision["warning"])

        same = (
            saved["write_head"] == live["write_head"]
            and saved["size"] == live["size"]
            and saved["total_added"] == live["total_added"]
        )
        decision["decision"] = "identical_snapshot" if same else "adopted_live_newer"
        self.run_metadata["buffer_resume_decision"] = decision
        if same:
            logger.info(
                "[BUFFER] Checkpoint and live buffer agree exactly (head=%d, size=%d, "
                "total_added=%d, generation=%s).",
                live["write_head"],
                live["size"],
                live["total_added"],
                live_gen or "unknown",
            )
        else:
            logger.info(
                "[BUFFER] Live buffer is newer than the checkpoint snapshot; continuing from the "
                "LIVE state (head=%d, size=%d, total_added=%d) instead of rewinding to the "
                "checkpoint's (head=%d, size=%d, total_added=%d). The checkpoint snapshot is "
                "recorded in run metadata as provenance.",
                live["write_head"],
                live["size"],
                live["total_added"],
                saved["write_head"],
                saved["size"],
                saved["total_added"],
            )

        # SR-04 append-arithmetic assertion (audit SR-04, integrated): a live
        # ring that is equal-or-newer by count but NOT explainable as appends
        # from the checkpoint's snapshot is inconsistent — the records were
        # replaced rather than appended. Legacy snapshots (no total_added)
        # cannot be verified this way and are handled conservatively above.
        classify = getattr(self.replay_buffer, "checkpoint_consistency", None)
        if (
            not saved.get("legacy")
            and isinstance(saved_state, dict)
            and saved_state.get("total_added") is not None
            and callable(classify)
        ):
            verdict, detail = classify(saved_state)
            if verdict == "mismatch":
                decision["decision"] = "refused_ring_not_explainable_as_appends"
                self.run_metadata["buffer_resume_decision"] = decision
                raise BufferCheckpointMismatchError(
                    f"Checkpoint '{path_str}' buffer snapshot is inconsistent with the live "
                    f"replay buffer: {detail}. The live ring is not explainable as appends "
                    "from the checkpoint's state; pass --allow-buffer-mismatch or "
                    "--reset-buffer to acknowledge explicitly."
                )
            logger.info(
                "[BUFFER] checkpoint/buffer snapshot %s the live state (%s)",
                "matches" if verdict == "match" else "is behind",
                detail,
            )

    def _assert_buffer_pairing(
        self,
        snapshot: Dict[str, Any],
        path_str: str,
        *,
        allow_override: bool = False,
    ) -> None:
        """SR-04 pairing assertion without adopting any buffer state (warm start)."""
        classify = getattr(self.replay_buffer, "checkpoint_consistency", None)
        if not callable(classify):
            return
        verdict, detail = classify(snapshot)
        if verdict in ("match", "ahead"):
            logger.info(
                "[BUFFER] checkpoint/buffer snapshot %s the live state (%s)",
                "matches" if verdict == "match" else "is behind",
                detail,
            )
            return
        message = (
            f"Checkpoint '{path_str}' buffer snapshot is inconsistent with the live replay "
            f"buffer: {detail}. The live buffer state is authoritative; refusing to pair this "
            "checkpoint with it unless --allow-buffer-mismatch or --reset-buffer is passed "
            "explicitly."
        )
        if not allow_override:
            raise BufferCheckpointMismatchError(message)
        logger.warning(
            "[BUFFER OVERRIDE] Proceeding despite buffer/checkpoint mismatch (explicitly "
            "acknowledged): %s",
            detail,
        )

    def _cuda_rng_state(self) -> Optional[List[torch.Tensor]]:
        """Per-device CUDA RNG states when training on CUDA (audit CKP-12).

        ``None`` on CPU-only runs: the project does not fabricate RNG state it
        does not have, and bitwise-exactness claims are scoped to CPU RNG plus
        whatever CUDA state was actually captured.
        """
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        try:
            return list(torch.cuda.get_rng_state_all())
        except Exception:  # pragma: no cover - defensive
            logger.debug("CUDA RNG capture failed", exc_info=True)
            return None

    def _restore_cuda_rng_state(self, saved: Any, path_str: str) -> None:
        """Restore captured CUDA RNG state when the device topology matches.

        Device-count differences are reported explicitly instead of being
        silently ignored or silently claimed as restored.
        """
        if saved is None:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            logger.info(
                "Checkpoint '%s' carries CUDA RNG state but this run is CPU-only; "
                "not restored (CUDA RNG exactness is not claimed for this run).",
                path_str,
            )
            return
        try:
            # ``set_rng_state_all`` requires CPU uint8 tensors, but the checkpoint
            # is loaded with ``map_location=self.device`` — so move them back.
            states = [torch.as_tensor(s, dtype=torch.uint8).cpu() for s in saved]
        except (TypeError, ValueError):
            logger.warning("Checkpoint '%s' CUDA RNG state is malformed; not restored", path_str)
            return
        present = torch.cuda.device_count()
        if len(states) != present:
            logger.warning(
                "Checkpoint '%s' captured CUDA RNG state for %d device(s) but %d are present; "
                "CUDA RNG state NOT restored (bitwise-exact CUDA continuation is not claimed "
                "across a device-count change).",
                path_str,
                len(states),
                present,
            )
            return
        try:
            torch.cuda.set_rng_state_all(states)
            logger.info("Restored CUDA RNG state for %d device(s) from '%s'", len(states), path_str)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to restore CUDA RNG state from '%s': %s", path_str, e)

    def _buffer_snapshot(self) -> Optional[Dict[str, Any]]:
        """Unified checkpoint buffer snapshot (CKP-05 triple + SR-04 identity).

        ``generation`` is the CKP-05 save generation (string id consumed by
        ``_restore_buffer_state``); ``dir_generation`` is the SR-04 directory
        re-creation counter consumed by ``checkpoint_consistency``. Carrying
        both under distinct keys keeps the two contracts independent — neither
        stage can misread the other's identity field.
        """
        buf = self.replay_buffer
        snap: Optional[Dict[str, Any]] = None
        if hasattr(buf, "buffer_state"):
            try:
                snap = dict(buf.buffer_state())
            except Exception:  # pragma: no cover - defensive
                logger.debug("buffer_state() failed", exc_info=True)
        if snap is None and hasattr(buf, "write_head"):
            snap = {
                "write_head": int(getattr(buf, "write_head", 0) or 0),
                "size": int(len(buf)) if hasattr(buf, "__len__") else None,
                "total_added": int(getattr(buf, "total_added", 0) or 0),
                "capacity": int(getattr(buf, "capacity", 0) or 0),
                "generation": getattr(buf, "buffer_generation", None),
            }
        if snap is not None and hasattr(buf, "snapshot"):
            try:
                identity = dict(buf.snapshot())
            except Exception:  # pragma: no cover - defensive
                identity = {}
            for key in ("buffer_id", "schema_version"):
                if identity.get(key) is not None:
                    snap[key] = identity[key]
            if identity.get("generation") is not None:
                snap["dir_generation"] = int(identity["generation"])
        return snap

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        keep_last_n: int = 3,
        *,
        checkpoint_type_override: Optional[str] = None,
    ) -> str:
        """Save RL training state atomically and optionally prune older checkpoints.

        The artifact carries the run-status ledger (CKP-01), the consistent
        replay-buffer snapshot triple plus generation (CKP-05), structured
        provenance (CKP-12), and the current schema version (CKP-10). The
        destination is validated against the production-write policy (CKP-04)
        and held under a cross-process lock (SUS-02) for the whole update,
        including the ``latest_checkpoint.pt`` mirror.
        """
        ckpt_path = (
            resolve_artifact_path(path)
            if path
            else self.checkpoint_dir / f"rl_ckpt_step_{self.step_num}.pt"
        )
        arch = (
            int(getattr(self.model, "num_blocks", 12)),
            int(getattr(self.model, "num_filters", 128)),
        )
        ckpt_path = assert_safe_artifact_destination(
            ckpt_path, ticket=self.artifact_ticket, arch=arch, what="RL checkpoint"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # The ledger is advanced, not replaced: a save during a running run
        # records "running", and a completed/failed/interrupted transition is
        # stamped by the caller before the terminal save.
        ledger = dict(self.run_metadata) if self.run_metadata else build_run_metadata(run_type="rl")
        if ledger.get("run_state") in ("created", None):
            ledger = stamp_run_state(ledger, "running", step=self.step_num)
        else:
            ledger = dict(ledger)
            ledger["last_update_at"] = time.time()
            ledger["step"] = int(self.step_num)
        self.run_metadata = ledger

        provenance = dict(self.provenance) if self.provenance else {}
        if not provenance:
            provenance = build_provenance(
                run_id=ledger.get("run_id"),
                run_type=ledger.get("run_type", "rl"),
                config=ledger.get("config"),
                model_arch=arch,
                param_count=int(sum(p.numel() for p in self.model.parameters())),
                dataset=ledger.get("dataset"),
                parent_checkpoint=getattr(self, "_resume_parent_identity", None),
                optimizer_config={
                    "type": type(self.optimizer).__name__,
                    "momentum": self.momentum,
                    "weight_decay": self.weight_decay,
                },
                scheduler_config={
                    "warmup_steps": self.warmup_steps,
                    "total_steps": self.total_steps,
                    "base_lr": self.base_lr,
                    "min_lr": self.min_lr,
                },
            )
        else:
            provenance = dict(provenance)
            provenance["model"] = {
                "num_blocks": arch[0],
                "num_filters": arch[1],
                "param_count": int(sum(p.numel() for p in self.model.parameters())),
            }
        provenance["parent_checkpoint"] = provenance.get("parent_checkpoint") or getattr(
            self, "_resume_parent_identity", None
        )
        self.provenance = provenance

        state = {
            "checkpoint_type": checkpoint_type_override or "rl",
            "checkpoint_format": CURRENT_RL_FORMAT,
            "_optimizer_type": type(self.optimizer).__name__,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": getattr(self.scheduler, "state_dict", lambda: {})(),
            "step": self.step_num,
            "model_version": int(getattr(self.direct_server, "model_version", getattr(self.model, "model_version", self.model_version))),
            "num_blocks": arch[0],
            "num_filters": arch[1],
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
            # Unified buffer association (SR-04 identity + CKP-05 triple):
            # identity (buffer_id, schema_version, dir_generation), the
            # (write_head, size, total_added, capacity) triple and the CKP-05
            # save generation. Consumed by both resume stages; the live
            # buffer's own meta stays authoritative.
            "buffer_state": self._buffer_snapshot(),
            "buffer_size": int(len(self.replay_buffer)) if hasattr(self.replay_buffer, "__len__") else None,
            "buffer_write_head": getattr(self.replay_buffer, "write_head", None),
            "last_push_step": self.last_push_step,
            "tactical_pool_size": getattr(self.replay_buffer, "tactical_pool_size", None),
            "run_metadata": dict(ledger),
            "provenance": dict(provenance),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": _safe_numpy_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": self._cuda_rng_state(),
        }

        # Whole update (step checkpoint + latest mirror) under one lock so two
        # trainers sharing a directory cannot interleave their writes.
        with artifact_lock(ckpt_path, timeout=0.0, owner=f"rl-trainer step {self.step_num}"):
            _atomic_torch_save(state, ckpt_path)

            # Single write, then byte-copy to latest_checkpoint.pt (candidate-selection
            # and crash-recovery paths resolve this stable name).
            latest_path = assert_safe_artifact_destination(
                ckpt_path.parent / "latest_checkpoint.pt",
                ticket=self.artifact_ticket,
                arch=arch,
                what="RL latest-checkpoint mirror",
            )
            latest_tmp = latest_path.with_name(latest_path.name + ".tmp")
            try:
                with open(ckpt_path, "rb") as fsrc, open(latest_tmp, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                    fdst.flush()
                    try:
                        os.fsync(fdst.fileno())
                    except OSError:
                        logger.debug("fsync of latest tmp failed", exc_info=True)
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

        # Record the published artifact's identity for lineage/hash reporting.
        identity = checkpoint_identity(ckpt_path, hash_file=True)
        self._last_checkpoint_identity = identity
        self.run_metadata["last_checkpoint"] = dict(identity)
        logger.info(
            "Saved RL checkpoint %s (%s, sha256=%s)",
            ckpt_path,
            describe_run_status(self.run_metadata),
            (identity.get("sha256") or "?")[:16],
        )
        return str(ckpt_path)

    def _sampling_stats_from_batch(self, batch_dict: dict[str, Any]) -> Dict[str, Any]:
        """Per-step sampling provenance (audit SR-13/SR-08).

        Counts how many sampled rows carried a producing game id (self-play
        rows) and how many distinct games they came from; human/tactical rows
        carry game id 0 (``NO_GAME_ID``) and are excluded from both the count
        and the distinct-game tally.
        """
        source = getattr(self.replay_buffer, "last_selfplay_source", None)
        stats: Dict[str, Any] = {
            "sample_source": source or "external",
            "sampled_selfplay_rows": int(getattr(self.replay_buffer, "last_selfplay_rows", 0) or 0),
            "sampled_rows_with_game_id": 0,
            "sampled_unique_games": 0,
            "staleness_fallbacks": int(
                getattr(self.replay_buffer, "staleness_fallback_count", 0) or 0
            ),
        }
        gids = batch_dict.get("game_ids")
        if gids is None:
            return stats
        try:
            g = gids.detach().to("cpu").to(torch.long)
            positive = g[g > 0]
            stats["sampled_rows_with_game_id"] = int(positive.numel())
            stats["sampled_unique_games"] = int(torch.unique(positive).numel())
        except (AttributeError, RuntimeError, TypeError):
            logger.debug("sampling stats unavailable for this batch", exc_info=True)
        return stats

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
        self._last_sampling_stats = self._sampling_stats_from_batch(batch_dict)

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
                **self._last_sampling_stats,
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
        # Sampling provenance for this step (audit SR-13/SR-08): how many
        # self-play rows carried a producing game id, and how many distinct
        # games they came from. Human/tactical rows carry game id 0 and are
        # excluded from the distinct-game count. Cheap (one small unique()).
        sampled = self._last_sampling_stats
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
            **sampled,
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
                from python.common.artifacts import (
                    ArtifactPathError,
                    assert_safe_artifact_destination,
                )

                # The sync artifact is a protected production name: validate the
                # destination and the architecture BEFORE writing, so a
                # test-scale model can never replace the production
                # live_weights.pt (finding CKP-04). Guard refusals fail closed.
                assert_safe_artifact_destination(
                    self.sync_weights_path,
                    ticket=self.artifact_ticket,
                    arch=(int(num_blocks), int(num_filters)),
                    what="live weight-sync artifact",
                )
                _atomic_torch_save(payload, self.sync_weights_path)
            except ArtifactPathError as e:
                # A refused destination is a configuration/authorization error,
                # not a retryable push failure: it propagates with its own type
                # (and is *never* swallowed like a transient disk error would be).
                self.push_failure_count += 1
                logger.error("RL weight sync refused for %s: %s", self.sync_weights_path, e)
                raise
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
        if ckpt_type == "supervised":
            raise RuntimeError(
                f"Checkpoint '{path_str}' is a supervised training checkpoint "
                "(checkpoint_type='supervised'). Per project policy, explicitly typed foreign "
                "artifacts are refused on every path here: RL warm starts run through the "
                "registry designation gate (validate_warm_start) and the inference server, and "
                "the weight payload is transferred as a weights-only container. Export the "
                "weights (or use the designated baseline path) instead of loading this file "
                "directly."
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
