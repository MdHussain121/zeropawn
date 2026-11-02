"""Formal checkpoint schema, run-status ledger, and provenance contract.

Established by the 2026-10-31 checkpoint & artifact lifecycle remediation
(findings CKP-01, CKP-10, CKP-12; roadmap P0-1/phase I/J/K). Before this module
``checkpoint_format`` was written by the RL trainer and read by nothing, so
backward compatibility was an accident of tolerant ``.get()`` access, and the
run-status ledger the orchestrator stamped never reached any artifact.

The contract has three parts:

* **Schema** — every family has a version, a required-field set, and a
  compatibility rule. Current writers emit :data:`CURRENT_FORMATS`; readers
  accept the current version and, explicitly and with a warning, *legacy*
  checkpoints (no version marker, or an older one) so the verified historical
  artifacts keep working. Newer/unknown versions are rejected loudly instead of
  being reinterpreted field-by-field.

* **Run status** — :data:`RUN_STATES` is the authoritative lifecycle
  (``created → running → completed | failed | interrupted | cancelled``).
  Every checkpoint family that records training state carries a
  ``run_metadata`` dict built by :func:`build_run_metadata` and advanced by
  :func:`stamp_run_state`. A checkpoint from a failed or interrupted run can no
  longer masquerade as a completed run: the state is *inside* the file, and
  :func:`describe_run_status` surfaces it on load.

* **Provenance** — :func:`build_provenance` records the immutable identity of
  the producing run (git commit, seed, timestamps, config, architecture,
  parameter count, dataset identity, parent checkpoint) so a file can be traced
  to a run without relying solely on external governance records.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

logger = logging.getLogger(__name__)

# ── Schema versions ──────────────────────────────────────────────────────────

#: Format written by current RL trainers. Bumped from 1 because format-2
#: checkpoints carry ``run_metadata``, ``buffer_state`` and ``provenance``.
CURRENT_RL_FORMAT = 2

#: Format written by current supervised trainers.
CURRENT_SUPERVISED_FORMAT = 1

#: Family → current supported format.
CURRENT_FORMATS: Dict[str, int] = {
    "rl": CURRENT_RL_FORMAT,
    "supervised": CURRENT_SUPERVISED_FORMAT,
}

#: Fields every current-format checkpoint of a family must carry. Weights-only
#: containers (exports, opening policy, curriculum) are deliberately *not* in
#: ``CURRENT_FORMATS``: they are inference/warm-start payloads, not training
#: state, and their readers must not treat them as resumable.
REQUIRED_FIELDS: Dict[str, tuple[str, ...]] = {
    "rl": (
        "checkpoint_type",
        "checkpoint_format",
        "model",
        "optimizer",
        "scheduler",
        "step",
        "num_blocks",
        "num_filters",
        "buffer_state",
        "run_metadata",
        "provenance",
    ),
    "supervised": (
        "checkpoint_type",
        "checkpoint_format",
        "model",
        "optimizer",
        "scheduler",
        "step",
        "epoch",
        "num_blocks",
        "num_filters",
        "run_metadata",
    ),
}

#: Minimal fields required even for legacy artifacts (weights must exist).
LEGACY_REQUIRED_FIELDS: tuple[str, ...] = ("model",)

#: Fields that make a training checkpoint *resumable* (full training state), as
#: opposed to a weights-only warm-start payload.
FULL_STATE_FIELDS: tuple[str, ...] = ("optimizer", "scheduler")


class CheckpointSchemaError(RuntimeError):
    """Raised when a checkpoint cannot be accepted under the schema contract."""


@dataclass
class SchemaVerdict:
    """Outcome of validating one checkpoint against the schema contract."""

    path: str
    family: str
    version: Optional[int]
    is_legacy: bool = False
    warnings: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)

    def describe(self) -> str:
        v = "legacy" if self.is_legacy else f"format {self.version}"
        base = f"{self.family} checkpoint ({v})"
        if self.missing_required:
            base += f"; missing {sorted(self.missing_required)}"
        return base


def detect_family(ckpt: Mapping[str, Any]) -> str:
    """Classify a checkpoint payload by its declared stamps and field shape."""
    declared = ckpt.get("checkpoint_type")
    if isinstance(declared, str) and declared:
        normalized = declared.strip().lower()
        if normalized in ("rl", "supervised"):
            return normalized
        return normalized
    # Legacy artifacts predate the stamps; classify structurally, conservatively.
    if "optimizer" in ckpt and "scheduler" in ckpt and "epoch" in ckpt:
        return "supervised"
    if "optimizer" in ckpt and "scheduler" in ckpt and "buffer_size" in ckpt:
        return "rl"
    if "model" in ckpt:
        return "weights_only"
    return "unknown"


def validate_checkpoint_schema(
    ckpt: Mapping[str, Any],
    *,
    path: Union[str, Path],
    expected_family: Optional[str] = None,
    allow_legacy: bool = True,
    require_full_state: bool = False,
) -> SchemaVerdict:
    """Validate ``ckpt`` against the schema contract; raise when unacceptable.

    ``require_full_state`` is the full-resume flag: it additionally demands the
    optimizer/scheduler state that a weights-only warm start legitimately lacks.
    Legacy acceptance (``allow_legacy``) never relaxes this — a legacy file with
    optimizer+scheduler state is resumable, a weights-only export is not.
    """
    if not isinstance(ckpt, Mapping):
        raise CheckpointSchemaError(f"'{path}' is not a checkpoint mapping (got {type(ckpt).__name__}).")

    family = detect_family(ckpt)
    if expected_family is not None and family != expected_family:
        if require_full_state and family not in CURRENT_FORMATS:
            raise CheckpointSchemaError(
                f"'{path}' is a {family!r} payload (no full training state: no optimizer/"
                "scheduler) and cannot be a full resume. Load it as a weights-only warm start "
                f"instead, or pass a {expected_family!r} training checkpoint."
            )
        raise CheckpointSchemaError(
            f"'{path}' is a {family!r} checkpoint but a {expected_family!r} checkpoint is required."
        )

    raw_version = ckpt.get("checkpoint_format")
    version: Optional[int]
    try:
        version = int(raw_version) if raw_version is not None else None
    except (TypeError, ValueError):
        raise CheckpointSchemaError(
            f"'{path}' declares a non-numeric checkpoint_format {raw_version!r}."
        )

    if version is None:
        if not allow_legacy:
            raise CheckpointSchemaError(
                f"'{path}' has no checkpoint_format marker; this consumer requires an "
                f"explicitly versioned checkpoint."
            )
        verdict = SchemaVerdict(path=str(path), family=family, version=None, is_legacy=True)
        missing = [f for f in LEGACY_REQUIRED_FIELDS if f not in ckpt]
        verdict.missing_required = missing
        if missing:
            raise CheckpointSchemaError(
                f"'{path}' is not a usable checkpoint: missing {missing}."
            )
        verdict.warnings.append(
            f"legacy artifact without checkpoint_format (family={family}); accepted through "
            "the explicit legacy path"
        )
        if require_full_state:
            absent = [f for f in FULL_STATE_FIELDS if f not in ckpt]
            if len(absent) == len(FULL_STATE_FIELDS):
                raise CheckpointSchemaError(
                    f"'{path}' carries no optimizer/scheduler state and cannot be a full resume; "
                    "load it as a weights-only warm start instead."
                )
        return verdict

    current = CURRENT_FORMATS.get(family)
    if current is not None and version > current:
        raise CheckpointSchemaError(
            f"'{path}' declares checkpoint_format={version} for family {family!r}, but this "
            f"build supports at most {current}. Refusing to reinterpret a newer schema; "
            "upgrade the code or load the artifact with the build that wrote it."
        )
    if current is None and version >= 1:
        # Weights-only containers may carry format 1 (the historical export).
        verdict = SchemaVerdict(path=str(path), family=family, version=version, is_legacy=True)
        verdict.warnings.append(
            f"{family} payload with checkpoint_format={version} is not a training-state family"
        )
        if require_full_state:
            raise CheckpointSchemaError(
                f"'{path}' is a {family} payload without full training state; it cannot be "
                "resumed — only used as weights."
            )
        return verdict
    if version < 1:
        raise CheckpointSchemaError(f"'{path}' declares an invalid checkpoint_format={version}.")

    verdict = SchemaVerdict(path=str(path), family=family, version=version, is_legacy=version < (current or version))
    if verdict.is_legacy:
        verdict.warnings.append(
            f"checkpoint_format={version} predates current format {current} for family "
            f"{family!r}; accepted through the explicit legacy path"
        )
    required = REQUIRED_FIELDS.get(family, ())
    missing = [f for f in required if f not in ckpt]
    verdict.missing_required = list(missing)
    if missing and not verdict.is_legacy:
        raise CheckpointSchemaError(
            f"'{path}' declares current format {version} for family {family!r} but is missing "
            f"required fields {sorted(missing)}; the artifact is incomplete and will not be "
            "accepted."
        )
    if require_full_state and any(f not in ckpt for f in FULL_STATE_FIELDS):
        missing_state = [f for f in FULL_STATE_FIELDS if f not in ckpt]
        raise CheckpointSchemaError(
            f"'{path}' lacks {missing_state} and cannot be a full resume; load it as a "
            "weights-only warm start instead."
        )
    return verdict


# ── Run-status ledger ────────────────────────────────────────────────────────

RUN_METADATA_VERSION = 1

RUN_STATE_CREATED = "created"
RUN_STATE_RUNNING = "running"
RUN_STATE_COMPLETED = "completed"
RUN_STATE_FAILED = "failed"
RUN_STATE_INTERRUPTED = "interrupted"
RUN_STATE_CANCELLED = "cancelled"

RUN_STATES = frozenset(
    {
        RUN_STATE_CREATED,
        RUN_STATE_RUNNING,
        RUN_STATE_COMPLETED,
        RUN_STATE_FAILED,
        RUN_STATE_INTERRUPTED,
        RUN_STATE_CANCELLED,
    }
)

TERMINAL_RUN_STATES = frozenset(
    {RUN_STATE_COMPLETED, RUN_STATE_FAILED, RUN_STATE_INTERRUPTED, RUN_STATE_CANCELLED}
)

_ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    # A run may terminate before its first observation as RUNNING (zero-step
    # runs, failures during startup): CREATED therefore allows every terminal
    # state. Terminal states themselves are final.
    RUN_STATE_CREATED: frozenset(
        {
            RUN_STATE_RUNNING,
            RUN_STATE_COMPLETED,
            RUN_STATE_FAILED,
            RUN_STATE_INTERRUPTED,
            RUN_STATE_CANCELLED,
        }
    ),
    RUN_STATE_RUNNING: frozenset(
        {RUN_STATE_COMPLETED, RUN_STATE_FAILED, RUN_STATE_INTERRUPTED, RUN_STATE_CANCELLED}
    ),
    RUN_STATE_COMPLETED: frozenset(),
    RUN_STATE_FAILED: frozenset(),
    RUN_STATE_INTERRUPTED: frozenset(),
    RUN_STATE_CANCELLED: frozenset(),
}


class RunStateError(RuntimeError):
    """Raised for an invalid run-state transition or an unreadable ledger."""


def new_run_id(prefix: str = "run", *, at: Optional[float] = None) -> str:
    """Deterministic-shape run id: ``<prefix>-<UTC stamp>-<pid>``."""
    import uuid
    from datetime import datetime, timezone

    stamp = datetime.fromtimestamp(at if at is not None else time.time(), tz=timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def build_run_metadata(
    *,
    run_id: Optional[str] = None,
    run_type: str,
    git_commit: Optional[str] = None,
    seed: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
    dataset: Optional[Mapping[str, Any]] = None,
    parent_checkpoint: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Create the run-status ledger a checkpoint embeds.

    ``run_type`` is a free-form label ("rl", "supervised", "resume", ...) that
    records *what kind of run* wrote the artifact, independent of the checkpoint
    family.
    """
    now = float(created_at if created_at is not None else time.time())
    ledger: Dict[str, Any] = {
        "run_metadata_version": RUN_METADATA_VERSION,
        "run_id": run_id or new_run_id(run_type),
        "run_type": str(run_type),
        "run_state": RUN_STATE_CREATED,
        "created_at": now,
        "started_at": None,
        "last_update_at": now,
        "completed_at": None,
        "failure_reason": None,
        "step": 0,
        "epoch": None,
        "git_commit": git_commit,
        "seed": seed,
        "config": dict(config) if config else {},
        "dataset": dict(dataset) if dataset else {},
        "parent_checkpoint": dict(parent_checkpoint) if parent_checkpoint else None,
    }
    if extra:
        ledger.update(dict(extra))
    return ledger


def stamp_run_state(
    ledger: Optional[Mapping[str, Any]],
    state: str,
    *,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    reason: Optional[str] = None,
    at: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a copy of ``ledger`` advanced to ``state`` with fresh timestamps.

    Raises :class:`RunStateError` for an unknown state or an illegal transition
    (terminal states are final; a resume must start a new ledger whose
    ``parent_checkpoint`` points at the predecessor).
    """
    if state not in RUN_STATES:
        raise RunStateError(f"Unknown run state {state!r}; expected one of {sorted(RUN_STATES)}.")
    now = float(at if at is not None else time.time())
    out: Dict[str, Any] = dict(ledger) if ledger else build_run_metadata(run_type="unknown")
    previous = out.get("run_state", RUN_STATE_CREATED)
    if previous not in RUN_STATES:
        raise RunStateError(f"Ledger carries invalid run_state {previous!r}.")
    if state != previous:
        allowed = _ALLOWED_TRANSITIONS.get(previous, frozenset())
        if state not in allowed:
            raise RunStateError(
                f"Illegal run-state transition {previous!r} → {state!r}. Run states are final; "
                "a continuation is a new run whose parent_checkpoint names the predecessor."
            )
    out["run_state"] = state
    out["last_update_at"] = now
    if state == RUN_STATE_RUNNING and not out.get("started_at"):
        out["started_at"] = now
    if state in TERMINAL_RUN_STATES:
        out["completed_at"] = now
        if reason:
            out["failure_reason"] = str(reason)
    if step is not None:
        out["step"] = int(step)
    if epoch is not None:
        out["epoch"] = int(epoch)
    return out


def advance_run_state_inplace(
    ledger: Dict[str, Any],
    state: str,
    *,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Advance ``ledger`` in place (keeping object identity) and return it.

    Callers that share the ledger object with other writers (e.g. the
    supervised run's telemetry manifest) must not replace the dict, or the two
    views diverge and the artifact records a different status than the run log.
    """
    ledger.update(stamp_run_state(ledger, state, step=step, epoch=epoch, reason=reason))
    return ledger


def validate_run_metadata(ledger: Any, *, path: Union[str, Path], required: bool = False) -> list[str]:
    """Check an embedded ledger; return warnings. Raises when unusable-but-present.

    ``required=True`` demands a present, structurally valid ledger with a state
    from :data:`RUN_STATES` (the current-format contract). When ``required=False``
    an absent ledger yields a warning (legacy artifact) instead of an error.

    Legacy tolerance covers *present* ledgers too (audit S3-02): a mapping that
    carries no ``run_state`` key at all predates the run-status schema (e.g. the
    supervised baseline ckpt-10) and is accepted with an explicit warning when
    ``required=False``, mirroring the absent-ledger branch. A ledger that
    *declares* a ``run_state`` is always validated strictly: an unknown/invalid
    value raises regardless of ``required``, and under ``required=True`` even a
    pre-schema ledger is refused because the current-format contract demands a
    state from :data:`RUN_STATES`.
    """
    if ledger is None:
        if required:
            raise CheckpointSchemaError(f"'{path}' has no run_metadata ledger (required by schema).")
        return [f"no run_metadata ledger inside '{Path(path).name}' (legacy artifact)"]
    if not isinstance(ledger, Mapping):
        raise CheckpointSchemaError(
            f"'{path}' run_metadata is not a mapping (got {type(ledger).__name__})."
        )
    warnings: list[str] = []
    if "run_state" not in ledger:
        # Pre-schema legacy ledger: the ledger predates the run-status contract
        # (no lifecycle state was recorded yet). Accepted exactly like an absent
        # ledger — loudly — never reinterpreted as a current-schema ledger.
        if required:
            raise CheckpointSchemaError(
                f"'{path}' run_metadata is a pre-schema legacy ledger (no 'run_state'); the "
                "current-format contract requires a valid run-status ledger."
            )
        warnings.append(
            f"pre-schema run_metadata ledger without 'run_state' inside '{Path(path).name}' "
            "(legacy ledger, predates the run-status schema); accepted through the explicit "
            "legacy path"
        )
        return warnings
    state = ledger["run_state"]
    if state not in RUN_STATES:
        raise CheckpointSchemaError(
            f"'{path}' run_metadata.run_state={state!r} is not one of {sorted(RUN_STATES)}."
        )
    if not ledger.get("run_id"):
        raise CheckpointSchemaError(f"'{path}' run_metadata carries no run_id.")
    if state in TERMINAL_RUN_STATES and not ledger.get("completed_at"):
        warnings.append(f"run_state={state} without completed_at")
    if state == RUN_STATE_RUNNING and not ledger.get("started_at"):
        warnings.append("run_state=running without started_at")
    return warnings


def describe_run_status(ledger: Optional[Mapping[str, Any]]) -> str:
    """Human-readable one-liner about a checkpoint's lifecycle status."""
    if not isinstance(ledger, Mapping) or not ledger:
        return "run status: unknown (no ledger embedded)"
    return (
        f"run {ledger.get('run_id')} type={ledger.get('run_type')} "
        f"state={ledger.get('run_state')} step={ledger.get('step')}"
        + (f" reason={ledger['failure_reason']}" if ledger.get("failure_reason") else "")
    )


def is_completed_checkpoint(ckpt: Mapping[str, Any]) -> Optional[bool]:
    """True/False for a ledger-bearing checkpoint; None when no ledger exists."""
    ledger = ckpt.get("run_metadata")
    if not isinstance(ledger, Mapping) or not ledger:
        return None
    return ledger.get("run_state") == RUN_STATE_COMPLETED


# ── Provenance ───────────────────────────────────────────────────────────────


def git_commit(short: bool = True) -> Optional[str]:
    """Best-effort current commit of the repository this code lives in."""
    try:
        from python.data.dataset_contract import detect_git_commit

        commit = detect_git_commit()
        if not commit:
            return None
        return commit[:12] if short else commit
    except Exception:  # pragma: no cover - defensive: provenance must never block a save
        logger.debug("git commit detection failed", exc_info=True)
        return None


def build_provenance(
    *,
    run_id: Optional[str],
    run_type: str,
    config: Optional[Mapping[str, Any]] = None,
    model_arch: Optional[tuple[int, int]] = None,
    param_count: Optional[int] = None,
    dataset: Optional[Mapping[str, Any]] = None,
    parent_checkpoint: Optional[Mapping[str, Any]] = None,
    optimizer_config: Optional[Mapping[str, Any]] = None,
    scheduler_config: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    created_at: Optional[float] = None,
    commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Immutable identity of the producing run, embedded in the artifact.

    ``checkpoint_sha256`` is intentionally absent: a file cannot contain its own
    hash. The hash of a saved artifact is recorded where it can be verified —
    the eval history / promotion record (``best_eval.json``) and the registry —
    and by :func:`python.common.artifacts.atomic_write_text` callers that emit
    sidecars.
    """
    provenance: Dict[str, Any] = {
        "provenance_version": 1,
        "run_id": run_id,
        "run_type": str(run_type),
        "created_at": float(created_at if created_at is not None else time.time()),
        "git_commit": commit if commit is not None else git_commit(),
        "model": {
            "num_blocks": int(model_arch[0]) if model_arch else None,
            "num_filters": int(model_arch[1]) if model_arch else None,
            "param_count": int(param_count) if param_count is not None else None,
        },
        "config": dict(config) if config else {},
        "dataset": dict(dataset) if dataset else {},
        "parent_checkpoint": dict(parent_checkpoint) if parent_checkpoint else None,
        "optimizer": dict(optimizer_config) if optimizer_config else {},
        "scheduler": dict(scheduler_config) if scheduler_config else {},
    }
    if extra:
        provenance.update(dict(extra))
    return provenance


def checkpoint_identity(path: Union[str, Path], *, hash_file: bool = True) -> Dict[str, Any]:
    """Identity record (path, size, mtime, optional sha256) for lineage links."""
    p = Path(path)
    record: Dict[str, Any] = {"path": str(p)}
    try:
        stat = p.stat()
        record["size"] = int(stat.st_size)
        record["mtime"] = float(stat.st_mtime)
    except OSError:
        record["size"] = None
        record["mtime"] = None
    if hash_file:
        try:
            record["sha256"] = compute_file_sha256(p)
        except OSError:
            record["sha256"] = None
    else:
        record["sha256"] = None
    return record


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def summarize_schema_support() -> Dict[str, Any]:
    """Machine-readable schema support matrix (used by docs and tests)."""
    return {
        "current_formats": dict(CURRENT_FORMATS),
        "required_fields": {k: list(v) for k, v in REQUIRED_FIELDS.items()},
        "legacy_required_fields": list(LEGACY_REQUIRED_FIELDS),
        "run_states": sorted(RUN_STATES),
        "terminal_run_states": sorted(TERMINAL_RUN_STATES),
        "run_metadata_version": RUN_METADATA_VERSION,
    }
