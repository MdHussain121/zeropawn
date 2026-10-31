"""Checkpoint provenance registry and warm-start validation.

Established 2026-10-26 after the search-quality audit (findings F-11/F-12):
three mutually inconsistent hash/step records existed for the "same"
supervised baselines, every historically certified ``best_model.pt`` artifact
was missing from disk, and an RL campaign had resumed from a checkpoint the
project's own baseline document said not to use.

Rules enforced here:

1. Every checkpoint record carries its actual SHA-256, computed from the file
   (never copied from documentation).
2. A documented hash that matches no existing file is recorded as
   ``status: "missing"`` — it is never presented as recovered.
3. The RL warm start must be a registry entry with
   ``status: "verified"`` and ``designated_warm_start: true``, or the operator
   must explicitly pass ``allow_unverified=True`` (logged as an override).
   Missing, invalid, or undesignated warm starts are rejected — the launcher
   must never silently fall back to whichever checkpoint is easiest to load.

Registry file: ``docs/checkpoint_registry.json`` (tracked in the repository;
``models/`` itself is gitignored, so the registry must not live there).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

REGISTRY_REL_PATH = Path("docs") / "checkpoint_registry.json"

# Architecture the whole project is locked to (vault decision 2026-10-14 and
# every manifest record): 12 ResBlocks x 128 filters, 19 input planes, 4,672 actions.
EXPECTED_ARCHITECTURE = {"num_blocks": 12, "num_filters": 128}


class WarmStartValidationError(RuntimeError):
    """Raised when an RL warm-start checkpoint cannot be trusted."""


@dataclass
class CheckpointInfo:
    path: Path
    sha256: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    model_version: Optional[int] = None
    best_val_loss: Optional[float] = None
    num_params: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream the file's SHA-256 (hex)."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    """Hash a checkpoint and read its metadata (without loading weights to GPU)."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict") or ck.get("model") or {}
    num_params = sum(t.numel() for t in state.values() if hasattr(t, "numel"))
    return CheckpointInfo(
        path=Path(path),
        sha256=compute_sha256(path),
        step=ck.get("step"),
        epoch=ck.get("epoch"),
        model_version=ck.get("model_version"),
        best_val_loss=ck.get("best_val_loss"),
        num_params=num_params,
        metadata={
            k: v
            for k, v in ck.items()
            if k
            not in (
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "model",
                "optimizer",
                "scheduler",
                "scaler",
                "rng_state",
                "numpy_rng_state",
                "python_rng_state",
            )
        },
    )


def load_registry(repo_root: str | Path = ".") -> Dict[str, Any]:
    """Load ``docs/checkpoint_registry.json``; raise if absent."""
    reg_path = Path(repo_root) / REGISTRY_REL_PATH
    if not reg_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint registry not found at {reg_path}. Run "
            "scripts/validate_checkpoints.py --write to create it from the "
            "current artifacts."
        )
    return json.loads(reg_path.read_text(encoding="utf-8"))


def _check_architecture(info: CheckpointInfo) -> None:
    arch = info.metadata.get("arch") or {}
    # Trainer checkpoints store num_blocks/num_filters at the top level.
    num_blocks = info.metadata.get("num_blocks", arch.get("num_blocks"))
    num_filters = info.metadata.get("num_filters", arch.get("num_filters"))
    if num_blocks is None and info.num_params is not None:
        # 8,631,454 params is the 12x128 ChessNet signature; a different count
        # means the architecture drifted.
        if info.num_params != 8_631_454:
            raise WarmStartValidationError(
                f"{info.path}: parameter count {info.num_params} does not match the "
                "project architecture signature 8,631,454 (12 blocks x 128 filters)."
            )
        return
    if (num_blocks, num_filters) != (
        EXPECTED_ARCHITECTURE["num_blocks"],
        EXPECTED_ARCHITECTURE["num_filters"],
    ):
        raise WarmStartValidationError(
            f"{info.path}: architecture {num_blocks}x{num_filters} does not match the "
            f"required {EXPECTED_ARCHITECTURE['num_blocks']}x{EXPECTED_ARCHITECTURE['num_filters']}."
        )


def validate_warm_start(
    path: str | Path,
    repo_root: str | Path = ".",
    allow_unverified: bool = False,
) -> Dict[str, Any]:
    """Validate an RL warm-start checkpoint; return its provenance record.

    Raises :class:`WarmStartValidationError` when the file is absent, unreadable,
    has the wrong architecture, or — unless ``allow_unverified`` is set by the
    operator — is not a designated warm start in the registry.
    """
    p = Path(path)
    if not p.is_absolute():
        candidate = Path(repo_root) / p
        p = candidate if candidate.is_file() else p
    if not p.is_file():
        raise WarmStartValidationError(
            f"Warm-start checkpoint not found: {p}. Refusing to start RL training "
            "from a missing checkpoint (no silent fallback)."
        )

    info = inspect_checkpoint(p)
    _check_architecture(info)

    sha16 = info.sha256[:16]
    try:
        registry = load_registry(repo_root)
    except FileNotFoundError:
        registry = None

    entry: Optional[Dict[str, Any]] = None
    if registry:
        for rec in registry.get("checkpoints", []):
            if rec.get("sha256") == info.sha256 or rec.get("sha256", "").startswith(sha16):
                entry = rec
                break

    if entry is None:
        if not allow_unverified:
            raise WarmStartValidationError(
                f"Warm-start {p} (sha256 {sha16}...) is not in the checkpoint registry. "
                "Designate it first (scripts/designate_baseline.py or manual registry "
                "edit + scripts/validate_checkpoints.py), or pass --allow-unverified-"
                "warm-start to override with a logged warning."
            )
        logger.warning(
            "[WARM-START OVERRIDE] Using unregistered checkpoint %s (sha256 %s...); "
            "record this decision in the run log and the registry.",
            p,
            sha16,
        )
        return {
            "path": str(p),
            "sha256": info.sha256,
            "status": "unverified-override",
            "step": info.step,
            "epoch": info.epoch,
            "model_version": info.model_version,
            "best_val_loss": info.best_val_loss,
        }

    status = entry.get("status")
    if status != "verified":
        if status == "disqualified":
            dq = entry.get("disqualification", {})
            raise WarmStartValidationError(
                f"Warm-start {p} is DISQUALIFIED in the registry "
                f"({dq.get('reason', 'undocumented')}): {dq.get('detail', '')} "
                "A disqualified checkpoint can never become a warm start; see "
                "docs/checkpoint_registry.json for the evidence trail."
            )
        raise WarmStartValidationError(
            f"Warm-start {p} is registered with status {status!r} (missing/unverified "
            "artifacts can never be warm starts). See docs/checkpoint_registry.json."
        )
    if not entry.get("designated_warm_start", False) and not allow_unverified:
        raise WarmStartValidationError(
            f"Warm-start {p} is verified but NOT designated as a warm start "
            f"(role={entry.get('role')!r}). Designate a baseline explicitly before "
            "starting an RL campaign, or pass --allow-unverified-warm-start to override."
        )

    logger.info(
        "[WARM-START] %s verified: sha256=%s step=%s epoch=%s role=%s designation_evidence=%s",
        p,
        sha16,
        info.step,
        info.epoch,
        entry.get("role"),
        entry.get("designation_evidence", "n/a"),
    )
    return {
        "path": str(p),
        "sha256": info.sha256,
        "status": "verified",
        "role": entry.get("role"),
        "step": info.step,
        "epoch": info.epoch,
        "model_version": info.model_version,
        "best_val_loss": info.best_val_loss,
    }
