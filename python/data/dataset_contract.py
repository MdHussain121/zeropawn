"""Dataset identity contract: roles, label semantics, schema, and provenance.

This module is the single source of truth for what a ZeroPawn binary dataset
split *means*:

- which role a dataset plays (game WDL supervision, policy-only tactics,
  endgame tablebase supervision, or a unified mixture),
- which outcome-label values are valid for that role,
- whether the role requires real WDL labels (a dataset whose every value
  target is masked must never be consumed as WDL-supervised game data), and
- how dataset artifacts are identified for provenance (hashes, sizes,
  schema/build-record versions).

Roles are declared by builders in ``metadata.json`` and must be declared
explicitly by consumers (trainers, replay buffers, evaluators). Role is never
inferred from directory names alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from python.constants import BOARD_SIZE, NUM_ACTIONS as NUM_ACTIONS, NUM_PLANES as NUM_PLANES

# Schema of the binary split arrays (planes/moves/outcomes/game_ids/plies and
# optional value/policy/dtz targets). Bump when the layout changes.
SCHEMA_VERSION = "1.1"

# Version of the validation receipt format written by dataset_validation.py.
RECEIPT_VERSION = "1.0"

# Version of the per-build provenance record written by dataset builders.
BUILD_RECORD_VERSION = "1.0"

OUTCOME_UNAVAILABLE: int = 255  # masked value target (policy-only example)
OUTCOME_WIN: int = 0  # side to move wins (mover perspective)
OUTCOME_DRAW: int = 1
OUTCOME_LOSS: int = 2

# Syzygy 5-class outcomes (cursed win / blessed loss distinctions).
OUTCOME_SYZYGY_DOMAIN: tuple[int, ...] = (0, 1, 2, 3, 4)

# Re-sourced from python.constants (S7-01 reconnection): the enforced contract
# mirrors cpp/include/zeropawn/encoder.hpp via python/constants.py.
PLANES_SHAPE = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

# Schema-2 datasets store the per-position legal-action mask bit-packed over
# NUM_ACTIONS actions (audit SA2-01: masks.bin is part of the validated and
# receipt-bound artifact set, so its exact layout lives in this contract).
# num_samples rows x MASK_PACKED_BYTES uint8 = the exact size of masks.bin.
MASK_PACKED_BYTES: int = (NUM_ACTIONS + 7) // 8  # 4672 actions -> 584 bytes

# Required array files for every split (the loader memory-maps all of these
# unconditionally, so any loadable dataset must carry them).
REQUIRED_ARRAYS: tuple[str, ...] = (
    "planes.bin",
    "moves.bin",
    "outcomes.bin",
    "game_ids.bin",
    "plies.bin",
)

# Optional arrays validated for exact size when present.
OPTIONAL_ARRAYS: tuple[str, ...] = ("value_targets.bin", "policy_targets.bin", "dtz_targets.bin")

# File that binds a build to its source and configuration (written by builders).
BUILD_RECORD_NAME = "build_record.json"
# File that binds a validation result to the exact bytes validated.
RECEIPT_NAME = "validation_receipt.json"


@dataclass(frozen=True)
class DatasetRole:
    """Semantic contract for a dataset role."""

    name: str
    description: str
    # Outcome values that may legally appear in outcomes.bin for this role.
    valid_outcomes: tuple[int, ...]
    # True when the role is meaningless without real WDL labels: a split whose
    # every outcome is OUTCOME_UNAVAILABLE must fail validation/consumption.
    requires_wdl_labels: bool
    # True when masked outcomes (OUTCOME_UNAVAILABLE) are an expected part of
    # the role's format (policy-only tactical data).
    allows_masked: bool


ROLE_GAMES_WDL = DatasetRole(
    name="games_wdl",
    description="Human/master game positions with mover-perspective WDL outcome labels.",
    valid_outcomes=(OUTCOME_WIN, OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_UNAVAILABLE),
    requires_wdl_labels=True,
    allows_masked=True,
)

ROLE_POLICY_TACTICS = DatasetRole(
    name="policy_tactics",
    description=(
        "Tactical puzzle positions training the policy head; value targets are masked "
        "(OUTCOME_UNAVAILABLE) for every row whose solution does not deliver checkmate. "
        "Mate-delivering rows carry a genuine mover-relative win label (outcome=0) at "
        "ply=10, and the loaders pass that label through so the trainer's WDL mask "
        "(ply >= 10 and non-zero target) supervises the value head for them (audit SLQ-06 "
        "contract decision: a genuinely available and supported target is never silently "
        "discarded, and unavailable targets remain masked)."
    ),
    valid_outcomes=(OUTCOME_UNAVAILABLE, OUTCOME_WIN, OUTCOME_DRAW, OUTCOME_LOSS),
    requires_wdl_labels=False,
    allows_masked=True,
)

ROLE_SYZYGY = DatasetRole(
    name="endgame_syzygy",
    description="Tablebase endgame positions with 5-class syzygy outcome labels.",
    valid_outcomes=OUTCOME_SYZYGY_DOMAIN + (OUTCOME_UNAVAILABLE,),
    requires_wdl_labels=True,
    allows_masked=True,
)

ROLE_UNIFIED = DatasetRole(
    name="unified",
    description="Mixture of game (WDL-labelled) and tactical (masked) positions.",
    valid_outcomes=(OUTCOME_WIN, OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_UNAVAILABLE)
    + OUTCOME_SYZYGY_DOMAIN,
    requires_wdl_labels=True,
    allows_masked=True,
)

# Metadata "kind"/"role" strings produced by builders, mapped to roles.
_KIND_TO_ROLE: Dict[str, DatasetRole] = {
    "games_wdl": ROLE_GAMES_WDL,
    "games": ROLE_GAMES_WDL,
    "master": ROLE_GAMES_WDL,
    "master-games": ROLE_GAMES_WDL,
    "master_games": ROLE_GAMES_WDL,
    "master_games_2500": ROLE_GAMES_WDL,
    "master_games_additional": ROLE_GAMES_WDL,
    "game_data": ROLE_GAMES_WDL,
    "policy_tactics": ROLE_POLICY_TACTICS,
    "tactics": ROLE_POLICY_TACTICS,
    "tactics_puzzles": ROLE_POLICY_TACTICS,
    "tactics-fullline": ROLE_POLICY_TACTICS,
    "puzzles": ROLE_POLICY_TACTICS,
    "dataset_tactics": ROLE_POLICY_TACTICS,
    "endgame_syzygy": ROLE_SYZYGY,
    "syzygy": ROLE_SYZYGY,
    "openings": ROLE_GAMES_WDL,
    "opening": ROLE_GAMES_WDL,
    "openings_policy": ROLE_GAMES_WDL,
    "unified": ROLE_UNIFIED,
    "unified_training": ROLE_UNIFIED,
}


class DatasetContractError(ValueError):
    """Raised when a dataset violates its declared or required role contract."""


def role_by_name(name: str) -> Optional[DatasetRole]:
    """Look up a role by its canonical name (e.g. 'games_wdl')."""
    registry = {
        ROLE_GAMES_WDL.name: ROLE_GAMES_WDL,
        ROLE_POLICY_TACTICS.name: ROLE_POLICY_TACTICS,
        ROLE_SYZYGY.name: ROLE_SYZYGY,
        ROLE_UNIFIED.name: ROLE_UNIFIED,
    }
    return registry.get(str(name).strip().lower())


def classify_role(
    metadata: Dict[str, Any],
    data_dir: Optional[Path] = None,
    default: Optional[DatasetRole] = None,
) -> Optional[DatasetRole]:
    """Resolve a split's role from declared metadata.

    Precedence: explicit ``role`` key, then ``label_semantics.role``, then the
    builder ``kind``. Directory names are intentionally NOT consulted here.
    Returns ``default`` (usually None) when the metadata declares nothing
    recognizable; callers must treat an unclassified dataset as a gate failure
    rather than silently assuming a role.
    """
    if not isinstance(metadata, dict):
        return default
    role_key = metadata.get("role")
    if isinstance(role_key, str) and role_key.strip():
        role = role_by_name(role_key)
        if role is not None:
            return role
    semantics = metadata.get("label_semantics")
    if isinstance(semantics, dict):
        role_key = semantics.get("role")
        if isinstance(role_key, str) and role_key.strip():
            role = role_by_name(role_key)
            if role is not None:
                return role
    kind = metadata.get("kind")
    if isinstance(kind, str) and kind.strip():
        role = _KIND_TO_ROLE.get(kind.strip().lower())
        if role is not None:
            return role
    return default


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Streamed SHA-256 of a file (hex digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    """Identity triple for a file: size, mtime_ns, and SHA-256.

    The full hash makes identity cryptographic; size+mtime_ns allow a cheap
    (non-cryptographic) staleness re-check for consumers that do not want to
    re-hash gigabyte-scale planes files.
    """
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def identity_matches(path: Path, identity: Dict[str, Any]) -> bool:
    """Cheap staleness check: size and mtime_ns against a recorded identity."""
    try:
        st = path.stat()
    except OSError:
        return False
    return st.st_size == identity.get("size") and st.st_mtime_ns == identity.get("mtime_ns")


def bytes_per_sample() -> int:
    """Exact byte size of one record across all required arrays."""
    return (
        PLANES_SHAPE[0] * PLANES_SHAPE[1] * PLANES_SHAPE[2] * 4  # float32 planes
        + 2  # int16 move
        + 1  # uint8 outcome
        + 8  # uint64 game id
        + 2  # int16 ply
    )


def remap_syzygy_outcomes(outcomes: Any) -> Any:
    """Remap 5-class Syzygy outcomes into 3-class mover-relative targets (SY-02).

    This is the RL-side twin of ``python.data.dataset._outcome_to_value_target``
    (is_syzygy=True): both MUST implement the identical SY-02 convention, where
    cursed wins and blessed losses are draws under FIDE rule 50
    (commit c3cc6cc, 2026-11-01). ``tests/test_syzygy_mapping_unification_20261103.py``
    pins the equality of the two paths.

    Convention:
      0 (Win)          -> 0 (Win for side to move)
      1 (Cursed Win)   -> 1 (Draw; cannot be won within 50 moves)
      2 (Draw)         -> 1 (Draw)
      3 (Blessed Loss) -> 1 (Draw; loss averted by the 50-move rule)
      4 (Loss)         -> 2 (Loss for side to move)
      255 (OUTCOME_UNAVAILABLE) -> 255 (Masked / unavailable; policy-only row)

    Supports numpy ndarray or torch Tensor, or scalar integers.
    Raises ValueError on out-of-domain values.
    """
    if isinstance(outcomes, np.ndarray):
        arr = outcomes.copy()
        mask_255 = arr == OUTCOME_UNAVAILABLE
        invalid = (~mask_255) & ((arr < 0) | (arr > 4))
        if np.any(invalid):
            bad_vals = np.unique(arr[invalid]).tolist()
            raise ValueError(f"Invalid Syzygy outcome value(s): {bad_vals}; expected 0..4 or 255")
        res = np.empty_like(arr)
        res[mask_255] = OUTCOME_UNAVAILABLE
        res[arr == 0] = OUTCOME_WIN
        res[(arr == 1) | (arr == 2) | (arr == 3)] = OUTCOME_DRAW
        res[arr == 4] = OUTCOME_LOSS
        return res

    import torch

    if isinstance(outcomes, torch.Tensor):
        t = outcomes.clone()
        mask_255 = t == OUTCOME_UNAVAILABLE
        invalid = (~mask_255) & ((t < 0) | (t > 4))
        if invalid.any():
            bad_vals = torch.unique(t[invalid]).tolist()
            raise ValueError(f"Invalid Syzygy outcome value(s): {bad_vals}; expected 0..4 or 255")
        res = torch.empty_like(t)
        res[mask_255] = OUTCOME_UNAVAILABLE
        res[t == 0] = OUTCOME_WIN
        res[(t == 1) | (t == 2) | (t == 3)] = OUTCOME_DRAW
        res[t == 4] = OUTCOME_LOSS
        return res

    # Scalar int
    val = int(outcomes)
    if val == OUTCOME_UNAVAILABLE:
        return OUTCOME_UNAVAILABLE
    if val == 0:
        return OUTCOME_WIN
    if val in (1, 2, 3):
        return OUTCOME_DRAW
    if val == 4:
        return OUTCOME_LOSS
    raise ValueError(f"Invalid Syzygy outcome value: {val}; expected 0..4 or 255")


@dataclass
class BuildRecord:
    """Provenance record for a dataset build (written next to the output).

    Every newly built artifact must be reproducible from the fields below:
    exact source identity, builder identity, configuration, measured counts,
    and output hashes.
    """

    builder: str
    builder_version: str = "1.0"
    schema_version: str = SCHEMA_VERSION
    source: Dict[str, Any] = field(default_factory=dict)  # path/sha256/rows
    config: Dict[str, Any] = field(default_factory=dict)
    label_semantics: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)  # per-split file hashes
    git_commit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        from datetime import datetime, timezone
        import sys

        env = {
            "python": sys.version.split()[0],
        }
        if "torch" in sys.modules:
            import torch

            env["torch"] = getattr(torch, "__version__", None)
        if "numpy" in sys.modules:
            import numpy as _np

            env["numpy"] = getattr(_np, "__version__", None)

        return {
            "build_record_version": BUILD_RECORD_VERSION,
            "builder": self.builder,
            "builder_version": self.builder_version,
            "schema_version": self.schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": self.git_commit,
            "environment": env,
            "source": self.source,
            "config": self.config,
            "label_semantics": self.label_semantics,
            "counts": self.counts,
            "outputs": self.outputs,
            "notes": self.notes,
        }

    def write(self, output_dir: Path) -> Path:
        path = Path(output_dir) / BUILD_RECORD_NAME
        with open(path, "w", encoding="utf-8") as f:
            import json

            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        return path


def detect_git_commit() -> Optional[str]:
    """Best-effort HEAD commit for provenance records; never raises."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None
