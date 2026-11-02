"""Checkpoint provenance registry, default resolution, and warm-start validation.

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
4. Default inference resolution (:func:`resolve_default_checkpoint`) follows
   the same fail-closed contract (added 2026-10-31, audit M-2): the single
   authoritative default is the registry-designated, verified artifact,
   hash-checked against the registry at resolution time.  When none is
   available, resolution raises :class:`DefaultCheckpointError` with the exact
   expected artifact and a recovery path.  It never falls back to "newest
   file on disk" — a checkpoint that merely exists (or loads) is not eligible.

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


class DefaultCheckpointError(RuntimeError):
    """Raised when no trustworthy default checkpoint can be resolved.

    The message is operator-facing and actionable: it names the expected
    artifact(s), why each candidate is ineligible, and the supported recovery
    actions.  Callers must surface it (CLI: print + exit 1) rather than
    substituting any checkpoint of convenience.
    """


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
    """Hash a checkpoint and read its metadata (without loading weights to GPU).

    Loading is tensor-only (``weights_only=True``, finding CKP-06): this is the
    path that *authenticates* warm-start candidates, so it must never execute
    pickle code from the file it is judging. A payload whose weights are absent
    is rejected here — an empty mapping must not slip past the architecture
    check as if it were a real model.
    """
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as e:  # pragma: no cover - torch without weights_only support
        raise RuntimeError(
            f"Refusing to inspect '{path}': this torch build cannot load with weights_only=True, "
            "and unrestricted unpickling is not permitted on the authentication path."
        ) from e
    if not isinstance(ck, dict):
        raise WarmStartValidationError(
            f"{path}: checkpoint payload is {type(ck).__name__}, not a dict."
        )
    state = ck.get("model_state_dict") or ck.get("model") or {}
    if not isinstance(state, dict) or not state:
        raise WarmStartValidationError(
            f"{path}: checkpoint carries no model weights (empty mapping); it cannot be "
            "authenticated as a warm-start candidate."
        )
    num_params = sum(t.numel() for t in state.values() if hasattr(t, "numel"))
    if num_params == 0:
        raise WarmStartValidationError(
            f"{path}: checkpoint weights contain no tensors; it cannot be authenticated."
        )
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
                "cuda_rng_state",
            )
        },
    )


def load_registry(repo_root: str | Path = ".") -> Dict[str, Any]:
    """Load ``docs/checkpoint_registry.json``; raise if absent.

    Resolution order (audit S6-01): the repository-checkout path first —
    byte-identical behavior for developers — then the copy staged into the
    installed package (``python/_data/checkpoint_registry.json``, see
    ``scripts/build_wheel.py``) so a wheel install outside a checkout can
    still load the registry instead of dying with ``FileNotFoundError``. When
    neither exists, the error names the canonical repository location.
    """
    from python.common.resource_paths import resolve_data_file

    repo_reg_path = Path(repo_root) / REGISTRY_REL_PATH
    reg_path = resolve_data_file(repo_reg_path, "_data", "checkpoint_registry.json")
    if not reg_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint registry not found at {repo_reg_path}. Run "
            "scripts/validate_checkpoints.py --write to create it from the "
            "current artifacts."
        )
    return json.loads(reg_path.read_text(encoding="utf-8"))


def registry_path(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / REGISTRY_REL_PATH


def find_registry_root(start: str | Path) -> Optional[Path]:
    """Nearest ancestor of ``start`` that contains ``docs/checkpoint_registry.json``.

    Checkpoints live in gitignored ``models/`` trees that may belong to a
    different checkout than the running code (worktrees, sibling trees). Walking
    up from the artifact finds the governance record that actually covers it;
    returns ``None`` when no ancestor carries a registry.
    """
    p = Path(start)
    if p.is_file():
        p = p.parent
    try:
        p = p.resolve()
    except OSError:
        return None
    for candidate in [p, *p.parents]:
        if (candidate / REGISTRY_REL_PATH).is_file():
            return candidate
    return None


def write_registry(
    registry: Dict[str, Any],
    repo_root: str | Path = ".",
    *,
    stamp_verified: bool = False,
) -> Path:
    """Atomically write the registry under an exclusive lock (finding CKP-14).

    The historical ``--write`` path rewrote the tracked governance file with a
    bare ``write_text``: a crash mid-write left torn JSON. Writes now go
    through tmp + fsync + ``os.replace`` while holding the registry's
    cross-process lock, so two verifier/designation sessions cannot interleave.
    """
    from python.common.artifacts import artifact_lock, atomic_write_json

    reg_path = registry_path(repo_root)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(registry)
    if stamp_verified:
        import time as _time

        payload["last_verified"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    with artifact_lock(reg_path, timeout=30.0, owner="registry-write"):
        atomic_write_json(reg_path, payload)
    return reg_path


def upsert_registry_entry(
    entry: Dict[str, Any],
    repo_root: str | Path = ".",
) -> Dict[str, Any]:
    """Insert or replace one registry entry (matched by ``id``, then path/sha).

    Returns the updated registry. The registry stays the single governance
    record: this is the in-repo appender path finding CKP-14 asked for, and it
    is atomic + locked like :func:`write_registry`.
    """
    registry = load_registry(repo_root)
    entries = registry.setdefault("checkpoints", [])
    entry_id = entry.get("id")
    entry_file = entry.get("file")
    entry_sha = entry.get("sha256")
    for idx, rec in enumerate(entries):
        same = False
        if entry_id and rec.get("id") == entry_id:
            same = True
        elif entry_file and rec.get("file") == entry_file:
            same = True
        elif entry_sha and rec.get("sha256") == entry_sha:
            same = True
        if same:
            entries[idx] = {**rec, **entry}
            break
    else:
        entries.append(entry)
    write_registry(registry, repo_root)
    return registry


def verify_checkpoint_identity(
    path: str | Path,
    *,
    repo_root: Optional[str | Path] = None,
    hash_file: bool = True,
) -> Dict[str, Any]:
    """Verify the exact file at ``path`` against the checkpoint registry (phase J).

    Returns a verdict dict with ``status`` one of ``verified``, ``unregistered``,
    ``hash_mismatch``, ``quarantined``, ``disqualified``, ``missing``, or
    ``unreadable``. An entry is matched by *path*, and a hash that matches a
    different entry's record is reported as ``copied_from_other_path`` instead
    of silently transferring that artifact's designation.

    ``hash_file=True`` (the default, used at LOAD time) computes the SHA-256 of
    the exact bytes being loaded, so a file replaced after registration cannot
    inherit its record. ``hash_file=False`` is the cheap variant for hot paths
    (default-model resolution runs on every CLI invocation): it compares the
    registry's recorded size when present and never reads the file contents.
    Status decisions (quarantined/disqualified/missing) are identical in both
    modes.
    """
    p = Path(path)
    if not p.is_file():
        return {"status": "unreadable", "sha256": None, "detail": "file not found"}
    try:
        stat = p.stat()
    except OSError as e:
        return {"status": "unreadable", "sha256": None, "detail": str(e)}
    if hash_file:
        try:
            sha: Optional[str] = compute_sha256(p)
        except OSError as e:
            return {"status": "unreadable", "sha256": None, "detail": str(e)}
    else:
        sha = None
    root: Optional[Path]
    if repo_root is not None:
        root = Path(repo_root)
    else:
        root = find_registry_root(p)
    if root is None:
        return {
            "status": "unregistered",
            "sha256": sha,
            "detail": "no registry found for this artifact's tree",
        }
    try:
        registry = load_registry(root)
    except FileNotFoundError:
        return {
            "status": "unregistered",
            "sha256": sha,
            "detail": "no registry in this checkout",
        }
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p

    def _entry_path_matches(rec: Dict[str, Any]) -> bool:
        rec_file = rec.get("file") or rec.get("path")
        if not rec_file:
            return False
        rec_path = Path(rec_file)
        if not rec_path.is_absolute():
            rec_path = root / rec_path
        try:
            return rec_path.resolve() == resolved
        except OSError:
            return str(rec_path) == str(resolved)

    same_path: Optional[Dict[str, Any]] = None
    same_hash: Optional[Dict[str, Any]] = None
    for rec in registry.get("checkpoints", []):
        if _entry_path_matches(rec):
            same_path = rec
            break
    if sha is not None:
        for rec in registry.get("checkpoints", []):
            rec_sha = rec.get("sha256")
            if rec_sha and rec_sha == sha:
                same_hash = rec
                break

    if same_path is None:
        if same_hash is not None:
            return {
                "status": "copied_from_other_path",
                "sha256": sha,
                "entry_id": same_hash.get("id"),
                "recorded_file": same_hash.get("file"),
                "detail": (
                    f"content matches registry entry {same_hash.get('id')!r} recorded at "
                    f"{same_hash.get('file')!r}, but this file lives elsewhere: a copy does "
                    "not inherit that artifact's designation"
                ),
            }
        return {
            "status": "unregistered",
            "sha256": sha,
            "detail": "no registry entry for this path or content",
        }

    status = same_path.get("status")
    recorded_sha = same_path.get("sha256")
    if hash_file and recorded_sha and recorded_sha != sha:
        return {
            "status": "hash_mismatch",
            "sha256": sha,
            "expected": recorded_sha,
            "entry_id": same_path.get("id"),
            "detail": f"entry {same_path.get('id')}: file changed since verification",
        }
    if not hash_file:
        # Cheap staleness signal: a recorded size that disagrees means the file
        # changed since verification.
        recorded_size = same_path.get("size_bytes") or same_path.get("size")
        if recorded_size is not None:
            try:
                if int(recorded_size) != int(stat.st_size):
                    return {
                        "status": "hash_mismatch",
                        "sha256": None,
                        "expected": recorded_sha,
                        "entry_id": same_path.get("id"),
                        "detail": (
                            f"entry {same_path.get('id')}: size changed since verification "
                            f"({recorded_size} -> {stat.st_size})"
                        ),
                    }
            except (TypeError, ValueError):
                pass
    if status == "quarantined":
        return {
            "status": "quarantined",
            "sha256": sha,
            "entry_id": same_path.get("id"),
            "detail": same_path.get("quarantine", {}).get("reason", "quarantined artifact"),
        }
    if status == "disqualified":
        return {
            "status": "disqualified",
            "sha256": sha,
            "entry_id": same_path.get("id"),
            "detail": same_path.get("disqualification", {}).get("reason", "disqualified artifact"),
        }
    if status == "missing":
        return {
            "status": "missing",
            "sha256": sha,
            "entry_id": same_path.get("id"),
            "detail": "entry is declared missing but a file is present at its path",
        }
    return {
        "status": "verified",
        "sha256": sha,
        "entry_id": same_path.get("id"),
        "role": same_path.get("role"),
        "designated_warm_start": bool(same_path.get("designated_warm_start", False)),
    }


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

    # Match by path first, then by content. A byte-identical *copy* at another
    # path must not inherit the original artifact's warm-start designation
    # (finding CKP-04 corollary: paths are identity, hashes are integrity).
    entry: Optional[Dict[str, Any]] = None
    content_match: Optional[Dict[str, Any]] = None
    if registry:
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        for rec in registry.get("checkpoints", []):
            rec_file = rec.get("file") or rec.get("path")
            rec_path = Path(rec_file) if rec_file else None
            if rec_path is not None and not rec_path.is_absolute():
                rec_path = Path(repo_root) / rec_path
            if rec_path is not None:
                try:
                    if rec_path.resolve() == resolved:
                        entry = rec
                        break
                except OSError:
                    pass
        if entry is None:
            for rec in registry.get("checkpoints", []):
                # Registry records for missing/quarantined artifacts legitimately
                # carry sha256: null; those must be skipped, not crash the
                # validator (`None.startswith` raised AttributeError and aborted
                # every warm-start construction; SR-02 / SLQ-07 hit the same bug).
                rec_sha = rec.get("sha256")
                if rec_sha and (rec_sha == info.sha256 or str(rec_sha).startswith(sha16)):
                    content_match = rec
                    break

    if entry is None and content_match is not None and not allow_unverified:
        raise WarmStartValidationError(
            f"Warm-start {p} (sha256 {sha16}...) is byte-identical to registry entry "
            f"{content_match.get('id')!r} recorded at {content_match.get('file')!r}, but it "
            "lives at a different path. A copy does not inherit another artifact's "
            "warm-start designation; use the registered path, or pass "
            "--allow-unverified-warm-start to override with a logged warning."
        )

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


# ---------------------------------------------------------------------------
# Default inference checkpoint resolution (audit M-2, 2026-10-31)
#
# Single authoritative resolver for every CLI/inference consumer.  The old
# world had two resolvers (python/cli/common.get_default_checkpoint_path and
# python/cli/models.get_default_model_path) with different candidate lists
# that could disagree, and both silently fell through to "newest .pt on disk"
# — which, after the certified artifacts went missing, meant every command
# quietly evaluated the disqualified weak scaled checkpoint.  The registry
# (docs/checkpoint_registry.json) is the source of truth; eligibility is
# status=="verified" AND designated_warm_start==true; the artifact's SHA-256
# is re-verified against the registry before it may be used.
# ---------------------------------------------------------------------------


@dataclass
class DefaultCheckpointResolution:
    """The outcome of :func:`resolve_default_checkpoint`."""

    path: Path
    sha256: str
    entry_id: str
    role: Optional[str] = None
    step: Optional[int] = None
    designated: bool = True

    @property
    def sha256_16(self) -> str:
        return self.sha256[:16]


def _entry_status_line(entry: Dict[str, Any]) -> str:
    status = entry.get("status")
    designated = "designated" if entry.get("designated_warm_start") else "not designated"
    reason = ""
    if status == "disqualified":
        dq = entry.get("disqualification") or {}
        reason = f" — disqualified: {dq.get('reason', 'undocumented')}"
    line = f"  - {entry.get('id', '<no id>')} [{status}, {designated}] {entry.get('path')}{reason}"
    return line


def _default_entry_path(
    entry: Dict[str, Any], repo_root: Path, models_root: Optional[Path]
) -> Path:
    # Registry entries record the artifact path under ``file`` (the canonical
    # key used by verify_checkpoint_identity and the tracked registry JSON);
    # ``path`` is the legacy alias some entries carry. Read both — an entry
    # with only one of them must still resolve.
    rel = Path(str(entry.get("file") or entry.get("path") or ""))
    if models_root is not None and rel.parts and rel.parts[0] == "models":
        return Path(models_root) / Path(*rel.parts[1:])
    return repo_root / rel


def _explain_registry(registry: Optional[Dict[str, Any]], registry_path: Path) -> str:
    if registry is None:
        return f"  (registry file not found: {registry_path})"
    lines = [_entry_status_line(e) for e in registry.get("checkpoints", [])]
    return "\n".join(lines) if lines else "  (registry contains no checkpoint entries)"


def _recovery_hint(repo_root: Path) -> str:
    return (
        "Recovery:\n"
        "  1. Restore or re-acquire the designated artifact at the exact path above, or\n"
        "  2. Designate a verified checkpoint in docs/checkpoint_registry.json "
        "(procedure in its 'designation_procedure' field) and re-verify with "
        "scripts/validate_checkpoints.py, or\n"
        "  3. Pass an explicit checkpoint path (e.g. --checkpoint <file.pt>).\n"
        "Refusing to fall back to an arbitrary or unqualified checkpoint."
    )


def resolve_default_checkpoint(
    repo_root: str | Path = ".",
    models_root: Optional[str | Path] = None,
    verify_hash: bool = True,
) -> DefaultCheckpointResolution:
    """Resolve the single authoritative default checkpoint (fail-closed).

    Policy: exactly one policy for every consumer.  The default is the
    registry entry with ``status == "verified"`` and
    ``designated_warm_start == true`` whose file exists on disk and whose
    SHA-256 matches the registry record.  If several entries are designated
    (a curation smell — a warning is logged), the one with the highest
    registry id wins (ids are assigned in designation order, so this is
    deterministic and documented).

    Raises:
        DefaultCheckpointError: when no such artifact exists, or the
            designated artifact is missing/hash-mismatched.  The message
            lists the expected artifact, the ineligibility reasons of every
            registry candidate, and recovery actions.
    """
    repo_root = Path(repo_root)
    models_root_p = Path(models_root) if models_root is not None else None
    registry_path = repo_root / REGISTRY_REL_PATH

    try:
        registry = load_registry(repo_root)
    except FileNotFoundError as e:
        raise DefaultCheckpointError(
            "No checkpoint registry found; cannot resolve a trustworthy default "
            f"checkpoint ({e})\n"
            f"Registry path expected: {registry_path}\n" + _recovery_hint(repo_root)
        ) from e

    entries = list(registry.get("checkpoints", []))
    eligible = [
        e
        for e in entries
        if e.get("status") == "verified" and e.get("designated_warm_start") and e.get("sha256")
    ]
    designated_but_shapeless = [
        e
        for e in entries
        if e.get("status") == "verified" and e.get("designated_warm_start") and not e.get("sha256")
    ]

    if not eligible:
        detail = ""
        if designated_but_shapeless:
            detail = (
                "\nA designated entry exists but records no SHA-256 "
                "(never verified from disk): "
                + ", ".join(str(e.get("id")) for e in designated_but_shapeless)
            )
        raise DefaultCheckpointError(
            "No designated, verified default checkpoint is available.\n"
            f"Registry: {registry_path}\n"
            f"Registry contents:\n{_explain_registry(registry, registry_path)}"
            f"{detail}\n" + _recovery_hint(repo_root)
        )

    eligible.sort(key=lambda e: str(e.get("id", "")), reverse=True)
    if len(eligible) > 1:
        logger.warning(
            "Multiple designated checkpoints in the registry (%s); selecting %s "
            "(highest registry id — deterministic rule). Consider undesignating "
            "the others.",
            ", ".join(str(e.get("id")) for e in eligible),
            eligible[0].get("id"),
        )
    entry = eligible[0]
    path = _default_entry_path(entry, repo_root, models_root_p)

    if not path.is_file():
        raise DefaultCheckpointError(
            "The designated default checkpoint is recorded in the registry but "
            f"the file is MISSING on disk.\n"
            f"  Expected artifact: {path}\n"
            f"  Registry entry:    {entry.get('id')} (status=verified, designated)\n"
            f"  Registry path:     {registry_path}\n" + _recovery_hint(repo_root)
        )

    size_bytes = entry.get("size_bytes")
    if size_bytes is not None:
        actual_size = path.stat().st_size
        if actual_size != int(size_bytes):
            raise DefaultCheckpointError(
                "The designated default checkpoint does not match its registry "
                "size (modified or replaced after verification).\n"
                f"  Artifact:        {path}\n"
                f"  Size on disk:    {actual_size}\n"
                f"  Registry size:   {size_bytes}\n"
                f"  Registry entry:  {entry.get('id')}\n" + _recovery_hint(repo_root)
            )

    sha256 = str(entry.get("sha256"))
    if verify_hash:
        actual_sha = compute_sha256(path)
        if actual_sha != sha256:
            raise DefaultCheckpointError(
                "The designated default checkpoint does not match its registry "
                "SHA-256 (modified or replaced after verification).\n"
                f"  Artifact:        {path}\n"
                f"  SHA-256 on disk: {actual_sha}\n"
                f"  Registry SHA-256:{sha256}\n"
                f"  Registry entry:  {entry.get('id')}\n" + _recovery_hint(repo_root)
            )

    resolution = DefaultCheckpointResolution(
        path=path,
        sha256=sha256,
        entry_id=str(entry.get("id")),
        role=entry.get("role"),
        step=entry.get("step"),
        designated=True,
    )
    logger.info(
        "Default checkpoint resolved via registry: %s (entry %s, sha256 %s..., step %s)",
        resolution.path,
        resolution.entry_id,
        resolution.sha256_16,
        resolution.step,
    )
    return resolution


def explicit_checkpoint_warnings(
    path: str | Path,
    repo_root: str | Path = ".",
    verify_hash: bool = True,
) -> Optional[str]:
    """Provenance warnings for an operator-supplied checkpoint path.

    Explicit paths are honored (the operator's choice is not silent), but the
    provenance is surfaced: disqualified/quarantined registry entries and
    unregistered files produce a loud warning string; a registered path whose
    file no longer matches the recorded hash warns about the mismatch.
    Returns ``None`` when the checkpoint is a verified registry artifact.
    """
    repo_root = Path(repo_root)
    p = Path(path)
    if not p.is_absolute():
        candidate = repo_root / p
        p = candidate if candidate.is_file() else p

    try:
        registry = load_registry(repo_root)
    except FileNotFoundError:
        return (
            f"Checkpoint {p} is being used without a checkpoint registry "
            f"({repo_root / REGISTRY_REL_PATH} not found); provenance unverified."
        )

    entry = None
    for rec in registry.get("checkpoints", []):
        rec_path = _default_entry_path(rec, repo_root, None)
        try:
            same = rec_path.resolve() == p.resolve()
        except OSError:  # pragma: no cover - defensive
            same = False
        if same:
            entry = rec
            break

    if entry is None:
        return (
            f"Checkpoint {p} is not in the checkpoint registry "
            "(unverified provenance — acceptable for explicit/experimental use, "
            "but it is not an approved default)."
        )

    status = entry.get("status")
    if status == "disqualified":
        dq = entry.get("disqualification") or {}
        return (
            f"Checkpoint {p} is DISQUALIFIED in the registry "
            f"({dq.get('reason', 'undocumented')}): {dq.get('detail', '')} "
            "This artifact must not be used as a default or warm start."
        )
    if status != "verified":
        return (
            f"Checkpoint {p} is registered with status {status!r} "
            "(not verified); treat its results as unverified provenance."
        )

    sha256 = entry.get("sha256")
    if verify_hash and sha256:
        actual = compute_sha256(p)
        if actual != sha256:
            return (
                f"Checkpoint {p} does not match its registry SHA-256 "
                f"(on disk {actual[:16]}..., registry {str(sha256)[:16]}...); "
                "the file was modified or replaced after verification — "
                "results from it are not covered by the registry record."
            )
    return None
