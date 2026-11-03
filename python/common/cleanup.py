"""Post-run cleanup and retention utilities for ZeroPawn.

Junk this module owns (never touches code / datasets / models):
- runs/<run>/buffer/*.bin + meta.json (preallocated replay memmaps)
- runs/**/events.out.tfevents.* (one per restart, unbounded)
- models/checkpoints*/checkpoint_step_*.pt + phase4_ckpt_v*.pt (periodic, unbounded)
- logs/*.log backups, __pycache__, .pytest_cache, .ruff_cache, tmp/, build/

Safe by design: every function takes explicit paths, never deletes
datasets (data/dataset*), never deletes protected checkpoints
(best_model.pt, latest_checkpoint.pt, live_weights.pt,
phase4_final_checkpoint.pt) unless asked.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

BUFFER_FILES = (
    "planes.bin",
    "visits.bin",
    "outcomes.bin",
    "full_search.bin",
    "versions.bin",
    "game_ids.bin",
    "meta.json",
    "meta.tmp",
)

# Checkpoints that are always kept (resume chain + live serving).
PROTECTED_CHECKPOINTS = frozenset(
    {
        "best_model.pt",
        "best_eval.pt",
        "best_eval.pt.tmp",
        "latest_checkpoint.pt",
        "live_weights.pt",
        "live_weights.pt.tmp",
        "phase4_final_checkpoint.pt",
        # Curriculum weights container (audit CKP-09 moved it out of the RL tree;
        # protected here so no future pattern ever matches it).
        "curriculum_final_latest.pt",
    }
)


def _rm_file(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)  # type: ignore[call-arg]
        return True
    except TypeError:
        # Python < 3.8 fallback (repo targets 3.10+, defensive only).
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                return True
        except OSError:
            logger.debug("cleanup: could not delete %s", path, exc_info=True)
        return False
    except OSError:
        # Locked (memmapped buffer of a live run) or permission error.
        logger.debug("cleanup: could not delete %s", path, exc_info=True)
        return False


def cleanup_buffer(buffer_dir: str | Path) -> dict[str, int]:
    """Close-safe delete of preallocated replay memmaps.

    Returns {"deleted": n, "failed": n, "freed_bytes": n}.
    Missing dir -> all zeros. Locked files count as failed, never raise.
    """
    buf = Path(buffer_dir)
    stats = {"deleted": 0, "failed": 0, "freed_bytes": 0}
    if not buf.is_dir():
        return stats
    for name in BUFFER_FILES:
        p = buf / name
        if p.is_file():
            try:
                stats["freed_bytes"] += p.stat().st_size
            except OSError:
                pass
            if _rm_file(p):
                stats["deleted"] += 1
            else:
                stats["failed"] += 1
    # Remove dir if fully empty (keeps log_dir tidy, recreates on next run).
    try:
        if not any(buf.iterdir()):
            buf.rmdir()
    except OSError:
        logger.debug("cleanup: could not rmdir %s", buf, exc_info=True)
    return stats


def prune_checkpoints(
    checkpoint_dir: str | Path,
    keep_last_n: int = 3,
    extra_keep: Sequence[str] = (),
) -> dict[str, int]:
    """Delete old periodic checkpoints, keep newest N + protected names.

    Only targets checkpoint_step_*.pt and phase4_ckpt_v*.pt patterns.
    Returns {"deleted": n, "kept": n, "freed_bytes": n}.
    """
    ckpt_dir = Path(checkpoint_dir)
    stats = {"deleted": 0, "kept": 0, "freed_bytes": 0}
    if not ckpt_dir.is_dir() or keep_last_n < 0:
        return stats
    keep_names = set(PROTECTED_CHECKPOINTS) | set(extra_keep)
    candidates: list[Path] = [
        p
        for p in ckpt_dir.glob("*.pt")
        if p.name not in keep_names
        and (p.name.startswith("checkpoint_step_") or p.name.startswith("phase4_ckpt_v"))
    ]
    # Newest first by mtime; keep the head, delete the tail.
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for p in candidates[keep_last_n:]:
        try:
            stats["freed_bytes"] += p.stat().st_size
        except OSError:
            pass
        if _rm_file(p):
            stats["deleted"] += 1
        # stale .tmp siblings and lock sentinels of deleted checkpoints
        _rm_file(p.with_name(p.name + ".tmp"))
        sweep_stale_lock_files(p)
    stats["kept"] = (
        len(candidates)
        - stats["deleted"]
        + sum(1 for p in ckpt_dir.glob("*.pt") if p.name in keep_names and p.is_file())
    )
    return stats


def prune_tfevents(log_dir: str | Path, keep_last_n: int = 1) -> dict[str, int]:
    """Keep newest N TensorBoard event files under log_dir, delete the rest."""
    log_path = Path(log_dir)
    stats = {"deleted": 0, "kept": 0, "freed_bytes": 0}
    if not log_path.is_dir() or keep_last_n < 0:
        return stats
    events = sorted(
        log_path.glob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for p in events[keep_last_n:]:
        try:
            stats["freed_bytes"] += p.stat().st_size
        except OSError:
            pass
        if _rm_file(p):
            stats["deleted"] += 1
    stats["kept"] = len(events) - stats["deleted"]
    return stats


def sweep_stale_lock_files(target: str | Path, *, min_age_sec: float = 600.0) -> int:
    """Remove *stale* lock sentinels for one artifact. Never raises.

    ``python.common.artifacts.artifact_lock`` creates ``<artifact>.lock`` (held by
    the OS for the duration of a write, auto-released if the process dies) plus a
    best-effort ``<artifact>.lock.owner`` diagnostic sidecar. Sentinel removal
    happens only when the lock is provably free of a live writer:

    * the ``.lock`` file is at least ``min_age_sec`` old, and
    * this process can acquire it non-blockingly right now.

    A held lock (and its sidecar) is therefore never touched. This is janitorial
    tidiness, not lock management: exclusion lives in the OS lock, so correctness
    never depends on the file's absence.
    """
    from python.common.artifacts import ArtifactLockError, artifact_lock

    base = Path(target)
    lock_path = base.with_name(base.name + ".lock")
    owner_path = base.with_name(base.name + ".lock.owner")
    lock_exists = lock_path.is_file()
    released = True
    if lock_exists:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = 0.0
        if age < min_age_sec:
            released = False
        else:
            try:
                with artifact_lock(base, timeout=0.0, owner="cleanup-sweep"):
                    pass
            except (ArtifactLockError, OSError):
                released = False
    if not released:
        return 0
    removed = 0
    for path in (lock_path, owner_path):
        try:
            if path.is_file() and _rm_file(path):
                removed += 1
        except OSError:
            logger.debug("could not remove lock sentinel %s", path, exc_info=True)
    return removed


def cleanup_run(
    log_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    *,
    delete_buffer: bool = False,
    keep_checkpoints: int = 3,
    keep_tfevents: int = 1,
) -> dict[str, dict[str, int]]:
    """One-call post-run janitor. Never raises on locked/missing files."""
    report: dict[str, dict[str, int]] = {}
    report["tfevents"] = prune_tfevents(log_dir, keep_last_n=keep_tfevents)
    if checkpoint_dir is not None:
        report["checkpoints"] = prune_checkpoints(checkpoint_dir, keep_last_n=keep_checkpoints)
    if delete_buffer:
        report["buffer"] = cleanup_buffer(Path(log_dir) / "buffer")
    # Lock sentinels of the run's own artifacts: remove only the ones provably
    # released (acquirable) and old, so an active writer's lock is never touched.
    sweeps = [Path(log_dir)]
    if checkpoint_dir is not None:
        sweeps.append(Path(checkpoint_dir))
    for sweep_dir in sweeps:
        if not sweep_dir.is_dir():
            continue
        seen_artifacts: set[Path] = set()
        for lock_file in list(sweep_dir.glob("*.lock")) + list(sweep_dir.glob("*.lock.owner")):
            name = lock_file.name
            for suffix in (".lock.owner", ".lock"):
                if name.endswith(suffix):
                    seen_artifacts.add(lock_file.with_name(name[: -len(suffix)]))
                    break
        for artifact in seen_artifacts:
            try:
                sweep_stale_lock_files(artifact)
            except OSError:
                logger.debug("lock sweep failed for %s", artifact, exc_info=True)
    return report
