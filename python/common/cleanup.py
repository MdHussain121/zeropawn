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
import shutil
from pathlib import Path
from typing import Iterable, Sequence

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
        # stale .tmp siblings of deleted checkpoints
        _rm_file(p.with_name(p.name + ".tmp"))
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


def cleanup_caches(repo_root: str | Path) -> dict[str, int]:
    """Delete regenerable caches: __pycache__, .pytest_cache, .ruff_cache, tmp/ files."""
    root = Path(repo_root)
    stats = {"deleted": 0, "failed": 0}
    for cache_name in (".pytest_cache", ".ruff_cache"):
        p = root / cache_name
        if p.exists():
            try:
                shutil.rmtree(p, ignore_errors=True)
                stats["deleted"] += 1
            except OSError:
                stats["failed"] += 1
                logger.debug("cleanup: could not remove %s", p, exc_info=True)
    for pycache in root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache, ignore_errors=True)
            stats["deleted"] += 1
        except OSError:
            stats["failed"] += 1
    tmp_dir = root / "tmp"
    if tmp_dir.is_dir():
        for p in tmp_dir.iterdir():
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
                stats["deleted"] += 1
            except OSError:
                stats["failed"] += 1
    return stats


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
    return report


def iter_junk_report(repo_root: str | Path) -> Iterable[tuple[str, int, int]]:
    """Yield (label, size_bytes, file_count) for known junk locations."""
    root = Path(repo_root)
    for label, sub in (
        ("replay buffers", "runs"),
        ("tensorboard events", "runs"),
        ("logs", "logs"),
        ("tmp", "tmp"),
        ("build", "build"),
    ):
        base = root / sub
        if not base.exists():
            continue
        if label == "replay buffers":
            files = list(base.rglob("buffer/*.bin"))
        elif label == "tensorboard events":
            files = list(base.rglob("events.out.tfevents.*"))
        else:
            files = [p for p in base.rglob("*") if p.is_file()]
        size = 0
        for p in files:
            try:
                size += p.stat().st_size
            except OSError:
                continue
        yield label, size, len(files)
