"""Hermetic artifact path policy, production-write protection, and cross-process locks.

Established by the 2026-10-31 checkpoint & artifact lifecycle remediation
(findings CKP-04, SUS-02). Three rules are enforced here:

1. **No CWD-relative resolution.** A relative artifact path is resolved against
   the repository root discovered from this file's location (``python.cli.common
   .REPO_ROOT``), never against :func:`os.getcwd`. A process whose working
   directory happens to be the repository root therefore cannot redirect
   production writes into the production tree; the path it names is the path it
   gets.

2. **Protected destinations need explicit production intent.** Writing a
   protected artifact name (``live_weights.pt``, ``latest_checkpoint.pt``,
   ``best_model.pt``, ``best_eval.pt``, ``phase4_final_checkpoint.pt``, the
   governance JSONs) inside the production artifact root requires a
   :class:`ProductionWriteTicket`. Tickets are minted only by naming an explicit
   destination (``issue_for_explicit_path``) or by a production entry point
   (``issue_for_launcher``) — never by a boolean switch alone, and never by an
   environment variable. Test code that does not deliberately mint one is
   refused with a diagnostic naming the guard.

3. **Architecture may not silently change under a name.** Replacing an existing
   checkpoint whose recorded architecture differs from the payload's
   architecture is refused. This is exactly the 2026-10-30 signature: a 2x32
   test-scale model overwrote the 12x128 production ``live_weights.pt``.

Cross-process serialization (:func:`artifact_lock`) uses OS-level file locks so
locks die with their process — no stale-lock recovery heuristic is needed. Lock
ownership is reported on contention via a best-effort ``.lock.owner`` sidecar.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

logger = logging.getLogger(__name__)

# Artifact names that are never silently replaced (mirrors
# python.common.cleanup.PROTECTED_CHECKPOINTS plus the governance records).
PROTECTED_ARTIFACT_NAMES = frozenset(
    {
        "best_model.pt",
        "best_eval.pt",
        "best_eval.pt.tmp",
        "best_eval.json",
        "latest_checkpoint.pt",
        "live_weights.pt",
        "live_weights.pt.tmp",
        "phase4_final_checkpoint.pt",
        "checkpoint_registry.json",
        "manifest.json",
        "run_manifest.json",
    }
)

# Directory (repository-relative) that holds production model artifacts.
PRODUCTION_ARTIFACT_DIRNAME = "models"


class ArtifactPathError(RuntimeError):
    """Raised for invalid or unsafe artifact destinations."""


class ProtectedArtifactWriteError(ArtifactPathError):
    """Raised when a write to a protected production artifact is not authorized."""


class ArtifactLockError(ArtifactPathError):
    """Raised when another process holds the artifact lock."""


@dataclass(frozen=True)
class ProductionWriteTicket:
    """Explicit, auditable authorization to write production artifact destinations.

    A ticket records *why* the caller believes it may touch production paths and
    *who* issued that decision. It is minted at two places only:

    * :meth:`issue_for_explicit_path` — the operator named the destination
      (``--checkpoint-dir models/checkpoints_rl`` or an absolute path), i.e. the
      write is an explicit act, not a default;
    * :meth:`issue_for_launcher` — a production launcher (``run_phase4_rl``,
      ``train_supervised`` CLI) declares the run's own output tree.

    Tests, diagnostics, and ad-hoc scripts hold none and are refused.
    """

    reason: str
    issued_by: str
    issued_at: float = field(default_factory=time.time)

    @classmethod
    def issue_for_explicit_path(cls, destination: Union[str, Path]) -> "ProductionWriteTicket":
        return cls(reason=f"explicit destination {destination}", issued_by="explicit-path")

    @classmethod
    def issue_for_launcher(cls, launcher: str, reason: str) -> "ProductionWriteTicket":
        return cls(reason=reason, issued_by=f"launcher:{launcher}")

    def describe(self) -> str:
        return f"ticket({self.issued_by}: {self.reason})"


def repo_root() -> Path:
    """Repository root discovered from this file's location (never the CWD)."""
    from python.cli.common import REPO_ROOT

    return Path(REPO_ROOT)


def production_artifact_root() -> Path:
    """The ``models/`` tree of the repository this code lives in."""
    return repo_root() / PRODUCTION_ARTIFACT_DIRNAME


def resolve_artifact_path(
    path: Union[str, Path],
    *,
    base: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve an artifact path without ever consulting the process CWD.

    Absolute paths are returned normalized. Relative paths are interpreted
    against ``base`` when given, else against the repository root: an artifact
    path in ZeroPawn is repository-relative by contract, and a changed working
    directory must not change what it points at.
    """
    p = Path(path)
    if p.is_absolute():
        return Path(os.path.normpath(str(p)))
    root = Path(base) if base is not None else repo_root()
    return Path(os.path.normpath(str(root / p)))


def is_production_artifact(path: Union[str, Path]) -> bool:
    """True when ``path`` lies inside the production artifact root."""
    try:
        resolved = Path(path).resolve()
        root = production_artifact_root().resolve()
    except OSError:
        return False
    return root == resolved or root in resolved.parents


def is_protected_artifact_name(path: Union[str, Path]) -> bool:
    return Path(path).name in PROTECTED_ARTIFACT_NAMES


def _describe_existing_artifact(path: Path) -> str:
    """Best-effort identity of the file about to be replaced (for diagnostics)."""
    try:
        stat = path.stat()
    except OSError:
        return "unreadable"
    return f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))} size={stat.st_size}"


def read_checkpoint_arch(path: Union[str, Path]) -> Optional[tuple[int, int]]:
    """Cheaply read ``(num_blocks, num_filters)`` from an existing checkpoint.

    Returns ``None`` when the header is absent or the file cannot be read as a
    weights-only torch archive — an unknown header never blocks a write by
    itself; it only means the architecture guard cannot fire.
    """
    import torch

    p = Path(path)
    if not p.is_file():
        return None
    try:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # older torch without mmap
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, KeyError, TypeError):
        return None
    if not isinstance(ckpt, dict):
        return None
    blocks = ckpt.get("num_blocks")
    filters = ckpt.get("num_filters")
    if blocks is None or filters is None:
        return None
    try:
        return (int(blocks), int(filters))
    except (TypeError, ValueError):
        return None


def assert_safe_artifact_destination(
    path: Union[str, Path],
    *,
    ticket: Optional[ProductionWriteTicket] = None,
    arch: Optional[tuple[int, int]] = None,
    allow_arch_change: bool = False,
    what: str = "artifact",
) -> Path:
    """Validate a destination before a create/replace and return the resolved path.

    Enforces the three rules in the module docstring. Raises
    :class:`ProtectedArtifactWriteError` (protected destination without a ticket)
    or :class:`ArtifactPathError` (architecture change under an existing name).
    """
    resolved = resolve_artifact_path(path)

    if is_production_artifact(resolved) and is_protected_artifact_name(resolved):
        if ticket is None:
            existing = _describe_existing_artifact(resolved)
            raise ProtectedArtifactWriteError(
                f"Refusing to write protected production {what} '{resolved}' without explicit "
                f"production authorization. This destination belongs to the production "
                f"artifact tree ({production_artifact_root()}); test/diagnostic writers must "
                f"use a temporary directory. Existing file: {existing}. An operator who "
                f"intends this write should pass an explicit output path or run the "
                f"production launcher, which mints a ProductionWriteTicket."
            )
        logger.info(
            "Authorized protected production write to %s (%s)", resolved, ticket.describe()
        )

    if arch is not None and not allow_arch_change:
        existing_arch = read_checkpoint_arch(resolved)
        if existing_arch is not None and existing_arch != arch:
            raise ArtifactPathError(
                f"Refusing to replace '{resolved}': existing checkpoint architecture "
                f"{existing_arch[0]}x{existing_arch[1]} differs from the payload "
                f"architecture {arch[0]}x{arch[1]}. This is the 2026-10-30 "
                f"production-contamination signature (a 2x32 test model overwrote "
                f"12x128 live_weights.pt). Write to a distinct directory, or pass "
                f"allow_arch_change=True only for a deliberate architecture migration."
            )
    return resolved


def atomic_write_text(
    path: Union[str, Path],
    text: str,
    *,
    ticket: Optional[ProductionWriteTicket] = None,
    encoding: str = "utf-8",
) -> Path:
    """Write text atomically (tmp + fsync + os.replace + dir fsync)."""
    resolved = assert_safe_artifact_destination(path, ticket=ticket, what="record")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_name(resolved.name + ".tmp")
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            logger.debug("fsync failed for %s", tmp)
    os.replace(tmp, resolved)
    _fsync_dir(resolved.parent)
    return resolved


def atomic_write_json(
    path: Union[str, Path],
    payload: Any,
    *,
    ticket: Optional[ProductionWriteTicket] = None,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=False) + "\n",
        ticket=ticket,
    )


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        logger.debug("could not open dir for fsync: %s", directory)
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("fsync failed for dir %s", directory)
    finally:
        os.close(fd)


# ── Cross-process locking ────────────────────────────────────────────────────


def _lock_file_handle(fh: Any) -> bool:
    """Try to take an exclusive OS lock on an open file handle (non-blocking)."""
    if sys.platform == "win32":
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:  # pragma: no cover - exercised on POSIX only
        import fcntl

        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def _unlock_file_handle(fh: Any) -> None:
    if sys.platform == "win32":
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            logger.debug("lock release failed", exc_info=True)
    else:  # pragma: no cover - exercised on POSIX only
        import fcntl

        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            logger.debug("lock release failed", exc_info=True)


def _owner_path(lock_path: Path) -> Path:
    return lock_path.with_name(lock_path.name + ".owner")


def describe_lock_owner(target: Union[str, Path]) -> str:
    """Best-effort description of the current lock holder (for diagnostics)."""
    owner_file = _owner_path(Path(target).with_name(Path(target).name + ".lock"))
    try:
        payload = json.loads(owner_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "owner unknown (owner sidecar unreadable); the OS lock is held by another process"
    age = time.time() - float(payload.get("acquired_at", 0.0) or 0.0)
    return (
        f"pid={payload.get('pid')} owner={payload.get('owner')!r} "
        f"acquired {age:.1f}s ago on host {payload.get('host')!r}"
    )


@contextmanager
def artifact_lock(
    target: Union[str, Path],
    *,
    timeout: float = 0.0,
    owner: str = "",
    poll_interval: float = 0.05,
) -> Iterator[Path]:
    """Hold an exclusive cross-process lock while writing ``target``.

    The lock lives on ``<target>.lock`` and is released by the OS even if the
    process dies mid-write, so no stale-lock heuristic is required. With
    ``timeout=0`` (default) contention raises :class:`ArtifactLockError`
    immediately — used for checkpoint writes, where two writers on one path is a
    configuration error that must be loud, not a wait. Longer timeouts suit
    append-style telemetry where brief contention is benign.
    """
    resolved = Path(target)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.with_name(resolved.name + ".lock")
    owner_path = _owner_path(lock_path)
    deadline = time.monotonic() + max(0.0, float(timeout))

    fh = open(lock_path, "a+b")
    acquired = False
    try:
        while True:
            if _lock_file_handle(fh):
                acquired = True
                break
            if time.monotonic() >= deadline:
                raise ArtifactLockError(
                    f"Artifact lock for '{resolved}' is held by another process "
                    f"({describe_lock_owner(resolved)}). Refusing to write concurrently: "
                    f"two writers on one artifact path silently destroy each other's work."
                )
            time.sleep(poll_interval)
        try:
            owner_path.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "owner": owner or "unknown",
                        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "",
                        "acquired_at": time.time(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("could not write lock owner sidecar %s", owner_path)
        yield resolved
    finally:
        if acquired:
            try:
                owner_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("could not remove lock owner sidecar %s", owner_path)
            _unlock_file_handle(fh)
        fh.close()
