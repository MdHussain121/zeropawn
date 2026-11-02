"""Resolve load-bearing data files: repository checkout first, package data fallback.

S6-01 (post-merge audit 2026-11-02): the built wheel previously shipped only
``python*`` with none of the data files the CLI needs (checkpoint registry,
presets), so every installed context died with ``FileNotFoundError``. The
resolution policy is:

1. **Repository-checkout path first** — byte-identical behavior for
   developers running from a checkout (the file the repo actually holds).
2. **Packaged copy fallback** — files staged into ``python/_data/`` at wheel
   build time (see ``scripts/build_wheel.py``) are resolved through
   ``importlib.resources`` so an installed wheel finds them in site-packages.

Writes never follow the fallback: packaged resources live inside
site-packages and are read-only by convention; write paths keep using the
repository-anchored path and fail naturally (with an OSError) when it does
not exist.

This module lives in ``python.common`` (not ``python.cli``) because the
checkpoint registry (``python.model``) is a lower layer that must not depend
on the CLI package.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

#: Package holding the staged copies (``python/_data/`` at build time).
DATA_PACKAGE = "python"

__all__ = ["resolve_data_file", "packaged_data_file"]


def packaged_data_file(*resource_parts: Union[str, Path]) -> Path:
    """Path of a staged copy inside the installed ``python`` package.

    Returns a real filesystem path (wheels are always installed extracted, so
    the traversable is a plain path). When the resource or the package is
    absent — repo checkouts without staging, exotic zip installs — the
    returned path does not exist and callers fall through to their original
    error handling.
    """
    try:
        traversable = resources.files(DATA_PACKAGE)
        for part in resource_parts:
            traversable = traversable / str(part)
        return Path(traversable)
    except (ModuleNotFoundError, TypeError, FileNotFoundError):
        # Package not importable under that name (or non-path traversable):
        # report a non-existent path so callers keep their normal failure mode.
        return Path("__missing_packaged_resource__") / Path(*map(str, resource_parts))


def resolve_data_file(repo_path: Union[str, Path], *resource_parts: str) -> Path:
    """Return ``repo_path`` when it exists, else the packaged copy.

    The repository path wins whenever present (dev behavior unchanged). The
    packaged copy (``python/_data/<parts>``) is used only when the repository
    path is absent — i.e. an installed wheel outside a checkout. If neither
    exists, the repository path is returned so callers raise their original,
    operator-facing error naming the canonical location.
    """
    repo = Path(repo_path)
    if repo.is_file():
        return repo
    packaged = packaged_data_file(*resource_parts)
    if packaged.is_file():
        logger.debug("Repo file %s absent; using packaged resource %s", repo, packaged)
        return packaged
    return repo
