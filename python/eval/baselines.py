"""Model puzzle baseline registry and persistent store."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_BASELINE: Optional[float] = None

# Cache for file sha256: (resolved_path_str, mtime_ns, size) -> sha256_hex
_SHA256_CACHE: Dict[Tuple[str, int, int], str] = {}


def get_baselines_file_path() -> Path:
    """Return path to puzzle_baselines.json in repo root models directory."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "models" / "puzzle_baselines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file with mtime/size in-memory caching.

    Reads in 1MB chunks to handle ~100MB model checkpoints efficiently without
    excessive memory consumption.
    """
    try:
        resolved = str(path.resolve())
        stat = path.stat()
        cache_key = (resolved, stat.st_mtime_ns, stat.st_size)
        cached = _SHA256_CACHE.get(cache_key)
        if cached:
            return cached

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        _SHA256_CACHE[cache_key] = digest
        return digest
    except Exception as e:
        logger.debug("Failed to compute SHA-256 for %s: %s", path, e)
        return ""


def load_baselines() -> Dict[str, Any]:
    """Load all model baselines from disk."""
    path = get_baselines_file_path()
    if not path.is_file():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug("Failed to read baselines from %s: %s", path, e)
    return {}


def get_baseline_for_model(
    model_name_or_path: Union[str, Path, None],
    default: Optional[float] = DEFAULT_BASELINE,
) -> Optional[float]:
    """Get recorded baseline solve rate (%) for a specific model checkpoint.

    Lookup precedence:
    1. SHA-256 hash match (full 64-char or 8-char short hash) if file exists on disk.
    2. Exact filename match (e.g. 'best_model.pt'). If the local file exists, verifies
       that the recorded 'sha256' matches the actual file hash. If the record lacks a
       hash or has a mismatched hash, the baseline is rejected and default is returned.
    3. Stem match with .pt or without .pt (for mock/unhashed names when file does not exist).
    4. Falls back to default (None by default).
    """
    if model_name_or_path is None:
        return default

    p = Path(model_name_or_path)
    baselines = load_baselines()
    if not baselines:
        return default

    file_exists = p.is_file()
    actual_sha = _file_sha256(p) if file_exists else ""

    # 1. Match by SHA-256 if file exists
    if file_exists and actual_sha:
        # Full hash lookup
        if actual_sha in baselines:
            val = baselines[actual_sha]
            if isinstance(val, dict):
                rate_val = val.get("policy_rate")
                if rate_val is not None:
                    try:
                        return float(rate_val)
                    except (ValueError, TypeError):
                        pass
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        # Short hash lookup (first 8 chars)
        short = actual_sha[:8]
        if short in baselines:
            val = baselines[short]
            if isinstance(val, dict):
                rate_val = val.get("policy_rate")
                if rate_val is not None:
                    try:
                        return float(rate_val)
                    except (ValueError, TypeError):
                        pass
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    name = p.name
    stem = p.stem

    def _extract_entry(entry: Any, key_name: str) -> tuple[bool, Optional[float]]:
        """Extract baseline score from entry with hash verification for physical files.

        Returns (matched, value). If matched is True, the key was found for this model;
        value is either the verified score or default if integrity check rejected it.
        """
        if file_exists:
            if isinstance(entry, dict):
                rec_sha = entry.get("sha256")
                if not rec_sha or not isinstance(rec_sha, str) or not rec_sha.strip():
                    logger.warning(
                        "Baseline for '%s' ignored: legacy record has no recorded SHA-256 hash. "
                        "Re-benchmark with --update-baseline to bind a verified baseline.",
                        name,
                    )
                    return True, default
                if not actual_sha or not actual_sha.startswith(rec_sha.strip()):
                    logger.warning(
                        "Baseline for '%s' ignored: file hash %s does not match recorded %s",
                        name,
                        (actual_sha[:8] if actual_sha else "unknown"),
                        rec_sha[:8],
                    )
                    return True, default
                rate_val = entry.get("policy_rate")
                if rate_val is not None:
                    try:
                        return True, float(rate_val)
                    except (ValueError, TypeError):
                        pass
                return True, default
            else:
                logger.warning(
                    "Baseline for '%s' ignored: legacy record has no recorded SHA-256 hash. "
                    "Re-benchmark with --update-baseline to bind a verified baseline.",
                    name,
                )
                return True, default
        else:
            if isinstance(entry, dict):
                rate_val = entry.get("policy_rate")
                if rate_val is not None:
                    try:
                        return True, float(rate_val)
                    except (ValueError, TypeError):
                        pass
                return True, default
            try:
                return True, float(entry)
            except (ValueError, TypeError):
                return False, default

    # 2. Match on filename
    if name in baselines:
        matched, val = _extract_entry(baselines[name], name)
        if matched:
            return val

    # 3. Match with .pt if query had no extension
    if f"{stem}.pt" in baselines:
        matched, val = _extract_entry(baselines[f"{stem}.pt"], f"{stem}.pt")
        if matched:
            return val

    # 4. Match stem (without .pt)
    if stem in baselines:
        matched, val = _extract_entry(baselines[stem], stem)
        if matched:
            return val

    # 5. Scan keys for stem match
    for k, v in baselines.items():
        if k != "default" and Path(k).stem == stem:
            matched, val = _extract_entry(v, k)
            if matched:
                return val

    return default


def set_baseline_for_model(
    model_name_or_path: Union[str, Path],
    rate: float,
    mcts_rate: Optional[float] = None,
    total_puzzles: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """Update and persist baseline solve rate (%) for a specific model bound by SHA-256.

    Writes atomically using a temporary file in the target directory and atomic replace.
    """
    path = get_baselines_file_path()
    baselines = load_baselines()
    p = Path(model_name_or_path)
    name = p.name
    sha = _file_sha256(p) if p.is_file() else ""

    existing = baselines.get(name, {}) if isinstance(baselines.get(name), dict) else {}
    if not existing and sha and sha in baselines and isinstance(baselines[sha], dict):
        existing = baselines[sha]

    record = {
        **existing,
        "policy_rate": round(float(rate), 2),
        "mcts_rate": round(float(mcts_rate), 2)
        if mcts_rate is not None
        else existing.get("mcts_rate"),
        "total_puzzles": total_puzzles
        if total_puzzles is not None
        else existing.get("total_puzzles"),
        "sha256": sha if sha else existing.get("sha256", ""),
    }
    if dataset_name is not None:
        record["dataset"] = dataset_name

    baselines[name] = record
    if sha:
        baselines[sha] = record

    # Atomic write to disk
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix="puzzle_baselines_",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as f:
            tmp_file = Path(f.name)
            json.dump(baselines, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, path)
    except Exception as e:
        logger.warning("Failed to save puzzle baselines to %s: %s", path, e)
        if tmp_file and tmp_file.is_file():
            try:
                tmp_file.unlink()
            except OSError:
                pass
    return path
