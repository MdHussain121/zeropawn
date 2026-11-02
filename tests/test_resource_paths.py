"""Offline tests for the S6-01 repo-first/package-data resource resolution."""

from __future__ import annotations

from pathlib import Path

from python.common.resource_paths import (
    packaged_data_file,
    resolve_data_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_path_wins_when_it_exists(tmp_path: Path):
    existing = tmp_path / "configs" / "presets.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    assert resolve_data_file(existing, "_data", "presets.json") == existing


def test_packaged_copy_used_when_repo_path_absent():
    """A nonexistent repo path falls back to the staged package copy."""
    resolved = resolve_data_file(
        "Z:/nonexistent-repo/configs/presets.json", "_data", "presets.json"
    )
    assert resolved.is_file(), f"expected staged package copy, got {resolved}"
    assert "_data" in str(resolved) and "presets.json" in str(resolved)


def test_absent_everywhere_returns_repo_path():
    """Neither source exists -> repo path returned so callers raise their normal error."""
    missing = "Z:/nonexistent-repo/configs/config.json"
    assert resolve_data_file(missing, "_data", "config.json") == Path(missing)


def test_staged_registry_matches_tracked_source():
    """Drift pin: the staged package copy must equal docs/checkpoint_registry.json."""
    staged = packaged_data_file("_data", "checkpoint_registry.json")
    tracked = REPO_ROOT / "docs" / "checkpoint_registry.json"
    if not staged.is_file():
        # A checkout without staging: the tracked file must still exist.
        assert tracked.is_file()
        return
    assert staged.read_bytes() == tracked.read_bytes(), (
        "python/_data/checkpoint_registry.json drifted from "
        "docs/checkpoint_registry.json — re-run scripts/build_wheel.py (staging step)"
    )


def test_staged_presets_match_tracked_source():
    """Drift pin: the staged package copy must equal configs/presets.json."""
    staged = packaged_data_file("_data", "presets.json")
    tracked = REPO_ROOT / "configs" / "presets.json"
    if not staged.is_file():
        assert tracked.is_file()
        return
    assert staged.read_bytes() == tracked.read_bytes(), (
        "python/_data/presets.json drifted from configs/presets.json — "
        "re-run scripts/build_wheel.py (staging step)"
    )
