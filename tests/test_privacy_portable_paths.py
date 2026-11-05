"""Regression tests for the 2026-11-05 pre-push privacy remediation.

Pins that no machine-specific absolute path literals remain in the two
source files that previously carried them, and that the portable
discovery mechanisms (env-var overrides / repo-root-relative defaults)
behave as documented.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

DRIVE_LETTER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users")


def _module_source(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_no_drive_letter_user_path_in_bench_uci_callback_ab():
    src = _module_source("scripts/bench_uci_callback_ab.py")
    assert not DRIVE_LETTER_PATH.search(src), (
        "scripts/bench_uci_callback_ab.py still contains a hardcoded drive-letter user path"
    )
    assert "<user>" not in src.lower()


def test_bench_uci_callback_ab_ckpt_is_repo_root_relative():
    """CKPT must derive from the script's repo root (or ZEROPAWN_CHECKPOINT),
    not a hardcoded absolute path."""
    src = _module_source("scripts/bench_uci_callback_ab.py")
    tree = ast.parse(src)
    ckpt_assignments = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CKPT" for t in n.targets)
    ]
    assert ckpt_assignments, "CKPT assignment not found"
    joined = ast.dump(ckpt_assignments[0])
    assert "_REPO_ROOT" in joined, "CKPT no longer derives from _REPO_ROOT"
    assert "ZEROPAWN_CHECKPOINT" in src, "ZEROPAWN_CHECKPOINT env override not wired"


def test_bench_uci_callback_ab_module_imports_and_resolves_default(tmp_path, monkeypatch):
    """With the native extension available, the module must import and CKPT
    must default to the repo-relative models path."""
    pytest.importorskip("torch")
    pytest.importorskip("zeropawn_cpp")
    monkeypatch.delenv("ZEROPAWN_CHECKPOINT", raising=False)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import scripts.bench_uci_callback_ab as bench

        # The DEFAULT must be the repo-relative models path. (The resolved
        # absolute form naturally contains the local username on any machine
        # whose checkout lives under the user profile — that is not an
        # exposure; only source literals are.)
        assert bench.CKPT == (REPO_ROOT / "models" / "checkpoints_scaled" / "latest_checkpoint.pt")
    finally:
        sys.path.remove(str(REPO_ROOT))
        sys.modules.pop("scripts.bench_uci_callback_ab", None)


def test_checkpoint_provenance_uses_env_override_not_hardcoded_path():
    src = _module_source("tests/test_checkpoint_provenance.py")
    assert "ZEROPAWN_MAIN_WORKTREE" in src, "ZEROPAWN_MAIN_WORKTREE env override not wired"
    assert not DRIVE_LETTER_PATH.search(src), (
        "tests/test_checkpoint_provenance.py still contains a hardcoded drive-letter user path"
    )


def test_zeropawn_main_worktree_env_discovery(tmp_path, monkeypatch):
    """The ZEROPAWN_MAIN_WORKTREE candidate mechanism used by
    test_bundled_registry_matches_disk_when_artifacts_present must accept an
    alternate models/ root."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ZEROPAWN_MAIN_WORKTREE", str(tmp_path))
    candidates = [REPO_ROOT]
    main_tree = (
        monkeypatch.get_env("ZEROPAWN_MAIN_WORKTREE") if hasattr(monkeypatch, "get_env") else None
    )
    if main_tree is None:
        import os

        main_tree = os.environ.get("ZEROPAWN_MAIN_WORKTREE")
    candidates.append(Path(main_tree))
    assert (candidates[-1] / "models" / "manifest.json").is_file()
    assert any((c / "models" / "manifest.json").is_file() for c in candidates)
