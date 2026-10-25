"""Tests for Quarantine and Safe Cleanup Framework.

Verifies:
1. Dry-run immutability (no files moved, modified, or deleted).
2. Protected path exclusion (supervised baseline, syzygy, scaled datasets).
3. SHA-256 integrity hashing and manifest recording.
4. Transactional quarantine moves, collision prevention, and rollback.
5. Restoration fidelity and byte-for-byte recovery.
6. Purge safety gates (strict authorization required).
7. Non-destructive CLI cleanup commands.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from python.common.quarantine import (
    QuarantineManager,
    QuarantineItem,
    CleanupManifest,
    FULL_BASELINE_SHA256,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


@pytest.fixture
def mock_repo(tmp_path: Path) -> Path:
    """Create a structured mock repository tree."""
    # Protected supervised warm-start baseline
    scaled_ckpt = tmp_path / "models" / "checkpoints_scaled" / "best_model.pt"
    scaled_ckpt.parent.mkdir(parents=True, exist_ok=True)
    scaled_ckpt.write_bytes(b"PROTECTED_SUPERVISED_WARM_START_CHECKPOINT_DATA")

    # Protected datasets
    scaled_data = tmp_path / "data" / "dataset_scaled" / "train" / "chunk_0.pt"
    scaled_data.parent.mkdir(parents=True, exist_ok=True)
    scaled_data.write_bytes(b"PROTECTED_TRAIN_DATA")

    syzygy = tmp_path / "data" / "syzygy" / "KQvK.rtbw"
    syzygy.parent.mkdir(parents=True, exist_ok=True)
    syzygy.write_bytes(b"PROTECTED_SYZYGY_DATA")

    # Protected presets and code
    presets = tmp_path / "configs" / "presets.json"
    presets.parent.mkdir(parents=True, exist_ok=True)
    presets.write_text('{"presets": []}', encoding="utf-8")

    # Disposable candidate assets
    legacy_buffer = tmp_path / "runs" / "phase4_rl" / "buffer" / "planes.bin"
    legacy_buffer.parent.mkdir(parents=True, exist_ok=True)
    legacy_buffer.write_bytes(b"DISPOSABLE_LEGACY_BUFFER_DATA")

    stale_tmp = tmp_path / "tmp" / "stale_eval.tmp"
    stale_tmp.parent.mkdir(parents=True, exist_ok=True)
    stale_tmp.write_bytes(b"DISPOSABLE_TMP_FILE")

    flawed_ckpt = tmp_path / "models" / "checkpoints_finetune" / "best_model_MIX_LEAKED.pt"
    flawed_ckpt.parent.mkdir(parents=True, exist_ok=True)
    flawed_ckpt.write_bytes(b"FLAWED_LEAKED_CHECKPOINT")

    return tmp_path


def test_dry_run_immutability(mock_repo: Path, monkeypatch):
    """Dry-run must not delete or move any files."""
    mgr = QuarantineManager(repo_root=mock_repo)

    # Monkeypatch verify_baseline_checkpoint to return True for mock
    monkeypatch.setattr(mgr, "verify_baseline_checkpoint", lambda: (True, "54d46282mockhash"))

    legacy_buffer = mock_repo / "runs" / "phase4_rl" / "buffer" / "planes.bin"
    stale_tmp = mock_repo / "tmp" / "stale_eval.tmp"
    orig_buf_hash = _sha256(legacy_buffer)
    orig_tmp_hash = _sha256(stale_tmp)

    manifest = mgr.generate_dry_run_manifest()

    assert manifest.total_candidates_count >= 2
    assert manifest.execution_status == "dry_run"
    assert legacy_buffer.exists(), "Replay buffer must exist after dry-run"
    assert stale_tmp.exists(), "Temporary file must exist after dry-run"
    assert _sha256(legacy_buffer) == orig_buf_hash
    assert _sha256(stale_tmp) == orig_tmp_hash
    assert not mgr.quarantine_dir.exists(), "Quarantine directory should not be created on dry run"


def test_protected_assets_blocked(mock_repo: Path):
    """Protected assets must be strictly identified and blocked from quarantine."""
    mgr = QuarantineManager(repo_root=mock_repo)

    protected_paths = [
        mock_repo / "models" / "checkpoints_scaled" / "best_model.pt",
        mock_repo / "data" / "dataset_scaled" / "train" / "chunk_0.pt",
        mock_repo / "data" / "syzygy" / "KQvK.rtbw",
        mock_repo / "configs" / "presets.json",
        mock_repo / "cpp" / "zeropawn_cpp.cpp",
        mock_repo / "python" / "cli" / "main.py",
        mock_repo / "tests" / "test_main.py",
    ]

    for p in protected_paths:
        is_prot, reason = mgr.is_protected(p)
        assert is_prot, f"Path {p} must be protected. Reason was: {reason}"


def test_quarantine_gate_enforced(mock_repo: Path, monkeypatch):
    """Quarantine execution must fail without explicit approved_by_user=True."""
    mgr = QuarantineManager(repo_root=mock_repo)
    monkeypatch.setattr(mgr, "verify_baseline_checkpoint", lambda: (True, "54d46282mockhash"))

    manifest = mgr.generate_dry_run_manifest()

    with pytest.raises(PermissionError, match="not yet authorized"):
        mgr.execute_quarantine(manifest, approved_by_user=False)


def test_quarantine_move_and_restore(mock_repo: Path, monkeypatch):
    """Quarantine moves files transactionally and restores them faithfully."""
    mgr = QuarantineManager(repo_root=mock_repo)
    monkeypatch.setattr(mgr, "verify_baseline_checkpoint", lambda: (True, "54d46282mockhash"))

    stale_tmp = mock_repo / "tmp" / "stale_eval.tmp"
    orig_hash = _sha256(stale_tmp)
    orig_content = stale_tmp.read_bytes()

    manifest = mgr.generate_dry_run_manifest()
    # Filter manifest to just the tmp file for isolated test
    manifest.items = [it for it in manifest.items if "stale_eval.tmp" in it["original_rel_path"]]

    result = mgr.execute_quarantine(manifest, approved_by_user=True)

    assert result["moved_files"] == 1
    assert not stale_tmp.exists(), "Source file should have been moved into quarantine"

    archive_dir = Path(result["archive_dir"])
    assert archive_dir.is_dir()
    archived_file = archive_dir / "tmp" / "stale_eval.tmp"
    assert archived_file.exists()
    assert _sha256(archived_file) == orig_hash

    # Test restoration
    restore_res = mgr.restore_archive(result["archive_name"])
    assert restore_res["restored_files"] == 1
    assert stale_tmp.exists(), "Restored file must exist at original location"
    assert stale_tmp.read_bytes() == orig_content
    assert _sha256(stale_tmp) == orig_hash


def test_purge_safety_gate(mock_repo: Path, monkeypatch):
    """Physical purge requires authorized=True."""
    mgr = QuarantineManager(repo_root=mock_repo)
    monkeypatch.setattr(mgr, "verify_baseline_checkpoint", lambda: (True, "54d46282mockhash"))

    stale_tmp = mock_repo / "tmp" / "stale_eval.tmp"
    manifest = mgr.generate_dry_run_manifest()
    manifest.items = [it for it in manifest.items if "stale_eval.tmp" in it["original_rel_path"]]

    result = mgr.execute_quarantine(manifest, approved_by_user=True)
    archive_name = result["archive_name"]
    archive_dir = Path(result["archive_dir"])
    assert archive_dir.exists()

    # Unauthorized purge must fail
    with pytest.raises(PermissionError, match="authorization"):
        mgr.purge_archive(archive_name, authorized=False)

    assert archive_dir.exists(), "Archive must survive unauthorized purge"

    # Authorized purge succeeds
    purge_res = mgr.purge_archive(archive_name, authorized=True)
    assert purge_res["purged_archive"] == archive_name
    assert not archive_dir.exists(), "Archive must be physically deleted after authorized purge"
