"""Fail-closed checkpoint provenance tests (Stage A §5, audit R-07/SL-04/E-4).

Covers python.model.checkpoint_registry.validate_warm_start and the
InferenceServer construction path: a missing, unreadable, architecture-
mismatched, unregistered, quarantined, or undesignated warm start must stop
the operation before any model exists — never a silent random-init fallback.
"""

import json
from pathlib import Path

import pytest
import torch

from python.model.chess_net import ChessNet
from python.model.checkpoint_registry import (
    DefaultCheckpointError,
    WarmStartValidationError,
    compute_sha256,
    resolve_default_checkpoint,
    validate_warm_start,
)
from python.selfplay.inference_server import InferenceServer


def _write_full_trainer_state(
    path: Path, num_blocks: int = 12, num_filters: int = 128, step: int = 5
) -> Path:
    model = ChessNet(num_blocks=num_blocks, num_filters=num_filters)
    torch.save(
        {
            "checkpoint_type": "supervised",
            "model": model.state_dict(),
            "num_blocks": num_blocks,
            "num_filters": num_filters,
            "step": step,
        },
        path,
    )
    return path


def _write_registry(repo_root: Path, checkpoints: list[dict]) -> None:
    docs = repo_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "checkpoint_registry.json").write_text(
        json.dumps({"version": 1, "checkpoints": checkpoints}), encoding="utf-8"
    )


def test_missing_checkpoint_fails_closed(tmp_path):
    with pytest.raises(WarmStartValidationError, match="not found"):
        validate_warm_start(tmp_path / "does_not_exist.pt", repo_root=tmp_path)


def test_unreadable_checkpoint_fails_closed(tmp_path):
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"not a torch checkpoint at all")
    with pytest.raises(Exception):
        validate_warm_start(bad, repo_root=tmp_path)


def test_incompatible_architecture_fails_closed(tmp_path):
    small = _write_full_trainer_state(tmp_path / "small.pt", num_blocks=2, num_filters=32)
    with pytest.raises(WarmStartValidationError, match="architecture"):
        validate_warm_start(small, repo_root=tmp_path)


def test_unregistered_checkpoint_rejected_without_override(tmp_path):
    ckpt = _write_full_trainer_state(tmp_path / "unregistered.pt")
    _write_registry(tmp_path, [])  # registry exists, entry absent
    with pytest.raises(WarmStartValidationError, match="not in the checkpoint registry"):
        validate_warm_start(ckpt, repo_root=tmp_path)


def test_verified_and_designated_warm_start_accepted(tmp_path):
    ckpt = _write_full_trainer_state(tmp_path / "good.pt", step=42)
    sha = compute_sha256(ckpt)
    _write_registry(
        tmp_path,
        [
            {
                "path": str(ckpt),
                "file": str(ckpt),
                "sha256": sha,
                "status": "verified",
                "role": "supervised_baseline",
                "step": 42,
                "designated_warm_start": True,
            }
        ],
    )
    record = validate_warm_start(ckpt, repo_root=tmp_path)
    assert record["status"] == "verified"
    assert record["sha256"] == sha
    assert record["role"] == "supervised_baseline"


def test_verified_but_undesignated_rejected(tmp_path):
    ckpt = _write_full_trainer_state(tmp_path / "good2.pt")
    _write_registry(
        tmp_path,
        [
            {
                "path": str(ckpt),
                "file": str(ckpt),
                "sha256": compute_sha256(ckpt),
                "status": "verified",
                "role": "candidate",
                "designated_warm_start": False,
            }
        ],
    )
    with pytest.raises(WarmStartValidationError, match="NOT designated"):
        validate_warm_start(ckpt, repo_root=tmp_path)


def test_quarantined_registry_entry_rejected(tmp_path):
    ckpt = _write_full_trainer_state(tmp_path / "quarantined.pt")
    _write_registry(
        tmp_path,
        [
            {
                "path": str(ckpt),
                "file": str(ckpt),
                "sha256": compute_sha256(ckpt),
                "status": "quarantined",
                "role": "contaminated_rl_snapshot",
                "designated_warm_start": False,
            }
        ],
    )
    with pytest.raises(WarmStartValidationError, match="quarantined"):
        validate_warm_start(ckpt, repo_root=tmp_path)


def test_missing_status_rejected_even_if_file_reappears_with_other_content(tmp_path):
    """A 'missing' registry record can never be a warm start while its status says missing."""
    ckpt = _write_full_trainer_state(tmp_path / "back_again.pt")
    _write_registry(
        tmp_path,
        [
            {
                "path": str(ckpt),
                "file": str(ckpt),
                "sha256": "0" * 64,  # hash of the historical artifact
                "status": "missing",
                "role": "historical",
                "designated_warm_start": False,
            }
        ],
    )
    with pytest.raises(WarmStartValidationError, match="missing"):
        validate_warm_start(ckpt, repo_root=tmp_path)


def test_allow_unverified_override_is_explicit(tmp_path):
    ckpt = _write_full_trainer_state(tmp_path / "override.pt")
    _write_registry(tmp_path, [])
    record = validate_warm_start(ckpt, repo_root=tmp_path, allow_unverified=True)
    assert record["status"] == "unverified-override"
    assert record["sha256"] == compute_sha256(ckpt)


def test_inference_server_init_fails_closed_on_invalid_checkpoint(tmp_path):
    """No silent random-init fallback: a bad checkpoint must stop construction."""
    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"\x00\x01garbage")
    with pytest.raises(RuntimeError, match="fail-closed"):
        InferenceServer(checkpoint_path=str(bad), device="cpu")


def test_inference_server_init_fails_closed_on_missing_checkpoint(tmp_path):
    with pytest.raises(Exception):
        InferenceServer(checkpoint_path=str(tmp_path / "absent.pt"), device="cpu")


def test_bundled_registry_matches_disk_when_artifacts_present():
    """Every committed registry entry marked verified must match the actual
    artifact on disk, and when the registry certifies no verified artifact
    (the post-cleanup state since commit 7099019) default checkpoint
    resolution must fail closed instead of silently substituting.
    (Skipped on checkouts without the gitignored models/ directory.)"""
    import os

    repo_root = Path(__file__).resolve().parent.parent
    # Artifacts live in the gitignored models/ tree. On the canonical checkout
    # repo_root itself contains it; for linked-worktree runs an alternate root
    # can be supplied via ZEROPAWN_MAIN_WORKTREE instead of a hardcoded path.
    candidates = [repo_root]
    main_tree = os.environ.get("ZEROPAWN_MAIN_WORKTREE")
    if main_tree:
        candidates.append(Path(main_tree))
    artifact_root = next(
        (c for c in candidates if (c / "models" / "manifest.json").is_file()), None
    )
    if artifact_root is None:
        pytest.skip("models/ artifacts not present on this checkout")
    registry = json.loads(
        (repo_root / "docs" / "checkpoint_registry.json").read_text(encoding="utf-8")
    )
    verified = [r for r in registry["checkpoints"] if r.get("status") == "verified"]
    for rec in verified:
        f = artifact_root / rec["file"]
        assert f.is_file(), f"{rec['file']} registered verified but missing on disk"
        assert compute_sha256(f) == rec["sha256"], f"{rec['file']} hash drift"
        assert rec.get("designated_warm_start") is False or rec.get("designation_evidence"), (
            f"{rec['file']} designated without evidence"
        )
    if not verified:
        # Post-cleanup governance (2026-11-03): no verified artifact remains
        # locally — the fail-closed resolver must refuse, naming the absent
        # designated baseline, rather than fall back to a disqualified asset.
        with pytest.raises(DefaultCheckpointError):
            resolve_default_checkpoint(repo_root)


def test_inference_server_empty_checkpoint_path_means_random_init():
    """checkpoint_path='' is 'no checkpoint requested' (None-like), not a bad file:
    explicit random-init construction stays available for callers that opt in."""
    server = InferenceServer(checkpoint_path="", device="cpu")
    assert server.model is not None
    assert server.model_version == 0
