"""Artifact governance: registry decisions, discovery filters, and recovery.

Covers audit findings CKP-08 (unregistered warm-start-capable artifact on disk),
CKP-09 (schema zoo / model-only container in the RL tree), CKP-11 (stale default
model chain resolving to disqualified artifacts), CKP-14 (governance files had no
atomic in-repo writer) and INF-03 (`runs recover` could suggest a non-checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from python.model.chess_net import ChessNet
from python.model.checkpoint_registry import (
    DefaultCheckpointError,
    WarmStartValidationError,
    load_registry,
    upsert_registry_entry,
    verify_checkpoint_identity,
    write_registry,
)


def _small_header_payload(blocks: int = 2, filters: int = 32, **extra) -> dict:
    payload = {
        "model": ChessNet(num_blocks=blocks, num_filters=filters).state_dict(),
        "num_blocks": blocks,
        "num_filters": filters,
        "step": 1,
    }
    payload.update(extra)
    return payload


# ── Registry writer (CKP-14) ─────────────────────────────────────────────────


def test_write_registry_is_atomic_and_locked(tmp_path):
    docs = tmp_path / "docs"
    registry = {"version": 1, "checkpoints": [{"id": "a", "file": "x.pt", "status": "verified"}]}
    path = write_registry(registry, tmp_path, stamp_verified=True)
    assert path == docs / "checkpoint_registry.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["checkpoints"][0]["id"] == "a"
    assert "last_verified" in written
    assert not list(docs.glob("*.tmp")), "atomic write must not leave temp files"


def test_upsert_registry_entry_inserts_and_updates(tmp_path):
    (tmp_path / "docs").mkdir()
    write_registry({"version": 1, "checkpoints": []}, tmp_path)
    upsert_registry_entry(
        {"id": "ckpt-new", "file": "models/x.pt", "sha256": "a" * 64, "status": "verified"},
        tmp_path,
    )
    reg = load_registry(tmp_path)
    assert [c["id"] for c in reg["checkpoints"]] == ["ckpt-new"]

    # Second upsert updates in place (matched by id), never duplicates.
    upsert_registry_entry({"id": "ckpt-new", "status": "quarantined"}, tmp_path)
    reg = load_registry(tmp_path)
    assert len(reg["checkpoints"]) == 1
    entry = reg["checkpoints"][0]
    assert entry["status"] == "quarantined"
    assert entry["sha256"] == "a" * 64  # untouched fields survive


def test_bundled_registry_declares_the_unattributed_container_disqualified():
    """CKP-08 registry decision: the stray weights-only container is governed."""
    repo_root = Path(__file__).resolve().parent.parent
    reg = json.loads((repo_root / "docs" / "checkpoint_registry.json").read_text(encoding="utf-8"))
    matches = [
        c
        for c in reg["checkpoints"]
        if c.get("file") == "runs/phase4_rl/warm_start_ckpts/phase4_weights_only.pt"
    ]
    assert matches, "the artifact found by the audit must be registered"
    entry = matches[0]
    assert entry["status"] == "disqualified"
    assert entry["sha256"] == "56cee5143e543e2a4a98ee3561feacc09004f903625fd26855f8d9dbbaab6e73"
    assert entry["disqualification"]["reason"]
    assert entry["designated_warm_start"] is False


def test_disqualified_container_is_refused_from_every_path(tmp_path):
    """Both the registered path and a copy of it are refused as a warm start."""
    artifact = tmp_path / "phase4_weights_only.pt"
    torch.save(
        {
            "model": ChessNet(num_blocks=12, num_filters=128).state_dict(),
            "num_blocks": 12,
            "num_filters": 128,
        },
        artifact,
    )
    (tmp_path / "docs").mkdir()
    from python.model.checkpoint_registry import compute_sha256

    sha = compute_sha256(artifact)
    write_registry(
        {
            "version": 1,
            "checkpoints": [
                {
                    "id": "ckpt-11",
                    "file": str(artifact),
                    "sha256": sha,
                    "status": "disqualified",
                    "designated_warm_start": False,
                    "disqualification": {"reason": "unattributed_weights_only_container"},
                }
            ],
        },
        tmp_path,
    )
    with pytest.raises(WarmStartValidationError, match="DISQUALIFIED"):
        validate_warm_start_path = _warm_start(artifact, tmp_path)

    copy = tmp_path / "copy_of_container.pt"
    copy.write_bytes(artifact.read_bytes())
    with pytest.raises(WarmStartValidationError, match="different path"):
        _warm_start(copy, tmp_path)
    verdict = verify_checkpoint_identity(copy, repo_root=tmp_path)
    assert verdict["status"] == "copied_from_other_path"


def _warm_start(path: Path, repo_root: Path):
    from python.model.checkpoint_registry import validate_warm_start

    return validate_warm_start(path, repo_root=repo_root)


# ── Resume-candidate filtering (CKP-08/CKP-09) ───────────────────────────────


def test_weights_only_container_is_not_a_resume_candidate(tmp_path):
    from python.train.train_rl import inspect_rl_checkpoint, is_resumable_candidate

    container = tmp_path / "curriculum_final_latest.pt"
    torch.save(
        {
            "checkpoint_type": "curriculum",
            "model": ChessNet(num_blocks=12, num_filters=128).state_dict(),
            "num_blocks": 12,
            "num_filters": 128,
        },
        container,
    )
    info = inspect_rl_checkpoint(container)
    assert info is not None
    acceptable, reason = is_resumable_candidate(info)
    assert acceptable is False
    assert "weights-only container" in reason


def test_declared_rl_checkpoint_without_optimizer_state_is_resumable(tmp_path):
    from python.train.train_rl import inspect_rl_checkpoint, is_resumable_candidate

    first_save = tmp_path / "rl_first.pt"
    torch.save(
        {
            "checkpoint_type": "rl",
            "checkpoint_format": 2,
            "model": ChessNet(num_blocks=12, num_filters=128).state_dict(),
            "optimizer": {"param_groups": [{"momentum": 0.9}], "state": {}},
            "scheduler": {"current_step": 0},
            "step": 0,
            "num_blocks": 12,
            "num_filters": 128,
        },
        first_save,
    )
    info = inspect_rl_checkpoint(first_save)
    acceptable, reason = is_resumable_candidate(info)
    assert acceptable, reason


def test_foreign_optimizer_state_is_not_a_resume_candidate(tmp_path):
    from python.train.train_rl import inspect_rl_checkpoint, is_resumable_candidate

    adamw = tmp_path / "adamw.pt"
    model = ChessNet(num_blocks=12, num_filters=128)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {
                "param_groups": [{"betas": (0.9, 0.999), "lr": 1e-5}],
                "state": {"0": {"exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}},
            },
            "num_blocks": 12,
            "num_filters": 128,
            "step": 6000,
        },
        adamw,
    )
    info = inspect_rl_checkpoint(adamw)
    acceptable, reason = is_resumable_candidate(info)
    assert acceptable is False
    assert "AdamW" in reason


def test_orchestrator_selection_ignores_weights_only_files_in_the_rl_dir(tmp_path):
    """A weights-only file in the RL directory must not become the resume point."""
    from python.model.chess_net import ChessNet as Net
    from scripts.run_phase4_rl import Phase4Orchestrator

    ckpt_dir = tmp_path / "checkpoints_rl"
    ckpt_dir.mkdir()
    old_container = ckpt_dir / "curriculum_final_latest.pt"
    torch.save(
        {
            "checkpoint_type": "curriculum",
            "model": Net(num_blocks=12, num_filters=128).state_dict(),
            "num_blocks": 12,
            "num_filters": 128,
        },
        old_container,
    )
    real = ckpt_dir / "phase4_ckpt_v5_step997.pt"
    model = Net(num_blocks=12, num_filters=128)
    torch.save(
        {
            "checkpoint_type": "rl",
            "checkpoint_format": 2,
            "model": model.state_dict(),
            "optimizer": {"param_groups": [{"momentum": 0.9, "lr": 0.01}], "state": {}},
            "scheduler": {"current_step": 997},
            "step": 997,
            "num_blocks": 12,
            "num_filters": 128,
        },
        real,
    )

    orch = Phase4Orchestrator.__new__(Phase4Orchestrator)
    orch.checkpoint_dir = ckpt_dir
    selected = orch._select_resume_checkpoint()
    assert selected is not None and selected.endswith("phase4_ckpt_v5_step997.pt")


def test_curriculum_writers_do_not_target_the_rl_directory():
    """No in-repo writer may put a model-only container into the RL tree (CKP-09)."""
    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in (repo_root / "python").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "models/checkpoints_rl/curriculum" in text:
            offenders.append(str(path))
    for path in (repo_root / "scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "models/checkpoints_rl/curriculum" in text:
            offenders.append(str(path))
    assert not offenders, f"model-only containers must not live in the RL tree: {offenders}"


# ── Default-model selection (CKP-11) ─────────────────────────────────────────


def test_default_chain_skips_disqualified_and_quarantined_artifacts(tmp_path, monkeypatch):
    import python.cli.common as common

    models = tmp_path / "models"
    (models / "checkpoints_scaled").mkdir(parents=True)
    disqualified = models / "checkpoints_scaled" / "best_model.pt"
    torch.save(_small_header_payload(12, 128), disqualified)
    (tmp_path / "docs").mkdir()
    write_registry(
        {
            "version": 1,
            "checkpoints": [
                {
                    "id": "ckpt-01",
                    "file": "models/checkpoints_scaled/best_model.pt",
                    "sha256": "0" * 64,
                    "status": "disqualified",
                    "designation_evidence": "n/a",
                }
            ],
        },
        tmp_path,
    )
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("python.common.artifacts.repo_root", lambda: tmp_path)

    # Merged M-2/CKP-11 contract: resolution is fail-closed.  A disqualified
    # artifact is never served as a default — and because no designated,
    # verified entry exists, resolution raises instead of falling back.
    with pytest.raises(DefaultCheckpointError):
        common.get_default_checkpoint_path()

    # A registry-verified artifact becomes the default instead.
    verified = models / "checkpoints_scaled" / "latest_checkpoint.pt"
    torch.save(_small_header_payload(12, 128), verified)
    from python.model.checkpoint_registry import compute_sha256

    write_registry(
        {
            "version": 1,
            "checkpoints": [
                {
                    "id": "ckpt-01",
                    "file": "models/checkpoints_scaled/best_model.pt",
                    "sha256": "0" * 64,
                    "status": "disqualified",
                },
                {
                    "id": "ckpt-10",
                    "file": "models/checkpoints_scaled/latest_checkpoint.pt",
                    "sha256": compute_sha256(verified),
                    "status": "verified",
                    "designated_warm_start": True,
                    "designation_evidence": "unit test",
                },
            ],
        },
        tmp_path,
    )
    assert Path(common.get_default_checkpoint_path()) == verified


def test_discovery_deduplicates_by_path_not_name(tmp_path, monkeypatch):
    import python.cli.common as common

    models = tmp_path / "models"
    for tree in ("checkpoints_scaled", "checkpoints_finetune"):
        (models / tree).mkdir(parents=True)
        torch.save(_small_header_payload(12, 128, step=10), models / tree / "latest_checkpoint.pt")
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("python.common.artifacts.repo_root", lambda: tmp_path)

    found = common.get_available_checkpoints()
    same_name = [c for c in found if c["name"] == "latest_checkpoint.pt"]
    assert len(same_name) == 2, "both trees must be listed (name-keyed dedup hid one)"


# ── runs recover (INF-03) ────────────────────────────────────────────────────


def test_runs_recover_never_suggests_a_non_checkpoint(tmp_path, monkeypatch, capsys):
    import python.cli.commands_runs as runs_cli

    models_rl = tmp_path / "models" / "checkpoints_rl"
    models_rl.mkdir(parents=True)
    # Newest file by mtime is a weights-only container; the resumable checkpoint
    # is older. The suggestion must be the resumable one.
    container = models_rl / "curriculum_final_latest.pt"
    torch.save(
        {
            "checkpoint_type": "curriculum",
            "model": ChessNet(num_blocks=12, num_filters=128).state_dict(),
            "num_blocks": 12,
            "num_filters": 128,
        },
        container,
    )
    real = models_rl / "phase4_ckpt_v5_step997.pt"
    torch.save(
        {
            "checkpoint_type": "rl",
            "model": ChessNet(num_blocks=12, num_filters=128).state_dict(),
            "optimizer": {"param_groups": [{"momentum": 0.9}], "state": {}},
            "scheduler": {"current_step": 1},
            "step": 997,
            "num_blocks": 12,
            "num_filters": 128,
        },
        real,
    )
    import os as _os
    import time as _time

    old = _time.time() - 3600
    _os.utime(real, (old, old))  # make the container the newest by mtime

    monkeypatch.setattr(runs_cli, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "runs" / "phase4_rl"
    run_dir.mkdir(parents=True)
    args = type("A", (), {"run_id": str(run_dir)})()
    assert runs_cli.run_runs_recover(args) == 0
    out = capsys.readouterr().out
    # Rich console hard-wraps the rendered line at the console width, which can
    # split the printed absolute path (and the filename inside it) across
    # lines on deep checkouts; compare with the wrap newlines removed.
    assert "phase4_ckpt_v5_step997.pt" in out.replace("\n", "")
