"""Schema contract, safe loading, warm-start vs resume, provenance, and export.

Covers audit findings CKP-02 (supervised `--checkpoint` aliased full resume),
CKP-06 (warm-start authentication loaded with ``weights_only=False``), CKP-10
(``checkpoint_format`` written but never read), CKP-12 (no git SHA/seed/CUDA RNG
inside artifacts) and CKP-13 (export key rename, non-atomic write, warn-only
compatibility check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from python.model.checkpoint_schema import (
    CURRENT_RL_FORMAT,
    CURRENT_SUPERVISED_FORMAT,
    CheckpointSchemaError,
    build_provenance,
    detect_family,
    summarize_schema_support,
    validate_checkpoint_schema,
)
from python.model.chess_net import ChessNet
from python.model.checkpoint_registry import (
    WarmStartValidationError,
    inspect_checkpoint,
    validate_warm_start,
    verify_checkpoint_identity,
)


def _small_state(num_blocks: int = 2, num_filters: int = 32) -> dict:
    return ChessNet(num_blocks=num_blocks, num_filters=num_filters).state_dict()


def _rl_payload(**overrides) -> dict:
    payload = {
        "checkpoint_type": "rl",
        "checkpoint_format": CURRENT_RL_FORMAT,
        "model": _small_state(),
        "optimizer": {"param_groups": [{"momentum": 0.9, "lr": 0.01}], "state": {}},
        "scheduler": {"current_step": 3},
        "step": 3,
        "num_blocks": 2,
        "num_filters": 32,
        "buffer_state": {
            "write_head": 3,
            "size": 3,
            "total_added": 3,
            "capacity": 100,
            "generation": "g1",
        },
        "run_metadata": {"run_id": "r1", "run_type": "rl", "run_state": "running"},
        "provenance": {"run_id": "r1", "run_type": "rl"},
    }
    payload.update(overrides)
    return payload


# ── Schema contract (CKP-10) ─────────────────────────────────────────────────


def test_schema_support_matrix_is_explicit():
    matrix = summarize_schema_support()
    assert matrix["current_formats"]["rl"] == CURRENT_RL_FORMAT
    assert matrix["current_formats"]["supervised"] == CURRENT_SUPERVISED_FORMAT
    assert "completed" in matrix["run_states"]
    assert matrix["run_metadata_version"] >= 1


def test_current_format_requires_all_fields(tmp_path):
    p = tmp_path / "full.pt"
    verdict = validate_checkpoint_schema(_rl_payload(), path=p, expected_family="rl")
    assert verdict.family == "rl"
    assert verdict.version == CURRENT_RL_FORMAT
    assert not verdict.is_legacy

    incomplete = _rl_payload()
    incomplete.pop("buffer_state")
    with pytest.raises(CheckpointSchemaError, match="missing required fields"):
        validate_checkpoint_schema(incomplete, path=p, expected_family="rl")


def test_newer_format_is_rejected_not_reinterpreted(tmp_path):
    future = _rl_payload(checkpoint_format=CURRENT_RL_FORMAT + 1)
    with pytest.raises(CheckpointSchemaError, match="supports at most"):
        validate_checkpoint_schema(future, path=tmp_path / "future.pt", expected_family="rl")


def test_invalid_version_marker_rejected(tmp_path):
    with pytest.raises(CheckpointSchemaError, match="non-numeric"):
        validate_checkpoint_schema(
            _rl_payload(checkpoint_format="v2"), path=tmp_path / "x.pt", expected_family="rl"
        )
    with pytest.raises(CheckpointSchemaError, match="invalid checkpoint_format"):
        validate_checkpoint_schema(
            _rl_payload(checkpoint_format=0), path=tmp_path / "x.pt", expected_family="rl"
        )


def test_legacy_artifact_accepted_explicitly_with_warning(tmp_path):
    legacy = _rl_payload()
    legacy.pop("checkpoint_format")
    legacy.pop("provenance")
    legacy.pop("run_metadata")
    verdict = validate_checkpoint_schema(legacy, path=tmp_path / "legacy.pt", expected_family="rl")
    assert verdict.is_legacy
    assert verdict.warnings, "legacy acceptance must be reported, not silent"


def test_legacy_rejected_when_legacy_not_allowed(tmp_path):
    legacy = _rl_payload()
    legacy.pop("checkpoint_format")
    with pytest.raises(CheckpointSchemaError, match="no checkpoint_format marker"):
        validate_checkpoint_schema(
            legacy, path=tmp_path / "legacy.pt", allow_legacy=False, expected_family="rl"
        )


def test_weights_only_container_cannot_be_a_full_resume(tmp_path):
    weights_only = {"model": _small_state(), "num_blocks": 2, "num_filters": 32, "step": 9}
    with pytest.raises(CheckpointSchemaError, match="warm start"):
        validate_checkpoint_schema(
            weights_only,
            path=tmp_path / "w.pt",
            expected_family="rl",
            require_full_state=True,
        )


def test_legacy_weights_only_detected_structurally():
    assert detect_family({"model": _small_state()}) == "weights_only"
    assert detect_family({"checkpoint_type": "curriculum", "model": {}}) == "curriculum"
    assert detect_family(_rl_payload()) == "rl"


# ── Safe loading (CKP-06) ────────────────────────────────────────────────────


def test_warm_start_inspection_rejects_weightless_payload(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({"step": 3, "num_blocks": 12, "num_filters": 128}, path)
    with pytest.raises(WarmStartValidationError, match="no model weights"):
        inspect_checkpoint(path)


def test_warm_start_inspection_rejects_non_dict_payload(tmp_path):
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(WarmStartValidationError, match="not a dict"):
        inspect_checkpoint(path)


def test_warm_start_inspection_uses_tensor_only_loading(tmp_path, monkeypatch):
    """The authentication path must never call torch.load with weights_only=False."""
    seen: list = []
    real_load = torch.load

    def spy_load(*args, **kwargs):
        seen.append(kwargs.get("weights_only", "default"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    path = tmp_path / "ok.pt"
    torch.save(
        {"model": _small_state(), "num_blocks": 2, "num_filters": 32, "step": 1},
        path,
    )
    inspect_checkpoint(path)
    assert seen and all(flag is True for flag in seen), seen


class _UnpicklablePayload:
    """Module-level class so torch.save can actually pickle an instance."""

    def __init__(self, marker: str = "evil") -> None:
        self.marker = marker


def test_pickled_object_payload_is_refused_without_execution(tmp_path):
    """A checkpoint carrying arbitrary pickled objects is not authenticatable."""
    path = tmp_path / "evil.pt"
    torch.save({"model": _small_state(), "payload": _UnpicklablePayload()}, path)
    # weights_only loading (the authentication path) refuses the foreign type
    # instead of executing its unpickling; never a silent accept.
    with pytest.raises(Exception) as excinfo:
        inspect_checkpoint(path)
    assert (
        "payload" in str(excinfo.value)
        or "Unpickl" in str(excinfo.value)
        or "weights" in str(excinfo.value)
    )


def test_copied_artifact_does_not_inherit_designation(tmp_path):
    """Registry matching is path-first: a byte-identical copy is not the artifact."""
    real = tmp_path / "real.pt"
    torch.save({"model": _small_state(), "num_blocks": 12, "num_filters": 128, "step": 1}, real)
    (tmp_path / "docs").mkdir()
    from python.model.checkpoint_registry import compute_sha256

    sha = compute_sha256(real)
    (tmp_path / "docs" / "checkpoint_registry.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "id": "ckpt-x",
                        "file": str(real),
                        "sha256": sha,
                        "status": "verified",
                        "designated_warm_start": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    copy = tmp_path / "copy.pt"
    copy.write_bytes(real.read_bytes())

    # The registered path verifies.
    assert validate_warm_start(real, repo_root=tmp_path)["status"] == "verified"
    # The copy does not inherit the designation.
    with pytest.raises(WarmStartValidationError, match="different path"):
        validate_warm_start(copy, repo_root=tmp_path)
    verdict = verify_checkpoint_identity(copy, repo_root=tmp_path)
    assert verdict["status"] == "copied_from_other_path"


# ── Load-time hash verification (phase J) ────────────────────────────────────


def test_verify_checkpoint_identity_reports_hash_mismatch(tmp_path):
    ckpt = tmp_path / "drift.pt"
    torch.save({"model": _small_state(), "num_blocks": 2, "num_filters": 32}, ckpt)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "checkpoint_registry.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {"id": "c1", "file": str(ckpt), "sha256": "0" * 64, "status": "verified"}
                ]
            }
        ),
        encoding="utf-8",
    )
    verdict = verify_checkpoint_identity(ckpt, repo_root=tmp_path)
    assert verdict["status"] == "hash_mismatch"
    assert verdict["expected"] == "0" * 64


def test_verify_checkpoint_identity_flags_quarantine_and_disqualification(tmp_path):
    ckpt = tmp_path / "q.pt"
    torch.save({"model": _small_state(), "num_blocks": 2, "num_filters": 32}, ckpt)
    (tmp_path / "docs").mkdir()
    from python.model.checkpoint_registry import compute_sha256

    sha = compute_sha256(ckpt)
    for status, key in (("quarantined", "quarantine"), ("disqualified", "disqualification")):
        (tmp_path / "docs" / "checkpoint_registry.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {
                            "id": "c",
                            "file": str(ckpt),
                            "sha256": sha,
                            "status": status,
                            key: {"reason": f"why-{status}"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        verdict = verify_checkpoint_identity(ckpt, repo_root=tmp_path)
        assert verdict["status"] == status
        assert verdict["detail"] == f"why-{status}"


def test_rl_resume_refuses_a_quarantined_registered_file(tmp_path, monkeypatch):
    from python.selfplay.replay_buffer import ReplayBuffer
    from python.train.train_rl import RLTrainer

    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=ReplayBuffer(buffer_dir=tmp_path / "b", capacity=32, create_new=True),
        device="cpu",
        batch_size=4,
        checkpoint_dir=tmp_path / "ckpt",
        sync_weights_path=tmp_path / "sync.pt",
    )
    ckpt = trainer.save_checkpoint(tmp_path / "q.pt", keep_last_n=-1)

    repo = Path(__file__).resolve().parent.parent
    from python.model.checkpoint_registry import compute_sha256

    def fake_verdict(path, repo_root="."):
        return {
            "status": "quarantined",
            "sha256": compute_sha256(path),
            "detail": "contaminated snapshot",
        }

    monkeypatch.setattr("python.model.checkpoint_registry.verify_checkpoint_identity", fake_verdict)
    trainer2 = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=ReplayBuffer(buffer_dir=tmp_path / "b2", capacity=32, create_new=True),
        device="cpu",
        batch_size=4,
        checkpoint_dir=tmp_path / "ckpt2",
        sync_weights_path=tmp_path / "sync2.pt",
    )
    with pytest.raises(RuntimeError, match="quarantined"):
        trainer2.load_checkpoint(ckpt)
    assert repo  # keeps the repo import used (path resolution sanity)


# ── Provenance (CKP-12) ──────────────────────────────────────────────────────


def test_provenance_records_the_required_identity_fields():
    prov = build_provenance(
        run_id="run-1",
        run_type="rl",
        config={"base_lr": 0.01},
        model_arch=(12, 128),
        param_count=123,
        dataset={"train": {"receipt_id": "sha256:abc"}},
        parent_checkpoint={"path": "p.pt", "sha256": "deadbeef"},
        optimizer_config={"type": "SGD", "momentum": 0.9},
        scheduler_config={"warmup_steps": 500},
    )
    assert prov["run_id"] == "run-1"
    assert prov["git_commit"]  # best-effort, present in this repo
    assert prov["created_at"]
    assert prov["model"] == {"num_blocks": 12, "num_filters": 128, "param_count": 123}
    assert prov["dataset"]["train"]["receipt_id"] == "sha256:abc"
    assert prov["parent_checkpoint"]["sha256"] == "deadbeef"
    assert prov["optimizer"]["type"] == "SGD"
    assert prov["scheduler"]["warmup_steps"] == 500


def test_rl_checkpoint_records_provenance_and_cuda_rng(tmp_path):
    from python.selfplay.replay_buffer import ReplayBuffer
    from python.train.train_rl import RLTrainer

    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=ReplayBuffer(buffer_dir=tmp_path / "bp", capacity=32, create_new=True),
        device="cpu",
        batch_size=4,
        checkpoint_dir=tmp_path / "ckptp",
        sync_weights_path=tmp_path / "syncp.pt",
    )
    ckpt = trainer.save_checkpoint(tmp_path / "prov.pt", keep_last_n=-1)
    saved = torch.load(ckpt, weights_only=False)
    prov = saved["provenance"]
    assert prov["run_id"] == saved["run_metadata"]["run_id"]
    assert prov["model"]["param_count"] > 0
    assert prov["optimizer"]["momentum"] == 0.9
    assert prov["scheduler"]["warmup_steps"] == 500
    # CPU run: no CUDA RNG is fabricated, and the key exists so consumers can
    # distinguish "not applicable" from "missing".
    assert "cuda_rng_state" in saved
    assert saved["cuda_rng_state"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG capture requires a GPU")
def test_cuda_rng_state_is_captured_and_restored(tmp_path):
    from python.selfplay.replay_buffer import ReplayBuffer
    from python.train.train_rl import RLTrainer

    def _trainer(name: str) -> RLTrainer:
        return RLTrainer(
            model=ChessNet(num_blocks=2, num_filters=32),
            replay_buffer=ReplayBuffer(
                buffer_dir=tmp_path / f"bc_{name}", capacity=32, create_new=True
            ),
            device="cuda",
            batch_size=4,
            checkpoint_dir=tmp_path / f"ckptc_{name}",
            sync_weights_path=tmp_path / f"sync_{name}.pt",
        )

    trainer = _trainer("src")
    torch.cuda.manual_seed_all(1234)
    expected = list(torch.cuda.get_rng_state_all())
    ckpt = trainer.save_checkpoint(tmp_path / "cuda.pt", keep_last_n=-1)
    saved = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert saved["cuda_rng_state"] is not None
    assert len(saved["cuda_rng_state"]) == torch.cuda.device_count()

    # Disturb the RNG, then resume and confirm the captured state returns.
    torch.cuda.manual_seed_all(999)
    trainer2 = _trainer("dst")
    trainer2.load_checkpoint(ckpt, reset_buffer=True)
    restored = list(torch.cuda.get_rng_state_all())
    assert torch.equal(restored[0].cpu(), expected[0].cpu())


# ── Supervised CLI semantics (CKP-02) ────────────────────────────────────────


def _supervised_args(**overrides) -> argparse.Namespace:
    base = dict(
        checkpoint=None,
        resume=None,
        resume_weights_only=False,
        override_schedule=False,
        data_role=None,
        allow_uncertified_data=False,
        verify_identity="full",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _run_supervised_cli(args, captured: dict) -> int:
    import python.cli.commands_train as ct

    def fake_train_supervised(**kwargs):
        captured.update(kwargs)
        return {"total_loss": 0.1, "top1": 10.0}

    original = ct.train_supervised
    import python.train.train_supervised as ts

    ts_original = ts.train_supervised
    ct.train_supervised = fake_train_supervised  # type: ignore[attr-defined]
    try:
        # The CLI imports train_supervised lazily inside the function.
        import sys

        sys.modules["python.train.train_supervised"].train_supervised = fake_train_supervised
        args.mode = "supervised"
        args.supervised = True
        return ct.run_supervised_training(args)
    finally:
        ct.train_supervised = original  # type: ignore[attr-defined]
        sys.modules["python.train.train_supervised"].train_supervised = ts_original


def test_supervised_cli_rejects_conflicting_initialization_flags(tmp_path):
    import python.cli.commands_train as ct

    args = _supervised_args(checkpoint="a.pt", resume="b.pt")
    args.mode = "supervised"
    args.supervised = True
    assert ct.run_supervised_training(args) == 1


def test_supervised_cli_rejects_resume_weights_only_without_path(tmp_path):
    import python.cli.commands_train as ct

    args = _supervised_args(resume_weights_only=True)
    args.mode = "supervised"
    args.supervised = True
    assert ct.run_supervised_training(args) == 1


def test_supervised_cli_rejects_missing_checkpoint_file(tmp_path):
    import python.cli.commands_train as ct

    args = _supervised_args(checkpoint=str(tmp_path / "nope.pt"))
    args.mode = "supervised"
    args.supervised = True
    assert ct.run_supervised_training(args) == 1


# ── Export contract (CKP-13) ─────────────────────────────────────────────────


def test_export_keeps_canonical_loss_key_and_is_weights_only(tmp_path):
    import python.cli.commands_models as cm

    src = tmp_path / "src.pt"
    torch.save(
        {
            "checkpoint_type": "supervised",
            "model": _small_state(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": 4,
            "best_val_loss": 1.25,
            "checkpoint_format": CURRENT_SUPERVISED_FORMAT,
        },
        src,
    )
    out = tmp_path / "exported.pt"
    args = argparse.Namespace(checkpoint=str(src), output=str(out), weights_only=True)
    assert cm.run_models_export(args) == 0
    exported = torch.load(out, weights_only=False)
    assert exported["best_val_loss"] == 1.25
    assert exported["val_loss"] == 1.25
    assert exported["checkpoint_type"] == "export"
    assert exported["inference_only"] is True
    assert "optimizer" not in exported
    assert exported["source_checkpoint"] == str(src)


def test_export_does_not_leave_tmp_files(tmp_path):
    import python.cli.commands_models as cm

    src = tmp_path / "src2.pt"
    torch.save({"model": _small_state(), "num_blocks": 2, "num_filters": 32}, src)
    out = tmp_path / "exported2.pt"
    cm.run_models_export(
        argparse.Namespace(checkpoint=str(src), output=str(out), weights_only=True)
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_check_compat_exits_nonzero_on_mismatch(tmp_path, capsys):
    import python.cli.commands_models as cm

    mismatched = tmp_path / "small.pt"
    torch.save({"model": _small_state(), "num_blocks": 2, "num_filters": 32}, mismatched)
    args = argparse.Namespace(checkpoint=str(mismatched), expected_blocks=12, expected_filters=128)
    assert cm.run_models_check_compat(args) == 1
