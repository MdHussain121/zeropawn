"""Regression tests for audit SA3-03: book-train resume provenance.

Covers the fail-closed resume contract added after the 2026-11-05 pre-push
audit (finding SA3-03, Low): ``book train --resume`` used to hardcode the
superseded ``models/checkpoints_scaled/best_model.pt`` and load whatever
bytes sat there with no provenance or compatibility validation. Now:

- the default resume checkpoint is the registry-designated, verified
  baseline resolved fail-closed (no hardcoded superseded artifact);
- a resume checkpoint's registry standing is validated before any load
  (quarantined/disqualified/missing/hash-mismatch refuse unless overridden);
- every run writes ``run_manifest.json`` (inputs, config, resume identity,
  output hashes) and a later resume must match it;
- trainer-output artifacts without a manifest refuse unless explicitly
  allowed (legacy interrupted runs stay reachable via override).

All fixtures are hermetic (tmp_path only); no production data/ or models/
tree is touched.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import chess
import pytest
import torch

import python.train.train_opening_policy as top_module
from python.cli.commands_book import register_parser
from python.data.build_opening_dataset import build_opening_dataset
from python.data.opening_corpus import OpeningCorpus
from python.data.dataset_validation import (
    DatasetContractError,
    validate_split,
    write_receipt,
)
from python.model.chess_net import ChessNet
from python.model.checkpoint_registry import (
    DefaultCheckpointError,
    compute_sha256,
)
from python.train.opening_run_manifest import (
    MANIFEST_NAME,
    OpeningRunProvenanceError,
    ResumeProvenance,
    read_run_manifest,
    receipt_files_digest,
    validate_manifest_for_resume,
)
from python.train.train_opening_policy import train_opening_policy


# ---------------------------------------------------------------------------
# Hermetic fixtures
# ---------------------------------------------------------------------------


def _tiny_dataset(tmp_path: Path, name: str, variant: int = 0) -> Path:
    """Build a small certified opening dataset (receipts written by the builder).

    ``variant`` changes the game counts so two datasets differ byte-wise and
    their validation receipts bind different file identities.
    """
    corpus = OpeningCorpus()
    board = chess.Board()
    for i in range(10 + variant):
        corpus.add_position_move(
            board, chess.Move.from_uci("e2e4"), game_id=f"{name}_e4_{i}", source="test"
        )
        corpus.add_position_move(
            board, chess.Move.from_uci("d2d4"), game_id=f"{name}_d4_{i}", source="test"
        )
    b_black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    for i in range(10 + variant):
        corpus.add_position_move(
            b_black, chess.Move.from_uci("c7c5"), game_id=f"{name}_c5_{i}", source="test"
        )
    ds_dir = tmp_path / name
    build_opening_dataset(corpus, output_dir=ds_dir, val_ratio=0.5, seed=42)
    return ds_dir


def _fake_checkpoint(path: Path, num_blocks: int = 2, num_filters: int = 32) -> Path:
    net = ChessNet(num_blocks=num_blocks, num_filters=num_filters)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "num_blocks": num_blocks,
            "num_filters": num_filters,
            "step": 1000,
        },
        path,
    )
    return path


def _write_registry(tmp_path: Path, entries: list) -> None:
    reg_dir = tmp_path / "docs"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "checkpoint_registry.json").write_text(
        json.dumps({"version": "test", "checkpoints": entries}), encoding="utf-8"
    )


def _run_trainer(ds_dir: Path, resume, out_dir: Path, tmp_path: Path, **overrides):
    kwargs = dict(
        opening_data_dir=ds_dir,
        resume_checkpoint=resume,
        output_checkpoint_dir=out_dir,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_steps=3,
        num_blocks=2,
        num_filters=32,
        device_name="cpu",
        repo_root=tmp_path,
    )
    kwargs.update(overrides)
    return train_opening_policy(**kwargs)


# ---------------------------------------------------------------------------
# Manifest written at run start, finalized with output hashes
# ---------------------------------------------------------------------------


def test_run_manifest_records_provenance_and_output_hashes(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, "ds_a")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out = tmp_path / "run1"

    res = _run_trainer(ds, baseline, out, tmp_path)

    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["kind"] == "opening_policy_run"
    assert manifest["status"] == "completed"
    # Resume identity: the exact baseline bytes this run started from.
    assert manifest["resume_checkpoint"]["sha256"] == compute_sha256(baseline)
    assert manifest["resume_checkpoint"]["path"] == str(baseline.resolve())
    # Dataset identity bound via the validation receipt digest.
    assert manifest["dataset_identity"]["train"]["certified"] is True
    assert manifest["dataset_identity"]["train"]["receipt_files_digest"] is not None
    assert manifest["dataset_identity"]["val"]["receipt_files_digest"] is not None
    # Config snapshot.
    assert manifest["config"]["num_blocks"] == 2
    assert manifest["config"]["lr"] == 1e-3
    # Every generated checkpoint is hash-bound at finalize time.
    best = out / "best_opening_model.pt"
    assert best.is_file()
    assert manifest["outputs"]["best_opening_model.pt"]["sha256"] == compute_sha256(best)
    assert manifest["outputs"]["latest_opening_model.pt"]["sha256"] == compute_sha256(
        out / "latest_opening_model.pt"
    )
    assert res["run_manifest"] == str(out / MANIFEST_NAME)


def test_interrupted_run_leaves_running_manifest_and_resumes(tmp_path: Path, monkeypatch):
    """The SA3-03 core case: an interrupted run's partial output stays resumable.

    The manifest written at run start keeps status ``running`` (no output
    hashes yet); a later run resuming from the partial artifact must validate
    against that manifest and continue.
    """
    ds = _tiny_dataset(tmp_path, "ds_b")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out = tmp_path / "run_interrupted"

    real_eval = top_module.evaluate_opening_policy
    eval_calls = {"n": 0}

    def flaky_eval(*args, **kwargs):
        eval_calls["n"] += 1
        if eval_calls["n"] >= 2:
            raise RuntimeError("simulated interruption")
        return {"val_policy_loss": 0.5, "val_top1": 0.0, "val_top3": 0.0, "val_samples": 4}

    monkeypatch.setattr(top_module, "evaluate_opening_policy", flaky_eval)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        # max_steps=None so epoch 0 completes (checkpoint + best are saved)
        # and epoch 1's evaluation is the interruption point.
        _run_trainer(ds, baseline, out, tmp_path, epochs=2, max_steps=None)
    monkeypatch.setattr(top_module, "evaluate_opening_policy", real_eval)

    partial = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert partial["status"] == "running"
    assert "outputs" not in partial
    assert (out / "best_opening_model.pt").is_file()

    # Resume from the partial artifact with the SAME configuration: valid.
    out2 = tmp_path / "run_continued"
    res = _run_trainer(ds, out / "best_opening_model.pt", out2, tmp_path, epochs=2, max_steps=None)
    assert res["final_step"] > 0
    continued = read_run_manifest(out2)
    assert continued is not None
    assert continued["resume_checkpoint"]["path"] == str((out / "best_opening_model.pt").resolve())


# ---------------------------------------------------------------------------
# Valid resume / mismatched provenance
# ---------------------------------------------------------------------------


def test_valid_resume_matches_previous_run_manifest(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, "ds_c")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)
    first = read_run_manifest(out1)
    assert first is not None and first["status"] == "completed"

    out2 = tmp_path / "run2"
    _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path)

    second = read_run_manifest(out2)
    assert second is not None
    assert (
        second["resume_checkpoint"]["sha256"] == first["outputs"]["best_opening_model.pt"]["sha256"]
    )


def test_resume_with_different_dataset_refused(tmp_path: Path):
    ds1 = _tiny_dataset(tmp_path, "ds_c1", variant=0)
    ds2 = _tiny_dataset(tmp_path, "ds_c2", variant=3)
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds1, baseline, out1, tmp_path)

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match="dataset identity"):
        _run_trainer(ds2, out1 / "best_opening_model.pt", out2, tmp_path)
    # The refused resume must not have written a new manifest over the old run.
    assert not (out2 / MANIFEST_NAME).exists()
    assert read_run_manifest(out1)["status"] == "completed"


def test_resume_with_config_drift_refused_but_override_reachable(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, "ds_d")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match="allow-resume-config-drift"):
        _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path, lr=2e-3)
    assert not (out2 / MANIFEST_NAME).exists()

    # Explicit override keeps legitimate tuning reachable (logged).
    _run_trainer(
        ds,
        out1 / "best_opening_model.pt",
        out2,
        tmp_path,
        lr=2e-3,
        allow_resume_config_drift=True,
    )
    assert read_run_manifest(out2)["status"] == "completed"


def test_resume_from_replaced_output_refused(tmp_path: Path):
    """A completed run's recorded output hash catches a swapped artifact."""
    ds = _tiny_dataset(tmp_path, "ds_e")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)

    # Replace the output with a *different but loadable* checkpoint: without
    # the manifest binding this would silently continue from foreign weights.
    foreign = ChessNet(num_blocks=2, num_filters=32)
    torch.save(
        {"model": foreign.state_dict(), "num_blocks": 2, "num_filters": 32, "step": 999},
        out1 / "best_opening_model.pt",
    )

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match="SHA-256 recorded"):
        _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path)
    assert not (out2 / MANIFEST_NAME).exists()

    _run_trainer(
        ds,
        out1 / "best_opening_model.pt",
        out2,
        tmp_path,
        allow_superseded_resume=True,
    )
    assert read_run_manifest(out2)["status"] == "completed"


def test_resume_with_different_architecture_refused(tmp_path: Path):
    """A different architecture is a hard binding, not tolerable drift."""
    ds = _tiny_dataset(tmp_path, "ds_arch")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match="architecture changed"):
        _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path, num_blocks=4)
    assert not (out2 / MANIFEST_NAME).exists()


def test_validate_manifest_refuses_architecture_and_dataset_digest_drift():
    """Unit pin of the two hard bindings: architecture and dataset identity."""
    prev = {
        "kind": "opening_policy_run",
        "manifest_version": "1.0",
        "config": {"num_blocks": 2, "num_filters": 32, "lr": 1e-3},
        "dataset_identity": {
            "train": {
                "data_dir": "/data/split",
                "certified": True,
                "receipt_files_digest": "sha256:aaaa",
            }
        },
        "outputs": {},
    }
    prov = ResumeProvenance(path=Path("/x/best_opening_model.pt"), sha256="0" * 64)

    # Architecture drift refuses even with the config-drift override set:
    # it is a checkpoint-compatibility error, not a tuning decision.
    with pytest.raises(OpeningRunProvenanceError, match="architecture changed"):
        validate_manifest_for_resume(
            prev,
            manifest_dir="/x",
            dataset_identity=prev["dataset_identity"],
            config={"num_blocks": 4, "num_filters": 32, "lr": 1e-3},
            resume_provenance=prov,
            allow_config_drift=True,
        )

    # Same directories but a changed validation-receipt digest (re-certified
    # different bytes) refuses: the resumed run would mix datasets.
    with pytest.raises(OpeningRunProvenanceError, match="dataset identity"):
        validate_manifest_for_resume(
            prev,
            manifest_dir="/x",
            dataset_identity={
                "train": {
                    "data_dir": "/data/split",
                    "certified": True,
                    "receipt_files_digest": "sha256:bbbb",
                }
            },
            config=prev["config"],
            resume_provenance=prov,
        )


def test_tampered_dataset_bytes_refused_on_resume(tmp_path: Path):
    """Flipping a receipt-bound byte after run 1 fails the split gate closed."""
    ds = _tiny_dataset(tmp_path, "ds_tamper")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)

    outcomes = ds / "train" / "outcomes.bin"
    raw = bytearray(outcomes.read_bytes())
    assert raw
    raw[0] ^= 0x01
    outcomes.write_bytes(bytes(raw))

    out2 = tmp_path / "run2"
    with pytest.raises(DatasetContractError):
        _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path)
    assert not (out2 / MANIFEST_NAME).exists()


def test_recertified_tampered_dataset_caught_by_manifest_digest(tmp_path: Path):
    """The full tamper-evidence chain: re-certified changed bytes still refuse.

    After tampering, re-running validation writes a receipt over the NEW
    bytes, so the split gate alone would pass — but the receipt digest then
    differs from run 1's manifest, and the provenance check refuses.
    """
    ds = _tiny_dataset(tmp_path, "ds_recert")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)
    old_digest = read_run_manifest(out1)["dataset_identity"]["train"]["receipt_files_digest"]
    assert old_digest is not None

    outcomes = ds / "train" / "outcomes.bin"
    raw = bytearray(outcomes.read_bytes())
    raw[0] ^= 0x01
    outcomes.write_bytes(bytes(raw))

    rep = validate_split(
        ds / "train", scan_planes="full", compute_hashes=True, check_build_record=False
    )
    assert rep.overall == "pass"
    write_receipt(ds / "train", {"train": rep}, overall=rep.overall)
    new_digest = receipt_files_digest(
        json.loads((ds / "train" / "validation_receipt.json").read_text(encoding="utf-8"))
    )
    assert new_digest != old_digest

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match="dataset identity"):
        _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path)
    assert not (out2 / MANIFEST_NAME).exists()


# ---------------------------------------------------------------------------
# Registry standing of the resume checkpoint
# ---------------------------------------------------------------------------


def _registry_entry(tmp_path: Path, artifact: Path, status: str, entry_id: str) -> dict:
    entry = {
        "id": entry_id,
        "file": str(artifact.relative_to(tmp_path)).replace("\\", "/"),
        "sha256": compute_sha256(artifact),
        "size_bytes": artifact.stat().st_size,
        "status": status,
    }
    if status == "quarantined":
        entry["quarantine"] = {"reason": "contaminated test artifact"}
    return entry


def test_resume_superseded_registry_checkpoint_refused(tmp_path: Path):
    baseline = _fake_checkpoint(tmp_path / "models" / "quarantined" / "baseline.pt")
    _write_registry(tmp_path, [_registry_entry(tmp_path, baseline, "quarantined", "ckpt-t-01")])
    ds = _tiny_dataset(tmp_path, "ds_f")

    out = tmp_path / "run1"
    with pytest.raises(OpeningRunProvenanceError, match="quarantined"):
        _run_trainer(ds, baseline, out, tmp_path)
    assert not (out / MANIFEST_NAME).exists()

    # Explicit override keeps the operation reachable (logged).
    _run_trainer(ds, baseline, out, tmp_path, allow_superseded_resume=True)
    assert read_run_manifest(out)["status"] == "completed"


def test_hardcoded_superseded_default_artifact_is_refused(tmp_path: Path):
    """Direct pin of SA3-03: the historical superseded baseline cannot resume.

    ckpt-08 (models/checkpoints_scaled/best_model.pt) is registered with
    status ``missing``; a file reappearing at that path is refused.
    """
    artifact = tmp_path / "models" / "checkpoints_scaled" / "best_model.pt"
    _fake_checkpoint(artifact)
    _write_registry(
        tmp_path,
        [
            {
                "id": "ckpt-t-missing-baseline",
                "file": "models/checkpoints_scaled/best_model.pt",
                "sha256": compute_sha256(artifact),
                "size_bytes": artifact.stat().st_size,
                "status": "missing",
                "role": "historical_certified_active_baseline",
            }
        ],
    )
    ds = _tiny_dataset(tmp_path, "ds_g")

    with pytest.raises(OpeningRunProvenanceError, match="missing"):
        _run_trainer(ds, artifact, tmp_path / "run1", tmp_path)


def test_unregistered_explicit_baseline_allowed_without_manifest_gate(tmp_path: Path):
    """An explicit, unregistered foreign baseline stays reachable (warning only)."""
    ds = _tiny_dataset(tmp_path, "ds_h")
    baseline = _fake_checkpoint(tmp_path / "foreign" / "some_other_baseline.pt")
    out = tmp_path / "run1"
    _run_trainer(ds, baseline, out, tmp_path)
    manifest = read_run_manifest(out)
    assert manifest["resume_checkpoint"]["registry_status"] == "unregistered"


def test_unregistered_trainer_output_without_manifest_still_gated(tmp_path: Path):
    """Gate-ordering pin: an unregistered-path warning must not short-circuit
    the missing-manifest gate — a manifest-less trainer-output artifact still
    needs its own --allow-missing-run-manifest override."""
    ds = _tiny_dataset(tmp_path, "ds_unreg")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)

    foreign = tmp_path / "foreign_dir"
    foreign.mkdir()
    shutil.copy2(out1 / "best_opening_model.pt", foreign / "best_opening_model.pt")

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match=MANIFEST_NAME):
        _run_trainer(ds, foreign / "best_opening_model.pt", out2, tmp_path)
    assert not (out2 / MANIFEST_NAME).exists()

    # The dedicated override keeps the legacy case reachable (logged).
    _run_trainer(
        ds,
        foreign / "best_opening_model.pt",
        out2,
        tmp_path,
        allow_missing_run_manifest=True,
    )
    assert read_run_manifest(out2)["status"] == "completed"


def test_missing_run_manifest_refused_with_legacy_override(tmp_path: Path):
    """A trainer-output artifact without its run manifest refuses (legacy partial)."""
    ds = _tiny_dataset(tmp_path, "ds_i")
    baseline = _fake_checkpoint(tmp_path / "baseline.pt")
    out1 = tmp_path / "run1"
    _run_trainer(ds, baseline, out1, tmp_path)
    (out1 / MANIFEST_NAME).unlink()

    out2 = tmp_path / "run2"
    with pytest.raises(OpeningRunProvenanceError, match=MANIFEST_NAME):
        _run_trainer(ds, out1 / "best_opening_model.pt", out2, tmp_path)
    assert not (out2 / MANIFEST_NAME).exists()

    _run_trainer(
        ds,
        out1 / "best_opening_model.pt",
        out2,
        tmp_path,
        allow_missing_run_manifest=True,
    )
    assert read_run_manifest(out2)["status"] == "completed"


# ---------------------------------------------------------------------------
# Honest default resolution (no hardcoded superseded artifact)
# ---------------------------------------------------------------------------


def test_default_resume_resolves_designated_baseline_fail_closed(tmp_path: Path):
    artifact = _fake_checkpoint(tmp_path / "models" / "baseline" / "best_model.pt")
    _write_registry(
        tmp_path,
        [
            {
                "id": "ckpt-t-designated",
                "file": "models/baseline/best_model.pt",
                "sha256": compute_sha256(artifact),
                "size_bytes": artifact.stat().st_size,
                "status": "verified",
                "designated_warm_start": True,
                "role": "supervised_baseline",
            }
        ],
    )
    ds = _tiny_dataset(tmp_path, "ds_j")
    out = tmp_path / "run1"

    _run_trainer(ds, None, out, tmp_path)
    manifest = read_run_manifest(out)
    assert manifest["resume_checkpoint"]["registry_entry_id"] == "ckpt-t-designated"
    assert manifest["resume_checkpoint"]["registry_status"] == "verified"
    assert manifest["resume_checkpoint"]["path"] == str(artifact.resolve())

    # Without any eligible designated artifact, the default refuses (fail-closed).
    _write_registry(tmp_path, [])
    with pytest.raises(DefaultCheckpointError):
        _run_trainer(ds, None, tmp_path / "run2", tmp_path)


# ---------------------------------------------------------------------------
# CLI surface: the hardcoded superseded default is gone
# ---------------------------------------------------------------------------


def _parse_book_train(argv: list) -> argparse.Namespace:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="top_command")
    register_parser(sub)
    return top.parse_args(argv)


def test_cli_book_train_resume_default_is_none_not_hardcoded():
    args = _parse_book_train(["book", "train"])
    assert args.resume is None
    assert args.resume != "models/checkpoints_scaled/best_model.pt"
    assert args.allow_superseded_resume is False
    assert args.allow_resume_config_drift is False
    assert args.allow_missing_run_manifest is False


def test_trainer_signature_default_is_not_hardcoded_path():
    """Pin the API-level default too: the trainer resolves the registry
    baseline when ``resume_checkpoint`` is omitted — never a hardcoded path."""
    sig = inspect.signature(train_opening_policy)
    assert sig.parameters["resume_checkpoint"].default is None


def test_cli_book_train_explicit_resume_and_overrides_parse():
    args = _parse_book_train(
        [
            "book",
            "train",
            "--resume",
            "models/checkpoints_opening/best_opening_model.pt",
            "--allow-superseded-resume",
            "--allow-resume-config-drift",
            "--allow-missing-run-manifest",
        ]
    )
    assert args.resume == "models/checkpoints_opening/best_opening_model.pt"
    assert args.allow_superseded_resume is True
    assert args.allow_resume_config_drift is True
    assert args.allow_missing_run_manifest is True
