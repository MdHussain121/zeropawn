"""Run-status ledger: creation, transitions, serialization, and lineage.

Regression coverage for audit finding CKP-01: the Phase-4 orchestrator stamped a
``run_status`` into the trainer's in-memory dict that never reached any
checkpoint, so a failed or interrupted run could masquerade as a completed one.

The tests here prove the ledger is *inside the artifact*, that terminal states
are final, and that a failed save cannot leave a completed-looking checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from python.model.checkpoint_schema import (
    RUN_STATE_CANCELLED,
    RUN_STATE_COMPLETED,
    RUN_STATE_CREATED,
    RUN_STATE_FAILED,
    RUN_STATE_INTERRUPTED,
    RUN_STATE_RUNNING,
    CheckpointSchemaError,
    RunStateError,
    advance_run_state_inplace,
    build_run_metadata,
    describe_run_status,
    is_completed_checkpoint,
    stamp_run_state,
    validate_run_metadata,
)
from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import RLTrainer


def _fill_buffer(buf: ReplayBuffer, n: int = 24) -> None:
    import numpy as np

    from python.selfplay.worker import SavedPosition

    positions = []
    for i in range(n):
        visits = np.zeros(4672, dtype=np.float32)
        visits[i % 4672] = 1.0
        positions.append(
            SavedPosition(
                planes=np.random.rand(19, 8, 8).astype(np.float32),
                visits=visits,
                outcome=i % 3,
                is_full_search=True,
                model_version=1,
                game_id=i + 1,  # ring records require game_id > 0 (integrated SR-07 contract)
            )
        )
    buf.add_positions(positions)


def _make_trainer(tmp_path: Path, name: str = "a") -> RLTrainer:
    buf = ReplayBuffer(buffer_dir=tmp_path / f"buf_{name}", capacity=128, create_new=True)
    _fill_buffer(buf)
    return RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
        checkpoint_dir=tmp_path / f"ckpt_{name}",
        sync_weights_path=tmp_path / f"sync_{name}.pt",
    )


# ── Ledger construction and transitions ──────────────────────────────────────


def test_new_ledger_starts_created_with_identity():
    ledger = build_run_metadata(
        run_type="rl", git_commit="abc123", seed=42, config={"base_lr": 0.01}
    )
    assert ledger["run_state"] == RUN_STATE_CREATED
    assert ledger["run_id"]
    assert ledger["started_at"] is None
    assert ledger["completed_at"] is None
    assert ledger["git_commit"] == "abc123"
    assert ledger["seed"] == 42
    assert ledger["config"] == {"base_lr": 0.01}
    assert validate_run_metadata(ledger, path="memory") == []


@pytest.mark.parametrize(
    "sequence",
    [
        (RUN_STATE_RUNNING, RUN_STATE_COMPLETED),
        (RUN_STATE_RUNNING, RUN_STATE_FAILED),
        (RUN_STATE_RUNNING, RUN_STATE_INTERRUPTED),
        (RUN_STATE_RUNNING, RUN_STATE_CANCELLED),
        (RUN_STATE_CREATED, RUN_STATE_FAILED),
    ],
)
def test_legal_transitions(sequence):
    ledger = build_run_metadata(run_type="rl")
    for state in sequence:
        ledger = stamp_run_state(ledger, state)
    assert ledger["run_state"] == sequence[-1]
    if sequence[-1] == RUN_STATE_COMPLETED:
        assert ledger["completed_at"] is not None
        assert ledger["started_at"] is not None


def test_terminal_states_are_final():
    for terminal in (
        RUN_STATE_COMPLETED,
        RUN_STATE_FAILED,
        RUN_STATE_INTERRUPTED,
        RUN_STATE_CANCELLED,
    ):
        ledger = stamp_run_state(
            stamp_run_state(build_run_metadata(run_type="rl"), RUN_STATE_RUNNING), terminal
        )
        for later in (RUN_STATE_RUNNING, RUN_STATE_COMPLETED, RUN_STATE_FAILED):
            if later == terminal:
                continue  # re-stamping the same state is a no-op refresh, not a transition
            with pytest.raises(RunStateError, match="Illegal run-state transition"):
                stamp_run_state(ledger, later)


def test_unknown_state_rejected():
    with pytest.raises(RunStateError, match="Unknown run state"):
        stamp_run_state(build_run_metadata(run_type="rl"), "finished")


def test_failure_reason_recorded():
    ledger = stamp_run_state(
        stamp_run_state(build_run_metadata(run_type="rl"), RUN_STATE_RUNNING),
        RUN_STATE_FAILED,
        reason="KeyError: 'momentum'",
    )
    assert ledger["failure_reason"] == "KeyError: 'momentum'"
    assert "reason=KeyError" in describe_run_status(ledger)


def test_in_place_advance_keeps_object_identity():
    ledger = build_run_metadata(run_type="supervised")
    same = advance_run_state_inplace(ledger, RUN_STATE_RUNNING, step=3)
    assert same is ledger
    assert ledger["run_state"] == RUN_STATE_RUNNING


def test_validate_run_metadata_rejects_corrupt_ledger():
    with pytest.raises(CheckpointSchemaError, match="run_state"):
        validate_run_metadata({"run_id": "x", "run_state": "finished"}, path="p.pt")
    with pytest.raises(CheckpointSchemaError, match="run_id"):
        validate_run_metadata({"run_state": "running"}, path="p.pt")
    with pytest.raises(CheckpointSchemaError, match="not a mapping"):
        validate_run_metadata(["nope"], path="p.pt", required=True)
    with pytest.raises(CheckpointSchemaError, match="no run_metadata"):
        validate_run_metadata(None, path="p.pt", required=True)
    # Absent ledger is a warning (legacy artifact) when not required.
    assert validate_run_metadata(None, path="p.pt", required=False)


# ── Serialization into RL artifacts ──────────────────────────────────────────


def test_rl_checkpoint_embeds_the_ledger_and_transitions_to_running(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert trainer.run_metadata["run_state"] == RUN_STATE_CREATED
    trainer.train_step()
    ckpt_path = trainer.save_checkpoint(tmp_path / "step1.pt", keep_last_n=-1)

    saved = torch.load(ckpt_path, weights_only=False)
    ledger = saved["run_metadata"]
    # The ledger is INSIDE the artifact (the CKP-01 defect was that it was not).
    assert ledger["run_state"] == RUN_STATE_RUNNING
    assert ledger["step"] == 1
    assert ledger["run_id"] == trainer.run_metadata["run_id"]
    assert ledger["run_type"] == "rl"
    assert saved["provenance"]["run_id"] == ledger["run_id"]
    assert validate_run_metadata(ledger, path=ckpt_path) == []
    assert is_completed_checkpoint(saved) is False


def test_completed_run_writes_completed_metadata(tmp_path):
    trainer = _make_trainer(tmp_path, "done")
    trainer.train_step()
    advance_run_state_inplace(trainer.run_metadata, RUN_STATE_COMPLETED, step=trainer.step_num)
    ckpt_path = trainer.save_checkpoint(tmp_path / "final.pt", keep_last_n=-1)
    saved = torch.load(ckpt_path, weights_only=False)
    assert is_completed_checkpoint(saved) is True
    assert saved["run_metadata"]["completed_at"] is not None


def test_failed_run_metadata_survives_the_save(tmp_path):
    trainer = _make_trainer(tmp_path, "failed")
    trainer.train_step()
    advance_run_state_inplace(
        trainer.run_metadata, RUN_STATE_FAILED, step=trainer.step_num, reason="division by zero"
    )
    ckpt_path = trainer.save_checkpoint(tmp_path / "failed.pt", keep_last_n=-1)
    saved = torch.load(ckpt_path, weights_only=False)
    assert is_completed_checkpoint(saved) is False
    assert saved["run_metadata"]["run_state"] == RUN_STATE_FAILED
    assert "division by zero" in saved["run_metadata"]["failure_reason"]


def test_interrupted_run_cannot_be_misread_as_completed(tmp_path):
    trainer = _make_trainer(tmp_path, "int")
    trainer.train_step()
    advance_run_state_inplace(
        trainer.run_metadata, RUN_STATE_INTERRUPTED, reason="KeyboardInterrupt"
    )
    ckpt_path = trainer.save_checkpoint(tmp_path / "int.pt", keep_last_n=-1)
    saved = torch.load(ckpt_path, weights_only=False)
    assert is_completed_checkpoint(saved) is False
    # And a *reload* keeps the original state (no reconstruction as completed).
    trainer2 = _make_trainer(tmp_path, "int2")
    trainer2.load_checkpoint(ckpt_path, reset_buffer=True)
    assert trainer2.run_metadata["run_state"] == RUN_STATE_CREATED  # the new run
    assert trainer2.run_metadata["predecessor_run"]["run_state"] == RUN_STATE_INTERRUPTED


def test_resume_creates_a_continuation_relationship(tmp_path):
    trainer = _make_trainer(tmp_path, "r1")
    trainer.train_step()
    advance_run_state_inplace(trainer.run_metadata, RUN_STATE_COMPLETED, step=trainer.step_num)
    first = trainer.save_checkpoint(tmp_path / "first.pt", keep_last_n=-1)
    first_ledger = torch.load(first, weights_only=False)["run_metadata"]

    trainer2 = _make_trainer(tmp_path, "r2")
    trainer2.load_checkpoint(first, reset_buffer=True)
    ledger2 = trainer2.run_metadata
    assert ledger2["run_id"] != first_ledger["run_id"]
    assert ledger2["run_type"] == "rl-resume"
    assert ledger2["parent_checkpoint"]["path"].endswith("first.pt")
    assert ledger2["parent_checkpoint"]["sha256"]
    assert ledger2["predecessor_run"]["run_id"] == first_ledger["run_id"]
    assert ledger2["predecessor_run"]["run_state"] == RUN_STATE_COMPLETED

    trainer2.train_step()
    second = trainer2.save_checkpoint(tmp_path / "second.pt", keep_last_n=-1)
    saved2 = torch.load(second, weights_only=False)
    assert saved2["provenance"]["parent_checkpoint"]["path"].endswith("first.pt")


def test_warm_start_ledger_names_the_source_as_parent(tmp_path):
    trainer = _make_trainer(tmp_path, "ws_src")
    trainer.train_step()
    source = trainer.save_checkpoint(tmp_path / "supervised_shaped.pt", keep_last_n=-1)

    trainer2 = _make_trainer(tmp_path, "ws_dst")
    trainer2.load_checkpoint(source, warm_start=True)
    assert trainer2.run_metadata["run_type"] == "rl-warm-start"
    assert trainer2.run_metadata["warm_start"].endswith("supervised_shaped.pt")
    assert trainer2.run_metadata["parent_checkpoint"]["path"].endswith("supervised_shaped.pt")
    assert trainer2.run_metadata["run_state"] == RUN_STATE_CREATED


def test_legacy_checkpoint_without_ledger_loads_with_a_warning(tmp_path, caplog):
    trainer = _make_trainer(tmp_path, "legacy")
    trainer.train_step()
    path = trainer.save_checkpoint(tmp_path / "legacy.pt", keep_last_n=-1)
    payload = torch.load(path, weights_only=False)
    payload.pop("run_metadata")
    payload.pop("checkpoint_format")
    torch.save(payload, path)

    fresh = _make_trainer(tmp_path, "legacy2")
    with caplog.at_level("WARNING", logger="python.train.train_rl"):
        fresh.load_checkpoint(path, reset_buffer=True)
    assert any("run_metadata" in rec.message or "legacy" in rec.message for rec in caplog.records)
    # The new run has its own ledger and records the predecessor as unknown-status.
    assert fresh.run_metadata["run_state"] == RUN_STATE_CREATED
    assert fresh.run_metadata.get("predecessor_run") is None


# ── Supervised ledger ────────────────────────────────────────────────────────


def test_supervised_checkpoint_write_carries_a_ledger(tmp_path):
    from python.train.train_supervised import WarmupCosineScheduler, save_checkpoint

    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1, total_steps=10, base_lr=1e-4)
    ledger = build_run_metadata(run_type="supervised", seed=7)
    advance_run_state_inplace(ledger, RUN_STATE_RUNNING, step=1)

    path = tmp_path / "sup.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        step=1,
        epoch=0,
        best_val_loss=1.0,
        num_blocks=2,
        num_filters=32,
        run_metadata=ledger,
    )
    saved = torch.load(path, weights_only=False)
    assert saved["run_metadata"]["run_state"] == RUN_STATE_RUNNING
    assert saved["run_metadata"]["run_id"] == ledger["run_id"]
    assert saved["checkpoint_format"] == 1
    assert saved["provenance"]["run_type"] == "supervised"


class _FakeGate:
    role = None
    certified = True
    masked_fraction = 0.0
    num_samples = 8
    warnings: list[str] = []
    receipt: dict | None = None


def _fake_gate(*args, **kwargs):
    return _FakeGate()


class _StubDataset:
    """Minimal supervised dataset yielding well-formed tiny batches."""

    def __init__(self, length: int = 8) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        value_target = torch.zeros(3, dtype=torch.float32)
        value_target[idx % 3] = 1.0
        return {
            "planes": torch.randn(19, 8, 8, dtype=torch.float32),
            "move": torch.tensor(idx % 4672, dtype=torch.long),
            "value_target": value_target,
            "ply": torch.tensor(10 + (idx % 5), dtype=torch.long),
        }


class _StubLoader:
    def __init__(self, dataset: _StubDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self._len = max(1, len(dataset) // max(1, batch_size))

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        batch = {
            key: torch.stack([self.dataset[i][key] for i in range(self.batch_size)])
            for key in ("planes", "move", "value_target", "ply")
        }
        for _ in range(self._len):
            yield batch

    def close(self) -> None:  # pragma: no cover - exercised via finally
        pass


def test_supervised_final_save_failure_is_loud_and_marks_the_run_failed(tmp_path, monkeypatch):
    """CKP-07: a failed final save must not look like a successful completion."""
    from python.train import train_supervised as ts

    telemetry_events: list[dict] = []

    class RecordingTelemetry:
        def __init__(self, ckpt_dir, manifest):
            self.manifest = manifest

        def log(self, record):
            telemetry_events.append(record)

        def checkpoint_event(self, path, reason, step):
            telemetry_events.append({"event": "checkpoint_saved", "reason": reason})

        def finalize(self, status, extra=None):
            telemetry_events.append({"event": "run_finished", "status": status})

    real_save = ts.save_checkpoint

    def flaky_save(*args, **kwargs):
        target = Path(args[0]).name if args else ""
        if target == "latest_checkpoint.pt":
            raise OSError("disk full")
        return real_save(*args, **kwargs)

    import python.data.dataset_validation as dv

    monkeypatch.setattr(dv, "require_validated_dataset", _fake_gate)
    monkeypatch.setattr(ts, "save_checkpoint", flaky_save)
    monkeypatch.setattr(ts, "_RunTelemetry", RecordingTelemetry)
    monkeypatch.setattr(ts, "BinaryChessDataset", lambda *a, **k: _StubDataset())
    monkeypatch.setattr(ts, "GpuResidentDataLoader", lambda *a, **k: _StubLoader(_StubDataset(), 2))
    monkeypatch.setattr(ts, "SourceAwareCombinedDataset", None)

    def _stub_dataloader(dataset, batch_size=1, **kwargs):
        if isinstance(dataset, _StubDataset):
            return _StubLoader(dataset, batch_size)
        return _StubLoader(_StubDataset(), batch_size)

    monkeypatch.setattr(ts, "DataLoader", _stub_dataloader)

    with pytest.raises(RuntimeError, match="final checkpoint could not be saved"):
        ts.train_supervised(
            train_dir=tmp_path / "train",
            val_dir=tmp_path / "val",
            epochs=1,
            max_steps=1,
            batch_size=2,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            checkpoint_dir=tmp_path / "ckpt",
            log_dir=tmp_path / "log",
            num_workers=0,
            gpu_resident=False,
            num_blocks=2,
            num_filters=32,
        )
    assert any(e.get("event") == "checkpoint_saved_failed" for e in telemetry_events), (
        f"the failed persistence must be recorded: {telemetry_events}"
    )
    assert any(
        e.get("event") == "run_finished" and e.get("status") == "failed" for e in telemetry_events
    ), f"the run must be finalized as failed: {telemetry_events}"
    assert not (tmp_path / "ckpt" / "latest_checkpoint.pt").exists()
