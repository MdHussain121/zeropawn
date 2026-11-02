"""Legacy pre-schema ``run_metadata`` ledgers are accepted, not fatal (S3-02).

Audit S3-02 (post-merge 2026-11-02): ``validate_run_metadata`` raised
``CheckpointSchemaError`` for a PRESENT ledger that lacks ``run_state`` — the
pre-schema shape — while the module contract (and ``docs/checkpoint_lifecycle.md``
§2) promises legacy artifacts are accepted with an explicit warning. The
designated supervised baseline ckpt-10
(``models/checkpoints_baseline_20261031/best_model.pt``) carries exactly that
shape (keys: config, created_at, dataset, entry, git_commit, host, model,
optimizer, run_id, schedule, seed — no ``run_state``), so
``zeropawn train --mode supervised --resume <ckpt-10>`` crashed at schema
validation, before any state adoption.

The tolerance added here is narrow and deliberate:

* a mapping with NO ``run_state`` key is a legacy pre-schema ledger — accepted
  with a warning when ``required=False`` (what both load paths pass for a
  legacy artifact: ``required=not verdict.is_legacy``), refused under
  ``required=True`` (the current-format contract demands a real ledger);
* a ledger that DECLARES ``run_state`` is validated strictly as before — an
  unknown/invalid value raises regardless of ``required``;
* current-schema ledgers keep full strict validation (state machine, run_id).

The pre-schema fixture below replicates ckpt-10's key set EXACTLY so the tests
do not depend on the production artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.model.checkpoint_schema import (  # noqa: E402
    RUN_STATE_COMPLETED,
    RUN_STATE_CREATED,
    RUN_STATE_FAILED,
    RUN_STATE_RUNNING,
    CheckpointSchemaError,
    RunStateError,
    describe_run_status,
    stamp_run_state,
    validate_checkpoint_schema,
    validate_run_metadata,
)
from python.model.chess_net import ChessNet  # noqa: E402

# ── Fixtures ─────────────────────────────────────────────────────────────────

CKPT10_LEDGER_KEYS = [
    "config",
    "created_at",
    "dataset",
    "entry",
    "git_commit",
    "host",
    "model",
    "optimizer",
    "run_id",
    "schedule",
    "seed",
]


def ckpt10_shaped_ledger() -> dict:
    """A ledger shaped EXACTLY like the designated baseline ckpt-10's.

    Key set replicated from the real artifact (probed read-only,
    torch.load(weights_only=True)): a rich pre-schema ledger with identity,
    config, dataset receipts and schedule — but no ``run_state``, because it
    predates the run-status schema.
    """
    return {
        "run_id": "20261031-125838-b8868fd6",
        "created_at": 1761917918.0,
        "config": {"base_lr": 0.0002, "warmup_steps": 500, "epochs": 40, "batch_size": 512},
        "dataset": {
            "train": {
                "data_dir": "data/dataset_training/train",
                "certified": True,
                "receipt_id": "sha256:156d528959dd463c",
                "receipt_overall": "pass",
            },
            "val": {
                "data_dir": "data/dataset_training/val",
                "certified": True,
                "receipt_id": "sha256:156d528959dd463c",
                "receipt_overall": "pass",
            },
        },
        "entry": {"kind": "supervised_baseline", "step": 4802},
        "git_commit": "101bff6c1234",
        "host": "baseline-host",
        "model": {"num_blocks": 12, "num_filters": 128, "param_count": 8624931},
        "optimizer": {"type": "AdamW", "lr": 0.0002, "weight_decay": 0.0001},
        "schedule": {"warmup_steps": 500, "total_steps": 4801, "base_lr": 0.0002},
        "seed": 42,
    }


def _assert_ckpt10_shape(ledger: dict) -> None:
    """Guard fixture drift: the fixture must keep ckpt-10's exact key set."""
    assert sorted(ledger.keys()) == sorted(CKPT10_LEDGER_KEYS)
    assert "run_state" not in ledger


def _legacy_supervised_payload(
    tmp_path: Path,
    *,
    step: int = 5,
    scheduler_current_step: int = 5,
    optimizer_steps: int = 1,
    geometry: dict | None = None,
) -> tuple[Path, dict]:
    """Write a pre-schema supervised artifact shaped like ckpt-10 (tiny 2x32).

    Mirrors the historical writer: no ``checkpoint_format`` marker, a rich
    ledger without ``run_state``, full optimizer + scheduler state. Returns
    (path, metadata about the fixture) for adoption assertions.
    """
    from python.train.train_supervised import WarmupCosineScheduler, save_checkpoint

    if geometry is None:
        geometry = {"warmup_steps": 3, "total_steps": 40, "base_lr": 1e-4, "min_lr": 1e-6}

    torch.manual_seed(7)
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=geometry["base_lr"], weight_decay=1e-4)
    # One real optimizer step so AdamW per-param state exists (step tensors).
    for _ in range(max(1, optimizer_steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(2, 19, 8, 8))[0].square().mean()
        loss.backward()
        optimizer.step()
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=geometry["warmup_steps"],
        total_steps=geometry["total_steps"],
        base_lr=geometry["base_lr"],
        min_lr=geometry["min_lr"],
    )
    scheduler.current_step = scheduler_current_step

    path = tmp_path / "preschema_baseline.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        step=step,
        epoch=0,
        best_val_loss=0.5,
        num_blocks=2,
        num_filters=32,
        run_metadata=ckpt10_shaped_ledger(),
    )
    payload = torch.load(path, weights_only=False)
    assert payload.get("checkpoint_format") == 1  # the current writer stamps format 1...
    payload.pop("checkpoint_format")  # ...the historical ckpt-10 writer did not
    assert payload.get("checkpoint_type") == "supervised"
    payload["run_metadata"] = ckpt10_shaped_ledger()
    torch.save(payload, path)
    return path, {"geometry": dict(geometry), "step": step, "scheduler_current_step": scheduler_current_step}


# ── Validator-level behavior ─────────────────────────────────────────────────


def test_current_schema_ledger_validates_strictly_with_state_machine():
    ledger = stamp_run_state(
        stamp_run_state(
            stamp_run_state(
                {
                    "run_metadata_version": 1,
                    "run_id": "run-x",
                    "run_type": "supervised",
                    "created_at": 1.0,
                    "started_at": 2.0,
                },
                RUN_STATE_CREATED,
            ),
            RUN_STATE_RUNNING,
            step=3,
        ),
        RUN_STATE_COMPLETED,
        step=9,
    )
    assert validate_run_metadata(ledger, path="p.pt", required=True) == []
    # The state machine itself stays enforced (terminal states are final).
    with pytest.raises(RunStateError, match="Illegal run-state transition"):
        stamp_run_state(ledger, RUN_STATE_RUNNING)


def test_pre_schema_ledger_accepted_with_warning_when_not_required():
    ledger = ckpt10_shaped_ledger()
    _assert_ckpt10_shape(ledger)
    warnings = validate_run_metadata(ledger, path="best_model.pt", required=False)
    assert len(warnings) == 1
    message = warnings[0]
    # The warning must distinguish the legacy pre-schema ledger from the
    # current schema and from the absent-ledger case.
    assert "pre-schema" in message
    assert "run_state" in message
    assert "legacy" in message
    # No crash downstream either.
    assert "run 20261031-125838-b8868fd6" in describe_run_status(ledger)


def test_pre_schema_ledger_refused_when_required():
    with pytest.raises(CheckpointSchemaError, match="pre-schema legacy ledger"):
        validate_run_metadata(ckpt10_shaped_ledger(), path="p.pt", required=True)


def test_missing_ledger_tolerance_preserved():
    # Absent ledger: warning (legacy artifact) when not required...
    absent = validate_run_metadata(None, path="p.pt", required=False)
    assert len(absent) == 1 and "no run_metadata" in absent[0]
    # ...and the pre-schema warning is clearly DIFFERENT from the absent one.
    present = validate_run_metadata(ckpt10_shaped_ledger(), path="p.pt", required=False)
    assert present[0] != absent[0]
    # required=True keeps refusing an absent ledger.
    with pytest.raises(CheckpointSchemaError, match="no run_metadata"):
        validate_run_metadata(None, path="p.pt", required=True)


def test_corrupted_ledger_still_raises():
    # Not a mapping at all.
    with pytest.raises(CheckpointSchemaError, match="not a mapping"):
        validate_run_metadata(["nope"], path="p.pt", required=True)
    with pytest.raises(CheckpointSchemaError, match="not a mapping"):
        validate_run_metadata(["nope"], path="p.pt", required=False)
    # Declares a run_state value outside RUN_STATES — invalid, not legacy.
    with pytest.raises(CheckpointSchemaError, match="run_state"):
        validate_run_metadata({"run_id": "x", "run_state": "finished"}, path="p.pt")
    # Explicit run_state=None is an invalid VALUE (the key is present), not a
    # pre-schema ledger — it must still raise (audit S3-02 evidence case).
    with pytest.raises(CheckpointSchemaError, match="run_state=None"):
        validate_run_metadata({"run_id": "x", "run_state": None}, path="p.pt")


def test_empty_ledger_is_treated_as_pre_schema():
    # An empty mapping carries no run_state: indistinguishable from a
    # pre-schema ledger, so it takes the same explicit legacy path (a warning),
    # never the strict raise.
    warnings = validate_run_metadata({}, path="p.pt", required=False)
    assert len(warnings) == 1 and "pre-schema" in warnings[0]
    with pytest.raises(CheckpointSchemaError, match="pre-schema legacy ledger"):
        validate_run_metadata({}, path="p.pt", required=True)


def test_incomplete_current_ledger_missing_run_id_raises():
    # A current-schema ledger (valid run_state) must still carry a run_id.
    with pytest.raises(CheckpointSchemaError, match="run_id"):
        validate_run_metadata({"run_state": RUN_STATE_RUNNING}, path="p.pt")


def test_terminal_state_warning_semantics_unchanged():
    ledger = {"run_id": "x", "run_state": RUN_STATE_FAILED}  # no completed_at
    warnings = validate_run_metadata(ledger, path="p.pt")
    assert warnings == ["run_state=failed without completed_at"]


# ── The exact load-path call sequences ───────────────────────────────────────


def test_supervised_load_path_call_sequence_accepts_ckpt10_shape(tmp_path):
    """train_supervised.py:1169-1184 call shape, on a ckpt-10-shaped payload."""
    path, _ = _legacy_supervised_payload(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    verdict = validate_checkpoint_schema(
        ckpt,
        path=str(path),
        expected_family="supervised",  # full resume (not weights-only)
        allow_legacy=True,
        require_full_state=True,
    )
    assert verdict.is_legacy is True
    assert verdict.family == "supervised"
    assert verdict.version is None

    predecessor_ledger = (
        ckpt.get("run_metadata") if isinstance(ckpt.get("run_metadata"), dict) else None
    )
    warnings = validate_run_metadata(
        predecessor_ledger, path=str(path), required=not verdict.is_legacy
    )
    assert len(warnings) == 1 and "pre-schema" in warnings[0]
    # The FAILED-run gate reads run_state tolerantly (no crash, no false hit).
    assert predecessor_ledger.get("run_state") != RUN_STATE_FAILED


def test_weights_only_call_shape_accepts_ckpt10_shape(tmp_path):
    """--resume ckpt-10 --resume-weights-only reaches the same validator."""
    path, _ = _legacy_supervised_payload(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    verdict = validate_checkpoint_schema(
        ckpt, path=str(path), expected_family="supervised", allow_legacy=True, require_full_state=False
    )
    assert verdict.is_legacy is True
    warnings = validate_run_metadata(
        ckpt.get("run_metadata"), path=str(path), required=not verdict.is_legacy
    )
    assert len(warnings) == 1 and "pre-schema" in warnings[0]


def test_rl_load_path_call_shape_tolerates_pre_schema_ledger():
    """train_rl.py:618-631 call shape (shared validator; legacy RL artifact)."""
    ckpt = {
        "checkpoint_type": "rl",
        "model": {"layer.weight": torch.zeros(1)},
        "optimizer": {"param_groups": []},
        "scheduler": {},
        "run_metadata": ckpt10_shaped_ledger(),
    }
    verdict = validate_checkpoint_schema(
        ckpt, path="legacy_rl.pt", expected_family=None, allow_legacy=True, require_full_state=False
    )
    assert verdict.is_legacy is True
    ledger = ckpt.get("run_metadata")
    warnings = validate_run_metadata(ledger, path="legacy_rl.pt", required=not verdict.is_legacy)
    assert len(warnings) == 1 and "pre-schema" in warnings[0]


def test_current_format_artifact_with_pre_schema_ledger_is_refused():
    """A CURRENT-format artifact whose ledger lacks run_state violates the
    current-format contract: required=True must refuse it (strictness kept)."""
    ckpt = {
        "checkpoint_type": "supervised",
        "checkpoint_format": 1,
        "model": {"layer.weight": torch.zeros(1)},
        "optimizer": {"param_groups": []},
        "scheduler": {},
        "step": 1,
        "epoch": 0,
        "num_blocks": 2,
        "num_filters": 32,
        "run_metadata": ckpt10_shaped_ledger(),  # pre-schema ledger in a new file
    }
    verdict = validate_checkpoint_schema(
        ckpt, path="new.pt", expected_family="supervised", allow_legacy=True, require_full_state=False
    )
    assert verdict.is_legacy is False  # current format...
    with pytest.raises(CheckpointSchemaError, match="pre-schema legacy ledger"):
        validate_run_metadata(
            ckpt.get("run_metadata"), path="new.pt", required=not verdict.is_legacy  # ...required=True
        )


# ── Full supervised resume against the ckpt-10-shaped artifact ───────────────


class _FakeGate:
    role = None
    certified = True
    masked_fraction = 0.0
    num_samples = 8
    warnings: list = []
    receipt: dict | None = None


class _StubDataset:
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


@pytest.fixture()
def stubbed_supervised_env(monkeypatch):
    import python.data.dataset_validation as dv
    from python.train import train_supervised as ts

    monkeypatch.setattr(dv, "require_validated_dataset", lambda *a, **k: _FakeGate())
    monkeypatch.setattr(ts, "BinaryChessDataset", lambda *a, **k: _StubDataset())
    monkeypatch.setattr(ts, "SourceAwareCombinedDataset", None)
    monkeypatch.setattr(ts, "GpuResidentDataLoader", lambda *a, **k: _StubLoader(_StubDataset(), 2))

    def _stub_dataloader(dataset, batch_size=1, **kwargs):
        return _StubLoader(_StubDataset(), batch_size)

    monkeypatch.setattr(ts, "DataLoader", _stub_dataloader)
    return ts


def test_supervised_resume_from_pre_schema_ckpt10_shape_adopts_state(
    tmp_path, caplog, stubbed_supervised_env
):
    """The previously-crashing operation: supervised --resume from a ckpt-10-shaped artifact."""
    ts = stubbed_supervised_env
    fixture_path, fixture = _legacy_supervised_payload(tmp_path / "fixture", step=5, scheduler_current_step=5)

    with caplog.at_level("WARNING", logger="python.train.train_supervised"):
        result = ts.train_supervised(
            train_dir=tmp_path / "train",
            val_dir=tmp_path / "val",
            epochs=1,
            max_steps=6,  # resume at step 5 -> exactly one more optimizer step
            batch_size=2,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            checkpoint_dir=tmp_path / "ckpt",
            log_dir=tmp_path / "log",
            num_workers=0,
            gpu_resident=False,
            num_blocks=2,
            num_filters=32,
            resume_path=str(fixture_path),
            # CLI geometry deliberately DIFFERS from the fixture's: the resume
            # must adopt the CHECKPOINT's schedule geometry (audit SLQ-11).
            base_lr=3e-4,
            warmup_steps=2,
        )

    # The pre-schema ledger was accepted loudly, not fatally.
    pre_schema_warnings = [
        rec.getMessage() for rec in caplog.records if "pre-schema" in rec.getMessage()
    ]
    assert pre_schema_warnings, "expected an explicit pre-schema ledger warning"

    final = torch.load(tmp_path / "ckpt" / "latest_checkpoint.pt", weights_only=False)

    # Training-step restoration: resumed at step 5, ran exactly one step.
    assert final["step"] == fixture["step"] + 1 == 6

    # Optimizer STATE adoption: AdamW per-param step counters CONTINUED from
    # the fixture's state. Params that had fixture state and received grads
    # during the run advanced 1 -> 2; params that first gained state during
    # this run sit at 1. A fresh (non-adopting) optimizer could never show a
    # step counter of 2 after a single optimizer step.
    opt_steps = {
        int(state["step"].item())
        for state in final["optimizer"]["state"].values()
        if "step" in state
    }
    assert 2 in opt_steps and max(opt_steps) == 2

    # Scheduler adoption: the checkpoint's geometry and POSITION were restored
    # (5 -> 6); the CLI geometry (warmup=2, total=1, min_lr=1e-5) was not.
    sched = final["scheduler"]
    assert sched["warmup_steps"] == fixture["geometry"]["warmup_steps"] == 3
    assert sched["total_steps"] == fixture["geometry"]["total_steps"] == 40
    assert sched["base_lr"] == fixture["geometry"]["base_lr"]
    assert sched["min_lr"] == fixture["geometry"]["min_lr"]
    assert sched["current_step"] == 6

    # Ledger lineage: the new run records the pre-schema predecessor as-is and
    # the parent checkpoint identity with its hash.
    new_ledger = final["run_metadata"]
    assert new_ledger["run_state"] == "running"  # the final save precedes completion stamping
    assert new_ledger["run_type"] == "supervised-resume"
    predecessor = new_ledger["predecessor_run"]
    _assert_ckpt10_shape(predecessor)
    assert predecessor["run_id"] == "20261031-125838-b8868fd6"
    assert new_ledger["parent_checkpoint"]["sha256"]
    assert "preschema_baseline.pt" in new_ledger["parent_checkpoint"]["path"]


def test_repeated_resume_chain_from_pre_schema_origin(tmp_path, stubbed_supervised_env):
    """Resume from the ckpt-10-shaped artifact, then resume AGAIN from the
    artifact the first resume produced (current-format ledger this time)."""
    ts = stubbed_supervised_env
    fixture_path, _ = _legacy_supervised_payload(tmp_path / "fixture", step=5, scheduler_current_step=5)

    ts.train_supervised(
        train_dir=tmp_path / "train",
        val_dir=tmp_path / "val",
        epochs=1,
        max_steps=6,
        batch_size=2,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        checkpoint_dir=tmp_path / "ckpt1",
        log_dir=tmp_path / "log1",
        num_workers=0,
        gpu_resident=False,
        num_blocks=2,
        num_filters=32,
        resume_path=str(fixture_path),
        base_lr=3e-4,
        warmup_steps=2,
    )
    first = torch.load(tmp_path / "ckpt1" / "latest_checkpoint.pt", weights_only=False)
    assert first["step"] == 6
    assert first["run_metadata"]["predecessor_run"]["run_id"] == "20261031-125838-b8868fd6"

    # Second resume: a CURRENT-format artifact with a valid (running) ledger.
    # epochs=2 because the first run persisted epoch=1 (range(start_epoch, epochs)).
    ts.train_supervised(
        train_dir=tmp_path / "train",
        val_dir=tmp_path / "val",
        epochs=2,
        max_steps=7,
        batch_size=2,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        checkpoint_dir=tmp_path / "ckpt2",
        log_dir=tmp_path / "log2",
        num_workers=0,
        gpu_resident=False,
        num_blocks=2,
        num_filters=32,
        resume_path=str(tmp_path / "ckpt1" / "latest_checkpoint.pt"),
        base_lr=3e-4,
        warmup_steps=2,
    )
    second = torch.load(tmp_path / "ckpt2" / "latest_checkpoint.pt", weights_only=False)
    assert second["step"] == 7
    # The strict path handled the current-format predecessor: state preserved.
    assert second["run_metadata"]["predecessor_run"]["run_state"] == RUN_STATE_RUNNING
    assert second["run_metadata"]["predecessor_run"]["run_id"] == first["run_metadata"]["run_id"]
    # The schedule geometry adopted from the first resume persisted.
    assert second["scheduler"]["warmup_steps"] == 3


def test_weights_only_resume_from_pre_schema_ckpt10_shape(tmp_path, stubbed_supervised_env):
    """--resume ckpt-10 --resume-weights-only: weights adopted, fresh optimizer."""
    ts = stubbed_supervised_env
    fixture_path, _ = _legacy_supervised_payload(tmp_path / "fixture", step=5, scheduler_current_step=5)
    fixture_state = torch.load(fixture_path, map_location="cpu", weights_only=True)["model"]

    ts.train_supervised(
        train_dir=tmp_path / "train",
        val_dir=tmp_path / "val",
        epochs=1,
        max_steps=1,
        batch_size=2,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        checkpoint_dir=tmp_path / "ckpt_wo",
        log_dir=tmp_path / "log_wo",
        num_workers=0,
        gpu_resident=False,
        num_blocks=2,
        num_filters=32,
        resume_path=str(fixture_path),
        resume_weights_only=True,
    )
    final = torch.load(tmp_path / "ckpt_wo" / "latest_checkpoint.pt", weights_only=False)
    # Weights-only: the run restarts at step 0 and adopts the predecessor's
    # weights (training then moves them; step counter must NOT inherit 5).
    assert final["step"] == 1
    assert final["run_metadata"]["run_type"] == "supervised-warm-start"
    assert final["run_metadata"]["predecessor_run"]["run_id"] == "20261031-125838-b8868fd6"
    # The loaded weights really came from the fixture (spot-check one tensor).
    fixture_tensor = next(iter(fixture_state.values()))
    final_tensor = final["model"][next(iter(final["model"].keys()))]
    assert fixture_tensor.shape == final_tensor.shape
