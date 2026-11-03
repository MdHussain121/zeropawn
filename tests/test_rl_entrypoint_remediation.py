"""Regression tests for the RL entry point, checkpoint safety and provenance.

Findings covered:

- SR-02  runnable RL entry point: no duplicate argparse registration, no
         ``_warm_start_info`` AttributeError on any construction path, data
         gate flags reachable from ``zeropawn train``.
- SR-03  auto-resume eligibility: checkpoint_type evidence, quarantine
         rejection, buffer-association prefilter.
- SR-04  checkpoint<->buffer consistency on resume (fail closed, no pointer
         rewind, explicit override).
- SR-08  telemetry/provenance: named-column CSV with upgrade, aborted games
         never counted as draws, run manifest/summary records.
- SR-09  synthetic end-to-end pipeline: records -> validated insertion ->
         persistence -> restore -> sampling -> batch collation -> training
         step -> checkpoint -> selection -> safe restoration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import python  # noqa: F401

from python.model.chess_net import ChessNet
from python.selfplay.metrics import MetricsTracker
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import GameSummary, SavedPosition, outcomes_from_result
from python.train.train_rl import (
    BufferCheckpointMismatchError,
    RLTrainer,
    inspect_rl_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extension_search_path() -> str:
    """Directory holding the importable zeropawn_cpp extension (for subprocesses)."""
    import zeropawn_cpp

    return str(Path(zeropawn_cpp.__file__).resolve().parent)


def _make_buffer(tmp_path: Path, name: str = "buf", capacity: int = 256, **kw) -> ReplayBuffer:
    return ReplayBuffer(buffer_dir=tmp_path / name, capacity=capacity, create_new=True, **kw)


def _one_hot(slot: int, num_actions: int = 4672) -> np.ndarray:
    v = np.zeros(num_actions, dtype=np.float32)
    v[slot % num_actions] = 1.0
    return v


def _record(n: int, game_id: int = 1, version: int = 1) -> SavedPosition:
    v = np.zeros(4672, dtype=np.float32)
    v[n % 4672] = 1.0
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    planes[0, 0, 0] = float(n % 2)
    return SavedPosition(
        planes=planes,
        visits=v,
        outcome=n % 3,
        is_full_search=bool(n % 2 == 0),
        model_version=version,
        game_id=game_id,
    )


def _make_trainer(tmp_path: Path, name: str, buf: ReplayBuffer | None = None) -> RLTrainer:
    if buf is None:
        buf = _make_buffer(tmp_path, name=f"buf_{name}", capacity=64)
        buf.add_positions([_record(i, game_id=i + 1) for i in range(16)])
    return RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=16),
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
        warmup_steps=5,
        total_steps=100,
        checkpoint_dir=tmp_path / f"ckpt_{name}",
        sync_weights_path=tmp_path / f"sync_{name}.pt",
    )


def _write_ckpt(
    path: Path,
    *,
    step: int,
    optimizer: str | None = "SGD",
    checkpoint_type: str | None = "rl",
    buffer_state: dict | None = None,
    model: ChessNet | None = None,
) -> None:
    """Write a small synthetic checkpoint with controllable provenance."""
    model = model or ChessNet(num_blocks=2, num_filters=16)
    if optimizer == "SGD":
        opt = {
            "state": {0: {"momentum_buffer": torch.zeros(1)}},
            "param_groups": [{"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "params": [0]}],
        }
    elif optimizer == "AdamW":
        opt = {
            "state": {0: {"exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}},
            "param_groups": [{"lr": 0.01, "params": [0]}],
        }
    elif optimizer == "empty":
        opt = {"state": {}, "param_groups": [{"lr": 0.01, "momentum": 0.9, "params": [0]}]}
    else:
        opt = None
    # No num_blocks/num_filters header: the architecture gate in
    # inspect_rl_checkpoint is skipped and the TYPE/quarantine logic under
    # test is what decides eligibility.
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt,
        "step": step,
    }
    if checkpoint_type is not None:
        ckpt["checkpoint_type"] = checkpoint_type
    if buffer_state is not None:
        ckpt["buffer_state"] = buffer_state
    torch.save(ckpt, path)


def _select(path: Path, buffer: ReplayBuffer | None = None, allow_mismatch: bool = False):
    from scripts.run_phase4_rl import Phase4Orchestrator

    fake = types.SimpleNamespace(
        checkpoint_dir=path,
        buffer=buffer,
        allow_buffer_mismatch=allow_mismatch,
    )
    return Phase4Orchestrator._select_resume_checkpoint(fake)


# ---------------------------------------------------------------------------
# SR-02 — entry point
# ---------------------------------------------------------------------------


def test_standalone_help_exits_zero_with_unique_options():
    """Before the fix, the duplicate registration made even --help raise."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _extension_search_path() + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "scripts/run_phase4_rl.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    # Exactly ONE option-definition line for the flag (it legitimately appears
    # in the usage synopsis as well, so count definition lines, not substrings).
    definitions = [
        line
        for line in out.stdout.splitlines()
        if line.startswith("  --allow-unverified-warm-start")
    ]
    assert len(definitions) == 1, definitions
    for flag in ("--data-role", "--allow-uncertified-data", "--allow-buffer-mismatch"):
        assert flag in out.stdout


def test_warm_start_validation_survives_null_registry_entries(tmp_path):
    """Registry records with sha256: null (missing/quarantined artifacts) must
    not crash warm-start validation (AttributeError on None.startswith)."""
    from python.model.checkpoint_registry import (
        WarmStartValidationError,
        validate_warm_start,
    )

    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    weak_model = tmp_path / "warm.pt"
    _write_ckpt(
        weak_model,
        step=10,
        checkpoint_type=None,
        optimizer=None,
        model=ChessNet(num_blocks=12, num_filters=128),
    )
    registry = {
        "version": 1,
        "warm_start_designation": "NONE",
        "checkpoints": [
            {"path": "models/gone.pt", "sha256": None, "status": "missing"},
            {"path": "models/q.pt", "sha256": None, "status": "quarantined"},
        ],
    }
    (repo / "docs" / "checkpoint_registry.json").write_text(json.dumps(registry), encoding="utf-8")

    # Unregistered + no override -> the documented clear error, not a crash.
    with pytest.raises(WarmStartValidationError):
        validate_warm_start(weak_model, repo_root=str(repo))
    # Unregistered + override -> logged override record.
    record = validate_warm_start(weak_model, repo_root=str(repo), allow_unverified=True)
    assert record["status"] == "unverified-override"
    assert record["sha256"]


def test_train_cli_exposes_data_gate_flags_and_is_idempotent():
    from python.cli.commands_train import _ensure_data_gate_arguments, register_parser

    parser = argparse.ArgumentParser(prog="zeropawn")
    sub = parser.add_subparsers(dest="command")
    register_parser(sub)
    # Registering the gate helper again must not raise (duplicate-option guard).
    train_parser = next(a for a in sub.choices.values() if a.prog.endswith("train"))
    _ensure_data_gate_arguments(train_parser)

    args = parser.parse_args(
        [
            "train",
            "--data-role",
            "games_wdl",
            "--allow-uncertified-data",
            "--allow-uncertified-eval",
            "--no-require-certified-eval",
            "--allow-buffer-mismatch",
        ]
    )
    assert args.data_role == "games_wdl"
    assert args.allow_uncertified_data is True
    assert args.allow_uncertified_eval is True
    assert args.require_certified_eval is False
    assert args.allow_buffer_mismatch is True

    defaults = parser.parse_args(["train"])
    assert defaults.data_role is None
    assert defaults.allow_uncertified_data is False
    assert defaults.allow_buffer_mismatch is False


def _tiny_puzzle_csv(tmp_path: Path) -> Path:
    p = tmp_path / "puzzles.csv"
    p.write_text("PuzzleId,FEN,Moves,Rating\n", encoding="utf-8")
    return p


def _construct_orchestrator(tmp_path: Path, **overrides):
    from scripts.run_phase4_rl import Phase4Orchestrator

    kwargs = dict(
        max_hours=0.01,
        eval_interval_sec=60.0,
        num_concurrent_games=2,
        buffer_capacity=512,
        gpu_cache_capacity=0,
        human_cache_size=10,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "run"),
        base_checkpoint=None,  # cold/random-init construction path
        resume_checkpoint=None,
        human_data_dir=None,
        puzzle_csv_path=str(_tiny_puzzle_csv(tmp_path)),
        openings_path=str(tmp_path / "openings.epd"),
        stockfish_path=str(tmp_path / "stockfish.exe"),
        require_certified_eval=False,
        skip_baseline_eval=True,
        tablebase_path=None,
        device="cpu",
        train_device="cpu",
        infer_device="cpu",
        seed=1234,
    )
    kwargs.update(overrides)
    return Phase4Orchestrator(**kwargs)


def _close_orchestrator(orch) -> None:
    """Release orchestrator resources without the heavy final-checkpoint path."""
    try:
        orch.worker.stop()
    finally:
        try:
            orch.metrics_tracker.close()
        finally:
            try:
                orch.buffer.close()
            except Exception:
                pass


def test_orchestrator_constructs_cold_without_attributeerror(tmp_path):
    """SR-02 P0: construction used to raise AttributeError on `_warm_start_info`."""
    orch = _construct_orchestrator(tmp_path)
    try:
        assert orch.trainer.step_num == 0
        assert orch.trainer.run_metadata.get("warm_start") == "random_init"
        assert orch._warm_start_record is None  # attribute consistency
        # Run manifest + shutdown summary (SR-08) are writable immediately.
        orch._write_run_manifest(resuming_from=None)
        manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text("utf-8"))
        assert manifest["run_id"] == orch.run_id
        assert manifest["seed"] == 1234
        assert manifest["buffer"]["buffer_id"] == orch.buffer.buffer_id
        assert manifest["buffer"]["schema_version"] >= 1
        orch._write_run_summary("TEST")
        summary = json.loads((tmp_path / "run" / "run_summary.json").read_text("utf-8"))
        assert summary["run_id"] == orch.run_id
        assert summary["buffer"]["buffer_id"] == orch.buffer.buffer_id
    finally:
        _close_orchestrator(orch)


def test_orchestrator_constructs_warm_start_with_consistent_provenance(tmp_path):
    warm = tmp_path / "warm.pt"
    # The registry/warm-start path enforces the project architecture 12x128.
    _write_ckpt(
        warm,
        step=4321,
        checkpoint_type=None,
        optimizer=None,
        model=ChessNet(num_blocks=12, num_filters=128),
    )

    orch = _construct_orchestrator(
        tmp_path,
        base_checkpoint=str(warm),
        allow_unverified_warm_start=True,
    )
    try:
        record = orch._warm_start_record
        assert record is not None
        assert record["sha256"]
        assert orch.trainer.run_metadata["warm_start_sha256"] == record["sha256"]
        assert orch.trainer.step_num == 0  # RL counters never inherit the source step
        # The inherited counter is provenance only (never pacing progress).
        assert orch.trainer.inherited_step_num == 4321
        assert orch.trainer.warm_start_step == 4321
        assert orch.trainer.run_metadata["warm_start_step"] == 4321
        assert orch.trainer.run_metadata["run_id"] == orch.run_id
    finally:
        _close_orchestrator(orch)


# ---------------------------------------------------------------------------
# SR-03 — auto-resume eligibility
# ---------------------------------------------------------------------------


def test_auto_resume_rejects_non_rl_checkpoint_type(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    _write_ckpt(d / "finetune_ckpt.pt", step=999, checkpoint_type="finetune")
    assert _select(d) is None


def test_auto_resume_rejects_unstamped_empty_optimizer(tmp_path):
    """The exact SR-03 hole: no type stamp AND no SGD evidence."""
    d = tmp_path / "ckpts"
    d.mkdir()
    _write_ckpt(d / "mys.pt", step=500, checkpoint_type=None, optimizer="empty")
    assert _select(d) is None


def test_auto_resume_accepts_stamped_rl_and_unstamped_with_sgd_evidence(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    _write_ckpt(d / "unstamped_sgd.pt", step=100, checkpoint_type=None, optimizer="SGD")
    _write_ckpt(d / "stamped_rl.pt", step=200, checkpoint_type="rl", optimizer="SGD")
    choice = _select(d)
    assert choice is not None and Path(choice).name == "stamped_rl.pt"  # higher step wins

    (d / "stamped_rl.pt").unlink()
    choice = _select(d)
    assert choice is not None and Path(choice).name == "unstamped_sgd.pt"


def test_auto_resume_rejects_quarantined_by_manifest(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    _write_ckpt(d / "phase4_final_checkpoint.pt", step=6000, checkpoint_type="rl")
    (d / "QUARANTINE.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file": "phase4_final_checkpoint.pt",
                        "sha256": "0" * 64,
                        "size_bytes": 1,
                        "reason": "contaminated",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _select(d) is None


def test_auto_resume_healthy_basename_not_blacklisted_by_registry(tmp_path):
    """A healthy latest_checkpoint.pt must not be rejected just because the
    registry knows a quarantined file with the same basename elsewhere."""
    d = tmp_path / "ckpts"
    d.mkdir()
    _write_ckpt(d / "latest_checkpoint.pt", step=42, checkpoint_type="rl")
    choice = _select(d)
    assert choice is not None and Path(choice).name == "latest_checkpoint.pt"


def test_auto_resume_never_scans_quarantined_subdirectory(tmp_path):
    d = tmp_path / "ckpts"
    (d / "quarantined").mkdir(parents=True)
    _write_ckpt(d / "quarantined" / "latest_checkpoint.pt", step=6000, checkpoint_type="rl")
    assert _select(d) is None


def test_auto_resume_rejects_quarantined_by_registry_hash(tmp_path):
    """A quarantined file copied back beside healthy ones stays ineligible."""
    from python.model.checkpoint_registry import compute_sha256

    d = tmp_path / "ckpts"
    d.mkdir()
    quarantined = d / "copied_back.pt"
    _write_ckpt(quarantined, step=6000, checkpoint_type="rl")
    digest = compute_sha256(quarantined)
    size = quarantined.stat().st_size

    import scripts.run_phase4_rl as runner

    index = {
        "names": set(),
        "paths": set(),
        "sha256": {digest, digest[:16]},
        "sizes": {size},
    }
    verdict = runner.quarantine_verdict(quarantined, index)
    assert verdict is not None and "sha256" in verdict
    # A healthy file with a different size is not falsely flagged.
    _write_ckpt(d / "healthy.pt", step=10, checkpoint_type="rl", model=ChessNet(2, 32))
    index2 = {"names": set(), "paths": set(), "sha256": {digest}, "sizes": {size}}
    assert runner.quarantine_verdict(d / "healthy.pt", index2) is None


def test_auto_resume_prefilter_rejects_buffer_mismatch_unless_overridden(tmp_path):
    buf = _make_buffer(tmp_path, name="live", capacity=64)
    buf.add_positions([_record(i, game_id=i + 1) for i in range(4)])
    d = tmp_path / "ckpts"
    d.mkdir()
    _write_ckpt(
        d / "rl.pt",
        step=50,
        checkpoint_type="rl",
        buffer_state={"buffer_id": "some-other-buffer", "capacity": 64},
    )
    assert _select(d, buffer=buf) is None
    choice = _select(d, buffer=buf, allow_mismatch=True)
    assert choice is not None and Path(choice).name == "rl.pt"
    # A matching buffer_id is eligible without any override.
    _write_ckpt(
        d / "rl2.pt",
        step=60,
        checkpoint_type="rl",
        buffer_state={"buffer_id": buf.buffer_id, "capacity": 64},
    )
    choice = _select(d, buffer=buf)
    assert choice is not None and Path(choice).name == "rl2.pt"


# ---------------------------------------------------------------------------
# SR-04 — checkpoint <-> buffer consistency on resume
# ---------------------------------------------------------------------------


def test_resume_with_matching_buffer_state_restores_state(tmp_path):
    buf = _make_buffer(tmp_path, name="buf", capacity=64)
    buf.add_positions([_record(i, game_id=i + 1) for i in range(16)])
    trainer = _make_trainer(tmp_path, "a", buf=buf)
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "c1.pt", keep_last_n=-1)

    # A second open of the SAME directory sees the same buffer id/state.
    buf2 = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64)
    trainer2 = _make_trainer(tmp_path, "b", buf=buf2)
    assert trainer2.load_checkpoint(ckpt) == 1
    assert trainer2.step_num == 1


def test_resume_with_mismatched_buffer_fails_closed_then_override_proceeds(tmp_path):
    src = _make_buffer(tmp_path, name="buf_src", capacity=64)
    src.add_positions([_record(i, game_id=i + 1) for i in range(16)])
    trainer = _make_trainer(tmp_path, "a", buf=src)
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "c1.pt", keep_last_n=-1)

    other = _make_buffer(tmp_path, name="buf_other", capacity=64)
    other.add_positions([_record(i, game_id=i + 50) for i in range(4)])
    trainer2 = _make_trainer(tmp_path, "b", buf=other)

    with pytest.raises(BufferCheckpointMismatchError):
        trainer2.load_checkpoint(ckpt)

    assert trainer2.load_checkpoint(ckpt, allow_buffer_mismatch=True) == 1
    # The live pointer was NOT rewound to the checkpoint's 16 records.
    assert other.size == 4 and other.write_head == 4


def test_resume_never_rewinds_pointer_when_live_buffer_is_ahead(tmp_path):
    buf = _make_buffer(tmp_path, name="buf", capacity=64)
    buf.add_positions([_record(i, game_id=i + 1) for i in range(8)])
    trainer = _make_trainer(tmp_path, "a", buf=buf)
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "c1.pt", keep_last_n=-1)

    buf.add_positions([_record(i + 8, game_id=i + 20) for i in range(6)])  # live moves ahead
    assert buf.size == 14 and buf.write_head == 14

    trainer2 = _make_trainer(tmp_path, "b", buf=buf)
    assert trainer2.load_checkpoint(ckpt) == 1  # 'ahead' is allowed
    assert buf.size == 14 and buf.write_head == 14  # never rewound
    assert buf.total_added == 14


def test_inspect_rl_checkpoint_reports_type_and_buffer_state(tmp_path):
    buf = _make_buffer(tmp_path, name="buf", capacity=64)
    buf.add_positions([_record(i, game_id=i + 1) for i in range(4)])
    trainer = RLTrainer(
        model=ChessNet(num_blocks=12, num_filters=128),  # project architecture gate
        replay_buffer=buf,
        device="cpu",
        batch_size=4,
        warmup_steps=2,
        total_steps=10,
        checkpoint_dir=tmp_path / "ckpt_a",
        sync_weights_path=tmp_path / "sync_a.pt",
    )
    ckpt = trainer.save_checkpoint(tmp_path / "c1.pt", keep_last_n=-1)
    info = inspect_rl_checkpoint(ckpt)
    assert info is not None
    assert info["checkpoint_type"] == "rl"
    assert info["buffer_state"]["buffer_id"] == buf.buffer_id
    assert info["buffer_state"]["size"] == 4
    assert info["buffer_state"]["schema_version"] >= 1


# ---------------------------------------------------------------------------
# SR-08 — telemetry and provenance
# ---------------------------------------------------------------------------


def test_metrics_csv_writes_named_provenance_columns(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    tracker = MetricsTracker(log_dir=tmp_path, csv_path=csv_path)
    try:
        tracker.run_id = "run123"
        tracker.log_step(
            step=1,
            buffer_size=42,
            model_version=7,
            buffer_total_added=99,
            buffer_write_head=42,
            buffer_schema_version=2,
            games_aborted=3,
            games_invalid=1,
            records_rejected=0,
            checkpoint_file="phase4_ckpt.pt",
            sampling={
                "sample_source": "ring+gpu_cache",
                "sampled_selfplay_rows": 200,
                "sampled_unique_games": 17,
                "staleness_fallbacks": 0,
            },
        )
    finally:
        tracker.close()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    header, values = rows
    assert len(header) == len(values)  # widths can never drift
    record = dict(zip(header, values))
    assert record["run_id"] == "run123"
    assert record["games_aborted"] == "3"
    assert record["games_invalid"] == "1"
    assert record["buffer_total_added"] == "99"
    assert record["checkpoint_file"] == "phase4_ckpt.pt"
    assert record["sample_source"] == "ring+gpu_cache"
    assert record["sampled_unique_games"] == "17"


def test_legacy_csv_header_is_upgraded_without_losing_rows(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    legacy_header = (
        "timestamp,step,simulations_per_sec,games_per_hour,gpu_utilization,"
        "batch_fill_rate,draw_rate,resign_rate,false_resign_rate,average_game_length,"
        "policy_entropy,buffer_size,human_mix_ratio,tactical_mix_ratio,model_version,"
        "rss_mb,vram_used_mb"
    )
    csv_path.write_text(legacy_header + "\n1.0,5,10,10,0,0,0.1,0.2,0,80,1.1,10,0.2,0.1,1,100,10\n")
    tracker = MetricsTracker(log_dir=tmp_path, csv_path=csv_path)
    try:
        tracker.log_step(step=6, buffer_size=11, model_version=1)
    finally:
        tracker.close()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3
    assert all(len(r) == len(rows[0]) for r in rows)  # old row padded, widths equal
    assert "run_id" in rows[0] and "games_aborted" in rows[0]
    assert rows[1][1] == "5"  # the legacy data row survived


def test_aborted_games_are_not_counted_as_draws_or_games():
    tracker = MetricsTracker(log_dir=Path("."), csv_path=None)
    try:
        tracker.record_game(
            GameSummary(
                game_id=1,
                length_plies=5,
                result="*",
                resigned=False,
                resigned_side_white=None,
                would_have_resigned=False,
                is_false_resign=False,
                duration_sec=0.1,
                total_simulations=1,
                model_version=1,
                aborted=True,
                termination_reason="aborted",
            )
        )
        assert tracker.total_games == 0
        assert tracker.total_draws == 0
        assert tracker.total_aborted_games == 1
    finally:
        tracker.close()


# ---------------------------------------------------------------------------
# SR-09 / Phase K — synthetic end-to-end integration
# ---------------------------------------------------------------------------


def test_synthetic_end_to_end_pipeline(tmp_path):
    """games -> records -> validated insert -> restore -> sample -> train ->
    checkpoint -> selection -> safe restore, with alignment checked at every hop."""
    buf_dir = tmp_path / "buffer"
    buf = ReplayBuffer(buffer_dir=buf_dir, capacity=128, gpu_cache_capacity=0, create_new=True)

    # 1. Two synthetic "games" produce mover-relative records.
    raw = [
        {
            "planes": np.zeros((19, 8, 8), dtype=np.float32),
            "visits": _one_hot(i),
            "is_white": i % 2 == 0,
            "is_full_search": True,
            "model_version": 3,
        }
        for i in range(6)
    ]
    game1 = outcomes_from_result(raw, white_won=True, game_id=1)
    game2 = outcomes_from_result(raw, white_won=None, game_id=2)
    assert [p.outcome for p in game1] == [0, 2, 0, 2, 0, 2]
    assert all(p.outcome == 1 for p in game2)

    buf.add_positions(game1)
    buf.add_positions(game2)
    assert buf.size == 12 and buf.total_added == 12
    assert (buf.game_ids[:12] == np.array([1] * 6 + [2] * 6, dtype=np.uint64)).all()

    # 2. Persistence + restoration round trip preserves identity and pointers.
    buf.close()
    restored = ReplayBuffer(buffer_dir=buf_dir, capacity=128)
    assert restored.size == 12 and restored.write_head == 12
    assert (restored.game_ids[:12] == np.array([1] * 6 + [2] * 6, dtype=np.uint64)).all()
    assert int(restored.versions[:12].max()) == 3

    # 3. Sampling + collation keep planes/visits/outcome/game_id aligned.
    batch, mix = restored.sample_batch_tensors(batch_size=8, step=0, device="cpu")
    assert batch["planes"].shape == (8, 19, 8, 8)
    assert batch["game_ids"].shape == (8,)
    assert set(int(g) for g in batch["game_ids"]) <= {1, 2}
    for row_outcome in batch["outcomes"].tolist():
        assert row_outcome in (0, 1, 2)

    # 4. A real training step consumes the batch (project architecture so the
    #    checkpoint-selection gate later accepts what we save).
    trainer = RLTrainer(
        model=ChessNet(num_blocks=12, num_filters=128),
        replay_buffer=restored,
        device="cpu",
        batch_size=8,
        warmup_steps=2,
        total_steps=10,
        checkpoint_dir=tmp_path / "ckpt",
        sync_weights_path=tmp_path / "sync.pt",
    )
    metrics = trainer.train_step()
    assert metrics["step"] == 1
    assert metrics["sampled_unique_games"] >= 1
    assert metrics["sampled_rows_with_game_id"] >= 1

    # 5. Checkpoint creation associates the buffer, selection finds it, and
    #    restoration into a fresh trainer on the same buffer succeeds.
    ckpt = trainer.save_checkpoint(tmp_path / "ckpt" / "step1.pt", keep_last_n=-1)
    selected = _select(tmp_path / "ckpt", buffer=restored)
    # save_checkpoint also mirrors the file to latest_checkpoint.pt; both are
    # eligible candidates with identical step/buffer association.
    assert selected is not None
    assert Path(selected).name in ("step1.pt", "latest_checkpoint.pt")

    trainer2 = RLTrainer(
        model=ChessNet(num_blocks=12, num_filters=128),
        replay_buffer=ReplayBuffer(buffer_dir=buf_dir, capacity=128),
        device="cpu",
        batch_size=8,
        warmup_steps=2,
        total_steps=10,
        checkpoint_dir=tmp_path / "ckpt2",
        sync_weights_path=tmp_path / "sync2.pt",
    )
    assert trainer2.load_checkpoint(ckpt) == 1
    assert trainer2.step_num == 1
