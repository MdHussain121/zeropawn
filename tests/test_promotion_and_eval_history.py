"""Atomic best_eval promotion, promotion metadata association, and eval history.

Regression coverage for audit findings CKP-03 (``best_eval.pt`` promotion used a
bare ``shutil.copyfile`` — the only protected-set writer without atomic
replacement) and CKP-15 (the promotion path never left a durable trace; the
phase-4 eval/telemetry CSVs on disk were header-only), plus SUS-01 (telemetry
CSV appends had no flush/fsync).

The orchestrator methods under test are exercised on a bare instance
(``__new__`` + attribute injection): they only touch the attributes injected
here, so no self-play workers, GPU work, or production artifacts are involved.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import torch

from python.model.chess_net import ChessNet
from scripts.run_phase4_rl import Phase4Orchestrator


def _candidate(path: Path, step: int = 5) -> Path:
    torch.save(
        {
            "checkpoint_type": "rl",
            "model": ChessNet(num_blocks=2, num_filters=32).state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": step,
        },
        path,
    )
    return path


class _TrainerStub:
    step_num = 5


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def __call__(self, kind: str, detail: str) -> None:
        self.events.append((kind, detail))


def _bare_orchestrator(tmp_path: Path) -> Phase4Orchestrator:
    orch = Phase4Orchestrator.__new__(Phase4Orchestrator)
    orch.checkpoint_dir = tmp_path / "ckpts"
    orch.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    orch._best_eval_state_path = orch.checkpoint_dir / "best_eval.json"
    orch._best_eval_comp_score = None
    orch.trainer = _TrainerStub()
    orch.artifact_ticket = None
    recorder = _EventRecorder()
    orch._event = recorder  # type: ignore[method-assign]
    orch._events = recorder.events  # convenience for assertions
    return orch


def _match_res(score: float = 0.6, games: int = 200) -> dict:
    return {
        "engine_score": score,
        "engine_wins": 60,
        "engine_draws": 40,
        "engine_losses": 100,
        "draw_rate": 0.2,
        "total_games": games,
    }


# ── Atomic promotion (CKP-03) ────────────────────────────────────────────────


def test_promotion_is_atomic_and_records_the_generation(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    candidate = _candidate(tmp_path / "cand.pt")

    promoted, generation, detail = orch._promote_best_eval(
        candidate,
        ckpt_name="cand.pt",
        comp_score=0.55,
        match_res=_match_res(),
        solve_rate=0.4,
        gate_reason="200 games, engine score 0.600",
    )
    assert promoted, detail
    assert generation.startswith("promo-")

    best_eval = orch.checkpoint_dir / "best_eval.pt"
    record = json.loads(orch._best_eval_state_path.read_text(encoding="utf-8"))
    # The artifact and its metadata describe the same bytes and the same event.
    assert best_eval.is_file()
    from python.model.checkpoint_registry import compute_sha256

    assert record["checkpoint_sha256"] == compute_sha256(best_eval)
    assert record["generation"] == generation
    assert record["comp_score"] == 0.55
    assert record["reason"]
    assert not list(orch.checkpoint_dir.glob("*.tmp")), "no tmp residue after success"


def test_promotion_keeps_previous_artifact_on_copy_failure(tmp_path, monkeypatch):
    orch = _bare_orchestrator(tmp_path)
    first = _candidate(tmp_path / "first.pt", step=1)
    assert orch._promote_best_eval(
        first, ckpt_name="first.pt", comp_score=0.5, match_res=_match_res(), solve_rate=0.4, gate_reason="g1"
    )[0]
    good_bytes = (orch.checkpoint_dir / "best_eval.pt").read_bytes()
    good_record = orch._best_eval_state_path.read_text(encoding="utf-8")

    second = _candidate(tmp_path / "second.pt", step=2)

    def broken_copy(src, dst):
        raise OSError("simulated I/O failure mid-copy")

    import shutil as _shutil

    monkeypatch.setattr(_shutil, "copyfileobj", broken_copy)
    orch._promote_best_eval(
        second, ckpt_name="second.pt", comp_score=0.9, match_res=_match_res(), solve_rate=0.5, gate_reason="g2"
    )
    # The previous valid promotion is untouched, and its metadata still matches.
    assert (orch.checkpoint_dir / "best_eval.pt").read_bytes() == good_bytes
    assert orch._best_eval_state_path.read_text(encoding="utf-8") == good_record
    assert orch._best_eval_comp_score == 0.5
    assert not list(orch.checkpoint_dir.glob("*.tmp"))


def test_promotion_reports_verification_failure_and_leaves_state_intact(tmp_path, monkeypatch):
    orch = _bare_orchestrator(tmp_path)
    candidate = _candidate(tmp_path / "cand2.pt")

    import scripts.run_phase4_rl as p4
    from python.model import checkpoint_registry as cr

    real_compute = cr.compute_sha256
    calls = {"n": 0}

    def flaky_compute(path, chunk_size=1 << 20):
        calls["n"] += 1
        if calls["n"] == 2:  # the temp-file verification
            return "0" * 64
        return real_compute(path, chunk_size)

    monkeypatch.setattr(cr, "compute_sha256", flaky_compute)
    promoted, generation, detail = orch._promote_best_eval(
        candidate, ckpt_name="cand2.pt", comp_score=0.7, match_res=_match_res(), solve_rate=0.5, gate_reason="g"
    )
    assert promoted is False
    assert "verification failed" in detail
    assert not (orch.checkpoint_dir / "best_eval.pt").exists()
    assert not (orch.checkpoint_dir / "best_eval.json").exists()
    assert not list(orch.checkpoint_dir.glob("*.tmp"))


def test_promotion_refuses_a_missing_candidate(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    promoted, generation, detail = orch._promote_best_eval(
        tmp_path / "absent.pt",
        ckpt_name="absent.pt",
        comp_score=0.9,
        match_res=_match_res(),
        solve_rate=0.9,
        gate_reason="g",
    )
    assert promoted is False
    assert "missing" in detail


def test_promotion_after_interruption_never_pairs_stale_metadata(tmp_path, monkeypatch):
    """An interrupted promotion cannot leave JSON pointing at absent bytes."""
    orch = _bare_orchestrator(tmp_path)
    candidate = _candidate(tmp_path / "cand3.pt")

    real_replace = os.replace

    def failing_replace(src, dst):
        # Fail the *artifact* rename, after the temp file was written+verified.
        if str(dst).endswith("best_eval.pt"):
            raise OSError("simulated interruption at replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    promoted, _generation, detail = orch._promote_best_eval(
        candidate, ckpt_name="cand3.pt", comp_score=0.8, match_res=_match_res(), solve_rate=0.6, gate_reason="g"
    )
    assert promoted is False
    assert "promotion failed" in detail
    # No checkpoint without metadata, and no metadata without a checkpoint.
    assert not (orch.checkpoint_dir / "best_eval.json").exists()
    assert not (orch.checkpoint_dir / "best_eval.pt").exists()


def test_protected_promotion_destination_requires_a_ticket(tmp_path, monkeypatch):
    """The promotion path respects the production-write policy (CKP-04/CKP-03)."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "models").mkdir(parents=True)
    monkeypatch.setattr("python.common.artifacts.repo_root", lambda: fake_repo)

    orch = Phase4Orchestrator.__new__(Phase4Orchestrator)
    orch.checkpoint_dir = fake_repo / "models" / "checkpoints_rl"
    orch.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    orch._best_eval_state_path = orch.checkpoint_dir / "best_eval.json"
    orch._best_eval_comp_score = None
    orch.trainer = _TrainerStub()
    orch.artifact_ticket = None
    orch._event = _EventRecorder()  # type: ignore[method-assign]

    candidate = _candidate(tmp_path / "cand4.pt")
    promoted, _gen, detail = orch._promote_best_eval(
        candidate, ckpt_name="cand4.pt", comp_score=0.8, match_res=_match_res(), solve_rate=0.6, gate_reason="g"
    )
    assert promoted is False
    assert "protected production" in detail
    assert not (orch.checkpoint_dir / "best_eval.pt").exists()


# ── Eval history (CKP-15) ────────────────────────────────────────────────────


def _bare_eval_csv_orchestrator(tmp_path: Path) -> Phase4Orchestrator:
    orch = Phase4Orchestrator.__new__(Phase4Orchestrator)
    orch.log_dir = tmp_path
    orch.eval_csv_path = tmp_path / "eval_history.csv"
    orch.match_games_csv_path = tmp_path / "match_games.csv"
    orch.puzzle_eval_jsonl_path = tmp_path / "eval_puzzle_history.jsonl"
    orch._last_saved_ckpt_path = None
    orch._last_saved_ckpt_name = None
    orch._run_commit = "deadbeef"
    return orch


def test_eval_history_header_is_written_and_flushed(tmp_path):
    orch = _bare_eval_csv_orchestrator(tmp_path)
    orch._init_eval_csv()
    first_line = orch.eval_csv_path.read_text(encoding="utf-8").splitlines()[0]
    for column in (
        "checkpoint_file",
        "model_sha256",
        "commit",
        "promotion_decision",
        "promotion_reason",
        "promotion_generation",
    ):
        assert column in first_line, column


def test_eval_history_appends_rows_with_promotion_evidence(tmp_path):
    orch = _bare_eval_csv_orchestrator(tmp_path)
    orch._init_eval_csv()
    orch._append_eval_row([1.0, 0.1] + [0] * 30 + ["promoted", "200 games, engine score 0.6", "promo-x"])
    rows = list(csv.reader(orch.eval_csv_path.open(encoding="utf-8")))
    assert len(rows) == 2, "header + one data row"
    assert rows[1][-3:] == ["promoted", "200 games, engine score 0.6", "promo-x"]


def test_eval_history_rejects_mixed_schemas(tmp_path):
    orch = _bare_eval_csv_orchestrator(tmp_path)
    orch._init_eval_csv()
    orch._append_eval_row([1.0] * 35)
    # A file with data rows under an older header must not be silently extended.
    orch.eval_csv_path.write_text("timestamp,old_column\n1.0,x\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="older column schema"):
        orch._init_eval_csv()


def test_eval_history_fsyncs_each_row(tmp_path, monkeypatch):
    orch = _bare_eval_csv_orchestrator(tmp_path)
    orch._init_eval_csv()
    calls = {"n": 0}
    real_fsync = os.fsync

    def counting_fsync(fd):
        calls["n"] += 1
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    orch._append_eval_row([0.1] * 36)
    assert calls["n"] >= 1, "each appended eval row must be fsynced (interruption durability)"


def test_puzzle_and_match_records_carry_checkpoint_identity(tmp_path):
    orch = _bare_eval_csv_orchestrator(tmp_path)
    orch._init_eval_csv()
    ckpt = _candidate(tmp_path / "last.pt")
    orch._last_saved_ckpt_path = str(ckpt)
    orch._last_saved_ckpt_name = "last.pt"
    orch.trainer = _TrainerStub()
    orch.server = type("S", (), {"model_version": 3})()

    orch.stockfish_elo = 1500
    orch.stockfish_nodes = 1000
    orch.eval_sims = 400
    orch._persist_puzzle_result(0.42, 42, 100)
    orch._persist_match_game({"opening_idx": 1, "engine_is_white": True, "result": "1-0", "seed": 42, "moves": ["e2e4"]})
    orch._persist_match_game({"opening_idx": 2, "engine_is_white": False, "result": "1/2-1/2", "seed": 43, "moves": []})

    puzzle = json.loads(orch.puzzle_eval_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert puzzle["model_sha256"] != "unknown"
    assert puzzle["checkpoint_file"] == "last.pt"
    rows = list(csv.reader(orch.match_games_csv_path.open(encoding="utf-8")))
    assert len(rows) == 3
    assert rows[1][4] != "unknown", "match rows must carry the evaluated model hash"


# ── Telemetry CSV durability (SUS-01) ────────────────────────────────────────


def test_telemetry_csv_append_is_flushed_and_fsynced(tmp_path, monkeypatch):
    from python.selfplay.metrics import MetricsTracker

    tracker = MetricsTracker(log_dir=tmp_path / "tb", csv_path=tmp_path / "telemetry.csv")
    calls = {"n": 0}
    real_fsync = os.fsync

    def counting_fsync(fd):
        calls["n"] += 1
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    tracker.log_step(step=1, buffer_size=10, model_version=1)
    tracker.close()
    assert calls["n"] >= 1, "telemetry rows must be fsynced (SUS-01)"
    rows = list(csv.reader((tmp_path / "telemetry.csv").open(encoding="utf-8")))
    assert len(rows) >= 2 and rows[-1], "the appended row must be durably present"
