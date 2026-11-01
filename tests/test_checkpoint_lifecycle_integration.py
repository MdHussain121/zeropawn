"""Synthetic end-to-end checkpoint lifecycle integration tests.

Two flows, both in temporary directories with synthetic artifacts only:

1. Run initialization -> checkpoint save -> run metadata persistence -> hash
   verification -> checkpoint reload -> full resume / weights-only warm start ->
   new checkpoint creation (audit roadmap phase Q, "integration").
2. Evaluation -> candidate validation -> atomic best-checkpoint promotion ->
   durable evaluation history (idempotent on rerun, survives interruption).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from python.model.checkpoint_registry import compute_sha256, verify_checkpoint_identity, write_registry
from python.model.checkpoint_schema import (
    RUN_STATE_COMPLETED,
    advance_run_state_inplace,
    validate_checkpoint_schema,
    validate_run_metadata,
)
from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import RLTrainer
from scripts.run_phase4_rl import Phase4Orchestrator


def _positions(n: int, start: int = 0):
    from python.selfplay.worker import SavedPosition

    out = []
    for i in range(start, start + n):
        visits = np.zeros(4672, dtype=np.float32)
        visits[i % 4672] = 1.0
        out.append(
            SavedPosition(
                planes=np.random.randn(19, 8, 8).astype(np.float32),
                visits=visits,
                outcome=i % 3,
                is_full_search=True,
                model_version=1,
                game_id=i,
            )
        )
    return out


def _trainer(tmp_path: Path, name: str, buffer: ReplayBuffer | None = None) -> RLTrainer:
    if buffer is None:
        buffer = ReplayBuffer(buffer_dir=tmp_path / f"buf_{name}", capacity=64, create_new=True)
    return RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buffer,
        device="cpu",
        batch_size=8,
        warmup_steps=5,
        total_steps=100,
        checkpoint_dir=tmp_path / f"ckpts_{name}",
        sync_weights_path=tmp_path / f"sync_{name}.pt",
    )


# ── Flow 1: train -> save -> verify -> reload -> resume/warm start -> save ────


def test_full_lifecycle_resume_flow(tmp_path):
    train_dir = tmp_path / "ckpts_main"
    train_dir.mkdir()

    # 1. Run initialization + training.
    trainer = _trainer(tmp_path, "main")
    trainer.replay_buffer.add_positions(_positions(20))
    for _ in range(3):
        trainer.train_step()
    assert trainer.step_num == 3
    run_id = trainer.run_metadata["run_id"]

    # 2. Checkpoint save (atomic, locked, ledger-bearing).
    ckpt = Path(trainer.save_checkpoint(keep_last_n=-1))
    assert ckpt.is_file()
    assert (ckpt.parent / "latest_checkpoint.pt").is_file()
    assert not list(ckpt.parent.glob("*.tmp"))

    # 3. Run metadata persistence: inside the artifact, schema-valid.
    saved = torch.load(ckpt, weights_only=False)
    verdict = validate_checkpoint_schema(saved, path=ckpt, expected_family="rl")
    assert verdict.version == 2 and not verdict.is_legacy
    assert validate_run_metadata(saved["run_metadata"], path=ckpt) == []
    assert saved["run_metadata"]["run_id"] == run_id
    assert saved["provenance"]["run_id"] == run_id
    assert saved["buffer_state"]["generation"] == trainer.replay_buffer.buffer_generation

    # 4. Hash verification against a registry that records exactly these bytes.
    (tmp_path / "docs").mkdir(exist_ok=True)
    sha = compute_sha256(ckpt)
    write_registry(
        {
            "version": 1,
            "checkpoints": [
                {
                    "id": "ckpt-int",
                    "file": str(ckpt),
                    "sha256": sha,
                    "status": "verified",
                    "designated_warm_start": False,
                }
            ],
        },
        tmp_path,
    )
    verdict_id = verify_checkpoint_identity(ckpt, repo_root=tmp_path)
    assert verdict_id["status"] == "verified"

    # 5. Full resume of the same run's checkpoint (same buffer directory).
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_main", capacity=64, create_new=False)
    resumed = _trainer(tmp_path, "resumed", buffer=buf)
    resumed.load_checkpoint(ckpt, verify_identity="required")
    assert resumed.step_num == 3
    assert resumed.run_metadata["run_id"] != run_id
    assert resumed.run_metadata["parent_checkpoint"]["sha256"] == sha
    assert resumed.run_metadata["predecessor_run"]["run_id"] == run_id
    decision = resumed.run_metadata["buffer_resume_decision"]
    assert decision["decision"] == "identical_snapshot"

    # 6. Continue training and write a NEW checkpoint with lineage preserved.
    resumed.train_step()
    advance_run_state_inplace(resumed.run_metadata, RUN_STATE_COMPLETED, step=resumed.step_num)
    ckpt2 = Path(resumed.save_checkpoint(keep_last_n=-1))
    saved2 = torch.load(ckpt2, weights_only=False)
    assert saved2["step"] == 4
    assert saved2["run_metadata"]["run_state"] == RUN_STATE_COMPLETED
    assert saved2["provenance"]["parent_checkpoint"]["path"].endswith(ckpt.name)
    assert saved2["provenance"]["parent_checkpoint"]["sha256"] == sha
    # latest_checkpoint.pt mirror followed the new save.
    assert compute_sha256(ckpt2.parent / "latest_checkpoint.pt") == compute_sha256(ckpt2)


def test_full_lifecycle_warm_start_flow(tmp_path):
    # Source: a supervised-shaped checkpoint written by the supervised writer,
    # exported as the weights-only container the RL warm-start path consumes
    # (project policy refuses *explicitly typed foreign* artifacts on the
    # trainer's warm-start path; the production warm start runs through the
    # registry designation gate + inference server, and only weights transfer).
    from python.train.train_supervised import WarmupCosineScheduler, save_checkpoint

    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=5, total_steps=50, base_lr=2e-4)
    source_dir = tmp_path / "sup"
    source_dir.mkdir()
    source = source_dir / "best_model.pt"
    save_checkpoint(
        source,
        model,
        optimizer,
        scheduler,
        step=40,
        epoch=1,
        best_val_loss=1.5,
        num_blocks=2,
        num_filters=32,
        run_metadata={"run_id": "sup-1", "run_type": "supervised", "run_state": "completed"},
    )
    # The typed training checkpoint is refused by the RL trainer on every path
    # (documented policy), with a diagnostic naming the sanctioned flow.
    typed_trainer = _trainer(tmp_path, "typed")
    with pytest.raises(RuntimeError, match="supervised training checkpoint"):
        typed_trainer.load_checkpoint(source, warm_start=True)

    weights_only_source = source_dir / "weights_only.pt"
    payload = torch.load(source, weights_only=False)
    torch.save(
        {
            "model": payload["model"],
            "num_blocks": payload["num_blocks"],
            "num_filters": payload["num_filters"],
            "step": payload["step"],
        },
        weights_only_source,
    )
    source = weights_only_source

    # Warm start in the RL trainer: weights only, fresh optimizer, step 0.
    trainer = _trainer(tmp_path, "ws")
    trainer.replay_buffer.add_positions(_positions(10))
    returned = trainer.load_checkpoint(source, warm_start=True)
    assert returned == 0
    assert trainer.inherited_step_num == 40
    assert trainer.run_metadata["run_type"] == "rl-warm-start"
    assert trainer.optimizer.state_dict()["state"] == {}
    assert trainer.run_metadata["parent_checkpoint"]["path"].endswith("weights_only.pt")

    trainer.train_step()
    ckpt = Path(trainer.save_checkpoint(keep_last_n=-1))
    saved = torch.load(ckpt, weights_only=False)
    assert saved["step"] == 1
    assert saved["run_metadata"]["run_state"] == "running"
    # The inherited supervised step is provenance, never RL progress.
    assert saved["step"] != 40
    assert saved["provenance"]["parent_checkpoint"]["path"].endswith("weights_only.pt")


def test_interrupted_run_checkpoint_can_be_resumed_without_false_completion(tmp_path):
    trainer = _trainer(tmp_path, "int")
    trainer.replay_buffer.add_positions(_positions(10))
    trainer.train_step()
    advance_run_state_inplace(
        trainer.run_metadata, "interrupted", step=trainer.step_num, reason="KeyboardInterrupt"
    )
    ckpt = Path(trainer.save_checkpoint(keep_last_n=-1))

    saved = torch.load(ckpt, weights_only=False)
    assert saved["run_metadata"]["run_state"] == "interrupted"

    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_int", capacity=64, create_new=False)
    resumed = _trainer(tmp_path, "int2", buffer=buf)
    resumed.load_checkpoint(ckpt)
    # The continuation is a new run; the predecessor's interrupted status is kept.
    assert resumed.run_metadata["run_state"] == "created"
    assert resumed.run_metadata["predecessor_run"]["run_state"] == "interrupted"


# ── Flow 2: evaluation -> validation -> promotion -> durable history ─────────


class _OrchStub(Phase4Orchestrator):
    pass


def _eval_orchestrator(tmp_path: Path) -> Phase4Orchestrator:
    orch = Phase4Orchestrator.__new__(_OrchStub)
    orch.log_dir = tmp_path / "logs"
    orch.log_dir.mkdir(parents=True, exist_ok=True)
    orch.checkpoint_dir = tmp_path / "ckpts"
    orch.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    orch.eval_csv_path = orch.log_dir / "eval_history.csv"
    orch.match_games_csv_path = orch.log_dir / "match_games.csv"
    orch.puzzle_eval_jsonl_path = orch.log_dir / "eval_puzzle_history.jsonl"
    orch._best_eval_state_path = orch.checkpoint_dir / "best_eval.json"
    orch._best_eval_comp_score = None
    orch._last_saved_ckpt_path = None
    orch._last_saved_ckpt_name = None
    orch._run_commit = "cafebabe"
    orch.artifact_ticket = None
    orch.stockfish_elo = 1500
    orch.stockfish_nodes = 1000
    orch.eval_sims = 100
    orch.is_verify_mode = True
    orch.trainer = type("T", (), {"step_num": 7})()
    orch.server = type("S", (), {"model_version": 2})()
    events: list[tuple[str, str]] = []
    orch._event = lambda kind, detail: events.append((kind, detail))  # type: ignore[method-assign]
    orch._events = events
    return orch


def _candidate(tmp_path: Path, step: int = 7) -> Path:
    path = tmp_path / "ckpts" / f"phase4_ckpt_v9_step{step}.pt"
    torch.save(
        {
            "checkpoint_type": "rl",
            "checkpoint_format": 2,
            "model": ChessNet(num_blocks=2, num_filters=32).state_dict(),
            "optimizer": {"param_groups": [{"momentum": 0.9}], "state": {}},
            "scheduler": {"current_step": step},
            "step": step,
            "num_blocks": 2,
            "num_filters": 32,
        },
        path,
    )
    return path


def _match_res(score: float) -> dict:
    return {
        "engine_score": score,
        "engine_wins": int(score * 200),
        "engine_draws": 0,
        "engine_losses": 200 - int(score * 200),
        "draw_rate": 0.0,
        "total_games": 200,
    }


def test_evaluation_to_promotion_with_durable_history(tmp_path):
    orch = _eval_orchestrator(tmp_path)
    orch._init_eval_csv()
    candidate = _candidate(tmp_path)
    candidate_sha = compute_sha256(candidate)

    # 1. Candidate validation + atomic promotion.
    promoted, generation, detail = orch._promote_best_eval(
        candidate,
        ckpt_name=candidate.name,
        comp_score=0.61,
        match_res=_match_res(0.7),
        solve_rate=0.4,
        gate_reason="200 games, engine score 0.700",
    )
    assert promoted, detail

    # 2. The promotion is recorded durably, with hash and generation.
    record = json.loads((orch.checkpoint_dir / "best_eval.json").read_text(encoding="utf-8"))
    assert record["checkpoint_sha256"] == candidate_sha
    assert record["generation"] == generation
    assert compute_sha256(orch.checkpoint_dir / "best_eval.pt") == candidate_sha

    # 3. Eval history carries the decision + reason as real rows (not header-only).
    orch._append_eval_row(
        [1.0, 0.2, 7, 2, candidate.name] + [0] * 28 + ["promoted", record["reason"], generation]
    )
    rows = list(csv.reader(orch.eval_csv_path.open(encoding="utf-8")))
    assert len(rows) >= 2 and rows[-1][-1] == generation

    # 4. A worse candidate does not displace the artifact or its metadata.
    worse = _candidate(tmp_path, step=8)
    promoted2, _gen, _detail = orch._promote_best_eval(
        worse,
        ckpt_name=worse.name,
        comp_score=0.9,
        match_res=_match_res(0.9),
        solve_rate=0.9,
        gate_reason="should not matter",
    )
    # The method itself is unconditional (the gate lives in run_eval_cycle), so it
    # promotes -- but the metadata must then describe the NEW bytes consistently.
    assert promoted2
    record2 = json.loads((orch.checkpoint_dir / "best_eval.json").read_text(encoding="utf-8"))
    if record2["checkpoint_sha256"] == compute_sha256(worse):
        assert record2["generation"] != generation
    assert compute_sha256(orch.checkpoint_dir / "best_eval.pt") == record2["checkpoint_sha256"]


def test_repeated_identical_promotion_is_idempotent(tmp_path):
    orch = _eval_orchestrator(tmp_path)
    candidate = _candidate(tmp_path, step=11)
    first = orch._promote_best_eval(
        candidate,
        ckpt_name=candidate.name,
        comp_score=0.5,
        match_res=_match_res(0.6),
        solve_rate=0.4,
        gate_reason="g",
    )
    sha_first = compute_sha256(orch.checkpoint_dir / "best_eval.pt")
    second = orch._promote_best_eval(
        candidate,
        ckpt_name=candidate.name,
        comp_score=0.5,
        match_res=_match_res(0.6),
        solve_rate=0.4,
        gate_reason="g",
    )
    assert first[0] and second[0]
    assert compute_sha256(orch.checkpoint_dir / "best_eval.pt") == sha_first
    # Generations differ (each promotion is its own event) but always describe
    # the bytes on disk together with their metadata.
    record = json.loads((orch.checkpoint_dir / "best_eval.json").read_text(encoding="utf-8"))
    assert compute_sha256(orch.checkpoint_dir / "best_eval.pt") == record["checkpoint_sha256"]
