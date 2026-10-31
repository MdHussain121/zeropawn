"""Regression tests for RL auto-resume selection and run-local pacing counters.

Covers the strength-audit findings PR-02 (poisoned auto-resume), PR-03
(pacing gate vs inherited supervised step counter), and RL-03 (staleness
filter arming + GPU-cache version provenance).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import (
    UNKNOWN_MODEL_VERSION,
    GpuResidentReplayCache,
    ReplayBuffer,
)
from python.selfplay.worker import SavedPosition
from python.train.train_rl import (
    RLTrainer,
    classify_optimizer_state,
    inspect_rl_checkpoint,
)
from scripts.run_phase4_rl import pacing_target_step


# ---------------------------------------------------------------------------
# Pacing gate (PR-03)
# ---------------------------------------------------------------------------


def test_pacing_target_step_floor_and_scaling():
    # Fresh buffer: floor of 500 supported steps, so a run can start training
    # before meaningful self-play production exists.
    assert pacing_target_step(0, 256, 16) == 500
    assert pacing_target_step(1000, 256, 16) == 500
    # Production-based growth: 100k positions * 16 reuses / 256 batch.
    assert pacing_target_step(100_000, 256, 16) == 6250


def test_warm_start_pacing_gate_is_open_immediately(tmp_path):
    """The Oct 26 failure: an inherited supervised step (6000) was compared
    against run-local buffer production, holding the gate closed ~100 minutes.
    With warm starts resetting the RL counter to 0, the gate must open on the
    first loop iteration (0 <= floor)."""
    target = pacing_target_step(0, 256, 16)
    run_local_step = 0  # warm start: trainer.step_num reset to 0
    assert run_local_step <= target


# ---------------------------------------------------------------------------
# Optimizer-state classification and inspection (PR-02)
# ---------------------------------------------------------------------------


def _opt_state(per_param_keys: dict) -> dict:
    return {
        "state": {0: per_param_keys},
        "param_groups": [{"lr": 0.01, "params": [0]}],
    }


def test_classify_optimizer_state_variants():
    assert classify_optimizer_state({"optimizer": None}) == "empty"
    assert classify_optimizer_state({}) == "empty"
    # Adam-family per-param state (the supervised warm-start poison).
    assert classify_optimizer_state({"optimizer": _opt_state({"exp_avg": 1, "exp_avg_sq": 1})}) == "AdamW"
    # SGD momentum buffer.
    assert classify_optimizer_state({"optimizer": _opt_state({"momentum_buffer": 1})}) == "SGD"
    # SGD before its first step has no per-param state.
    assert classify_optimizer_state({"optimizer": _opt_state({})}) == "empty"
    # A stamp always wins.
    assert (
        classify_optimizer_state(
            {"_optimizer_type": "SGD", "optimizer": _opt_state({"exp_avg": 1})}
        )
        == "SGD"
    )
    # Unrecognizable keys.
    assert classify_optimizer_state({"optimizer": _opt_state({"something_else": 1})}) == "unknown"


def test_inspect_rl_checkpoint_metadata_and_rejection(tmp_path):
    ckpt = {
        "model": {"stem_conv.weight": torch.zeros(4)},
        "optimizer": _opt_state({"momentum_buffer": 1}),
        "_optimizer_type": "SGD",
        "step": 997,
        "model_version": 5,
        "buffer_write_head": 15947,
        "last_push_step": 800,
    }
    p = tmp_path / "phase4_ckpt_v5_step997.pt"
    torch.save(ckpt, p)
    info = inspect_rl_checkpoint(p)
    assert info is not None
    assert info["step"] == 997
    assert info["optimizer_class"] == "SGD"
    assert info["buffer_write_head"] == 15947

    # Garbage file → rejected, not raised.
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"not a torch file")
    assert inspect_rl_checkpoint(bad) is None

    # Missing weights → rejected.
    no_model = tmp_path / "no_model.pt"
    torch.save({"step": 3}, no_model)
    assert inspect_rl_checkpoint(no_model) is None

    # Absent file → None.
    assert inspect_rl_checkpoint(tmp_path / "missing.pt") is None


# ---------------------------------------------------------------------------
# Auto-resume selection (PR-02 / B2)
# ---------------------------------------------------------------------------


def _write_ckpt(path: Path, *, step: int, optimizer_class: str, stamp: bool = True) -> None:
    if optimizer_class == "AdamW":
        opt = _opt_state({"exp_avg": 1, "exp_avg_sq": 1})
        opt_type = "AdamW"
    elif optimizer_class == "SGD":
        opt = _opt_state({"momentum_buffer": 1})
        opt_type = "SGD"
    else:  # empty SGD state (fresh optimizer, never stepped)
        opt = _opt_state({})
        opt_type = "SGD"
    ckpt = {
        "model": {"stem_conv.weight": torch.zeros(4)},
        "optimizer": opt,
        "step": step,
    }
    if stamp:
        ckpt["_optimizer_type"] = opt_type
    torch.save(ckpt, path)


def _select(path: Path):
    from scripts.run_phase4_rl import Phase4Orchestrator

    fake = types.SimpleNamespace(checkpoint_dir=path)
    return Phase4Orchestrator._select_resume_checkpoint(fake)


def test_auto_resume_skips_poisoned_and_picks_highest_step(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    # The poisoned artifacts of the Oct 26 failed runs: full-strength AdamW
    # state under the highest-priority filenames.
    _write_ckpt(ckpt_dir / "phase4_final_checkpoint.pt", step=6000, optimizer_class="AdamW")
    _write_ckpt(ckpt_dir / "latest_checkpoint.pt", step=6000, optimizer_class="AdamW")
    _write_ckpt(ckpt_dir / "phase4_ckpt_v1_step6000.pt", step=6000, optimizer_class="AdamW")
    # The genuine Oct 22-23 SGD session checkpoint.
    _write_ckpt(ckpt_dir / "phase4_ckpt_v5_step997.pt", step=997, optimizer_class="SGD")
    # Unreadable junk that must not crash the scan.
    (ckpt_dir / "corrupt.pt").write_bytes(b"junk")

    choice = _select(ckpt_dir)
    assert choice is not None
    assert Path(choice).name == "phase4_ckpt_v5_step997.pt"


def test_auto_resume_all_poisoned_returns_none(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    _write_ckpt(ckpt_dir / "phase4_final_checkpoint.pt", step=6000, optimizer_class="AdamW")
    assert _select(ckpt_dir) is None


def test_auto_resume_empty_dir_returns_none(tmp_path):
    assert _select(tmp_path / "does_not_exist") is None


def test_auto_resume_unstamped_healthy_sgd_selected(tmp_path):
    """Pre-stamp checkpoints (e.g. the Oct 22-23 step-997 file) are classified
    by their per-param state keys, not rejected for lacking a stamp."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    _write_ckpt(ckpt_dir / "phase4_ckpt_v5_step997.pt", step=997, optimizer_class="SGD", stamp=False)
    _write_ckpt(ckpt_dir / "phase4_ckpt_v1_step6000.pt", step=6000, optimizer_class="AdamW", stamp=False)
    choice = _select(ckpt_dir)
    assert choice is not None
    assert Path(choice).name == "phase4_ckpt_v5_step997.pt"


# ---------------------------------------------------------------------------
# Warm-start counter reset and provenance (PR-03 / B3)
# ---------------------------------------------------------------------------


def _one_hot(slot: int) -> np.ndarray:
    """Normalized one-hot visit distribution (valid policy target, SR-07)."""
    v = np.zeros(4672, dtype=np.float32)
    v[slot] = 1.0
    return v


def _make_trainer(tmp_path, name: str, reopen: bool = False) -> RLTrainer:
    """Build a trainer on a synthetic buffer.

    ``reopen=True`` reopens an EXISTING buffer directory (same buffer id and
    generation, no reset) — the legitimate same-buffer resume case of the
    audit SR-04 contract.
    """
    buf = ReplayBuffer(
        buffer_dir=tmp_path / f"buf_{name}", capacity=100, create_new=not reopen
    )
    if reopen:
        model = ChessNet(num_blocks=2, num_filters=32)
        return RLTrainer(
            model=model,
            replay_buffer=buf,
            batch_size=8,
            warmup_steps=5,
            total_steps=100,
            checkpoint_dir=tmp_path / f"ckpt_{name}",
            sync_weights_path=tmp_path / f"sync_{name}.pt",
        )
    # A few positions so train_step can sample immediately.
    positions = []
    for i in range(16):
        positions.append(
            SavedPosition(
                planes=(np.random.rand(19, 8, 8) < 0.5).astype(np.float32),
                visits=_one_hot(i % 4672),
                outcome=i % 3,
                is_full_search=True,
                model_version=1,
                game_id=i + 1,
            )
        )
    buf.add_positions(positions)
    model = ChessNet(num_blocks=2, num_filters=32)
    return RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=8,
        warmup_steps=5,
        total_steps=100,
        checkpoint_dir=tmp_path / f"ckpt_{name}",
        sync_weights_path=tmp_path / f"sync_{name}.pt",
    )


def test_warm_start_resets_rl_counters_and_records_provenance(tmp_path):
    trainer = _make_trainer(tmp_path, "a")
    # Simulate a supervised AdamW checkpoint at step 6000 (the finetune shape).
    sup_ckpt = {
        "model": trainer.model.state_dict(),
        "optimizer": _opt_state({"exp_avg": 1, "exp_avg_sq": 1}),
        "_optimizer_type": "AdamW",
        "step": 6000,
    }
    p = tmp_path / "supervised_latest.pt"
    torch.save(sup_ckpt, p)

    trainer.load_checkpoint(p, reset_step=True)
    assert trainer.step_num == 0
    assert trainer.last_push_step == 0
    assert trainer.inherited_step_num == 6000
    assert trainer.warm_start_step == 6000
    assert trainer.warm_start_path == str(p)
    # The first training step must be reachable immediately (no KeyError).
    metrics = trainer.train_step()
    assert metrics["step"] == 1


def test_rl_resume_keeps_step_counter_and_momentum(tmp_path):
    trainer = _make_trainer(tmp_path, "b")
    # A real optimizer step creates the momentum buffers a resume must preserve.
    metrics = trainer.train_step()
    assert metrics["step"] == 1
    p = Path(trainer.save_checkpoint(tmp_path / "resume_ckpt.pt", keep_last_n=-1))
    saved = torch.load(p, weights_only=False)
    assert saved["_optimizer_type"] == "SGD"
    assert saved["step"] == 1
    assert "inherited_supervised_step" not in saved or saved["inherited_supervised_step"] == 0

    # Same-buffer resume (audit SR-04): reopening the SAME buffer directory
    # is a matching/legitimately-ahead state and must restore step+momentum.
    trainer2 = _make_trainer(tmp_path, "b", reopen=True)
    trainer2.load_checkpoint(p)
    assert trainer2.step_num == 1
    # Momentum buffers survived the round trip.
    state = trainer2.optimizer.state_dict()["state"]
    assert any("momentum_buffer" in per_param for per_param in state.values())

    # A checkpoint paired with a DIFFERENT buffer directory fails closed
    # unless the operator overrides it explicitly (SR-04).
    from python.train.train_rl import BufferCheckpointMismatchError

    trainer3 = _make_trainer(tmp_path, "c")
    with pytest.raises(BufferCheckpointMismatchError):
        trainer3.load_checkpoint(p)
    assert trainer3.load_checkpoint(p, allow_buffer_mismatch=True) == 1


# ---------------------------------------------------------------------------
# Staleness filter and version provenance (RL-03)
# ---------------------------------------------------------------------------


def _row(plane_val: float) -> np.ndarray:
    return np.full((19, 8, 8), plane_val, dtype=np.float32)


def test_staleness_filter_excludes_unknown_and_stale_rows():
    cache = GpuResidentReplayCache(capacity=64, device="cpu")
    # 32 known-fresh rows (version 30) and 32 rows with unknown provenance.
    n_known, n_unknown = 32, 32
    planes = np.stack([_row(1.0)] * n_known + [_row(2.0)] * n_unknown)
    visits = np.zeros((n_known + n_unknown, 4672), dtype=np.float32)
    # Outcomes encode row class so sampled rows are identifiable.
    outcomes = np.array([1] * n_known + [2] * n_unknown, dtype=np.int64)
    full = np.ones(n_known + n_unknown, dtype=np.bool_)
    versions = np.array([30] * n_known + [UNKNOWN_MODEL_VERSION] * n_unknown, dtype=np.int32)
    cache.add_positions(planes=planes, visits=visits, outcomes=outcomes, full_search=full, versions=versions)

    batch = cache.sample_batch(batch_size=16)
    assert int(batch["outcomes"].min()) == 1  # unknown rows never sampled
    assert int(batch["outcomes"].max()) == 1


def test_staleness_filter_is_relative_from_first_push():
    cache = GpuResidentReplayCache(capacity=128, device="cpu")
    n = 128
    # Versions 0..127: rows more than 20 pushes behind the max (version < 107)
    # must be excluded even though the absolute max is well past any gate.
    versions = np.arange(n, dtype=np.int32)
    planes = np.stack([_row(1.0)] * n)
    visits = np.zeros((n, 4672), dtype=np.float32)
    outcomes = versions.astype(np.int64)  # outcome == version
    full = np.ones(n, dtype=np.bool_)
    cache.add_positions(planes=planes, visits=visits, outcomes=outcomes, full_search=full, versions=versions)

    # batch_size must not exceed the valid (fresh) row count, otherwise the
    # documented fallback (sample everything) kicks in.
    batch = cache.sample_batch(batch_size=16)
    assert int(batch["outcomes"].min()) >= 127 - cache.STALENESS_WINDOW


def test_staleness_filter_fallback_when_only_unknown_rows():
    cache = GpuResidentReplayCache(capacity=16, device="cpu")
    n = 8
    planes = np.stack([_row(1.0)] * n)
    visits = np.zeros((n, 4672), dtype=np.float32)
    outcomes = np.ones(n, dtype=np.int64)
    full = np.ones(n, dtype=np.bool_)
    # No versions argument → all rows stamped UNKNOWN.
    cache.add_positions(planes=planes, visits=visits, outcomes=outcomes, full_search=full)
    assert int(cache.gpu_versions[: cache.size].min()) == UNKNOWN_MODEL_VERSION
    batch = cache.sample_batch(batch_size=4)  # fallback must still sample
    assert batch["planes"].shape[0] == 4


def test_gpu_cache_restore_preserves_real_versions(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=32, create_new=True)
    n = 16
    versions = np.arange(10, 10 + n, dtype=np.int32)
    positions = []
    for i in range(n):
        positions.append(
            types.SimpleNamespace(
                planes=_row(1.0),
                visits=_one_hot(i % 4672),
                outcome=int(i % 3),
                is_full_search=True,
                model_version=int(versions[i]),
                game_id=i + 1,
            )
        )
    buf.add_positions(positions)
    buf.enable_gpu_cache(capacity=8, device="cpu")
    assert buf.gpu_cache is not None
    restored = buf.gpu_cache.gpu_versions[: buf.gpu_cache.size].tolist()
    # The 8 most recent rows carry their true versions (18..25), not 0/unknown.
    assert restored == [18, 19, 20, 21, 22, 23, 24, 25]
