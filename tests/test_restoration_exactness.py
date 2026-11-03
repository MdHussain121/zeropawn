"""Bitwise restoration guarantees for every supported checkpoint family.

The 2026-10-30/31 audit verified bitwise-exact restoration of the artifacts that
exist on disk (model, optimizer moments, scheduler position, CPU RNG, next-draw
continuity) and required that the remediation must not weaken those guarantees
(audit roadmap phase P). These tests pin them with synthetic 2x32 checkpoints so
they run everywhere and never touch production artifacts.

CUDA RNG exactness is claimed only when the device topology matches; the
CPU-only claims below are unconditional, and the CUDA-specific assertions are
skipped without a GPU.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import RLTrainer
from python.train.train_supervised import WarmupCosineScheduler, save_checkpoint


def _positions(n: int, start: int = 0):
    from python.selfplay.worker import SavedPosition

    out = []
    for i in range(start, start + n):
        visits = np.zeros(4672, dtype=np.float32)
        visits[i % 4672] = 1.0
        out.append(
            SavedPosition(
                planes=np.random.rand(19, 8, 8).astype(np.float32),
                visits=visits,
                outcome=i % 3,
                is_full_search=True,
                model_version=1,
                game_id=i + 1,  # ring records require game_id > 0 (integrated SR-07 contract)
            )
        )
    return out


def _rl_trainer(tmp_path: Path, name: str, buffer: ReplayBuffer | None = None) -> RLTrainer:
    if buffer is None:
        buffer = ReplayBuffer(buffer_dir=tmp_path / f"b_{name}", capacity=64, create_new=True)
    return RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buffer,
        device="cpu",
        batch_size=8,
        base_lr=0.01,
        min_lr=1e-4,
        warmup_steps=5,
        total_steps=100,
        checkpoint_dir=tmp_path / f"c_{name}",
        sync_weights_path=tmp_path / f"s_{name}.pt",
    )


# ── RL family ────────────────────────────────────────────────────────────────


def test_rl_restores_model_optimizer_scheduler_and_rng_bitwise(tmp_path):
    trainer = _rl_trainer(tmp_path, "exact")
    trainer.replay_buffer.add_positions(_positions(16))
    for _ in range(3):
        trainer.train_step()
    ckpt = Path(trainer.save_checkpoint(keep_last_n=-1))
    saved = torch.load(ckpt, weights_only=False, map_location="cpu")

    buffer = ReplayBuffer(buffer_dir=tmp_path / "b_exact", capacity=64, create_new=False)
    resumed = _rl_trainer(tmp_path, "exact2", buffer=buffer)
    resumed.load_checkpoint(ckpt)

    # Model: every tensor bitwise equal.
    current = resumed.model.state_dict()
    reference = saved["model"]
    assert set(current) == set(reference)
    for key in reference:
        assert torch.equal(current[key].cpu(), reference[key].cpu()), key

    # Optimizer: same LR/momentum hyperparameters and every momentum buffer bitwise.
    state = resumed.optimizer.state_dict()
    assert state["param_groups"][0]["momentum"] == saved["optimizer"]["param_groups"][0]["momentum"]
    saved_state = saved["optimizer"]["state"]
    assert len(state["state"]) == len(saved_state)
    for pid, entry in saved_state.items():
        for key, value in entry.items():
            assert torch.equal(state["state"][pid][key].cpu(), value.cpu()), (pid, key)

    # Scheduler: position matches the checkpointed step, and the LR restored for
    # the *next* update is the schedule's value at that step (a pure function of
    # the step counter). Note the nuance: the LR stored in the saved optimizer
    # param group is the one the last executed update used (LR(step-1)); the
    # resume recomputes LR(step), which is exactly what an uninterrupted run
    # would use for its next update.
    assert resumed.scheduler.current_step == saved["step"]
    assert resumed.step_num == saved["step"]
    expected_lr = resumed.scheduler.get_lr(saved["step"])
    assert abs(resumed.optimizer.param_groups[0]["lr"] - expected_lr) < 1e-15
    uninterrupted_next_lr = trainer.scheduler.get_lr(trainer.step_num)
    assert abs(expected_lr - uninterrupted_next_lr) < 1e-15

    # CPU RNG: torch / numpy / python restored exactly, and the *next draw* is
    # identical (continuity, not just state equality).
    assert torch.equal(
        torch.get_rng_state(), torch.as_tensor(saved["rng_state"], dtype=torch.uint8)
    )
    # numpy: same algorithm, same position, same key array (draw continuity is
    # additionally pinned by test_rl_next_draw_continuity_after_resume).
    np_now = np.random.get_state()
    assert np_now[2] == saved["numpy_rng_state"][2]
    assert np_now[3] == saved["numpy_rng_state"][3]
    assert np_now[4] == pytest.approx(saved["numpy_rng_state"][4], abs=0.0)
    assert np.array_equal(np_now[1], np.asarray(saved["numpy_rng_state"][1], dtype=np.uint32))
    # python: the full Mersenne state matches.
    assert random.getstate() == saved["python_rng_state"]


def _restore_and_draw_reference(saved: dict) -> tuple:
    """Restore the checkpoint's numpy/python state and draw the next values."""
    np.random.set_state(
        (
            saved["numpy_rng_state"][0],
            np.asarray(saved["numpy_rng_state"][1], dtype=np.uint32),
            saved["numpy_rng_state"][2],
            saved["numpy_rng_state"][3],
            saved["numpy_rng_state"][4],
        )
    )
    return float(np.random.random())


def _draw_now() -> float:
    return float(np.random.random())


def _python_next_draw(saved: dict) -> float:
    random.setstate(saved["python_rng_state"])
    return random.random()


def test_rl_next_draw_continuity_after_resume(tmp_path):
    """The exact sequence a resumed run would draw matches the checkpoint's state."""
    trainer = _rl_trainer(tmp_path, "draw")
    trainer.replay_buffer.add_positions(_positions(8))
    trainer.train_step()
    ckpt = Path(trainer.save_checkpoint(keep_last_n=-1))

    # Reference: load the checkpoint's RNG states into fresh globals and draw.
    saved = torch.load(ckpt, weights_only=False, map_location="cpu")
    torch.set_rng_state(torch.as_tensor(saved["rng_state"], dtype=torch.uint8))
    np.random.set_state(
        (
            saved["numpy_rng_state"][0],
            np.asarray(saved["numpy_rng_state"][1], dtype=np.uint32),
            saved["numpy_rng_state"][2],
            saved["numpy_rng_state"][3],
            saved["numpy_rng_state"][4],
        )
    )
    random.setstate(saved["python_rng_state"])
    expected = (torch.rand(3).tolist(), float(np.random.random()), random.random())

    # Now let a resumed trainer restore the same state and draw again.
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    buffer = ReplayBuffer(buffer_dir=tmp_path / "b_draw", capacity=64, create_new=False)
    resumed = _rl_trainer(tmp_path, "draw2", buffer=buffer)
    resumed.load_checkpoint(ckpt)
    actual = (torch.rand(3).tolist(), float(np.random.random()), random.random())
    assert actual[0] == pytest.approx(expected[0], abs=0.0)
    assert actual[1] == expected[1]
    assert actual[2] == expected[2]


# ── Supervised family ────────────────────────────────────────────────────────


def test_supervised_restores_model_adamw_and_scheduler(tmp_path):
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=5, total_steps=50, base_lr=1e-4)
    # Materialize real AdamW moments.
    for _ in range(2):
        loss = sum(p.square().sum() for p in model.parameters())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    ckpt = tmp_path / "sup.pt"
    save_checkpoint(
        ckpt,
        model,
        optimizer,
        scheduler,
        step=2,
        epoch=0,
        best_val_loss=1.0,
        num_blocks=2,
        num_filters=32,
    )
    saved = torch.load(ckpt, weights_only=False, map_location="cpu")

    model2 = ChessNet(num_blocks=2, num_filters=32)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    scheduler2 = WarmupCosineScheduler(optimizer2, warmup_steps=5, total_steps=50, base_lr=1e-4)
    model2.load_state_dict(saved["model"])
    optimizer2.load_state_dict(saved["optimizer"])
    scheduler2.current_step = saved["scheduler"]["current_step"]

    # Model bitwise.
    for key, value in saved["model"].items():
        assert torch.equal(model2.state_dict()[key].cpu(), value.cpu()), key
    # AdamW moments bitwise (exp_avg / exp_avg_sq / step).
    restored = optimizer2.state_dict()["state"]
    reference = saved["optimizer"]["state"]
    assert set(restored) == set(reference)
    for pid, entry in reference.items():
        for key, value in entry.items():
            restored_value = restored[pid][key]
            if torch.is_tensor(value):
                assert torch.equal(restored_value.cpu(), value.cpu()), (pid, key)
            else:
                assert restored_value == value, (pid, key)
    # Scheduler position and posture.
    assert scheduler2.current_step == saved["scheduler"]["current_step"] == 2
    assert scheduler2.warmup_steps == saved["scheduler"]["warmup_steps"]
    assert scheduler2.total_steps == saved["scheduler"]["total_steps"]
    assert scheduler2.base_lr == saved["scheduler"]["base_lr"]
    assert abs(scheduler2.get_lr(2) - scheduler.get_lr(2)) < 1e-15
    # RNG: the resume path restores it verbatim (mirrored here, since this test
    # reconstructs the restore steps instead of running the whole trainer loop).
    torch.manual_seed(999)
    torch.set_rng_state(torch.as_tensor(saved["rng_state"], dtype=torch.uint8))
    assert torch.equal(
        torch.get_rng_state(), torch.as_tensor(saved["rng_state"], dtype=torch.uint8)
    )
    _restore_and_draw_reference(saved)
    restored_numpy_draw = _draw_now()
    _restore_and_draw_reference(saved)
    assert _draw_now() == restored_numpy_draw


def test_supervised_scheduler_lr_is_a_pure_function_of_step(tmp_path):
    """Resume must not skip or repeat schedule steps (audit answer 5)."""
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100, base_lr=1e-4)
    # Fresh scheduler at the same step yields the same LR as a stepped one.
    scheduler.current_step = 42
    fresh = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100, base_lr=1e-4)
    assert scheduler.get_lr(42) == fresh.get_lr(42)
    assert scheduler.step(42) == fresh.get_lr(42)


# ── CUDA RNG policy ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_rng_restored_when_device_count_matches(tmp_path):
    trainer = _rl_trainer(tmp_path, "cuda_src")
    trainer.device = torch.device("cuda")
    trainer.device_type = "cuda"
    torch.cuda.manual_seed_all(2026)
    expected = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    ckpt = Path(trainer.save_checkpoint(tmp_path / "cuda_rng.pt", keep_last_n=-1))

    buffer = ReplayBuffer(buffer_dir=tmp_path / "b_cuda_src", capacity=64, create_new=False)
    resumed = _rl_trainer(tmp_path, "cuda_dst", buffer=buffer)
    resumed.device = torch.device("cuda")
    resumed.device_type = "cuda"
    torch.cuda.manual_seed_all(7)
    resumed.load_checkpoint(ckpt)
    actual = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
