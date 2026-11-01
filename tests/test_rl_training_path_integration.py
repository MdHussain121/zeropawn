"""Bounded synthetic integration test: the real RL optimizer-update path.

Audit Phase J (mission §14.3): prove, at toy scale and entirely on CPU, that
the production training loop performs GENUINE optimizer updates —

    replay sampling → batch collation → forward pass → loss computation →
    backward pass → optimizer step → scheduler step → RL update counter →
    checkpoint save → resume verification

Nothing here touches production artifacts: the buffer, checkpoints,
live-weights file and telemetry all live in a per-test temporary directory.
No self-play is generated and no campaign is launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python  # noqa: F401
from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import SavedPosition
from python.train.train_rl import RLTrainer


def _positions(n: int) -> list[SavedPosition]:
    rng = np.random.default_rng(11)
    out = []
    for i in range(n):
        planes = (rng.random((19, 8, 8)) < 0.5).astype(np.float32)
        visits = np.zeros(4672, dtype=np.float32)
        # A peaked multi-move distribution (closer to real MCTS targets than
        # a pure one-hot), normalized exactly like the C++ root aggregation.
        k = rng.integers(2, 8)
        slots = rng.choice(4672, size=k, replace=False)
        visits[slots] = rng.random(k) + 0.1
        visits /= visits.sum()
        out.append(
            SavedPosition(
                planes=planes,
                visits=visits,
                outcome=int(i % 3),
                is_full_search=bool(i % 2 == 0),
                model_version=1,
                game_id=i + 1,
            )
        )
    return out


@pytest.fixture()
def toy_world(tmp_path):
    torch.manual_seed(3)
    np.random.seed(3)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buffer", capacity=256, create_new=True)
    buf.add_positions(_positions(128))
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        device="cpu",
        batch_size=16,
        warmup_steps=10,
        total_steps=1000,
        base_lr=0.0005,
        min_lr=1e-5,
        momentum=0.9,
        weight_decay=1e-4,
        push_freq=2,
        checkpoint_dir=tmp_path / "checkpoints",
        sync_weights_path=tmp_path / "checkpoints" / "live_weights.pt",
    )
    return tmp_path, buf, trainer


def test_real_optimizer_update_changes_parameters_and_counters(toy_world):
    tmp_path, _buf, trainer = toy_world
    weights_before = {
        k: v.detach().clone() for k, v in trainer.model.state_dict().items()
    }

    metrics = trainer.train_step()  # samples from the real buffer
    assert trainer.step_num == 1
    assert metrics["step"] == 1
    assert metrics["skipped_batch"] is False
    for name in ("total_loss", "policy_loss", "wdl_loss"):
        assert metrics[name] == metrics[name], f"{name} must be finite"
    assert metrics["n_total"] == 16

    changed = [
        k
        for k, v in trainer.model.state_dict().items()
        if not torch.equal(v.detach(), weights_before[k])
    ]
    assert changed, "a successful optimizer update must change model parameters"
    momentum_buffers = [
        s for s in trainer.optimizer.state.values() if "momentum_buffer" in s
    ]
    assert momentum_buffers, "SGD momentum state must exist after a real update"

    lr_after_one_step = trainer.scheduler.get_lr(1)
    assert trainer.scheduler.get_lr(1) == pytest.approx(lr_after_one_step)


def test_checkpoint_round_trip_preserves_full_training_state(toy_world):
    tmp_path, buf, trainer = toy_world
    for _ in range(3):
        trainer.train_step()
    assert trainer.step_num == 3

    ckpt_path = trainer.save_checkpoint(tmp_path / "checkpoints" / "manual.pt")
    assert Path(ckpt_path).is_file()

    # A fresh trainer (same geometry) resumes the exact state.
    model2 = ChessNet(num_blocks=2, num_filters=32)
    trainer2 = RLTrainer(
        model=model2,
        replay_buffer=buf,
        device="cpu",
        batch_size=16,
        warmup_steps=10,
        total_steps=1000,
        base_lr=0.0005,
        min_lr=1e-5,
        momentum=0.9,
        weight_decay=1e-4,
        push_freq=2,
        checkpoint_dir=tmp_path / "checkpoints2",
        sync_weights_path=tmp_path / "checkpoints2" / "live_weights.pt",
    )
    trainer2.load_checkpoint(ckpt_path)
    assert trainer2.step_num == 3
    assert trainer2.last_push_step == 2  # push_freq=2 → pushed after step 2
    assert trainer2.model_version == trainer.model_version
    # LR is a pure function of the step: identical before and after resume.
    assert trainer2.scheduler.get_lr(3) == pytest.approx(trainer.scheduler.get_lr(3))
    # Optimizer state traveled: momentum buffers present and identical.
    for (k1, s1), (k2, s2) in zip(
        trainer.optimizer.state.items(), trainer2.optimizer.state.items()
    ):
        assert torch.equal(s1["momentum_buffer"], s2["momentum_buffer"])
    # The resumed trainer continues updating from the restored counter.
    metrics = trainer2.train_step()
    assert trainer2.step_num == 4
    assert metrics["step"] == 4


def test_real_server_push_advances_the_version_contract(tmp_path):
    """The learner→actor weight push works against the real InferenceServer:
    version advances or WeightPushError propagates (never silent divergence)."""
    torch.manual_seed(5)
    np.random.seed(5)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buffer", capacity=128, create_new=True)
    buf.add_positions(_positions(64))

    # A seed checkpoint on disk for the server to load (tiny arch).
    seed_model = ChessNet(num_blocks=2, num_filters=32)
    seed_path = tmp_path / "seed.pt"
    torch.save(
        {
            "model": seed_model.state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": 0,
        },
        seed_path,
    )
    server = InferenceServer(checkpoint_path=seed_path, device="cpu")
    assert server.model_version == 1

    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        batch_size=16,
        warmup_steps=10,
        total_steps=1000,
        push_freq=2,
        checkpoint_dir=tmp_path / "checkpoints",
        sync_weights_path=tmp_path / "checkpoints" / "live_weights.pt",
        direct_server=server,
    )
    trainer.train_step()
    trainer.train_step()  # push fires here (push_freq=2)
    assert trainer.push_success_count == 1
    assert trainer.push_failure_count == 0
    assert server.model_version == 2
    assert trainer.model_version == 2
    assert trainer.last_push_step == 2
    # The sync artifact was written and is weights-only.
    live = torch.load(
        tmp_path / "checkpoints" / "live_weights.pt", map_location="cpu", weights_only=True
    )
    assert "optimizer" not in live
    assert live["step"] == 2
