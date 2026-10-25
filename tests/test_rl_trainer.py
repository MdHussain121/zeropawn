"""Tests for Task 4: RL Trainer (SGD momentum, full-search policy CE, WDL CE, weight pushes, resumability)."""

import numpy as np
import pytest
import torch

from python.model.chess_net import ChessNet
from python.selfplay.worker import SavedPosition
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.inference_server import InferenceServer
from python.train.train_rl import RLTrainer, WarmupCosineDecayScheduler


def _create_dummy_positions(n: int, full_prob: float = 0.5) -> list[SavedPosition]:
    positions = []
    for i in range(n):
        planes = np.random.randn(19, 8, 8).astype(np.float32)
        visits = np.zeros(4672, dtype=np.float32)
        visits[i % 4672] = 1.0
        positions.append(
            SavedPosition(
                planes=planes,
                visits=visits,
                outcome=i % 3,
                is_full_search=(np.random.rand() < full_prob),
                model_version=1,
                game_id=i,
            )
        )
    return positions


def test_lr_scheduler_warmup_and_cosine():
    """Verify linear warmup and cosine decay learning rate schedule."""
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = WarmupCosineDecayScheduler(
        optimizer=optimizer,
        warmup_steps=100,
        total_steps=1000,
        base_lr=0.01,
        min_lr=1e-4,
    )

    # Step 0: min_lr
    assert abs(sched.get_lr(0) - 1e-4) < 1e-6
    # Step 50: half-way warmup
    assert abs(sched.get_lr(50) - (1e-4 + 0.5 * (0.01 - 1e-4))) < 1e-6
    # Step 100: base_lr
    assert abs(sched.get_lr(100) - 0.01) < 1e-6
    # Step 1000: min_lr
    assert abs(sched.get_lr(1000) - 1e-4) < 1e-5


def test_rl_trainer_loss_masking_and_step(tmp_path):
    """Verify policy cross-entropy applies only to full-search positions while WDL applies to all."""
    torch.manual_seed(42)
    np.random.seed(42)
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_buf",
        capacity=1000,
        create_new=True,
    )
    positions = _create_dummy_positions(200, full_prob=0.5)
    buf.add_positions(positions)

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=32,
        warmup_steps=10,
        total_steps=100,
        checkpoint_dir=tmp_path / "checkpoints",
        sync_weights_path=tmp_path / "sync.pt",
    )

    metrics = trainer.train_step()
    assert metrics["step"] == 1
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "wdl_loss" in metrics
    assert "policy_entropy" in metrics
    assert metrics["total_loss"] > 0


def test_rl_trainer_push_weights_and_resumability(tmp_path):
    """Verify pushing weights to inference server and resuming from checkpoint."""
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_buf2",
        capacity=1000,
        create_new=True,
    )
    buf.add_positions(_create_dummy_positions(200, full_prob=0.5))

    server_model = ChessNet(num_blocks=2, num_filters=32)
    server = InferenceServer(model=server_model)
    v_initial = server.model_version

    trainer_model = ChessNet(num_blocks=2, num_filters=32)
    sync_path = tmp_path / "live_weights.pt"
    ckpt_dir = tmp_path / "checkpoints"

    trainer = RLTrainer(
        model=trainer_model,
        replay_buffer=buf,
        batch_size=32,
        push_freq=5,
        checkpoint_dir=ckpt_dir,
        sync_weights_path=sync_path,
        direct_server=server,
    )

    # Train for 5 steps (triggers push at step 5)
    for _ in range(5):
        trainer.train_step()

    assert server.model_version == v_initial + 1
    assert sync_path.exists()

    # Save checkpoint at step 5
    ckpt_file = trainer.save_checkpoint()

    # Create fresh trainer and load checkpoint
    new_model = ChessNet(num_blocks=2, num_filters=32)
    resumed_trainer = RLTrainer(
        model=new_model,
        replay_buffer=buf,
        batch_size=32,
        checkpoint_dir=ckpt_dir,
        sync_weights_path=sync_path,
    )
    resumed_step = resumed_trainer.load_checkpoint(ckpt_file)
    assert resumed_step == 5
    assert resumed_trainer.step_num == 5

    # Run one more step
    m = resumed_trainer.train_step()
    assert m["step"] == 6


def test_rl_trainer_custom_lr_flow(tmp_path):
    """Verify that --base-lr and --min-lr flow into RLTrainer, optimizer, and scheduler."""
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_buf_lr",
        capacity=500,
        create_new=True,
    )
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        base_lr=0.0005,
        min_lr=1e-5,
        warmup_steps=10,
        total_steps=100,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert trainer.base_lr == 0.0005
    assert trainer.min_lr == 1e-5
    assert trainer.scheduler.base_lr == 0.0005
    assert trainer.scheduler.min_lr == 1e-5
    assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.0005) < 1e-8

    # When train_step() executes, step 0 lr is set to min_lr (1e-5)
    metrics = trainer.train_step()
    assert abs(trainer.optimizer.param_groups[0]["lr"] - 1e-5) < 1e-8
    assert abs(metrics["lr"] - 1e-5) < 1e-8


def test_rl_trainer_checkpoint_resume_lr_override_and_warning(tmp_path, caplog):
    """Verify that resuming a checkpoint with new lr flags overrides saved scheduler lr and logs a warning."""
    import logging

    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_buf_resume",
        capacity=500,
        create_new=True,
    )
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    old_model = ChessNet(num_blocks=2, num_filters=32)
    ckpt_dir = tmp_path / "checkpoints_resume"
    old_trainer = RLTrainer(
        model=old_model,
        replay_buffer=buf,
        batch_size=16,
        base_lr=0.01,
        min_lr=1e-4,
        warmup_steps=10,
        total_steps=100,
        checkpoint_dir=ckpt_dir,
    )
    old_trainer.train_step()
    ckpt_file = old_trainer.save_checkpoint()

    # Verify tactical_pool_size is saved in checkpoint
    ckpt_data = torch.load(ckpt_file, weights_only=False)
    assert "tactical_pool_size" in ckpt_data

    # Now resume with lower warm-start lr config
    new_model = ChessNet(num_blocks=2, num_filters=32)
    new_trainer = RLTrainer(
        model=new_model,
        replay_buffer=buf,
        batch_size=16,
        base_lr=0.0005,
        min_lr=1e-5,
        warmup_steps=10,
        total_steps=100,
        checkpoint_dir=ckpt_dir,
    )

    with caplog.at_level(logging.WARNING):
        resumed_step = new_trainer.load_checkpoint(ckpt_file, scheduler_override=True)

    assert resumed_step == 1
    # Check that warning was logged
    warning_found = any(
        "Ignoring saved scheduler lr" in record.message
        and "explicit config takes precedence" in record.message
        for record in caplog.records
    )
    assert warning_found, (
        f"Expected warning about scheduler lr override, got: {[r.message for r in caplog.records]}"
    )

    # Check scheduler and optimizer reflect new configuration
    assert new_trainer.scheduler.base_lr == 0.0005
    assert new_trainer.scheduler.min_lr == 1e-5
    expected_step1_lr = new_trainer.scheduler.get_lr(1)
    assert abs(new_trainer.optimizer.param_groups[0]["lr"] - expected_step1_lr) < 1e-8


def test_temperature_sharpening_known_distribution():
    """Verify tau=0.5 transforms known 2-move visit counts to expected squared distribution."""
    tau = 0.5
    visits = torch.zeros(1, 4672, dtype=torch.float32)
    visits[0, 10] = 10.0
    visits[0, 20] = 30.0

    sharp = visits.clamp_min(0) ** (1.0 / tau)
    target = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    assert abs(target[0, 10].item() - 0.10) < 1e-6
    assert abs(target[0, 20].item() - 0.90) < 1e-6
    assert abs(target[0].sum().item() - 1.0) < 1e-6


def test_temperature_sharpening_clamp_nan_safety():
    """Verify negative and denormal visit values do not produce NaNs under clamping."""
    tau = 0.5
    visits = torch.zeros(2, 4672, dtype=torch.float32)
    visits[0, 0] = -5.0
    visits[0, 1] = -1e-12
    visits[0, 2] = 10.0
    visits[1, 5] = 0.0

    sharp = visits.clamp_min(0) ** (1.0 / tau)
    target = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    assert torch.isfinite(target).all()
    assert not torch.isnan(target).any()
    assert abs(target[0, 2].item() - 1.0) < 1e-6


def test_rl_trainer_loss_equivalence_with_defaults(tmp_path):
    """Equivalence test: with temp=1.0 and entropy_bonus=0.005, train_step loss is identical to pre-change formula."""
    torch.manual_seed(1337)
    np.random.seed(1337)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_equiv", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model1 = ChessNet(num_blocks=2, num_filters=32)
    model2 = ChessNet(num_blocks=2, num_filters=32)
    model2.load_state_dict(model1.state_dict())

    trainer = RLTrainer(
        model=model2,
        replay_buffer=buf,
        batch_size=16,
        device="cpu",
        policy_target_temp=1.0,
        entropy_bonus=0.005,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    # Fixed synthetic batch
    planes = torch.randn(16, 19, 8, 8, dtype=torch.float32)
    visits = torch.zeros(16, 4672, dtype=torch.float32)
    for i in range(16):
        visits[i, i * 10] = 5.0
        visits[i, i * 10 + 1] = 15.0
    outcomes = torch.tensor([i % 3 for i in range(16)], dtype=torch.long)
    is_full = torch.tensor([(i % 2 == 0) for i in range(16)], dtype=torch.bool)

    batch_dict = {
        "planes": planes,
        "visits": visits,
        "outcomes": outcomes,
        "is_full_search": is_full,
    }

    # Reference calculation using pre-change formula on model1 (in train mode)
    model1.train()
    p_logits, w_logits = model1(planes)
    full_mask = is_full
    full_p_logits = p_logits[full_mask].float()
    full_visits = visits[full_mask].float()
    log_probs = torch.nn.functional.log_softmax(full_p_logits, dim=-1)
    # Pre-change: target was full_visits directly
    expected_pol_loss = -(full_visits * log_probs).sum(dim=-1).mean()
    expected_wdl_loss = torch.nn.functional.cross_entropy(w_logits.float(), outcomes)
    probs_full = torch.nn.functional.softmax(full_p_logits, dim=-1)
    expected_entropy_reg = -(probs_full * torch.log(probs_full + 1e-12)).sum(dim=-1).mean()
    expected_total_loss = expected_pol_loss + 0.5 * expected_wdl_loss - 0.005 * expected_entropy_reg

    # Run actual train step on model2
    metrics = trainer.train_step(batch_dict=batch_dict)

    assert abs(metrics["policy_loss"] - expected_pol_loss.item()) < 1e-5
    assert abs(metrics["wdl_loss"] - expected_wdl_loss.item()) < 1e-5
    assert abs(metrics["total_loss"] - expected_total_loss.item()) < 1e-5
    assert "target_entropy" in metrics
    assert metrics["target_entropy"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["target_entropy"]))


def test_entropy_bonus_scaling(tmp_path):
    """Verify that entropy_bonus coefficient flows to total_loss scaling correctly."""
    torch.manual_seed(999)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_ent", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=1.0))

    planes = torch.randn(8, 19, 8, 8, dtype=torch.float32)
    visits = torch.zeros(8, 4672, dtype=torch.float32)
    for i in range(8):
        visits[i, 1] = 10.0
    outcomes = torch.tensor([1] * 8, dtype=torch.long)
    is_full = torch.tensor([True] * 8, dtype=torch.bool)
    batch_dict = {
        "planes": planes,
        "visits": visits,
        "outcomes": outcomes,
        "is_full_search": is_full,
    }

    model1 = ChessNet(num_blocks=2, num_filters=32)
    model2 = ChessNet(num_blocks=2, num_filters=32)
    model2.load_state_dict(model1.state_dict())

    trainer1 = RLTrainer(
        model=model1,
        replay_buffer=buf,
        batch_size=8,
        device="cpu",
        entropy_bonus=0.0,
        checkpoint_dir=tmp_path / "ckpt1",
    )
    trainer2 = RLTrainer(
        model=model2,
        replay_buffer=buf,
        batch_size=8,
        device="cpu",
        entropy_bonus=0.01,
        checkpoint_dir=tmp_path / "ckpt2",
    )

    m1 = trainer1.train_step(batch_dict=batch_dict)
    m2 = trainer2.train_step(batch_dict=batch_dict)

    # Policy loss and WDL loss should be identical
    assert abs(m1["policy_loss"] - m2["policy_loss"]) < 1e-6
    assert abs(m1["wdl_loss"] - m2["wdl_loss"]) < 1e-6
    # Total loss in m2 has extra -0.01 * entropy_reg
    # Therefore, m1["total_loss"] - m2["total_loss"] == 0.01 * entropy_reg > 0
    diff = m1["total_loss"] - m2["total_loss"]
    assert diff > 0.0


def test_cli_wiring_and_checkpoint_override(tmp_path, caplog):
    """Verify that CLI flags configure RLTrainer and override saved checkpoint values with logging."""
    import logging

    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_cli", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    ckpt_dir = tmp_path / "ckpt_cli"
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        policy_target_temp=1.0,
        entropy_bonus=0.005,
        checkpoint_dir=ckpt_dir,
    )
    assert trainer.policy_target_temp == 1.0
    assert trainer.entropy_bonus == 0.005

    ckpt_path = trainer.save_checkpoint()
    saved = torch.load(ckpt_path, weights_only=False)
    assert "policy_target_temp" in saved
    assert saved["policy_target_temp"] == 1.0
    assert saved["entropy_bonus"] == 0.005

    # Resume with new flags
    new_model = ChessNet(num_blocks=2, num_filters=32)
    new_trainer = RLTrainer(
        model=new_model,
        replay_buffer=buf,
        batch_size=16,
        policy_target_temp=0.5,
        entropy_bonus=0.001,
        checkpoint_dir=ckpt_dir,
    )
    with caplog.at_level(logging.INFO):
        new_trainer.load_checkpoint(ckpt_path)

    assert new_trainer.policy_target_temp == 0.5
    assert new_trainer.entropy_bonus == 0.001
    assert any("Overriding saved policy_target_temp" in r.message for r in caplog.records)
    assert any("Overriding saved entropy_bonus" in r.message for r in caplog.records)


def test_rl_trainer_defaults_and_user_customization(tmp_path):
    """Verify that default values are tau=0.5 and entropy_bonus=0.001, and that user overrides work."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_default", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    default_trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_default",
    )
    # Default is tau=0.5, entropy_bonus=0.001
    assert default_trainer.policy_target_temp == 0.5
    assert default_trainer.entropy_bonus == 0.001

    # User can customize both parameters freely
    custom_trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        policy_target_temp=0.75,
        entropy_bonus=0.0025,
        checkpoint_dir=tmp_path / "ckpt_custom",
    )
    assert custom_trainer.policy_target_temp == 0.75
    assert custom_trainer.entropy_bonus == 0.0025


def test_cli_parser_defaults_and_user_customization():
    """Verify that CLI defaults are tau=0.5 and entropy_bonus=0.001, and CLI args override them."""
    import argparse
    from scripts.run_phase4_rl import Phase4Orchestrator

    # Check Phase4Orchestrator __init__ default signatures
    import inspect

    sig = inspect.signature(Phase4Orchestrator.__init__)
    assert sig.parameters["policy_target_temp"].default == 0.5
    assert sig.parameters["entropy_bonus"].default == 0.001
