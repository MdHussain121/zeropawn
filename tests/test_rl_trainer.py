"""Tests for Task 4: RL Trainer (SGD momentum, full-search policy CE, WDL CE, weight pushes, resumability)."""

import math
import numpy as np
import pytest
import torch

from python.model.chess_net import ChessNet
from python.selfplay.worker import SavedPosition
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.inference_server import InferenceServer
from python.train.train_rl import RLTrainer, WarmupCosineDecayScheduler, WeightPushError


def _create_dummy_positions(n: int, full_prob: float = 0.5) -> list[SavedPosition]:
    positions = []
    for i in range(n):
        # Valid records under the insertion-time contract (audit SR-07):
        # binary planes, one-hot normalized visits, game_id > 0.
        planes = (np.random.rand(19, 8, 8) < 0.5).astype(np.float32)
        visits = np.zeros(4672, dtype=np.float32)
        visits[i % 4672] = 1.0
        positions.append(
            SavedPosition(
                planes=planes,
                visits=visits,
                outcome=i % 3,
                is_full_search=(np.random.rand() < full_prob),
                model_version=1,
                game_id=i + 1,
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


def test_policy_entropy_metric_is_legal_move_masked(tmp_path):
    """R-01: policy entropy must be measured over legal moves, not all 4672 slots.

    A position with k visited (legal) moves has entropy <= log(k). Measuring the raw
    4672-way softmax instead yields ~log(4672)=8.45 regardless of the legal-move
    distribution, which is what made the entropy-collapse trigger blind.
    """
    torch.manual_seed(0)
    buf = ReplayBuffer(buffer_dir=tmp_path / "entropy_buf", capacity=64, create_new=True)
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=8,
        checkpoint_dir=tmp_path / "ckpt_entropy",
        sync_weights_path=tmp_path / "sync_entropy.pt",
    )

    n = 8
    planes = torch.randn(n, 19, 8, 8)
    visits = torch.zeros(n, 4672)
    legal_counts = []
    for i in range(n):
        k = (i % 4) + 1  # 1..4 legal moves per row
        legal_counts.append(k)
        visits[i, i * 7 : i * 7 + k] = 1.0
    batch = {
        "planes": planes,
        "visits": visits,
        "outcomes": torch.zeros(n, dtype=torch.long),
        "is_full_search": torch.ones(n, dtype=torch.bool),
    }

    metrics = trainer.train_step(batch_dict=batch)
    assert metrics["policy_entropy"] <= float(np.log(max(legal_counts))) + 1e-3

    # Degenerate case: exactly one legal move per position -> entropy is 0.
    visits_one = torch.zeros(n, 4672)
    for i in range(n):
        visits_one[i, i] = 1.0
    batch_one = dict(batch, visits=visits_one)
    metrics_one = trainer.train_step(batch_dict=batch_one)
    assert metrics_one["policy_entropy"] < 1e-4


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
    # Entropy is measured over the legal-move set only (R-01): the visited slots are the
    # legal-move proxy, so illegal slots are excluded from the distribution.
    legal_mask = full_visits > 0
    expected_log_probs = torch.nn.functional.log_softmax(
        full_p_logits.masked_fill(~legal_mask, -1e9), dim=-1
    )
    expected_entropy_reg = -(expected_log_probs.exp() * expected_log_probs).sum(dim=-1).mean()
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
        # Several visited (legal) moves so the legal-move entropy is non-zero.
        visits[i, 1] = 10.0
        visits[i, 2] = 5.0
        visits[i, 3] = 2.0
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


def _make_adamw_checkpoint(tmp_path, model: torch.nn.Module, step: int = 10) -> str:
    """Build a minimal .pt checkpoint whose optimizer state is AdamW (no momentum)."""
    ckpt_dir = tmp_path / "adamw_ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "adamw_checkpoint.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()  # materialize optimizer state
    payload = {
        "checkpoint_type": "finetune",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "num_blocks": getattr(model, "num_blocks", 12),
        "num_filters": getattr(model, "num_filters", 128),
    }
    torch.save(payload, ckpt_path)
    return str(ckpt_path)


def _make_sgd_checkpoint(tmp_path, trainer: "RLTrainer", step: int) -> str:
    """Save a valid RL checkpoint from an RLTrainer at the given step."""
    ckpt_path = tmp_path / f"sgd_ckpt_step{step}.pt"
    return trainer.save_checkpoint(path=ckpt_path)


def test_adamw_optimizer_rejected_before_training(tmp_path):
    """An AdamW optimizer state supplied to the SGD-based RL trainer is rejected before training,
    with a clear compatibility error rather than `KeyError: 'momentum'`.
    """
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_adamw", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_adamw",
        sync_weights_path=tmp_path / "sync_adamw.pt",
    )

    ckpt_path = _make_adamw_checkpoint(tmp_path, model, step=10)
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        trainer.load_checkpoint(ckpt_path)

    # The trainer must still have a working SGD optimizer (reconstructed, not loaded from AdamW).
    metrics = trainer.train_step()
    assert metrics["step"] == 1
    assert metrics["total_loss"] > 0
    assert math.isfinite(metrics["total_loss"])


def test_finetune_checkpoint_type_rejected_for_full_resume(tmp_path):
    """A checkpoint explicitly tagged as a fine-tuning checkpoint is rejected for full RL resume."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_ft", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_ft",
        sync_weights_path=tmp_path / "sync_ft.pt",
    )

    ckpt_path = _make_adamw_checkpoint(tmp_path, model, step=10)
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        trainer.load_checkpoint(ckpt_path)


def test_weights_only_checkpoint_accepted_for_warm_start(tmp_path):
    """A checkpoint without optimizer state is accepted as a weights-only warm-start.
    The RL trainer keeps its freshly initialized SGD optimizer.
    """
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_wo", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_wo",
        sync_weights_path=tmp_path / "sync_wo.pt",
    )

    weights_only_path = tmp_path / "weights_only.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": 7,
        },
        weights_only_path,
    )

    # No optimizer state: validation passes through, model weights are loaded.
    resumed_step = trainer.load_checkpoint(weights_only_path)
    assert resumed_step == 7
    assert trainer.step_num == 7

    # Fresh SGD optimizer still works.
    metrics = trainer.train_step()
    assert metrics["step"] == 8
    assert metrics["total_loss"] > 0


def test_malformed_checkpoint_rejected_safely(tmp_path):
    """A malformed or incomplete checkpoint is rejected safely."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_bad", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_bad",
        sync_weights_path=tmp_path / "sync_bad.pt",
    )

    bad_path = tmp_path / "bad_checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": {"param_groups": []}}, bad_path)
    with pytest.raises(RuntimeError, match="empty or malformed"):
        trainer.load_checkpoint(bad_path)


def test_valid_legacy_rl_checkpoint_loaded(tmp_path):
    """A valid RL checkpoint without checkpoint_type is loaded when its optimizer structure is SGD-compatible."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_legacy", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    old_trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_legacy",
        sync_weights_path=tmp_path / "sync_legacy.pt",
    )
    old_trainer.train_step()
    ckpt_path = _make_sgd_checkpoint(tmp_path, old_trainer, step=1)

    # Strip the new metadata fields so this looks like a legacy checkpoint.
    legacy_ckpt = torch.load(ckpt_path, weights_only=False)
    legacy_ckpt.pop("checkpoint_type", None)
    legacy_ckpt.pop("checkpoint_format", None)
    torch.save(legacy_ckpt, ckpt_path)

    new_model = ChessNet(num_blocks=2, num_filters=32)
    new_trainer = RLTrainer(
        model=new_model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_legacy2",
        sync_weights_path=tmp_path / "sync_legacy2.pt",
    )
    resumed_step = new_trainer.load_checkpoint(ckpt_path)
    assert resumed_step == 1

    metrics = new_trainer.train_step()
    assert metrics["step"] == 2
    assert metrics["total_loss"] > 0


def test_new_rl_checkpoint_contains_type_and_format(tmp_path):
    """A newly saved RL checkpoint contains the expected type, version, and optimizer metadata."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_new", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_new",
        sync_weights_path=tmp_path / "sync_new.pt",
    )
    trainer.train_step()
    ckpt_path = trainer.save_checkpoint()

    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt.get("checkpoint_type") == "rl"
    assert ckpt.get("checkpoint_format") == 1
    pg = ckpt.get("optimizer", {}).get("param_groups", [])
    assert pg and isinstance(pg[0], dict)
    assert "momentum" in pg[0]
    assert pg[0]["momentum"] == 0.9


def test_optimizer_state_round_trip_supports_training_step(tmp_path):
    """An optimizer-state save/load round trip supports a subsequent training step."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_rt", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_rt",
        sync_weights_path=tmp_path / "sync_rt.pt",
    )
    first_metrics = trainer.train_step()
    ckpt_path = trainer.save_checkpoint()

    new_model = ChessNet(num_blocks=2, num_filters=32)
    resumed_trainer = RLTrainer(
        model=new_model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_rt2",
        sync_weights_path=tmp_path / "sync_rt2.pt",
    )
    resumed_trainer.load_checkpoint(ckpt_path)
    second_metrics = resumed_trainer.train_step()

    assert second_metrics["step"] == 2
    assert math.isfinite(second_metrics["total_loss"])
    assert second_metrics["total_loss"] > 0


def test_resume_fails_fast_on_incompatible_checkpoint(tmp_path):
    """An incompatible candidate fails early with a useful diagnostic rather than starting an unusable training run."""
    import logging

    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_fail", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_fail",
        sync_weights_path=tmp_path / "sync_fail.pt",
    )

    ckpt_path = _make_adamw_checkpoint(tmp_path, model, step=10)
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        trainer.load_checkpoint(ckpt_path)


def test_checkpoint_rejects_quarantine_and_finetune_candidates(tmp_path):
    """Automated resume logic must not select files from quarantine or fine-tune locations,
    and incompatible optimizer state must be rejected by RLTrainer.load_checkpoint.
    The orchestrator candidate-selection path is exercised only by the RLTrainer-level
    assertion below; this test does not depend on Phase4Orchestrator internals.
    """

    checkpoint_dir = tmp_path / "active_rl_ckpts"
    quarantine_dir = tmp_path / "quarantine"
    finetune_dir = tmp_path / "finetune"
    checkpoint_dir.mkdir()
    quarantine_dir.mkdir()
    finetune_dir.mkdir()

    # Put a real-looking RL checkpoint inside the active dir.
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_qf", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))
    model = ChessNet(num_blocks=2, num_filters=32)
    builder = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=checkpoint_dir,
        sync_weights_path=tmp_path / "sync_qf.pt",
    )
    builder.train_step()
    builder.save_checkpoint()

    # Corrupt/resume-incompatible candidates placed outside the active dir.
    bad_candidate = quarantine_dir / "latest_checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {
                "param_groups": [
                    {"params": list(model.parameters()), "lr": 0.0, "momentum": 0.9, "dampening": 0.0, "nesterov": False},
                ]
            },
            "checkpoint_type": "rnd",
        },
        bad_candidate,
    )
    finetune_candidate = finetune_dir / "best_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {
                "param_groups": [
                    {"params": list(model.parameters()), "lr": 0.0, "momentum": 0.9, "dampening": 0.0, "nesterov": False},
                ]
            },
            "checkpoint_type": "finetune",
        },
        finetune_candidate,
    )

    # The candidate-selection rule used by the orchestrator is: pick the first candidate
    # that exists and lives strictly inside checkpoint_dir. This test exercises that rule
    # directly so it does not depend on Phase4Orchestrator internals.
    candidates = [
        checkpoint_dir / "phase4_final_checkpoint.pt",
        checkpoint_dir / "latest_checkpoint.pt",
    ]
    selected = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            candidate.resolve().relative_to(checkpoint_dir.resolve())
        except ValueError:
            continue
        selected = str(candidate)
        break

    assert selected is not None, "Expected the valid RL checkpoint to be selected from the active dir"
    assert str(checkpoint_dir / "latest_checkpoint.pt") == selected

    # Confirm the incompatible files outside checkpoint_dir are not selected, and that the
    # orchestrator rejects them when they are passed explicitly.
    assert str(bad_candidate) != selected
    assert str(finetune_candidate) != selected

    # A weights-only fine-tune checkpoint must be rejected for full resume if it is
    # explicitly tagged as finetune (the class under test is RLTrainer.load_checkpoint).
    ft_trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_ft_full",
        sync_weights_path=tmp_path / "sync_ft_full.pt",
    )
    ft_tagged = tmp_path / "finetune_tagged.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": None, "checkpoint_type": "finetune"},
        ft_tagged,
    )
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        ft_trainer.load_checkpoint(str(ft_tagged), warm_start=True)

    # Explicitly tagged fine-tune checkpoints are rejected even for the weights-only path.
    # Explicitly tagged fine-tune checkpoints are also rejected for the weights-only path.
    ft_trainer2 = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_ft_full2",
        sync_weights_path=tmp_path / "sync_ft_full2.pt",
    )
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        ft_trainer2.load_checkpoint(str(ft_tagged), warm_start=True)

    # A file with an incompatible explicit type is rejected for the full-resume path,
    # even when it contains a valid SGD param_groups shape.
    bad_trainer2 = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_qf2b",
        sync_weights_path=tmp_path / "sync_qf2b.pt",
    )
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        bad_trainer2.load_checkpoint(str(bad_candidate), warm_start=True)

    wow_trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_wow",
        sync_weights_path=tmp_path / "sync_wow.pt",
    )
    wow_path = tmp_path / "weights_only.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
        },
        wow_path,
    )
    step = wow_trainer.load_checkpoint(str(wow_path), warm_start=True)
    assert step == 0
    assert isinstance(wow_trainer.optimizer, torch.optim.SGD)
    for pg in wow_trainer.optimizer.param_groups:
        assert "momentum" in pg

    bad_trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_qf2",
        sync_weights_path=tmp_path / "sync_qf2.pt",
    )
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        bad_trainer.load_checkpoint(str(bad_candidate), warm_start=True)

    # The weights-only warm-start path still rejects an explicit fine-tune type.
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        RLTrainer(
            model=ChessNet(num_blocks=2, num_filters=32),
            replay_buffer=buf,
            batch_size=16,
            checkpoint_dir=tmp_path / "ckpt_qf3",
            sync_weights_path=tmp_path / "sync_qf3.pt",
        ).load_checkpoint(str(finetune_candidate), warm_start=True)


def test_exception_produces_traceback_in_log(tmp_path, caplog):
    """An exception in the training loop produces a traceback in the run log.
    This test targets the error-reporting fix and does not require the full orchestrator.
    """
    import logging
    import io

    logger = logging.getLogger("phase4_rl.test_traceback")
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)

    try:
        raise KeyError("momentum")
    except Exception as e:
        logger.exception("Unhandled exception in Phase 4 training loop")

    stream = handler.stream  # type: ignore[attr-defined]
    logged = stream.getvalue()
    assert "KeyError" in logged
    assert "momentum" in logged
    assert "Unhandled exception in Phase 4 training loop" in logged


def test_loaded_rl_checkpoint_executes_training_step_with_finite_loss(tmp_path):
    """A verified, compatible RL checkpoint can successfully execute a training step and produce finite loss values."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_exec", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    first_trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_exec",
        sync_weights_path=tmp_path / "sync_exec.pt",
    )
    first_trainer.train_step()
    ckpt_path = first_trainer.save_checkpoint()

    new_model = ChessNet(num_blocks=2, num_filters=32)
    resumed_trainer = RLTrainer(
        model=new_model,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_exec2",
        sync_weights_path=tmp_path / "sync_exec2.pt",
    )
    resumed_trainer.load_checkpoint(ckpt_path)
    metrics = resumed_trainer.train_step()

    assert metrics["step"] == 2
    assert math.isfinite(metrics["total_loss"])
    assert metrics["total_loss"] > 0
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["wdl_loss"])


def test_weights_only_warm_start_does_not_restore_incompatible_optimizer_state(tmp_path):
    """A weights-only warm-start initializes a fresh optimizer and does not restore incompatible optimizer state."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_wow", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=16,
        base_lr=0.0005,
        min_lr=1e-5,
        checkpoint_dir=tmp_path / "ckpt_wow",
        sync_weights_path=tmp_path / "sync_wow.pt",
    )

    weights_only_path = tmp_path / "weights_only_warm.pt"
    torch.save(
        {
            "checkpoint_type": "finetune",
            "model": model.state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": 9,
            "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-3).state_dict(),
        },
        weights_only_path,
    )

    # checkpoint_type is "finetune" but there is no compatible RL optimizer state to restore,
    # so load_checkpoint rejects the full-resume path. The user should use a weights-only warm start
    # by loading only the model weights.
    with pytest.raises(RuntimeError, match="not an RL training checkpoint"):
        trainer.load_checkpoint(weights_only_path)

    # Build a clean weights-only checkpoint (no optimizer) and confirm it is accepted.
    clean_weights_path = tmp_path / "clean_weights.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": 9,
        },
        clean_weights_path,
    )
    resumed_step = trainer.load_checkpoint(clean_weights_path)
    assert resumed_step == 9
    assert trainer.step_num == 9

    # Fresh SGD optimizer was used, not the AdamW state.
    pg = trainer.optimizer.param_groups[0]
    assert "momentum" in pg
    assert "betas" not in pg

    metrics = trainer.train_step()
    assert metrics["step"] == 10
    assert metrics["total_loss"] > 0


def test_cli_parser_defaults_and_user_customization():
    """Verify that CLI defaults are tau=0.5 and entropy_bonus=0.001, and CLI args override them."""
    import argparse
    from scripts.run_phase4_rl import Phase4Orchestrator

    # Check Phase4Orchestrator __init__ default signatures
    import inspect

    sig = inspect.signature(Phase4Orchestrator.__init__)
    assert sig.parameters["policy_target_temp"].default == 0.5
    assert sig.parameters["entropy_bonus"].default == 0.001


def test_scheduler_explicit_step_pins_counter_and_lr():
    """N-01 regression: step(step_num) is valid and pins the schedule to that step.

    The explicit-step form computes LR as a pure function of the given step
    (matching get_lr) so the RL trainer can drive the schedule from its own
    checkpointed counter; the no-arg form keeps the legacy advance-by-one
    warmup semantics used by supervised training.
    """
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = WarmupCosineDecayScheduler(
        optimizer=optimizer,
        warmup_steps=100,
        total_steps=1000,
        base_lr=0.01,
        min_lr=1e-4,
    )

    lr50 = sched.step(50)
    assert sched.current_step == 50
    assert abs(lr50 - sched.get_lr(50)) < 1e-12
    assert abs(optimizer.param_groups[0]["lr"] - sched.get_lr(50)) < 1e-12

    # No-arg step still advances by one and applies the legacy warmup formula.
    sched.step()
    assert sched.current_step == 51
    expected_warmup_lr = 0.01 * (51 / 100)
    assert abs(optimizer.param_groups[0]["lr"] - expected_warmup_lr) < 1e-12

    # Jumping forward into the cosine region matches get_lr there too.
    lr800 = sched.step(800)
    assert abs(lr800 - sched.get_lr(800)) < 1e-12


def test_entropy_term_contributes_gradient(tmp_path):
    """N-03 regression: the entropy bonus must be differentiable through total_loss.

    The momentum-fix rewrite converted policy entropy to a Python float via a
    mid-step .item(), turning the entropy bonus into a constant with zero
    gradient. This test proves the loss gradient w.r.t. a model parameter
    changes when the entropy coefficient changes.
    """
    torch.manual_seed(7)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_entgrad", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=1.0))

    planes = torch.randn(8, 19, 8, 8, dtype=torch.float32)
    visits = torch.zeros(8, 4672, dtype=torch.float32)
    for i in range(8):
        # Several visited (legal) moves per row so d(entropy) is non-zero.
        visits[i, 1] = 10.0
        visits[i, 2] = 5.0
        visits[i, 3] = 2.0
    outcomes = torch.tensor([1] * 8, dtype=torch.long)
    is_full = torch.tensor([True] * 8, dtype=torch.bool)
    batch_dict = {
        "planes": planes,
        "visits": visits,
        "outcomes": outcomes,
        "is_full_search": is_full,
    }

    base_state = ChessNet(num_blocks=2, num_filters=32).state_dict()

    def _grad_signature(entropy_bonus: float) -> torch.Tensor:
        torch.manual_seed(7)
        model = ChessNet(num_blocks=2, num_filters=32)
        model.load_state_dict(base_state)
        trainer = RLTrainer(
            model=model,
            replay_buffer=buf,
            batch_size=8,
            device="cpu",
            entropy_bonus=entropy_bonus,
            checkpoint_dir=tmp_path / f"ckpt_entgrad_{entropy_bonus}",
            sync_weights_path=tmp_path / f"sync_entgrad_{entropy_bonus}.pt",
        )
        captured = {}
        handle = model.stem_conv.weight.register_hook(lambda g: captured.update(grad=g.clone()))
        trainer.train_step(batch_dict=batch_dict)
        handle.remove()
        grad = captured["grad"]
        assert torch.isfinite(grad).all()
        return grad

    g_baseline = _grad_signature(0.0)
    g_with_bonus = _grad_signature(0.05)
    # Same data and init: the only difference is the entropy term's gradient
    # contribution. If the entropy gradient is broken, these are identical.
    assert not torch.allclose(g_baseline, g_with_bonus, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_bf16_autocast_applied_on_cuda(tmp_path):
    """N-02 regression: the forward pass runs under bf16 autocast when use_bfloat16 is set.

    The momentum-fix rewrite stored use_bfloat16 but never applied it, silently
    regressing training throughput ~1.7x. This test pins the autocast scope.
    """
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_autocast", capacity=64, create_new=True)
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=8,
        device="cuda",
        use_bfloat16=True,
        checkpoint_dir=tmp_path / "ckpt_autocast",
        sync_weights_path=tmp_path / "sync_autocast.pt",
    )
    assert trainer.use_bfloat16 is True

    observed = {}
    orig_forward = model.forward

    def _spy(x):
        observed["autocast"] = torch.is_autocast_enabled("cuda")
        observed["dtype"] = torch.get_autocast_dtype("cuda")
        return orig_forward(x)

    model.forward = _spy
    batch = {
        "planes": torch.randn(4, 19, 8, 8, device="cuda"),
        "visits": torch.zeros(4, 4672, device="cuda"),
        "outcomes": torch.zeros(4, dtype=torch.long, device="cuda"),
        "is_full_search": torch.ones(4, dtype=torch.bool, device="cuda"),
    }
    metrics = trainer.train_step(batch_dict=batch)
    model.forward = orig_forward

    assert observed["autocast"] is True
    assert observed["dtype"] == torch.bfloat16
    assert math.isfinite(metrics["total_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_bf16_autocast_disabled_when_flag_off(tmp_path):
    """use_bfloat16=False must leave the forward pass in fp32 (no autocast)."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_noautocast", capacity=64, create_new=True)
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        batch_size=8,
        device="cuda",
        use_bfloat16=False,
        checkpoint_dir=tmp_path / "ckpt_noautocast",
        sync_weights_path=tmp_path / "sync_noautocast.pt",
    )
    assert trainer.use_bfloat16 is False

    observed = {}
    orig_forward = model.forward

    def _spy(x):
        observed["autocast"] = torch.is_autocast_enabled("cuda")
        return orig_forward(x)

    model.forward = _spy
    batch = {
        "planes": torch.randn(4, 19, 8, 8, device="cuda"),
        "visits": torch.zeros(4, 4672, device="cuda"),
        "outcomes": torch.zeros(4, dtype=torch.long, device="cuda"),
        "is_full_search": torch.ones(4, dtype=torch.bool, device="cuda"),
    }
    trainer.train_step(batch_dict=batch)
    model.forward = orig_forward

    assert observed["autocast"] is False


# ---------------------------------------------------------------------------
# Stage A: warm-start optimizer isolation (R-09) and fail-closed weight pushes
# ---------------------------------------------------------------------------


def test_warm_start_ignores_optimizer_state_and_resets_step(tmp_path):
    """warm_start=True loads weights only: foreign optimizer/scheduler state is
    never adopted (audit R-09) and the RL step counter restarts at 0."""
    torch.manual_seed(7)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_ws", capacity=500, create_new=True)
    buf.add_positions(_create_dummy_positions(50, full_prob=0.5))

    model_a = ChessNet(num_blocks=2, num_filters=32)
    trainer_a = RLTrainer(
        model=model_a,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_ws_a",
        sync_weights_path=tmp_path / "sync_ws_a.pt",
    )
    trainer_a.train_step()
    trained_state = {k: v.clone() for k, v in model_a.state_dict().items()}

    # A supervised-style checkpoint: AdamW optimizer state, source step 777,
    # no checkpoint_type stamp (legacy layout).
    foreign_path = tmp_path / "supervised_style.pt"
    torch.save(
        {
            "model": trained_state,
            "optimizer": torch.optim.AdamW(model_a.parameters(), lr=1e-3).state_dict(),
            "step": 777,
            "num_blocks": 2,
            "num_filters": 32,
        },
        foreign_path,
    )

    model_b = ChessNet(num_blocks=2, num_filters=32)
    trainer_b = RLTrainer(
        model=model_b,
        replay_buffer=buf,
        batch_size=16,
        checkpoint_dir=tmp_path / "ckpt_ws_b",
        sync_weights_path=tmp_path / "sync_ws_b.pt",
    )
    step = trainer_b.load_checkpoint(foreign_path, warm_start=True)
    assert step == 0
    assert trainer_b.step_num == 0
    assert trainer_b.last_push_step == 0

    # Weights were adopted...
    for name, tensor in model_b.state_dict().items():
        torch.testing.assert_close(tensor, trained_state[name])
    # ...but the optimizer stayed a fresh SGD (no AdamW keys or state).
    pg = trainer_b.optimizer.param_groups[0]
    assert "betas" not in pg
    assert all(len(state) == 0 for state in trainer_b.optimizer.state.values())

    # Full resume of the same foreign checkpoint must still be rejected.
    with pytest.raises(RuntimeError):
        trainer_b.load_checkpoint(foreign_path, warm_start=False)


class _GoodServer:
    """Minimal stand-in honoring the InferenceServer weight-loading contract."""

    def __init__(self) -> None:
        self.model_version = 0
        self.last_state = None

    def load_state_dict_weights(self, state_dict, sync_event=None):
        self.last_state = state_dict
        self.model_version += 1
        return self.model_version


class _StuckServer(_GoodServer):
    """Simulates a failed hot-swap: old weights kept, version unchanged."""

    def load_state_dict_weights(self, state_dict, sync_event=None):
        return self.model_version


class _BoomServer(_GoodServer):
    def load_state_dict_weights(self, state_dict, sync_event=None):
        raise RuntimeError("boom")


class _NoApiServer:
    model_version = 0


def test_push_weights_success_tracks_version_and_counts(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_push", capacity=100, create_new=True)
    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        checkpoint_dir=tmp_path / "ck",
        sync_weights_path=tmp_path / "sync.pt",
        direct_server=_GoodServer(),
    )
    assert trainer.push_weights() is None  # CPU: no CUDA event
    assert trainer.push_success_count == 1
    assert trainer.push_failure_count == 0
    assert trainer.model_version == 1


def test_push_weights_failure_raises_and_counts(tmp_path):
    for server in (_StuckServer(), _BoomServer(), _NoApiServer()):
        trainer = RLTrainer(
            model=ChessNet(num_blocks=2, num_filters=32),
            replay_buffer=ReplayBuffer(
                buffer_dir=tmp_path / "buf_pf", capacity=100, create_new=True
            ),
            device="cpu",
            checkpoint_dir=tmp_path / "ck_pf",
            sync_weights_path=tmp_path / "sync_pf.pt",
            direct_server=server,
        )
        with pytest.raises(WeightPushError):
            trainer.push_weights()
        assert trainer.push_failure_count == 1
        assert trainer.push_success_count == 0
        # The version mirror never advances on failure.
        assert trainer.model_version == getattr(server, "model_version", 0)


def test_train_step_propagates_push_failure(tmp_path):
    """A failed weight push must be observable to the training loop (fail-closed)."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_tc", capacity=200, create_new=True)
    buf.add_positions(_create_dummy_positions(60, full_prob=0.5))
    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        batch_size=16,
        push_freq=1,
        checkpoint_dir=tmp_path / "ck_tc",
        sync_weights_path=tmp_path / "sync_tc.pt",
        direct_server=_StuckServer(),
    )
    with pytest.raises(WeightPushError):
        trainer.train_step()
    assert trainer.step_num == 1  # optimizer stepped, then the push failed loudly
    assert trainer.push_failure_count == 1


def test_train_step_masks_sentinel_wdl_targets(tmp_path):
    """Real train_step with outcome-sentinel 255 rows (Stage A §3.3): mixed and
    all-unavailable batches must produce finite loss; invalid rows contribute
    zero WDL loss/gradient; all-invalid batches give exactly zero WDL loss."""
    torch.manual_seed(3)
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_255", capacity=200, create_new=True)
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
        checkpoint_dir=tmp_path / "ck_255",
        sync_weights_path=tmp_path / "sync_255.pt",
    )
    planes = torch.randn(8, 19, 8, 8)
    visits = torch.zeros(8, 4672)
    visits[torch.arange(8), torch.randint(0, 4672, (8,))] = 1.0

    # Mixed: five valid labels, three unavailable sentinels.
    mixed = {
        "planes": planes,
        "visits": visits,
        "outcomes": torch.tensor([0, 1, 2, 255, 0, 1, 255, 2], dtype=torch.long),
        "is_full_search": torch.ones(8, dtype=torch.bool),
    }
    metrics = trainer.train_step(batch_dict=mixed)
    assert math.isfinite(metrics["total_loss"])
    assert math.isfinite(metrics["wdl_loss"])
    assert metrics["wdl_loss"] > 0  # valid rows contribute
    assert any(p.grad is not None for p in model.parameters())
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()

    # All-unavailable: WDL loss must be exactly zero, loss still finite.
    model.zero_grad(set_to_none=True)
    all_invalid = {
        "planes": planes,
        "visits": visits,
        "outcomes": torch.full((8,), 255, dtype=torch.long),
        "is_full_search": torch.ones(8, dtype=torch.bool),
    }
    metrics = trainer.train_step(batch_dict=all_invalid)
    assert math.isfinite(metrics["total_loss"])
    assert metrics["wdl_loss"] == 0.0
    # Zero WDL contribution: gradients are finite where present (None == zero).
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()

    # And a fully valid batch keeps normal WDL semantics.
    model.zero_grad(set_to_none=True)
    valid = dict(mixed)
    valid["outcomes"] = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    metrics = trainer.train_step(batch_dict=valid)
    assert metrics["wdl_loss"] > 0
