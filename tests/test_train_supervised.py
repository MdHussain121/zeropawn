import pytest
import torch
import torch.nn as nn
from python.model.chess_net import ChessNet
from python.train.train_supervised import (
    WarmupCosineScheduler,
    compute_accuracy_topk,
    compute_policy_entropy,
    train_step,
    save_checkpoint,
)


def test_warmup_cosine_scheduler():
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    sched = WarmupCosineScheduler(
        optimizer, warmup_steps=100, total_steps=1000, base_lr=1e-3, min_lr=1e-5
    )

    # Step 1: linear warmup start
    lr1 = sched.step()
    assert 0.0 < lr1 < 1e-3

    # Step 100: exactly peak base_lr
    for _ in range(99):
        lr_peak = sched.step()
    assert pytest.approx(lr_peak, rel=1e-4) == 1e-3

    # Step 1000: decayed to min_lr
    for _ in range(900):
        lr_end = sched.step()
    assert pytest.approx(lr_end, rel=1e-3) == 1e-5


def test_compute_accuracy_and_entropy():
    logits = torch.zeros(10, 4672)
    logits[:, 42] = 10.0  # Class 42 has highest probability
    targets = torch.full((10,), 42, dtype=torch.long)

    top1, top3 = compute_accuracy_topk(logits, targets, topk=(1, 3))
    assert top1 == 100.0
    assert top3 == 100.0

    entropy = compute_policy_entropy(logits)
    assert entropy >= 0.0


def test_train_step_dummy_batch():
    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    batch = {
        "planes": torch.randn(2, 19, 8, 8),
        "move": torch.tensor([10, 20], dtype=torch.long),
        "value_target": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "ply": torch.tensor([5, 15], dtype=torch.long),
    }

    tot, pol, wdl, top1, top3, ent, *rest = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        use_autocast=False,
    )

    assert not torch.isnan(torch.tensor(tot))
    assert not torch.isnan(torch.tensor(pol))
    assert not torch.isnan(torch.tensor(wdl))
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top3 <= 100.0
    assert ent >= 0.0


def test_save_and_resume_checkpoint(tmp_path):
    ckpt_file = tmp_path / "test_ckpt.pt"
    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100, base_lr=1e-3)

    sched.step()
    save_checkpoint(ckpt_file, model, optimizer, sched, step=1, epoch=0, best_val_loss=1.234)

    assert ckpt_file.exists()
    loaded = torch.save, torch.load(ckpt_file, map_location="cpu")
    data = loaded[1]
    assert data["step"] == 1
    assert data["epoch"] == 0
    assert data["best_val_loss"] == 1.234
    assert "model" in data
    assert "optimizer" in data
    assert "scheduler" in data
