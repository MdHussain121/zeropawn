"""Tests for Opening Policy Target Construction, Side-to-Move, and Training Step."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

import python
import zeropawn_cpp
from python.data.build_opening_dataset import build_opening_dataset
from python.data.opening_corpus import OpeningCorpus
from python.model.chess_net import ChessNet
from python.train.train_opening_policy import (
    evaluate_opening_policy,
    train_opening_policy,
    train_opening_step,
)


def test_opening_dataset_builder_and_targets(tmp_path: Path):
    """Verify build_opening_dataset constructs valid planes, moves, outcomes, and value targets."""
    corpus = OpeningCorpus()
    board = chess.Board()

    # Add 1. e4 and 1. d4 with high frequency
    m_e4 = chess.Move.from_uci("e2e4")
    m_d4 = chess.Move.from_uci("d2d4")

    for i in range(10):
        corpus.add_position_move(board, m_e4, game_id=f"game_e4_{i}", source="test")
        corpus.add_position_move(board, m_d4, game_id=f"game_d4_{i}", source="test")

    out_dir = tmp_path / "dataset_opening_test"
    summary = build_opening_dataset(corpus, output_dir=out_dir, val_ratio=0.5, seed=42)

    assert summary["total_samples"] >= 1
    assert (out_dir / "train" / "metadata.json").is_file()
    assert (out_dir / "val" / "metadata.json").is_file()

    # Load dataset with BinaryChessDataset
    from python.data.dataset import BinaryChessDataset

    ds = BinaryChessDataset(out_dir / "train")
    assert len(ds) >= 1
    sample = ds[0]

    assert sample["planes"].shape == (19, 8, 8)
    assert 0 <= int(sample["move"]) < 4672
    assert int(sample["outcome"]) in (0, 1, 2)
    assert sample["value_target"].shape == (3,)
    assert torch.isclose(sample["value_target"].sum(), torch.tensor(1.0))


def test_opening_training_smoke_step():
    """Verify train_opening_step executes forward and backward pass stably."""
    device = torch.device("cpu")
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 4
    batch = {
        "planes": torch.randn(B, 19, 8, 8),
        "move": torch.randint(0, 4672, (B,)),
        "outcome": torch.randint(0, 3, (B,)),
        "value_target": torch.tensor([[0.3, 0.4, 0.3]] * B, dtype=torch.float32),
        "ply": torch.tensor([2, 4, 6, 8], dtype=torch.long),
    }

    tot_l, pol_l, wdl_l, top1, top3, entropy = train_opening_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        use_autocast=False,
    )

    assert np.isfinite(tot_l)
    assert np.isfinite(pol_l)
    assert np.isfinite(wdl_l)
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top3 <= 100.0


def test_checkpoint_isolation_rule(tmp_path: Path):
    """Verify that training creates a new checkpoint and does not alter the baseline."""
    # Build tiny dataset
    corpus = OpeningCorpus()
    board = chess.Board()
    for i in range(10):
        corpus.add_position_move(board, chess.Move.from_uci("e2e4"), game_id=f"g{i}", source="test")
        corpus.add_position_move(
            board, chess.Move.from_uci("d2d4"), game_id=f"g_d{i}", source="test"
        )

    # Add Black-to-move position
    b_black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    for i in range(10):
        corpus.add_position_move(
            b_black, chess.Move.from_uci("c7c5"), game_id=f"gb_{i}", source="test"
        )

    ds_dir = tmp_path / "dataset_tiny"
    build_opening_dataset(corpus, output_dir=ds_dir, val_ratio=0.5, seed=42)

    # Fake initial checkpoint
    fake_baseline = tmp_path / "baseline_model.pt"
    initial_net = ChessNet(num_blocks=2, num_filters=32)
    torch.save(
        {
            "model": initial_net.state_dict(),
            "num_blocks": 2,
            "num_filters": 32,
            "step": 1000,
        },
        fake_baseline,
    )

    baseline_mtime_before = fake_baseline.stat().st_mtime
    baseline_bytes_before = fake_baseline.read_bytes()

    out_ckpt_dir = tmp_path / "exp_checkpoints"
    res = train_opening_policy(
        opening_data_dir=ds_dir,
        resume_checkpoint=fake_baseline,
        output_checkpoint_dir=out_ckpt_dir,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_steps=5,
        num_blocks=2,
        num_filters=32,
        device_name="cpu",
    )

    # 1. Output checkpoint directory must have new files
    assert (out_ckpt_dir / "latest_opening_model.pt").is_file()
    assert res["final_step"] > 0

    # 2. Original baseline must remain completely unmodified
    assert fake_baseline.stat().st_mtime == baseline_mtime_before
    assert fake_baseline.read_bytes() == baseline_bytes_before
