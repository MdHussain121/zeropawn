"""Unit tests for tactical puzzle dataset compilation and multi-dataset supervised training."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import pytest
import torch

from python.data.build_puzzle_dataset import build_puzzle_dataset
from python.data.dataset import BinaryChessDataset
from python.model.chess_net import ChessNet
from python.train.train_opening_policy import train_opening_policy


@pytest.fixture
def dummy_puzzle_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_puzzles.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "PuzzleId",
                "FEN",
                "Moves",
                "Rating",
                "RatingDeviation",
                "Popularity",
                "NbPlays",
                "Themes",
                "GameUrl",
                "OpeningTags",
                "DailyDate",
            ]
        )
        # Puzzle 1: Standard tactical sequence
        # FEN before opponent move: 8/7R/8/5p2/4bk1P/8/2r2K2/6R1 w - - 7 51
        # Opponent move: f2f1, Player solution: f4f3
        writer.writerow(
            [
                "puz1",
                "8/7R/8/5p2/4bk1P/8/2r2K2/6R1 w - - 7 51",
                "f2f1 f4f3",
                "2167",
                "76",
                "94",
                "2029",
                "endgame",
                "url",
                "",
                "",
            ]
        )
        # Puzzle 2: Another tactical line (rating 2400)
        # FEN before: 6k1/pp1r1pp1/1qp1p2p/4P2P/5Q2/1P4R1/P1Pr1PP1/R5K1 b - - 4 23
        # Opponent: b6d4, Player solution: f4f6
        writer.writerow(
            [
                "puz2",
                "6k1/pp1r1pp1/1qp1p2p/4P2P/5Q2/1P4R1/P1Pr1PP1/R5K1 b - - 4 23",
                "b6d4 f4f6",
                "2391",
                "115",
                "66",
                "35",
                "endgame pin",
                "url",
                "",
                "",
            ]
        )
        # Puzzle 3: Low rating (below 2000 threshold)
        writer.writerow(
            [
                "puz3",
                "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
                "f2g3 e6e7",
                "1600",
                "76",
                "95",
                "10282",
                "hangingPiece",
                "url",
                "",
                "",
            ]
        )
    return csv_file


def test_build_puzzle_dataset(dummy_puzzle_csv: Path, tmp_path: Path):
    out_dir = tmp_path / "dataset_puzzles"
    summary = build_puzzle_dataset(
        csv_path=dummy_puzzle_csv,
        output_dir=out_dir,
        min_rating=2000,
        val_ratio=0.5,
    )

    assert summary["total_samples"] == 2  # Puz3 was filtered out (< 2000)
    assert (out_dir / "train" / "planes.bin").exists()
    assert (out_dir / "train" / "policy_targets.bin").exists()
    assert (out_dir / "val" / "planes.bin").exists()

    ds = BinaryChessDataset(out_dir / "train")
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["planes"].shape == (19, 8, 8)
    assert sample["move"].dtype == torch.long
    assert 0 <= sample["move"].item() < 4672
    assert sample["value_target"].shape == (3,)


def test_multi_dataset_training_with_puzzles(dummy_puzzle_csv: Path, tmp_path: Path):
    # 1. Build a dummy opening dataset
    op_dir = tmp_path / "op_ds"
    build_puzzle_dataset(dummy_puzzle_csv, op_dir, min_rating=1500, val_ratio=0.5)

    # 2. Build a dummy puzzle dataset
    puz_dir = tmp_path / "puz_ds"
    build_puzzle_dataset(dummy_puzzle_csv, puz_dir, min_rating=2000, val_ratio=0.5)

    # 3. Create dummy checkpoint
    dummy_model = ChessNet(num_blocks=2, num_filters=32)
    ckpt_path = tmp_path / "dummy_base.pt"
    torch.save({"model": dummy_model.state_dict(), "step": 1}, ckpt_path)

    out_ckpt_dir = tmp_path / "out_ckpts"
    res = train_opening_policy(
        opening_data_dir=op_dir,
        extra_data_dirs=[puz_dir],
        resume_checkpoint=ckpt_path,
        output_checkpoint_dir=out_ckpt_dir,
        epochs=2,
        batch_size=2,
        max_steps=3,
        num_blocks=2,
        num_filters=32,
    )

    assert res["final_step"] == 3
    assert (out_ckpt_dir / "best_opening_model.pt").exists()
