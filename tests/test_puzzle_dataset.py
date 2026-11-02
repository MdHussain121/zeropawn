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


def _certify_for_gate(root: Path) -> None:
    """Certify a builder-produced dataset the way production does (audit S2-03.1):
    exhaustive validate_split per split + a validation receipt at the dataset
    root, so the P2-8 fail-closed consumption gate inside train_opening_policy
    certifies the bytes. ``check_build_record=False`` because the puzzle
    builder's BuildRecord carries no per-split output hashes (provenance
    agreement is exercised on the real rebuilt dataset); the full plane scan
    is required because a bounded scan reports BOUNDED => overall='incomplete'
    (P2-5), which is not certifiable."""
    from python.data.dataset_validation import validate_split, write_receipt

    reports = {}
    for split in ("train", "val"):
        split_dir = root / split
        if split_dir.is_dir():
            report = validate_split(split_dir, scan_planes="full", check_build_record=False)
            assert report.overall == "pass", (
                f"{split_dir} fixture must validate exhaustively: {report.summary_line()}"
            )
            reports[split] = report
    assert set(reports) == {"train", "val"}, "puzzle builder must produce train and val splits"
    write_receipt(root, reports, overall="pass")


def test_multi_dataset_training_with_puzzles(dummy_puzzle_csv: Path, tmp_path: Path):
    # 1. Build a dummy opening dataset
    op_dir = tmp_path / "op_ds"
    build_puzzle_dataset(dummy_puzzle_csv, op_dir, min_rating=1500, val_ratio=0.5)

    # 2. Build a dummy puzzle dataset
    puz_dir = tmp_path / "puz_ds"
    build_puzzle_dataset(dummy_puzzle_csv, puz_dir, min_rating=2000, val_ratio=0.5)

    # 3. Certify both builds the way production would (audit S2-03.1): the
    #    builder writes a BuildRecord but no receipt, and train_opening_policy's
    #    P2-8 gate fails closed on uncertified data.
    _certify_for_gate(op_dir)
    _certify_for_gate(puz_dir)

    # 4. Create dummy checkpoint
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

    # The deterministic PuzzleId-hash split (audit F-08) assigns the 3-row
    # fixture differently than the legacy per-row RNG split, so the exact
    # step count differs; assert that training actually ran and produced
    # the checkpoint artifact.
    assert res["final_step"] >= 1
    assert (out_ckpt_dir / "best_opening_model.pt").exists()
