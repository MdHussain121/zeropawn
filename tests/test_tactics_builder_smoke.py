"""Real integration smoke test for the tactical dataset builder (build_tactics.py).

Verifies the complete tactical generation and serialization pipeline:
1. Constructs a valid synthetic puzzle CSV with real FEN and UCI moves.
2. Invokes build_tactics_dataset to exercise the actual builder API.
3. Proves PositionData validation accepts the OUTCOME_UNAVAILABLE (255) sentinel.
4. Proves serialization completes and generates all required binary files on disk.
5. Loads the emitted records back using BinaryChessDataset.
6. Verifies tactical source identification (is_tactical = True).
7. Verifies value target masking (value_target = [0, 0, 0], outcome = -1, ply = 0).
8. Contrasts against standard master positions to prove master-game value encoding is unaltered.
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pytest
import torch

from python.data.build_dataset import OUTCOME_UNAVAILABLE, PositionData, save_dataset_split
from python.data.build_tactics import build_tactics_dataset
from python.data.dataset import BinaryChessDataset


@pytest.fixture
def synthetic_puzzle_csv(tmp_path: Path) -> Path:
    """Create a minimal valid Lichess puzzle CSV with 2 synthetic puzzles."""
    csv_file = tmp_path / "synthetic_puzzles.csv"
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
            ]
        )
        # Puzzle 1: Initial position, opponent plays e4, solution is e5
        writer.writerow(
            [
                "puz01",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "e2e4 e7e5",
                "1500",
                "75",
                "95",
                "200",
                "opening",
                "https://lichess.org/test01",
                "King's Pawn",
            ]
        )
        # Puzzle 2: After 1. e4 e5, opponent plays Nf3, solution is Nc6
        writer.writerow(
            [
                "puz02",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "g1f3 b8c6",
                "1600",
                "70",
                "90",
                "150",
                "opening",
                "https://lichess.org/test02",
                "Defense",
            ]
        )
    return csv_file


def test_tactical_builder_synthetic_smoke(synthetic_puzzle_csv: Path, tmp_path: Path, zeropawn_cpp):
    """End-to-end smoke test of build_tactics_dataset using actual generator APIs."""
    out_dir = tmp_path / "dataset_tactics"

    n_train, n_val = build_tactics_dataset(
        csv_path=synthetic_puzzle_csv,
        output_dir=out_dir,
        max_solver_plies=2,
        chunk_size=10,
    )

    # 1. Total records generated
    total_generated = n_train + n_val
    assert total_generated >= 2, f"Expected at least 2 positions generated, got {total_generated}"

    # 2. Check binary files on disk for train split
    train_dir = out_dir / "train"
    assert (train_dir / "planes.bin").is_file(), "planes.bin missing"
    assert (train_dir / "moves.bin").is_file(), "moves.bin missing"
    assert (train_dir / "outcomes.bin").is_file(), "outcomes.bin missing"
    assert (train_dir / "game_ids.bin").is_file(), "game_ids.bin missing"
    assert (train_dir / "plies.bin").is_file(), "plies.bin missing"
    assert (train_dir / "metadata.json").is_file(), "metadata.json missing"

    # 3. Verify outcomes.bin stores OUTCOME_UNAVAILABLE (255) as uint8
    train_outcomes = np.fromfile(train_dir / "outcomes.bin", dtype=np.uint8)
    assert len(train_outcomes) == n_train
    assert np.all(train_outcomes == OUTCOME_UNAVAILABLE), (
        f"Expected all outcomes to be {OUTCOME_UNAVAILABLE}, got {train_outcomes}"
    )

    # 4. Open with BinaryChessDataset and verify semantics
    with BinaryChessDataset(train_dir) as ds:
        assert len(ds) == n_train
        assert ds.is_tactical is True, "Dataset should be identified as tactical"

        for idx in range(len(ds)):
            sample = ds[idx]
            # Policy head has legal move target
            assert 0 <= sample["move"].item() < 4672
            # Value target is masked to zero
            assert torch.allclose(sample["value_target"], torch.zeros(3)), (
                f"Tactical value target was not masked: {sample['value_target']}"
            )
            # Sentinel outcome remapped to -1
            assert sample["outcome"].item() == -1
            # Ply reset to 0 for tactical positions
            assert sample["ply"].item() == 0
            # Explicit boolean is_tactical flag
            assert sample["is_tactical"].item() is True


def test_master_builder_contrast_smoke(tmp_path: Path):
    """Verify that master game records preserve legitimate value supervision and are not altered."""
    master_dir = tmp_path / "dataset_master" / "train"
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    master_positions = [
        PositionData(planes=planes, move_idx=10, outcome=0, game_id=101, ply=15, pos_hash=1),
        PositionData(planes=planes, move_idx=20, outcome=1, game_id=102, ply=25, pos_hash=2),
        PositionData(planes=planes, move_idx=30, outcome=2, game_id=103, ply=35, pos_hash=3),
    ]

    save_dataset_split(
        master_dir, master_positions, "train", extra_metadata={"kind": "master_games"}
    )

    with BinaryChessDataset(master_dir) as ds:
        assert len(ds) == 3
        assert ds.is_tactical is False

        s0 = ds[0]
        assert s0["outcome"].item() == 0
        assert torch.allclose(s0["value_target"], torch.tensor([1.0, 0.0, 0.0]))
        assert s0["ply"].item() == 15
        assert s0["is_tactical"].item() is False

        s1 = ds[1]
        assert s1["outcome"].item() == 1
        assert torch.allclose(s1["value_target"], torch.tensor([0.0, 1.0, 0.0]))
        assert s1["ply"].item() == 25
        assert s1["is_tactical"].item() is False

        s2 = ds[2]
        assert s2["outcome"].item() == 2
        assert torch.allclose(s2["value_target"], torch.tensor([0.0, 0.0, 1.0]))
        assert s2["ply"].item() == 35
        assert s2["is_tactical"].item() is False
