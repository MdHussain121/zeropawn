"""Regression tests for Data Integrity P2 remediations (DI-15, DI-16, DI-17, DI-21, DI-22, DI-30)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pytest
import torch
import numpy as np

from python.eval.match_runner import eval_config_hash
from python.eval.baselines import _file_sha256
from python.data.dataset import BinaryChessDataset, SourceAwareCombinedDataset
from python.train.train_opening_policy import train_opening_step


def test_eval_config_hash_content_sensitive(tmp_path: Path):
    """Verify eval_config_hash incorporates file content hash, seed, and stockfish_elo (DI-15)."""
    openings_1 = tmp_path / "openings1.epd"
    openings_1.write_text(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n", encoding="utf-8"
    )

    openings_2 = tmp_path / "openings2.epd"
    openings_2.write_text(
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1\n", encoding="utf-8"
    )

    h1 = eval_config_hash(1000, 400, 2.0, openings_1, 8, stockfish_elo=2000, seed=42)
    h2 = eval_config_hash(1000, 400, 2.0, openings_2, 8, stockfish_elo=2000, seed=42)
    # Different content -> different hash
    assert h1 != h2

    # Different seed -> different hash
    h3 = eval_config_hash(1000, 400, 2.0, openings_1, 8, stockfish_elo=2000, seed=43)
    assert h1 != h3

    # Different elo -> different hash
    h4 = eval_config_hash(1000, 400, 2.0, openings_1, 8, stockfish_elo=2500, seed=42)
    assert h1 != h4


def test_baseline_file_sha256_fresh_computation(tmp_path: Path):
    """Verify _file_sha256 does not return stale digests across modifications (DI-16)."""
    target = tmp_path / "model.pt"
    target.write_bytes(b"content_version_one")
    digest1 = _file_sha256(target)

    # Overwrite file with same length content
    target.write_bytes(b"content_version_two")
    digest2 = _file_sha256(target)

    assert digest1 != digest2
    assert digest2 == hashlib.sha256(b"content_version_two").hexdigest()


def test_opening_policy_step_masks_early_plies():
    """Verify train_opening_step applies zero WDL loss when plies < 10 (DI-21)."""
    from python.model.chess_net import ChessNet

    device = torch.device("cpu")
    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 4
    # All plies < 10
    batch_early = {
        "planes": torch.randn(B, 19, 8, 8),
        "move": torch.randint(0, 4672, (B,)),
        "outcome": torch.randint(0, 3, (B,)),
        "value_target": torch.tensor([[0.3, 0.4, 0.3]] * B, dtype=torch.float32),
        "ply": torch.tensor([1, 2, 3, 4], dtype=torch.long),
    }

    tot_l, pol_l, wdl_l, _, _, _ = train_opening_step(
        model=model,
        batch=batch_early,
        optimizer=optimizer,
        device=device,
        use_autocast=False,
    )
    # WDL loss should be 0.0 because all plies are < 10
    assert wdl_l == 0.0

    # Plies >= 10
    batch_late = {
        "planes": torch.randn(B, 19, 8, 8),
        "move": torch.randint(0, 4672, (B,)),
        "outcome": torch.randint(0, 3, (B,)),
        "value_target": torch.tensor([[1.0, 0.0, 0.0]] * B, dtype=torch.float32),
        "ply": torch.tensor([10, 12, 14, 16], dtype=torch.long),
    }

    tot_l2, pol_l2, wdl_l2, _, _, _ = train_opening_step(
        model=model,
        batch=batch_late,
        optimizer=optimizer,
        device=device,
        use_autocast=False,
    )
    # WDL loss should be non-zero for plies >= 10
    assert wdl_l2 > 0.0


def test_value_targets_consistency_check(tmp_path: Path):
    """Verify BinaryChessDataset checks value_targets.bin against outcomes.bin (DI-22)."""
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    N = 10

    # Create planes, moves, outcomes, game_ids, plies
    (split_dir / "planes.bin").write_bytes(np.zeros((N, 19, 8, 8), dtype=np.float32).tobytes())
    (split_dir / "moves.bin").write_bytes(np.zeros(N, dtype=np.int16).tobytes())
    (split_dir / "outcomes.bin").write_bytes(
        np.zeros(N, dtype=np.uint8).tobytes()
    )  # outcome = 0 (Win)
    (split_dir / "game_ids.bin").write_bytes(np.zeros(N, dtype=np.uint64).tobytes())
    (split_dir / "plies.bin").write_bytes(np.zeros(N, dtype=np.int16).tobytes())

    # Create metadata
    meta = {
        "num_samples": N,
        "planes_shape": [19, 8, 8],
        "split": "train",
        "kind": "games_wdl",
    }
    (split_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Inconsistent value_targets: all zeros while outcome is 0 (unmasked win)
    (split_dir / "value_targets.bin").write_bytes(np.zeros((N, 3), dtype=np.float32).tobytes())

    with pytest.raises(ValueError, match="value_targets.bin row sum is 0"):
        BinaryChessDataset(split_dir)
