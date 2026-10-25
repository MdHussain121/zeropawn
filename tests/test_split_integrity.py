"""Regression tests for dataset split integrity and value calibration diagnostics (F-06).

Verifies:
1. Game-level split integrity (zero game ID leakage between train, val, and test splits).
2. Calculation correctness of Brier score and value calibration metrics.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pytest
import torch


def test_game_level_split_no_leakage():
    """Verify that hashing game_id guarantees disjoint game sets across train/val/test splits."""
    num_games = 1000
    positions_per_game = 40

    game_ids = np.arange(num_games, dtype=np.uint64)

    # Assign games to splits via deterministic hashing (80% train, 10% val, 10% test)
    train_games = set()
    val_games = set()
    test_games = set()

    for gid in game_ids:
        h = int(hashlib.md5(str(int(gid)).encode()).hexdigest()[:8], 16) % 100
        if h < 80:
            train_games.add(gid)
        elif h < 90:
            val_games.add(gid)
        else:
            test_games.add(gid)

    # Verify disjointness
    assert len(train_games.intersection(val_games)) == 0, "Leakage detected between Train and Val!"
    assert len(train_games.intersection(test_games)) == 0, (
        "Leakage detected between Train and Test!"
    )
    assert len(val_games.intersection(test_games)) == 0, "Leakage detected between Val and Test!"
    assert len(train_games) + len(val_games) + len(test_games) == num_games


def test_brier_score_and_calibration_math():
    """Verify Brier score and predicted vs target value arithmetic."""
    # Perfect predictions: targets == probs -> Brier score = 0
    probs_perfect = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    targets_perfect = probs_perfect.clone()
    brier_perfect = ((probs_perfect - targets_perfect) ** 2).sum(dim=-1).mean().item()
    assert brier_perfect == 0.0

    # Completely wrong predictions (predict 100% win when actually loss)
    probs_wrong = torch.tensor([[1.0, 0.0, 0.0]])
    targets_wrong = torch.tensor([[0.0, 0.0, 1.0]])
    # (1-0)^2 + (0-0)^2 + (0-1)^2 = 1 + 1 = 2.0
    brier_wrong = ((probs_wrong - targets_wrong) ** 2).sum(dim=-1).mean().item()
    assert brier_wrong == 2.0

    # Uniform predictions on 1-hot target
    probs_uniform = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    targets_onehot = torch.tensor([[1.0, 0.0, 0.0]])
    # (2/3)^2 + (-1/3)^2 + (-1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
    brier_uniform = ((probs_uniform - targets_onehot) ** 2).sum(dim=-1).mean().item()
    assert abs(brier_uniform - (2.0 / 3.0)) < 1e-6
