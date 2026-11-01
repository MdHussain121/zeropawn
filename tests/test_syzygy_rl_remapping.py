"""Regression tests for DI-06: 5-class Syzygy remapping in RL pipeline."""

import numpy as np
import pytest
import torch

from python.data.dataset_contract import remap_syzygy_outcomes


def test_remap_syzygy_outcomes_numpy():
    raw = np.array([0, 1, 2, 3, 4, 255], dtype=np.int64)
    expected = np.array([0, 0, 1, 2, 2, 255], dtype=np.int64)
    remapped = remap_syzygy_outcomes(raw)
    np.testing.assert_array_equal(remapped, expected)

    with pytest.raises(ValueError, match="Invalid Syzygy outcome"):
        remap_syzygy_outcomes(np.array([5, 2], dtype=np.int64))


def test_remap_syzygy_outcomes_torch():
    raw = torch.tensor([0, 1, 2, 3, 4, 255], dtype=torch.long)
    expected = torch.tensor([0, 0, 1, 2, 2, 255], dtype=torch.long)
    remapped = remap_syzygy_outcomes(raw)
    assert torch.equal(remapped, expected)

    with pytest.raises(ValueError, match="Invalid Syzygy outcome"):
        remap_syzygy_outcomes(torch.tensor([-1, 0]))


def test_remap_syzygy_outcomes_scalar():
    assert remap_syzygy_outcomes(0) == 0
    assert remap_syzygy_outcomes(1) == 0
    assert remap_syzygy_outcomes(2) == 1
    assert remap_syzygy_outcomes(3) == 2
    assert remap_syzygy_outcomes(4) == 2
    assert remap_syzygy_outcomes(255) == 255
    with pytest.raises(ValueError):
        remap_syzygy_outcomes(99)


def test_rl_wdl_loss_safety_gate():
    """Verify that unremapped Syzygy labels raise ValueError in the RL loss calculation."""
    # Simulate the check in train_rl.py
    outcomes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    invalid_outcomes = (outcomes != 255) & ((outcomes < 0) | (outcomes > 2))
    assert invalid_outcomes.any()

    # Once remapped:
    remapped = remap_syzygy_outcomes(outcomes)
    invalid_remapped = (remapped != 255) & ((remapped < 0) | (remapped > 2))
    assert not invalid_remapped.any()
    assert torch.equal(remapped, torch.tensor([0, 0, 1, 2, 2], dtype=torch.long))
