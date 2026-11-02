"""Regression tests for the Syzygy 5-class remapping consumed by the RL pipeline.

Semantics (SY-02, commit c3cc6cc): cursed wins and blessed losses are draws
under FIDE rule 50, identical to the supervised path
(``python.data.dataset._outcome_to_value_target`` with is_syzygy=True).
The DI-06 convention (0/1->W, 3/4->L) is retired; audit S2-02 unified both
paths and ``tests/test_syzygy_mapping_unification_20261103.py`` pins the
SL/RL equality explicitly.
"""

import numpy as np
import pytest
import torch

from python.data.dataset_contract import remap_syzygy_outcomes


def test_remap_syzygy_outcomes_numpy():
    raw = np.array([0, 1, 2, 3, 4, 255], dtype=np.int64)
    expected = np.array([0, 1, 1, 1, 2, 255], dtype=np.int64)
    remapped = remap_syzygy_outcomes(raw)
    np.testing.assert_array_equal(remapped, expected)
    # The input must not be mutated in place (contract: returns a new array).
    np.testing.assert_array_equal(raw, np.array([0, 1, 2, 3, 4, 255], dtype=np.int64))

    with pytest.raises(ValueError, match="Invalid Syzygy outcome"):
        remap_syzygy_outcomes(np.array([5, 2], dtype=np.int64))


def test_remap_syzygy_outcomes_torch():
    raw = torch.tensor([0, 1, 2, 3, 4, 255], dtype=torch.long)
    expected = torch.tensor([0, 1, 1, 1, 2, 255], dtype=torch.long)
    remapped = remap_syzygy_outcomes(raw)
    assert torch.equal(remapped, expected)
    # The input must not be mutated in place.
    assert torch.equal(raw, torch.tensor([0, 1, 2, 3, 4, 255], dtype=torch.long))

    with pytest.raises(ValueError, match="Invalid Syzygy outcome"):
        remap_syzygy_outcomes(torch.tensor([-1, 0]))


def test_remap_syzygy_outcomes_scalar():
    assert remap_syzygy_outcomes(0) == 0  # Win -> Win
    assert remap_syzygy_outcomes(1) == 1  # Cursed Win -> Draw (SY-02)
    assert remap_syzygy_outcomes(2) == 1  # Draw -> Draw
    assert remap_syzygy_outcomes(3) == 1  # Blessed Loss -> Draw (SY-02)
    assert remap_syzygy_outcomes(4) == 2  # Loss -> Loss
    assert remap_syzygy_outcomes(255) == 255  # masked passes through
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
    assert torch.equal(remapped, torch.tensor([0, 1, 1, 1, 2], dtype=torch.long))


def test_remap_matches_supervised_onehot_targets():
    """The RL class-id remap and the SL one-hot mapping must agree for every
    5-class input, after identical one-hot encoding of the RL class ids
    (masked rows stay all-zero on both sides)."""
    from python.data.dataset import _outcome_to_value_target

    raw = np.array([0, 1, 2, 3, 4, 255], dtype=np.int64)
    sl_onehot = _outcome_to_value_target(raw, is_syzygy=True)

    ids = remap_syzygy_outcomes(raw)
    rl_onehot = np.zeros((raw.shape[0], 3), dtype=np.float32)
    valid = ids != 255
    rl_onehot[np.nonzero(valid)[0], ids[valid].astype(np.int64)] = 1.0

    np.testing.assert_allclose(rl_onehot, sl_onehot, atol=0.0)
    np.testing.assert_allclose(
        sl_onehot,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=0.0,
    )
