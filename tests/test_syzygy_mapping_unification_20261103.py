"""Shared-contract test (audit S2-02 remediation, 2026-11-03).

Pins that the supervised value-target path
(``python.data.dataset._outcome_to_value_target`` with ``is_syzygy=True``)
and the RL-side class-id remap
(``python.data.dataset_contract.remap_syzygy_outcomes``) implement the
IDENTICAL SY-02 5-class -> 3-class convention:

    0 (Win)          -> 0 / [1, 0, 0]
    1 (Cursed Win)   -> 1 / [0, 1, 0]   (draw under FIDE rule 50)
    2 (Draw)         -> 1 / [0, 1, 0]
    3 (Blessed Loss) -> 1 / [0, 1, 0]   (draw under FIDE rule 50)
    4 (Loss)         -> 2 / [0, 0, 1]
    255 (masked)     -> 255 / [0, 0, 0] (policy-only row)

The DI-06 mapping (0/1 -> W, 3/4 -> L) is retired; a divergence between the
two paths must fail this test.
"""

import numpy as np
import torch

from python.data.dataset import _outcome_to_value_target
from python.data.dataset_contract import (
    OUTCOME_DRAW,
    OUTCOME_LOSS,
    OUTCOME_UNAVAILABLE,
    OUTCOME_WIN,
    remap_syzygy_outcomes,
)

# Fixed 5-class input vector covering every legal domain value.
FIVE_CLASS_VECTOR = [0, 1, 2, 3, 4, 255]


def _onehot_from_class_ids(ids: np.ndarray) -> np.ndarray:
    """Encode class ids into (N, 3) one-hot rows exactly the way the loaders
    encode supervised targets; masked rows (255) stay all-zero."""
    onehot = np.zeros((ids.shape[0], 3), dtype=np.float32)
    valid = ids != OUTCOME_UNAVAILABLE
    onehot[np.nonzero(valid)[0], ids[valid].astype(np.int64)] = 1.0
    return onehot


def test_supervised_and_rl_paths_produce_identical_targets():
    """The canonical S2-02 equality: SL one-hot targets == RL remap, after
    identical one-hot encoding of the RL class ids."""
    raw = np.array(FIVE_CLASS_VECTOR, dtype=np.int64)

    sl_targets = _outcome_to_value_target(raw, is_syzygy=True)
    rl_ids = np.asarray(remap_syzygy_outcomes(raw))
    rl_targets = _onehot_from_class_ids(rl_ids)

    np.testing.assert_array_equal(rl_targets, sl_targets)
    np.testing.assert_array_equal(
        sl_targets,
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
    )


def test_per_value_scalar_agreement_with_supervised_path():
    """Every scalar class maps to the same 3-class target on both paths."""
    raw = np.array(FIVE_CLASS_VECTOR, dtype=np.int64)
    sl_rows = _outcome_to_value_target(raw, is_syzygy=True)
    for idx, value in enumerate(FIVE_CLASS_VECTOR):
        rl_id = remap_syzygy_outcomes(int(value))
        expected_row = sl_rows[idx]
        if value == OUTCOME_UNAVAILABLE:
            assert rl_id == OUTCOME_UNAVAILABLE
            np.testing.assert_array_equal(expected_row, np.zeros(3, dtype=np.float32))
        else:
            onehot = np.zeros(3, dtype=np.float32)
            onehot[int(rl_id)] = 1.0
            np.testing.assert_array_equal(onehot, expected_row)


def test_remap_torch_and_numpy_paths_agree():
    """The numpy and torch branches of remap_syzygy_outcomes agree with each
    other and with the supervised one-hot table."""
    raw_np = np.array(FIVE_CLASS_VECTOR, dtype=np.int64)
    raw_t = torch.tensor(FIVE_CLASS_VECTOR, dtype=torch.long)

    ids_np = np.asarray(remap_syzygy_outcomes(raw_np))
    ids_t = remap_syzygy_outcomes(raw_t)
    np.testing.assert_array_equal(ids_np, ids_t.numpy())

    np.testing.assert_array_equal(
        _onehot_from_class_ids(ids_np),
        _outcome_to_value_target(raw_np, is_syzygy=True),
    )


def test_remapped_values_stay_inside_trainer_domain_gate():
    """The RL trainer's DI-06 domain gate requires post-remap values in
    {0, 1, 2, 255}; SY-02 must keep satisfying it."""
    ids = np.asarray(remap_syzygy_outcomes(np.array(FIVE_CLASS_VECTOR, dtype=np.int64)))
    assert set(np.unique(ids).tolist()) <= {OUTCOME_WIN, OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_UNAVAILABLE}


def test_retired_di06_mapping_is_not_resurrected():
    """The retired DI-06 table (0/1 -> W, 3/4 -> L) must not come back:
    class 1 is a Draw and class 3 is a Draw under SY-02."""
    ids = np.asarray(remap_syzygy_outcomes(np.array([1, 3], dtype=np.int64)))
    np.testing.assert_array_equal(ids, np.array([OUTCOME_DRAW, OUTCOME_DRAW], dtype=np.int64))
