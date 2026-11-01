"""L-1 regression tests: the raw batched-evaluator contract must fail loudly.

The 2026-10-31 model-architecture/inference audit (finding L-1) found that
``MCTS.search_batched`` consumed the raw evaluator's policy/value arrays via
``std::min`` memcpy with zero-fill: a short value array became 0.0 ("draw")
for the trailing rows instead of raising.  Element-count validation landed
with the C++/Python parity remediation (F-12, same defect class); the
remediation for this audit additionally enforces the exact rank/shape
(transposed / flat / extra-dimension arrays previously slipped through the
count check) and rejects non-finite values (NaN priors silently degraded to
the uniform fallback; Inf values propagated into ``root_value``).

The No-Truncation rule under test: malformed evaluator output raises a clear
exception; it is never truncated, zero-filled, or silently padded; and a
failed evaluation leaves the search instance usable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zeropawn_cpp  # noqa: E402

NUM_ACTIONS = 4672


def _make_mcts(seed: int = 1):
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = seed
    return zeropawn_cpp.MCTS(cfg)


def _uniform(n: int) -> np.ndarray:
    return np.full((n, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)


def _run(ev, sims: int = 6, batch_size: int = 4):
    mcts = _make_mcts()
    return mcts.search_batched(zeropawn_cpp.ChessGame(), sims, batch_size, ev)


# ---------------------------------------------------------------------------
# Valid contract
# ---------------------------------------------------------------------------


def test_correct_output_is_accepted_and_search_completes():
    seen = []

    def ev(planes):
        n = planes.shape[0]
        seen.append(n)
        return (_uniform(n), np.zeros(n, dtype=np.float32))

    res = _run(ev)
    assert seen, "evaluator must be called"
    assert all(n >= 1 for n in seen), f"evaluator called with empty batch: {seen}"
    assert isinstance(res.best_action, int)
    assert np.isfinite(res.root_value)


def test_valid_mixed_size_batches_are_supported():
    """Partial (final) batches legitimately have n < batch_size; each call must
    receive exactly (n, NUM_ACTIONS)/(n,)."""
    seen = []

    def ev(planes):
        n = planes.shape[0]
        seen.append(n)
        assert planes.shape == (n, 1216), planes.shape  # 19*8*8 state_size
        return (_uniform(n), np.zeros(n, dtype=np.float32))

    res = _run(ev, sims=9, batch_size=4)
    assert max(seen) <= 4
    assert len(set(seen)) >= 2, f"expected mixed batch sizes, saw {seen}"
    assert np.isfinite(res.root_value)


# ---------------------------------------------------------------------------
# Malformed output must raise (never truncate / zero-fill / pad)
# ---------------------------------------------------------------------------


def test_short_policy_array_raises():
    def ev(planes):
        n = planes.shape[0]
        return (_uniform(max(n - 1, 0)), np.zeros(n, dtype=np.float32))

    with pytest.raises(RuntimeError, match="policy shape"):
        _run(ev)


def test_long_policy_array_raises():
    def ev(planes):
        n = planes.shape[0]
        return (_uniform(n + 1), np.zeros(n, dtype=np.float32))

    with pytest.raises(RuntimeError, match="policy shape"):
        _run(ev)


def test_short_value_array_raises_instead_of_zero_fill():
    """The original L-1 defect: a short value array used to leave trailing
    rows at 0.0 ('draw') via std::min memcpy + zero-fill."""

    def ev(planes):
        n = planes.shape[0]
        return (_uniform(n), np.zeros(max(n - 1, 0), dtype=np.float32))

    with pytest.raises(RuntimeError, match="values shape"):
        _run(ev)


def test_long_value_array_raises():
    def ev(planes):
        n = planes.shape[0]
        return (_uniform(n), np.zeros(n + 1, dtype=np.float32))

    with pytest.raises(RuntimeError, match="values shape"):
        _run(ev)


def test_transposed_policy_shape_raises():
    """(NUM_ACTIONS, n) has the same element count as (n, NUM_ACTIONS) and
    previously slipped through the count-only check as row-major garbage."""

    def ev(planes):
        n = planes.shape[0]
        return (
            np.full((NUM_ACTIONS, n), 1.0 / NUM_ACTIONS, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    with pytest.raises(RuntimeError, match="policy shape"):
        _run(ev)


def test_flat_policy_array_raises():
    def ev(planes):
        n = planes.shape[0]
        return (
            np.full(n * NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    with pytest.raises(RuntimeError, match="policy shape"):
        _run(ev)


def test_extra_dimension_policy_raises():
    def ev(planes):
        n = planes.shape[0]
        return (
            np.full((n, NUM_ACTIONS, 1), 1.0 / NUM_ACTIONS, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    with pytest.raises(RuntimeError, match="policy shape"):
        _run(ev)


def test_two_dimensional_values_raise():
    def ev(planes):
        n = planes.shape[0]
        return (_uniform(n), np.zeros((n, 1), dtype=np.float32))

    with pytest.raises(RuntimeError, match="values shape"):
        _run(ev)


def test_empty_output_raises():
    def ev(planes):
        return (np.zeros((0, NUM_ACTIONS), np.float32), np.zeros(0, np.float32))

    with pytest.raises(RuntimeError, match="policy shape"):
        _run(ev)


def test_unsupported_dtype_raises():
    def ev(planes):
        n = planes.shape[0]
        return (
            np.full((n, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float64),
            np.zeros(n, dtype=np.float32),
        )

    with pytest.raises(TypeError):
        _run(ev)


def test_nan_policy_raises():
    """NaN priors used to pass silently: NaN comparisons are false, so the
    prior renormalization fell through to the uniform fallback."""

    def ev(planes):
        n = planes.shape[0]
        return (np.full((n, NUM_ACTIONS), np.nan, dtype=np.float32), np.zeros(n, np.float32))

    with pytest.raises(RuntimeError, match="non-finite"):
        _run(ev)


def test_inf_policy_raises():
    def ev(planes):
        n = planes.shape[0]
        policy = _uniform(n)
        policy[0, 0] = np.inf
        return (policy, np.zeros(n, np.float32))

    with pytest.raises(RuntimeError, match="non-finite"):
        _run(ev)


def test_nan_values_raise():
    def ev(planes):
        n = planes.shape[0]
        vals = np.zeros(n, np.float32)
        vals[-1] = np.nan
        return (_uniform(n), vals)

    with pytest.raises(RuntimeError, match="non-finite"):
        _run(ev)


def test_inf_values_raise():
    """Inf values used to propagate straight into SearchResult.root_value."""

    def ev(planes):
        n = planes.shape[0]
        return (_uniform(n), np.full(n, np.inf, dtype=np.float32))

    with pytest.raises(RuntimeError, match="non-finite"):
        _run(ev)


# ---------------------------------------------------------------------------
# Exception behavior and search-state integrity
# ---------------------------------------------------------------------------


def test_python_evaluator_exception_propagates():
    def ev(planes):
        raise ValueError("evaluator exploded")

    with pytest.raises(ValueError, match="evaluator exploded"):
        _run(ev)


def test_search_remains_usable_after_malformed_output_failure():
    """A failed evaluation must not corrupt the MCTS instance: a subsequent
    search with a correct evaluator must succeed."""
    mcts = _make_mcts(seed=3)

    def bad_ev(planes):
        n = planes.shape[0]
        return (_uniform(n), np.zeros(max(n - 1, 0), np.float32))

    with pytest.raises(RuntimeError):
        mcts.search_batched(zeropawn_cpp.ChessGame(), 4, 2, bad_ev)

    def good_ev(planes):
        n = planes.shape[0]
        return (_uniform(n), np.zeros(n, np.float32))

    res = mcts.search_batched(zeropawn_cpp.ChessGame(), 6, 4, good_ev)
    assert isinstance(res.best_action, int)
    assert np.isfinite(res.root_value)


def test_evaluator_tuple_arity_is_enforced():
    def ev_not_a_tuple(planes):
        return "nope"

    with pytest.raises(Exception):
        _run(ev_not_a_tuple)

    def ev_one_element(planes):
        n = planes.shape[0]
        return (_uniform(n),)

    with pytest.raises(Exception):
        _run(ev_one_element)
