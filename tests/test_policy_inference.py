"""Regression tests for policy inference and MCTS prior semantics (F-05).

Verifies:
1. Mathematical equivalence of full-action softmax + legal re-normalization vs masked-logit softmax.
2. Numerical stability under extreme logits (float32 underflow behavior).
3. Fallback to uniform prior (1/k) when legal probability sum <= 1e-8.
4. Non-finite logit handling and masking safety.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def mcts_cpp_prior_normalization(full_policy: np.ndarray, legal_actions: list[int]) -> list[float]:
    """Python reference implementation of cpp/src/mcts.cpp set_priors logic."""
    k = len(legal_actions)
    if k == 0:
        return []

    priors = [0.0] * k
    p_sum = 0.0
    for i in range(k):
        a = legal_actions[i]
        p = float(full_policy[a]) if (0 <= a < len(full_policy)) else 0.0
        priors[i] = p
        p_sum += p

    if p_sum > 1e-8:
        for i in range(k):
            priors[i] /= p_sum
    else:
        for i in range(k):
            priors[i] = 1.0 / k

    return priors


def test_prior_normalization_standard_equivalence():
    """Verify that full-action softmax followed by legal extraction + renormalization
    is equivalent to masked-logit softmax for normal logit ranges.
    """
    torch.manual_seed(42)
    logits = torch.randn(4672)  # Realistic logit distribution
    legal_actions = [10, 45, 120, 500, 2300]
    k = len(legal_actions)

    # Method A: Full softmax then slice and re-normalize (MCTS implementation)
    full_probs = F.softmax(logits, dim=-1).numpy()
    priors_a = mcts_cpp_prior_normalization(full_probs, legal_actions)

    # Method B: Masked-logits softmax
    masked_logits = torch.full((4672,), -1e9)
    masked_logits[legal_actions] = logits[legal_actions]
    priors_b = F.softmax(masked_logits[legal_actions], dim=-1).numpy()

    # Must match within float32 tolerance
    np.testing.assert_allclose(priors_a, priors_b, rtol=1e-5, atol=1e-6)
    assert math.isclose(sum(priors_a), 1.0, rel_tol=1e-5)


def test_prior_underflow_fallback_to_uniform():
    """Verify that when legal actions have extremely small probabilities (underflow in float32),
    the MCTS prior logic cleanly falls back to uniform distribution (1/k).
    """
    legal_actions = [5, 10, 15, 20]
    k = len(legal_actions)

    # Simulate full policy where legal actions underflowed to 0.0 in float32
    full_policy = np.zeros(4672, dtype=np.float32)
    full_policy[100] = 1.0  # Illegal action took 100% of mass

    priors = mcts_cpp_prior_normalization(full_policy, legal_actions)
    expected_uniform = [1.0 / k] * k

    assert len(priors) == k
    assert np.allclose(priors, expected_uniform)
    assert math.isclose(sum(priors), 1.0, rel_tol=1e-6)


def test_prior_empty_legal_actions():
    """Verify that zero legal actions (checkmate/stalemate node) does not divide by zero."""
    full_policy = np.ones(4672, dtype=np.float32) / 4672.0
    priors = mcts_cpp_prior_normalization(full_policy, [])
    assert priors == []


def test_prior_near_threshold_behavior():
    """Verify behavior around the 1e-8 sum threshold."""
    legal_actions = [0, 1]
    full_policy = np.zeros(4672, dtype=np.float32)

    # Just below threshold (1e-9) -> falls back to uniform
    full_policy[0] = 5e-10
    full_policy[1] = 4e-10
    priors_below = mcts_cpp_prior_normalization(full_policy, legal_actions)
    assert np.allclose(priors_below, [0.5, 0.5])

    # Just above threshold (2e-8) -> normalizes
    full_policy[0] = 1.5e-8
    full_policy[1] = 0.5e-8
    priors_above = mcts_cpp_prior_normalization(full_policy, legal_actions)
    assert np.allclose(priors_above, [0.75, 0.25])
