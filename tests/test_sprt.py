"""Tests for the SPRT math module (EV-03) and the Elo confidence interval (EV-06).

Boundaries and H0/H1 semantics follow the validated 2026-10-16 screen harness:
H0 = +0 Elo, H1 = +30 Elo, alpha = beta = 0.05, Wald bounds ±2.944 (ln(19)).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python.eval.sprt import (
    compute_sprt_llr,
    elo_ci,
    elo_to_score,
    score_to_elo,
    sprt_bounds,
    sprt_result_to_json,
    wilson_interval,
)


def test_sprt_bounds_match_wald_formula():
    lo, hi = sprt_bounds(0.05, 0.05)
    assert abs(lo - math.log(0.05 / 0.95)) < 1e-12
    assert abs(hi - math.log(0.95 / 0.05)) < 1e-12
    assert abs(lo + 2.9444389791664403) < 1e-9
    assert abs(hi - 2.9444389791664403) < 1e-9


def test_score_elo_round_trip():
    for elo in (0.0, 30.0, 100.0, -147.2):
        assert abs(score_to_elo(elo_to_score(elo)) - elo) < 1e-6


def test_sprt_neutral_score_continues():
    # 50% score over many games is exactly H0; LLR must hover near 0.
    r = compute_sprt_llr(50, 0, 50)
    assert r["decision"] == "CONTINUE"
    assert abs(r["llr"]) < 1.0
    assert r["n"] == 100


def test_sprt_accepts_h1_on_strong_gain():
    # A sustained score clearly above H1 (60% >> the 54.3% H1 score) must
    # accept H1. (Tallies sampled exactly AT the H1 score are marginal by
    # construction and legitimately continue.)
    n = 800
    wins = int(n * 0.60)
    losses = n - wins
    r = compute_sprt_llr(wins, 0, losses)
    assert r["decision"] == "ACCEPT_H1"
    assert r["llr"] >= r["upper_bound"]


def test_sprt_rejects_h1_on_no_gain():
    # Sustained 45% score (worse than H1's 54.3%) must eventually reject.
    n = 800
    wins = int(n * 0.45)
    losses = n - wins
    r = compute_sprt_llr(wins, 0, losses)
    assert r["decision"] == "REJECT_H1"
    assert r["llr"] <= r["lower_bound"]


def test_sprt_monotone_in_score():
    strong = compute_sprt_llr(600, 100, 300)
    weak = compute_sprt_llr(400, 100, 500)
    assert strong["llr"] > weak["llr"]


def test_sprt_llr_closed_form_fixed_sigma():
    """S5-06: sigma^2 is the FIXED conservative 0.25, so the LLR is exactly
    4 * (s1 - s0) * sum(x_i - mu_bar) for any W/D/L mix (tolerance 1e-9).

    The historical "empirical variance" branch was mathematically dead (the
    Bhatia-Davis bound caps the outcome variance at s(1-s) <= 0.25, so the
    max(var, 0.25) floor was always active)."""
    for elo0, elo1 in [(0.0, 30.0), (-10.0, 20.0), (0.0, 5.0)]:
        s0 = elo_to_score(elo0)
        s1 = elo_to_score(elo1)
        mu_bar = (s0 + s1) / 2.0
        for w, d, l in [
            (10, 0, 0),
            (0, 0, 10),
            (3, 4, 3),
            (0, 100, 0),
            (70, 20, 10),
            (500, 100, 200),
            (1, 1, 1),
            (13, 0, 87),
        ]:
            r = compute_sprt_llr(w, d, l, elo0=elo0, elo1=elo1)
            sum_centered = w * (1.0 - mu_bar) + d * (0.5 - mu_bar) + l * (0.0 - mu_bar)
            expected = 4.0 * (s1 - s0) * sum_centered
            assert abs(r["llr"] - expected) <= 1e-9, (elo0, elo1, w, d, l, r["llr"], expected)


def test_sprt_llr_fixed_variance_ignores_outcome_spread():
    """S5-06: with fixed sigma^2 = 0.25 the LLR depends only on n and the mean
    score (sum(x_i - mu_bar) = n * (score - mu_bar)), NOT on the W/D/L spread —
    a (25, 50, 25) split and a (50, 0, 50) split (same score, same n) produce
    identical LLRs."""
    a = compute_sprt_llr(25, 50, 25, elo0=0.0, elo1=30.0)
    b = compute_sprt_llr(50, 0, 50, elo0=0.0, elo1=30.0)
    assert a["score"] == b["score"] == 0.5
    assert abs(a["llr"] - b["llr"]) <= 1e-9


def test_sprt_zero_games_continues():
    r = compute_sprt_llr(0, 0, 0)
    assert r["decision"] == "CONTINUE"
    assert r["llr"] == 0.0
    assert r["elo_diff"] is None


def test_sprt_result_serializes_cleanly():
    for tally in [(10, 0, 0), (0, 0, 10), (3, 4, 3), (0, 0, 0), (500, 100, 200)]:
        r = compute_sprt_llr(*tally)
        text = sprt_result_to_json(r)
        assert "NaN" not in text and "Infinity" not in text


def test_elo_ci_finite_at_shutouts():
    # 10-0-0: the old Wald clip produced artificial certainty; the add-2
    # interval must stay finite and honest (wide).
    lo, hi = elo_ci(10, 0, 0)
    assert lo is not None and hi is not None
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < hi
    # A shutout over 10 games cannot certify more than a few hundred Elo.
    assert hi < 1200.0
    assert lo > 0.0

    lo0, hi0 = elo_ci(0, 0, 10)
    assert math.isfinite(lo0) and math.isfinite(hi0)
    assert hi0 < 0.0  # lost every game: interval entirely negative


def test_elo_ci_centered_on_score():
    lo, hi = elo_ci(50, 0, 50)
    assert abs(lo + hi) < 1e-6  # symmetric around 0 Elo
    assert lo < 0.0 < hi

    lo, hi = elo_ci(30, 40, 30)
    assert lo < 0.0 < hi  # 50% score with draws


def test_elo_ci_narrows_with_games():
    lo10, hi10 = elo_ci(7, 2, 1)
    lo200, hi200 = elo_ci(140, 40, 20)
    assert (hi200 - lo200) < (hi10 - lo10)


def test_elo_ci_all_draws_is_zero_with_finite_width():
    lo, hi = elo_ci(0, 100, 0)
    assert abs(lo + hi) < 1e-6
    assert math.isfinite(lo) and math.isfinite(hi)


def test_wilson_interval_basics():
    p, lo, hi = wilson_interval(640, 1000)
    assert abs(p - 64.0) < 1e-9
    assert 60.5 < lo < 64.0 < hi < 67.0
    assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)
    p, lo, hi = wilson_interval(10, 10)
    assert lo > 0.0 and hi <= 100.0  # never degenerate
