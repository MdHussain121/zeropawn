"""Sequential-probability-ratio testing (SPRT) and Elo confidence intervals.

Ported from the validated Stage 1 parameter-screen harness
(``scripts/run_sprt_screen.py`` on branch ``search-study``, 2026-10-16) and
extended with a shutout-safe Elo interval.

Conventions (verified against the harness and the vault study record):
- H0: candidate strength = ``elo0`` (default 0 Elo better than base)
- H1: candidate strength = ``elo1`` (default +30 Elo)
- Wald LLR bounds for ``alpha = beta = 0.05``: ``ln(beta/(1-alpha))`` and
  ``ln((1-beta)/alpha)`` — approximately ±2.944.
- Decisions: ``ACCEPT_H1`` (real gain proven), ``REJECT_H1`` (no gain
  proven), ``CONTINUE`` (keep playing), and an outer ``INCONCLUSIVE`` when
  the game budget is exhausted while still ``CONTINUE``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

# Wald bounds for alpha = beta = 0.05 (computed, not hardcoded, so a different
# error-rate choice stays consistent).
def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def elo_to_score(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def score_to_elo(score: float) -> Optional[float]:
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def compute_sprt_llr(
    wins: int,
    draws: int,
    losses: int,
    elo0: float = 0.0,
    elo1: float = 30.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict[str, Any]:
    """SPRT log-likelihood ratio for a W/D/L tally under score-based H0/H1.

    Fishtest/cutechess-style normalized LLR: ``(delta/var) * (sum of score
    deviations)`` where ``var`` is the empirical outcome variance. Returns the
    decision (``ACCEPT_H1``/``REJECT_H1``/``CONTINUE``), the Elo point
    estimate, and an Agresti-Caffo (add-2) Elo interval that stays finite for
    shutouts and other extreme scores (EV-06 fix).
    """
    n = wins + draws + losses
    lower_bound, upper_bound = sprt_bounds(alpha, beta)
    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "score": 0.0,
            "elo_diff": None,
            "elo_lo": None,
            "elo_hi": None,
            "llr": 0.0,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "decision": "CONTINUE",
        }

    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    score = (wins + 0.5 * draws) / n
    mu = (s0 + s1) / 2.0
    delta_s = s1 - s0

    var = (wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2) / n
    if var < 1e-4:
        var = 0.25

    llr = (delta_s / var) * (score * n - n * mu)
    decision = "CONTINUE"
    if llr <= lower_bound:
        decision = "REJECT_H1"
    elif llr >= upper_bound:
        decision = "ACCEPT_H1"

    elo_lo, elo_hi = elo_ci(wins, draws, losses)
    return {
        "n": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_diff": score_to_elo(score),
        "elo_lo": elo_lo,
        "elo_hi": elo_hi,
        "llr": llr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "decision": decision,
    }


def elo_ci(
    wins: int,
    draws: int,
    losses: int,
    z: float = 1.959963984540054,
) -> tuple[Optional[float], Optional[float]]:
    """95% Elo interval for a W/D/L tally — finite for shutouts (EV-06).

    The plain Wald interval on the score is degenerate at 0/1 (the old
    ``max(score - 1.96*se, 0.001)`` clip invented artificial certainty). We
    use the Agresti-Caffo add-2 interval: ``p_c = (score*n + 2) / (n + 4)``
    with variance ``p_c(1-p_c)/(n+4)`` on the logit scale, then map the
    bounds back through the logistic/Elo transform. Every output is a finite
    float whenever any game has been played.
    """
    n = wins + draws + losses
    if n == 0:
        return None, None
    score = (wins + 0.5 * draws) / n
    k = score * n
    p_c = (k + 2.0) / (n + 4.0)
    denom = 1.0 / (p_c * (1.0 - p_c))
    se = math.sqrt(denom / (n + 4.0))
    logit = math.log(p_c / (1.0 - p_c))
    lo_p = 1.0 / (1.0 + math.exp(-(logit - z * se)))
    hi_p = 1.0 / (1.0 + math.exp(-(logit + z * se)))
    return score_to_elo(lo_p), score_to_elo(hi_p)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a proportion, returned as (pct, lo_pct, hi_pct)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return p * 100.0, lo, hi


def sprt_result_to_json(result: dict[str, Any]) -> str:
    """Canonical serialization of an SPRT result (all floats finite or None)."""
    return json.dumps(result, sort_keys=True, allow_nan=False)
