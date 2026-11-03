"""Sequential-probability-ratio testing (SPRT) and Elo confidence intervals.

Ported from the validated Stage 1 parameter-screen harness
(``scripts/run_sprt_screen.py`` on branch ``search-study``, 2026-10-16) and
backed by the authoritative ``python.eval.stats`` module (Audit F-03, F-13).

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

from python.eval.stats import (
    agresti_caffo_elo_ci,
    # Facade re-exports (explicit ``as`` idiom — audit S7-10): scripts/run_sprt.py
    # and tests/test_sprt.py import these through this module.
    compute_sprt_llr as compute_sprt_llr,
    elo_to_score as elo_to_score,
    score_to_elo as score_to_elo,
    wilson_score_interval,
)


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    """Wald LLR stopping bounds for given alpha and beta error rates."""
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def elo_ci(
    wins: int,
    draws: int,
    losses: int,
    z: float = 1.959963984540054,
) -> tuple[Optional[float], Optional[float]]:
    """95% Elo interval for a W/D/L tally — finite for shutouts (EV-06, F-03).

    Delegates to canonical Agresti-Caffo implementation in python.eval.stats.
    """
    return agresti_caffo_elo_ci(wins, draws, losses)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a proportion, returned as (pct, lo_pct, hi_pct)."""
    return wilson_score_interval(k, n, confidence=confidence)


def sprt_result_to_json(result: dict[str, Any]) -> str:
    """Canonical serialization of an SPRT result (all floats finite or None)."""
    return json.dumps(result, sort_keys=True, allow_nan=False)
