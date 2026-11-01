"""ZeroPawn Authoritative Evaluation Statistics Module.

Provides canonical, statistically rigorous implementations of:
- Logistic score <-> Elo transforms
- Agresti-Caffo, Wald, and Delta-method Elo confidence intervals
- Paired-observation (cluster-robust) variance and confidence intervals
- Proportional Wilson score intervals
- Normalized GSPRT / Fishtest-family Sequential Probability Ratio Test (SPRT)
- Backward-compatible adapters for existing evaluators

Addresses Audit Findings:
- F-03: Unification of divergent Elo/CI implementations into one authoritative module
- F-13: SPRT LLR calibration and reference properties
- F-16: Paired-opening correlation and cluster-adjusted variance modeling
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union


def score_to_elo(score: float) -> Optional[float]:
    """Convert winning score s in (0, 1) to Elo difference: -400 * log10(1/s - 1).

    Returns None for non-positive or >= 1.0 scores (infinite / unbounded).
    """
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    """Convert Elo difference to expected score: 1 / (1 + 10^(-elo / 400))."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def z_for_confidence(confidence: float = 0.95) -> float:
    """Return standard normal critical value z for two-sided confidence level."""
    if math.isclose(confidence, 0.95, rel_tol=1e-3):
        return 1.959963984540054
    if math.isclose(confidence, 0.99, rel_tol=1e-3):
        return 2.5758293035489004
    if math.isclose(confidence, 0.90, rel_tol=1e-3):
        return 1.6448536269514722
    # Inverse normal CDF approximation (Abramowitz & Stegun formula)
    alpha = 1.0 - confidence
    p = 1.0 - alpha / 2.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)


def agresti_caffo_elo_ci(
    wins: int,
    draws: int,
    losses: int,
    confidence: float = 0.95,
) -> tuple[Optional[float], Optional[float]]:
    """Agresti-Caffo add-2 confidence interval on the logit scale, mapped to Elo.

    Computes p_c = (score * n + 2) / (n + 4) with variance on logit scale:
      se_logit = sqrt(1 / ((n + 4) * p_c * (1 - p_c)))
      logit_bounds = logit(p_c) +/- z * se_logit
    and maps the probability bounds back through score_to_elo().

    Properties:
    - Finite and centered for shutouts (all-wins or all-losses).
    - Eliminates artificial finite bounds created by arbitrary score-space clamping (F-03).
    - Conservative coverage near boundaries while agreeing with Wald for large, balanced samples.
    """
    n = wins + draws + losses
    if n <= 0:
        return (None, None)
    z = z_for_confidence(confidence)
    score = (wins + 0.5 * draws) / n
    k = score * n
    p_c = (k + 2.0) / (n + 4.0)
    denom = 1.0 / (p_c * (1.0 - p_c))
    se_logit = math.sqrt(denom / (n + 4.0))
    logit = math.log(p_c / (1.0 - p_c))
    lo_p = 1.0 / (1.0 + math.exp(-(logit - z * se_logit)))
    hi_p = 1.0 / (1.0 + math.exp(-(logit + z * se_logit)))
    return (score_to_elo(lo_p), score_to_elo(hi_p))


def delta_method_elo_ci(
    wins: int,
    draws: int,
    losses: int,
    confidence: float = 0.95,
) -> tuple[Optional[float], Optional[float]]:
    """Wald delta-method interval directly on the Elo scale.

    d(Elo)/ds = 400 / (ln(10) * s * (1 - s))
    SE_elo = (400 / ln(10)) * sqrt(1 / (n * s * (1 - s)))
    Returns (None, None) for score <= 0 or >= 1 (unbounded).
    """
    n = wins + draws + losses
    if n <= 0:
        return (None, None)
    s = (wins + 0.5 * draws) / n
    if s <= 0.0 or s >= 1.0:
        return (None, None)
    z = z_for_confidence(confidence)
    elo = score_to_elo(s)
    if elo is None:
        return (None, None)
    se_elo = (400.0 / math.log(10.0)) * math.sqrt(1.0 / (n * s * (1.0 - s)))
    return (elo - z * se_elo, elo + z * se_elo)


def elo_confidence_interval(
    wins: int,
    draws: int,
    losses: int,
    confidence: float = 0.95,
    method: str = "agresti_caffo",
) -> dict[str, Any]:
    """Authoritative Elo confidence interval calculation for evaluation reporting.

    Supported methods:
    - 'agresti_caffo' (default, canonical): Robust add-2 logit interval, honest at shutouts.
    - 'delta_method': Wald on Elo scale, returns None bounds at 0% or 100%.

    Returns dictionary containing point estimate, bounds, standard error, and metadata.
    """
    n = wins + draws + losses
    if n <= 0:
        return {
            "n": 0,
            "score": 0.0,
            "elo_diff": None,
            "lo": None,
            "hi": None,
            "confidence": confidence,
            "method": method,
            "is_shutout": False,
        }

    score = (wins + 0.5 * draws) / n
    elo_diff = score_to_elo(score)
    is_shutout = (score <= 0.0 or score >= 1.0)

    if method == "agresti_caffo":
        lo, hi = agresti_caffo_elo_ci(wins, draws, losses, confidence=confidence)
    elif method == "delta_method":
        lo, hi = delta_method_elo_ci(wins, draws, losses, confidence=confidence)
    else:
        raise ValueError(f"Unknown confidence interval method: {method}")

    return {
        "n": n,
        "score": score,
        "elo_diff": elo_diff,
        "lo": lo,
        "hi": hi,
        "confidence": confidence,
        "method": method,
        "is_shutout": is_shutout,
    }


def paired_score_and_ci(
    pairs: list[tuple[float, float]],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster-robust / paired variance calculation for color-reversed opening pairs.

    In a paired evaluation with M opening positions, opening i is played twice:
      - once with Candidate as White (score s_W in {0, 0.5, 1})
      - once with Candidate as Black (score s_B in {0, 0.5, 1})

    The paired experimental unit score is y_i = s_W + s_B in [0, 2].
    The overall candidate score is s_bar = y_bar / 2.
    The sample variance across pairs is:
      s_pair^2 = (1 / (M - 1)) * sum((y_i - y_bar)^2)
    and the cluster-adjusted standard error of the mean score is:
      SE_paired = s_pair / (2 * sqrt(M))

    This correctly models intra-opening correlation (F-16) rather than
    assuming all 2*M games are i.i.d. observations.

    Returns:
    - pair_count: number of complete pairs M
    - score: overall candidate score s_bar in [0, 1]
    - elo_diff: point Elo estimate
    - lo, hi: paired cluster-adjusted Elo confidence bounds
    - intra_pair_correlation: Pearson r between White and Black scores across openings
    """
    m = len(pairs)
    if m <= 0:
        return {
            "pair_count": 0,
            "total_games": 0,
            "score": 0.0,
            "elo_diff": None,
            "lo": None,
            "hi": None,
            "se_score": 0.0,
            "intra_pair_correlation": 0.0,
        }

    w_scores = [p[0] for p in pairs]
    b_scores = [p[1] for p in pairs]
    pair_totals = [w + b for w, b in pairs]

    y_bar = sum(pair_totals) / m
    s_bar = y_bar / 2.0
    total_games = 2 * m

    if m > 1:
        s_pair_sq = sum((y - y_bar) ** 2 for y in pair_totals) / (m - 1)
        se_score = math.sqrt(max(s_pair_sq, 0.0)) / (2.0 * math.sqrt(m))
    else:
        se_score = 0.0

    # Intra-pair Pearson correlation between White score and Black score
    mean_w = sum(w_scores) / m
    mean_b = sum(b_scores) / m
    var_w = sum((w - mean_w) ** 2 for w in w_scores)
    var_b = sum((b - mean_b) ** 2 for b in b_scores)
    cov_wb = sum((w - mean_w) * (b - mean_b) for w, b in zip(w_scores, b_scores))
    if var_w > 0.0 and var_b > 0.0:
        corr = cov_wb / math.sqrt(var_w * var_b)
    else:
        corr = 0.0

    z = z_for_confidence(confidence)
    # Delta-method transform for paired SE
    elo_diff = score_to_elo(s_bar)
    if elo_diff is not None and 0.0 < s_bar < 1.0 and se_score > 0.0:
        d_elo_ds = 400.0 / (math.log(10.0) * s_bar * (1.0 - s_bar))
        se_elo = d_elo_ds * se_score
        lo_elo = elo_diff - z * se_elo
        hi_elo = elo_diff + z * se_elo
    else:
        # Fall back to Agresti-Caffo if extreme or single pair
        w_int = int(round(sum(w_scores) + sum(b_scores)))
        # Map pair totals to pseudo W/D/L for fallback
        lo_elo, hi_elo = agresti_caffo_elo_ci(
            int(round(sum(1 for s in w_scores + b_scores if s > 0.75))),
            int(round(sum(1 for s in w_scores + b_scores if 0.25 <= s <= 0.75))),
            int(round(sum(1 for s in w_scores + b_scores if s < 0.25))),
            confidence=confidence,
        )

    return {
        "pair_count": m,
        "total_games": total_games,
        "score": s_bar,
        "elo_diff": elo_diff,
        "lo": lo_elo,
        "hi": hi_elo,
        "se_score": se_score,
        "intra_pair_correlation": corr,
    }


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Wilson score interval for proportions, returning (pct, lo_pct, hi_pct)."""
    if total <= 0:
        return (0.0, 0.0, 0.0)
    z = z_for_confidence(confidence)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return (p * 100.0, lo, hi)


def compute_sprt_llr(
    wins: int,
    draws: int,
    losses: int,
    elo0: float = 0.0,
    elo1: float = 30.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict[str, Any]:
    """Normalized GSPRT log-likelihood ratio matching Fishtest and ZeroPawn conventions.

    LLR = (Delta_s / sigma^2) * sum(x_i - mu_bar)
    where:
      s0 = elo_to_score(elo0), s1 = elo_to_score(elo1)
      Delta_s = s1 - s0
      mu_bar = (s0 + s1) / 2
      sigma^2 = outcome variance (floored at 0.25)
      Wald bounds: lower = ln(beta / (1 - alpha)), upper = ln((1 - beta) / alpha)

    Decisions:
      LLR >= upper -> 'ACCEPT_H1' (candidate confirmed stronger)
      LLR <= lower -> 'REJECT_H1' (candidate not stronger / rejected)
      otherwise    -> 'CONTINUE'
    """
    n = wins + draws + losses
    lower_bound = math.log(beta / (1.0 - alpha))
    upper_bound = math.log((1.0 - beta) / alpha)

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
    delta_s = s1 - s0
    mu_bar = (s0 + s1) / 2.0

    score = (wins + 0.5 * draws) / n
    # Empirical variance across games x in {1.0, 0.5, 0.0}
    var_empirical = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / n
    sigma_sq = max(var_empirical, 0.25)

    sum_centered = wins * (1.0 - mu_bar) + draws * (0.5 - mu_bar) + losses * (0.0 - mu_bar)
    llr = (delta_s / sigma_sq) * sum_centered

    if llr >= upper_bound:
        decision = "ACCEPT_H1"
    elif llr <= lower_bound:
        decision = "REJECT_H1"
    else:
        decision = "CONTINUE"

    lo_elo, hi_elo = agresti_caffo_elo_ci(wins, draws, losses, confidence=1.0 - alpha)
    elo_diff = score_to_elo(score)

    return {
        "n": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_diff": elo_diff,
        "elo_lo": lo_elo,
        "elo_hi": hi_elo,
        "llr": llr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "decision": decision,
    }


# Backward-compatible helper matching legacy MatchRunner signature
def elo_interval(wins: int, draws: int, losses: int) -> dict[str, Optional[float]]:
    """Backward-compatible replacement for MatchRunner.elo_interval.

    Replaces the defective clamped-Wald variant with canonical Agresti-Caffo (F-03).
    Returns {'elo_diff': ..., 'lo': ..., 'hi': ..., 'score': ...}.
    """
    res = elo_confidence_interval(wins, draws, losses, confidence=0.95, method="agresti_caffo")
    return {
        "elo_diff": res["elo_diff"],
        "lo": res["lo"],
        "hi": res["hi"],
        "score": res["score"],
    }
