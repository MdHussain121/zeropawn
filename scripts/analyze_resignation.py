"""Analysis script for Phase 4 resignation diagnosis.

Computes:
1. Reliability table with Wilson 95% confidence intervals
2. Resignation grid: evaluates t in {-0.90, -0.95, -0.97, -0.99} x k in {4, 6, 8} x min_ply in {0, 30}
3. End-to-end sign check across decisive games by ply range
4. Termination-reason breakdown, draw rate, and 400-ply cap rate
5. Dual-trigger frequency (games where both sides triggered would-resign)
6. Distribution of first would-resign ply
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def load_games(jsonl_path: str) -> List[Dict[str, Any]]:
    games = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def wilson_interval(successes: float, trials: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculate Wilson score interval for score proportion."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + (z**2) / trials
    centre = (p + (z**2) / (2.0 * trials)) / denom
    spread = (
        z * math.sqrt(max(0.0, (p * (1.0 - p)) / trials + (z**2) / (4.0 * (trials**2))))
    ) / denom
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (lower, upper)


def compute_reliability_table(games: List[Dict[str, Any]]) -> str:
    bins = [
        (-1.01, -0.95),
        (-0.95, -0.90),
        (-0.90, -0.80),
        (-0.80, -0.60),
        (-0.60, -0.40),
        (-0.40, -0.20),
        (-0.20, 0.00),
        (0.00, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.80),
        (0.80, 0.90),
        (0.90, 0.95),
        (0.95, 1.01),
    ]

    bin_counts = [0] * len(bins)
    bin_scores = [0.0] * len(bins)
    bin_values = [0.0] * len(bins)

    for g in games:
        result = g["result"]
        if result == "1-0":
            white_score = 1.0
        elif result == "0-1":
            white_score = 0.0
        else:
            white_score = 0.5

        for p in g.get("plies", []):
            side = p["side_to_move"]
            val = p["root_value"]
            actual_score = white_score if side == "white" else (1.0 - white_score)

            for b_idx, (b_low, b_high) in enumerate(bins):
                if b_low <= val < b_high or (b_idx == len(bins) - 1 and b_low <= val <= b_high):
                    bin_counts[b_idx] += 1
                    bin_scores[b_idx] += actual_score
                    bin_values[b_idx] += val
                    break

    lines = []
    header = f"{'Value Bin':<18} {'Count':<8} {'Pct':<8} {'Mean Val':<10} {'Mean Score':<12} {'Wilson 95% CI':<20} {'Expected':<10} {'Cal Gap':<10}"
    lines.append(header)
    lines.append("-" * len(header))

    total_plies = sum(bin_counts)
    for b_idx, (b_low, b_high) in enumerate(bins):
        cnt = bin_counts[b_idx]
        pct = (cnt / total_plies * 100.0) if total_plies > 0 else 0.0
        mean_v = (bin_values[b_idx] / cnt) if cnt > 0 else 0.0
        mean_s = (bin_scores[b_idx] / cnt) if cnt > 0 else 0.0
        ci_low, ci_high = wilson_interval(bin_scores[b_idx], cnt)
        ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]" if cnt > 0 else "N/A"
        exp_s = (mean_v + 1.0) / 2.0
        gap = mean_s - exp_s
        bin_label = f"[{max(-1.0, b_low):.2f}, {min(1.0, b_high):.2f})"
        lines.append(
            f"{bin_label:<18} {cnt:<8} {pct:<7.2f}% {mean_v:<+10.4f} {mean_s:<12.4f} {ci_str:<20} {exp_s:<10.4f} {gap:<+10.4f}"
        )

    lines.append("-" * len(header))
    lines.append(f"Total Evaluated Plies: {total_plies}")
    return "\n".join(lines)


def evaluate_sign_by_outcome(games: List[Dict[str, Any]]) -> str:
    ranges = [(0, 30), (30, 60), (60, 100), (100, 401)]
    winner_vals = {r: [] for r in ranges}
    loser_vals = {r: [] for r in ranges}
    decisive_count = 0

    for g in games:
        res = g["result"]
        if res not in ("1-0", "0-1"):
            continue
        decisive_count += 1
        winner_side = "white" if res == "1-0" else "black"

        for p in g.get("plies", []):
            ply_num = p["ply"]
            stm = p["side_to_move"]
            val = p["root_value"]
            for r in ranges:
                if r[0] <= ply_num < r[1]:
                    if stm == winner_side:
                        winner_vals[r].append(val)
                    else:
                        loser_vals[r].append(val)
                    break

    lines = []
    lines.append("=" * 85)
    lines.append(f"END-TO-END SIGN CHECK (Decisive Games: N={decisive_count} of {len(games)})")
    lines.append("Side-to-move root value on eventual winner's turns vs eventual loser's turns:")
    lines.append("-" * 85)
    lines.append(
        f"{'Ply Range':<15} {'Winner Turns (Count, Mean)':<32} {'Loser Turns (Count, Mean)':<32} {'Separation':<12}"
    )
    lines.append("-" * 85)
    for r in ranges:
        w_v = winner_vals[r]
        l_v = loser_vals[r]
        w_mean = np.mean(w_v) if w_v else 0.0
        l_mean = np.mean(l_v) if l_v else 0.0
        sep = w_mean - l_mean
        label = f"Plies {r[0]}-{r[1] - 1}"
        w_str = f"N={len(w_v):<5} mean={w_mean:<+7.4f}"
        l_str = f"N={len(l_v):<5} mean={l_mean:<+7.4f}"
        lines.append(f"{label:<15} {w_str:<32} {l_str:<32} sep={sep:<+7.4f}")
    lines.append("=" * 85)
    return "\n".join(lines)


def evaluate_grid(games: List[Dict[str, Any]]) -> str:
    thresholds = [-0.90, -0.95, -0.97, -0.99]
    consecutives = [4, 6, 8]
    min_plies = [0, 30]

    lines = []
    lines.append("=" * 115)
    lines.append(
        f"{'min_ply':<8} {'thresh (t)':<12} {'k':<4} {'Triggered':<10} {'True Resign':<13} {'Win Turnaround':<17} {'Draw Turnaround':<18} {'Total False Rate':<18} {'Both Trig':<10}"
    )
    lines.append("-" * 115)

    num_games = len(games)

    for mp in min_plies:
        for t in thresholds:
            for k in consecutives:
                triggered_count = 0
                true_resign = 0
                win_turnaround = 0
                draw_turnaround = 0
                both_sides_trig = 0

                for g in games:
                    res = g["result"]
                    g_trig_white = False
                    g_trig_black = False

                    for target_side in ["white", "black"]:
                        side_won = (res == "1-0") if target_side == "white" else (res == "0-1")
                        side_drew = res == "1/2-1/2"
                        side_lost = (res == "0-1") if target_side == "white" else (res == "1-0")

                        consec = 0
                        triggered = False

                        for p in g.get("plies", []):
                            if p["side_to_move"] != target_side:
                                continue
                            ply_num = p["ply"]
                            val = p["root_value"]

                            if ply_num >= mp and val < t:
                                consec += 1
                                if consec >= k:
                                    triggered = True
                                    break
                            else:
                                consec = 0

                        if triggered:
                            if target_side == "white":
                                g_trig_white = True
                            else:
                                g_trig_black = True
                            triggered_count += 1
                            if side_lost:
                                true_resign += 1
                            elif side_won:
                                win_turnaround += 1
                            elif side_drew:
                                draw_turnaround += 1

                    if g_trig_white and g_trig_black:
                        both_sides_trig += 1

                total_false = win_turnaround + draw_turnaround
                false_rate = (total_false / triggered_count * 100.0) if triggered_count > 0 else 0.0
                win_rate = (
                    (win_turnaround / triggered_count * 100.0) if triggered_count > 0 else 0.0
                )
                draw_rate = (
                    (draw_turnaround / triggered_count * 100.0) if triggered_count > 0 else 0.0
                )

                lines.append(
                    f"{mp:<8} {t:<12.2f} {k:<4} {triggered_count:<10} {true_resign:<13} "
                    f"{win_turnaround:>4} ({win_rate:4.1f}%)     "
                    f"{draw_turnaround:>4} ({draw_rate:4.1f}%)      "
                    f"{total_false:>4} ({false_rate:5.1f}%)     "
                    f"{both_sides_trig:<10}"
                )

        lines.append("-" * 115)

    lines.append(f"Evaluated across {num_games} total games.")
    lines.append("=" * 115)
    return "\n".join(lines)


def evaluate_termination(games: List[Dict[str, Any]]) -> str:
    reasons = defaultdict(int)
    results = defaultdict(int)
    lengths = []
    both_triggered_t95_k4 = 0

    first_plies_t95_k4 = []

    for g in games:
        reasons[g.get("termination_reason", "unknown")] += 1
        results[g.get("result", "unknown")] += 1
        lengths.append(g.get("length_plies", len(g.get("moves", []))))

        white_trig = False
        black_trig = False
        earliest_ply = None

        w_consec = 0
        b_consec = 0
        for p in g.get("plies", []):
            if p["ply"] < 30:
                continue
            if p["side_to_move"] == "white":
                if p["root_value"] < -0.95:
                    w_consec += 1
                    if w_consec >= 4:
                        white_trig = True
                        if earliest_ply is None or p["ply"] < earliest_ply:
                            earliest_ply = p["ply"]
                else:
                    w_consec = 0
            else:
                if p["root_value"] < -0.95:
                    b_consec += 1
                    if b_consec >= 4:
                        black_trig = True
                        if earliest_ply is None or p["ply"] < earliest_ply:
                            earliest_ply = p["ply"]
                else:
                    b_consec = 0

        if white_trig and black_trig:
            both_triggered_t95_k4 += 1
        if earliest_ply is not None:
            first_plies_t95_k4.append(earliest_ply)

    total = len(games)
    lines = []
    lines.append("=" * 60)
    lines.append("TERMINATION REASONS BREAKDOWN:")
    lines.append("-" * 60)
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100.0 if total > 0 else 0.0
        lines.append(f"  {r:<25} {cnt:>4} / {total} ({pct:5.1f}%)")

    lines.append("-" * 60)
    lines.append("GAME RESULTS BREAKDOWN:")
    lines.append("-" * 60)
    for res, cnt in sorted(results.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100.0 if total > 0 else 0.0
        lines.append(f"  {res:<25} {cnt:>4} / {total} ({pct:5.1f}%)")

    lines.append("-" * 60)
    lines.append("GAME LENGTH STATISTICS (Plies):")
    lines.append("-" * 60)
    if lengths:
        lines.append(f"  Mean:   {np.mean(lengths):.1f}")
        lines.append(f"  Median: {np.median(lengths):.1f}")
        lines.append(f"  Min:    {np.min(lengths)}")
        lines.append(f"  Max:    {np.max(lengths)}")
        cap_400 = sum(1 for l in lengths if l >= 400)
        lines.append(f"  Hit 400-ply cap: {cap_400} ({cap_400 / total * 100:.1f}%)")

    lines.append("-" * 60)
    lines.append("DUAL TRIGGER (Both sides triggered t=-0.95, k=4, ply>=30):")
    lines.append(
        f"  Count: {both_triggered_t95_k4} / {total} ({both_triggered_t95_k4 / total * 100:.1f}%)"
    )

    lines.append("-" * 60)
    lines.append("FIRST WOULD-RESIGN PLY DISTRIBUTION (t=-0.95, k=4, ply>=30):")
    if first_plies_t95_k4:
        lines.append(
            f"  Games triggered: {len(first_plies_t95_k4)} / {total} ({len(first_plies_t95_k4) / total * 100:.1f}%)"
        )
        lines.append(f"  Mean:   {np.mean(first_plies_t95_k4):.1f} ply")
        lines.append(f"  Median: {np.median(first_plies_t95_k4):.1f} ply")
        lines.append(f"  Min:    {np.min(first_plies_t95_k4)} ply")
        lines.append(f"  Max:    {np.max(first_plies_t95_k4)} ply")
        q25, q75 = np.percentile(first_plies_t95_k4, [25, 75])
        lines.append(f"  IQR:    [25th={q25:.1f}, 75th={q75:.1f}]")
    else:
        lines.append("  No games triggered.")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Resignation Games")
    parser.add_argument("input", type=str, help="Path to diagnosis_games.jsonl")
    args = parser.parse_args()

    games = load_games(args.input)
    print(f"\nLoaded {len(games)} games from {args.input}\n")

    print(evaluate_termination(games))
    print("\n" + evaluate_sign_by_outcome(games))
    print("\n" + "=" * 105)
    print("ROOT VALUE RELIABILITY TABLE")
    print("=" * 105)
    print(compute_reliability_table(games))
    print("\n" + "=" * 115)
    print("RESIGNATION GRID EVALUATION")
    print("=" * 115)
    print(evaluate_grid(games))


if __name__ == "__main__":
    main()
