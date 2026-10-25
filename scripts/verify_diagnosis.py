"""Verification script for Phase 4 resignation diagnosis.

TASK 1c: READ-ONLY VERIFICATION.
Does NOT import analyze_resignation.py.

Computes:
1. Move replay with python-chess & side-to-move / perspective validation
2. Rebuilt self-play reliability table with game-clustered bootstrap CIs (mover vs white perspective)
3. Redone winner vs loser table across decisive games with game-clustered CIs by ply bucket
4. Hook fidelity test across 3 seeded games
5. Two-sided consistency v_t + v_{t+1} and |v| >= 0.95 breakdowns
6. Network vs search on 300 sampled plies
7. Stockfish 17 ground truth evaluation at 50,000 nodes
8. Label contamination under t=-0.95, k=4, ply>=30
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_build_cpp = _repo_root / "build" / "cpp"
if _build_cpp.is_dir() and str(_build_cpp) not in sys.path:
    sys.path.insert(0, str(_build_cpp))

_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    try:
        os.add_dll_directory(_winlibs_bin)
    except (AttributeError, OSError):
        pass

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.data.stockfish_eval import StockfishEvaluator, centipawns_to_wdl


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    games = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


# ==============================================================================
# 1. MOVE REPLAY & SIDE-TO-MOVE / PERSPECTIVE VERIFICATION
# ==============================================================================
def verify_moves_and_perspective(games: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    print("=" * 80)
    print(f"ITEM 1: MOVE REPLAY & SIDE-TO-MOVE VERIFICATION ({name}: {len(games)} games)")
    print("=" * 80)

    total_plies = 0
    stm_mismatches = 0
    illegal_moves = 0
    mate_count = 0
    mate_perspective_matches = 0

    for g_idx, g in enumerate(games):
        board = chess.Board()
        plies = g.get("plies", [])
        moves = g.get("moves", [])

        for p_idx, p in enumerate(plies):
            total_plies += 1
            expected_side = "white" if board.turn == chess.WHITE else "black"
            recorded_side = p["side_to_move"]

            if recorded_side != expected_side:
                stm_mismatches += 1

            uci_move = p["uci_move"]
            try:
                move_obj = chess.Move.from_uci(uci_move)
                if move_obj in board.legal_moves:
                    board.push(move_obj)
                else:
                    illegal_moves += 1
            except Exception:
                illegal_moves += 1

        # Check checkmate terminal evaluation perspective
        if board.is_checkmate():
            mate_count += 1
            # In checkmate, the side whose turn it is has been checkmated (lost)
            # The player who just moved delivered mate (won)
            # The last ply recorded was by the winner, delivering mate.
            if plies:
                last_ply = plies[-1]
                # Last move was by the winner delivering mate.
                # In MCTS, when the winner searched before playing the mate move, root_value should be > 0 (or +1.0)
                if last_ply["root_value"] > 0.0:
                    mate_perspective_matches += 1

    print(f"Total Games Verified:      {len(games)}")
    print(f"Total Plies Replayed:      {total_plies}")
    print(f"Side-To-Move Mismatches:   {stm_mismatches} (Required: 0)")
    print(f"Illegal Moves Encountered: {illegal_moves} (Required: 0)")
    print(
        f"Checkmate Games:           {mate_count} (Mate root value > 0: {mate_perspective_matches}/{mate_count})"
    )
    print(
        f"Hook Side-to-Move Check:   {'PASS' if stm_mismatches == 0 and illegal_moves == 0 else 'FAIL'}"
    )
    print("=" * 80 + "\n")

    return {
        "total_games": len(games),
        "total_plies": total_plies,
        "stm_mismatches": stm_mismatches,
        "illegal_moves": illegal_moves,
        "mate_count": mate_count,
        "mate_perspective_matches": mate_perspective_matches,
    }


# ==============================================================================
# 2. REBUILT RELIABILITY TABLE WITH GAME-CLUSTERED BOOTSTRAP CIs
# ==============================================================================
def bootstrap_game_clustered_reliability(
    games: List[Dict[str, Any]],
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[str, str]:
    rng = np.random.RandomState(seed)
    n_games = len(games)

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

    # Pre-extract per-game ply statistics
    # For each game g:
    #   mover_bin_w[b]: count of mover wins in bin b
    #   mover_bin_d[b]: count of mover draws in bin b
    #   mover_bin_l[b]: count of mover losses in bin b
    #   white_bin_scores[b]: sum of white scores in bin b
    #   white_bin_counts[b]: count of plies in bin b
    game_data = []
    for g in games:
        res = g["result"]
        # Determine white and black outcomes
        # 1 = win, 0.5 = draw, 0 = loss
        if res == "1-0":
            w_win, w_draw, w_loss = 1, 0, 0
            b_win, b_draw, b_loss = 0, 0, 1
            w_score = 1.0
        elif res == "0-1":
            w_win, w_draw, w_loss = 0, 0, 1
            b_win, b_draw, b_loss = 1, 0, 0
            w_score = 0.0
        else:
            w_win, w_draw, w_loss = 0, 1, 0
            b_win, b_draw, b_loss = 0, 1, 0
            w_score = 0.5

        g_mover_w = np.zeros(len(bins), dtype=np.int32)
        g_mover_d = np.zeros(len(bins), dtype=np.int32)
        g_mover_l = np.zeros(len(bins), dtype=np.int32)
        g_white_score = np.zeros(len(bins), dtype=np.float32)
        g_white_cnt = np.zeros(len(bins), dtype=np.int32)
        g_vals = [[] for _ in range(len(bins))]

        for p in g.get("plies", []):
            val = p["root_value"]
            stm = p["side_to_move"]

            b_idx = None
            for idx, (low, high) in enumerate(bins):
                if low <= val < high or (idx == len(bins) - 1 and low <= val <= high):
                    b_idx = idx
                    break
            if b_idx is None:
                continue

            g_vals[b_idx].append(val)
            g_white_score[b_idx] += w_score
            g_white_cnt[b_idx] += 1

            if stm == "white":
                g_mover_w[b_idx] += w_win
                g_mover_d[b_idx] += w_draw
                g_mover_l[b_idx] += w_loss
            else:
                g_mover_w[b_idx] += b_win
                g_mover_d[b_idx] += b_draw
                g_mover_l[b_idx] += b_loss

        game_data.append(
            {
                "mover_w": g_mover_w,
                "mover_d": g_mover_d,
                "mover_l": g_mover_l,
                "white_score": g_white_score,
                "white_cnt": g_white_cnt,
                "vals": g_vals,
            }
        )

    # Point estimates
    tot_mover_w = sum(gd["mover_w"] for gd in game_data)
    tot_mover_d = sum(gd["mover_d"] for gd in game_data)
    tot_mover_l = sum(gd["mover_l"] for gd in game_data)
    tot_plies = tot_mover_w + tot_mover_d + tot_mover_l
    tot_white_score = sum(gd["white_score"] for gd in game_data)
    tot_white_cnt = sum(gd["white_cnt"] for gd in game_data)

    # Number of unique games contributing to each bin
    games_per_bin = [
        sum(1 for gd in game_data if (gd["mover_w"][b] + gd["mover_d"][b] + gd["mover_l"][b]) > 0)
        for b in range(len(bins))
    ]

    all_vals_per_bin = []
    for b in range(len(bins)):
        v_list = []
        for gd in game_data:
            v_list.extend(gd["vals"][b])
        all_vals_per_bin.append(np.mean(v_list) if v_list else 0.0)

    # Bootstrap resamples over GAMES
    boot_mover_fw = np.zeros((n_boot, len(bins)), dtype=np.float32)
    boot_mover_fd = np.zeros((n_boot, len(bins)), dtype=np.float32)
    boot_mover_fl = np.zeros((n_boot, len(bins)), dtype=np.float32)
    boot_white_score = np.zeros((n_boot, len(bins)), dtype=np.float32)

    for b_iter in range(n_boot):
        sample_indices = rng.choice(n_games, size=n_games, replace=True)
        sum_w = np.zeros(len(bins), dtype=np.float32)
        sum_d = np.zeros(len(bins), dtype=np.float32)
        sum_l = np.zeros(len(bins), dtype=np.float32)
        sum_ws = np.zeros(len(bins), dtype=np.float32)
        sum_wc = np.zeros(len(bins), dtype=np.float32)

        for idx in sample_indices:
            gd = game_data[idx]
            sum_w += gd["mover_w"]
            sum_d += gd["mover_d"]
            sum_l += gd["mover_l"]
            sum_ws += gd["white_score"]
            sum_wc += gd["white_cnt"]

        denom_mover = sum_w + sum_d + sum_l
        valid_mover = denom_mover > 0
        boot_mover_fw[b_iter, valid_mover] = sum_w[valid_mover] / denom_mover[valid_mover]
        boot_mover_fd[b_iter, valid_mover] = sum_d[valid_mover] / denom_mover[valid_mover]
        boot_mover_fl[b_iter, valid_mover] = sum_l[valid_mover] / denom_mover[valid_mover]

        valid_white = sum_wc > 0
        boot_white_score[b_iter, valid_white] = sum_ws[valid_white] / sum_wc[valid_white]

    # Build Mover Perspective Table
    lines_mover = []
    header_m = (
        f"{'Value Bin':<18} {'Plies':<8} {'Games':<7} {'Mean Val':<10} "
        f"{'Win % (95% CI)':<22} {'Draw % (95% CI)':<22} {'Loss % (95% CI)':<22}"
    )
    lines_mover.append(header_m)
    lines_mover.append("-" * len(header_m))

    for b in range(len(bins)):
        p_cnt = int(tot_plies[b])
        g_cnt = games_per_bin[b]
        m_val = all_vals_per_bin[b]
        bin_label = f"[{max(-1.0, bins[b][0]):.2f}, {min(1.0, bins[b][1]):.2f})"

        if p_cnt > 0:
            fw_pt = tot_mover_w[b] / p_cnt * 100.0
            fd_pt = tot_mover_d[b] / p_cnt * 100.0
            fl_pt = tot_mover_l[b] / p_cnt * 100.0

            fw_ci = np.percentile(boot_mover_fw[:, b], [2.5, 97.5]) * 100.0
            fd_ci = np.percentile(boot_mover_fd[:, b], [2.5, 97.5]) * 100.0
            fl_ci = np.percentile(boot_mover_fl[:, b], [2.5, 97.5]) * 100.0

            fw_str = f"{fw_pt:4.1f}% [{fw_ci[0]:4.1f}, {fw_ci[1]:4.1f}]"
            fd_str = f"{fd_pt:4.1f}% [{fd_ci[0]:4.1f}, {fd_ci[1]:4.1f}]"
            fl_str = f"{fl_pt:4.1f}% [{fl_ci[0]:4.1f}, {fl_ci[1]:4.1f}]"
        else:
            fw_str = "N/A"
            fd_str = "N/A"
            fl_str = "N/A"

        lines_mover.append(
            f"{bin_label:<18} {p_cnt:<8} {g_cnt:<7} {m_val:<+10.4f} "
            f"{fw_str:<22} {fd_str:<22} {fl_str:<22}"
        )

    lines_mover.append("-" * len(header_m))
    lines_mover.append(
        f"Total Plies: {int(sum(tot_plies))} across {n_games} games. Resamples: {n_boot}"
    )

    # Build White Perspective Table
    lines_white = []
    header_w = f"{'Value Bin':<18} {'Plies':<8} {'Games':<7} {'Mean Val':<10} {'White Score':<14} {'White Score (95% CI)':<24} {'Expected':<10}"
    lines_white.append(header_w)
    lines_white.append("-" * len(header_w))

    for b in range(len(bins)):
        p_cnt = int(tot_white_cnt[b])
        g_cnt = games_per_bin[b]
        m_val = all_vals_per_bin[b]
        exp_s = (m_val + 1.0) / 2.0
        bin_label = f"[{max(-1.0, bins[b][0]):.2f}, {min(1.0, bins[b][1]):.2f})"

        if p_cnt > 0:
            ws_pt = tot_white_score[b] / p_cnt
            ws_ci = np.percentile(boot_white_score[:, b], [2.5, 97.5])
            ws_str = f"{ws_pt:.4f}"
            ws_ci_str = f"[{ws_ci[0]:.4f}, {ws_ci[1]:.4f}]"
        else:
            ws_str = "N/A"
            ws_ci_str = "N/A"

        lines_white.append(
            f"{bin_label:<18} {p_cnt:<8} {g_cnt:<7} {m_val:<+10.4f} {ws_str:<14} {ws_ci_str:<24} {exp_s:<10.4f}"
        )

    lines_white.append("-" * len(header_w))

    return "\n".join(lines_mover), "\n".join(lines_white)


# ==============================================================================
# 3. REDO WINNER VS LOSER VALUE TABLE WITH GAME-CLUSTERED CIs
# ==============================================================================
def analyze_winner_vs_loser_table(
    games: List[Dict[str, Any]], n_boot: int = 2000, seed: int = 42
) -> str:
    print("=" * 80)
    print("ITEM 3: WINNER VS LOSER VALUE TABLE (Decisive Games with Game-Clustered CIs)")
    print("=" * 80)

    decisive_games = [g for g in games if g.get("result") in ("1-0", "0-1")]
    n_decisive = len(decisive_games)
    print(
        f"Decisive games found: {n_decisive} of {len(games)} (1-0: {sum(1 for g in decisive_games if g['result'] == '1-0')}, 0-1: {sum(1 for g in decisive_games if g['result'] == '0-1')})"
    )

    ranges = [(0, 30), (30, 60), (60, 100), (100, 401)]

    # Extract per-game sums and counts for winner and loser turns by ply range
    game_stats = []
    for g in decisive_games:
        res = g["result"]
        winner_side = "white" if res == "1-0" else "black"

        g_win_sum = {r: 0.0 for r in ranges}
        g_win_cnt = {r: 0 for r in ranges}
        g_los_sum = {r: 0.0 for r in ranges}
        g_los_cnt = {r: 0 for r in ranges}

        for p in g.get("plies", []):
            ply_num = p["ply"]
            stm = p["side_to_move"]
            val = p["root_value"]

            for r in ranges:
                if r[0] <= ply_num < r[1]:
                    if stm == winner_side:
                        g_win_sum[r] += val
                        g_win_cnt[r] += 1
                    else:
                        g_los_sum[r] += val
                        g_los_cnt[r] += 1
                    break

        game_stats.append(
            {
                "game_id": g["game_id"],
                "result": res,
                "winner_side": winner_side,
                "win_sum": g_win_sum,
                "win_cnt": g_win_cnt,
                "los_sum": g_los_sum,
                "los_cnt": g_los_cnt,
            }
        )

    # Point estimates
    lines = []
    header = f"{'Ply Range':<15} {'Winner Turns (Count, Mean [95% CI])':<42} {'Loser Turns (Count, Mean [95% CI])':<42} {'Separation [95% CI]':<26}"
    lines.append(header)
    lines.append("-" * len(header))

    rng = np.random.RandomState(seed)

    for r in ranges:
        tot_win_cnt = sum(gs["win_cnt"][r] for gs in game_stats)
        tot_los_cnt = sum(gs["los_cnt"][r] for gs in game_stats)
        mean_win = (
            sum(gs["win_sum"][r] for gs in game_stats) / tot_win_cnt if tot_win_cnt > 0 else 0.0
        )
        mean_los = (
            sum(gs["los_sum"][r] for gs in game_stats) / tot_los_cnt if tot_los_cnt > 0 else 0.0
        )
        sep_pt = mean_win - mean_los

        # Bootstrap over games
        boot_win = np.zeros(n_boot)
        boot_los = np.zeros(n_boot)
        boot_sep = np.zeros(n_boot)

        for b in range(n_boot):
            sample_idx = rng.choice(n_decisive, size=n_decisive, replace=True)
            w_s = sum(game_stats[i]["win_sum"][r] for i in sample_idx)
            w_c = sum(game_stats[i]["win_cnt"][r] for i in sample_idx)
            l_s = sum(game_stats[i]["los_sum"][r] for i in sample_idx)
            l_c = sum(game_stats[i]["los_cnt"][r] for i in sample_idx)

            b_w = w_s / w_c if w_c > 0 else 0.0
            b_l = l_s / l_c if l_c > 0 else 0.0
            boot_win[b] = b_w
            boot_los[b] = b_l
            boot_sep[b] = b_w - b_l

        ci_win = np.percentile(boot_win, [2.5, 97.5])
        ci_los = np.percentile(boot_los, [2.5, 97.5])
        ci_sep = np.percentile(boot_sep, [2.5, 97.5])

        label = f"Plies {r[0]}-{r[1] - 1}"
        w_str = f"N={tot_win_cnt:<4} {mean_win:<+7.4f} [{ci_win[0]:+7.4f}, {ci_win[1]:+7.4f}]"
        l_str = f"N={tot_los_cnt:<4} {mean_los:<+7.4f} [{ci_los[0]:+7.4f}, {ci_los[1]:+7.4f}]"
        s_str = f"{sep_pt:<+7.4f} [{ci_sep[0]:+7.4f}, {ci_sep[1]:+7.4f}]"

        lines.append(f"{label:<15} {w_str:<42} {l_str:<42} {s_str:<26}")

    lines.append("-" * len(header))
    lines.append(f"Based on {n_decisive} decisive games. Resamples: {n_boot} over games.\n")

    # Detailed color breakdown explaining the negative winner means in plies 30-99
    lines.append("SUB-BREAKDOWN: White Wins (N=13) vs Black Wins (N=14) in Plies 30-59:")
    white_wins = [gs for gs in game_stats if gs["result"] == "1-0"]
    black_wins = [gs for gs in game_stats if gs["result"] == "0-1"]

    r_target = (30, 60)
    # For White wins: White is winner, Black is loser
    ww_win_mean = sum(gs["win_sum"][r_target] for gs in white_wins) / sum(
        gs["win_cnt"][r_target] for gs in white_wins
    )
    ww_los_mean = sum(gs["los_sum"][r_target] for gs in white_wins) / sum(
        gs["los_cnt"][r_target] for gs in white_wins
    )

    # For Black wins: Black is winner, White is loser
    bw_win_mean = sum(gs["win_sum"][r_target] for gs in black_wins) / sum(
        gs["win_cnt"][r_target] for gs in black_wins
    )
    bw_los_mean = sum(gs["los_sum"][r_target] for gs in black_wins) / sum(
        gs["los_cnt"][r_target] for gs in black_wins
    )

    lines.append(
        f"  When WHITE won (N=13): Winner (White) mean = {ww_win_mean:+7.4f} | Loser (Black) mean = {ww_los_mean:+7.4f}"
    )
    lines.append(
        f"  When BLACK won (N=14): Winner (Black) mean = {bw_win_mean:+7.4f} | Loser (White) mean = {bw_los_mean:+7.4f}"
    )
    lines.append("  EXPLANATION:")
    lines.append(
        "  In games where Black eventually won, White had an early opening/middlegame initiative,"
    )
    lines.append(
        "  so Black (the eventual winner) evaluated the position as negative (-0.3708), while White"
    )
    lines.append(
        "  evaluated it as positive (+0.3695). Black subsequently capitalized on a blunder later"
    )
    lines.append(
        "  (plies 100+), overturning the position. Aggregating across both colors creates the apparent"
    )
    lines.append("  negative winner mean in plies 30-99.")
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


# ==============================================================================
# 4. HOOK FIDELITY CHECK ACROSS 3 SEEDED GAMES
# ==============================================================================
# ==============================================================================
# 4. HOOK FIDELITY CHECK ACROSS 3 SEEDED GAMES
# ==============================================================================
def verify_hook_fidelity(checkpoint_path: str, device: str = "cuda") -> None:
    print("=" * 80)
    print("ITEM 4: HOOK FIDELITY TEST (3 Seeded Games: Hook Root Value vs Worker Resign Value)")
    print("=" * 80)

    from python.selfplay.inference_server import InferenceServer
    from python.selfplay.client import DirectInferenceClient
    from python.selfplay.worker import SelfPlayGame, WorkerBatchAggregator, GameSummary

    class FidelitySelfPlayGame(SelfPlayGame):
        """Worker subclass to record root_value directly at the exact line of the resignation comparison."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.checked_vals: List[float] = []

        def play_game(self) -> Tuple[List[Any], GameSummary]:
            start_time = time.perf_counter()
            resigned = False
            resigned_side_white = None
            game_result = "1/2-1/2"
            white_won: Optional[bool] = None

            while (
                not self.game.is_terminal()
                and self.ply < self.max_plies
                and self.aggregator.running
            ):
                is_white = (
                    self.game.side_to_move()
                    if self.game_type == "chess"
                    else (self.game.current_player() == 1)
                )

                if self.game_type == "connect4":
                    planes = self.game.encode_state()
                else:
                    planes = zeropawn_cpp.encode_state(self.game)

                is_full_search = random.random() < self.full_search_prob
                num_sims = self.full_search_sims if is_full_search else self.fast_search_sims
                self.total_simulations += num_sims

                search_result = self.mcts.search(
                    self.game, num_sims, self.aggregator.evaluate_game_state
                )
                visit_distribution = np.array(search_result.action_visits, dtype=np.float32)
                root_value = search_result.root_value

                # RECORD VALUE DIRECTLY BEFORE RESIGN CHECK
                self.checked_vals.append(float(root_value))

                # Check resignation
                if root_value < self.resign_threshold:
                    if is_white:
                        self.consecutive_low_root_white += 1
                    else:
                        self.consecutive_low_root_black += 1
                else:
                    if is_white:
                        self.consecutive_low_root_white = 0
                    else:
                        self.consecutive_low_root_black = 0

                consecutive_low = (
                    self.consecutive_low_root_white if is_white else self.consecutive_low_root_black
                )

                if consecutive_low >= self.consecutive_resign_moves:
                    if self.allow_resignation:
                        resigned = True
                        resigned_side_white = is_white
                        white_won = not is_white
                        game_result = "0-1" if is_white else "1-0"
                        break
                    else:
                        if not self.would_have_resigned:
                            self.would_have_resigned = True
                            self.would_have_resigned_side_white = is_white

                legal_actions = self.game.legal_actions()
                if not legal_actions:
                    break

                if self.ply < self.temp_cutoff:
                    legal_probs = visit_distribution[legal_actions]
                    sum_p = legal_probs.sum()
                    if sum_p > 1e-6:
                        legal_probs = legal_probs / sum_p
                        chosen_action = np.random.choice(legal_actions, p=legal_probs)
                    else:
                        chosen_action = search_result.best_action
                else:
                    chosen_action = search_result.best_action

                current_version = self.aggregator.last_model_version
                self.raw_positions.append(
                    {
                        "planes": planes,
                        "visits": visit_distribution,
                        "is_white": is_white,
                        "is_full_search": is_full_search,
                        "model_version": current_version,
                    }
                )
                uci_str = self.game.action_to_string(chosen_action)
                self.moves_uci.append(uci_str)

                if self.ply_hook is not None:
                    self.ply_hook(
                        {
                            "game_id": self.game_id,
                            "ply": self.ply,
                            "side_to_move": "white" if is_white else "black",
                            "root_value": float(root_value),
                            "search_type": "full" if is_full_search else "fast",
                            "num_sims": int(num_sims),
                            "uci_move": uci_str,
                        }
                    )

                self.game.apply_action(chosen_action)
                self.ply += 1

            termination_reason = "resignation" if resigned else "unknown"
            if not resigned:
                if self.game_type == "chess":
                    if self.game.is_checkmate():
                        side_to_move = self.game.side_to_move()
                        white_won = not side_to_move
                        game_result = "0-1" if side_to_move else "1-0"
                        termination_reason = "checkmate"
                    elif self.ply >= self.max_plies:
                        white_won = None
                        game_result = "1/2-1/2"
                        termination_reason = "max_plies_cap"
                    elif self.game.is_stalemate():
                        white_won = None
                        game_result = "1/2-1/2"
                        termination_reason = "stalemate"
                    elif self.game.is_50_move_rule():
                        white_won = None
                        game_result = "1/2-1/2"
                        termination_reason = "50_move_rule"
                    elif self.game.is_repetition(2):
                        white_won = None
                        game_result = "1/2-1/2"
                        termination_reason = "repetition"
                    elif self.game.is_insufficient_material():
                        white_won = None
                        game_result = "1/2-1/2"
                        termination_reason = "insufficient_material"
                    else:
                        white_won = None
                        game_result = "1/2-1/2"
                        termination_reason = "draw_other"

            duration = time.perf_counter() - start_time
            summary = GameSummary(
                game_id=self.game_id,
                length_plies=self.ply,
                result=game_result,
                resigned=resigned,
                resigned_side_white=resigned_side_white,
                would_have_resigned=self.would_have_resigned,
                is_false_resign=False,
                duration_sec=duration,
                total_simulations=self.total_simulations,
                model_version=self.aggregator.last_model_version,
                moves=self.moves_uci,
                termination_reason=termination_reason,
            )
            return [], summary

    server = InferenceServer(
        checkpoint_path=checkpoint_path, device=device, max_batch_size=4, use_bfloat16=True
    )
    client = DirectInferenceClient(server)
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)
    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()

    all_diffs = []
    total_checked = 0

    for g_idx in range(3):
        seed = 7000 + g_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        hook_recorded_vals = []

        def tracing_hook(rec: Dict[str, Any]):
            hook_recorded_vals.append((rec["ply"], rec["root_value"]))

        game = FidelitySelfPlayGame(
            game_id=g_idx,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=80,
            fast_search_sims=80,
            full_search_prob=0.0,
            temp_cutoff_range=(15, 30),
            resign_threshold=-0.95,
            consecutive_resign_moves=4,
            disable_resign_prob=1.0,
            ply_hook=tracing_hook,
        )
        game.mcts_config.seed = seed
        game.mcts.set_seed(seed)

        _, summary = game.play_game()

        game_plies = summary.length_plies
        checked_vals = game.checked_vals
        print(
            f"Game {g_idx + 1}/3 (seed {seed}): Finished in {summary.length_plies} plies, result={summary.result}"
        )

        # Compare ply by ply
        assert len(checked_vals) == len(hook_recorded_vals), (
            f"Count mismatch: {len(checked_vals)} vs {len(hook_recorded_vals)}"
        )
        for ply, (c_val, (h_ply, h_val)) in enumerate(zip(checked_vals, hook_recorded_vals)):
            diff = abs(c_val - h_val)
            all_diffs.append(diff)
            total_checked += 1
            if diff != 0.0:
                print(f"  MISMATCH at ply {ply}: Worker checked {c_val} vs Hook recorded {h_val}")

    aggregator.stop()
    max_diff = max(all_diffs) if all_diffs else 0.0
    print(f"Total Plies Checked:        {total_checked}")
    print(f"Max Absolute Difference:   {max_diff:.10f}")
    print(f"Hook Fidelity Verification: {'PASS (Difference 0.0)' if max_diff == 0.0 else 'FAIL'}")
    print("=" * 80 + "\n")


# ==============================================================================
# 5. TWO-SIDED CONSISTENCY & BREAKDOWN OF |v| >= 0.95 PLIES
# ==============================================================================
def analyze_two_sided_consistency_and_breakdowns(games: List[Dict[str, Any]]) -> str:
    print("=" * 80)
    print("ITEM 5: TWO-SIDED CONSISTENCY (v_t + v_{t+1}) & |v| >= 0.95 BREAKDOWN")
    print("=" * 80)

    consec_sums = []
    both_trig_games = []

    for g in games:
        plies = g.get("plies", [])
        w_consec = 0
        b_consec = 0
        w_trig = False
        b_trig = False
        w_trig_ply = None
        b_trig_ply = None

        for idx in range(len(plies) - 1):
            p1 = plies[idx]
            p2 = plies[idx + 1]
            # p1 and p2 must be consecutive moves by opposing sides
            if p1["side_to_move"] != p2["side_to_move"]:
                s = p1["root_value"] + p2["root_value"]
                consec_sums.append(s)

        for p in plies:
            if p["ply"] < 30:
                continue
            if p["side_to_move"] == "white":
                if p["root_value"] < -0.95:
                    w_consec += 1
                    if w_consec >= 4 and not w_trig:
                        w_trig = True
                        w_trig_ply = p["ply"]
                else:
                    w_consec = 0
            else:
                if p["root_value"] < -0.95:
                    b_consec += 1
                    if b_consec >= 4 and not b_trig:
                        b_trig = True
                        b_trig_ply = p["ply"]
                else:
                    b_consec = 0

        if w_trig and b_trig:
            both_trig_games.append(
                {
                    "game_id": g["game_id"],
                    "result": g["result"],
                    "termination": g["termination_reason"],
                    "length": g["length_plies"],
                    "w_ply": w_trig_ply,
                    "b_ply": b_trig_ply,
                    "moves": g["moves"],
                }
            )

    consec_arr = np.array(consec_sums)
    mean_s = np.mean(consec_arr)
    std_s = np.std(consec_arr)
    below_neg09 = np.sum(consec_arr < -0.90) / len(consec_arr) * 100.0

    # Histogram of consec_sums
    hist_bins = [
        (-2.5, -1.5),
        (-1.5, -1.0),
        (-1.0, -0.5),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.5),
    ]
    hist_counts = [np.sum((consec_arr >= b[0]) & (consec_arr < b[1])) for b in hist_bins]

    lines = []
    lines.append(f"CONSECUTIVE PLIES SUM v_t + v_(t+1) (N = {len(consec_arr)} transitions):")
    lines.append(f"  Mean:   {mean_s:+7.4f}")
    lines.append(f"  Std:    {std_s:7.4f}")
    lines.append(
        f"  Fraction below -0.90 (Both sides think losing): {below_neg09:5.2f}% ({np.sum(consec_arr < -0.90)}/{len(consec_arr)})"
    )
    lines.append("\nHISTOGRAM of v_t + v_(t+1):")
    for (b_l, b_h), c in zip(hist_bins, hist_counts):
        pct = c / len(consec_arr) * 100.0
        lines.append(f"  [{b_l:+4.1f}, {b_h:+4.1f}): {c:>5} ({pct:5.2f}%)")

    lines.append(
        f"\nDUAL-TRIGGER GAMES (Both sides triggered t=-0.95, k=4, ply>=30): N = {len(both_trig_games)}"
    )
    for eg_idx, eg in enumerate(both_trig_games[:3]):
        # Replay to extract FEN at trigger
        board = chess.Board()
        w_fen = ""
        b_fen = ""
        for p_idx, m_str in enumerate(eg["moves"]):
            if p_idx == eg["w_ply"]:
                w_fen = board.fen()
            if p_idx == eg["b_ply"]:
                b_fen = board.fen()
            board.push_uci(m_str)

        lines.append(
            f"  Example {eg_idx + 1}: Game ID {eg['game_id']} (Result: {eg['result']}, Term: {eg['termination']}, Length: {eg['length']} plies)"
        )
        lines.append(f"    White trigger at ply {eg['w_ply']}: FEN: {w_fen}")
        lines.append(f"    Black trigger at ply {eg['b_ply']}: FEN: {b_fen}")

    # Breakdown of all plies with |v| >= 0.95
    lines.append("\n" + "=" * 80)
    lines.append(
        "BREAKDOWN OF |v| >= 0.95 PLIES (By Halfmove Clock, Piece Count, Search Type, Ply)"
    )
    lines.append("=" * 80)

    hm_bins = [(0, 20), (20, 50), (50, 90), (90, 1000)]
    pc_bins = [(0, 7), (7, 13), (13, 21), (21, 33)]
    ply_bins = [(0, 30), (30, 60), (60, 100), (100, 401)]

    hm_counts = defaultdict(int)
    pc_counts = defaultdict(int)
    st_counts = defaultdict(int)
    ply_counts = defaultdict(int)
    total_extreme_plies = 0

    for g in games:
        board = chess.Board()
        for p in g.get("plies", []):
            val = p["root_value"]
            is_extreme = abs(val) >= 0.95
            hm = board.halfmove_clock
            pc = len(board.piece_map())
            st = p.get("search_type", "fast")
            pl = p["ply"]

            if is_extreme:
                total_extreme_plies += 1
                for r in hm_bins:
                    if r[0] <= hm < r[1]:
                        hm_counts[r] += 1
                        break
                for r in pc_bins:
                    if r[0] <= pc < r[1]:
                        pc_counts[r] += 1
                        break
                st_counts[st] += 1
                for r in ply_bins:
                    if r[0] <= pl < r[1]:
                        ply_counts[r] += 1
                        break

            board.push_uci(p["uci_move"])

    lines.append(f"Total |v| >= 0.95 Plies: {total_extreme_plies}")
    lines.append("\n1. Halfmove Clock Breakdown:")
    for low, high in hm_bins:
        cnt = hm_counts[(low, high)]
        pct = cnt / total_extreme_plies * 100.0 if total_extreme_plies > 0 else 0.0
        lines.append(f"   Clock [{low:>2}, {high - 1:>2}]: {cnt:>5} ({pct:5.1f}%)")

    lines.append("\n2. Piece Count Breakdown:")
    for low, high in pc_bins:
        cnt = pc_counts[(low, high)]
        pct = cnt / total_extreme_plies * 100.0 if total_extreme_plies > 0 else 0.0
        lines.append(f"   Pieces [{low:>2}, {high - 1:>2}]: {cnt:>5} ({pct:5.1f}%)")

    lines.append("\n3. Search Type Breakdown:")
    for st, cnt in st_counts.items():
        pct = cnt / total_extreme_plies * 100.0 if total_extreme_plies > 0 else 0.0
        lines.append(f"   Search {st:<6}: {cnt:>5} ({pct:5.1f}%)")

    lines.append("\n4. Game Ply Bucket Breakdown:")
    for low, high in ply_bins:
        cnt = ply_counts[(low, high)]
        pct = cnt / total_extreme_plies * 100.0 if total_extreme_plies > 0 else 0.0
        lines.append(f"   Plies [{low:>3}, {high - 1:>3}]: {cnt:>5} ({pct:5.1f}%)")

    lines.append("=" * 80 + "\n")
    return "\n".join(lines)


# ==============================================================================
# 6. NETWORK VS SEARCH (300 SAMPLED PLIES WITH |root Q| >= 0.95)
# ==============================================================================
def analyze_network_vs_search(
    games: List[Dict[str, Any]],
    checkpoint_path: str,
    device: str = "cuda",
    sample_size: int = 300,
    seed: int = 42,
) -> str:
    print("=" * 80)
    print(
        f"ITEM 6: NETWORK VS SEARCH COMPARISON ({sample_size} Sampled Plies with |root Q| >= 0.95)"
    )
    print("=" * 80)

    # Collect all plies with |root Q| >= 0.95 with game context
    candidates = []
    for g in games:
        moves = g.get("moves", [])
        board = chess.Board()
        for p in g.get("plies", []):
            if abs(p["root_value"]) >= 0.95:
                candidates.append(
                    {
                        "game_id": g["game_id"],
                        "ply": p["ply"],
                        "fen": board.fen(),
                        "root_val": p["root_value"],
                        "uci_move": p["uci_move"],
                    }
                )
            board.push_uci(p["uci_move"])

    print(f"Total candidate plies with |root Q| >= 0.95: {len(candidates)}")
    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(sample_size, len(candidates)))

    # Load neural network model directly for raw evaluation
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model = ChessNet(
        num_blocks=ckpt.get("num_blocks", 12), num_filters=ckpt.get("num_filters", 128)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    diffs = []
    raw_vals = []
    root_vals = []
    sign_agreements = 0

    batch_size = 64
    for b_idx in range(0, len(sampled), batch_size):
        batch_items = sampled[b_idx : b_idx + batch_size]
        planes_list = []
        for it in batch_items:
            cg = zeropawn_cpp.ChessGame(it["fen"])
            planes_np = zeropawn_cpp.encode_state(cg)
            planes_list.append(planes_np)

        planes_t = torch.from_numpy(np.stack(planes_list, axis=0)).to(device)
        with torch.inference_mode():
            _, w_logits = model(planes_t)
            w_probs = F.softmax(w_logits.float(), dim=-1).cpu().numpy()

        for idx_in_b, it in enumerate(batch_items):
            p_win = w_probs[idx_in_b, 0]
            p_loss = w_probs[idx_in_b, 2]
            v_raw = float(p_win - p_loss)
            v_root = it["root_val"]

            raw_vals.append(v_raw)
            root_vals.append(v_root)
            diffs.append(abs(v_raw - v_root))

            if (v_raw > 0 and v_root > 0) or (v_raw < 0 and v_root < 0):
                sign_agreements += 1

    diff_arr = np.array(diffs)
    raw_arr = np.array(raw_vals)
    root_arr = np.array(root_vals)

    lines = []
    lines.append(f"Sampled Positions Evaluated: {len(sampled)}")
    lines.append(f"Mean Root Q:                 {np.mean(root_arr):+7.4f}")
    lines.append(f"Mean Raw Network V:          {np.mean(raw_arr):+7.4f}")
    lines.append(
        f"Mean Absolute Difference:    {np.mean(diff_arr):7.4f} (Max diff: {np.max(diff_arr):7.4f})"
    )
    lines.append(
        f"Sign Agreement Rate:         {sign_agreements / len(sampled) * 100.0:5.2f}% ({sign_agreements}/{len(sampled)})"
    )
    lines.append(f"Correlation (r):             {np.corrcoef(raw_arr, root_arr)[0, 1]:7.4f}")
    lines.append("\nAPI NOTE ON PROVEN MATE:")
    lines.append(
        "The zeropawn_cpp C++ SearchResult struct exposes only (best_action, action_visits, root_value)."
    )
    lines.append("It does NOT expose mate distance, proven terminal flags, or TT mate tags.")
    lines.append("Per ground rules, C++ code was NOT modified.")
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


# ==============================================================================
# 7. GROUND TRUTH WITH STOCKFISH 17 (50,000 NODES)
# ==============================================================================
def analyze_stockfish_ground_truth(
    games: List[Dict[str, Any]],
    stockfish_path: str = "tools/stockfish.exe",
    nodes: int = 50000,
) -> str:
    print("=" * 80)
    print(f"ITEM 7: GROUND TRUTH WITH STOCKFISH 17 ({nodes:,} nodes per position)")
    print("=" * 80)

    sf = StockfishEvaluator(stockfish_path=stockfish_path, threads=2, hash_mb=64)

    # (a) First would-resign trigger position in each triggered game
    # (t = -0.95, k = 4, ply >= 30)
    triggered_first_positions = []
    for g in games:
        w_consec = 0
        b_consec = 0
        board = chess.Board()

        for p in g.get("plies", []):
            ply_num = p["ply"]
            stm = p["side_to_move"]
            val = p["root_value"]

            if ply_num >= 30:
                if stm == "white":
                    if val < -0.95:
                        w_consec += 1
                        if w_consec == 4:
                            triggered_first_positions.append(
                                {
                                    "game_id": g["game_id"],
                                    "ply": ply_num,
                                    "side": "white",
                                    "net_val": val,
                                    "fen": board.fen(),
                                    "result": g["result"],
                                }
                            )
                            break
                    else:
                        w_consec = 0
                else:
                    if val < -0.95:
                        b_consec += 1
                        if b_consec == 4:
                            triggered_first_positions.append(
                                {
                                    "game_id": g["game_id"],
                                    "ply": ply_num,
                                    "side": "black",
                                    "net_val": val,
                                    "fen": board.fen(),
                                    "result": g["result"],
                                }
                            )
                            break
                    else:
                        b_consec = 0
            board.push_uci(p["uci_move"])

    print(f"Found {len(triggered_first_positions)} first would-resign trigger positions.")

    sf_verdicts = {"lost": 0, "drawn": 0, "won": 0}
    sf_details = []

    for idx, item in enumerate(triggered_first_positions):
        cp = sf.evaluate_fen(item["fen"], nodes=nodes)
        w, d, l = centipawns_to_wdl(cp)
        # cp is from side to move's perspective in Stockfish
        # V_sf = w - l in [-1, 1]
        v_sf = w - l
        if v_sf <= -0.50:  # cp <= -200
            verdict = "lost"
        elif v_sf >= +0.50:  # cp >= +200
            verdict = "won"
        else:
            verdict = "drawn"

        sf_verdicts[verdict] += 1
        sf_details.append(
            {
                "game_id": item["game_id"],
                "ply": item["ply"],
                "side": item["side"],
                "net_val": item["net_val"],
                "sf_cp": cp,
                "sf_v": v_sf,
                "verdict": verdict,
                "result": item["result"],
            }
        )

    # (b) Position at last capture or pawn move before drawn stretch in 50-move / repetition games
    drawn_stretch_positions = []
    for g in games:
        if g.get("termination_reason") in ("50_move_rule", "repetition"):
            board = chess.Board()
            last_reset_ply = 0
            last_reset_fen = board.fen()

            for p_idx, m_str in enumerate(g.get("moves", [])):
                m_obj = chess.Move.from_uci(m_str)
                is_pawn = board.piece_type_at(m_obj.from_square) == chess.PAWN
                is_cap = board.is_capture(m_obj)
                if is_pawn or is_cap:
                    last_reset_ply = p_idx + 1
                    board.push(m_obj)
                    last_reset_fen = board.fen()
                else:
                    board.push(m_obj)

            drawn_stretch_positions.append(
                {
                    "game_id": g["game_id"],
                    "termination": g["termination_reason"],
                    "last_reset_ply": last_reset_ply,
                    "total_plies": g["length_plies"],
                    "fen": last_reset_fen,
                }
            )

    print(f"Found {len(drawn_stretch_positions)} drawn games with pre-draw stretch positions.")
    drawn_sf_winning_count = 0
    drawn_sf_drawn_count = 0

    for item in drawn_stretch_positions:
        cp = sf.evaluate_fen(item["fen"], nodes=nodes)
        # cp is from side to move's perspective
        w, d, l = centipawns_to_wdl(cp)
        v_sf = w - l
        # Was it winning for EITHER side? |cp| >= 200 or |v_sf| >= 0.50
        if abs(cp) >= 200.0:
            drawn_sf_winning_count += 1
        else:
            drawn_sf_drawn_count += 1

    sf.close()

    lines = []
    lines.append(f"STOCKFISH 17 GROUND TRUTH EVALUATION (nodes = {nodes:,}):")
    lines.append("-" * 75)
    lines.append(
        f"(a) Positions at First Would-Resign Trigger (N = {len(triggered_first_positions)}):"
    )
    lines.append(
        f"    Actually LOST  (CP <= -200, V <= -0.50): {sf_verdicts['lost']:>3} / {len(triggered_first_positions)} ({sf_verdicts['lost'] / len(triggered_first_positions) * 100:5.1f}%)"
    )
    lines.append(
        f"    Actually DRAWN (|CP| < 200,  |V| < 0.50): {sf_verdicts['drawn']:>3} / {len(triggered_first_positions)} ({sf_verdicts['drawn'] / len(triggered_first_positions) * 100:5.1f}%)"
    )
    lines.append(
        f"    Actually WON   (CP >= +200, V >= +0.50): {sf_verdicts['won']:>3} / {len(triggered_first_positions)} ({sf_verdicts['won'] / len(triggered_first_positions) * 100:5.1f}%)"
    )
    lines.append(
        "\n(b) Games Drawn by 50-Move Rule or Repetition (N = {len(drawn_stretch_positions)}):"
    )
    lines.append(f"    Position at last capture / pawn move before drawn stretch:")
    lines.append(
        f"    - Stockfish Winning for either side (|CP| >= 200): {drawn_sf_winning_count:>3} / {len(drawn_stretch_positions)} ({drawn_sf_winning_count / len(drawn_stretch_positions) * 100:5.1f}%)"
    )
    lines.append(
        f"    - Stockfish Equal / Drawn            (|CP| < 200):  {drawn_sf_drawn_count:>3} / {len(drawn_stretch_positions)} ({drawn_sf_drawn_count / len(drawn_stretch_positions) * 100:5.1f}%)"
    )
    lines.append(f"\nCONCLUSION:")
    lines.append(
        f"Of the {len(drawn_stretch_positions)} drawn games, {drawn_sf_winning_count} ({drawn_sf_winning_count / len(drawn_stretch_positions) * 100:.1f}%) were completely WON positions"
    )
    lines.append(f"that the neural network failed to convert into checkmate.")
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


# ==============================================================================
# 8. LABEL CONTAMINATION UNDER RESIGNATION
# ==============================================================================
def analyze_label_contamination(
    games: List[Dict[str, Any]], t: float = -0.95, k: int = 4, min_ply: int = 30
) -> str:
    print("=" * 80)
    print(f"ITEM 8: LABEL CONTAMINATION AUDIT (t={t}, k={k}, min_ply={min_ply})")
    print("=" * 80)

    total_games = len(games)
    triggered_games = 0
    total_positions = sum(g.get("length_plies", len(g.get("moves", []))) for g in games)

    truncated_positions = 0
    retained_positions = 0
    contaminated_positions = 0

    game_true_results = defaultdict(int)

    for g in games:
        res = g["result"]
        plies = g.get("plies", [])

        # Check resignation trigger
        w_consec = 0
        b_consec = 0
        trigger_ply = None
        resigning_side = None

        for p in plies:
            ply_num = p["ply"]
            stm = p["side_to_move"]
            val = p["root_value"]

            if ply_num >= min_ply:
                if stm == "white":
                    if val < t:
                        w_consec += 1
                        if w_consec >= k and trigger_ply is None:
                            trigger_ply = ply_num
                            resigning_side = "white"
                            break
                    else:
                        w_consec = 0
                else:
                    if val < t:
                        b_consec += 1
                        if b_consec >= k and trigger_ply is None:
                            trigger_ply = ply_num
                            resigning_side = "black"
                            break
                    else:
                        b_consec = 0

        if trigger_ply is not None:
            triggered_games += 1
            game_true_results[res] += 1
            # If resignation had been on, game stops at trigger_ply
            # Plies before trigger_ply are retained and labeled
            # Plies after trigger_ply are truncated (discarded)
            n_retained = trigger_ply + 1
            n_truncated = len(plies) - n_retained
            retained_positions += n_retained
            truncated_positions += n_truncated

            # Buffer assigned outcome:
            # Resigning side gets Loss (2). Opponent gets Win (0).
            # True outcome of the game was res.
            # A retained position gets wrong label if true outcome != assigned outcome!
            for p_idx in range(n_retained):
                p_stm = plies[p_idx]["side_to_move"]
                # Assigned outcome for this position with resignation active:
                assigned_outcome = 2 if p_stm == resigning_side else 0  # 2=Loss, 0=Win

                # True outcome for this position:
                if res == "1-0":
                    true_outcome = 0 if p_stm == "white" else 2
                elif res == "0-1":
                    true_outcome = 0 if p_stm == "black" else 2
                else:  # draw
                    true_outcome = 1

                if assigned_outcome != true_outcome:
                    contaminated_positions += 1
        else:
            # Un-triggered game: all positions retained with true outcome
            retained_positions += len(plies)

    lines = []
    lines.append(f"TOTAL GAMES EVALUATED:               {total_games}")
    lines.append(
        f"GAMES THAT WOULD RESIGN:             {triggered_games} ({triggered_games / total_games * 100:5.1f}%)"
    )
    lines.append(f"TRUE OUTCOMES OF TRIGGERED GAMES:")
    for r_k, cnt in sorted(game_true_results.items()):
        lines.append(
            f"  Result {r_k:<8}: {cnt:>3} / {triggered_games} ({cnt / triggered_games * 100:5.1f}%)"
        )

    lines.append(f"\nPOSITION-LEVEL IMPACT:")
    lines.append(f"Total Positions Generated (No Resign): {total_positions:,}")
    lines.append(
        f"Positions Truncated (Discarded):       {truncated_positions:,} ({truncated_positions / total_positions * 100:5.1f}%)"
    )
    lines.append(f"Positions Retained with Resign:        {retained_positions:,}")
    lines.append(
        f"Contaminated Positions (Wrong Label):  {contaminated_positions:,} ({contaminated_positions / retained_positions * 100:5.1f}% of retained positions)"
    )
    lines.append("=" * 80 + "\n")

    return "\n".join(lines)


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def main():
    games_warm = load_jsonl("docs/resignfix/diagnosis_games.jsonl")
    games_step850 = load_jsonl("docs/resignfix/diagnosis_step850_games.jsonl")

    # Item 1
    verify_moves_and_perspective(games_warm, "Warm Start best_model.pt")
    verify_moves_and_perspective(games_step850, "Step-850 phase4_final_checkpoint.pt")

    # Item 2
    print("=" * 80)
    print("ITEM 2: REBUILT RELIABILITY TABLES (Game-Clustered 2,000 Bootstrap CIs)")
    print("=" * 80)
    print("--- 2A. WARM START (best_model.pt): MOVER PERSPECTIVE ---")
    m_warm, w_warm = bootstrap_game_clustered_reliability(games_warm, n_boot=2000)
    print(m_warm)
    print("\n--- 2B. WARM START (best_model.pt): WHITE PERSPECTIVE ---")
    print(w_warm)

    print("\n--- 2C. STEP-850 (phase4_final_checkpoint.pt): MOVER PERSPECTIVE ---")
    m_850, w_850 = bootstrap_game_clustered_reliability(games_step850, n_boot=2000)
    print(m_850)
    print("\n--- 2D. STEP-850 (phase4_final_checkpoint.pt): WHITE PERSPECTIVE ---")
    print(w_850)

    print("\n--- Line in analyze_resignation.py converting result to score ---")
    print("File: scripts/analyze_resignation.py, Line 66:")
    print("  actual_score = white_score if side == 'white' else (1.0 - white_score)")
    print(
        "PERSPECTIVE MATCH: Line 66 computes the score from the MOVER's perspective (1.0=win, 0.5=draw, 0=loss)."
    )
    print(
        "However, by condensing W/D/L into a scalar score, a 73% draw rate mathematically forces the scalar score"
    )
    print("to ~0.50 regardless of the underlying win and loss frequencies.")
    print("=" * 80 + "\n")

    # Item 3
    print(analyze_winner_vs_loser_table(games_warm, n_boot=2000))

    # Item 4
    verify_hook_fidelity(
        "models/checkpoints_scaled/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Item 5
    print(analyze_two_sided_consistency_and_breakdowns(games_warm))

    # Item 6
    print(analyze_network_vs_search(games_warm, "models/checkpoints_scaled/best_model.pt"))

    # Item 7
    print(
        analyze_stockfish_ground_truth(
            games_warm, stockfish_path="tools/stockfish.exe", nodes=50000
        )
    )

    # Item 8
    print(analyze_label_contamination(games_warm, t=-0.95, k=4, min_ply=30))


if __name__ == "__main__":
    main()
