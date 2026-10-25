"""Standalone feasibility analysis for Syzygy tablebase adjudication.

NOTE: Analysis script only. Does NOT use python-chess in self-play worker or search.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import chess

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.cli.common import setup_logging
from python.common.progress import (
    banner,
    emit_error,
    emit_report,
    error_block,
    event_line,
    format_kv_line,
    header_block,
    parameters_table,
    results_table,
    run_context,
    summary_block,
)


def probe_syzygy_wdl(fen: str) -> Optional[str]:
    """Query Lichess Syzygy Tablebase API for FEN with <= 5 pieces."""
    url = f"https://tablebase.lichess.ovh/standard?fen={urllib.parse.quote(fen)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ZeroPawn-Audit/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("category")  # 'win', 'loss', 'draw', 'cursed-win', 'blessed-loss'
    except Exception:
        return None


def analyze_dataset(path: str, name: str, max_games: Optional[int] = None) -> None:
    with open(path, "r", encoding="utf-8") as f:
        games = [json.loads(line) for line in f if line.strip()]
    total_available = len(games)
    if max_games is not None:
        games = games[:max_games]

    emit_report(event_line("AUDIT", f"TABLEBASE FEASIBILITY: {name} ({len(games)} games)"))

    reaches_le5 = []
    end_piece_counts = []
    cursed_wins = 0
    blessed_losses = 0
    draw_to_win = 0
    draw_remains_draw = 0
    decisive_unchanged = 0
    plies_saved_list = []

    for g in games:
        board = chess.Board()
        first_ply = None
        first_fen = None
        first_stm = None

        for p_idx, m_str in enumerate(g.get("moves", [])):
            board.push_uci(m_str)
            pc = len(board.piece_map())
            if pc <= 5 and not bool(board.castling_rights) and first_ply is None:
                first_ply = p_idx + 1
                first_fen = board.fen()
                first_stm = "white" if board.turn == chess.WHITE else "black"

        end_piece_counts.append(len(board.piece_map()))

        if first_ply is not None:
            reaches_le5.append(
                {
                    "game_id": g["game_id"],
                    "ply": first_ply,
                    "fen": first_fen,
                    "stm": first_stm,
                    "orig_result": g["result"],
                    "total_plies": g["length_plies"],
                }
            )

    emit_report(
        format_kv_line(
            [
                (
                    "reaching_le5",
                    f"{len(reaches_le5)}/{len(games)} "
                    f"({len(reaches_le5) / len(games) * 100:.1f}%) (M)",
                )
            ]
        )
    )

    for idx, it in enumerate(reaches_le5):
        if idx % 10 == 0:
            emit_report(format_kv_line([("probed", f"{idx}/{len(reaches_le5)}")]))
        cat = probe_syzygy_wdl(it["fen"])
        it["cat"] = cat
        if cat in ("cursed-win", "blessed-loss"):
            if cat == "cursed-win":
                cursed_wins += 1
            else:
                blessed_losses += 1
            adj_res = "1/2-1/2"
        elif cat == "win":
            adj_res = "1-0" if it["stm"] == "white" else "0-1"
        elif cat == "loss":
            adj_res = "0-1" if it["stm"] == "white" else "1-0"
        else:
            adj_res = "1/2-1/2"

        orig = it["orig_result"]
        if orig in ("1/2-1/2", "draw"):
            if adj_res in ("1-0", "0-1"):
                draw_to_win += 1
            else:
                draw_remains_draw += 1
        else:
            decisive_unchanged += 1

        plies_saved_list.append(it["total_plies"] - it["ply"])

    orig_draws = sum(1 for g in games if g["result"] in ("1/2-1/2", "draw"))
    unreached_draws = orig_draws - sum(
        1 for it in reaches_le5 if it["orig_result"] in ("1/2-1/2", "draw")
    )
    adjudicated_draws = sum(
        1 for it in reaches_le5 if it.get("cat") in ("draw", "cursed-win", "blessed-loss", None)
    )
    new_draws = unreached_draws + adjudicated_draws
    new_decisive = len(games) - new_draws
    mean_saved = sum(plies_saved_list) / len(plies_saved_list) if plies_saved_list else 0.0

    subset_note = "" if max_games is None else f" (subset {len(games)} of {total_available} (D))"
    draw_conv = f"{draw_to_win / orig_draws * 100:.1f}%" if orig_draws else "n/a (no draws)"
    unreached_pct = f"{unreached_draws / orig_draws * 100:.1f}%" if orig_draws else "n/a (no draws)"
    emit_report(
        results_table(
            [
                ("dataset", f"{name}{subset_note}"),
                ("draw_to_win", f"{draw_to_win} games ({draw_conv} of draws) (M)"),
                ("cursed_blessed", f"cursed={cursed_wins} blessed={blessed_losses} (M)"),
                ("plies_saved_avg", f"{mean_saved:.1f} (M)"),
                (
                    "draw_rate_orig",
                    f"{orig_draws}/{len(games)} ({orig_draws / len(games) * 100:.1f}%) (M)",
                ),
                (
                    "draw_rate_new",
                    f"{new_draws}/{len(games)} ({new_draws / len(games) * 100:.1f}%) (M)",
                ),
                (
                    "decisive_new",
                    f"{new_decisive}/{len(games)} ({new_decisive / len(games) * 100:.1f}%) (M)",
                ),
                ("draws_never_le5", f"{unreached_draws}/{orig_draws} ({unreached_pct}) (M)"),
            ]
        )
    )
    for pc in sorted(set(end_piece_counts)):
        cnt = end_piece_counts.count(pc)
        emit_report(
            format_kv_line(
                [("end_pieces", pc), ("games", f"{cnt} ({cnt / len(games) * 100:.1f}%)")]
            )
        )


if __name__ == "__main__":
    setup_logging(log_dir="logs", log_filename="tb_feasibility.log")
    parser = argparse.ArgumentParser(description="Feasibility analysis for Syzygy adjudication.")
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Analyze at most N games per dataset (default: None = all; subset runs are labeled).",
    )
    fargs = parser.parse_args()

    ctx = run_context("tablebase-feasibility")
    t0 = time.time()
    log_path = "logs/tb_feasibility.log"
    emit_report(banner("tablebase-feasibility", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu+http",
            "n/a (saved games)",
        )
    )
    emit_report(
        parameters_table(
            [
                (
                    "max_games",
                    fargs.max_games,
                    "user" if fargs.max_games is not None else "default",
                ),
            ]
        )
    )
    try:
        analyze_dataset(
            "docs/resignfix/diagnosis_games.jsonl",
            "Warm-Start best_model.pt",
            max_games=fargs.max_games,
        )
        analyze_dataset(
            "docs/resignfix/diagnosis_step850_games.jsonl",
            "Step-850 phase4_final_checkpoint.pt",
            max_games=fargs.max_games,
        )
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"tablebase-feasibility failed: {e}",
                "missing game JSONL or no network for the Lichess API",
                "check docs/resignfix/*.jsonl and network, then see LOG",
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)
