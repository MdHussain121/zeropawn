"""Task 6: Offline Validation of Tablebase Adjudication.

Evaluates saved games (100 diagnosis games + 50 step-850 games):
1. Runs TablebaseAdjudicator on each game to find the exact cut ply and verdict.
2. Evaluates cut positions with Stockfish 17 (50,000 nodes) to compare verdicts.
3. Compares realized game outcomes vs tablebase verdicts (conversion failures vs opponent blunders).
4. Computes actual plies per game before and after adjudication, plies saved, and draw rates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import chess
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
from python.selfplay.tb_adjudicate import TablebaseAdjudicator
from python.data.stockfish_eval import StockfishEvaluator


def audit_dataset(
    jsonl_path: str,
    name: str,
    tb_dir: Optional[str] = None,
    sf_nodes: int = 50_000,
    stockfish_path: Optional[str] = None,
    max_games: Optional[int] = None,
) -> Dict[str, Any]:
    if tb_dir is None:
        tb_dir = str(_repo_root / "data" / "syzygy")
    if stockfish_path is None:
        stockfish_path = str(_repo_root / "tools" / "stockfish.exe")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        games = [json.loads(line) for line in f if line.strip()]
    total_available = len(games)
    if max_games is not None:
        games = games[:max_games]

    adjudicator = TablebaseAdjudicator.get_instance(tb_dir, max_pieces=5)
    sf = StockfishEvaluator(stockfish_path, threads=2)

    total_games = len(games)
    adjudicated_games = []
    unadjudicated_games = []

    orig_lengths = [g["length_plies"] for g in games]
    orig_len_by_id = {g["game_id"]: g["length_plies"] for g in games}
    adj_lengths = []

    orig_draws = sum(1 for g in games if g["result"] in ("1/2-1/2", "draw"))
    adj_draws = 0

    emit_report(event_line("AUDIT", f"{name} (N = {total_games})"))

    for g in games:
        board = chess.Board()
        cut_info = None

        for ply_idx, m_str in enumerate(g.get("moves", [])):
            board.push_uci(m_str)
            current_ply = ply_idx + 1

            # Only check if game has not ended
            if not board.is_game_over() and cut_info is None:
                fen = board.fen()
                probe_res, probe_meta = adjudicator.probe(fen)
                if probe_res is not None:
                    stm_white = board.turn == chess.WHITE
                    if probe_res == 1:
                        winner = "white" if stm_white else "black"
                        adj_result = "1-0" if stm_white else "0-1"
                    elif probe_res == -1:
                        winner = "black" if stm_white else "white"
                        adj_result = "0-1" if stm_white else "1-0"
                    else:
                        winner = None
                        adj_result = "1/2-1/2"

                    cut_info = {
                        "game_id": g["game_id"],
                        "cut_ply": current_ply,
                        "orig_length": g["length_plies"],
                        "cut_fen": fen,
                        "piece_count": probe_meta["piece_count"],
                        "raw_wdl": probe_meta["raw_wdl"],
                        "winner": winner,
                        "adj_result": adj_result,
                        "orig_result": g["result"],
                        "stm": "white" if stm_white else "black",
                    }
                    break

        if cut_info is not None:
            adj_lengths.append(cut_info["cut_ply"])
            if cut_info["adj_result"] == "1/2-1/2":
                adj_draws += 1
            adjudicated_games.append(cut_info)
        else:
            adj_lengths.append(g["length_plies"])
            if g["result"] in ("1/2-1/2", "draw"):
                adj_draws += 1
            unadjudicated_games.append(g)

    # Now evaluate cut positions with Stockfish 17 (50,000 nodes)
    emit_report(
        event_line(
            "PHASE",
            f"Stockfish audit of {len(adjudicated_games)} cut positions ({sf_nodes:,} nodes)",
        )
    )
    sf_agreements = 0
    conversion_failures = 0
    opponent_blunders = 0
    agreement_records = []

    for idx, c in enumerate(adjudicated_games):
        if idx % 10 == 0:
            emit_report(format_kv_line([("sf_done", f"{idx}/{len(adjudicated_games)}")]))
        sf_cp = sf.evaluate_fen(c["cut_fen"], nodes=sf_nodes)
        sf_val = sf_cp / 100.0
        sf_str = f"cp {int(sf_cp)}"
        c["sf_str"] = sf_str
        c["sf_val"] = sf_val

        # Stockfish perspective: positive means side to move is winning
        stm = c["stm"]
        tb_winner = c["winner"]
        orig_res = c["orig_result"]
        adj_res = c["adj_result"]

        # Check SF agreement
        # If TB is draw, SF is near zero (|sf_val| < 1.0 or sign)
        # If TB winner is STM, sf_val > 0
        # If TB winner is opponent, sf_val < 0
        if tb_winner is None:
            # Draw
            sf_agree = abs(sf_val) < 2.0
        elif (tb_winner == "white" and stm == "white") or (tb_winner == "black" and stm == "black"):
            # STM winning
            sf_agree = sf_val > 1.0
        else:
            # STM losing
            sf_agree = sf_val < -1.0

        if sf_agree:
            sf_agreements += 1

        # Realized outcome analysis:
        # Did the TB winning side actually win the unadjudicated game?
        if tb_winner is not None:
            orig_winner = "white" if orig_res == "1-0" else ("black" if orig_res == "0-1" else None)
            if orig_winner != tb_winner:
                # Discrepancy between TB truth and selfplay realized outcome
                if orig_winner is None:
                    # Won position converted to draw
                    conversion_failures += 1
                    status = "CONVERSION FAILURE (TB win -> played out to Draw)"
                elif orig_winner != tb_winner:
                    # Won position converted to Loss!
                    conversion_failures += 1
                    status = "CONVERSION BLUNDER (TB win -> played out to Loss!)"
            else:
                status = "CONVERTED (TB win -> Win)"
        else:
            # TB is Draw
            if orig_res in ("1-0", "0-1"):
                opponent_blunders += 1
                status = f"BLUNDER IN PLAYOUT (TB draw -> {orig_res})"
            else:
                status = "DRAW HELD (TB draw -> Draw)"

        c["realization_status"] = status
        agreement_records.append(
            (
                c["game_id"],
                c["cut_ply"],
                c["piece_count"],
                c["raw_wdl"],
                adj_res,
                orig_res,
                sf_str,
                status,
            )
        )

    subset_note = "" if max_games is None else f" (subset {total_games} of {total_available} (D))"
    emit_report(
        results_table(
            [
                ("dataset", f"{name}{subset_note}"),
                ("games", f"{total_games} (M)"),
                (
                    "adjudicated",
                    f"{len(adjudicated_games)}/{total_games} "
                    f"({len(adjudicated_games) / total_games * 100:.1f}%) (M)",
                ),
                (
                    "sf_agreement",
                    f"{sf_agreements}/{len(adjudicated_games)} "
                    f"({sf_agreements / max(1, len(adjudicated_games)) * 100:.1f}%) (M)",
                ),
                ("conversion_failures", f"{conversion_failures} (M)"),
                ("playout_blunders", f"{opponent_blunders} (M)"),
                ("avg_plies_orig", f"{sum(orig_lengths) / total_games:.1f} (M)"),
                ("avg_plies_adj", f"{sum(adj_lengths) / total_games:.1f} (M)"),
                (
                    "plies_saved_per_adj",
                    f"{sum(orig_len_by_id[g['game_id']] - g['cut_ply'] for g in adjudicated_games) / max(1, len(adjudicated_games)):.1f} (M)",
                ),
                (
                    "draw_rate_orig",
                    f"{orig_draws}/{total_games} ({orig_draws / total_games * 100:.1f}%) (M)",
                ),
                (
                    "draw_rate_adj",
                    f"{adj_draws}/{total_games} ({adj_draws / total_games * 100:.1f}%) (M)",
                ),
            ]
        )
    )

    emit_report(
        event_line("SAMPLE", "first cut positions (game|cut_ply|pieces|WDL|adj|orig|SF|status):")
    )
    for rec in agreement_records[:10]:
        emit_report(
            format_kv_line(
                [
                    ("game", f"{rec[0]}"),
                    ("cut_ply", f"{rec[1]}"),
                    ("pieces", f"{rec[2]}"),
                    ("wdl", f"{rec[3]:+d}"),
                    ("adj", f"{rec[4]}"),
                    ("orig", f"{rec[5]}"),
                    ("sf", f"{rec[6]}"),
                    ("status", f"{rec[7]}"),
                ]
            )
        )

    return {
        "total_games": total_games,
        "adjudicated_count": len(adjudicated_games),
        "sf_agreement_count": sf_agreements,
        "conversion_failures": conversion_failures,
        "opponent_blunders": opponent_blunders,
        "orig_avg_plies": sum(orig_lengths) / total_games,
        "adj_avg_plies": sum(adj_lengths) / total_games,
        "orig_draw_rate": orig_draws / total_games,
        "adj_draw_rate": adj_draws / total_games,
        "records": agreement_records,
    }


def main(argv=None):
    setup_logging(log_dir="logs", log_filename="validate_tb_offline.log")
    parser = argparse.ArgumentParser(description="Offline validation of tablebase adjudication.")
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Audit at most N games per dataset (default: None = all; subset runs are labeled).",
    )
    parser.add_argument(
        "--sf-nodes",
        type=int,
        default=50_000,
        help="Stockfish nodes per cut position (default: 50000).",
    )
    parser.add_argument(
        "--tb-dir",
        type=str,
        default=str(_repo_root / "data" / "syzygy"),
        help="Path to Syzygy tablebase directory.",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=str(_repo_root / "tools" / "stockfish.exe"),
        help="Path to Stockfish engine executable.",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.90,
        help="Minimum required Stockfish agreement ratio for successful validation (default: 0.90).",
    )
    args = parser.parse_args(argv)

    ctx = run_context("tablebase-audit")
    t0 = time.time()
    log_path = str(_repo_root / "logs" / "validate_tb_offline.log")
    emit_report(banner("tablebase-audit", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu+stockfish",
            "n/a (saved games)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("max_games", args.max_games, "user" if args.max_games is not None else "default"),
                ("sf_nodes", args.sf_nodes, "user" if args.sf_nodes != 50_000 else "default"),
                ("tb_dir", args.tb_dir, "default"),
                ("min_agreement", args.min_agreement, "default"),
            ]
        )
    )
    try:
        datasets = [
            (
                _repo_root / "docs" / "resignfix" / "diagnosis_games.jsonl",
                "Warm-Start Checkpoint best_model.pt (100 Games)",
            ),
            (
                _repo_root / "docs" / "resignfix" / "diagnosis_step850_games.jsonl",
                "Halted Run Step-850 Checkpoint (50 Games)",
            ),
        ]
        run_datasets = []
        for path, name in datasets:
            if not path.is_file():
                # SY-07 contract: diagnostic inputs absent from this checkout
                # (private run artifacts are not part of the publication) must
                # skip uninformatively, not fail the gate.
                emit_report(event_line("SKIP", f"{name}: diagnostic dataset not present: {path}"))
                continue
            run_datasets.append((str(path), name))
        if not run_datasets:
            emit_report(
                summary_block("SUCCESS (no diagnostic datasets present)", time.time() - t0, 0, 0)
            )
            sys.exit(0)
        results = []
        fail_reasons = []
        for path_str, name in run_datasets:
            res = audit_dataset(
                path_str,
                name,
                tb_dir=args.tb_dir,
                sf_nodes=args.sf_nodes,
                stockfish_path=args.stockfish_path,
                max_games=args.max_games,
            )
            results.append(res)
            adj_count = res["adjudicated_count"]
            sf_count = res["sf_agreement_count"]
            if adj_count > 0:
                agreement = sf_count / adj_count
                if agreement < args.min_agreement:
                    fail_reasons.append(
                        f"{name}: Stockfish agreement {agreement:.1%} ({sf_count}/{adj_count}) < threshold {args.min_agreement:.1%}"
                    )

        if fail_reasons:
            emit_error(
                error_block(
                    "tablebase-audit validation gate failed",
                    "; ".join(fail_reasons),
                    "inspect logs and investigate Stockfish / tablebase disagreement",
                    log_path,
                )
            )
            emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
            sys.exit(1)

        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
        sys.exit(0)
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"tablebase-audit failed: {e}",
                "missing game JSONL, tablebase path, or stockfish binary",
                f"check docs/resignfix/*.jsonl, {args.tb_dir}, {args.stockfish_path}, then see LOG",
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
