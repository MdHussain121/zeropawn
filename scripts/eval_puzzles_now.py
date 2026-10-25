"""Standalone puzzle evaluation script to benchmark latest model weights."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
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
    short_hash,
    summary_block,
)
from python.model.chess_net import ChessNet
from python.eval.puzzles import PuzzleEvaluator, load_puzzles
from python.eval.baselines import get_baseline_for_model, set_baseline_for_model


def main() -> None:
    setup_logging(log_dir="logs", log_filename="eval_puzzles_now.log")
    ctx = run_context("puzzles-script")
    t0 = time.time()
    log_path = "logs/eval_puzzles_now.log"

    def _fail(what: str, likely: str, try_: str, code: int = 1) -> None:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 0, code))
        sys.exit(code)

    parser = argparse.ArgumentParser(
        description="Evaluate latest model on fixed 1,000 Lichess puzzles."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/checkpoints_rl/live_weights.pt",
        help="Path to model weights or checkpoint.",
    )
    parser.add_argument(
        "--puzzles",
        type=str,
        default="data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        help="Path to fixed puzzle dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--mcts-puzzles",
        type=int,
        default=50,
        help="Number of puzzles to test with MCTS (default: 50, 0 to skip).",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="MCTS simulations per puzzle (default: 50).",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Baseline solve rate (%%) to compare against (default: auto-detected for model or 35.6%%).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update the baseline for this model in puzzle_baselines.json with this result.",
    )
    args = parser.parse_args()

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    emit_report(banner("puzzles-script", ctx["start"] + " (local)"))
    emit_report(
        parameters_table(
            [
                (
                    "weights",
                    args.weights,
                    "user"
                    if args.weights != "models/checkpoints_rl/live_weights.pt"
                    else "default",
                ),
                (
                    "puzzles",
                    args.puzzles,
                    "user"
                    if args.puzzles != "data/puzzles/lichess_puzzles_1000_1200_1600.csv"
                    else "default",
                ),
                ("device", args.device, "user" if args.device != auto_device else "default"),
                (
                    "mcts_puzzles",
                    args.mcts_puzzles,
                    "user" if args.mcts_puzzles != 50 else "default",
                ),
                ("mcts_sims", args.mcts_sims, "user" if args.mcts_sims != 50 else "default"),
                ("baseline", args.baseline, "user" if args.baseline is not None else "default"),
                (
                    "update_baseline",
                    args.update_baseline,
                    "user" if args.update_baseline else "default",
                ),
            ]
        )
    )

    weights_path = Path(args.weights)
    if not weights_path.exists():
        fallback_candidates = [
            Path("models/checkpoints_rl/phase4_final_checkpoint.pt"),
            Path("models/checkpoints_scaled/best_model.pt"),
            Path("models/checkpoints_rl/latest_checkpoint.pt"),
        ]
        found = False
        for c in fallback_candidates:
            if c.is_file():
                emit_report(event_line("FALLBACK", f"'{weights_path}' not found, using '{c}'"))
                weights_path = c
                found = True
                break
        if not found:
            _fail(
                f"weights not found at {weights_path}",
                "no checkpoint at --weights or fallback paths",
                "train a model or pass an existing --weights, then see LOG",
            )

    emit_report(event_line("LOAD", f"loading weights from {weights_path}"))
    ckpt = torch.load(weights_path, map_location="cpu")
    step = ckpt.get("step", "unknown")
    num_blocks = ckpt.get("num_blocks", 12)
    num_filters = ckpt.get("num_filters", 128)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    model = ChessNet(num_blocks=num_blocks, num_filters=num_filters)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    ckpt_id = f"{weights_path.name} step={step} sha={short_hash(str(weights_path))} (M)"
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            ckpt_id,
            f"{args.device} (M)",
            "n/a (fixed puzzle order)",
        )
    )
    emit_report(
        format_kv_line([("net", f"{num_blocks}x{num_filters} (M)"), ("step", f"{step} (M)")])
    )

    puzzles = load_puzzles(args.puzzles)
    emit_report(format_kv_line([("puzzles", f"{len(puzzles)} (M)"), ("file", args.puzzles)]))

    evaluator = PuzzleEvaluator(model=model, device=args.device, puzzles=puzzles)

    model_baseline = (
        float(args.baseline) if args.baseline is not None else get_baseline_for_model(weights_path)
    )

    # 1. Policy Top-1 Evaluation across all puzzles
    emit_report(event_line("PHASE", f"policy top-1 across {len(puzzles)} puzzles"))
    t_phase = time.perf_counter()
    policy_res = evaluator.evaluate_policy(batch_size=128)
    t_policy = time.perf_counter() - t_phase

    solved = policy_res["solved"]
    total = policy_res["total"]
    rate = policy_res["solve_rate"] * 100.0
    if model_baseline is not None:
        diff = rate - model_baseline
        direction = "+" if diff >= 0 else ""
        vs_baseline_str = (
            f"; vs_baseline {direction}{diff:.2f}% (D, baseline {model_baseline:.2f}%)"
        )
    else:
        vs_baseline_str = "; vs_baseline N/A (no baseline recorded)"
    emit_report(
        results_table(
            [
                (
                    "policy_top1",
                    f"{solved}/{total} = {rate:.2f}% (M) in {t_policy:.2f}s (M){vs_baseline_str}",
                ),
            ]
        )
    )

    # 2. Fast MCTS Search Evaluation on subset
    mcts_rate = 0.0
    if args.mcts_puzzles > 0:
        num_mcts = min(args.mcts_puzzles, len(puzzles))
        emit_report(event_line("PHASE", f"MCTS {args.mcts_sims} sims across {num_mcts} puzzles"))
        t_phase = time.perf_counter()
        mcts_res = evaluator.evaluate_mcts(num_simulations=args.mcts_sims, max_puzzles=num_mcts)
        t_mcts = time.perf_counter() - t_phase
        mcts_rate = mcts_res["solve_rate"] * 100.0
        if model_baseline is not None:
            diff_mcts = mcts_rate - model_baseline
            dir_mcts = "+" if diff_mcts >= 0 else ""
            vs_mcts_str = (
                f"; vs_baseline {dir_mcts}{diff_mcts:.2f}% (D, baseline {model_baseline:.2f}%)"
            )
        else:
            vs_mcts_str = "; vs_baseline N/A (no baseline recorded)"
        emit_report(
            results_table(
                [
                    (
                        f"mcts_{args.mcts_sims}sims",
                        f"{mcts_res['solved']}/{mcts_res['total']} = {mcts_rate:.2f}% (M) "
                        f"in {t_mcts:.2f}s (M){vs_mcts_str}",
                    ),
                ]
            )
        )

    if args.update_baseline:
        set_baseline_for_model(
            weights_path,
            rate,
            mcts_rate=mcts_rate if args.mcts_puzzles > 0 else None,
            total_puzzles=total,
        )
        emit_report(
            event_line(
                "BASELINE",
                f"updated baseline for '{weights_path.name}' "
                f"to {rate:.2f}% in puzzle_baselines.json",
            )
        )
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))


if __name__ == "__main__":
    main()
