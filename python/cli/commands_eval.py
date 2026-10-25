"""Evaluation subcommands for ZeroPawn: baseline, heldout, reliability, and connect4."""

from __future__ import annotations

import argparse

from python.cli.models import get_default_model_path
from python.cli.runner import execute_script_with_monitoring


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluation commands: Stockfish Elo baseline, held-out accuracy, and reliability.",
        description="Run evaluation benchmarks against Stockfish levels or validation datasets.",
    )
    sub = eval_parser.add_subparsers(dest="eval_subcommand", help="Evaluation actions")

    # 1. Baseline
    p_bl = sub.add_parser(
        "baseline", help="Run calibrated baseline vs Stockfish Elo level (1320, 1500, 1700)."
    )
    p_bl.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint path")
    p_bl.add_argument(
        "--elo", type=int, default=1500, choices=[1320, 1500, 1700], help="Stockfish UCI_Elo level"
    )
    p_bl.add_argument(
        "--games", "-n", type=int, default=None, help="Total games to play (openings = games / 2)"
    )
    p_bl.add_argument("--openings", type=int, default=50, help="Number of openings (2 games each)")
    p_bl.add_argument("--sims", type=int, default=400, help="MCTS simulations per move")
    p_bl.add_argument(
        "--smoke", action="store_true", help="Quick smoke test (5 openings, 10 games)"
    )
    p_bl.add_argument("--workers", type=int, default=8, help="Worker count")
    p_bl.set_defaults(func=run_baseline)

    # 2. All Baseline
    p_abl = sub.add_parser(
        "all-baseline", help="Run full sequential chain across all Stockfish Elo levels."
    )
    p_abl.set_defaults(func=run_all_baseline)

    # 3. Heldout
    p_ho = sub.add_parser(
        "heldout", help="Evaluate policy top-1, top-3, and loss on held-out dataset."
    )
    p_ho.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Model checkpoint path (default: auto-detected)",
    )
    p_ho.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Validation dataset directory (default: auto-detected data/dataset_training/val or data/dataset_scaled/val)",
    )
    p_ho.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p_ho.set_defaults(func=run_heldout)

    # 4. Reliability
    p_rel = sub.add_parser("reliability", help="Evaluate value head calibration and reliability.")
    p_rel.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint path")
    p_rel.add_argument("--val-dir", type=str, default=None, help="Validation dataset directory")
    p_rel.add_argument(
        "--max-samples", type=int, default=None, help="Maximum validation samples to evaluate"
    )
    p_rel.set_defaults(func=run_reliability)

    # 5. Connect 4
    p_c4 = sub.add_parser(
        "connect4", help="Evaluate Connect 4 agent over games against random player."
    )
    p_c4.add_argument("--games", type=int, default=100, help="Number of games to evaluate")
    p_c4.set_defaults(func=run_connect4)


def run_baseline(args: argparse.Namespace) -> int:
    ckpt = args.checkpoint or str(
        get_default_model_path() or "models/checkpoints_scaled/best_model.pt"
    )
    openings = (
        max(1, args.games // 2) if getattr(args, "games", None) is not None else args.openings
    )
    cmd_args = [
        "--ckpt",
        ckpt,
        "--elo",
        str(args.elo),
        "--openings",
        str(openings),
        "--sims",
        str(args.sims),
        "--workers",
        str(args.workers),
    ]
    if args.smoke:
        cmd_args.append("--smoke")

    return execute_script_with_monitoring(
        script_rel_path="scripts/run_calibrated_baseline.py",
        args=cmd_args,
        op_name="eval baseline",
        description=f"Calibrated strength baseline vs Stockfish Elo {args.elo}",
        parameters=vars(args),
        model_path=ckpt,
    )


def run_all_baseline(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/run_all_baseline.py",
        args=[],
        op_name="eval all-baseline",
        description="Sequential chain across 6 Stockfish Elo levels",
    )


def run_heldout(args: argparse.Namespace) -> int:
    ckpt = args.checkpoint or str(
        get_default_model_path() or "models/checkpoints_scaled/best_model.pt"
    )
    cmd_args = [
        "--checkpoint",
        ckpt,
        "--batch-size",
        str(args.batch_size),
    ]
    if getattr(args, "val_dir", None):
        cmd_args.extend(["--val-dir", args.val_dir])
    return execute_script_with_monitoring(
        script_rel_path="scripts/eval_heldout.py",
        args=cmd_args,
        op_name="eval heldout",
        description="Held-out validation games evaluation",
        parameters=vars(args),
        model_path=ckpt,
    )


def run_reliability(args: argparse.Namespace) -> int:
    cmd_args = []
    if args.checkpoint:
        cmd_args.extend(["--checkpoint", args.checkpoint])
    if getattr(args, "val_dir", None):
        cmd_args.extend(["--val-dir", args.val_dir])
    if getattr(args, "max_samples", None):
        cmd_args.extend(["--max-samples", str(args.max_samples)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/eval_val_reliability.py",
        args=cmd_args,
        op_name="eval reliability",
        description="Value head calibration and reliability evaluation",
        parameters=vars(args),
    )


def run_connect4(args: argparse.Namespace) -> int:
    cmd_args = ["--games", str(args.games)]
    return execute_script_with_monitoring(
        script_rel_path="scripts/eval_connect4.py",
        args=cmd_args,
        op_name="eval connect4",
        description="Connect 4 agent evaluation against random player",
        parameters=vars(args),
    )
