"""CLI wrapper to run ZeroPawn in standard Universal Chess Interface (UCI) mode."""

from __future__ import annotations

import argparse

from python.uci.uci_engine import UCIEngine
from python.cli.common import (
    resolve_path,
    print_error,
)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "uci",
        help="Run ZeroPawn in standard Universal Chess Interface (UCI) protocol mode.",
        description="Run ZeroPawn engine in UCI mode for GUIs (Arena, Lucas Chess, cutechess-cli) or stdin/stdout.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to neural network model checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device for neural inference (default: auto).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=400,
        help="Default MCTS simulation count per move (default: 400).",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="PUCT exploration constant (default: 2.0).",
    )
    parser.add_argument(
        "--reuse-tree",
        action="store_true",
        default=False,
        help="Retain and advance MCTS search tree between moves during play-time search (default: False).",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="1",
        help="Inference batch size ('auto' or integer 1-64; default: 1).",
    )
    parser.set_defaults(func=run_uci)


def run_uci(args: argparse.Namespace) -> int:
    ckpt_path = None
    if args.checkpoint:
        try:
            ckpt_path = resolve_path(args.checkpoint)
        except Exception as e:
            print_error(f"Invalid checkpoint path: {e}")
            return 1

    raw_bs = getattr(args, "batch_size", "1")
    bs_str = str(raw_bs).strip().lower()
    if bs_str != "auto":
        try:
            bs_val = int(bs_str)
            if not (1 <= bs_val <= 64):
                print_error("--batch-size must be in 1..64 or 'auto'.")
                return 1
        except ValueError:
            print_error("--batch-size must be an integer (1..64) or 'auto'.")
            return 1

    engine = UCIEngine(
        checkpoint_path=ckpt_path,
        device=args.device,
        default_sims=args.sims,
        c_puct=args.c_puct,
        reuse_tree=args.reuse_tree,
        batch_size=raw_bs,
    )
    engine.run_stdio_loop()
    return 0
