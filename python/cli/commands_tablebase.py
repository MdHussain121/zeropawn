"""Tablebase subcommands for ZeroPawn: download, offline validation, and feasibility."""

from __future__ import annotations

import argparse

from python.cli.runner import execute_script_with_monitoring


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    tb_parser = subparsers.add_parser(
        "tablebase",
        help="Syzygy tablebase tools: download, offline validation, and feasibility.",
        description="Endgame tablebase integration and verification utilities.",
    )
    sub = tb_parser.add_subparsers(dest="tb_subcommand", help="Tablebase actions")

    # 1. Download & Verify
    p_dl = sub.add_parser(
        "download",
        aliases=["verify", "check"],
        help="Download and verify 145 Syzygy 3-4-5 piece WDL (.rtbw) files.",
    )
    p_dl.set_defaults(func=run_tb_download)

    # 2. Validate
    p_val = sub.add_parser(
        "validate", help="Offline validation of tablebase adjudication vs Stockfish."
    )
    p_val.add_argument("--max-games", type=int, default=2, help="Maximum games to validate")
    p_val.set_defaults(func=run_tb_validate)

    # 3. Feasibility
    p_feas = sub.add_parser(
        "feasibility", help="Tablebase early adjudication feasibility analysis."
    )
    p_feas.add_argument("--max-games", type=int, default=2, help="Maximum games to analyze")
    p_feas.set_defaults(func=run_tb_feasibility)

    # 4. Build Dataset
    p_bds = sub.add_parser(
        "build-dataset", help="Generate offline Syzygy 3-4-5 piece endgame supervision dataset."
    )
    p_bds.add_argument(
        "--output-dir", type=str, default="data/dataset_syzygy", help="Output dataset directory"
    )
    p_bds.add_argument(
        "--count", type=int, default=500, help="Number of legal endgame positions to generate"
    )
    p_bds.add_argument(
        "--tb-dir", type=str, default="data/syzygy", help="Syzygy tablebase directory"
    )
    p_bds.set_defaults(func=run_tb_build_dataset)


def run_tb_download(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/download_syzygy.py",
        args=[],
        op_name="tablebase download",
        description="Download and verify Syzygy 3-4-5 tablebase files",
    )


def run_tb_validate(args: argparse.Namespace) -> int:
    cmd_args = ["--max-games", str(args.max_games)]
    return execute_script_with_monitoring(
        script_rel_path="scripts/validate_tb_offline.py",
        args=cmd_args,
        op_name="tablebase validate",
        description="Offline tablebase adjudication validation",
        parameters=vars(args),
    )


def run_tb_feasibility(args: argparse.Namespace) -> int:
    cmd_args = ["--max-games", str(args.max_games)]
    return execute_script_with_monitoring(
        script_rel_path="scripts/tb_feasibility.py",
        args=cmd_args,
        op_name="tablebase feasibility",
        description="Tablebase early adjudication feasibility analysis",
        parameters=vars(args),
    )


def run_tb_build_dataset(args: argparse.Namespace) -> int:
    from python.cli.common import print_banner, print_success, print_error
    from python.data.syzygy_dataset import SyzygySupervisionGenerator

    print_banner("Generating Syzygy 3-4-5 Piece Endgame Supervision Dataset")
    gen = SyzygySupervisionGenerator(tablebase_path=args.tb_dir)
    if gen.tb is None:
        print_error(f"Cannot open tablebase at {args.tb_dir}")
        return 1

    res = gen.build_endgame_dataset(output_dir=args.output_dir, synthetic_count=args.count)
    gen.close()
    print_success(
        f"Built Syzygy endgame dataset: {res['total_samples']} samples ({res['train_samples']} train, {res['val_samples']} val) in {res['output_dir']}"
    )
    return 0
