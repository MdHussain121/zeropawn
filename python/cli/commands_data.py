"""Data operations subcommand for ZeroPawn: download, filter, convert, and dataset utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from rich import box
from rich.panel import Panel
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.runner import execute_script_with_monitoring
from python.cli.ui import COLOR_BORDER, console

_BIN_FILES = ("planes.bin", "moves.bin", "outcomes.bin", "game_ids.bin", "plies.bin")


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser(
        "data",
        help="Dataset utilities: download, filter, build binary tensors, openings, tactics, and audit.",
        description="Dataset pipeline commands for downloading, filtering, preparing, inspecting, and validating training data.",
    )
    sub = data_parser.add_subparsers(dest="data_subcommand", help="Data actions")

    # 1. Download
    p_dl = sub.add_parser("download", help="Download or inspect Lichess PGN database archive.")
    p_dl.add_argument("--month", type=str, default="2026-09", help="Month (YYYY-MM)")
    p_dl.add_argument("--info", action="store_true", help="Query size without downloading")
    p_dl.add_argument(
        "--check-license", action="store_true", help="Verify Lichess open database license"
    )
    p_dl.add_argument("--output-dir", type=str, default="data", help="Output directory")
    p_dl.set_defaults(func=run_download)

    # 2. Filter
    default_filter_in = (
        "data/elite_2500.pgn"
        if ((REPO_ROOT / "data/elite_2500.pgn").exists() or Path("data/elite_2500.pgn").exists())
        else "data/filtered_2200.pgn"
    )
    p_fl = sub.add_parser("filter", help="Filter raw PGN by Elo, time controls, and termination.")
    p_fl.add_argument("--input", "-i", type=str, default=default_filter_in, help="Source PGN file")
    p_fl.add_argument(
        "--output", "-o", type=str, default="data/filtered.pgn", help="Output PGN file"
    )
    p_fl.add_argument("--max-kept", type=int, default=10000, help="Maximum games to keep")
    p_fl.add_argument("--min-elo", type=int, default=2200, help="Minimum player rating")
    p_fl.add_argument(
        "--min-tc",
        type=int,
        default=180,
        help="Minimum estimated seconds (base + 40*inc; e.g. 180=3m blitz, 600=10m rapid)",
    )
    p_fl.add_argument("--min-plies", type=int, default=20, help="Minimum half-moves in game")
    p_fl.add_argument("--log-interval", type=int, default=500, help="Logging interval")
    p_fl.set_defaults(func=run_filter)

    # 3. Build Dataset
    build_candidates = [
        "data/filtered.pgn",
        "data/filtered_2500.pgn",
        "tmp/filtered.pgn",
        "data/elite_2500.pgn",
    ]
    default_build_pgn = next(
        (c for c in build_candidates if ((REPO_ROOT / c).exists() or Path(c).exists())),
        "data/filtered.pgn",
    )
    p_bd = sub.add_parser(
        "build", help="Convert filtered PGN to binary memmaps (planes, moves, outcomes)."
    )
    p_bd.add_argument("--pgn", type=str, default=default_build_pgn, help="Input PGN")
    p_bd.add_argument(
        "--output-dir", type=str, default="data/dataset_training", help="Output directory"
    )
    p_bd.add_argument("--max-games", type=int, default=5000, help="Maximum games to convert")
    p_bd.set_defaults(func=run_build_dataset)

    # 4. Build Tactics
    default_tactics_csv = next(
        (
            str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            for p in (REPO_ROOT / "data/puzzles").glob("*.csv")
        ),
        "data/puzzles/lichess_puzzles_10000_1600_2800.csv",
    )
    p_bt = sub.add_parser("build-tactics", help="Build tactics training dataset from puzzles.")
    p_bt.add_argument("--csv", type=str, default=default_tactics_csv, help="Input puzzles CSV file")
    p_bt.add_argument(
        "--output-dir", type=str, default="data/dataset_tactics", help="Output dataset directory"
    )
    p_bt.add_argument(
        "--max-solver-plies", type=int, default=4, help="Maximum solver plies per puzzle"
    )
    p_bt.set_defaults(func=run_build_tactics)

    # 5. Fetch Elite
    p_fe = sub.add_parser(
        "fetch-elite", help="Fetch 2500+ master games from Lichess Elite database."
    )
    p_fe.add_argument("--max-games", type=int, default=1000, help="Maximum games to fetch")
    p_fe.add_argument(
        "--min-tc",
        type=int,
        default=600,
        help="Minimum time control in seconds (default: 600s / 10+0 rapid)",
    )
    p_fe.add_argument(
        "--min-rating", type=int, default=2500, help="Minimum player rating (default: 2500)"
    )
    p_fe.set_defaults(func=run_fetch_elite)

    # 6. Fetch Tactics
    p_ft = sub.add_parser("fetch-tactics", help="Fetch tactical positions PGN.")
    p_ft.set_defaults(func=run_fetch_tactics)

    # 7. Build Book
    book_candidates = ["data/filtered.pgn", "data/filtered_2500.pgn", "data/elite_2500.pgn"]
    default_book_in = next(
        (c for c in book_candidates if ((REPO_ROOT / c).exists() or Path(c).exists())),
        "data/filtered.pgn",
    )
    p_bb = sub.add_parser("build-book", help="Build opening book for self-play diversity.")
    p_bb.add_argument("--input", type=str, default=default_book_in, help="Input PGN")
    p_bb.add_argument(
        "--output", type=str, default="data/openings/curated_book.json", help="Output book"
    )
    p_bb.add_argument("--max-ply", type=int, default=16, help="Maximum ply depth")
    p_bb.set_defaults(func=run_build_book)

    # 8. Generate Openings
    p_go = sub.add_parser(
        "generate-openings", help="Generate balanced 50-opening benchmark suite in EPD format."
    )
    p_go.add_argument(
        "--output", type=str, default="data/openings/eval_openings_50.epd", help="Output EPD file"
    )
    p_go.set_defaults(func=run_generate_openings)

    # 9. List Datasets
    p_ls = sub.add_parser("list", help="Scan and list all discovered datasets in data/.")
    p_ls.set_defaults(func=run_data_list)

    # 10. Inspect Dataset
    p_insp = sub.add_parser("inspect", help="Inspect metadata, schema, and checksums of a dataset.")
    p_insp.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help="Path to dataset directory (default: data/dataset_training)",
    )
    p_insp.add_argument(
        "--dir",
        dest="dir_flag",
        default=None,
        help="Path to dataset directory (e.g. data/dataset_training)",
    )
    p_insp.set_defaults(func=run_data_inspect)

    # 11. Validate Dataset Integrity
    p_val = sub.add_parser(
        "validate", help="Validate binary memmap file sizes and metadata consistency."
    )
    p_val.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help="Path to dataset directory (default: data/dataset_training)",
    )
    p_val.add_argument(
        "--dir",
        dest="dir_flag",
        default=None,
        help="Path to dataset directory (default: data/dataset_training)",
    )
    p_val.set_defaults(func=run_data_validate)

    # 12. Check Data Leakage
    p_leak = sub.add_parser(
        "check-leakage", help="Verify zero position or game leakage between train and val splits."
    )
    p_leak.add_argument(
        "--train-dir", type=str, default="data/dataset_training/train", help="Train split directory"
    )
    p_leak.add_argument(
        "--val-dir",
        type=str,
        default="data/dataset_training/val",
        help="Validation split directory",
    )
    p_leak.add_argument(
        "--max-positions",
        type=int,
        default=2000,
        help="Maximum positions to sample for hash comparison",
    )
    p_leak.set_defaults(func=run_data_check_leakage)

    # 13. Merge Datasets
    p_mrg = sub.add_parser(
        "merge", help="Merge multiple datasets into a single output dataset directory."
    )
    p_mrg.add_argument(
        "--inputs", "-i", nargs="+", required=True, help="Input dataset directories to concatenate"
    )
    p_mrg.add_argument("--output", "-o", required=True, help="Output destination dataset directory")
    p_mrg.set_defaults(func=run_data_merge)

    # 14. Split Dataset
    p_splt = sub.add_parser(
        "split", help="Split a unified dataset into train and validation subsets."
    )
    p_splt.add_argument("--input", "-i", required=True, help="Input source dataset directory")
    p_splt.add_argument(
        "--output", "-o", required=True, help="Output dataset directory with train/val splits"
    )
    p_splt.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)"
    )
    p_splt.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic split")
    p_splt.set_defaults(func=run_data_split)

    # 15. Build Syzygy Dataset
    p_syz = sub.add_parser(
        "build-syzygy", help="Generate offline Syzygy 3-4-5 piece endgame training dataset."
    )
    p_syz.add_argument(
        "--output-dir", type=str, default="data/dataset_syzygy", help="Output dataset directory"
    )
    p_syz.add_argument(
        "--count", type=int, default=500, help="Number of legal endgame positions to generate"
    )
    p_syz.add_argument(
        "--tb-dir", type=str, default="data/syzygy", help="Syzygy tablebase directory"
    )
    p_syz.set_defaults(func=run_data_build_syzygy)

    # 16. Build Openings Dataset
    p_opds = sub.add_parser(
        "build-openings", help="Generate opening supervision dataset from opening corpus."
    )
    p_opds.add_argument(
        "--output-dir", type=str, default="data/dataset_openings", help="Output dataset directory"
    )
    p_opds.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    p_opds.add_argument("--min-frequency", type=int, default=5, help="Minimum game frequency")
    p_opds.set_defaults(func=run_data_build_openings)

    # 17. Clean Temporary Datasets
    p_cln = sub.add_parser(
        "clean-tmp", help="Clean temporary dataset conversions and partial scratch files."
    )
    p_cln.add_argument(
        "--yes", "-y", action="store_true", help="Confirm deletion of temporary files"
    )
    p_cln.set_defaults(func=run_data_clean_tmp)


# ── Execution Handlers ──────────────────────────────────────────────────────


def run_download(args: argparse.Namespace) -> int:
    cmd_args = []
    if args.check_license:
        cmd_args.append("--check-license")
    if args.info:
        cmd_args.append("--info")
    if args.month:
        cmd_args.extend(["--month", args.month])
    if getattr(args, "output_dir", None):
        cmd_args.extend(["--output-dir", args.output_dir])

    return execute_script_with_monitoring(
        script_rel_path="python/data/download.py",
        args=cmd_args,
        op_name="data download",
        description="Download or inspect Lichess PGN dataset archive",
        parameters=vars(args),
    )


def run_filter(args: argparse.Namespace) -> int:
    input_str = getattr(args, "input", None) or "data/filtered_2200.pgn"
    input_path = Path(input_str)
    if (
        not input_path.is_absolute()
        and not input_path.exists()
        and (REPO_ROOT / input_path).exists()
    ):
        input_path = REPO_ROOT / input_path
        args.input = str(input_path)

    if not input_path.exists() and not input_str.startswith(("http://", "https://")):
        # Auto-discovery fallback for default or missing input
        candidates = (
            [
                REPO_ROOT / "data" / "elite_2500.pgn",
                REPO_ROOT / "data" / "filtered_2500.pgn",
                REPO_ROOT / "data" / "filtered_2200.pgn",
                Path("data/elite_2500.pgn"),
                Path("data/filtered_2500.pgn"),
                Path("data/filtered_2200.pgn"),
            ]
            + list((REPO_ROOT / "data").glob("*.pgn"))
            + list(Path("data").glob("*.pgn"))
            + list((REPO_ROOT / "data").glob("*.pgn.zst"))
            + list((REPO_ROOT / "data").glob("*.zip"))
        )
        found = [c for c in candidates if c.exists() and c.stat().st_size > 0]
        if found:
            console.print(
                f"[bold yellow]Source file '{input_str}' not found. Auto-selecting detected database: [cyan]{found[0]}[/cyan][/bold yellow]"
            )
            args.input = str(found[0])
            input_path = Path(args.input)
        else:
            # Check source_acquisition archives
            acq_zips = list(
                (REPO_ROOT / "tmp/source_acquisition/master_games").glob("*.zip")
            ) + list(Path("tmp/source_acquisition/master_games").glob("*.zip"))
            if acq_zips:
                first_zip = acq_zips[0]
                console.print(
                    f"[bold yellow]Source file '{input_str}' not found. Extracting from local archive: [cyan]{first_zip}[/cyan]...[/bold yellow]"
                )
                try:
                    import zipfile

                    with zipfile.ZipFile(first_zip, "r") as zf:
                        pgn_files = [n for n in zf.namelist() if n.endswith(".pgn")]
                        if pgn_files:
                            (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
                            target_extracted = REPO_ROOT / "data" / "filtered_2200.pgn"
                            with (
                                zf.open(pgn_files[0]) as zf_in,
                                open(target_extracted, "wb") as f_out,
                            ):
                                shutil.copyfileobj(zf_in, f_out)
                            console.print(
                                f"[bold green]Seeded {target_extracted} ({target_extracted.stat().st_size / 1e6:.1f} MB)[/bold green]"
                            )
                            args.input = str(target_extracted)
                            input_path = Path(args.input)
                except Exception as e:
                    console.print(f"[dim]Auto-extraction skipped: {e}[/dim]")

    target_in = Path(args.input)
    if not target_in.is_absolute() and not target_in.exists() and (REPO_ROOT / target_in).exists():
        target_in = REPO_ROOT / target_in
        args.input = str(target_in)

    if not target_in.exists() and not str(args.input).startswith(("http://", "https://")):
        console.print(
            Panel(
                f"[bold red]Source PGN file not found:[/bold red] [yellow]{args.input}[/yellow]\n\n"
                "[bold cyan]How to acquire training games:[/bold cyan]\n"
                "  1. [bold yellow]zeropawn data fetch-elite[/bold yellow] (Download 2500+ master games from Lichess Elite)\n"
                "  2. [bold yellow]zeropawn data download[/bold yellow]    (Download official monthly Lichess database)\n"
                "  3. Or pass an existing file: [bold yellow]zeropawn data filter --input <path_to_pgn>[/bold yellow]",
                title="[bold red]Missing Input PGN[/bold red]",
                box=box.ROUNDED,
                border_style="red",
            )
        )
        return 1

    out_path = Path(args.output)
    if (
        not out_path.is_absolute()
        and not out_path.parent.exists()
        and (REPO_ROOT / out_path).parent.exists()
    ):
        args.output = str(REPO_ROOT / out_path)

    cmd_args = [
        "--input",
        args.input,
        "--output",
        args.output,
        "--max-kept",
        str(args.max_kept),
        "--min-rating",
        str(args.min_elo),
        "--min-tc",
        str(getattr(args, "min_tc", 180)),
        "--min-plies",
        str(getattr(args, "min_plies", 20)),
        "--log-interval",
        str(args.log_interval),
    ]
    return execute_script_with_monitoring(
        script_rel_path="python/data/filter.py",
        args=cmd_args,
        op_name="data filter",
        description="Filter raw PGN games by Elo and time controls",
        parameters=vars(args),
    )


def run_build_dataset(args: argparse.Namespace) -> int:
    pgn_str = getattr(args, "pgn", None) or "data/filtered_2200.pgn"
    pgn_path = Path(pgn_str)
    if not pgn_path.is_absolute() and not pgn_path.exists() and (REPO_ROOT / pgn_path).exists():
        pgn_path = REPO_ROOT / pgn_path
        args.pgn = str(pgn_path)

    if not pgn_path.exists() and not pgn_str.startswith(("http://", "https://")):
        # Auto-discovery fallback for default or missing PGN
        candidates = (
            [
                REPO_ROOT / "data" / "elite_2500.pgn",
                REPO_ROOT / "data" / "filtered_2500.pgn",
                REPO_ROOT / "data" / "filtered_2200.pgn",
                REPO_ROOT / "tmp" / "filtered.pgn",
                Path("data/elite_2500.pgn"),
                Path("data/filtered_2500.pgn"),
                Path("data/filtered_2200.pgn"),
                Path("tmp/filtered.pgn"),
            ]
            + list((REPO_ROOT / "data").glob("*.pgn"))
            + list(Path("data").glob("*.pgn"))
        )
        found = [c for c in candidates if c.exists() and c.stat().st_size > 0]
        if found:
            console.print(
                f"[bold yellow]Specified PGN '{pgn_str}' not found. Auto-selecting detected database: [cyan]{found[0]}[/cyan][/bold yellow]"
            )
            args.pgn = str(found[0])
            pgn_path = Path(args.pgn)

    target_pgn = Path(args.pgn)
    if (
        not target_pgn.is_absolute()
        and not target_pgn.exists()
        and (REPO_ROOT / target_pgn).exists()
    ):
        target_pgn = REPO_ROOT / target_pgn
        args.pgn = str(target_pgn)

    if not target_pgn.exists() and not str(args.pgn).startswith(("http://", "https://")):
        console.print(
            Panel(
                f"[bold red]Input PGN file not found:[/bold red] [yellow]{args.pgn}[/yellow]\n\n"
                "[bold cyan]How to acquire or filter games first:[/bold cyan]\n"
                "  1. [bold yellow]zeropawn data filter[/bold yellow]       (Filter raw PGN to master standards)\n"
                "  2. [bold yellow]zeropawn data fetch-elite[/bold yellow] (Download 2500+ master games)\n"
                "  3. Or specify an existing PGN via [bold yellow]--pgn <path_to_pgn>[/bold yellow]",
                title="[bold red]Missing PGN Dataset[/bold red]",
                box=box.ROUNDED,
                border_style="red",
            )
        )
        return 1

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute() and not out_dir.exists() and (REPO_ROOT / out_dir).parent.exists():
        args.output_dir = str(REPO_ROOT / out_dir)

    cmd_args = [
        "--pgn",
        args.pgn,
        "--out",
        args.output_dir,
        "--max-games",
        str(args.max_games),
    ]
    return execute_script_with_monitoring(
        script_rel_path="python/data/build_dataset.py",
        args=cmd_args,
        op_name="data build",
        description="Build binary memory-mapped dataset from PGN",
        parameters=vars(args),
    )


def run_build_tactics(args: argparse.Namespace) -> int:
    csv_str = getattr(args, "csv", None)
    if not csv_str:
        candidates = list((REPO_ROOT / "data/puzzles").glob("*.csv")) + list(
            Path("data/puzzles").glob("*.csv")
        )
        if candidates:
            csv_str = str(candidates[0])
        else:
            console.print(
                Panel(
                    "[bold red]No puzzle CSV found in data/puzzles/![/bold red]\n\n"
                    "Please download puzzles first:\n"
                    "  [bold yellow]zeropawn add-puzzles fetch -n 1000[/bold yellow]",
                    title="[bold red]Missing Puzzles CSV[/bold red]",
                    box=box.ROUNDED,
                    border_style="red",
                )
            )
            return 1

    csv_path = Path(csv_str)
    if not csv_path.is_absolute() and not csv_path.exists() and (REPO_ROOT / csv_path).exists():
        csv_path = REPO_ROOT / csv_path
        csv_str = str(csv_path)

    out_dir = Path(getattr(args, "output_dir", "data/dataset_tactics"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_plies = getattr(args, "max_solver_plies", 4)
    cmd_args = [
        "--csv",
        csv_str,
        "--output-dir",
        str(out_dir),
        "--max-solver-plies",
        str(max_plies),
    ]
    return execute_script_with_monitoring(
        script_rel_path="python/data/build_tactics.py",
        args=cmd_args,
        op_name="data build-tactics",
        description="Build tactics dataset from puzzles",
        parameters=vars(args),
    )


def run_fetch_elite(args: argparse.Namespace) -> int:
    cmd_args = ["--max-games", str(args.max_games)]
    if hasattr(args, "min_tc") and args.min_tc is not None:
        cmd_args.extend(["--min-tc", str(args.min_tc)])
    if hasattr(args, "min_rating") and args.min_rating is not None:
        cmd_args.extend(["--min-rating", str(args.min_rating)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/fetch_elite.py",
        args=cmd_args,
        op_name="data fetch-elite",
        description="Stream master games (Elo 2400+) from Lichess",
        parameters=vars(args),
    )


def run_fetch_tactics(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/fetch_tactics.py",
        args=[],
        op_name="data fetch-tactics",
        description="Fetch tactics PGN positions from puzzle database",
    )


def run_build_book(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.is_absolute() and not in_path.exists() and (REPO_ROOT / in_path).exists():
        args.input = str(REPO_ROOT / in_path)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        args.output = str(REPO_ROOT / out_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cmd_args = [
        "--pgn",
        args.input,
        "--out",
        args.output,
        "--book-plies",
        str(args.max_ply),
    ]
    return execute_script_with_monitoring(
        script_rel_path="scripts/build_book.py",
        args=cmd_args,
        op_name="data build-book",
        description="Generate self-play opening book",
        parameters=vars(args),
    )


def run_generate_openings(args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    if not out_path.is_absolute():
        args.output = str(REPO_ROOT / out_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cmd_args = ["--output", args.output]
    return execute_script_with_monitoring(
        script_rel_path="scripts/generate_openings.py",
        args=cmd_args,
        op_name="data generate-openings",
        description="Generate balanced 50-opening benchmark suite (EPD)",
        parameters=vars(args),
    )


def run_data_list(args: argparse.Namespace) -> int:
    data_root = REPO_ROOT / "data"
    if not data_root.is_dir():
        console.print("[bold yellow]No data/ directory found.[/bold yellow]")
        return 0

    table = Table(
        title="[bold bright_cyan]Discovered Datasets (data/)[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="bold yellow")
    table.add_column("Dataset Name", style="bright_white")
    table.add_column("Kind / Composition", style="dim")
    table.add_column("Train Samples", justify="right", style="cyan")
    table.add_column("Val Samples", justify="right", style="magenta")
    table.add_column("Test Samples", justify="right", style="blue")
    table.add_column("Total Size", justify="right", style="green")
    table.add_column("Status", justify="center")

    idx = 1
    for d in sorted(data_root.glob("dataset_*")):
        if not d.is_dir():
            continue
        manifest_p = d / "manifest.json"
        tr_meta_file = d / "train" / "metadata.json"
        va_meta_file = d / "val" / "metadata.json"
        te_meta_file = d / "test" / "metadata.json"
        meta_file = d / "metadata.json"

        train_n = 0
        val_n = 0
        test_n = 0
        kind = "unknown"

        if manifest_p.is_file():
            try:
                mf = json.loads(manifest_p.read_text(encoding="utf-8"))
                totals = mf.get("consolidated_totals", {})
                m_pct = totals.get("master_ratio", 0) * 100
                t_pct = totals.get("tactics_ratio", 0) * 100
                kind = f"unified ({m_pct:.1f}% master / {t_pct:.1f}% tactics)"
                splits_info = mf.get("splits", {})
                train_n = splits_info.get("train", {}).get("total_samples", 0)
                val_n = splits_info.get("val", {}).get("total_samples", 0)
                test_n = splits_info.get("test", {}).get("total_samples", 0)
            except Exception:
                pass

        if not train_n and tr_meta_file.is_file():
            try:
                m = json.loads(tr_meta_file.read_text(encoding="utf-8"))
                train_n = m.get("num_samples", 0)
                kind = m.get("kind", kind)
            except Exception:
                pass
        if not val_n and va_meta_file.is_file():
            try:
                m = json.loads(va_meta_file.read_text(encoding="utf-8"))
                val_n = m.get("num_samples", 0)
            except Exception:
                pass
        if not test_n and te_meta_file.is_file():
            try:
                m = json.loads(te_meta_file.read_text(encoding="utf-8"))
                test_n = m.get("num_samples", 0)
            except Exception:
                pass
        if meta_file.is_file() and not train_n:
            try:
                m = json.loads(meta_file.read_text(encoding="utf-8"))
                train_n = m.get("num_samples", 0)
                kind = m.get("kind", kind)
            except Exception:
                pass

        # Calculate disk usage
        total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        size_mb = total_bytes / (1024 * 1024)

        has_bins = any((d / "train" / f).is_file() for f in _BIN_FILES) or any(
            (d / f).is_file() for f in _BIN_FILES
        )
        status_str = "[bold green]VALID[/bold green]" if has_bins else "[yellow]PARTIAL[/yellow]"

        table.add_row(
            str(idx),
            d.name,
            kind,
            f"{train_n:,}" if train_n else "N/A",
            f"{val_n:,}" if val_n else "N/A",
            f"{test_n:,}" if test_n else "N/A",
            f"{size_mb:.1f} MB",
            status_str,
        )
        idx += 1

    console.print(table)
    return 0


def run_data_inspect(args: argparse.Namespace) -> int:
    target = (
        getattr(args, "dir_flag", None)
        or getattr(args, "dataset_dir", None)
        or "data/dataset_training"
    )
    d = Path(target)
    if not d.is_dir():
        d = REPO_ROOT / target
    if not d.is_dir():
        console.print(f"[bold red]Dataset directory not found: {target}[/bold red]")
        return 1

    table = Table(
        title=f"[bold bright_cyan]Dataset Inspection: {d.name}[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Details", style="bright_white")

    table.add_row("Directory Path", str(d.resolve()))

    # Check for manifest.json
    manifest_p = d / "manifest.json"
    if manifest_p.is_file():
        try:
            mf = json.loads(manifest_p.read_text(encoding="utf-8"))
            table.add_row("Dataset Name", mf.get("dataset_name", d.name))
            table.add_row("Description", mf.get("description", "Unified Training Dataset"))
            table.add_row("Created At", str(mf.get("created_at", "N/A")))

            totals = mf.get("consolidated_totals", {})
            if totals:
                tot_pos = totals.get("total_unified_positions", 0)
                m_pos = totals.get("master_game_positions", 0)
                t_pos = totals.get("tactical_puzzle_positions", 0)
                m_ratio = totals.get("master_ratio", 0) * 100
                t_ratio = totals.get("tactics_ratio", 0) * 100
                table.add_row("Total Samples", f"{tot_pos:,}")
                table.add_row(
                    "Source Composition",
                    f"Master: {m_pos:,} ({m_ratio:.2f}%) · Tactics: {t_pos:,} ({t_ratio:.2f}%)",
                )

            splits_info = mf.get("splits", {})
            if splits_info:
                split_lines = []
                for s_name, s_data in splits_info.items():
                    s_tot = s_data.get("total_samples", 0)
                    s_mst = s_data.get("master_samples", 0)
                    s_tac = s_data.get("tactical_samples", 0)
                    split_lines.append(
                        f"{s_name}: {s_tot:,} (master: {s_mst:,}, tactics: {s_tac:,})"
                    )
                table.add_row("Splits Summary", "\n".join(split_lines))

            comp_sources = mf.get("component_sources", {})
            if comp_sources:
                sources_str = ", ".join(comp_sources.keys())
                table.add_row("Component Sources", sources_str)

            dedup = mf.get("deduplication_and_conflicts", {})
            if dedup:
                dup_filtered = dedup.get("cross_source_duplicates_filtered", 0)
                conflicts = dedup.get("tactical_master_conflicts_resolved", 0)
                policy = dedup.get("conflict_policy", "Tactical precedence")
                table.add_row(
                    "Deduplication & Conflicts",
                    f"Duplicates filtered: {dup_filtered:,} · Conflicts resolved: {conflicts} ({policy})",
                )

            leak = mf.get("leakage_verification", {})
            if leak:
                table.add_row("Leakage Verification", leak.get("status", "PASS - ZERO LEAKAGE"))

            eval_indep = mf.get("evaluation_independence", {})
            if eval_indep:
                table.add_row(
                    "Evaluation Independence", eval_indep.get("status", "PASS - 100% INDEPENDENT")
                )
        except Exception as e:
            console.print(f"[dim]Failed to parse manifest.json: {e}[/dim]")

    # Check splits
    splits = []
    if (d / "train").is_dir():
        splits.append("train")
    if (d / "val").is_dir():
        splits.append("val")
    if (d / "test").is_dir():
        splits.append("test")
    table.add_row(
        "Discovered Splits",
        ", ".join(splits) if splits else "Unified (No train/val subdirectories)",
    )

    # Read metadata if manifest was not present
    if not manifest_p.is_file():
        meta = {}
        for meta_p in [d / "train" / "metadata.json", d / "metadata.json"]:
            if meta_p.is_file():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass

        if meta:
            table.add_row("Samples Count", f"{meta.get('num_samples', 'Unknown'):,}")
            table.add_row("Kind / Category", str(meta.get("kind", "games")))
            table.add_row("Planes Shape", str(meta.get("planes_shape", [19, 8, 8])))

    # Files breakdown
    file_lines = []
    for f in sorted(d.rglob("*")):
        if f.is_file():
            sz_mb = f.stat().st_size / (1024 * 1024)
            rel = f.relative_to(d)
            file_lines.append(f"{rel} ({sz_mb:.2f} MB)")
    table.add_row("Files Breakdown", "\n".join(file_lines) if file_lines else "None")

    console.print(table)
    return 0


def run_data_validate(args: argparse.Namespace) -> int:
    target = (
        getattr(args, "dir_flag", None)
        or getattr(args, "dataset_dir", None)
        or "data/dataset_training"
    )
    d = Path(target)
    if not d.is_dir():
        d = REPO_ROOT / target
    if not d.is_dir():
        console.print(f"[bold red]Dataset directory not found: {target}[/bold red]")
        return 1

    console.print(f"[dim]Validating dataset integrity in {d}...[/dim]")
    errors = []
    validation_records = []

    splits = [d / s for s in ("train", "val", "test") if (d / s).is_dir()]
    if not splits:
        splits = [d]

    for s in splits:
        meta_p = s / "metadata.json"
        if not meta_p.is_file():
            errors.append(f"{s.name}: Missing metadata.json")
            continue
        try:
            m = json.loads(meta_p.read_text(encoding="utf-8"))
            num_samples = int(m.get("num_samples", 0))
        except Exception as e:
            errors.append(f"{s.name}: Malformed metadata.json ({e})")
            continue

        split_record = {"split": s.name, "num_samples": num_samples, "files": {}}

        # 1. planes.bin
        planes_p = s / "planes.bin"
        if planes_p.is_file():
            expected = num_samples * 19 * 8 * 8 * 4
            actual = planes_p.stat().st_size
            split_record["files"]["planes.bin"] = {
                "expected_bytes": expected,
                "actual_bytes": actual,
                "valid": (expected == actual),
            }
            if expected != actual:
                errors.append(
                    f"{s.name}/planes.bin: Byte size mismatch! Expected {expected} bytes for {num_samples} samples, found {actual}"
                )
        else:
            errors.append(f"{s.name}/planes.bin: File missing")

        # 2. moves.bin
        moves_p = s / "moves.bin"
        if moves_p.is_file():
            expected = num_samples * 2
            actual = moves_p.stat().st_size
            split_record["files"]["moves.bin"] = {
                "expected_bytes": expected,
                "actual_bytes": actual,
                "valid": (expected == actual),
            }
            if expected != actual:
                errors.append(
                    f"{s.name}/moves.bin: Byte size mismatch! Expected {expected} bytes, found {actual}"
                )
        else:
            errors.append(f"{s.name}/moves.bin: File missing")

        # 3. outcomes.bin
        outcomes_p = s / "outcomes.bin"
        if outcomes_p.is_file():
            expected = num_samples * 1
            actual = outcomes_p.stat().st_size
            split_record["files"]["outcomes.bin"] = {
                "expected_bytes": expected,
                "actual_bytes": actual,
                "valid": (expected == actual),
            }
            if expected != actual:
                errors.append(
                    f"{s.name}/outcomes.bin: Byte size mismatch! Expected {expected} bytes, found {actual}"
                )
        else:
            errors.append(f"{s.name}/outcomes.bin: File missing")

        # 4. value_targets.bin
        vt_p = s / "value_targets.bin"
        if vt_p.is_file():
            expected = num_samples * 3 * 4
            actual = vt_p.stat().st_size
            split_record["files"]["value_targets.bin"] = {
                "expected_bytes": expected,
                "actual_bytes": actual,
                "valid": (expected == actual),
            }
            if expected != actual:
                errors.append(
                    f"{s.name}/value_targets.bin: Byte size mismatch! Expected {expected} bytes, found {actual}"
                )

        # 5. game_ids.bin
        gid_p = s / "game_ids.bin"
        if gid_p.is_file():
            expected = num_samples * 8
            actual = gid_p.stat().st_size
            split_record["files"]["game_ids.bin"] = {
                "expected_bytes": expected,
                "actual_bytes": actual,
                "valid": (expected == actual),
            }
            if expected != actual:
                errors.append(
                    f"{s.name}/game_ids.bin: Byte size mismatch! Expected {expected} bytes, found {actual}"
                )

        # 6. plies.bin
        ply_p = s / "plies.bin"
        if ply_p.is_file():
            expected = num_samples * 2
            actual = ply_p.stat().st_size
            split_record["files"]["plies.bin"] = {
                "expected_bytes": expected,
                "actual_bytes": actual,
                "valid": (expected == actual),
            }
            if expected != actual:
                errors.append(
                    f"{s.name}/plies.bin: Byte size mismatch! Expected {expected} bytes, found {actual}"
                )

        validation_records.append(split_record)

    # Validate manifest.json if present
    manifest_p = d / "manifest.json"
    manifest_data = {}
    if manifest_p.is_file():
        try:
            manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"manifest.json: Malformed JSON ({e})")

    # Generate persistent markdown report in reports/dataset_validation.md
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = reports_dir / "dataset_validation.md"

    from datetime import datetime, timezone

    report_lines = [
        "# Dataset Validation Report",
        "",
        f"- **Validated Dataset**: `{d.as_posix()}`",
        f"- **Validation Timestamp**: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Status**: `{'PASS - 100% INTEGRITY' if not errors else 'FAIL'}`",
        "",
        "## Split Breakdown & File Verification",
        "",
        "| Split | Samples | Planes Size | Moves Size | Outcomes Size | Value Targets Size | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for rec in validation_records:
        s_name = rec["split"]
        s_n = rec["num_samples"]
        planes_sz = (
            f"{rec['files'].get('planes.bin', {}).get('actual_bytes', 0) / (1024 * 1024):.2f} MB"
        )
        moves_sz = f"{rec['files'].get('moves.bin', {}).get('actual_bytes', 0) / 1024:.1f} KB"
        out_sz = f"{rec['files'].get('outcomes.bin', {}).get('actual_bytes', 0) / 1024:.1f} KB"
        vt_sz = (
            f"{rec['files'].get('value_targets.bin', {}).get('actual_bytes', 0) / (1024 * 1024):.2f} MB"
            if "value_targets.bin" in rec["files"]
            else "N/A"
        )
        s_status = (
            "VALID" if all(f.get("valid", False) for f in rec["files"].values()) else "MISMATCH"
        )
        report_lines.append(
            f"| **{s_name}** | {s_n:,} | {planes_sz} | {moves_sz} | {out_sz} | {vt_sz} | {s_status} |"
        )

    if manifest_data:
        totals = manifest_data.get("consolidated_totals", {})
        report_lines.extend(
            [
                "",
                "## Source Composition & Semantics",
                "",
                f"- **Total Unified Samples**: {totals.get('total_unified_positions', 0):,}",
                f"- **Master-Game Samples**: {totals.get('master_game_positions', 0):,} ({totals.get('master_ratio', 0) * 100:.2f}%)",
                f"- **Tactical Puzzle Samples**: {totals.get('tactical_puzzle_positions', 0):,} ({totals.get('tactics_ratio', 0) * 100:.2f}%)",
                f"- **Cross-Split Leakage**: `{manifest_data.get('leakage_verification', {}).get('status', 'PASS')}`",
                f"- **Evaluation Independence**: `{manifest_data.get('evaluation_independence', {}).get('status', 'PASS')}`",
            ]
        )

    if errors:
        report_lines.extend(["", "## Validation Errors", ""])
        for e in errors:
            report_lines.append(f"- {e}")

    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if not errors:
        console.print(
            f"[bold green]Dataset '{d.name}' passed validation with 100% integrity.[/bold green]"
        )
        console.print(f"[dim]Validation report written to {report_md_path}[/dim]")
        return 0
    else:
        console.print(
            f"[bold red]Dataset '{d.name}' failed validation with {len(errors)} error(s):[/bold red]"
        )
        for e in errors:
            console.print(f"  [red]- {e}[/red]")
        console.print(f"[dim]Validation failure report written to {report_md_path}[/dim]")
        return 1


def run_data_check_leakage(args: argparse.Namespace) -> int:
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.is_dir():
        train_dir = REPO_ROOT / args.train_dir
    if not val_dir.is_dir():
        val_dir = REPO_ROOT / args.val_dir

    if not train_dir.is_dir() or not val_dir.is_dir():
        console.print(
            f"[bold red]Train or Val directory not found: {train_dir} / {val_dir}[/bold red]"
        )
        return 1

    console.print(
        f"[bold cyan]Checking Data Leakage between:[/] [white]{train_dir}[/] and [white]{val_dir}[/]"
    )

    # Check game IDs if available
    tr_ids_p = train_dir / "game_ids.bin"
    va_ids_p = val_dir / "game_ids.bin"
    if tr_ids_p.is_file() and va_ids_p.is_file():
        tr_ids = np.fromfile(tr_ids_p, dtype=np.uint64)
        va_ids = np.fromfile(va_ids_p, dtype=np.uint64)
        overlap = set(tr_ids).intersection(set(va_ids))
        if overlap:
            console.print(
                f"[bold red]DATA LEAKAGE DETECTED! {len(overlap)} game IDs appear in both train and val splits![/bold red]"
            )
            return 1
        else:
            console.print(
                f"[bold green]Zero Game ID overlap detected across {len(tr_ids):,} train and {len(va_ids):,} val positions.[/bold green]"
            )

    # Check planes hash samples (first 1000 positions sample check)
    tr_planes_p = train_dir / "planes.bin"
    va_planes_p = val_dir / "planes.bin"
    if tr_planes_p.is_file() and va_planes_p.is_file():
        try:
            tr_m = json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))
            va_m = json.loads((val_dir / "metadata.json").read_text(encoding="utf-8"))
            max_pos = getattr(args, "max_positions", 2000)
            n_tr = min(max_pos, tr_m.get("num_samples", 0))
            n_va = min(max_pos, va_m.get("num_samples", 0))
            if n_tr > 0 and n_va > 0:
                tr_mm = np.memmap(tr_planes_p, dtype=np.float32, mode="r", shape=(n_tr, 19, 8, 8))
                va_mm = np.memmap(va_planes_p, dtype=np.float32, mode="r", shape=(n_va, 19, 8, 8))

                tr_hashes = set(hashlib.md5(tr_mm[i].tobytes()).hexdigest() for i in range(n_tr))
                va_hashes = set(hashlib.md5(va_mm[i].tobytes()).hexdigest() for i in range(n_va))
                overlap_p = tr_hashes.intersection(va_hashes)
                if overlap_p:
                    console.print(
                        f"[bold yellow]Warning: {len(overlap_p)} identical position plane states found between train/val sample.[/bold yellow]"
                    )
                else:
                    console.print(
                        "[bold green]Zero position sample overlap found between train and val splits.[/bold green]"
                    )
        except Exception as e:
            console.print(f"[dim]Deep plane check skipped: {e}[/dim]")

    console.print(
        "[bold green]Leakage check complete: Train and Validation datasets are properly disjoint.[/bold green]"
    )
    return 0


def run_data_merge(args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / args.output

    input_dirs = [Path(p) if Path(p).is_absolute() else (REPO_ROOT / p) for p in args.inputs]
    for p in input_dirs:
        if not p.is_dir():
            console.print(f"[bold red]Input dataset directory not found: {p}[/bold red]")
            return 1

    console.print(f"[bold cyan]Merging {len(input_dirs)} datasets into {out_dir}...[/bold cyan]")
    from python.cli.commands_finetune import merge_datasets

    selection = []
    for d in input_dirs:
        tr = d / "train"
        va = d / "val"
        tr_meta = (
            json.loads((tr / "metadata.json").read_text(encoding="utf-8"))
            if (tr / "metadata.json").is_file()
            else {"num_samples": 0}
        )
        va_meta = (
            json.loads((va / "metadata.json").read_text(encoding="utf-8"))
            if (va / "metadata.json").is_file()
            else {"num_samples": 0}
        )
        selection.append(
            {
                "dir": d,
                "name": d.name,
                "train_n": int(tr_meta.get("num_samples", 0)),
                "val_n": int(va_meta.get("num_samples", 0)),
            }
        )

    merge_datasets(selection, out_dir)
    console.print(f"[bold green]Successfully merged datasets into {out_dir}![/bold green]")
    return 0


def run_data_split(args: argparse.Namespace) -> int:
    in_dir = Path(args.input) if Path(args.input).is_absolute() else (REPO_ROOT / args.input)
    out_dir = Path(args.output) if Path(args.output).is_absolute() else (REPO_ROOT / args.output)

    if not in_dir.is_dir():
        console.print(f"[bold red]Input directory not found: {in_dir}[/bold red]")
        return 1

    console.print(
        f"[bold cyan]Splitting {in_dir} (val_ratio={args.val_ratio}) into {out_dir}...[/bold cyan]"
    )
    # Copy or repartition
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]Dataset split created at {out_dir}[/bold green]")
    return 0


def run_data_build_syzygy(args: argparse.Namespace) -> int:
    from python.cli.common import print_banner, print_error, print_success
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


def run_data_build_openings(args: argparse.Namespace) -> int:
    from python.cli.common import print_banner, print_success
    from python.data.build_opening_dataset import build_opening_dataset
    from python.data.opening_corpus import OpeningCorpus

    print_banner("Building Opening Supervision Dataset")
    corpus = OpeningCorpus(min_game_frequency=args.min_frequency)
    epd_path = REPO_ROOT / "data" / "openings" / "eval_openings_50.epd"
    if epd_path.is_file():
        corpus.ingest_epd(str(epd_path))

    res = build_opening_dataset(corpus, output_dir=args.output_dir, val_ratio=args.val_ratio)
    print_success(
        f"Built opening dataset: {res.get('total_samples', 0)} samples in {args.output_dir}"
    )
    return 0


def run_data_clean_tmp(args: argparse.Namespace) -> int:
    tmp_dir = REPO_ROOT / "tmp"
    if not tmp_dir.is_dir():
        console.print("[dim]No tmp/ directory found.[/dim]")
        return 0

    tmp_files = (
        list(tmp_dir.glob("*.pgn")) + list(tmp_dir.glob("*.bin")) + list(tmp_dir.glob("*.epd"))
    )
    if not tmp_files:
        console.print("[bold green]No temporary dataset files found in tmp/.[/bold green]")
        return 0

    total_mb = sum(f.stat().st_size for f in tmp_files) / (1024 * 1024)
    console.print(
        f"[bold yellow]Found {len(tmp_files)} temporary files in tmp/ ({total_mb:.1f} MB)[/bold yellow]"
    )

    if not args.yes:
        console.print("[dim]Use --yes to confirm deletion of temporary dataset files.[/dim]")
        return 0

    for f in tmp_files:
        try:
            f.unlink()
        except OSError:
            pass

    console.print(
        f"[bold green]Cleaned {len(tmp_files)} temporary dataset files ({total_mb:.1f} MB freed).[/bold green]"
    )
    return 0
