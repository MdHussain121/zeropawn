"""Data operations subcommand for ZeroPawn: download, filter, convert, and dataset utilities."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

from rich import box
from rich.panel import Panel
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.runner import execute_script_with_monitoring
from python.cli.ui import COLOR_BORDER, console
from python.constants import NUM_PLANES, BOARD_SIZE

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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output dataset directory. REQUIRED: there is deliberately no default target "
            "(audit SA2-02) — omitting it fails closed instead of building into "
            "data/dataset_training."
        ),
    )
    p_bd.add_argument("--max-games", type=int, default=5000, help="Maximum games to convert")
    p_bd.add_argument(
        "--force",
        action="store_true",
        help=(
            "Explicitly allow building into an existing non-empty dataset directory "
            "(replaces its contents; certification-aware guard refuses without this flag)."
        ),
    )
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
    p_bt.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite if target directory already contains a dataset",
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
        "validate",
        help="Layered fail-closed dataset validation (structural, semantic, provenance, split).",
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
        help="Path to dataset directory (e.g. data/dataset_training)",
    )
    p_val.add_argument(
        "--level",
        choices=("structural", "semantic", "full"),
        default="full",
        help="Validation depth: structural (Layer A), semantic (A+B), full (A+B+C+D).",
    )
    p_val.add_argument(
        "--scan-planes",
        choices=("off", "bounded", "full"),
        default="bounded",
        help="Plane numeric scan coverage (bounded scans the first --scan-rows rows).",
    )
    p_val.add_argument(
        "--scan-rows", type=int, default=20000, help="Rows scanned by bounded plane checks."
    )
    p_val.add_argument(
        "--role",
        choices=("games_wdl", "policy_tactics", "endgame_syzygy", "unified"),
        default=None,
        help="Explicit dataset role (otherwise resolved from metadata declarations).",
    )
    p_val.add_argument(
        "--write-receipt",
        dest="write_receipt",
        action="store_true",
        default=False,
        help="Certify the dataset by writing validation_receipt.json into the dataset "
        "root on success. Explicit by default-else-off: validation is read-only unless "
        "certification is requested (a stray bounded-scan validate run must never "
        "downgrade an existing definitive receipt).",
    )
    p_val.add_argument(
        "--no-write-receipt",
        dest="write_receipt",
        action="store_false",
        help="Do not write a validation receipt (the default).",
    )
    p_val.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even when required checks could not complete (result stays marked incomplete).",
    )
    p_val.set_defaults(func=run_data_validate)

    # 12. Check Data Leakage
    p_leak = sub.add_parser(
        "check-leakage",
        help="Fail-closed train/val leakage check (group IDs + encoded states).",
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
        "--test-dir",
        type=str,
        default=None,
        help="Optional test split directory; when given, train/test and val/test disjointness "
        "are checked as well (audit SLQ-09).",
    )
    p_leak.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Bound the encoded-state scan to the first N rows of each split (bounded result; "
        "cannot certify disjointness).",
    )
    p_leak.add_argument(
        "--allow-bounded",
        action="store_true",
        help="Exit 0 on a bounded (non-exhaustive) scan with no overlap found; without this flag "
        "a bounded scan exits 3 (incomplete).",
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


def _existing_dataset_entries(target: Path) -> list[str]:
    """Enumerate entries of an existing target directory (empty list if absent/empty)."""
    if not target.is_dir():
        return []
    return sorted(p.name for p in target.iterdir())


def _describe_existing_dataset(target: Path) -> str:
    """Human description of an existing target: certified / built / unknown contents."""
    from python.data.dataset_contract import RECEIPT_NAME

    receipt_path = target / RECEIPT_NAME
    if not receipt_path.is_file() and (target / "train").is_dir():
        receipt_path = target / "train" / RECEIPT_NAME
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            overall = receipt.get("overall")
            if overall == "pass":
                return "a CERTIFIED dataset (validation receipt overall=pass)"
            return f"a validated dataset (validation receipt overall={overall!r})"
        except Exception:
            return "a dataset with a malformed validation receipt"
    if (target / "metadata.json").is_file() or any(target.glob("*.bin")):
        return "a built dataset (array artifacts present, no validation receipt)"
    return f"a non-empty directory ({len(_existing_dataset_entries(target))} entries)"


def run_build_dataset(args: argparse.Namespace) -> int:
    # ── Fail-closed target selection (audit SA2-02) ──────────────────────
    # There is deliberately no default output directory: an invocation without
    # an explicit target must not touch persistent datasets. Resolve relative
    # targets the way the build subprocess will (its cwd is always REPO_ROOT).
    raw_target = getattr(args, "output_dir", None)
    if raw_target is None or not str(raw_target).strip():
        console.print(
            Panel(
                "[bold red]No output directory specified.[/bold red]\n\n"
                "'data build' has no default target (audit SA2-02 fail-closed): it will not\n"
                "build into data/dataset_training or any other persistent directory implicitly.\n\n"
                "[bold cyan]Re-run with an explicit target:[/bold cyan]\n"
                "  [yellow]zeropawn data build --pgn <input.pgn> --output-dir <staging_dir>[/yellow]\n"
                "  e.g. a scratch/staging path; pass --force to replace an existing dataset.",
                title="[bold red]Missing Output Directory[/bold red]",
                box=box.ROUNDED,
                border_style="red",
            )
        )
        return 1

    target = Path(str(raw_target))
    if not target.is_absolute():
        target = REPO_ROOT / target

    # Never a repository-critical directory, even with --force.
    protected = {REPO_ROOT, REPO_ROOT / "data", REPO_ROOT / "models"}
    if target.resolve() in {p.resolve() for p in protected}:
        console.print(
            f"[bold red]Refusing to build into {target}: repository-critical directory.[/bold red]"
        )
        return 1

    if target.exists() and not target.is_dir():
        console.print(f"[bold red]Output target exists and is not a directory: {target}[/bold red]")
        return 1

    existing_entries = _existing_dataset_entries(target)
    if existing_entries and not getattr(args, "force", False):
        console.print(
            Panel(
                f"[bold red]Refusing to build into {target}:[/bold red] it already contains "
                f"{_describe_existing_dataset(target)}.\n\n"
                "Building here would replace existing dataset bytes. Re-run with\n"
                "[yellow]--force[/yellow] to explicitly overwrite, or choose a fresh staging "
                "directory for [cyan]--output-dir[/cyan].\n\n"
                "[dim]Certification note: overwriting a certified dataset invalidates its "
                "validation receipt; re-run 'zeropawn data validate --dir <dataset> "
                "--write-receipt' after any rebuild.[/dim]",
                title="[bold red]Existing Dataset Target[/bold red]",
                box=box.ROUNDED,
                border_style="red",
            )
        )
        return 1

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
        "--output-dir",
        args.output_dir,
        "--max-games",
        str(args.max_games),
    ]
    if getattr(args, "force", False):
        cmd_args.append("--force")
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

    protected = {REPO_ROOT, REPO_ROOT / "data", REPO_ROOT / "models"}
    if out_dir.resolve() in {p.resolve() for p in protected}:
        console.print(
            f"[bold red]Refusing to build into {out_dir}: repository-critical directory.[/bold red]"
        )
        return 1

    if out_dir.exists() and not out_dir.is_dir():
        console.print(f"[bold red]Output target exists and is not a directory: {out_dir}[/bold red]")
        return 1

    existing_entries = _existing_dataset_entries(out_dir)
    if existing_entries and not getattr(args, "force", False):
        console.print(
            Panel(
                f"[bold red]Refusing to build into {out_dir}:[/bold red] it already contains "
                f"{_describe_existing_dataset(out_dir)}.\n\n"
                "Building here would replace existing dataset bytes. Re-run with\n"
                "[yellow]--force[/yellow] to explicitly overwrite, or choose a fresh staging "
                "directory for [cyan]--output-dir[/cyan].\n\n"
                "[dim]Certification note: overwriting a certified dataset invalidates its "
                "validation receipt; re-run 'zeropawn data validate --dir <dataset> "
                "--write-receipt' after any rebuild.[/dim]",
                title="[bold red]Existing Dataset Target[/bold red]",
                box=box.ROUNDED,
                border_style="red",
            )
        )
        return 1

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

        train_has_all = all((d / "train" / f).is_file() for f in _BIN_FILES)
        root_has_all = all((d / f).is_file() for f in _BIN_FILES)
        has_all_bins = train_has_all or root_has_all
        if has_all_bins:
            receipt = d / "validation_receipt.json"
            if receipt.is_file():
                try:
                    from python.data.dataset_validation import verify_receipt

                    status_str = (
                        "[bold green]VALID[/bold green]"
                        if verify_receipt(d)
                        else "[bold red]INVALID[/bold red]"
                    )
                except Exception:
                    status_str = "[yellow]UNCERTIFIED[/yellow]"
            else:
                status_str = "[yellow]UNCERTIFIED[/yellow]"
        else:
            status_str = "[yellow]PARTIAL[/yellow]"

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
                table.add_row("Leakage Verification", leak.get("status", "NOT RECORDED"))

            eval_indep = mf.get("evaluation_independence", {})
            if eval_indep:
                table.add_row("Evaluation Independence", eval_indep.get("status", "NOT RECORDED"))
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
            table.add_row(
                "Planes Shape", str(meta.get("planes_shape", [NUM_PLANES, BOARD_SIZE, BOARD_SIZE]))
            )

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
    """Layered, fail-closed dataset validation (audit F-04).

    Exit codes: 0 = pass (definitive), 1 = validation failure,
    3 = incomplete (required checks skipped/bounded, no definitive result).
    """
    from python.data.dataset_contract import (
        RECEIPT_NAME,
        classify_role,
        role_by_name,
    )
    from python.data.dataset_validation import (
        CheckStatus,
        ValidationReport,
        check_split_integrity,
        validate_split,
        write_receipt,
    )

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

    level = args.level
    scan_planes = args.scan_planes if level != "structural" else "off"
    explicit_role = role_by_name(args.role) if getattr(args, "role", None) else None

    # Determine the split layout: a dataset root has train/val(/test) children;
    # a bare split directory is validated on its own.
    split_names = [s for s in ("train", "val", "test") if (d / s).is_dir()]
    is_root = bool(split_names)
    splits = [(d / s) for s in split_names] if is_root else [d]

    console.print(f"[dim]Validating dataset at {d} (level={level}, planes={scan_planes})...[/dim]")

    reports: dict[str, ValidationReport] = {}
    for split_dir in splits:
        role = explicit_role
        if role is None:
            meta_path = split_dir / "metadata.json"
            if meta_path.is_file():
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    role = classify_role(metadata)
                except Exception:
                    role = None
        reports[split_dir.name] = validate_split(
            split_dir,
            role=role,
            scan_planes=scan_planes,
            plane_scan_rows=args.scan_rows,
            compute_hashes=(level == "full"),
            check_build_record=(level == "full"),
        )

    # Layer D: cross-split integrity between every pair of splits (dataset root only).
    if is_root and level == "full" and len(splits) >= 2:
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                exhaustive = args.scan_planes != "bounded"
                pair = check_split_integrity(
                    splits[i],
                    splits[j],
                    exhaustive_states=exhaustive,
                    max_bounded_rows=args.scan_rows,
                )
                reports[f"{splits[i].name}+{splits[j].name}"] = ValidationReport(
                    dataset_dir=f"{splits[i]} vs {splits[j]}",
                    role=next(iter(reports.values())).role,
                )
                for r in pair:
                    reports[f"{splits[i].name}+{splits[j].name}"].add(r)

    overall_pass = True
    incomplete = False
    for name, report in reports.items():
        for r in report.results:
            if r.status == CheckStatus.FAIL:
                overall_pass = False
                console.print(f"[bold red]FAIL[/bold red] [{name}] {r.name}: {r.detail}")
            elif r.status == CheckStatus.SKIPPED:
                incomplete = True
                console.print(f"[yellow]SKIPPED[/yellow] [{name}] {r.name}: {r.detail}")
            elif r.status == CheckStatus.BOUNDED:
                incomplete = True
                console.print(
                    f"[yellow]BOUNDED[/yellow] [{name}] {r.name}: {r.detail} "
                    f"(coverage: {r.coverage})"
                )
        console.print(f"[dim]{name}: {report.summary_line()}[/dim]")
        if report.label_stats:
            for split, stats in report.label_stats.items():
                console.print(
                    f"[dim]labels[{split}]: {stats['wdl_labelled']:,} WDL-labelled, "
                    f"{stats['masked']:,} masked ({stats['masked_fraction']:.2%})[/dim]"
                )

    receipt_path = None
    if args.write_receipt and overall_pass:
        try:
            # An incomplete validation is recorded as such; consumption gates
            # reject receipts whose overall is not "pass".
            receipt_overall = "incomplete" if incomplete else "pass"
            receipt_path = write_receipt(d, reports, overall=receipt_overall)
        except Exception as e:
            console.print(f"[yellow]Receipt write failed: {e}[/yellow]")

    if not overall_pass:
        console.print(f"[bold red]Dataset '{d.name}' FAILED validation.[/bold red]")
        return 1
    if incomplete and not args.allow_incomplete:
        console.print(
            f"[bold yellow]Dataset '{d.name}' passed executed checks but validation is "
            "INCOMPLETE (skipped/bounded checks above). No definitive integrity claim is "
            "made; re-run with --scan-planes full for exhaustive coverage, or pass "
            "--allow-incomplete to accept an incomplete result.[/bold yellow]"
        )
        return 3
    console.print(
        f"[bold green]Dataset '{d.name}' passed {'definitive' if not incomplete else 'incomplete but accepted'} "
        f"validation.[/bold green]"
        + (f" Receipt: {receipt_path}" if receipt_path else "")
        + (
            f" ({RECEIPT_NAME} binds these results to the exact validated bytes.)"
            if receipt_path
            else ""
        )
    )
    return 0


def run_data_check_leakage(args: argparse.Namespace) -> int:
    """Fail-closed leakage check between two splits (audit F-04).

    Exit codes: 0 = definitively disjoint (exhaustive, all checks completed),
    1 = leakage detected, 2 = check could not complete (missing data / errors),
    3 = bounded scan only (no definitive result; requires --allow-bounded to
    accept). A definitive "disjoint" claim is never printed after a skipped
    or bounded check.
    """
    from python.data.dataset_validation import CheckStatus, check_split_integrity

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.is_dir():
        train_dir = REPO_ROOT / args.train_dir
    if not val_dir.is_dir():
        val_dir = REPO_ROOT / args.val_dir
    test_dir: Optional[Path] = None
    test_arg = getattr(args, "test_dir", None)
    if test_arg:
        test_dir = Path(test_arg)
        if not test_dir.is_dir():
            test_dir = REPO_ROOT / test_arg
        if not test_dir.is_dir():
            console.print(f"[bold red]Test directory not found: {test_dir}[/bold red]")
            return 2

    if not train_dir.is_dir() or not val_dir.is_dir():
        console.print(
            f"[bold red]Train or Val directory not found: {train_dir} / {val_dir}[/bold red]"
        )
        return 2

    exhaustive = args.max_positions is None
    console.print(
        f"[bold cyan]Checking Data Leakage between:[/] [white]{train_dir}[/] and [white]{val_dir}[/] "
        + (f"and [white]{test_dir}[/] " if test_dir else "")
        + f"({'exhaustive' if exhaustive else 'BOUNDED'})"
    )

    try:
        results = check_split_integrity(
            train_dir,
            val_dir,
            exhaustive_states=exhaustive,
            max_bounded_rows=args.max_positions or 2000,
            test_dir=test_dir,
        )
    except Exception as e:
        console.print(f"[bold red]Leakage check failed to complete: {e}[/bold red]")
        return 2

    exit_code = 0
    for r in results:
        if r.status == CheckStatus.FAIL:
            console.print(f"[bold red]FAIL[/bold red] {r.name}: {r.detail}")
            exit_code = 1
        elif r.status == CheckStatus.BOUNDED:
            console.print(f"[yellow]BOUNDED[/yellow] {r.name}: {r.detail} (coverage: {r.coverage})")
            exit_code = 3
        elif r.status == CheckStatus.SKIPPED:
            console.print(f"[yellow]SKIPPED[/yellow] {r.name}: {r.detail}")
            exit_code = 2
        else:
            console.print(f"[bold green]PASS[/bold green] {r.name}: {r.detail} ({r.coverage})")

    if exit_code == 0:
        console.print(
            "[bold green]Result: splits are DEFINITIVELY disjoint under the checked identities "
            "(exhaustive group-ID and encoded-state scans completed).[/bold green]"
        )
    elif exit_code == 3:
        if args.allow_bounded:
            console.print(
                "[bold yellow]Result: bounded scan found no overlap, but this is NOT an "
                "exhaustive result; no definitive disjoint claim is made (--allow-bounded).[/bold yellow]"
            )
            return 0
        console.print(
            "[bold yellow]Result: INCOMPLETE — the encoded-state scan was bounded. Pass --allow-bounded "
            "to accept a bounded result, or run without --max-positions for the exhaustive scan.[/bold yellow]"
        )
    elif exit_code == 2:
        console.print(
            "[bold red]Result: INCOMPLETE — a required check could not run. No disjoint claim is made.[/bold red]"
        )
    else:
        console.print(
            "[bold red]Result: LEAKAGE DETECTED — the splits share group IDs and/or encoded states.[/bold red]"
        )
    return exit_code


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
    # This command has never implemented an actual record split; historically it
    # silently created empty train/val directories (audit F-04 section 8).
    # Failing loudly instead of producing a "successful" no-op.
    console.print(
        Panel(
            "[bold red]`data split` is not implemented.[/bold red] It previously created only "
            "empty train/val directories without copying any records, which could be mistaken "
            "for a completed split. Use a real builder instead:\n\n"
            "  [bold yellow]zeropawn data build --pgn <pgn> --output-dir <dir>[/bold yellow] "
            "(98/2 game-hash split), or\n"
            "  [bold yellow]zeropawn data build-tactics --csv <csv> --output-dir <dir>[/bold yellow]",
            title="[bold red]Unimplemented Operation[/bold red]",
            box=box.ROUNDED,
            border_style="red",
        )
    )
    return 2


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
