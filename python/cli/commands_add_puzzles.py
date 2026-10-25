"""Add, fetch, import, and manage tactical chess puzzles."""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import zstandard

from python.cli.common import (
    REPO_ROOT,
    console,
    print_banner,
    print_error,
    print_success,
    safe_resolve,
    sanitize_theme,
    validate_fen_or_raise,
)
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
import logging

logger = logging.getLogger(__name__)

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_PUZZLE_CSV = REPO_ROOT / "data" / "puzzles" / "lichess_puzzles_1000_1200_1600.csv"


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "add-puzzles",
        aliases=["add-puzzle", "fetch-puzzles", "puzzle-tool"],
        help="Fetch new puzzles from Lichess, add custom puzzles, or list puzzle sets.",
        description="Stream puzzles from Lichess database by rating/theme, add custom tactics, or inspect puzzle sets.",
    )

    puzzle_actions = parser.add_subparsers(dest="add_action", help="Action to perform")

    # 1. Fetch from Lichess
    fetch_parser = puzzle_actions.add_parser(
        "fetch", help="Stream & download puzzles from Lichess database."
    )
    _add_fetch_args(fetch_parser)

    # 2. Manual Add
    manual_parser = puzzle_actions.add_parser(
        "manual", aliases=["add"], help="Add a custom puzzle with move validation."
    )
    _add_manual_args(manual_parser)

    # 3. List puzzle sets
    list_parser = puzzle_actions.add_parser(
        "list", aliases=["ls"], help="List all puzzle sets and statistics."
    )
    _add_list_args(list_parser)

    # 4. Import from CSV
    import_parser = puzzle_actions.add_parser(
        "import", help="Import puzzles from external CSV file."
    )
    _add_import_args(import_parser)

    # Set default action to fetch if arguments provided directly
    _add_fetch_args(parser)
    parser.set_defaults(func=route_add_puzzles)


def _add_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1000,
        help="Number of puzzles to collect (default: 1000)",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1200,
        help="Minimum puzzle rating (default: 1200)",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=1600,
        help="Maximum puzzle rating (default: 1600)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Filter by specific theme (e.g. 'mateIn1', 'fork', 'pin', 'endgame')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination CSV path (default: auto-named in data/puzzles/)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing puzzle file without duplicate puzzle IDs",
    )


def _add_manual_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--id", type=str, default=None, help="Custom puzzle ID (default: auto-generated)"
    )
    parser.add_argument(
        "--fen", type=str, default=None, help="Starting FEN position before opponent blunder"
    )
    parser.add_argument(
        "--opponent-move", type=str, default=None, help="Opponent setup move (UCI, e.g. 'd3d6')"
    )
    parser.add_argument(
        "--solution", type=str, default=None, help="Solution move(s) in UCI format (e.g. 'f8d8')"
    )
    parser.add_argument("--rating", type=int, default=1500, help="Puzzle rating (default: 1500)")
    parser.add_argument("--themes", type=str, default="custom", help="Themes separated by spaces")
    parser.add_argument(
        "--target-file",
        type=str,
        default="data/puzzles/custom_puzzles.csv",
        help="Target CSV file to save puzzle into (default: data/puzzles/custom_puzzles.csv)",
    )


def _add_list_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        type=str,
        default="data/puzzles",
        help="Puzzles directory to scan (default: data/puzzles)",
    )


def _add_import_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source_file", type=str, help="Source CSV file path to import")
    parser.add_argument(
        "--target-file",
        type=str,
        default="data/puzzles/imported_puzzles.csv",
        help="Target destination CSV file",
    )


def _validate_fetch_numerics(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "count", None) is not None and args.count < 1:
        return "--count must be >= 1."
    for attr in ("min_rating", "max_rating"):
        v = getattr(args, attr, None)
        if v is not None and not (200 <= v <= 3500):
            return f"--{attr.replace('_', '-')} must be in 200..3500."
    mn, mx = getattr(args, "min_rating", None), getattr(args, "max_rating", None)
    if mn is not None and mx is not None and mn > mx:
        return "--min-rating must be <= --max-rating."
    if getattr(args, "rating", None) is not None and not (200 <= args.rating <= 3500):
        return "--rating must be in 200..3500."
    return None


def route_add_puzzles(args: argparse.Namespace) -> int:
    action = getattr(args, "add_action", None)
    if action in ["manual", "add"]:
        return run_manual_add(args)
    elif action in ["list", "ls"]:
        return run_list_puzzles(args)
    elif action == "import":
        return run_import_puzzles(args)
    else:
        return run_fetch_puzzles(args)


def run_fetch_puzzles(args: argparse.Namespace) -> int:
    import time

    ctx = run_context("puzzles-fetch")
    t0 = time.time()
    log_path = "logs/zeropawn.log"

    def _fail(what: str, likely: str, try_: str) -> int:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1

    err = _validate_fetch_numerics(args)
    if err:
        return _fail(err, "argument out of range", "fix the flagged argument, then see LOG")
    theme = sanitize_theme(args.theme)

    emit_report(banner("puzzles-fetch", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu",
            "n/a (stream order)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("count", args.count, "user" if args.count != 1000 else "default"),
                ("min_rating", args.min_rating, "user" if args.min_rating != 1200 else "default"),
                ("max_rating", args.max_rating, "user" if args.max_rating != 1600 else "default"),
                (
                    "theme",
                    args.theme if args.theme else "none",
                    "user" if args.theme else "default",
                ),
                (
                    "output",
                    str(args.output) if args.output else "auto",
                    "user" if args.output else "default",
                ),
                ("append", args.append, "user" if args.append else "default"),
            ]
        )
    )

    try:
        if args.output:
            out_path = safe_resolve(args.output)
        else:
            theme_suffix = f"_{theme}" if theme else ""
            filename = f"lichess_puzzles_{args.count}_{args.min_rating}_{args.max_rating}{theme_suffix}.csv"
            out_path = safe_resolve(Path("data/puzzles") / filename)
    except (ValueError, OSError) as e:
        return _fail(
            f"Invalid --output path: {e}",
            "path escapes repo or is malformed",
            "pass a repo-relative CSV path, then see LOG",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    existing_rows = []
    header = [
        "PuzzleId",
        "FEN",
        "Moves",
        "Rating",
        "RatingDeviation",
        "Popularity",
        "NbPlays",
        "Themes",
        "GameUrl",
        "OpeningTags",
        "DailyDate",
    ]

    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            if file_header:
                header = file_header
            for r in reader:
                if r:
                    existing_ids.add(r[0])
                    existing_rows.append(r)
        if not args.append and len(existing_rows) >= args.count:
            emit_report(
                format_kv_line([("cached", f"{len(existing_rows)} (M)"), ("file", str(out_path))])
            )
            emit_report(
                event_line("CACHED", f"target file already has {len(existing_rows)} puzzles")
            )
            emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
            return 0

    emit_report(event_line("FETCH", f"connecting to {LICHESS_PUZZLE_URL}"))
    req = urllib.request.Request(LICHESS_PUZZLE_URL, headers={"User-Agent": "ZeroPawn/1.0"})

    collected_rows = []
    target_count = args.count

    try:
        # NOTE: single-attempt fetch; retry the command on transient network errors.
        with urllib.request.urlopen(req, timeout=30) as resp:
            dctx = zstandard.ZstdDecompressor()
            stream_reader = dctx.stream_reader(resp)
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
            reader = csv.reader(text_stream)
            _ = next(reader, header)

            for row in reader:
                if not row or len(row) < 4:
                    continue
                p_id = row[0]
                if p_id in existing_ids:
                    continue

                try:
                    rating = int(row[3])
                except ValueError:
                    continue

                if not (args.min_rating <= rating <= args.max_rating):
                    continue

                # Theme filter
                if theme:
                    row_themes = row[7].lower() if len(row) > 7 else ""
                    if theme.lower() not in row_themes:
                        continue

                collected_rows.append(row)
                existing_ids.add(p_id)
                if len(collected_rows) % 100 == 0:
                    emit_report(
                        format_kv_line(
                            [
                                ("collected", f"{len(collected_rows)}/{target_count}"),
                            ]
                        )
                    )

                if len(collected_rows) >= target_count:
                    break

    except KeyboardInterrupt:
        emit_report(
            event_line("INTERRUPTED", f"saving {len(collected_rows)} collected puzzles so far")
        )
    except (OSError, ValueError, EOFError, zstandard.ZstdError) as e:
        logger.debug("puzzle download failed: %s", e)
        return _fail(
            f"Download failed: {e}",
            "network or stream decode failure",
            "retry the command, then see LOG",
        )

    if not collected_rows and not existing_rows:
        emit_report(event_line("EMPTY", "no matching puzzles found"))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
        return 0

    all_to_write = existing_rows + collected_rows if args.append else collected_rows

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_to_write)

    emit_report(results_table([("saved", f"{len(all_to_write):,} puzzles in {out_path} (M)")]))
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    return 0


def run_manual_add(args: argparse.Namespace) -> int:
    import zeropawn_cpp

    print_banner("ADD CUSTOM PUZZLE", "Create and validate a new tactical puzzle")

    err = _validate_fetch_numerics(args)
    if err:
        print_error(err)
        return 1
    try:
        target_path = safe_resolve(args.target_file)
    except (ValueError, OSError) as e:
        print_error(f"Invalid --target-file path: {e}")
        return 1
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect inputs if not passed
    fen = args.fen
    if not fen:
        fen = console.input("[bold cyan]Enter FEN before opponent move > [/bold cyan]").strip()

    opp_move = args.opponent_move
    if not opp_move:
        opp_move = console.input(
            "[bold cyan]Enter opponent move (UCI, e.g. 'e2e4') > [/bold cyan]"
        ).strip()

    solution = args.solution
    if not solution:
        solution = console.input(
            "[bold cyan]Enter solution move(s) (UCI, e.g. 'e7e5') > [/bold cyan]"
        ).strip()

    rating = args.rating
    themes = args.themes
    p_id = args.id or f"usr_{uuid.uuid4().hex[:6]}"

    # Validate with zeropawn_cpp (FEN pre-validated)
    try:
        fen = validate_fen_or_raise(fen)
        game = zeropawn_cpp.ChessGame(fen)
    except (ValueError, RuntimeError) as e:
        logger.debug("invalid FEN: %s", e)
        print_error(f"Invalid FEN string: {e}")
        return 1

    if not game.is_legal(opp_move):
        print_error(f"Opponent move '{opp_move}' is illegal in position: {fen}")
        return 1

    game.make_move_uci(opp_move)
    first_sol_move = solution.split()[0]
    if not game.is_legal(first_sol_move):
        print_error(f"Solution move '{first_sol_move}' is illegal after '{opp_move}'!")
        return 1

    full_moves = f"{opp_move} {solution}"

    # Check if target file exists
    write_header = not target_path.exists()
    with open(target_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "PuzzleId",
                    "FEN",
                    "Moves",
                    "Rating",
                    "RatingDeviation",
                    "Popularity",
                    "NbPlays",
                    "Themes",
                    "GameUrl",
                    "OpeningTags",
                    "DailyDate",
                ]
            )
        writer.writerow(
            [
                p_id,
                fen,
                full_moves,
                rating,
                100,
                100,
                1,
                themes,
                "",
                "",
                "",
            ]
        )

    print_success(
        f"Added puzzle '{p_id}' (Rating: {rating}) to [bold green]{target_path}[/bold green]!"
    )
    return 0


def run_list_puzzles(args: argparse.Namespace) -> int:
    import time

    ctx = run_context("puzzles-list")
    t0 = time.time()

    try:
        puzzles_dir = safe_resolve(args.dir)
    except (ValueError, OSError) as e:
        emit_error(
            error_block(
                f"Invalid --dir path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative dir, then see LOG",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1
    emit_report(banner("puzzles-list", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu",
            "n/a",
        )
    )
    emit_report(
        parameters_table(
            [
                ("dir", str(args.dir), "user" if str(args.dir) != "data/puzzles" else "default"),
            ]
        )
    )

    if not puzzles_dir.is_dir():
        emit_error(
            error_block(
                f"Directory not found: {puzzles_dir}", "wrong --dir", "check --dir, then see LOG"
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1

    csv_files = list(puzzles_dir.glob("*.csv"))
    if not csv_files:
        emit_report(event_line("EMPTY", f"no CSV puzzle sets in {puzzles_dir}"))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
        return 0

    rows = []
    for f in csv_files:
        count = 0
        ratings = []
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reader = csv.reader(fp)
                next(reader, None)
                for r in reader:
                    if r and len(r) >= 4:
                        count += 1
                        try:
                            ratings.append(int(r[3]))
                        except ValueError:
                            pass
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            logger.debug("skipping unreadable puzzle set %s: %s", f, e)
            continue

        min_r = str(min(ratings)) if ratings else "N/A"
        max_r = str(max(ratings)) if ratings else "N/A"
        avg_r = f"{sum(ratings) / len(ratings):.0f}" if ratings else "N/A"
        size_kb = f"{f.stat().st_size / 1024:.1f} KB"

        rows.append(
            (f.name, f"{count:,} puzzles, rating {min_r}-{max_r} avg {avg_r}, {size_kb} (M)")
        )

    emit_report(results_table(rows))
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    return 0


def run_import_puzzles(args: argparse.Namespace) -> int:
    import time

    ctx = run_context("puzzles-import")
    t0 = time.time()
    log_path = "logs/zeropawn.log"

    def _fail(what: str, likely: str, try_: str) -> int:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1

    emit_report(banner("puzzles-import", ctx["start"] + " (local)"))
    try:
        src = safe_resolve(args.source_file)
        dst = safe_resolve(args.target_file)
    except (ValueError, OSError) as e:
        return _fail(
            f"Invalid file path: {e}",
            "path escapes repo or is malformed",
            "pass repo-relative CSV paths, then see LOG",
        )

    if not src.is_file():
        return _fail(
            f"Source file not found: {src}", "wrong --source-file", "check the path, then see LOG"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if dst.exists():
        with open(dst, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if r:
                    existing_ids.add(r[0])

    imported_count = 0
    with open(src, "r", encoding="utf-8") as sf, open(dst, "a", newline="", encoding="utf-8") as df:
        s_reader = csv.reader(sf)
        d_writer = csv.writer(df)
        if not dst.exists() or dst.stat().st_size == 0:
            header = next(s_reader, None)
            if header:
                d_writer.writerow(header)
        else:
            next(s_reader, None)

        for row in s_reader:
            if not row or row[0] in existing_ids:
                continue
            d_writer.writerow(row)
            existing_ids.add(row[0])
            imported_count += 1

    emit_report(results_table([("imported", f"{imported_count} new puzzles in {dst} (M)")]))
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    return 0
