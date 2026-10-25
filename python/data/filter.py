"""PGN filtering script for Lichess game databases.

Filter criteria:
1. Rated games only.
2. Both players rated >= 2200.
3. Normal terminations only (drops abandoned, rules infraction, time forfeit).
4. Time-control filter: base + 40 * increment >= 180 seconds.
   JUSTIFICATION:
   - Filters out Ultra-Bullet (30s) and 1-minute Bullet (<180s) where high-rated
     play is dominated by premove races, mouse slips, and scramble blunders.
   - Retains Blitz (3+0, 3+2, 5+3), Rapid (10+0, 15+10), and Classical games
     where 2200+ players calculate deep tactical and strategic variations.
5. Minimum length: >= 20 plies (drops short draws, move-1 resignations/disconnects).

Logs retention statistics across each filtering stage.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO
import urllib.request
import zipfile
import zstandard as zstd
import chess.pgn

logger = logging.getLogger("data.filter")

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.cli.common import setup_logging
from python.common.progress import (
    artifacts_list,
    banner,
    emit_error,
    emit_report,
    error_block,
    file_record,
    format_kv_line,
    header_block,
    parameters_table,
    run_context,
    summary_block,
)


TIME_CONTROL_JUSTIFICATION = (
    "Proposed time-control filter: estimated game length (base + 40 * increment) >= 180 seconds. "
    "Justification: Excludes Ultra-Bullet (30s) and fast Bullet (<180s) where moves are largely "
    "pre-moves and flag-induced scrambles with low tactical accuracy. Retains standard Blitz, "
    "Rapid, and Classical where 2200+ masters demonstrate genuine chess calculation and depth."
)


class FilterStats:
    def __init__(self) -> None:
        self.total_seen: int = 0
        self.passed_rated: int = 0
        self.passed_rating_2200: int = 0
        self.passed_normal_term: int = 0
        self.passed_time_control: int = 0
        self.passed_min_plies: int = 0
        self.total_kept: int = 0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("LICHESS FILTERING PIPELINE REPORT")
        print("=" * 60)
        print(f"1. Total games parsed:           {self.total_seen:>8}")
        pct_rated = (self.passed_rated / self.total_seen * 100) if self.total_seen else 0
        print(f"2. Rated games:                  {self.passed_rated:>8} ({pct_rated:5.1f}%)")
        pct_2200 = (self.passed_rating_2200 / self.total_seen * 100) if self.total_seen else 0
        print(f"3. Both players >= 2200:         {self.passed_rating_2200:>8} ({pct_2200:5.2f}%)")
        pct_term = (self.passed_normal_term / self.total_seen * 100) if self.total_seen else 0
        print(f"4. Normal termination:           {self.passed_normal_term:>8} ({pct_term:5.2f}%)")
        pct_tc = (self.passed_time_control / self.total_seen * 100) if self.total_seen else 0
        print(f"5. Time control (>=180s):        {self.passed_time_control:>8} ({pct_tc:5.2f}%)")
        pct_plies = (self.passed_min_plies / self.total_seen * 100) if self.total_seen else 0
        print(f"6. Min length (>=20 plies):      {self.passed_min_plies:>8} ({pct_plies:5.2f}%)")
        print("-" * 60)
        pct_kept = (self.total_kept / self.total_seen * 100) if self.total_seen else 0
        print(f"Total games kept:                {self.total_kept:>8} ({pct_kept:5.2f}%)")
        print("=" * 60 + "\n")


def parse_time_control(tc_str: str) -> Optional[int]:
    """Parse TimeControl header into estimated seconds (base + 40 * inc)."""
    if not tc_str or tc_str == "-":
        return None
    try:
        parts = tc_str.split("+")
        base = int(parts[0])
        inc = int(parts[1]) if len(parts) > 1 else 0
        return base + 40 * inc
    except (ValueError, IndexError):
        return None


def is_rated(headers: dict[str, str]) -> bool:
    """Check if game was rated."""
    event = headers.get("Event", "").lower()
    if "rated" in event:
        return True
    if headers.get("Rated", "").lower() in ("true", "yes", "rated"):
        return True
    return False


def is_rating_above_threshold(headers: dict[str, str], threshold: int = 2200) -> bool:
    """Check if both players are rated >= threshold."""
    try:
        w_elo = int(headers.get("WhiteElo", "0"))
        b_elo = int(headers.get("BlackElo", "0"))
        return w_elo >= threshold and b_elo >= threshold
    except ValueError:
        return False


def is_normal_termination(headers: dict[str, str]) -> bool:
    """Check for normal game termination (exclude abandoned, forfeit, etc.)."""
    term = headers.get("Termination", "").strip()
    return term.lower() == "normal"


def check_time_control(headers: dict[str, str], min_seconds: int = 180) -> bool:
    """Check if game time control meets the minimum estimated duration threshold."""
    tc = headers.get("TimeControl", "")
    est_sec = parse_time_control(tc)
    if est_sec is None:
        return False
    return est_sec >= min_seconds


def count_plies_in_game(game: chess.pgn.Game) -> int:
    """Count number of plies (half-moves) in a parsed chess.pgn.Game."""
    node = game
    count = 0
    while node.variations:
        node = node.variation(0)
        count += 1
    return count


def stream_pgn_raw_games(stream: TextIO) -> Iterator[str]:
    """Yield raw PGN game text blocks from stream."""
    lines: list[str] = []
    in_game = False

    for line in stream:
        if line.startswith("[Event "):
            if lines and in_game:
                yield "".join(lines)
                lines = []
            in_game = True
        if in_game:
            lines.append(line)

    if lines and in_game:
        yield "".join(lines)


def filter_games(
    source_stream: TextIO,
    stats: FilterStats,
    min_rating: int = 2200,
    min_plies: int = 20,
    min_tc_seconds: int = 180,
    max_kept: Optional[int] = None,
) -> Iterator[chess.pgn.Game]:
    """Filter raw PGN games yielding parsed chess.pgn.Game objects."""
    for raw_game_str in stream_pgn_raw_games(source_stream):
        stats.total_seen += 1

        # Fast header extraction from raw lines before full parsing
        headers: dict[str, str] = {}
        for line in raw_game_str.splitlines():
            if not line.startswith("["):
                break
            line_str = line.strip("[] \r\n")
            if " " in line_str:
                k, v = line_str.split(" ", 1)
                headers[k] = v.strip('"')

        # Filter step 1: Rated only
        if not is_rated(headers):
            continue
        stats.passed_rated += 1

        # Filter step 2: Both ratings >= min_rating
        if not is_rating_above_threshold(headers, min_rating):
            continue
        stats.passed_rating_2200 += 1

        # Filter step 3: Normal termination
        if not is_normal_termination(headers):
            continue
        stats.passed_normal_term += 1

        # Filter step 4: Time control filter
        if not check_time_control(headers, min_tc_seconds):
            continue
        stats.passed_time_control += 1

        # Filter step 5: Minimum length in plies
        game = chess.pgn.read_game(io.StringIO(raw_game_str))
        if game is None:
            continue
        plies = count_plies_in_game(game)
        if plies < min_plies:
            continue
        stats.passed_min_plies += 1

        stats.total_kept += 1
        yield game

        if max_kept is not None and stats.total_kept >= max_kept:
            break


import gc
import shutil


class _ClosingTextIOWrapper(io.TextIOWrapper):
    """TextIOWrapper that also closes the underlying file/buffer when closed."""

    def __init__(self, buffer: Any, underlying_file: Any = None, **kwargs: Any) -> None:
        super().__init__(buffer, **kwargs)
        self._underlying_file = underlying_file

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._underlying_file is not None and hasattr(self._underlying_file, "close"):
                try:
                    self._underlying_file.close()
                except Exception:
                    pass


def open_input_stream(input_path_or_url: str) -> TextIO:
    """Open a stream from URL, .pgn.zst file, or standard .pgn file."""
    if input_path_or_url.startswith("http://") or input_path_or_url.startswith("https://"):
        req = urllib.request.Request(input_path_or_url, headers={"User-Agent": "ZeroPawn/1.0"})
        resp = urllib.request.urlopen(req, timeout=60.0)
        if input_path_or_url.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(resp)
            return _ClosingTextIOWrapper(
                stream_reader, underlying_file=resp, encoding="utf-8", errors="replace"
            )
        return _ClosingTextIOWrapper(resp, underlying_file=resp, encoding="utf-8", errors="replace")

    path = Path(input_path_or_url)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".zst":
        f = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        return _ClosingTextIOWrapper(
            stream_reader, underlying_file=f, encoding="utf-8", errors="replace"
        )

    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path, "r")
        pgn_names = [n for n in zf.namelist() if n.endswith(".pgn")]
        if not pgn_names:
            raise ValueError(f"No .pgn file found inside zip archive: {path}")
        f = zf.open(pgn_names[0], "r")
        return _ClosingTextIOWrapper(f, underlying_file=zf, encoding="utf-8", errors="replace")

    return open(path, "r", encoding="utf-8", errors="replace")


def run_filter(
    input_source: str,
    output_pgn_path: str,
    min_rating: int = 2200,
    min_plies: int = 20,
    min_tc_seconds: int = 180,
    max_kept: Optional[int] = None,
    log_interval: int = 1000,
) -> FilterStats:
    """Run full filtering pipeline and save matching games to output PGN."""
    print("=" * 60)
    print("LICHESS 2200+ FILTER RUNNER")
    print(f"Source: {input_source}")
    print(f"Output: {output_pgn_path}")
    print(f"Min rating: {min_rating}")
    print(f"Min plies:  {min_plies}")
    print(f"Min TC sec: {min_tc_seconds}s")
    print(TIME_CONTROL_JUSTIFICATION)
    print("=" * 60)

    stats = FilterStats()
    out_path = Path(output_pgn_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check free disk space before filtering
    try:
        free_bytes = shutil.disk_usage(out_path.parent).free
        if free_bytes < 100 * 1024 * 1024:
            raise OSError(
                f"Insufficient disk space ({free_bytes / 1e6:.1f} MB available, need >= 100 MB)"
            )
    except Exception as e:
        logger.debug("Disk usage check warning: %s", e)

    tmp_out_path = out_path.with_suffix(".tmp")
    stream = open_input_stream(input_source)
    try:
        with open(tmp_out_path, "w", encoding="utf-8") as out_file:
            exporter = chess.pgn.FileExporter(out_file)
            for game in filter_games(
                source_stream=stream,
                stats=stats,
                min_rating=min_rating,
                min_plies=min_plies,
                min_tc_seconds=min_tc_seconds,
                max_kept=max_kept,
            ):
                game.accept(exporter)
                if stats.total_seen % 5000 == 0:
                    gc.collect(1)
                if stats.total_kept % log_interval == 0:
                    emit_report(
                        format_kv_line(
                            [
                                ("kept", stats.total_kept),
                                ("examined", stats.total_seen),
                            ]
                        )
                    )

        # Atomic replacement on success
        if tmp_out_path.exists():
            tmp_out_path.replace(out_path)
    except Exception:
        if tmp_out_path.exists():
            try:
                tmp_out_path.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            stream.close()
        except Exception:
            pass

    stats.print_summary()
    return stats


def main() -> None:
    setup_logging(log_dir="logs", log_filename="filter.log")
    ctx = run_context("data-filter")
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Filter Lichess PGN databases.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to .pgn, .pgn.zst, or Lichess URL."
    )
    parser.add_argument(
        "--output", type=str, default="data/filtered_2200.pgn", help="Output PGN file."
    )
    parser.add_argument(
        "--min-rating",
        "--min-elo",
        dest="min_rating",
        type=int,
        default=2200,
        help="Minimum rating for both players.",
    )
    parser.add_argument("--min-plies", type=int, default=20, help="Minimum half-moves in game.")
    parser.add_argument(
        "--min-tc", type=int, default=180, help="Minimum estimated seconds (base + 40*inc)."
    )
    parser.add_argument(
        "--max-kept", type=int, default=None, help="Stop after keeping this many games."
    )
    parser.add_argument(
        "--log-interval", type=int, default=1000, help="Log progress every N kept games."
    )

    args = parser.parse_args()

    emit_report(banner("data-filter", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu",
            "n/a (input order)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("input", args.input, "user"),
                (
                    "output",
                    args.output,
                    "user" if args.output != "data/filtered_2200.pgn" else "default",
                ),
                ("min_rating", args.min_rating, "user" if args.min_rating != 2200 else "default"),
                ("min_plies", args.min_plies, "user" if args.min_plies != 20 else "default"),
                ("min_tc", args.min_tc, "user" if args.min_tc != 180 else "default"),
                ("max_kept", args.max_kept, "user" if args.max_kept is not None else "default"),
                (
                    "log_interval",
                    args.log_interval,
                    "user" if args.log_interval != 1000 else "default",
                ),
            ]
        )
    )

    try:
        run_filter(
            input_source=args.input,
            output_pgn_path=args.output,
            min_rating=args.min_rating,
            min_plies=args.min_plies,
            min_tc_seconds=args.min_tc,
            max_kept=args.max_kept,
            log_interval=args.log_interval,
        )
        emit_report(artifacts_list([file_record(args.output)]))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except Exception as e:
        emit_error(
            error_block(
                f"data-filter failed: {e}",
                "unreadable input PGN/URL or unwritable output path",
                "check --input exists and --output directory is writable, then see LOG",
                "logs/filter.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
