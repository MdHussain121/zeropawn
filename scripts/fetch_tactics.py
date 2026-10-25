"""Fetch a training-grade tactics set from the Lichess puzzle DB.

Unlike fetch_puzzles.py (first-N-wins, eval benchmark), this builds a LARGE,
quality-gated, rating-spread training set in ONE streaming pass using quotas:

- Quality gates: RatingDeviation <= 100 (stable rating), Popularity >= 50
  (community-approved, filters engine-artifact junk), NbPlays >= 30.
- Rating spread quotas: uniform-ish coverage 1100-2400 so the net learns
  basic motifs AND hard tactics (NOT just the 1200-1600 eval band, which would
  leak the benchmark into training).
- Theme caps: no single theme (e.g. mateIn1) may exceed its share, forcing
  motif diversity (pins, forks, skewers, deflections, endgames...).
- Dedupe by PuzzleId.

Output CSV keeps the official header:
PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,DailyDate

Usage:
    python scripts/fetch_tactics.py --target 300000 --min-rating 1100 --max-rating 2400 \\
        --output data/puzzles/tactics_train_300k.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import zstandard

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

# rating bins (inclusive lower bound) with default quotas summing to 1.0
DEFAULT_BINS = [
    (1100, 1400, 0.15),  # basic motifs
    (1400, 1700, 0.25),  # core tactics (engine's current weakness band)
    (1700, 2000, 0.30),  # stretch band (live failures live here)
    (2000, 2200, 0.20),  # hard tactics
    (2200, 2401, 0.10),  # expert (rare, high value per position)
]

# a single theme may claim at most this fraction of the total set
MAX_THEME_SHARE = 0.12


def bin_index(rating: int, lo: int, hi: int) -> int | None:
    for i, (blo, bhi, _share) in enumerate(DEFAULT_BINS):
        _ = (lo, hi)
        if blo <= rating < bhi:
            return i
    return None


def fetch_tactics(
    output_path: Path,
    target_count: int = 300000,
    min_rating: int = 1100,
    max_rating: int = 2400,
    min_popularity: int = 50,
    min_plays: int = 30,
    max_rd: int = 100,
    exclude_csvs: list[Path] | None = None,
) -> Path:
    # EVAL-SET HYGIENE: any puzzle id appearing in an eval CSV must never
    # enter training (a 72% overlap once inflated solve-rate by +17pp).
    exclude: set = set()
    for p in exclude_csvs or []:
        try:
            with open(p, encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row and row[0] != "PuzzleId":
                        exclude.add(row[0])
            print(f"Excluding eval-set puzzles from {p.name}: {len(exclude)} ids so far")
        except OSError as e:
            print(f"WARNING: cannot read exclusion CSV {p}: {e}")
    quotas = [int(target_count * share) for _, _, share in DEFAULT_BINS]
    quotas[-1] += target_count - sum(quotas)  # fix rounding
    filled = [0] * len(quotas)
    theme_counts: Counter = Counter()
    seen: set = set()
    selected: list = []
    total_needed = sum(quotas)
    examined = 0
    kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {PUZZLE_URL}")
    print(
        f"Target {target_count} tactics ({min_rating}-{max_rating}), "
        f"pop>={min_popularity}, plays>={min_plays}, RD<={max_rd}"
    )

    req = urllib.request.Request(PUZZLE_URL, headers={"User-Agent": "ZeroPawn/1.0"})
    with urllib.request.urlopen(req) as resp:
        dctx = zstandard.ZstdDecompressor()
        reader = csv.reader(io.TextIOWrapper(dctx.stream_reader(resp), encoding="utf-8"))
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}

        for row in reader:
            if not row or len(row) < len(header):
                continue
            examined += 1
            try:
                pid = row[idx["PuzzleId"]]
                rating = int(row[idx["Rating"]])
                rd = int(row[idx["RatingDeviation"]])
                pop = int(row[idx["Popularity"]])
                plays = int(row[idx["NbPlays"]])
                themes = row[idx["Themes"]].split()
            except (ValueError, KeyError, IndexError):
                continue
            if pid in seen or pid in exclude:
                continue
            if not (min_rating <= rating <= max_rating):
                continue
            if rd > max_rd or pop < min_popularity or plays < min_plays:
                continue
            b = bin_index(rating, min_rating, max_rating)
            if b is None or filled[b] >= quotas[b]:
                continue
            # theme diversity cap (checked against dominant theme)
            if themes:
                top = max(themes, key=lambda t: theme_counts[t])
                if theme_counts[top] >= int(target_count * MAX_THEME_SHARE) and len(themes) == 1:
                    continue
            seen.add(pid)
            selected.append(row)
            filled[b] += 1
            for t in themes:
                theme_counts[t] += 1
            kept += 1
            if kept % 20000 == 0:
                print(f"  kept {kept}/{total_needed} (examined {examined}) bins={filled}")
            if kept >= total_needed:
                break

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(selected)

    print(f"Saved {len(selected)} tactics to {output_path} (examined {examined})")
    print(f"bin fill: {filled} / quotas {quotas}")
    print(f"top themes: {theme_counts.most_common(12)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch quality-gated tactics set.")
    parser.add_argument("--target", type=int, default=300000)
    parser.add_argument("--min-rating", type=int, default=1100)
    parser.add_argument("--max-rating", type=int, default=2400)
    parser.add_argument("--min-popularity", type=int, default=50)
    parser.add_argument("--min-plays", type=int, default=30)
    parser.add_argument("--max-rd", type=int, default=100)
    parser.add_argument(
        "--exclude-csvs",
        nargs="*",
        type=Path,
        default=[
            Path("data/puzzles/lichess_puzzles_5000_1000_1600.csv"),
            Path("data/puzzles/lichess_puzzles_5000_1800_2800.csv"),
        ],
        help="Eval CSVs whose puzzle IDs must NOT appear in training.",
    )
    parser.add_argument("--output", type=str, default="data/puzzles/tactics_train_300k.csv")
    args = parser.parse_args()
    fetch_tactics(
        Path(args.output),
        args.target,
        args.min_rating,
        args.max_rating,
        args.min_popularity,
        args.min_plays,
        args.max_rd,
        exclude_csvs=args.exclude_csvs,
    )


if __name__ == "__main__":
    main()
