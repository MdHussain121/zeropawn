"""Stream and download ~750,000 high-quality Lichess puzzles to build a 2M position tactical dataset."""

from __future__ import annotations

import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

import zstandard

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def fetch_tactics_stream(
    output_path: Path,
    target_count: int = 750_000,
    min_rating: int = 1000,
    max_rating: int = 2800,
    min_popularity: int = 40,
    min_plays: int = 20,
    max_rd: int = 105,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Lichess Puzzle Stream: {LICHESS_PUZZLE_URL}")
    print(
        f"Targeting: {target_count:,} quality puzzles (Rating {min_rating}-{max_rating}, Pop>={min_popularity}, Plays>={min_plays})"
    )

    t0 = time.time()
    req = urllib.request.Request(LICHESS_PUZZLE_URL, headers={"User-Agent": "ZeroPawn/1.0"})

    kept = 0
    examined = 0

    with urllib.request.urlopen(req, timeout=60) as resp:
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(resp)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        reader = csv.reader(text_stream)

        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}

        with open(output_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)

            for row in reader:
                if not row or len(row) < len(header):
                    continue
                examined += 1

                try:
                    rating = int(row[idx["Rating"]])
                    rd = int(row[idx["RatingDeviation"]])
                    pop = int(row[idx["Popularity"]])
                    plays = int(row[idx["NbPlays"]])
                except (ValueError, KeyError, IndexError):
                    continue

                if not (min_rating <= rating <= max_rating):
                    continue
                if rd > max_rd or pop < min_popularity or plays < min_plays:
                    continue

                writer.writerow(row)
                kept += 1

                if kept % 50_000 == 0:
                    elapsed = time.time() - t0
                    rate = kept / max(0.1, elapsed)
                    print(
                        f"[{elapsed:5.1f}s] Collected {kept:,} / {target_count:,} puzzles ({rate:.0f} puzzles/sec, examined {examined:,})"
                    )

                if kept >= target_count:
                    break

    elapsed = time.time() - t0
    print(
        f"\nDone! Saved {kept:,} puzzles to {output_path} in {elapsed:.1f}s ({kept / max(0.1, elapsed):.0f} puzzles/sec)."
    )
    return kept


if __name__ == "__main__":
    out = Path("data/puzzles/lichess_puzzles_750k.csv")
    fetch_tactics_stream(out, target_count=750_000)
