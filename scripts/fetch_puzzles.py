"""Fetch and freeze a fixed set of 1,000 Lichess puzzles rated 1200-1600.

Saves directly to data/puzzles/lichess_puzzles_1000_1200_1600.csv.
The set is deterministic and must never be changed once created.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path
import zstandard

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
OUTPUT_PATH = Path("data/puzzles/lichess_puzzles_1000_1200_1600.csv")
TARGET_COUNT = 1000
MIN_RATING = 1200
MAX_RATING = 1600


def fetch_puzzles(
    output_path: Path = OUTPUT_PATH,
    target_count: int = TARGET_COUNT,
    min_rating: int = MIN_RATING,
    max_rating: int = MAX_RATING,
) -> Path:
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            count = sum(1 for _ in reader)
        if count == target_count:
            print(f"Puzzle file already exists with {count} puzzles at {output_path}")
            return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Streaming from {PUZZLE_URL} to select {target_count} puzzles (ratings {min_rating}-{max_rating})..."
    )

    req = urllib.request.Request(PUZZLE_URL, headers={"User-Agent": "ZeroPawn/1.0"})
    selected_rows = []

    with urllib.request.urlopen(req) as resp:
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(resp)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        reader = csv.reader(text_stream)
        header = next(reader)

        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                rating = int(row[3])
            except ValueError:
                continue

            if min_rating <= rating <= max_rating:
                selected_rows.append(row)
                if len(selected_rows) % 100 == 0:
                    print(f"Collected {len(selected_rows)}/{target_count} puzzles...")
                if len(selected_rows) >= target_count:
                    break

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(selected_rows)

    # Verify the serialized output row count before publishing (audit F-07:
    # a short write must never appear as a complete frozen set).
    with open(tmp_path, "r", encoding="utf-8") as f:
        written = max(0, sum(1 for _ in f) - 1)
    if written != len(selected_rows):
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Puzzle CSV write verification failed: {written} rows on disk, "
            f"{len(selected_rows)} selected. Nothing was published."
        )
    tmp_path.replace(output_path)

    import hashlib
    import json
    from datetime import datetime, timezone

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    Path(str(output_path) + ".provenance.json").write_text(
        json.dumps(
            {
                "url": PUZZLE_URL,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "target_count": target_count,
                "rating_band": [min_rating, max_rating],
                "rows_written": len(selected_rows),
                "sha256": digest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Successfully saved {len(selected_rows)} fixed puzzles to {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_puzzles()
