"""Build the 2500+ training PGN from the Lichess Elite Database.

Why elite (database.nikonoel.fr) instead of raw monthly DBs:
  - raw standard month (2026-07) = ~29 GB compressed; elite month ~= 85 MB
    (pre-filtered: both players high-rated, bullet already excluded).
  - this script then applies OUR bar on top: both >= 2500, TC >= 600s
    (drops ALL blitz: 5+3=420s, 3+2=260s; keeps 10+0 rapid and slower),
    normal termination, >= 20 plies. TC "-" (correspondence/unknown) is
    dropped: quality unverifiable, engine-assistance risk.

Usage:
    python scripts/fetch_elite.py --months 2026-06 2026-05 2026-04 \\
        --output data/elite_2500.pgn --workdir data/elite_tmp
    python scripts/fetch_elite.py --last-n 12 --output data/elite_2500.pgn
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from python.data.filter import FilterStats, filter_games

ELITE_URL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"


def download(url: str, dest: Path) -> Path | None:
    """Download, or return None if the server serves an HTML stub (month missing)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists, skipping: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    # Pre-flight disk space check: ensure at least 1 GB free
    free_bytes = shutil.disk_usage(dest.parent).free
    if free_bytes < 1_000_000_000:
        raise OSError(
            f"Insufficient disk space in {dest.parent} ({free_bytes / 1e9:.2f} GB free, need >= 1.0 GB)"
        )

    # probe: elite DB serves HTML 200-stubs for months it doesn't have
    try:
        req0 = urllib.request.Request(url, headers={"User-Agent": "ZeroPawn/1.0"}, method="HEAD")
        with urllib.request.urlopen(req0, timeout=30) as x0:
            if "text/html" in (x0.headers.get("Content-Type") or ""):
                print(f"  SKIP {url.rsplit('/', 1)[-1]}: no data on server (HTML stub)")
                return None
    except Exception as e:
        print(f"  probe failed ({str(e)[:80]}), will try GET anyway")

    part_dest = dest.with_name(f"{dest.name}.part")
    print(f"  downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ZeroPawn/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(part_dest, "wb") as f:
            shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
        part_dest.replace(dest)
    except Exception:
        part_dest.unlink(missing_ok=True)
        raise

    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def fetch_elite(
    months: list[str],
    output_pgn: Path,
    workdir: Path,
    min_rating: int = 2500,
    min_tc: int = 600,
    keep_downloads: bool = False,
    max_games: int | None = None,
) -> FilterStats:
    import io
    import chess.pgn

    workdir.mkdir(parents=True, exist_ok=True)
    output_pgn.parent.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(output_pgn.parent).free
    if free_bytes < 1_000_000_000:
        raise OSError(
            f"Insufficient disk space in {output_pgn.parent} ({free_bytes / 1e9:.2f} GB free, need >= 1.0 GB)"
        )

    stats = FilterStats()
    total_kept = 0
    tmp_output = output_pgn.with_name(f"{output_pgn.name}.tmp")
    out = open(tmp_output, "w", encoding="utf-8")
    try:
        exporter = chess.pgn.FileExporter(out)
        for month in months:
            zpath = download(ELITE_URL.format(month=month), workdir / f"elite_{month}.zip")
            if zpath is None:
                continue
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if n.endswith(".pgn")]
                if not names:
                    print(f"  WARNING: no .pgn inside {zpath.name}: {zf.namelist()[:5]}")
                    continue
                with zf.open(names[0]) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    for game in filter_games(
                        text, stats, min_rating=min_rating, min_plies=20, min_tc_seconds=min_tc
                    ):
                        game.accept(exporter)
                        total_kept += 1
                        if total_kept % 1000 == 0:
                            print(f"  ... kept {total_kept} games so far")
                        if max_games and total_kept >= max_games:
                            break
            print(f"  month {month} done (cumulative kept {total_kept})")
            if not keep_downloads:
                zpath.unlink(missing_ok=True)
            if max_games and total_kept >= max_games:
                print(f"  Reached target limit of {max_games} games.")
                break
        out.close()
        tmp_output.replace(output_pgn)
    except Exception:
        out.close()
        tmp_output.unlink(missing_ok=True)
        raise
    print(f"\nWrote {total_kept} games to {output_pgn}")
    stats.print_summary()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2500+ PGN from elite DB.")
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to fetch across all processed months.",
    )
    parser.add_argument(
        "--months", nargs="*", default=[], help="Explicit months YYYY-MM. If empty, use --last-n."
    )
    parser.add_argument(
        "--last-n", type=int, default=0, help="Use the N most recent months (needs hardcoded list)."
    )
    parser.add_argument("--output", type=str, default="data/elite_2500.pgn")
    parser.add_argument("--workdir", type=str, default="data/elite_tmp")
    parser.add_argument("--min-rating", type=int, default=2500)
    parser.add_argument("--min-tc", type=int, default=600)
    parser.add_argument("--keep-downloads", action="store_true", default=False)
    args = parser.parse_args()

    months = list(args.months)
    last_n = args.last_n
    if not months and not last_n:
        last_n = 3  # Default to recent 3 months if none specified

    if not months and last_n:
        # probe backwards from a recent anchor until last_n valid months found
        import urllib.request as _ur

        y, m = 2026, 7
        while len(months) < last_n and (y > 2024 or (y == 2024 and m >= 1)):
            ym = f"{y:04d}-{m:02d}"
            u = ELITE_URL.format(month=ym)
            try:
                req0 = _ur.Request(u, headers={"User-Agent": "ZeroPawn/1.0"}, method="HEAD")
                with _ur.urlopen(req0, timeout=15) as x0:
                    if "zip" in (x0.headers.get("Content-Type") or ""):
                        months.append(ym)
            except Exception:
                pass
            m -= 1
            if m == 0:
                m, y = 12, y - 1
    if not months:
        parser.error("give --months or --last-n")
    fetch_elite(
        months,
        Path(args.output),
        Path(args.workdir),
        args.min_rating,
        args.min_tc,
        args.keep_downloads,
        max_games=args.max_games,
    )


if __name__ == "__main__":
    main()
