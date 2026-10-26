"""Check that an evaluation puzzle set is disjoint from tactics training data.

The RL loop reports a puzzle solve rate as a generalisation signal and the
puzzle-decline safety trigger stops the run on it. Both are only meaningful if the
evaluation puzzles never appear in training data. Lichess puzzle CSVs share one
database, so an eval set drawn from it will silently overlap any tactics training
set unless it is checked.

This tool compares every evaluation puzzle against one or more training CSVs by
Lichess PuzzleId and by position key (board, side to move, castling rights and
en-passant square of the FEN) and reports the contamination. With ``--output`` it
writes a certified eval CSV containing only non-overlapping puzzles plus a
``.certified.json`` manifest recording provenance and the measured overlap.

Usage:
    # Report only
    python scripts/check_puzzle_overlap.py \
        data/puzzles/lichess_puzzles_1000_1200_1600.csv data/tactics_train_300k.csv

    # Build a certified, contamination-free eval set from a larger pool
    python scripts/check_puzzle_overlap.py \
        --eval-csv data/puzzles/lichess_puzzles_50000_1200_3000.csv \
        --train-csv data/tactics_train_300k.csv \
        --output data/puzzles/eval_certified_1200_1800.csv \
        --max-puzzles 1000 --min-rating 1200 --max-rating 1800
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file() and not p.is_absolute():
        candidate = _REPO_ROOT / p
        if candidate.is_file():
            return candidate
    return p


def position_key(fen: str) -> str:
    """Position identity: piece placement, side to move, castling, en passant.

    Halfmove clock and fullmove number are excluded because they do not change the
    position's identity for training/evaluation contamination purposes.
    """
    fields = fen.strip().split()
    if len(fields) < 4:
        return fen.strip()
    placement, side, castling, ep = fields[0], fields[1], fields[2], fields[3]
    if ep == "-":
        ep = ""
    return f"{placement} {side} {castling} {ep}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r and r.get("FEN")]
    if not rows:
        print(f"WARNING: no usable rows in {path}", file=sys.stderr)
    return rows


def load_overlap_keys(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    fens: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            if row.get("PuzzleId"):
                ids.add(row["PuzzleId"])
            fens.add(position_key(row["FEN"]))
    return ids, fens


def evaluate(
    eval_rows: Sequence[dict[str, str]], train_ids: set[str], train_fens: set[str]
) -> dict[str, int]:
    id_hits = 0
    fen_hits = 0
    either = 0
    for row in eval_rows:
        by_id = bool(row.get("PuzzleId")) and row["PuzzleId"] in train_ids
        by_fen = position_key(row["FEN"]) in train_fens
        id_hits += int(by_id)
        fen_hits += int(by_fen)
        either += int(by_id or by_fen)
    return {"total": len(eval_rows), "id": id_hits, "fen": fen_hits, "either": either}


def certified_subset(
    eval_rows: Sequence[dict[str, str]],
    train_ids: set[str],
    train_fens: set[str],
    max_puzzles: int | None,
    min_rating: int | None,
    max_rating: int | None,
) -> list[dict[str, str]]:
    """Non-overlapping puzzles, deterministically sampled from the rating band."""
    candidates: list[dict[str, str]] = []
    for row in eval_rows:
        if row.get("PuzzleId") and row["PuzzleId"] in train_ids:
            continue
        if position_key(row["FEN"]) in train_fens:
            continue
        try:
            rating = int(float(row.get("Rating", "")))
        except (TypeError, ValueError):
            continue
        if min_rating is not None and rating < min_rating:
            continue
        if max_rating is not None and rating > max_rating:
            continue
        candidates.append(row)

    # Deterministic order: by rating then puzzle id, so the selection is reproducible.
    candidates.sort(key=lambda r: (int(float(r["Rating"])), r["PuzzleId"]))
    if max_puzzles is not None and len(candidates) > max_puzzles:
        step = len(candidates) / float(max_puzzles)
        candidates = [candidates[int(i * step)] for i in range(max_puzzles)]
    return candidates


def write_certified(
    rows: Sequence[dict[str, str]],
    output: Path,
    provenance: dict,
    fieldnames: Sequence[str],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    manifest = output.with_name(output.name + ".certified.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check eval puzzle CSV(s) for contamination against tactics training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "paths",
        nargs="*",
        help="[EVAL_CSV TRAIN_CSV ...] positional form; the first path is the eval set",
    )
    p.add_argument("--eval-csv", action="append", default=[], help="Evaluation puzzle CSV")
    p.add_argument(
        "--train-csv",
        action="append",
        default=[],
        help="Tactics training CSV to check against (repeatable)",
    )
    p.add_argument("--output", default=None, help="Write a certified disjoint eval CSV here")
    p.add_argument("--max-puzzles", type=int, default=None, help="Cap the certified set size")
    p.add_argument("--min-rating", type=int, default=None, help="Minimum puzzle rating")
    p.add_argument("--max-rating", type=int, default=None, help="Maximum puzzle rating")
    p.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Exit non-zero if any eval puzzle appears in the training data",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    eval_paths = [_resolve(p) for p in args.eval_csv]
    train_paths = [_resolve(p) for p in args.train_csv]
    if args.paths:
        positional = [_resolve(p) for p in args.paths]
        if not eval_paths:
            eval_paths = [positional[0]]
        train_paths.extend(positional[1:])

    if not eval_paths or not train_paths:
        print(
            "ERROR: need at least one eval CSV and one train CSV "
            "(positional: EVAL TRAIN..., or --eval-csv/--train-csv)",
            file=sys.stderr,
        )
        return 2

    for path in [*eval_paths, *train_paths]:
        if not path.is_file():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2

    eval_rows: list[dict[str, str]] = []
    for path in eval_paths:
        eval_rows.extend(read_rows(path))
    train_ids, train_fens = load_overlap_keys(train_paths)

    stats = evaluate(eval_rows, train_ids, train_fens)
    pct = (100.0 * stats["either"] / stats["total"]) if stats["total"] else 0.0
    print("=" * 68)
    print("PUZZLE EVAL CONTAMINATION REPORT")
    print("=" * 68)
    print(f"  eval files   : {', '.join(str(p) for p in eval_paths)}")
    print(f"  train files  : {', '.join(str(p) for p in train_paths)}")
    print(f"  eval puzzles : {stats['total']}")
    print(f"  overlap by PuzzleId : {stats['id']}")
    print(f"  overlap by FEN      : {stats['fen']}")
    print(f"  overlap (either)    : {stats['either']}  ({pct:.2f}%)")
    print("=" * 68)

    if args.output:
        subset = certified_subset(
            eval_rows,
            train_ids,
            train_fens,
            max_puzzles=args.max_puzzles,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
        )
        if not subset:
            print("ERROR: no disjoint puzzles available for the requested band", file=sys.stderr)
            return 1
        fieldnames = list(eval_rows[0].keys())
        provenance = {
            "certified": True,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "scripts/check_puzzle_overlap.py",
            "eval_sources": [{"path": str(p), "sha256": _sha256(p)} for p in eval_paths],
            "train_sources": [{"path": str(p), "sha256": _sha256(p)} for p in train_paths],
            "candidate_puzzles": stats["total"],
            "overlap_by_id": stats["id"],
            "overlap_by_fen": stats["fen"],
            "overlap_either": stats["either"],
            "output_puzzles": len(subset),
            "min_rating": args.min_rating,
            "max_rating": args.max_rating,
            "overlap_in_output": 0,
        }
        output = _resolve(args.output)
        manifest = write_certified(subset, output, provenance, fieldnames)
        print(f"Certified {len(subset)} puzzles -> {output}")
        print(f"Manifest                     -> {manifest}")

    if args.fail_on_overlap and stats["either"] > 0:
        print(
            f"FAIL: {stats['either']} of {stats['total']} eval puzzles appear in training data",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
