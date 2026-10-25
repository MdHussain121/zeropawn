"""Convert a tactics CSV (PuzzleId,FEN,Moves,Rating,...) to binary training bins.

Each puzzle becomes ONE training position:
  - board = FEN with the opponent move (Moves[0]) applied  (player to move)
  - policy target = solution move (Moves[1])
  - outcome = draw placeholder, ply = 0  -> the supervised trainer's
    `val_mask = (plies >= 10)` rule masks the value loss, so tactics rows
    train the POLICY head only. No pipeline changes needed.

Uses the same 98/2 game-hash split + val dedup as build_dataset.py, so
`--train-dir/--val-dir` point straight at the output.

Usage:
    python python/data/build_tactics.py --csv data/puzzles/tactics_train_300k.csv \\
        --output-dir data/dataset_tactics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp
from python.data.build_dataset import (
    OUTCOME_UNAVAILABLE,
    PositionData,
    hash_game_id,
    save_dataset_split,
    _flush_positions_chunk,
)


def build_tactics_dataset(
    csv_path: str | Path,
    output_dir: str | Path,
    max_solver_plies: int = 4,
    chunk_size: int = 10_000,
) -> tuple[int, int]:
    import json

    out_dir = Path(output_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_part_paths = {
        "planes": train_dir / "planes.bin.part",
        "moves": train_dir / "moves.bin.part",
        "outcomes": train_dir / "outcomes.bin.part",
        "game_ids": train_dir / "game_ids.bin.part",
        "plies": train_dir / "plies.bin.part",
    }
    train_final_paths = {k: train_dir / f"{k}.bin" for k in train_part_paths}

    train_chunk: list[PositionData] = []
    candidate_val: list[PositionData] = []
    seen_pids: set = set()
    train_hashes: set[int] = set()
    n_train = 0
    skipped = 0
    total = 0

    with (
        open(train_part_paths["planes"], "wb") as f_tr_planes,
        open(train_part_paths["moves"], "wb") as f_tr_moves,
        open(train_part_paths["outcomes"], "wb") as f_tr_outcomes,
        open(train_part_paths["game_ids"], "wb") as f_tr_game_ids,
        open(train_part_paths["plies"], "wb") as f_tr_plies,
    ):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                try:
                    pid = row["PuzzleId"]
                    fen = row["FEN"]
                    moves = row["Moves"].split()
                    if len(moves) < 2 or pid in seen_pids:
                        skipped += 1
                        continue
                    seen_pids.add(pid)

                    # Full-line training: EVERY solver move of the solution becomes a training position
                    game = zeropawn_cpp.ChessGame(fen)
                    gid = hash_game_id(f"tactics_{pid}")
                    solver_plies = 0
                    i = 0
                    while i + 1 < len(moves) and solver_plies < max_solver_plies:
                        opp_move, sol_move = moves[i], moves[i + 1]
                        if not game.is_legal(opp_move):
                            break
                        game.make_move_uci(opp_move)
                        i += 1
                        move_idx = zeropawn_cpp.move_to_index(game, sol_move)
                        if not (0 <= move_idx < 4672):
                            break
                        if sol_move not in game.legal_moves_uci():
                            break

                        planes = zeropawn_cpp.encode_state(game)
                        pos_hash = game.get_hash()
                        pos = PositionData(
                            planes=planes,
                            move_idx=move_idx,
                            outcome=OUTCOME_UNAVAILABLE,  # puzzle position: masked value target
                            game_id=gid,
                            ply=0,
                            pos_hash=pos_hash,
                        )
                        solver_plies += 1
                        if (gid % 100) < 2:
                            candidate_val.append(pos)
                        else:
                            train_chunk.append(pos)
                            train_hashes.add(pos_hash)
                            n_train += 1
                            if len(train_chunk) >= chunk_size:
                                _flush_positions_chunk(
                                    train_chunk,
                                    f_tr_planes,
                                    f_tr_moves,
                                    f_tr_outcomes,
                                    f_tr_game_ids,
                                    f_tr_plies,
                                )
                                train_chunk.clear()

                        game.make_move_uci(sol_move)
                        i += 1
                    if solver_plies == 0:
                        skipped += 1
                except Exception:
                    skipped += 1
                    continue
                if total % 50000 == 0:
                    print(
                        f"  processed {total:,} puzzles "
                        f"(train {n_train:,}, val-cand {len(candidate_val):,}, skipped {skipped:,})"
                    )

            # Flush remaining train positions
            if train_chunk:
                _flush_positions_chunk(
                    train_chunk,
                    f_tr_planes,
                    f_tr_moves,
                    f_tr_outcomes,
                    f_tr_game_ids,
                    f_tr_plies,
                )
                train_chunk.clear()

    # Atomically promote train binary files
    for k in train_part_paths:
        if train_final_paths[k].exists():
            train_final_paths[k].unlink()
        train_part_paths[k].rename(train_final_paths[k])

    print(f"\nTotal puzzles: {total:,}, skipped: {skipped:,}")
    print(f"Train positions: {n_train:,}, val candidates: {len(candidate_val):,}")

    # val dedup against train (same rule as game pipeline)
    final_val = [p for p in candidate_val if p.pos_hash not in train_hashes]
    print(
        f"Val dedup removed {len(candidate_val) - len(final_val):,}, final val {len(final_val):,}"
    )

    meta_train = {
        "split": "train",
        "num_samples": n_train,
        "planes_shape": [19, 8, 8],
        "dtype_planes": "float32",
        "dtype_moves": "int16",
        "dtype_outcomes": "uint8",
        "dtype_game_ids": "uint64",
        "dtype_plies": "int16",
        "source": str(csv_path),
        "kind": "tactics-fullline",
        "value_loss": "masked (outcome=255, ply=0)",
        "max_solver_plies": max_solver_plies,
    }
    with open(train_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_train, f, indent=2)

    save_dataset_split(
        val_dir,
        final_val,
        "val",
        extra_metadata={"source": str(csv_path), "kind": "tactics-fullline"},
    )
    return n_train, len(final_val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tactics CSV to training bins.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/dataset_tactics")
    parser.add_argument(
        "--max-solver-plies",
        type=int,
        default=4,
        help="Cap solver positions per puzzle (bounds veryLong skew).",
    )
    args = parser.parse_args()
    n_tr, n_val = build_tactics_dataset(args.csv, args.output_dir, args.max_solver_plies)
    print(f"DONE: train={n_tr} val={n_val} in {args.output_dir}")


if __name__ == "__main__":
    main()
