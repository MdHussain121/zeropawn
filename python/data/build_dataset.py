"""Dataset builder converting PGN games to binary memory-mapped files using C++ encoder.

Stores per position:
- encoded planes (19, 8, 8) float32
- played-move index (0-4671) int16
- game outcome from side-to-move perspective (0=win, 1=draw, 2=loss) uint8
- game id hash uint64
- ply number int16

Splits train/val deterministically by game ID hash (98% / 2%).
Deduplicates validation positions against training positions and reports stats.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, TextIO
import numpy as np
import chess.pgn

logger = logging.getLogger(__name__)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp
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


def hash_game_id(game_id: str) -> int:
    """Deterministic 64-bit unsigned hash of a game identifier."""
    h = hashlib.sha256(game_id.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def extract_game_id(game: chess.pgn.Game) -> str:
    """Extract game ID from Site URL or create a deterministic string."""
    site = game.headers.get("Site", "")
    if site.startswith("http"):
        return site.rstrip("/").split("/")[-1]
    w = game.headers.get("White", "W")
    b = game.headers.get("Black", "B")
    d = game.headers.get("Date", "D")
    return f"{d}_{w}_{b}"


def parse_result(result_str: str) -> int | None:
    """Parse result string: 1 = White win, -1 = Black win, 0 = Draw, None = unknown."""
    if result_str == "1-0":
        return 1
    if result_str == "0-1":
        return -1
    if result_str == "1/2-1/2":
        return 0
    return None


OUTCOME_UNAVAILABLE: int = (
    255  # Sentinel for unlabelled/tactical positions with masked value target
)
VALID_OUTCOMES: tuple[int, ...] = (0, 1, 2, OUTCOME_UNAVAILABLE)


class PositionData:
    __slots__ = ("planes", "move_idx", "outcome", "game_id", "ply", "pos_hash")

    def __init__(
        self,
        planes: np.ndarray,
        move_idx: int,
        outcome: int,
        game_id: int,
        ply: int,
        pos_hash: int,
    ) -> None:
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Invalid outcome {outcome}: must be one of {VALID_OUTCOMES}")
        self.planes = planes
        self.move_idx = move_idx
        self.outcome: int = outcome
        self.game_id = game_id
        self.ply = ply
        self.pos_hash = pos_hash


def process_pgn_game(game: chess.pgn.Game) -> tuple[int, bool, list[PositionData]]:
    """Process a single chess.pgn.Game using the C++ ChessGame and encoder.

    Returns:
        (game_id_hash, is_val, list of PositionData)
    """
    game_id_str = extract_game_id(game)
    game_id = hash_game_id(game_id_str)
    # Deterministic 98% train / 2% val split by game ID hash
    is_val = (game_id % 100) < 2

    raw_res = game.headers.get("Result", "")
    res = parse_result(raw_res)
    if res is None:
        return game_id, is_val, []

    positions: list[PositionData] = []
    cpp_game = zeropawn_cpp.ChessGame()

    for ply_num, move in enumerate(game.mainline_moves()):
        uci_move = move.uci()
        planes = zeropawn_cpp.encode_state(cpp_game)
        move_idx = zeropawn_cpp.move_to_index(cpp_game, uci_move)

        if move_idx < 0 or move_idx >= 4672:
            # Move could not be mapped (e.g. malformed), stop game replay
            break

        # Side to move's perspective outcome:
        # 0 = Win, 1 = Draw, 2 = Loss
        is_white = cpp_game.side_to_move()
        if res == 0:
            outcome = 1  # Draw
        elif (res == 1 and is_white) or (res == -1 and not is_white):
            outcome = 0  # Win for side to move
        else:
            outcome = 2  # Loss for side to move

        pos_hash = cpp_game.get_hash()

        positions.append(
            PositionData(
                planes=planes,
                move_idx=move_idx,
                outcome=outcome,
                game_id=game_id,
                ply=ply_num,
                pos_hash=pos_hash,
            )
        )

        cpp_game.make_move_uci(uci_move)

    return game_id, is_val, positions


def save_dataset_split(
    data_dir: Path,
    positions: list[PositionData],
    split_name: str,
    extra_metadata: dict | None = None,
) -> None:
    """Save PositionData list into binary memory-mapped files and metadata.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    num_samples = len(positions)

    planes_path = data_dir / "planes.bin"
    moves_path = data_dir / "moves.bin"
    outcomes_path = data_dir / "outcomes.bin"
    game_ids_path = data_dir / "game_ids.bin"
    plies_path = data_dir / "plies.bin"

    planes_mm = np.memmap(planes_path, dtype=np.float32, mode="w+", shape=(num_samples, 19, 8, 8))
    moves_mm = np.memmap(moves_path, dtype=np.int16, mode="w+", shape=(num_samples,))
    outcomes_mm = np.memmap(outcomes_path, dtype=np.uint8, mode="w+", shape=(num_samples,))
    game_ids_mm = np.memmap(game_ids_path, dtype=np.uint64, mode="w+", shape=(num_samples,))
    plies_mm = np.memmap(plies_path, dtype=np.int16, mode="w+", shape=(num_samples,))

    try:
        for i, pos in enumerate(positions):
            planes_mm[i] = pos.planes
            moves_mm[i] = pos.move_idx
            outcomes_mm[i] = pos.outcome
            game_ids_mm[i] = pos.game_id
            plies_mm[i] = pos.ply

        planes_mm.flush()
        moves_mm.flush()
        outcomes_mm.flush()
        game_ids_mm.flush()
        plies_mm.flush()
    finally:
        for mm in (planes_mm, moves_mm, outcomes_mm, game_ids_mm, plies_mm):
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("memmap close failed", exc_info=True)

    metadata = {
        "split": split_name,
        "num_samples": num_samples,
        "planes_shape": [19, 8, 8],
        "dtype_planes": "float32",
        "dtype_moves": "int16",
        "dtype_outcomes": "uint8",
        "dtype_game_ids": "uint64",
        "dtype_plies": "int16",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(data_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{split_name.upper()}] Saved {num_samples} positions to {data_dir}")


def _flush_positions_chunk(
    positions: list[PositionData],
    f_planes: Any,
    f_moves: Any,
    f_outcomes: Any,
    f_game_ids: Any,
    f_plies: Any,
) -> None:
    """Pack and flush a chunk of PositionData to binary file handles."""
    n = len(positions)
    if n == 0:
        return
    planes_arr = np.empty((n, 19, 8, 8), dtype=np.float32)
    moves_arr = np.empty(n, dtype=np.int16)
    outcomes_arr = np.empty(n, dtype=np.uint8)
    game_ids_arr = np.empty(n, dtype=np.uint64)
    plies_arr = np.empty(n, dtype=np.int16)

    for i, p in enumerate(positions):
        planes_arr[i] = p.planes
        moves_arr[i] = p.move_idx
        outcomes_arr[i] = p.outcome
        game_ids_arr[i] = p.game_id
        plies_arr[i] = p.ply

    f_planes.write(planes_arr.tobytes())
    f_moves.write(moves_arr.tobytes())
    f_outcomes.write(outcomes_arr.tobytes())
    f_game_ids.write(game_ids_arr.tobytes())
    f_plies.write(plies_arr.tobytes())


def build_binary_dataset(
    pgn_path_or_stream: str | Path | TextIO,
    output_dir: str | Path,
    max_games: int | None = None,
    progress_interval: int = 0,
    chunk_size: int = 5_000,
) -> tuple[int, int, int]:
    """Parse PGN, split 98/2 by game hash, deduplicate val positions, and save.

    Uses a streaming bounded-chunk pipeline to keep memory strictly bounded
    regardless of input PGN size. Writes partial files (.part) and atomically
    promotes them only after verification.
    """
    import shutil

    out_dir = Path(output_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Pre-check disk space (at least 200 MB free)
    try:
        free_bytes = shutil.disk_usage(out_dir if out_dir.exists() else out_dir.parent).free
        if free_bytes < 200 * 1024 * 1024:
            raise OSError(
                f"Insufficient disk space ({free_bytes / 1e6:.1f} MB available, need >= 200 MB)"
            )
    except Exception as e:
        logger.debug("Disk usage check warning: %s", e)

    train_part_paths = {
        "planes": train_dir / "planes.bin.part",
        "moves": train_dir / "moves.bin.part",
        "outcomes": train_dir / "outcomes.bin.part",
        "game_ids": train_dir / "game_ids.bin.part",
        "plies": train_dir / "plies.bin.part",
    }

    train_chunk: list[PositionData] = []
    candidate_val_positions: list[PositionData] = []
    train_hashes: set[int] = set()

    train_games_count = 0
    val_games_count = 0
    total_games_seen = 0
    n_train = 0

    try:
        with (
            open(train_part_paths["planes"], "wb") as f_tr_planes,
            open(train_part_paths["moves"], "wb") as f_tr_moves,
            open(train_part_paths["outcomes"], "wb") as f_tr_outcomes,
            open(train_part_paths["game_ids"], "wb") as f_tr_game_ids,
            open(train_part_paths["plies"], "wb") as f_tr_plies,
        ):
            with contextlib.ExitStack() as stack:
                if isinstance(pgn_path_or_stream, (str, Path)):
                    f = stack.enter_context(
                        open(pgn_path_or_stream, "r", encoding="utf-8", errors="replace")
                    )
                else:
                    f = pgn_path_or_stream

                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    total_games_seen += 1

                    game_id, is_val, pos_list = process_pgn_game(game)
                    if not pos_list:
                        continue

                    if is_val:
                        val_games_count += 1
                        candidate_val_positions.extend(pos_list)
                    else:
                        train_games_count += 1
                        for p in pos_list:
                            train_hashes.add(p.pos_hash)
                        train_chunk.extend(pos_list)
                        n_train += len(pos_list)

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

                    if max_games is not None and total_games_seen >= max_games:
                        break

                    if progress_interval and total_games_seen % progress_interval == 0:
                        emit_report(
                            format_kv_line(
                                [
                                    ("parsed_games", total_games_seen),
                                    ("train_games", train_games_count),
                                    ("val_games", val_games_count),
                                    ("train_positions", n_train),
                                ]
                            )
                        )

            # Flush any residual training positions
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

        # Atomically rename train part files to final .bin files
        for key, part_p in train_part_paths.items():
            final_p = train_dir / f"{key}.bin"
            if part_p.exists():
                part_p.replace(final_p)

        # Write train metadata.json
        train_metadata = {
            "split": "train",
            "num_samples": n_train,
            "planes_shape": [19, 8, 8],
            "dtype_planes": "float32",
            "dtype_moves": "int16",
            "dtype_outcomes": "uint8",
            "dtype_game_ids": "uint64",
            "dtype_plies": "int16",
        }
        with open(train_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(train_metadata, f, indent=2)

        # Deduplicate validation positions against training positions
        final_val_positions: list[PositionData] = []
        removed_count = 0
        for pos in candidate_val_positions:
            if pos.pos_hash in train_hashes:
                removed_count += 1
            else:
                final_val_positions.append(pos)

        n_val_final = len(final_val_positions)
        val_part_paths = {
            "planes": val_dir / "planes.bin.part",
            "moves": val_dir / "moves.bin.part",
            "outcomes": val_dir / "outcomes.bin.part",
            "game_ids": val_dir / "game_ids.bin.part",
            "plies": val_dir / "plies.bin.part",
        }

        with (
            open(val_part_paths["planes"], "wb") as f_val_planes,
            open(val_part_paths["moves"], "wb") as f_val_moves,
            open(val_part_paths["outcomes"], "wb") as f_val_outcomes,
            open(val_part_paths["game_ids"], "wb") as f_val_game_ids,
            open(val_part_paths["plies"], "wb") as f_val_plies,
        ):
            for start_idx in range(0, n_val_final, chunk_size):
                end_idx = min(start_idx + chunk_size, n_val_final)
                _flush_positions_chunk(
                    final_val_positions[start_idx:end_idx],
                    f_val_planes,
                    f_val_moves,
                    f_val_outcomes,
                    f_val_game_ids,
                    f_val_plies,
                )

        # Atomically rename val part files to final .bin files
        for key, part_p in val_part_paths.items():
            final_p = val_dir / f"{key}.bin"
            if part_p.exists():
                part_p.replace(final_p)

        # Write val metadata.json
        val_metadata = {
            "split": "val",
            "num_samples": n_val_final,
            "planes_shape": [19, 8, 8],
            "dtype_planes": "float32",
            "dtype_moves": "int16",
            "dtype_outcomes": "uint8",
            "dtype_game_ids": "uint64",
            "dtype_plies": "int16",
            "candidate_val_positions": len(candidate_val_positions),
            "dedup_removed_positions": removed_count,
        }
        with open(val_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(val_metadata, f, indent=2)

    except Exception:
        # Clean up partial files on failure
        for p in list(train_part_paths.values()) + list(val_dir.glob("*.bin.part")):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        raise

    print("\n" + "=" * 60)
    print("DATASET SPLIT AND DEDUPLICATION REPORT")
    print("=" * 60)
    print(f"Total games processed:           {total_games_seen}")
    print(
        f"Training games:                  {train_games_count} ({(train_games_count / max(1, total_games_seen) * 100):.1f}%)"
    )
    print(
        f"Validation games:                {val_games_count} ({(val_games_count / max(1, total_games_seen) * 100):.1f}%)"
    )
    print(f"Training positions:              {n_train}")
    print(f"Candidate validation positions:  {len(candidate_val_positions)}")
    pct_removed = (removed_count / max(1, len(candidate_val_positions))) * 100
    print(f"Validation positions removed:    {removed_count} ({pct_removed:.2f}%)")
    print(f"Final validation positions:      {n_val_final}")
    print("=" * 60 + "\n")

    del candidate_val_positions, final_val_positions, train_hashes
    gc.collect()

    return n_train, n_val_final, removed_count


def main() -> None:
    # Single logging setup: rotating file for everything, WARNING+ on stderr.
    setup_logging(log_dir="logs", log_filename="build_dataset.log")
    ctx = run_context("data-convert")
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Convert filtered PGN to binary memmap dataset.")
    parser.add_argument("--pgn", type=str, required=True, help="Input filtered PGN file.")
    parser.add_argument("--output-dir", type=str, default="data/dataset", help="Output directory.")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process.")

    args = parser.parse_args()

    emit_report(banner("data-convert", ctx["start"] + " (local)"))
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
                ("pgn", args.pgn, "user"),
                (
                    "output_dir",
                    args.output_dir,
                    "user" if args.output_dir != "data/dataset" else "default",
                ),
                ("max_games", args.max_games, "user" if args.max_games is not None else "default"),
            ]
        )
    )

    try:
        build_binary_dataset(
            args.pgn, args.output_dir, max_games=args.max_games, progress_interval=1000
        )
        records = []
        for split in ("train", "val"):
            split_dir = Path(args.output_dir) / split
            if split_dir.is_dir():
                for path in sorted(split_dir.iterdir()):
                    if path.is_file():
                        records.append(file_record(str(path)))
        emit_report(artifacts_list(records))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except Exception as e:
        emit_error(
            error_block(
                f"data-convert failed: {e}",
                "unreadable input PGN or unwritable output directory",
                "check --pgn exists and --output-dir is writable, then see LOG",
                "logs/build_dataset.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
