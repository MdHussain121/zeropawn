"""Build Supervised Tactical Puzzle Training Dataset from Lichess / Master Puzzle CSVs.

Converts tactical puzzles into canonical binary memory-mapped datasets
(planes.bin, moves.bin, outcomes.bin, value_targets.bin, policy_targets.bin, metadata.json)
compatible with BinaryChessDataset.

Features:
- Parses Lichess puzzle CSV format (PuzzleId, FEN, Moves, Rating, etc.).
- Plays initial opponent setup move to reach the tactical dilemma for the engine.
- Encodes state via zeropawn_cpp.encode_state into (19, 8, 8) feature planes.
- Sets target policy to 100% on the tactical solution move.
- Sets value target to winning distribution [0.98, 0.01, 0.01] (mover is winning).
- Supports rating filters (e.g., >= 2000, >= 2400) and sample caps.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chess
import numpy as np

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp

from python.cli.common import REPO_ROOT

logger = logging.getLogger(__name__)


def validate_label_with_stockfish(
    sol_move_uci: str, fen_before_opponent: str, opponent_move_uci: str, depth: int = 14
) -> dict:
    """Validate a puzzle label against Stockfish (audit Â§9, P2).

    Returns {
        "agreement": bool,
        "sf_top1": str | None,
        "label_is_sf_top": bool,
        "tie_count_25cp": int,
        "label_cp_loss": float,
    }
    Disabled by default (Stockfish subprocess overhead); call explicitly.
    """
    try:
        stockfish_path = (REPO_ROOT / "tools" / "stockfish.exe").resolve()
        if not stockfish_path.exists():
            return {
                "agreement": False,
                "sf_top1": None,
                "label_is_sf_top": False,
                "tie_count_25cp": 0,
                "label_cp_loss": 9999,
                "error": "stockfish not found",
            }
        # Write temporary FEN + move for analysis
        board = chess.Board(fen_before_opponent)
        if opponent_move_uci:
            board.push_uci(opponent_move_uci)
        board.fen()
        # Quick UCI-based probe: we don't run a full SF session here (too slow for batch),
        # but we expose the interface so build scripts can call it per-sample.
        # For production use, run SF depth 14 externally and compare bestmove to sol_move_uci.
        return {
            "agreement": None,
            "sf_top1": None,
            "label_is_sf_top": None,
            "tie_count_25cp": 0,
            "label_cp_loss": 0.0,
            "note": "Stockfish validation requires external subprocess call; interface provided.",
        }
    except Exception as exc:
        return {
            "agreement": False,
            "sf_top1": None,
            "label_is_sf_top": False,
            "tie_count_25cp": 0,
            "label_cp_loss": 9999,
            "error": str(exc),
        }


def build_puzzle_dataset(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    min_rating: int = 1800,
    max_rating: Optional[int] = None,
    max_positions: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compile tactical puzzles into binary train and val datasets."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Puzzle CSV not found at {csv_path}")

    rng = np.random.RandomState(seed)
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Determine column indices from header
        id_col = 0
        fen_col = 1
        moves_col = 2
        rating_col = 3
        if header:
            for idx, col in enumerate(header):
                name = col.strip().lower()
                if name == "puzzleid":
                    id_col = idx
                elif name == "fen":
                    fen_col = idx
                elif name == "moves":
                    moves_col = idx
                elif name == "rating":
                    rating_col = idx

        count = 0
        for row in reader:
            if not row or len(row) <= max(fen_col, moves_col):
                continue

            # Rating filtering
            if len(row) > rating_col:
                try:
                    rating = int(row[rating_col])
                    if rating < min_rating:
                        continue
                    if max_rating is not None and rating > max_rating:
                        continue
                except ValueError:
                    pass

            fen = row[fen_col].strip()
            move_tokens = row[moves_col].strip().split()
            if len(move_tokens) < 2:
                continue

            opp_move_uci = move_tokens[0]
            sol_move_uci = move_tokens[1]

            # Play opponent move to reach tactical puzzle position
            try:
                board = chess.Board(fen)
                opp_move = chess.Move.from_uci(opp_move_uci)
                if opp_move not in board.legal_moves:
                    continue
                board.push(opp_move)

                sol_move = chess.Move.from_uci(sol_move_uci)
                if sol_move not in board.legal_moves:
                    continue
            except Exception:
                continue

            puzzle_fen = board.fen()
            cpp_game = zeropawn_cpp.ChessGame(puzzle_fen)
            planes = zeropawn_cpp.encode_state(cpp_game)

            try:
                sol_idx = zeropawn_cpp.move_to_index(cpp_game, sol_move_uci)
                if not (0 <= sol_idx < 4672):
                    continue
            except Exception:
                continue

            # Build policy distribution: sharp target on winning tactical solution
            policy_dist = np.zeros(4672, dtype=np.float32)
            policy_dist[sol_idx] = 1.0

            # Value target: masked when no real WDL outcome is available.
            # Only supervised WDL targets should come from verified game outcomes
            # (audit Â§5.2, Â§9.2 â€” synthetic [0.98,0.01,0.01] creates the value-head saturation).
            value_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # masked / unknown
            # If a verified forced-mate label is added in future, set to [1.0, 0.0, 0.0]
            # with outcome=0 and a separate value_mask=1.0.

            sample = {
                "fen": puzzle_fen,
                "planes": planes,
                "move_idx": int(sol_idx),
                "outcome": 1,
                "value_target": value_target,
                "policy_dist": policy_dist,
                "ply": int(board.fullmove_number * 2 - (1 if board.turn == chess.WHITE else 0)),
                "puzzle_id": row[id_col] if len(row) > id_col else f"puz_{count}",
            }

            # Split train / val
            if rng.uniform() < val_ratio:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

            count += 1
            if max_positions is not None and count >= max_positions:
                break

    if not train_samples and val_samples:
        train_samples.append(val_samples.pop())
    if not val_samples and train_samples:
        val_samples.append(train_samples[-1])
    if not train_samples:
        raise RuntimeError("No qualified tactical puzzle samples found to build dataset.")

    _write_dataset_split(train_dir, train_samples, "train")
    _write_dataset_split(val_dir, val_samples, "val")

    summary = {
        "output_dir": str(output_dir),
        "total_samples": len(train_samples) + len(val_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "min_rating": min_rating,
    }
    logger.info("Built tactical puzzle dataset: %s", summary)
    return summary


def _write_dataset_split(split_dir: Path, samples: List[Dict[str, Any]], split_name: str) -> None:
    num = len(samples)
    planes_path = split_dir / "planes.bin"
    moves_path = split_dir / "moves.bin"
    outcomes_path = split_dir / "outcomes.bin"
    game_ids_path = split_dir / "game_ids.bin"
    plies_path = split_dir / "plies.bin"
    val_targets_path = split_dir / "value_targets.bin"
    policy_targets_path = split_dir / "policy_targets.bin"
    meta_path = split_dir / "metadata.json"

    planes_mm = np.memmap(planes_path, dtype=np.float32, mode="w+", shape=(num, 19, 8, 8))
    moves_mm = np.memmap(moves_path, dtype=np.int16, mode="w+", shape=(num,))
    outcomes_mm = np.memmap(outcomes_path, dtype=np.uint8, mode="w+", shape=(num,))
    game_ids_mm = np.memmap(game_ids_path, dtype=np.uint64, mode="w+", shape=(num,))
    plies_mm = np.memmap(plies_path, dtype=np.int16, mode="w+", shape=(num,))
    val_targets_mm = np.memmap(val_targets_path, dtype=np.float32, mode="w+", shape=(num, 3))
    policy_targets_mm = np.memmap(
        policy_targets_path, dtype=np.float32, mode="w+", shape=(num, 4672)
    )

    for i, s in enumerate(samples):
        planes_mm[i] = s["planes"]
        moves_mm[i] = np.int16(s["move_idx"])
        outcomes_mm[i] = np.uint8(s["outcome"])
        game_ids_mm[i] = np.uint64(i + 1)
        plies_mm[i] = np.int16(s["ply"])
        val_targets_mm[i] = s["value_target"]
        policy_targets_mm[i] = s["policy_dist"]

    planes_mm.flush()
    moves_mm.flush()
    outcomes_mm.flush()
    game_ids_mm.flush()
    plies_mm.flush()
    val_targets_mm.flush()
    policy_targets_mm.flush()

    metadata = {
        "num_samples": num,
        "planes_shape": [19, 8, 8],
        "split": split_name,
        "kind": "tactics_puzzles",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
