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

from python.constants import NUM_ACTIONS, NUM_PLANES, BOARD_SIZE

logger = logging.getLogger(__name__)


def build_puzzle_dataset(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    min_rating: int = 1800,
    max_rating: Optional[int] = None,
    max_positions: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compile tactical puzzles into binary train and val datasets.

    Split integrity (audit F-08): assignment is by the deterministic
    ``hash(PuzzleId)`` rule, so every row of a puzzle (including duplicate
    PuzzleId rows) lands in the same split; the real source PuzzleId hash is
    stored in ``game_ids.bin`` (not an ordinal); and the degenerate fallbacks
    that duplicated an example across train and val are removed — a build that
    cannot produce both splits fails explicitly.
    """
    from python.data.build_dataset import OUTCOME_UNAVAILABLE, hash_game_id
    from python.data.dataset_contract import BuildRecord, detect_git_commit, sha256_file

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Puzzle CSV not found at {csv_path}")

    val_pct = int(round(val_ratio * 100))
    if not (0 <= val_pct <= 99):
        raise ValueError(f"val_ratio {val_ratio} outside [0.0, 0.99]")

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    rows_seen = 0

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
            rows_seen += 1
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
                if not (0 <= sol_idx < NUM_ACTIONS):
                    continue
            except Exception:
                continue

            # Build policy distribution: sharp target on winning tactical solution
            policy_dist = np.zeros(NUM_ACTIONS, dtype=np.float32)
            policy_dist[sol_idx] = 1.0

            # Value target: masked when no real WDL outcome is available.
            # Only supervised WDL targets should come from verified game outcomes
            # (audit Â§5.2, Â§9.2 â€” synthetic [0.98,0.01,0.01] creates the value-head saturation).
            value_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # masked / unknown
            # If a verified forced-mate label is added in future, set to [1.0, 0.0, 0.0]
            # with outcome=0 and a separate value_mask=1.0.

            puzzle_id = row[id_col] if len(row) > id_col else ""
            gid = hash_game_id(f"puzzle_{puzzle_id}") if puzzle_id else 0

            sample = {
                "fen": puzzle_fen,
                "planes": planes,
                "move_idx": int(sol_idx),
                # Explicit policy-only semantics (audit F-08): a placeholder
                # draw label is never written; masked outcomes make the role
                # unambiguous to loaders and validators.
                "outcome": OUTCOME_UNAVAILABLE,
                "value_target": value_target,
                "policy_dist": policy_dist,
                "ply": int(board.fullmove_number * 2 - (1 if board.turn == chess.WHITE else 0)),
                "puzzle_id": puzzle_id,
                "gid": gid,
            }

            # Deterministic PuzzleId-family split (audit F-08): all rows of a
            # PuzzleId land in the same split; no per-row RNG.
            if gid % 100 < val_pct:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

            count += 1
            if max_positions is not None and count >= max_positions:
                break

    if not train_samples or not val_samples:
        raise RuntimeError(
            "Cannot produce the required split structure: "
            f"train={len(train_samples)} val={len(val_samples)} puzzles qualified "
            f"(val_ratio={val_ratio}). Provide more puzzles or a different --val-ratio; "
            "examples are never duplicated across splits."
        )

    _write_dataset_split(train_dir, train_samples, "train")
    _write_dataset_split(val_dir, val_samples, "val")

    summary = {
        "output_dir": str(output_dir),
        "total_samples": len(train_samples) + len(val_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "skipped_rows": rows_seen - count,
        "min_rating": min_rating,
        # Audit S2-03.1: the builder writes a BuildRecord but no validation
        # receipt, and the training consumers' fail-closed receipt gate
        # (require_validated_dataset) refuses uncertified data. Tell the
        # operator how to certify before gated training consumes this build.
        "certification_hint": (
            "Fresh build is NOT certified: run "
            f"'zeropawn data validate --dir {output_dir} --scan-planes full --write-receipt' "
            "before gated training (train_supervised / train_opening_policy / replay buffer) "
            "can consume it."
        ),
    }
    logger.info("%s", summary["certification_hint"])
    BuildRecord(
        builder="python.data.build_puzzle_dataset.build_puzzle_dataset",
        source={"path": str(csv_path), "sha256": sha256_file(csv_path)},
        config={
            "val_ratio": val_ratio,
            "split_rule": f"hash_game_id('puzzle_'+PuzzleId) % 100 < {val_pct} -> val",
            "min_rating": min_rating,
            "max_rating": max_rating,
            "max_positions": max_positions,
            "seed_note": "seed retained for API compatibility; split is hash-based (no RNG)",
        },
        label_semantics={
            "role": "policy_tactics",
            "outcome_encoding": "255=masked (policy-only example)",
            "wdl_supervised": False,
        },
        counts={k: v for k, v in summary.items() if k not in ("output_dir", "certification_hint")},
        git_commit=detect_git_commit(),
    ).write(output_dir)
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

    planes_mm = np.memmap(
        planes_path, dtype=np.float32, mode="w+", shape=(num, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    )
    moves_mm = np.memmap(moves_path, dtype=np.int16, mode="w+", shape=(num,))
    outcomes_mm = np.memmap(outcomes_path, dtype=np.uint8, mode="w+", shape=(num,))
    game_ids_mm = np.memmap(game_ids_path, dtype=np.uint64, mode="w+", shape=(num,))
    plies_mm = np.memmap(plies_path, dtype=np.int16, mode="w+", shape=(num,))
    val_targets_mm = np.memmap(val_targets_path, dtype=np.float32, mode="w+", shape=(num, 3))
    policy_targets_mm = np.memmap(
        policy_targets_path, dtype=np.float32, mode="w+", shape=(num, NUM_ACTIONS)
    )

    for i, s in enumerate(samples):
        planes_mm[i] = s["planes"]
        moves_mm[i] = np.int16(s["move_idx"])
        outcomes_mm[i] = np.uint8(s["outcome"])
        game_ids_mm[i] = np.uint64(s["gid"])
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
        "schema_version": "1.1",
        "role": "policy_tactics",
        "label_semantics": {
            "role": "policy_tactics",
            "outcome_encoding": "255=masked (policy-only example)",
            "wdl_supervised": False,
        },
        "game_identity_rule": "hash_game_id('puzzle_'+PuzzleId); source PuzzleId family preserved",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
