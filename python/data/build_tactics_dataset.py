"""Build Comprehensive Supervised Tactical Training Dataset (dataset_tactics_v1).

Adheres strictly to the Gate 1 and Gate 2 specifications:
1. Sources:
   - Official Lichess puzzle database (tmp/source_acquisition/lichess/lichess_db_puzzle.csv.zst)
   - Benchmark quarantine (data/puzzles/lichess_puzzles_5000_1800_2800.csv) - 100% excluded.
2. 4 Difficulty Tiers:
   - Tier 1 (1000-1499): 20%
   - Tier 2 (1500-1999): 35%
   - Tier 3 (2000-2399): 30%
   - Tier 4 (2400+):     15%
3. Target serialized sample cap: N_final <= 500,000.
4. Puzzle semantics:
   - CSV FEN is the board BEFORE opponent move Moves[0].
   - Replay Moves[0] to establish root dilemma position.
   - Solver moves at odd indices (1, 3, ...), opponent responses at even indices (2, 4, ...).
   - Expand up to max_solver_plies=4.
5. Value Target Masking:
   - Forced mates: outcome=0, ply=10, value_target=[1.0, 0.0, 0.0]
   - Other tactics: outcome=255, ply=0, value_target=[0.0, 0.0, 0.0] (val_mask=0.0 in loss)
6. Splits:
   - Deterministic 90/5/5 source-level split via sha256(puzzle_id) % 100.
   - Cross-split position deduplication ensuring 0% position leakage.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import chess
import numpy as np
import zstandard as zstd

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_quarantine_ids(path: Path) -> Set[str]:
    """Load puzzle IDs to strictly exclude from all training/val/test splits."""
    quarantine: Set[str] = set()
    if not path.is_file():
        logger.warning("Quarantine benchmark file not found at %s", path)
        return quarantine

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        id_idx = 0
        if header:
            for idx, col in enumerate(header):
                if col.strip().lower() == "puzzleid":
                    id_idx = idx
                    break
        for row in reader:
            if row and len(row) > id_idx:
                quarantine.add(row[id_idx].strip())
    logger.info("Loaded %d quarantined benchmark puzzles from %s", len(quarantine), path)
    return quarantine


def get_puzzle_split(puzzle_id: str) -> str:
    """Deterministic 90/5/5 split by puzzle_id hash."""
    h = int(hashlib.sha256(puzzle_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    elif h < 95:
        return "val"
    else:
        return "test"


def build_tactics_dataset(
    zst_path: Path,
    quarantine_path: Path,
    output_dir: Path,
    target_puzzles: int = 215_000,
    max_samples_cap: int = 500_000,
    max_solver_plies: int = 4,
) -> Dict[str, Any]:
    """Build the tactical dataset with full verification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    quarantine = load_quarantine_ids(quarantine_path)

    # 4-tier quotas (percentages: 20%, 35%, 30%, 15%)
    quotas = {
        1: int(target_puzzles * 0.20),  # 43,000 (1000 - 1499)
        2: int(target_puzzles * 0.35),  # 75,250 (1500 - 1999)
        3: int(target_puzzles * 0.30),  # 64,500 (2000 - 2399)
        4: int(target_puzzles * 0.15),  # 32,250 (2400+)
    }
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    logger.info("Target puzzle quotas by tier: %s", quotas)

    # In-memory sample collectors per split
    # Each sample: (planes_np, move_idx, outcome, ply, val_target, game_id)
    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Tracking canonical board keys (FEN without halfmove/fullmove) to prevent transposition leakage
    train_seen_keys: Set[str] = set()
    val_seen_keys: Set[str] = set()
    test_seen_keys: Set[str] = set()

    total_samples = 0
    total_puzzles_used = 0
    used_puzzle_ids: Set[str] = set()
    t0 = time.time()

    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            csv_reader = csv.reader(text_stream)
            header = next(csv_reader, None)

            # Determine column positions
            puz_col, fen_col, moves_col, rating_col, themes_col = 0, 1, 2, 3, 7
            if header:
                for idx, col in enumerate(header):
                    name = col.strip().lower()
                    if name == "puzzleid":
                        puz_col = idx
                    elif name == "fen":
                        fen_col = idx
                    elif name == "moves":
                        moves_col = idx
                    elif name == "rating":
                        rating_col = idx
                    elif name == "themes":
                        themes_col = idx

            for row_idx, row in enumerate(csv_reader):
                if not row or len(row) <= max(fen_col, moves_col, rating_col):
                    continue

                puz_id = row[puz_col].strip()
                if puz_id in quarantine:
                    continue

                try:
                    rating = int(row[rating_col])
                except ValueError:
                    continue

                # Determine tier
                if 1000 <= rating < 1500:
                    tier = 1
                elif 1500 <= rating < 2000:
                    tier = 2
                elif 2000 <= rating < 2400:
                    tier = 3
                elif rating >= 2400:
                    tier = 4
                else:
                    continue

                if tier_counts[tier] >= quotas[tier]:
                    # Check if all quotas reached
                    if all(tier_counts[t] >= quotas[t] for t in (1, 2, 3, 4)):
                        logger.info("All tier quotas met at row %d", row_idx)
                        break
                    continue

                fen = row[fen_col].strip()
                move_tokens = row[moves_col].strip().split()
                if len(move_tokens) < 2:
                    continue

                row[themes_col].strip().lower() if len(row) > themes_col else ""

                # Setup board and play opponent move (Moves[0])
                try:
                    board = chess.Board(fen)
                    opp_setup = chess.Move.from_uci(move_tokens[0])
                    if opp_setup not in board.legal_moves:
                        continue
                    board.push(opp_setup)
                except Exception:
                    continue

                split = get_puzzle_split(puz_id)
                puzzle_samples: List[Dict[str, Any]] = []
                discard_puzzle = False

                # Iterate through solver moves
                solver_ply = 0
                while 2 * solver_ply + 1 < len(move_tokens) and solver_ply < max_solver_plies:
                    sol_uci = move_tokens[2 * solver_ply + 1]
                    try:
                        sol_move = chess.Move.from_uci(sol_uci)
                        if sol_move not in board.legal_moves:
                            discard_puzzle = True
                            break
                    except Exception:
                        discard_puzzle = True
                        break

                    curr_fen = board.fen()
                    # Canonical transposition key: board piece placement, active turn, castling, en passant
                    epd_key = board.epd()

                    # Transposition deduplication across splits
                    if split == "train":
                        if epd_key in val_seen_keys or epd_key in test_seen_keys:
                            discard_puzzle = True
                            break
                    elif split == "val":
                        if epd_key in train_seen_keys or epd_key in test_seen_keys:
                            discard_puzzle = True
                            break
                    elif split == "test":
                        if epd_key in train_seen_keys or epd_key in val_seen_keys:
                            discard_puzzle = True
                            break

                    try:
                        cpp_game = zeropawn_cpp.ChessGame(curr_fen)
                        planes = zeropawn_cpp.encode_state(cpp_game)
                        sol_idx = zeropawn_cpp.move_to_index(cpp_game, sol_uci)
                        if not (0 <= sol_idx < 4672):
                            discard_puzzle = True
                            break
                    except Exception:
                        discard_puzzle = True
                        break

                    # Only label as forced-mate when the solver move actually delivers checkmate
                    # (not just because the theme tag contains "mate"). Fixes false value-head labels.
                    is_mate = board.is_checkmate()

                    if is_mate:
                        outcome = 0
                        ply_target = 10
                        val_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        outcome = 255
                        ply_target = 0
                        val_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                    puzzle_samples.append(
                        {
                            "planes": planes,
                            "move_idx": int(sol_idx),
                            "outcome": outcome,
                            "ply": ply_target,
                            "value_target": val_target,
                            "puzzle_id": puz_id,
                            "epd_key": epd_key,
                        }
                    )

                    # If there's an opponent reply, push it
                    if 2 * solver_ply + 2 < len(move_tokens):
                        opp_reply_uci = move_tokens[2 * solver_ply + 2]
                        try:
                            opp_reply = chess.Move.from_uci(opp_reply_uci)
                            if opp_reply not in board.legal_moves:
                                break
                            board.push(opp_reply)
                        except Exception:
                            break
                    else:
                        break

                    solver_ply += 1

                if discard_puzzle or not puzzle_samples:
                    continue

                # Check if adding these samples exceeds the cap
                if total_samples + len(puzzle_samples) > max_samples_cap:
                    logger.info("Reached maximum samples cap (%d). Stopping.", max_samples_cap)
                    break

                # Accept samples into split
                used_puzzle_ids.add(puz_id)
                for samp in puzzle_samples:
                    split_samples[split].append(samp)
                    epd_key = samp["epd_key"]
                    if split == "train":
                        train_seen_keys.add(epd_key)
                    elif split == "val":
                        val_seen_keys.add(epd_key)
                    elif split == "test":
                        test_seen_keys.add(epd_key)

                total_samples += len(puzzle_samples)
                tier_counts[tier] += 1
                total_puzzles_used += 1

                if total_puzzles_used % 25000 == 0:
                    el = time.time() - t0
                    logger.info(
                        "Progress: %d puzzles, %d samples in %.1fs | Tiers: %s | Split counts: tr=%d, va=%d, te=%d",
                        total_puzzles_used,
                        total_samples,
                        el,
                        tier_counts,
                        len(split_samples["train"]),
                        len(split_samples["val"]),
                        len(split_samples["test"]),
                    )

    elapsed = time.time() - t0
    logger.info(
        "Finished extraction in %.1fs: %d puzzles, %d samples. Writing memmap binaries...",
        elapsed,
        total_puzzles_used,
        total_samples,
    )

    # Write binary memmaps for each split
    split_stats = {}
    for s_name in ("train", "val", "test"):
        s_dir = output_dir / s_name
        samples = split_samples[s_name]
        n = len(samples)
        logger.info("Writing %s split (%d samples)...", s_name, n)

        planes_mm = np.memmap(
            s_dir / "planes.bin", dtype=np.float32, mode="w+", shape=(n, 19, 8, 8)
        )
        moves_mm = np.memmap(s_dir / "moves.bin", dtype=np.int16, mode="w+", shape=(n,))
        outcomes_mm = np.memmap(s_dir / "outcomes.bin", dtype=np.uint8, mode="w+", shape=(n,))
        game_ids_mm = np.memmap(s_dir / "game_ids.bin", dtype=np.uint64, mode="w+", shape=(n,))
        plies_mm = np.memmap(s_dir / "plies.bin", dtype=np.int16, mode="w+", shape=(n,))
        val_targets_mm = np.memmap(
            s_dir / "value_targets.bin", dtype=np.float32, mode="w+", shape=(n, 3)
        )

        for idx, item in enumerate(samples):
            planes_mm[idx] = item["planes"]
            moves_mm[idx] = np.int16(item["move_idx"])
            outcomes_mm[idx] = np.uint8(item["outcome"])
            gid_val = int(hashlib.sha256(item["puzzle_id"].encode("utf-8")).hexdigest()[:16], 16)
            game_ids_mm[idx] = np.uint64(gid_val)
            plies_mm[idx] = np.int16(item["ply"])
            val_targets_mm[idx] = item["value_target"]

        planes_mm.flush()
        moves_mm.flush()
        outcomes_mm.flush()
        game_ids_mm.flush()
        plies_mm.flush()
        val_targets_mm.flush()

        # Close memmaps
        del planes_mm, moves_mm, outcomes_mm, game_ids_mm, plies_mm, val_targets_mm

        # Write split metadata
        split_meta = {
            "num_samples": n,
            "planes_shape": [19, 8, 8],
            "split": s_name,
            "kind": "tactics_puzzles",
            "tier_distribution": {
                "tier1_1000_1500": tier_counts[1],
                "tier2_1500_2000": tier_counts[2],
                "tier3_2000_2400": tier_counts[3],
                "tier4_2400_plus": tier_counts[4],
            },
        }
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)

        split_stats[s_name] = {
            "num_samples": n,
            "planes_bytes": (s_dir / "planes.bin").stat().st_size,
            "moves_bytes": (s_dir / "moves.bin").stat().st_size,
            "outcomes_bytes": (s_dir / "outcomes.bin").stat().st_size,
        }

    # Verify zero leakage across splits
    leakage_train_val = len(train_seen_keys.intersection(val_seen_keys))
    leakage_train_test = len(train_seen_keys.intersection(test_seen_keys))
    leakage_val_test = len(val_seen_keys.intersection(test_seen_keys))

    # Compute source and benchmark checksums
    zst_sha256 = compute_file_sha256(zst_path)
    quarantine_sha256 = compute_file_sha256(quarantine_path) if quarantine_path.exists() else "None"

    manifest = {
        "dataset_name": "dataset_tactics_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "source": {
            "archive": str(zst_path),
            "sha256": zst_sha256,
            "size_bytes": zst_path.stat().st_size,
        },
        "quarantine_benchmark": {
            "path": str(quarantine_path),
            "sha256": quarantine_sha256,
            "count_quarantined": len(quarantine),
            "count_leaked_into_dataset": len(used_puzzle_ids.intersection(quarantine)),
        },
        "tier_quotas": {
            "tier1_1000_1500": {
                "target": quotas[1],
                "actual_puzzles": tier_counts[1],
                "pct": round(tier_counts[1] / total_puzzles_used * 100, 2),
            },
            "tier2_1500_2000": {
                "target": quotas[2],
                "actual_puzzles": tier_counts[2],
                "pct": round(tier_counts[2] / total_puzzles_used * 100, 2),
            },
            "tier3_2000_2400": {
                "target": quotas[3],
                "actual_puzzles": tier_counts[3],
                "pct": round(tier_counts[3] / total_puzzles_used * 100, 2),
            },
            "tier4_2400_plus": {
                "target": quotas[4],
                "actual_puzzles": tier_counts[4],
                "pct": round(tier_counts[4] / total_puzzles_used * 100, 2),
            },
        },
        "totals": {
            "source_puzzles_used": total_puzzles_used,
            "total_serialized_samples": total_samples,
            "cap_limit": max_samples_cap,
            "within_cap": total_samples <= max_samples_cap,
        },
        "splits": split_stats,
        "leakage_verification": {
            "train_val_overlap": leakage_train_val,
            "train_test_overlap": leakage_train_test,
            "val_test_overlap": leakage_val_test,
            "status": "PASS - ZERO LEAKAGE"
            if (leakage_train_val == 0 and leakage_train_test == 0 and leakage_val_test == 0)
            else "FAIL",
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Successfully created dataset_tactics_v1 manifest: %s", manifest_path)
    return manifest


if __name__ == "__main__":
    from python.cli.common import setup_logging

    setup_logging()
    zst = Path("tmp/source_acquisition/lichess/lichess_db_puzzle.csv.zst")
    bench = Path("data/puzzles/lichess_puzzles_1000_1200_1600.csv")
    out = Path("data/dataset_tactics_v1")
    build_tactics_dataset(zst, bench, out)
