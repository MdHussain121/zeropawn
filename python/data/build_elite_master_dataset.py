"""Build Supervised 2500+ Master Chess Dataset v2 (dataset_elite_master_2500_v2).

Eligible criteria:
- Both White and Black rated >= 2500 Elo at game time.
- Both players verified human (strict BOT and engine account purge).
- Minimum base time >= 600 seconds with explicit, valid TimeControl tag.
- Move legality, clean termination, game deduplication.
- Model input contract: (19, 8, 8) float32 planes, 4672-action space.
- Perspective WDL value targets: win=[1,0,0], draw=[0,1,0], loss=[0,0,1].
- Deterministic 90/5/5 split by sha256(game_id) % 100.
- Transposition deduplication to guarantee 0% cross-split position leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import chess
import chess.pgn
import numpy as np

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_game_split(game_id: str) -> str:
    h = int(hashlib.sha256(game_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    elif h < 95:
        return "val"
    else:
        return "test"


def build_elite_master_dataset_v2(
    source_pgn_paths: List[Path],
    output_dir: Path,
    min_elo: int = 2500,
    min_base_time_sec: int = 600,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    source_pgns: List[Path] = []
    for d in source_pgn_paths:
        if d.is_file() and d.suffix == ".pgn":
            source_pgns.append(d)
        elif d.is_dir():
            source_pgns.extend(d.glob("*.pgn"))

    logger.info("Found %d source PGN files to inspect for 2500+ master pool.", len(source_pgns))

    # Tracking audit statistics
    total_games_inspected = 0
    both_2500_count = 0
    engine_rejected = 0
    tc_rejected = 0
    metadata_rejected = 0
    duplicate_games = 0
    legal_games_count = 0
    illegal_games_count = 0

    seen_game_keys: Set[str] = set()
    source_checksums: Dict[str, str] = {}

    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    train_seen_keys: Set[str] = set()
    val_seen_keys: Set[str] = set()
    test_seen_keys: Set[str] = set()

    for pgn_path in source_pgns:
        # Avoid HTML pages
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as test_f:
            first_chars = test_f.read(100).strip()
            if first_chars.startswith("<") or "Page not found" in first_chars:
                logger.warning("Skipping non-PGN HTML file: %s", pgn_path)
                continue

        sha = compute_file_sha256(pgn_path)
        source_checksums[pgn_path.name] = sha

        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                except Exception as e:
                    logger.warning("Skipping corrupted record in %s: %s", pgn_path.name, e)
                    continue

                if game is None:
                    break

                total_games_inspected += 1
                h = game.headers

                try:
                    we = int(h.get("WhiteElo", 0))
                    be = int(h.get("BlackElo", 0))
                except ValueError:
                    metadata_rejected += 1
                    continue

                # Both players must be >= 2500
                if we < min_elo or be < min_elo:
                    continue
                both_2500_count += 1

                # Human verification
                white = h.get("White", "")
                black = h.get("Black", "")
                wt = h.get("WhiteTitle", "")
                bt = h.get("BlackTitle", "")
                if any(x in white.lower() or x in black.lower() for x in ["bot", "engine"]):
                    engine_rejected += 1
                    continue
                if "BOT" in wt or "BOT" in bt:
                    engine_rejected += 1
                    continue

                # Minimum time control (>= 600s)
                tc = h.get("TimeControl", "")
                base = 0
                if "+" in tc:
                    pts = tc.split("+")
                    if pts[0].isdigit():
                        base = int(pts[0])
                elif tc.isdigit():
                    base = int(tc)
                if base < min_base_time_sec:
                    tc_rejected += 1
                    continue

                # Clean termination & valid result
                res = h.get("Result", "*")
                if res not in ("1-0", "0-1", "1/2-1/2"):
                    metadata_rejected += 1
                    continue

                # Deduplicate games by URL / key
                url = h.get("LichessURL", "").strip()
                site = h.get("Site", "").strip()
                if not url and site and site != "?":
                    url = site
                date_str = h.get("UTCDate", "") or h.get("Date", "")
                time_str = h.get("UTCTime", "") or h.get("Time", "")
                game_key = url if url else f"{white}_{we}_{black}_{be}_{date_str}_{time_str}_{res}"

                if game_key in seen_game_keys:
                    duplicate_games += 1
                    continue
                seen_game_keys.add(game_key)

                # Split by game hash
                split = get_game_split(game_key)
                board = game.board()
                game_samples: List[Dict[str, Any]] = []
                game_legal = True

                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        game_legal = False
                        break

                    curr_fen = board.fen()
                    epd_key = board.epd()

                    # Transposition deduplication across splits
                    if split == "train":
                        if epd_key in val_seen_keys or epd_key in test_seen_keys:
                            board.push(move)
                            continue
                    elif split == "val":
                        if epd_key in train_seen_keys or epd_key in test_seen_keys:
                            board.push(move)
                            continue
                    elif split == "test":
                        if epd_key in train_seen_keys or epd_key in val_seen_keys:
                            board.push(move)
                            continue

                    try:
                        cpp_game = zeropawn_cpp.ChessGame(curr_fen)
                        planes = zeropawn_cpp.encode_state(cpp_game)
                        sol_idx = zeropawn_cpp.move_to_index(cpp_game, move.uci())
                        if not (0 <= sol_idx < 4672):
                            game_legal = False
                            break
                    except Exception:
                        game_legal = False
                        break

                    # Outcome from mover's perspective
                    if board.turn == chess.WHITE:
                        if res == "1-0":
                            outcome, val_target = 0, [1.0, 0.0, 0.0]
                        elif res == "0-1":
                            outcome, val_target = 2, [0.0, 0.0, 1.0]
                        else:
                            outcome, val_target = 1, [0.0, 1.0, 0.0]
                    else:
                        if res == "0-1":
                            outcome, val_target = 0, [1.0, 0.0, 0.0]
                        elif res == "1-0":
                            outcome, val_target = 2, [0.0, 0.0, 1.0]
                        else:
                            outcome, val_target = 1, [0.0, 1.0, 0.0]

                    game_samples.append(
                        {
                            "planes": planes,
                            "move_idx": int(sol_idx),
                            "outcome": outcome,
                            "ply": int(board.ply()),
                            "value_target": np.array(val_target, dtype=np.float32),
                            "epd_key": epd_key,
                            "game_key": game_key,
                        }
                    )

                    board.push(move)

                if not game_legal or not game_samples:
                    illegal_games_count += 1
                    continue

                legal_games_count += 1
                for samp in game_samples:
                    split_samples[split].append(samp)
                    epd_k = samp["epd_key"]
                    if split == "train":
                        train_seen_keys.add(epd_k)
                    elif split == "val":
                        val_seen_keys.add(epd_k)
                    elif split == "test":
                        test_seen_keys.add(epd_k)

                if legal_games_count % 1000 == 0:
                    el = time.time() - t0
                    logger.info(
                        "Processed %d legal games in %.1fs | Samples: tr=%d, va=%d, te=%d",
                        legal_games_count,
                        el,
                        len(split_samples["train"]),
                        len(split_samples["val"]),
                        len(split_samples["test"]),
                    )

    total_positions = sum(len(split_samples[s]) for s in ("train", "val", "test"))
    elapsed = time.time() - t0
    logger.info(
        "Extracted %d legal master games (%d total positions) in %.1fs. Writing memmaps...",
        legal_games_count,
        total_positions,
        elapsed,
    )

    # Write binary memmaps
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
            gid_val = int(hashlib.sha256(item["game_key"].encode("utf-8")).hexdigest()[:16], 16)
            game_ids_mm[idx] = np.uint64(gid_val)
            plies_mm[idx] = np.int16(item["ply"])
            val_targets_mm[idx] = item["value_target"]

        planes_mm.flush()
        moves_mm.flush()
        outcomes_mm.flush()
        game_ids_mm.flush()
        plies_mm.flush()
        val_targets_mm.flush()

        del planes_mm, moves_mm, outcomes_mm, game_ids_mm, plies_mm, val_targets_mm

        split_meta = {
            "num_samples": n,
            "planes_shape": [19, 8, 8],
            "split": s_name,
            "kind": "master_games_2500",
        }
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)

        split_stats[s_name] = {
            "num_samples": n,
            "planes_bytes": (s_dir / "planes.bin").stat().st_size,
            "moves_bytes": (s_dir / "moves.bin").stat().st_size,
            "outcomes_bytes": (s_dir / "outcomes.bin").stat().st_size,
        }

    leakage_train_val = len(train_seen_keys.intersection(val_seen_keys))
    leakage_train_test = len(train_seen_keys.intersection(test_seen_keys))
    leakage_val_test = len(val_seen_keys.intersection(test_seen_keys))

    manifest = {
        "dataset_name": "dataset_elite_master_2500_v2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "criteria": {
            "white_rating_min": min_elo,
            "black_rating_min": min_elo,
            "min_base_time_sec": min_base_time_sec,
            "human_only": True,
            "explicit_time_control_required": True,
        },
        "audit_statistics": {
            "total_games_inspected": total_games_inspected,
            "both_players_2500_plus": both_2500_count,
            "engine_bot_rejected": engine_rejected,
            "tc_sub_600s_rejected": tc_rejected,
            "metadata_invalid_rejected": metadata_rejected,
            "duplicate_games_filtered": duplicate_games,
            "games_passing_legality_and_termination": legal_games_count,
            "games_failing_legality": illegal_games_count,
            "positions_extracted_before_dedup": total_positions,
            "unique_positions_remaining": len(train_seen_keys | val_seen_keys | test_seen_keys),
            "observed_positions_per_game": round(total_positions / max(1, legal_games_count), 2),
        },
        "source_files": source_checksums,
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

    logger.info("Successfully built dataset_elite_master_2500_v2 manifest at %s", manifest_path)
    return manifest


if __name__ == "__main__":
    from python.cli.common import setup_logging

    setup_logging()
    sources = [
        Path("data/elite_2500.pgn"),
        Path("data/filtered_combined.pgn"),
        Path("data/filtered_2026_08.pgn"),
        Path("data/filtered_2200.pgn"),
        Path("tmp/source_acquisition/master_rapid"),
    ]
    out = Path("data/dataset_elite_master_2500_v2")
    build_elite_master_dataset_v2(sources, out)
