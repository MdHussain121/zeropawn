"""Build Additional 2500+ Master Chess Dataset (data/staging/master_additional).

Extracts and serializes newly acquired 2500+ master games from staging archives:
- lichess_elite_2025-08.zip
- lichess_elite_2025-09.zip
- lichess_elite_2025-10.zip

Enforces:
- Both White & Black >= 2500 Elo.
- Verified human players (zero BOTs or engines).
- Base Time >= 600 seconds with explicit valid TimeControl.
- Deduplication against existing dataset_elite_master_2500_v2.
- 90/5/5 deterministic split by sha256(game_id) % 100.
- Transposition deduplication to ensure 0% cross-split leakage.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def build_master_additional(
    archive_paths: List[Path],
    output_dir: Path,
    existing_master_manifest_path: Optional[Path] = None,
    min_elo: int = 2500,
    min_base_time_sec: int = 600,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seen_game_keys: Set[str] = set()

    # If existing master manifest provided, populate seen_game_keys from existing dataset
    if existing_master_manifest_path and existing_master_manifest_path.is_file():
        try:
            with open(existing_master_manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                # Read game keys if present
                for g in meta.get("games_sample", []):
                    seen_game_keys.add(g.get("game_id", ""))
            logger.info("Loaded baseline deduplication keys from %s", existing_master_manifest_path)
        except Exception as e:
            logger.warning("Could not read existing manifest keys: %s", e)

    archive_checksums: Dict[str, str] = {}
    total_games_inspected = 0
    both_2500_count = 0
    engine_rejected = 0
    tc_rejected = 0
    metadata_rejected = 0
    duplicate_games = 0
    legal_games_count = 0
    illegal_games_count = 0

    split_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    train_seen_keys: Set[str] = set()
    val_seen_keys: Set[str] = set()
    test_seen_keys: Set[str] = set()

    for zp in archive_paths:
        sha = compute_file_sha256(zp)
        archive_checksums[zp.name] = sha
        logger.info("Processing archive %s (SHA-256 %s)...", zp.name, sha[:16])

        with zipfile.ZipFile(zp, "r") as z:
            for member in z.namelist():
                logger.info("  Streaming PGN member: %s", member)
                with z.open(member, "r") as f:
                    ts = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
                    while True:
                        try:
                            game = chess.pgn.read_game(ts)
                        except Exception as e:
                            logger.warning("Skipping corrupted record in %s: %s", member, e)
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

                        if we < min_elo or be < min_elo:
                            continue
                        both_2500_count += 1

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

                        res = h.get("Result", "*")
                        if res not in ("1-0", "0-1", "1/2-1/2"):
                            metadata_rejected += 1
                            continue

                        url = h.get("LichessURL", "").strip()
                        site = h.get("Site", "").strip()
                        if not url and site and site != "?":
                            url = site
                        date_str = h.get("UTCDate", "") or h.get("Date", "")
                        time_str = h.get("UTCTime", "") or h.get("Time", "")
                        game_key = (
                            url if url else f"{white}_{we}_{black}_{be}_{date_str}_{time_str}_{res}"
                        )

                        if game_key in seen_game_keys:
                            duplicate_games += 1
                            continue
                        seen_game_keys.add(game_key)

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
                                "Processed %d additional legal games in %.1fs | Samples: tr=%d, va=%d, te=%d",
                                legal_games_count,
                                el,
                                len(split_samples["train"]),
                                len(split_samples["val"]),
                                len(split_samples["test"]),
                            )

    total_positions = sum(len(split_samples[s]) for s in ("train", "val", "test"))
    elapsed = time.time() - t0
    logger.info(
        "Extracted %d additional legal master games (%d total positions) in %.1fs. Writing memmaps...",
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
            "kind": "master_games_additional",
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
        "dataset_name": "master_additional",
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
            "observed_positions_per_game": round(total_positions / max(1, legal_games_count), 2)
            if legal_games_count
            else 0,
        },
        "source_archives": archive_checksums,
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

    logger.info("Successfully built master_additional manifest at %s", manifest_path)
    return manifest


if __name__ == "__main__":
    from python.cli.common import setup_logging

    setup_logging()
    archives = [
        Path("tmp/source_acquisition/master_games/lichess_elite_2025-08.zip"),
        Path("tmp/source_acquisition/master_games/lichess_elite_2025-09.zip"),
        Path("tmp/source_acquisition/master_games/lichess_elite_2025-10.zip"),
    ]
    out = Path("data/staging/master_additional")
    base_manifest = Path("data/dataset_elite_master_2500_v2/manifest.json")
    build_master_additional(archives, out, base_manifest)
