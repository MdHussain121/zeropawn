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
from typing import Any, Optional, TextIO
import numpy as np
import chess.pgn

logger = logging.getLogger(__name__)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp
from python.cli.common import setup_logging
from python.constants import BOARD_SIZE, NUM_ACTIONS, NUM_PLANES
from python.data.dataset_contract import (
    SCHEMA_VERSION,
    BuildRecord,
    detect_git_commit,
    sha256_file,
)
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


def extract_game_id(game: chess.pgn.Game, mainline_uci: list[str] | None = None) -> str:
    """Extract a stable, collision-resistant game identity.

    Sources with a URL ``Site`` header use the URL's last path segment (stable
    source ID). Otherwise the identity combines the game headers with a digest
    of the canonical mainline content, so distinct games sharing the same
    date and player names (audit F-03) no longer collapse into one group while
    genuinely identical duplicate games still resolve to the same identity.

    ``mainline_uci`` is the ordered UCI move list of the game's mainline; when
    omitted (legacy callers without move context) the ambiguous headers-only
    fallback is returned unchanged.
    """
    site = game.headers.get("Site", "")
    if site.startswith("http"):
        return site.rstrip("/").split("/")[-1]
    w = game.headers.get("White", "W")
    b = game.headers.get("Black", "B")
    d = game.headers.get("Date", "D")
    if mainline_uci is None:
        return f"{d}_{w}_{b}"
    content_digest = hashlib.sha256(" ".join(mainline_uci).encode("utf-8")).hexdigest()[:16]
    return f"{d}_{w}_{b}#{content_digest}"


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
    __slots__ = (
        "planes",
        "move_idx",
        "outcome",
        "game_id",
        "ply",
        "pos_hash",
        "state_key",
        "legal_mask",
        "stm",
    )

    # NUM_ACTIONS actions packed into bits: ceil(NUM_ACTIONS/8) bytes per position.
    MASK_NUM_ACTIONS = NUM_ACTIONS
    MASK_PACKED_BYTES = (NUM_ACTIONS + 7) // 8

    def __init__(
        self,
        planes: np.ndarray,
        move_idx: int,
        outcome: int,
        game_id: int,
        ply: int,
        pos_hash: int,
        state_key: bytes | None = None,
        legal_mask: Optional[np.ndarray] = None,
        stm: int = 1,
    ) -> None:
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Invalid outcome {outcome}: must be one of {VALID_OUTCOMES}")
        self.planes = planes
        self.move_idx = move_idx
        self.outcome: int = outcome
        self.game_id = game_id
        self.ply = ply
        self.pos_hash = pos_hash
        # Exact serialized-state identity: MD5 of the plane bytes. The C++
        # engine hash distinguishes rule state the 19-plane encoder renders
        # identically, so val dedup MUST key on the serialized bytes or
        # identical encoded states survive across splits (audit finding).
        self.state_key = (
            state_key
            if state_key is not None
            else hashlib.md5(np.ascontiguousarray(planes).tobytes()).digest()
        )
        self.legal_mask = legal_mask
        # Side to move: 1 = white, 0 = black. The 19-plane encoder is fully
        # side-relative (no absolute side plane), so the side cannot be
        # recovered from the planes alone — store it explicitly so held-out
        # evaluation never has to guess it from ply parity (audit SLQ-13;
        # ply parity is invalid for FEN-rooted tactical rows).
        self.stm = 1 if stm else 0


def compute_legal_mask(cpp_game) -> np.ndarray:
    """Packed bit mask of legal action indices from a live C++ ChessGame."""
    legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    actions = cpp_game.legal_actions()
    idx = np.asarray(actions, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= NUM_ACTIONS):
        raise ValueError(f"legal_actions returned out-of-range indices: {idx.min()}..{idx.max()}")
    legal[idx] = True
    return np.packbits(legal)  # uint8, 584 bytes


def process_pgn_game(
    game: chess.pgn.Game,
    diag_out: Optional[dict[str, Any]] = None,
) -> tuple[int, str, list[PositionData]]:
    """Process a single chess.pgn.Game using the C++ ChessGame and encoder.

    Splits games deterministically by game-ID hash into three buckets
    (audit SLQ-09): ``val`` = hash % 100 < 2 (unchanged from the historical
    rule, so previously-val games stay val), ``test`` = 2 <= hash % 100 < 3
    (an independent held-out test set, never used for selection), and
    ``train`` = the remaining 97 %.

    Returns:
        (game_id_hash, split_name, list of PositionData)
    """
    if diag_out is not None:
        diag_out["unparsable_result"] = False
        diag_out["truncated_replay"] = False
        diag_out["game_errors"] = len(game.errors) if hasattr(game, "errors") else 0
        diag_out["header_encoding_error"] = any(
            "\ufffd" in str(v)
            for v in (
                game.headers.get("Date", ""),
                game.headers.get("White", ""),
                game.headers.get("Black", ""),
                game.headers.get("Site", ""),
            )
        )

    # Snapshot the canonical mainline first so the fallback game identity can
    # bind to game content (collision-resistant, audit F-03).
    mainline_uci = [move.uci() for move in game.mainline_moves()]
    game_id_str = extract_game_id(game, mainline_uci)
    game_id = hash_game_id(game_id_str)
    # Deterministic 97% train / 2% val / 1% test split by game ID hash
    bucket = game_id % 100
    if bucket < 2:
        split = "val"
    elif bucket < 3:
        split = "test"
    else:
        split = "train"

    raw_res = game.headers.get("Result", "")
    res = parse_result(raw_res)
    if res is None:
        if diag_out is not None:
            diag_out["unparsable_result"] = True
        return game_id, split, []

    positions: list[PositionData] = []
    cpp_game = zeropawn_cpp.ChessGame()

    for ply_num, uci_move in enumerate(mainline_uci):
        planes = zeropawn_cpp.encode_state(cpp_game)
        move_idx = zeropawn_cpp.move_to_index(cpp_game, uci_move)

        if move_idx < 0 or move_idx >= NUM_ACTIONS:
            # Move could not be mapped (e.g. malformed), stop game replay (DI-19)
            if diag_out is not None:
                diag_out["truncated_replay"] = True
            break

        legal_mask = compute_legal_mask(cpp_game)

        # Data-integrity gate (SL-01): the training label must be a legal
        # action of the position it is stored with. A violation means the
        # replay or the action mapping is broken — fail loudly instead of
        # training on (or smoothing away) an impossible move.
        unpacked = np.unpackbits(legal_mask)
        if not unpacked[move_idx]:
            raise ValueError(
                f"Illegal training label in game {game_id_str} at ply {ply_num}: "
                f"action {move_idx} ({uci_move}) is not legal in the stored position."
            )

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
                legal_mask=legal_mask,
                stm=1 if is_white else 0,
            )
        )

        cpp_game.make_move_uci(uci_move)

    if diag_out is not None and len(positions) < len(mainline_uci):
        diag_out["truncated_replay"] = True

    return game_id, split, positions


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
    masks_path = data_dir / "masks.bin"
    stm_path = data_dir / "stm.bin"
    has_masks = all(p.legal_mask is not None for p in positions)

    planes_mm = np.memmap(
        planes_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_samples, NUM_PLANES, BOARD_SIZE, BOARD_SIZE),
    )
    moves_mm = np.memmap(moves_path, dtype=np.int16, mode="w+", shape=(num_samples,))
    outcomes_mm = np.memmap(outcomes_path, dtype=np.uint8, mode="w+", shape=(num_samples,))
    game_ids_mm = np.memmap(game_ids_path, dtype=np.uint64, mode="w+", shape=(num_samples,))
    plies_mm = np.memmap(plies_path, dtype=np.int16, mode="w+", shape=(num_samples,))
    masks_mm = (
        np.memmap(
            masks_path,
            dtype=np.uint8,
            mode="w+",
            shape=(num_samples, PositionData.MASK_PACKED_BYTES),
        )
        if has_masks
        else None
    )
    stm_mm = np.memmap(stm_path, dtype=np.uint8, mode="w+", shape=(num_samples,))

    try:
        for i, pos in enumerate(positions):
            planes_mm[i] = pos.planes
            moves_mm[i] = pos.move_idx
            outcomes_mm[i] = pos.outcome
            game_ids_mm[i] = pos.game_id
            plies_mm[i] = pos.ply
            stm_mm[i] = pos.stm
            if masks_mm is not None:
                masks_mm[i] = pos.legal_mask

        planes_mm.flush()
        moves_mm.flush()
        outcomes_mm.flush()
        game_ids_mm.flush()
        plies_mm.flush()
        stm_mm.flush()
        if masks_mm is not None:
            masks_mm.flush()
    finally:
        for mm in (planes_mm, moves_mm, outcomes_mm, game_ids_mm, plies_mm, masks_mm, stm_mm):
            if mm is None:
                continue
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("memmap close failed", exc_info=True)

    metadata = {
        "split": split_name,
        "num_samples": num_samples,
        "planes_shape": [NUM_PLANES, BOARD_SIZE, BOARD_SIZE],
        "dtype_planes": "float32",
        "dtype_moves": "int16",
        "dtype_outcomes": "uint8",
        "dtype_game_ids": "uint64",
        "dtype_plies": "int16",
        "dtype_stm": "uint8 (1 = white to move, 0 = black)",
        # SCHEMA_VERSION constant (not a hardcoded 2): the validator compares
        # this field against dataset_contract.SCHEMA_VERSION, and the
        # discrepancy made helper-built splits fail validation (SLQ-08).
        "schema_version": SCHEMA_VERSION,
        # Role declaration (SL-01/F-04 contract): the games builder produces
        # mover-relative WDL supervision for every replayed position.
        "role": "games_wdl",
        "label_semantics": {
            "role": "games_wdl",
            "outcome_encoding": "mover perspective: 0=win, 1=draw, 2=loss, 255=masked",
            "wdl_supervised": True,
        },
    }
    if has_masks:
        metadata["dtype_masks"] = "uint8-bitpacked-4672"
        metadata["masks_shape"] = [num_samples, PositionData.MASK_PACKED_BYTES]
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
    f_masks: Any = None,
    f_stm: Any = None,
) -> None:
    """Pack and flush a chunk of PositionData to binary file handles."""
    n = len(positions)
    if n == 0:
        return
    planes_arr = np.empty((n, NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    moves_arr = np.empty(n, dtype=np.int16)
    outcomes_arr = np.empty(n, dtype=np.uint8)
    game_ids_arr = np.empty(n, dtype=np.uint64)
    plies_arr = np.empty(n, dtype=np.int16)
    masks_arr = (
        np.empty((n, PositionData.MASK_PACKED_BYTES), dtype=np.uint8)
        if f_masks is not None
        else None
    )
    stm_arr = np.empty(n, dtype=np.uint8) if f_stm is not None else None

    for i, p in enumerate(positions):
        planes_arr[i] = p.planes
        moves_arr[i] = p.move_idx
        outcomes_arr[i] = p.outcome
        game_ids_arr[i] = p.game_id
        plies_arr[i] = p.ply
        if stm_arr is not None:
            stm_arr[i] = p.stm
        if masks_arr is not None:
            if p.legal_mask is None:
                raise ValueError("masks stream provided but a position lacks a legal mask")
            masks_arr[i] = p.legal_mask

    f_planes.write(planes_arr.tobytes())
    f_moves.write(moves_arr.tobytes())
    f_outcomes.write(outcomes_arr.tobytes())
    f_game_ids.write(game_ids_arr.tobytes())
    f_plies.write(plies_arr.tobytes())
    if f_stm is not None:
        f_stm.write(stm_arr.tobytes())
    if f_masks is not None:
        f_masks.write(masks_arr.tobytes())


def build_binary_dataset(
    pgn_path_or_stream: str | Path | TextIO,
    output_dir: str | Path,
    max_games: int | None = None,
    progress_interval: int = 0,
    chunk_size: int = 5_000,
    force: bool = False,
) -> tuple[int, int, int]:
    """Parse PGN, split 98/2 by game hash, deduplicate val positions, and save.

    Uses a streaming bounded-chunk pipeline to keep memory strictly bounded
    regardless of input PGN size. Writes partial files (.part) and atomically
    promotes them only after verification.
    """
    import shutil

    out_dir = Path(output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory {out_dir} already exists and is non-empty. "
            "Pass force=True (or CLI --force) to overwrite, or build into a staging directory."
        )

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Pre-check disk space (at least 200 MB free)
    try:
        free_bytes = shutil.disk_usage(out_dir if out_dir.exists() else out_dir.parent).free
        if free_bytes < 200 * 1024 * 1024:
            raise OSError(
                f"Insufficient disk space ({free_bytes / 1e6:.1f} MB available, need >= 200 MB)"
            )
    except Exception as e:
        logger.debug("Disk usage check warning: %s", e)

    def _part_paths(split_dir: Path) -> dict[str, Path]:
        return {
            "planes": split_dir / "planes.bin.part",
            "moves": split_dir / "moves.bin.part",
            "outcomes": split_dir / "outcomes.bin.part",
            "game_ids": split_dir / "game_ids.bin.part",
            "plies": split_dir / "plies.bin.part",
            "masks": split_dir / "masks.bin.part",
            "stm": split_dir / "stm.bin.part",
        }

    train_part_paths = _part_paths(train_dir)

    train_chunk: list[PositionData] = []
    candidate_val_positions: list[PositionData] = []
    candidate_test_positions: list[PositionData] = []
    # Serialized-state identity of every accepted train position: val dedup
    # keys on the exact encoded bytes, not the C++ engine hash (which can
    # distinguish states the encoder renders identically).
    train_state_keys: set[bytes] = set()

    train_games_count = 0
    val_games_count = 0
    test_games_count = 0
    total_games_seen = 0
    n_train = 0
    # Exact-duplicate game handling (audit F-03): games with identical content
    # identity are processed once; conflicting results are counted, never mixed.
    seen_games: dict[int, str] = {}
    duplicate_games = 0
    conflicting_duplicate_games = 0

    # Silent skip and replay error accounting (DI-19)
    unparsable_results_count = 0
    truncated_replays_count = 0
    game_errors_count = 0
    header_encoding_errors_count = 0

    try:
        with (
            open(train_part_paths["planes"], "wb") as f_tr_planes,
            open(train_part_paths["moves"], "wb") as f_tr_moves,
            open(train_part_paths["outcomes"], "wb") as f_tr_outcomes,
            open(train_part_paths["game_ids"], "wb") as f_tr_game_ids,
            open(train_part_paths["plies"], "wb") as f_tr_plies,
            open(train_part_paths["masks"], "wb") as f_tr_masks,
            open(train_part_paths["stm"], "wb") as f_tr_stm,
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

                    diag: dict[str, Any] = {}
                    game_id, split, pos_list = process_pgn_game(game, diag_out=diag)
                    if diag.get("unparsable_result"):
                        unparsable_results_count += 1
                    if diag.get("truncated_replay"):
                        truncated_replays_count += 1
                    if diag.get("game_errors", 0) > 0:
                        game_errors_count += 1
                    if diag.get("header_encoding_error"):
                        header_encoding_errors_count += 1

                    if not pos_list:
                        continue

                    res_raw = game.headers.get("Result", "")
                    if game_id in seen_games:
                        if seen_games[game_id] == res_raw:
                            duplicate_games += 1
                        else:
                            conflicting_duplicate_games += 1
                        continue
                    seen_games[game_id] = res_raw

                    if split == "val":
                        val_games_count += 1
                        candidate_val_positions.extend(pos_list)
                    elif split == "test":
                        test_games_count += 1
                        candidate_test_positions.extend(pos_list)
                    else:
                        train_games_count += 1
                        for p in pos_list:
                            train_state_keys.add(p.state_key)
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
                                f_masks=f_tr_masks,
                                f_stm=f_tr_stm,
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
                                    ("test_games", test_games_count),
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
                    f_masks=f_tr_masks,
                    f_stm=f_tr_stm,
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
            "planes_shape": [NUM_PLANES, BOARD_SIZE, BOARD_SIZE],
            "dtype_planes": "float32",
            "dtype_moves": "int16",
            "dtype_outcomes": "uint8",
            "dtype_game_ids": "uint64",
            "dtype_plies": "int16",
            "dtype_stm": "uint8 (1 = white to move, 0 = black)",
            "schema_version": SCHEMA_VERSION,
            "role": "games_wdl",
            "label_semantics": {
                "role": "games_wdl",
                "outcome_encoding": "mover perspective: 0=win, 1=draw, 2=loss, 255=masked",
                "wdl_supervised": True,
            },
            "split_rule": (
                "hash_game_id(game_id) % 100: <2 -> val, 2..<3 -> test (independent "
                "held-out set, audit SLQ-09), else train"
            ),
            "game_identity_rule": (
                "sha256 game identity: Site URL last segment when present, else "
                "Date_White_Black + sha256(mainline UCI) digest (audit F-03)"
            ),
        }
        with open(train_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(train_metadata, f, indent=2)

        # Deduplicate validation positions against training positions by the
        # exact serialized state identity, AND within-val deduplicate (DI-29).
        final_val_positions: list[PositionData] = []
        val_seen_keys: set[bytes] = set()
        val_removed_cross = 0
        val_removed_within = 0
        for pos in candidate_val_positions:
            if pos.state_key in train_state_keys:
                val_removed_cross += 1
            elif pos.state_key in val_seen_keys:
                val_removed_within += 1
            else:
                val_seen_keys.add(pos.state_key)
                final_val_positions.append(pos)

        n_val_final = len(final_val_positions)
        removed_count = val_removed_cross + val_removed_within

        # The test split must be independent of BOTH train and the final val
        # split (audit SLQ-09), AND within-test deduplicate (DI-32).
        val_state_keys = val_seen_keys
        final_test_positions: list[PositionData] = []
        test_seen_keys: set[bytes] = set()
        test_removed_cross = 0
        test_removed_within = 0
        for pos in candidate_test_positions:
            if pos.state_key in train_state_keys or pos.state_key in val_state_keys:
                test_removed_cross += 1
            elif pos.state_key in test_seen_keys:
                test_removed_within += 1
            else:
                test_seen_keys.add(pos.state_key)
                final_test_positions.append(pos)

        n_test_final = len(final_test_positions)
        test_removed_count = test_removed_cross + test_removed_within

        def _write_candidate_split(
            split_dir: Path,
            split_name: str,
            positions: list[PositionData],
            candidate_count: int,
            dedup_removed: int,
            dedup_removed_cross: int = 0,
            dedup_removed_within: int = 0,
        ) -> None:
            part_paths = _part_paths(split_dir)
            with (
                open(part_paths["planes"], "wb") as fp,
                open(part_paths["moves"], "wb") as fm,
                open(part_paths["outcomes"], "wb") as fo,
                open(part_paths["game_ids"], "wb") as fg,
                open(part_paths["plies"], "wb") as fl,
                open(part_paths["masks"], "wb") as fk,
                open(part_paths["stm"], "wb") as fs,
            ):
                for start_idx in range(0, len(positions), chunk_size):
                    end_idx = min(start_idx + chunk_size, len(positions))
                    _flush_positions_chunk(
                        positions[start_idx:end_idx],
                        fp,
                        fm,
                        fo,
                        fg,
                        fl,
                        f_masks=fk,
                        f_stm=fs,
                    )
            for key, part_p in part_paths.items():
                final_p = split_dir / f"{key}.bin"
                if part_p.exists():
                    part_p.replace(final_p)

            metadata = {
                "split": split_name,
                "num_samples": len(positions),
                "planes_shape": [NUM_PLANES, BOARD_SIZE, BOARD_SIZE],
                "dtype_planes": "float32",
                "dtype_moves": "int16",
                "dtype_outcomes": "uint8",
                "dtype_game_ids": "uint64",
                "dtype_plies": "int16",
                "dtype_stm": "uint8 (1 = white to move, 0 = black)",
                "schema_version": SCHEMA_VERSION,
                "role": "games_wdl",
                "label_semantics": {
                    "role": "games_wdl",
                    "outcome_encoding": "mover perspective: 0=win, 1=draw, 2=loss, 255=masked",
                    "wdl_supervised": True,
                },
                "split_rule": (
                    "hash_game_id(game_id) % 100: <2 -> val, 2..<3 -> test (independent "
                    "held-out set, audit SLQ-09), else train"
                ),
                "game_identity_rule": (
                    "sha256 game identity: Site URL last segment when present, else "
                    "Date_White_Black + sha256(mainline UCI) digest (audit F-03)"
                ),
                "candidate_positions": candidate_count,
                "dedup_removed_positions": dedup_removed,
                "dedup_removed_cross": dedup_removed_cross,
                "dedup_removed_within": dedup_removed_within,
                "dedup_rule": (
                    "positions whose exact serialized-state identity (MD5 of encoded "
                    "planes) occurs in an earlier split or earlier within the split are dropped (DI-29/DI-32)"
                ),
            }
            with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        _write_candidate_split(
            val_dir,
            "val",
            final_val_positions,
            len(candidate_val_positions),
            removed_count,
            dedup_removed_cross=val_removed_cross,
            dedup_removed_within=val_removed_within,
        )
        _write_candidate_split(
            test_dir,
            "test",
            final_test_positions,
            len(candidate_test_positions),
            test_removed_count,
            dedup_removed_cross=test_removed_cross,
            dedup_removed_within=test_removed_within,
        )

    except Exception:
        # Clean up partial files on failure
        for split_dir in (train_dir, val_dir, test_dir):
            for p in split_dir.glob("*.bin.part"):
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
    print(
        f"Test games:                      {test_games_count} ({(test_games_count / max(1, total_games_seen) * 100):.1f}%)"
    )
    print(f"Training positions:              {n_train}")
    print(f"Candidate validation positions:  {len(candidate_val_positions)}")
    pct_removed = (removed_count / max(1, len(candidate_val_positions))) * 100
    print(
        f"Validation positions removed:    {removed_count} ({pct_removed:.2f}%) [cross: {val_removed_cross}, within-val: {val_removed_within}]"
    )
    print(f"Final validation positions:      {n_val_final}")
    print(f"Candidate test positions:        {len(candidate_test_positions)}")
    print(
        f"Test positions removed:          {test_removed_count} [cross: {test_removed_cross}, within-test: {test_removed_within}]"
    )
    print(f"Final test positions:            {n_test_final}")
    print(f"Duplicate games skipped:         {duplicate_games}")
    print(f"Conflicting duplicate games:     {conflicting_duplicate_games}")
    print(f"Unparsable results skipped:      {unparsable_results_count}")
    print(f"Truncated replays:               {truncated_replays_count}")
    print(f"Game errors encountered:         {game_errors_count}")
    print(f"Header encoding errors:          {header_encoding_errors_count}")
    print("=" * 60 + "\n")

    # Provenance build record (audit F-04 Layer C): binds this build to its
    # exact source bytes, configuration, measured counts, and output hashes.
    if isinstance(pgn_path_or_stream, (str, Path)):
        src_path = Path(pgn_path_or_stream)
        source_identity: dict[str, Any] = {
            "path": str(src_path),
            "sha256": sha256_file(src_path),
            "size": src_path.stat().st_size,
        }
    else:
        source_identity = {"kind": "TextIO stream (not hashable)"}

    output_hashes: dict[str, dict[str, str]] = {"train": {}, "val": {}, "test": {}}
    for split_name, split_dir in (("train", train_dir), ("val", val_dir), ("test", test_dir)):
        for array_file in sorted(split_dir.glob("*.bin")) + [split_dir / "metadata.json"]:
            if array_file.is_file():
                output_hashes[split_name][array_file.name] = sha256_file(array_file)

    BuildRecord(
        builder="python.data.build_dataset.build_binary_dataset",
        source=source_identity,
        config={
            "split_rule": (
                "hash_game_id(game_id) % 100: <2 -> val, 2..<3 -> test (independent "
                "held-out set, audit SLQ-09), else train"
            ),
            "val_dedup": "val positions whose exact serialized-state identity occurs in train or within val are dropped",
            "test_dedup": "test positions whose exact serialized-state identity occurs in train, val, or within test are dropped",
            "stm_sidecar": "stm.bin uint8 (1 = white to move, 0 = black), stored per position because the 19-plane encoder is side-relative",
            "max_games": max_games,
            "chunk_size": chunk_size,
            "force": force,
        },
        label_semantics={
            "role": "games_wdl",
            "outcome_encoding": "mover perspective: 0=win, 1=draw, 2=loss, 255=masked",
            "wdl_supervised": True,
            "outcome_source": "game Result header mapped to the side to move at each ply",
        },
        counts={
            "total_games_seen": total_games_seen,
            "train_games": train_games_count,
            "val_games": val_games_count,
            "test_games": test_games_count,
            "train_positions": n_train,
            "candidate_val_positions": len(candidate_val_positions),
            "dedup_removed_val_positions": removed_count,
            "dedup_removed_val_cross": val_removed_cross,
            "dedup_removed_val_within": val_removed_within,
            "final_val_positions": n_val_final,
            "candidate_test_positions": len(candidate_test_positions),
            "dedup_removed_test_positions": test_removed_count,
            "dedup_removed_test_cross": test_removed_cross,
            "dedup_removed_test_within": test_removed_within,
            "final_test_positions": n_test_final,
            "duplicate_games_skipped": duplicate_games,
            "conflicting_duplicate_games_skipped": conflicting_duplicate_games,
            "unparsable_results_skipped": unparsable_results_count,
            "truncated_replays": truncated_replays_count,
            "game_errors": game_errors_count,
            "header_encoding_errors": header_encoding_errors_count,
        },
        outputs=output_hashes,
        git_commit=detect_git_commit(),
    ).write(out_dir)

    del candidate_val_positions, final_val_positions, train_state_keys
    gc.collect()

    return n_train, n_val_final, removed_count


def main() -> None:
    # Single logging setup: rotating file for everything, WARNING+ on stderr.
    setup_logging(log_dir="logs", log_filename="build_dataset.log")
    ctx = run_context("data-convert")
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Convert filtered PGN to binary memmap dataset.")
    parser.add_argument("--pgn", type=str, required=True, help="Input filtered PGN file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (explicit target required; no production default, audit SA2-02).",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process.")
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite non-empty output directory.",
    )

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
                ("output_dir", args.output_dir, "user"),
                ("max_games", args.max_games, "user" if args.max_games is not None else "default"),
                ("force", args.force, "user" if args.force else "default"),
            ]
        )
    )

    try:
        build_binary_dataset(
            args.pgn,
            args.output_dir,
            max_games=args.max_games,
            progress_interval=1000,
            force=args.force,
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
