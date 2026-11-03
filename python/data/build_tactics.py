"""Convert a tactics CSV (PuzzleId,FEN,Moves,Rating,...) to binary training bins.

Every solver ply (up to ``max_solver_plies``) of every puzzle becomes one
policy example:

  - board = FEN with the opponent move(s) applied (player to move)
  - policy target = that solver move (moves.bin), value target masked
    (outcome = 255, ply = 0), so the supervised trainer's value loss is
    masked for these rows (policy-only role).

Split integrity (audit F-02/F-06):

  Puzzles are assigned to train/val candidates by the deterministic
  98/2 ``hash(PuzzleId)`` rule, then puzzle families are grouped by shared
  encoded state: every two puzzles that serialize the SAME encoded position
  (exact 19x8x8 plane bytes) are joined into one connected component, and
  every component is assigned to a single split (train if any member hashes
  to train, else val). A position family therefore can never be independently
  assigned across train and validation merely because its source PuzzleIds
  differ, and no puzzle spans both splits.

  Same-state/different-target conflicts and exact duplicates inside a split
  are detected, counted, and recorded in the build record (first-by-(PuzzleId,
  sequence) wins; never silently discarded).

  The SERIALIZED OUTPUT is re-read and exhaustively verified (zero cross-split
  encoded states, counts match) before the build reports success.

Usage:
    python python/data/build_tactics.py --csv data/tactics_train_300k.csv \\
        --output-dir data/dataset_tactics --max-solver-plies 6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    _flush_positions_chunk,
)
from python.data.dataset_contract import (
    SCHEMA_VERSION,
    BuildRecord,
    detect_git_commit,
    sha256_file,
)

SKIP_REASONS = (
    "short_line",
    "duplicate_puzzle_id",
    "illegal_opponent_move",
    "unencodable_move",
    "illegal_solution_move",
    "no_solver_positions",
    "malformed_row",
)


class _PuzzleRecord:
    __slots__ = ("puzzle_id", "gid", "row_no", "fen", "moves", "entries")

    def __init__(self, puzzle_id: str, gid: int, row_no: int, fen: str, moves: list[str]):
        self.puzzle_id = puzzle_id
        self.gid = gid
        self.row_no = row_no
        self.fen = fen
        self.moves = moves
        # entries: list of (solver_seq, pos_key_md5_bytes, move_idx)
        self.entries: list[tuple[int, bytes, int]] = []


def _collect_puzzle_records(
    csv_path: Path, max_solver_plies: int
) -> tuple[dict[str, _PuzzleRecord], dict[str, int], int]:
    """Parse the CSV once, encoding every candidate solver position to derive
    its exact serialized-state key (planes bytes MD5)."""
    import numpy as np

    puzzles: dict[str, _PuzzleRecord] = {}
    skip_reasons = {reason: 0 for reason in SKIP_REASONS}
    total_rows = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_no, row in enumerate(reader):
            total_rows += 1
            try:
                pid = (row.get("PuzzleId") or "").strip()
                fen = (row.get("FEN") or "").strip()
                moves = (row.get("Moves") or "").split()
            except Exception:
                skip_reasons["malformed_row"] += 1
                continue
            if not pid or not fen or len(moves) < 2:
                skip_reasons["short_line"] += 1
                continue
            if pid in puzzles:
                skip_reasons["duplicate_puzzle_id"] += 1
                continue

            try:
                game = zeropawn_cpp.ChessGame(fen)
            except Exception:
                skip_reasons["malformed_row"] += 1
                continue

            record = _PuzzleRecord(pid, hash_game_id(f"tactics_{pid}"), row_no, fen, moves)
            i = 0
            solver_seq = 0
            while i + 1 < len(moves) and solver_seq < max_solver_plies:
                opp_move, sol_move = moves[i], moves[i + 1]
                if not game.is_legal(opp_move):
                    skip_reasons["illegal_opponent_move"] += 1
                    break
                game.make_move_uci(opp_move)
                i += 1
                try:
                    move_idx = zeropawn_cpp.move_to_index(game, sol_move)
                except Exception:
                    move_idx = -1
                if not (0 <= move_idx < 4672):
                    skip_reasons["unencodable_move"] += 1
                    break
                if sol_move not in game.legal_moves_uci():
                    skip_reasons["illegal_solution_move"] += 1
                    break
                planes = zeropawn_cpp.encode_state(game)
                pos_key = hashlib.md5(np.ascontiguousarray(planes).tobytes()).digest()
                record.entries.append((solver_seq, pos_key, int(move_idx)))
                solver_seq += 1
                game.make_move_uci(sol_move)
                i += 1
            if solver_seq == 0:
                skip_reasons["no_solver_positions"] += 1
                continue
            puzzles[pid] = record

    return puzzles, skip_reasons, total_rows


def _assign_split_by_family(
    puzzles: dict[str, _PuzzleRecord],
) -> tuple[dict[str, str], dict[str, int]]:
    """Group puzzles into connected components by shared encoded state and
    assign each component to one split (audit F-02/F-06).

    Component rule: train if any member puzzle's own hash-based candidate is
    train (preserving the 98/2 balance), else val. Deterministic and input
    order independent. Returns (puzzle_id -> split, component size histogram).
    """
    candidate_split = {
        pid: ("val" if rec.gid % 100 < 2 else "train") for pid, rec in puzzles.items()
    }

    # Union-find over puzzles sharing an exact encoded state.
    parent: dict[str, str] = {pid: pid for pid in puzzles}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pos_owners: dict[bytes, str] = {}
    for pid, rec in puzzles.items():
        for _seq, pos_key, _move in rec.entries:
            prev = pos_owners.get(pos_key)
            if prev is None:
                pos_owners[pos_key] = pid
            elif prev != pid:
                union(prev, pid)

    component_split: dict[str, str] = {}
    for pid in puzzles:
        root = find(pid)
        if root not in component_split:
            # A component's initial split is its FIRST member's hash-based
            # candidate (not "train" — that default made every component
            # train and left the val split permanently empty, a latent bug in
            # this builder surfaced by the 2026-10-31 rebuild, audit SLQ-10).
            component_split[root] = candidate_split[pid]
        if candidate_split[pid] == "train":
            component_split[root] = "train"

    final_split = {pid: component_split[find(pid)] for pid in puzzles}

    sizes: dict[str, int] = {}
    components: dict[str, list[str]] = {}
    for pid in puzzles:
        components.setdefault(find(pid), []).append(pid)
    for members in components.values():
        sizes[str(len(members))] = sizes.get(str(len(members)), 0) + 1
    return final_split, sizes


def _detect_conflicts_and_dedup(
    puzzles: dict[str, _PuzzleRecord], split_of: dict[str, str]
) -> dict[str, dict]:
    """Within each split, resolve exact duplicates and same-state/different-target
    conflicts explicitly (first by (PuzzleId, solver_seq) wins)."""
    stats = {
        "train": {"exact_duplicates_removed": 0, "target_conflicts": 0},
        "val": {"exact_duplicates_removed": 0, "target_conflicts": 0},
    }
    conflict_examples: list[dict] = []
    kept: dict[str, dict[bytes, tuple[str, int, int]]] = {"train": {}, "val": {}}
    for pid in sorted(puzzles):
        split = split_of[pid]
        seen = kept[split]
        for seq, pos_key, move_idx in puzzles[pid].entries:
            prev = seen.get(pos_key)
            if prev is None:
                seen[pos_key] = (pid, seq, move_idx)
                continue
            prev_pid, prev_seq, prev_move = prev
            if prev_move == move_idx:
                stats[split]["exact_duplicates_removed"] += 1
            else:
                stats[split]["target_conflicts"] += 1
                if len(conflict_examples) < 20:
                    conflict_examples.append(
                        {
                            "split": split,
                            "state_md5": pos_key.hex(),
                            "kept": {
                                "puzzle_id": prev_pid,
                                "solver_seq": prev_seq,
                                "move": prev_move,
                            },
                            "discarded": {"puzzle_id": pid, "solver_seq": seq, "move": move_idx},
                        }
                    )
    return {
        "stats": stats,
        "conflict_examples": conflict_examples,
        "kept": kept,
    }


def _write_split_from_records(
    split_dir: Path,
    kept: dict[bytes, tuple[str, int, int]],
    puzzles: dict[str, _PuzzleRecord],
    chunk_size: int,
) -> int:
    """Re-encode and stream the accepted records to split part files, then
    atomically promote them. Rows are written in ascending (PuzzleId,
    solver_seq) order, so each puzzle's solution line is replayed
    incrementally exactly once. Returns (rows_written, mate_rows_labelled)."""
    import numpy as np

    from python.data.build_dataset import compute_legal_mask

    split_dir.mkdir(parents=True, exist_ok=True)
    part_paths = {
        k: split_dir / f"{k}.bin.part"
        for k in ("planes", "moves", "outcomes", "game_ids", "plies", "masks", "stm")
    }

    rows: list[tuple[str, int, bytes, int]] = []
    for pos_key, (pid, seq, move_idx) in kept.items():
        rows.append((pid, seq, pos_key, move_idx))
    rows.sort()

    n_rows = 0
    n_mate = 0
    chunk: list[PositionData] = []
    with (
        open(part_paths["planes"], "wb") as f_planes,
        open(part_paths["moves"], "wb") as f_moves,
        open(part_paths["outcomes"], "wb") as f_outcomes,
        open(part_paths["game_ids"], "wb") as f_gids,
        open(part_paths["plies"], "wb") as f_plies,
        open(part_paths["masks"], "wb") as f_masks,
        open(part_paths["stm"], "wb") as f_stm,
    ):
        current_pid: str | None = None
        game = None
        current_seq = 0
        move_idx_in_line = 0
        for pid, seq, pos_key, move_idx in rows:
            rec = puzzles[pid]
            if pid != current_pid:
                game = zeropawn_cpp.ChessGame(rec.fen)
                current_pid = pid
                current_seq = 0
                move_idx_in_line = 0
            assert game is not None
            # Advance the incremental replay up to this row's solver position.
            while current_seq <= seq:
                opp_move = rec.moves[move_idx_in_line]
                sol_move = rec.moves[move_idx_in_line + 1]
                game.make_move_uci(opp_move)
                move_idx_in_line += 1
                row: PositionData | None = None
                if current_seq == seq:
                    planes = zeropawn_cpp.encode_state(game)
                    actual_key = hashlib.md5(np.ascontiguousarray(planes).tobytes()).digest()
                    if actual_key != pos_key:
                        raise RuntimeError(
                            f"Re-encode mismatch for puzzle {pid} solver_seq {seq}: "
                            "serialized state changed between collection and write"
                        )
                    # Schema-2 sidecars at the exact stored position: the
                    # packed legal mask (masked policy loss) and the side to
                    # move (held-out STM resolution; the 19-plane encoder is
                    # side-relative).
                    row = PositionData(
                        planes=planes,
                        move_idx=move_idx,
                        outcome=OUTCOME_UNAVAILABLE,
                        game_id=rec.gid,
                        ply=0,
                        pos_hash=0,
                        legal_mask=compute_legal_mask(game),
                        stm=1 if game.side_to_move() else 0,
                    )
                    chunk.append(row)
                    n_rows += 1
                    if len(chunk) >= chunk_size:
                        _flush_positions_chunk(
                            chunk,
                            f_planes,
                            f_moves,
                            f_outcomes,
                            f_gids,
                            f_plies,
                            f_masks=f_masks,
                            f_stm=f_stm,
                        )
                        chunk.clear()
                game.make_move_uci(sol_move)
                move_idx_in_line += 1
                current_seq += 1
                # Mate contract (audit SLQ-06, builder side): when the solver
                # move delivers checkmate the outcome IS available and must
                # not be silently discarded — label the row as a mover-relative
                # win at ply 10 so the trainer's WDL mask (ply >= 10 and non-zero
                # target) supervises the value head. The line ends at mate.
                if row is not None and game.is_checkmate():
                    row.outcome = 0
                    row.ply = 10
                    n_mate += 1
                    break
        if chunk:
            _flush_positions_chunk(
                chunk, f_planes, f_moves, f_outcomes, f_gids, f_plies, f_masks=f_masks, f_stm=f_stm
            )
            chunk.clear()

    for name, part_path in part_paths.items():
        final_path = split_dir / f"{name}.bin"
        if final_path.exists():
            final_path.unlink()
        part_path.replace(final_path)
    return n_rows, n_mate


def _verify_serialized_output(output_dir: Path, n_train: int, n_val: int) -> dict:
    """Exhaustive final verification of the SERIALIZED artifacts: re-read both
    splits' planes row-wise and assert zero cross-split encoded states."""
    import numpy as np

    def row_hashes(split_dir: Path, n: int) -> set[bytes]:
        if n == 0:
            return set()
        mm = np.memmap(split_dir / "planes.bin", dtype=np.float32, mode="r", shape=(n, 19, 8, 8))
        keys: set[bytes] = set()
        chunk = 2048
        try:
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                block = np.ascontiguousarray(mm[start:end])
                for offset in range(end - start):
                    keys.add(hashlib.md5(block[offset].tobytes()).digest())
        finally:
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        return keys

    train_keys = row_hashes(output_dir / "train", n_train)
    val_keys = row_hashes(output_dir / "val", n_val)
    overlap = train_keys & val_keys
    return {
        "exhaustive": True,
        "train_unique_states": len(train_keys),
        "val_unique_states": len(val_keys),
        "cross_split_overlap": len(overlap),
        "passed": len(overlap) == 0,
    }


def build_tactics_dataset(
    csv_path: str | Path,
    output_dir: str | Path,
    max_solver_plies: int = 4,
    chunk_size: int = 10_000,
) -> tuple[int, int]:
    """Build the tactics dataset with position-family split integrity."""
    csv_path = Path(csv_path)
    out_dir = Path(output_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    source_sha = sha256_file(csv_path)
    puzzles, skip_reasons, total_rows = _collect_puzzle_records(csv_path, max_solver_plies)
    print(f"Parsed {len(puzzles):,} puzzles from {total_rows:,} rows; skip reasons: {skip_reasons}")

    split_of, component_sizes = _assign_split_by_family(puzzles)
    n_cross_family = sum(
        1
        for pid, rec in puzzles.items()
        if ("val" if rec.gid % 100 < 2 else "train") != split_of[pid]
    )
    if n_cross_family:
        print(
            f"Position-family integrity: {n_cross_family:,} puzzles reassigned across the "
            "hash-based candidate split to keep shared encoded states in one split"
        )

    conflict_report = _detect_conflicts_and_dedup(puzzles, split_of)
    for split in ("train", "val"):
        s = conflict_report["stats"][split]
        print(
            f"{split}: exact duplicates removed {s['exact_duplicates_removed']:,}, "
            f"target conflicts {s['target_conflicts']:,}"
        )

    n_train, train_mate_rows = _write_split_from_records(
        train_dir, conflict_report["kept"]["train"], puzzles, chunk_size
    )
    n_val, val_mate_rows = _write_split_from_records(
        val_dir, conflict_report["kept"]["val"], puzzles, chunk_size
    )
    print(
        f"Serialized train {n_train:,} rows ({train_mate_rows:,} mate-labelled), "
        f"val {n_val:,} rows ({val_mate_rows:,} mate-labelled)"
    )

    verification = _verify_serialized_output(out_dir, n_train, n_val)
    if not verification["passed"]:
        raise RuntimeError(
            "Final serialized-output verification FAILED: "
            f"{verification['cross_split_overlap']} encoded states cross the train/val boundary"
        )
    print(
        f"Final serialized verification PASSED: 0 cross-split states "
        f"(train {verification['train_unique_states']:,} unique, val {verification['val_unique_states']:,} unique)"
    )

    label_semantics = {
        "role": "policy_tactics",
        "outcome_encoding": (
            "mover perspective: 0=win (solver move delivers checkmate, ply=10), "
            "255=masked (policy-only example, ply=0)"
        ),
        "wdl_supervised": False,
        "value_loss": "masked except mate-delivering rows (outcome=0, ply=10) per the SLQ-06 contract",
    }
    metadata_common = {
        "planes_shape": [19, 8, 8],
        "dtype_planes": "float32",
        "dtype_moves": "int16",
        "dtype_outcomes": "uint8",
        "dtype_game_ids": "uint64",
        "dtype_plies": "int16",
        "dtype_stm": "uint8 (1 = white to move, 0 = black)",
        "dtype_masks": "uint8-bitpacked-4672",
        "schema_version": SCHEMA_VERSION,
        "role": "policy_tactics",
        "label_semantics": label_semantics,
        "kind": "tactics-fullline",
        "source": str(csv_path),
        "source_sha256": source_sha,
    }
    with open(train_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"split": "train", "num_samples": n_train, **metadata_common}, f, indent=2)
    with open(val_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"split": "val", "num_samples": n_val, **metadata_common}, f, indent=2)

    output_hashes = {"train": {}, "val": {}}
    for split_name, split_dir in (("train", train_dir), ("val", val_dir)):
        for array_file in sorted(split_dir.glob("*.bin")) + [split_dir / "metadata.json"]:
            output_hashes[split_name][array_file.name] = sha256_file(array_file)

    BuildRecord(
        builder="python.data.build_tactics.build_tactics_dataset",
        source={"path": str(csv_path), "sha256": source_sha, "rows": total_rows},
        config={
            "max_solver_plies": max_solver_plies,
            "split_rule": "hash_game_id('tactics_'+PuzzleId) % 100 < 2 -> val candidate; "
            "puzzle families connected by shared encoded states assigned to one split "
            "(train if any member is a train candidate)",
            "write_order": "(PuzzleId, solver_seq) ascending",
            "conflict_policy": "same-state/different-target: first by (PuzzleId, solver_seq) wins; "
            "all conflicts counted and exampled in this record",
        },
        label_semantics=label_semantics,
        counts={
            "csv_rows": total_rows,
            "puzzles_accepted": len(puzzles),
            "puzzles_reassigned_for_family_integrity": n_cross_family,
            "component_size_histogram": component_sizes,
            "skip_reasons": skip_reasons,
            "train_rows": n_train,
            "train_mate_rows": train_mate_rows,
            "val_rows": n_val,
            "val_mate_rows": val_mate_rows,
            **{
                f"{split}_{key}": value
                for split in ("train", "val")
                for key, value in conflict_report["stats"][split].items()
            },
        },
        outputs=output_hashes,
        git_commit=detect_git_commit(),
        notes={
            "serialized_output_verification": verification,
            "target_conflict_examples": conflict_report["conflict_examples"],
        },
    ).write(out_dir)

    return n_train, n_val


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
