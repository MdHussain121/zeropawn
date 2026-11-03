"""Merge and Consolidate All Validated Datasets into data/dataset_training/.

Consolidates:
1. data/dataset_elite_master_2500_v2/ (Existing 2500+ Master Games)
2. data/staging/master_additional/      (Additional Acquired 2500+ Master Games)
3. data/dataset_tactics_v1/             (Curated Tactical Puzzle Dataset)

Enforces all merge requirements:
1. Position-level deduplication across sources.
2. Conflict resolution: Tactical solver moves supersede master moves on tactical dilemma positions.
3. Preservation of value-target masking on tactical positions (outcome=255, ply=0 -> val_mask=0.0).
4. Zero cross-split leakage (Train vs Val, Train vs Test, Val vs Test).
5. Quarantine confirmation of 5,000-puzzle evaluation benchmark.
6. Unified binary memmap structure compatible with BinaryChessDataset and GpuResidentDataLoader.
7. Memory-efficient streaming run-length copy to handle >1.3M positions under 250MB RAM footprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def merge_unified_dataset(
    master_v2_dir: Path,
    master_add_dir: Path,
    tactics_v1_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (output_dir / s).mkdir(parents=True, exist_ok=True)

    time.time()
    component_manifests = {}
    for name, d in [
        ("master_existing", master_v2_dir),
        ("master_additional", master_add_dir),
        ("tactics_v1", tactics_v1_dir),
    ]:
        m_file = d / "manifest.json"
        if m_file.is_file():
            with open(m_file, "r", encoding="utf-8") as f:
                component_manifests[name] = json.load(f)

    logger.info("Starting consolidation into %s...", output_dir)

    # Tracks seen positions across splits to enforce 0% leakage
    # Stores 16-byte MD5 digest for minimal RAM usage
    train_seen_hashes: Set[bytes] = set()
    val_seen_hashes: Set[bytes] = set()
    test_seen_hashes: Set[bytes] = set()

    total_conflicts_resolved = 0
    total_duplicates_filtered = 0
    all_conflict_examples: List[Dict[str, Any]] = []
    split_summaries = {}

    for split in ("train", "val", "test"):
        s_t0 = time.time()
        logger.info("Merging %s split...", split)

        # Open input memmaps for each source
        sources_data = []
        for src_name, src_dir, src_type in [
            ("master_existing", master_v2_dir, 0),
            ("master_additional", master_add_dir, 0),
            ("tactics_v1", tactics_v1_dir, 1),
        ]:
            s_dir = src_dir / split
            meta_p = s_dir / "metadata.json"
            if not meta_p.is_file():
                continue
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            n = int(meta.get("num_samples", 0))
            if n == 0:
                continue

            planes_mm = np.memmap(
                s_dir / "planes.bin", dtype=np.float32, mode="r", shape=(n, 19, 8, 8)
            )
            moves_mm = np.memmap(s_dir / "moves.bin", dtype=np.int16, mode="r", shape=(n,))
            outcomes_mm = np.memmap(s_dir / "outcomes.bin", dtype=np.uint8, mode="r", shape=(n,))
            game_ids_mm = np.memmap(s_dir / "game_ids.bin", dtype=np.uint64, mode="r", shape=(n,))
            plies_mm = np.memmap(s_dir / "plies.bin", dtype=np.int16, mode="r", shape=(n,))
            val_targets_mm = np.memmap(
                s_dir / "value_targets.bin", dtype=np.float32, mode="r", shape=(n, 3)
            )

            sources_data.append(
                {
                    "name": src_name,
                    "type": src_type,
                    "n": n,
                    "planes": planes_mm,
                    "moves": moves_mm,
                    "outcomes": outcomes_mm,
                    "game_ids": game_ids_mm,
                    "plies": plies_mm,
                    "val_targets": val_targets_mm,
                }
            )

        # Step 1: Collect and resolve indices memory-efficiently.
        # chosen_entries stores (source_idx, sample_idx).
        # Position identity is the serialized 19-plane state; duplicates and
        # conflicts are CLASSIFIED and recorded (audit F-09) instead of being
        # silently resolved by first-wins/tactical precedence alone:
        #   - exact duplicate: same state, same move target, same outcome label
        #   - label conflict:  same state + move target, different outcome label
        #   - target conflict: same state, different move target
        # Conflict policy: the tactical solver record supersedes the master
        # record; a superseding or discarded conflict is counted and exampled.
        chosen_entries: List[Tuple[int, int]] = []
        current_split_pos_map: Dict[bytes, Dict[str, Any]] = {}
        split_master_count = 0
        split_tactics_count = 0
        split_stats = {
            "exact_duplicates_removed": 0,
            "label_conflicts": 0,
            "target_conflicts": 0,
            "cross_split_duplicates_filtered": 0,
        }
        conflict_examples: List[Dict[str, Any]] = []

        def _record_conflict_example(
            kind: str, kept: Dict[str, Any], dropped: Dict[str, Any]
        ) -> None:
            if len(conflict_examples) < 50:
                conflict_examples.append(
                    {"kind": kind, "split": split, "kept": kept, "dropped": dropped}
                )

        for s_idx, s_info in enumerate(sources_data):
            s_type = s_info["type"]
            n = s_info["n"]
            planes_mm = s_info["planes"]
            logger.info("Scanning %s (%d samples) in %s...", s_info["name"], n, split)

            for i in range(n):
                pos_hash = hashlib.md5(planes_mm[i].tobytes()).digest()
                cur_move = int(s_info["moves"][i])
                cur_outcome = int(s_info["outcomes"][i])

                # Cross-split leakage filter (a state seen in an earlier split
                # never enters this one).
                if split == "val" and pos_hash in train_seen_hashes:
                    split_stats["cross_split_duplicates_filtered"] += 1
                    continue
                if split == "test" and (
                    pos_hash in train_seen_hashes or pos_hash in val_seen_hashes
                ):
                    split_stats["cross_split_duplicates_filtered"] += 1
                    continue

                prev = current_split_pos_map.get(pos_hash)
                if prev is not None:
                    prev_type = sources_data[prev["s_idx"]]["type"]
                    same_move = cur_move == prev["move"]
                    same_outcome = cur_outcome == prev["outcome"]
                    if same_move and same_outcome:
                        split_stats["exact_duplicates_removed"] += 1
                        continue
                    kind = "target_conflict" if not same_move else "label_conflict"
                    if s_type == 1 and prev_type == 0:
                        # Tactical solver record supersedes the master record.
                        chosen_entries[prev["entry_idx"]] = (s_idx, i)
                        current_split_pos_map[pos_hash] = {
                            "s_idx": s_idx,
                            "sample_idx": i,
                            "entry_idx": prev["entry_idx"],
                            "move": cur_move,
                            "outcome": cur_outcome,
                        }
                        split_master_count -= 1
                        split_tactics_count += 1
                        _record_conflict_example(
                            kind,
                            {
                                "source": s_info["name"],
                                "row": i,
                                "move": cur_move,
                                "outcome": cur_outcome,
                            },
                            {
                                "source": sources_data[prev["s_idx"]]["name"],
                                "row": prev["sample_idx"],
                                "move": prev["move"],
                                "outcome": prev["outcome"],
                            },
                        )
                    else:
                        # Incoming record loses to the kept one; the conflict is
                        # still counted and exampled, never silently dropped.
                        _record_conflict_example(
                            kind,
                            {
                                "source": sources_data[prev["s_idx"]]["name"],
                                "row": prev["sample_idx"],
                                "move": prev["move"],
                                "outcome": prev["outcome"],
                            },
                            {
                                "source": s_info["name"],
                                "row": i,
                                "move": cur_move,
                                "outcome": cur_outcome,
                            },
                        )
                    if kind == "target_conflict":
                        split_stats["target_conflicts"] += 1
                    else:
                        split_stats["label_conflicts"] += 1
                    continue

                # Unique position
                entry_idx = len(chosen_entries)
                current_split_pos_map[pos_hash] = {
                    "s_idx": s_idx,
                    "sample_idx": i,
                    "entry_idx": entry_idx,
                    "move": cur_move,
                    "outcome": cur_outcome,
                }
                chosen_entries.append((s_idx, i))

                if s_type == 0:
                    split_master_count += 1
                else:
                    split_tactics_count += 1

                if split == "train":
                    train_seen_hashes.add(pos_hash)
                elif split == "val":
                    val_seen_hashes.add(pos_hash)
                elif split == "test":
                    test_seen_hashes.add(pos_hash)

        total_split_samples = len(chosen_entries)
        logger.info(
            "Split %s indexing finished in %.1fs: %d positions (%d master, %d tactical). Writing binary files...",
            split,
            time.time() - s_t0,
            total_split_samples,
            split_master_count,
            split_tactics_count,
        )

        # Step 2: Write binary files via contiguous block streaming
        out_s_dir = output_dir / split
        planes_out = np.memmap(
            out_s_dir / "planes.bin",
            dtype=np.float32,
            mode="w+",
            shape=(total_split_samples, 19, 8, 8),
        )
        moves_out = np.memmap(
            out_s_dir / "moves.bin", dtype=np.int16, mode="w+", shape=(total_split_samples,)
        )
        outcomes_out = np.memmap(
            out_s_dir / "outcomes.bin", dtype=np.uint8, mode="w+", shape=(total_split_samples,)
        )
        game_ids_out = np.memmap(
            out_s_dir / "game_ids.bin", dtype=np.uint64, mode="w+", shape=(total_split_samples,)
        )
        plies_out = np.memmap(
            out_s_dir / "plies.bin", dtype=np.int16, mode="w+", shape=(total_split_samples,)
        )
        val_targets_out = np.memmap(
            out_s_dir / "value_targets.bin",
            dtype=np.float32,
            mode="w+",
            shape=(total_split_samples, 3),
        )

        # Build contiguous run-length slices
        runs: List[Tuple[int, int, int, int]] = []
        if chosen_entries:
            curr_s_idx, curr_in_start = chosen_entries[0]
            curr_out_start = 0
            curr_count = 1
            for out_pos in range(1, total_split_samples):
                s_idx, in_idx = chosen_entries[out_pos]
                if s_idx == curr_s_idx and in_idx == curr_in_start + curr_count:
                    curr_count += 1
                else:
                    runs.append((curr_s_idx, curr_in_start, curr_out_start, curr_count))
                    curr_s_idx = s_idx
                    curr_in_start = in_idx
                    curr_out_start = out_pos
                    curr_count = 1
            runs.append((curr_s_idx, curr_in_start, curr_out_start, curr_count))

        logger.info("Executing %d contiguous copy runs for %s...", len(runs), split)
        for s_idx, in_start, out_start, count in runs:
            src = sources_data[s_idx]
            in_end = in_start + count
            out_end = out_start + count
            planes_out[out_start:out_end] = src["planes"][in_start:in_end]
            moves_out[out_start:out_end] = src["moves"][in_start:in_end]
            outcomes_out[out_start:out_end] = src["outcomes"][in_start:in_end]
            game_ids_out[out_start:out_end] = src["game_ids"][in_start:in_end]
            plies_out[out_start:out_end] = src["plies"][in_start:in_end]
            val_targets_out[out_start:out_end] = src["val_targets"][in_start:in_end]

        planes_out.flush()
        moves_out.flush()
        outcomes_out.flush()
        game_ids_out.flush()
        plies_out.flush()
        val_targets_out.flush()

        del planes_out, moves_out, outcomes_out, game_ids_out, plies_out, val_targets_out
        for s_info in sources_data:
            del (
                s_info["planes"],
                s_info["moves"],
                s_info["outcomes"],
                s_info["game_ids"],
                s_info["plies"],
                s_info["val_targets"],
            )

        split_meta = {
            "num_samples": total_split_samples,
            "planes_shape": [19, 8, 8],
            "split": split,
            "kind": "unified_training",
            "composition": {
                "master_game_positions": split_master_count,
                "tactical_puzzle_positions": split_tactics_count,
                "master_percentage": round(
                    split_master_count / max(1, total_split_samples) * 100, 2
                ),
                "tactical_percentage": round(
                    split_tactics_count / max(1, total_split_samples) * 100, 2
                ),
            },
        }
        with open(out_s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)

        split_summaries[split] = {
            "total_samples": total_split_samples,
            "master_samples": split_master_count,
            "tactical_samples": split_tactics_count,
            "planes_bytes": (out_s_dir / "planes.bin").stat().st_size,
            "moves_bytes": (out_s_dir / "moves.bin").stat().st_size,
            "outcomes_bytes": (out_s_dir / "outcomes.bin").stat().st_size,
            "exact_duplicates_removed": split_stats["exact_duplicates_removed"],
            "label_conflicts": split_stats["label_conflicts"],
            "target_conflicts": split_stats["target_conflicts"],
            "cross_split_duplicates_filtered": split_stats["cross_split_duplicates_filtered"],
        }
        total_duplicates_filtered += (
            split_stats["exact_duplicates_removed"] + split_stats["cross_split_duplicates_filtered"]
        )
        total_conflicts_resolved += split_stats["label_conflicts"] + split_stats["target_conflicts"]
        all_conflict_examples.extend(conflict_examples)

    leakage_train_val = len(train_seen_hashes.intersection(val_seen_hashes))
    leakage_train_test = len(train_seen_hashes.intersection(test_seen_hashes))
    leakage_val_test = len(val_seen_hashes.intersection(test_seen_hashes))

    total_unified = sum(s["total_samples"] for s in split_summaries.values())
    total_master = sum(s["master_samples"] for s in split_summaries.values())
    total_tactics = sum(s["tactical_samples"] for s in split_summaries.values())

    # Evaluation independence is MEASURED from the actual benchmark file and
    # the actual selected outputs (audit F-09) — never hard-coded.
    benchmark_path = Path("data/puzzles/lichess_puzzles_5000_1800_2800.csv")
    if not benchmark_path.is_file():
        benchmark_path = Path(__file__).resolve().parent.parent.parent / (
            "data/puzzles/lichess_puzzles_5000_1800_2800.csv"
        )
    if benchmark_path.is_file():
        import csv as _csv

        def _tactics_gid(puzzle_id: str) -> int:
            return int(hashlib.sha256(f"tactics_{puzzle_id}".encode("utf-8")).hexdigest()[:16], 16)

        benchmark_gids: set[int] = set()
        benchmark_rows = 0
        with open(benchmark_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row and row.get("PuzzleId"):
                    benchmark_rows += 1
                    benchmark_gids.add(_tactics_gid(row["PuzzleId"]))
        leaked_by_split: Dict[str, int] = {}
        for split, summary in split_summaries.items():
            gids = np.unique(
                np.memmap(
                    output_dir / split / "game_ids.bin",
                    dtype=np.uint64,
                    mode="r",
                    shape=(summary["total_samples"],),
                )
            )
            leaked_by_split[split] = (
                int(np.isin(np.int64(gids), np.int64(list(benchmark_gids))).sum())
                if benchmark_gids
                else 0
            )
        total_leaked = sum(leaked_by_split.values())
        evaluation_independence = {
            "benchmark_file": str(benchmark_path),
            "benchmark_puzzles": benchmark_rows,
            "identity_rule": "sha256('tactics_'+PuzzleId) game-id join against output game_ids.bin",
            "benchmark_puzzles_leaked": total_leaked,
            "leaked_by_split": leaked_by_split,
            "status": (
                "MEASURED - benchmark PuzzleIds absent from all merged splits"
                if total_leaked == 0
                else f"FAIL - {total_leaked} benchmark puzzle IDs present in merged splits"
            ),
        }
    else:
        evaluation_independence = {
            "benchmark_file": str(benchmark_path),
            "status": "NOT_VERIFIED - benchmark file not found; no independence claim is made",
        }

    # Output hashes bind the manifest to the exact published bytes.
    output_hashes: Dict[str, Dict[str, str]] = {}
    for split in ("train", "val", "test"):
        s_dir = output_dir / split
        if s_dir.is_dir():
            output_hashes[split] = {
                p.name: compute_file_sha256(p) for p in sorted(s_dir.glob("*.bin"))
            }

    manifest = {
        "dataset_name": "dataset_training",
        "description": "Unified Canonical Training Dataset containing consolidated 2500+ master-game positions and curated tactical puzzle positions.",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "schema_version": "1.1",
        "builder": "python.data.merge_unified_training_dataset.merge_unified_dataset",
        "component_sources": component_manifests,
        "consolidated_totals": {
            "total_unified_positions": total_unified,
            "master_game_positions": total_master,
            "tactical_puzzle_positions": total_tactics,
            "master_ratio": round(total_master / max(1, total_unified), 4),
            "tactics_ratio": round(total_tactics / max(1, total_unified), 4),
        },
        "deduplication_and_conflicts": {
            "identity_rule": (
                "MD5 of serialized 19-plane state; duplicates classified as exact, "
                "label-conflict, or target-conflict before resolution"
            ),
            "conflict_policy": (
                "Tactical puzzle solution record supersedes the master record on shared "
                "states; every conflict is counted and exampled in conflict_examples"
            ),
            "cross_source_duplicates_filtered": total_duplicates_filtered,
            "tactical_master_conflicts_resolved": total_conflicts_resolved,
            "per_split": {
                name: {
                    k: summary[k]
                    for k in (
                        "exact_duplicates_removed",
                        "label_conflicts",
                        "target_conflicts",
                        "cross_split_duplicates_filtered",
                    )
                    if k in summary
                }
                for name, summary in split_summaries.items()
            },
            "conflict_examples": all_conflict_examples,
        },
        "splits": split_summaries,
        "leakage_verification": {
            "train_val_overlap": leakage_train_val,
            "train_test_overlap": leakage_train_test,
            "val_test_overlap": leakage_val_test,
            "identity_rule": "MD5 of serialized 19-plane state, accumulated across split order train->val->test",
            "status": "MEASURED - ZERO LEAKAGE"
            if (leakage_train_val == 0 and leakage_train_test == 0 and leakage_val_test == 0)
            else f"FAIL - cross-split overlaps: {leakage_train_val}/{leakage_train_test}/{leakage_val_test}",
        },
        "evaluation_independence": evaluation_independence,
        "output_sha256": output_hashes,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Successfully published unified dataset manifest at %s", manifest_path)
    return manifest


if __name__ == "__main__":
    from python.cli.common import setup_logging

    setup_logging()
    mv2 = Path("data/dataset_elite_master_2500_v2")
    madd = Path("data/staging/master_additional")
    tact = Path("data/dataset_tactics_v1")
    out = Path("data/dataset_training")
    merge_unified_dataset(mv2, madd, tact, out)
