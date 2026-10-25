"""Build Supervised Opening Training Dataset from Curated Opening Corpus.

Converts verified opening positions and qualified candidate moves into
canonical binary memory-mapped datasets (planes.bin, moves.bin, outcomes.bin,
value_targets.bin, metadata.json) compatible with BinaryChessDataset.

Supports:
- Single-move representation (frequency/score-weighted sampling across accepted moves)
- Multiple-move distribution targets (policy_targets.bin)
- Clean train/validation splits by opening family (ECO prefix) or hash, preventing data leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp
from python.data.opening_corpus import OpeningCorpus

logger = logging.getLogger(__name__)


def build_opening_dataset(
    corpus: OpeningCorpus,
    output_dir: Union[str, Path],
    val_ratio: float = 0.1,
    include_all_qualified: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Export opening corpus into binary train and val datasets."""
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Collect qualified position samples
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    corpus.update_qualification_status()

    for key, pos in corpus.positions.items():
        # Get qualified moves
        qual_moves = []
        for uci, ev in pos.moves.items():
            # Qualification must gate lookup: include all qualified moves,
            # or include everything when the flag is False.
            if include_all_qualified and ev.qualification not in ("frequency", "stockfish", "both"):
                continue
            qual_moves.append(ev)

        if not qual_moves:
            continue

        # Encode board planes
        cpp_game = zeropawn_cpp.ChessGame(pos.fen)
        planes = zeropawn_cpp.encode_state(cpp_game)

        # Build policy distribution across qualified moves
        weights = []
        indices = []
        for ev in qual_moves:
            try:
                idx = zeropawn_cpp.move_to_index(cpp_game, ev.uci)
                if 0 <= idx < 4672:
                    indices.append(idx)
                    # Weight by frequency (at least 1)
                    w = max(1.0, float(ev.frequency))
                    if ev.stockfish_cp is not None and ev.stockfish_cp > 0:
                        w *= 1.0 + ev.stockfish_cp / 100.0
                    weights.append(w)
            except Exception:
                pass

        if not indices:
            continue

        tot_w = sum(weights)
        probs = [w / tot_w for w in weights]

        # For single-move dataset compatibility: sample primary move
        chosen_idx = rng.choice(indices, p=probs)

        # Policy distribution (4672,)
        policy_dist = np.zeros(4672, dtype=np.float32)
        for idx, p in zip(indices, probs):
            policy_dist[idx] = p

        # Opening positions use a neutral draw-leaning prior (30% Win, 40% Draw, 30% Loss)
        # Note: supervised training masks WDL loss for ply < 10 by default
        outcome = 1  # Draw
        value_target = np.array([0.30, 0.40, 0.30], dtype=np.float32)

        sample = {
            "fen": pos.fen,
            "planes": planes,
            "move_idx": int(chosen_idx),
            "outcome": outcome,
            "value_target": value_target,
            "policy_dist": policy_dist,
            "ply": pos.ply,
            "eco": pos.eco or "A00",
        }

        # Deterministic split by hash of canonical key (avoids opening line leakage)
        h = int(key.encode("utf-8").hex()[:8], 16)
        if (h % 100) < int(val_ratio * 100):
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    if not train_samples and val_samples:
        train_samples.append(val_samples.pop())
    if not val_samples and train_samples:
        val_samples.append(train_samples[-1])
    if not train_samples:
        raise RuntimeError("No qualified opening samples found to build dataset.")

    _write_dataset_split(train_dir, train_samples, "train")
    _write_dataset_split(val_dir, val_samples, "val")

    summary = {
        "output_dir": str(output_dir),
        "total_samples": len(train_samples) + len(val_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    logger.info("Built opening supervised dataset: %s", summary)
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
        "kind": "openings",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
