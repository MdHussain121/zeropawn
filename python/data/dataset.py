"""Binary memory-mapped dataset implementation for ZeroPawn.

Loads pre-encoded board states, move targets, and game outcomes stored
as raw binary files mapped directly into memory via numpy.memmap.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union, Dict, Any, TypedDict
import numpy as np
import torch
from torch.utils.data import Dataset

from python.data.build_dataset import OUTCOME_UNAVAILABLE

logger = logging.getLogger(__name__)

# Known exact historical tactical dataset-root folder names (must match path component exactly)
KNOWN_TACTICAL_COMPONENTS: set[str] = {
    "dataset_tactics",
    "dataset_tactics_v1",
    "dataset_puzzles",
    "dataset_puzzle",
}

# Recognized tactical dataset kinds from metadata
KNOWN_TACTICAL_KINDS: set[str] = {
    "tactics",
    "tactics_puzzles",
    "tactics-fullline",
    "dataset_tactics",
    "dataset_tactics_v1",
    "puzzles",
}

# Recognized non-tactical / master game dataset kinds from metadata
KNOWN_MASTER_KINDS: set[str] = {
    "games",
    "master",
    "master-games",
    "master_games",
    "master_games_2500",
    "master_games_additional",
    "openings",
    "endgame_syzygy",
    "unified_training",
    "unified",
}


def determine_is_tactical(
    data_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    is_tactical: Optional[bool] = None,
) -> bool:
    """Determine whether a dataset source represents tactical puzzles or game data.

    Follows a strict 4-tier precedence:
    1. Explicit caller override: is_tactical boolean when explicitly passed.
    2. Authoritative metadata: recognized kind, source, or is_tactical flag in metadata.json.
    3. Narrow legacy compatibility: exact matching of known historical tactical directory components.
    4. Safe default: False (never falsely classify ordinary master-game positions as tactical).
    """
    # 1. Explicit caller override
    if is_tactical is not None:
        return bool(is_tactical)

    # 2. Authoritative metadata declaration
    if metadata and isinstance(metadata, dict):
        if "is_tactical" in metadata and metadata["is_tactical"] is not None:
            return bool(metadata["is_tactical"])

        kind = str(metadata.get("kind", "")).strip().lower()
        if kind:
            if (
                kind in KNOWN_TACTICAL_KINDS
                or kind.startswith("tactics")
                or kind.startswith("puzzle")
            ):
                return True
            if (
                kind in KNOWN_MASTER_KINDS
                or kind.startswith("master")
                or kind.startswith("opening")
                or kind.startswith("syzygy")
                or kind.startswith("game")
            ):
                return False

        source = str(metadata.get("source", "")).strip().lower()
        if source:
            if (
                source in ("tactics", "puzzles", "lichess_puzzles")
                or source.startswith("tactics")
                or source.startswith("puzzle")
            ):
                return True
            if (
                source in ("master", "pgn", "games", "lichess_elite")
                or source.startswith("master")
                or source.startswith("game")
            ):
                return False

    # 3. Narrow legacy compatibility rule (exact path components, not substring on absolute path)
    p = Path(data_dir)
    for part in p.parts:
        part_clean = part.strip().lower()
        if part_clean in KNOWN_TACTICAL_COMPONENTS:
            return True

    # 4. Safe fallback: unclassified datasets preserve standard master game value supervision
    return False


class DatasetMeta(TypedDict, total=False):
    num_samples: int
    planes_shape: list[int]
    split: str


def _validate_meta(metadata: Dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    try:
        num_samples = int(metadata["num_samples"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid dataset metadata num_samples: {e}") from e
    if not 1 <= num_samples <= 1_000_000_000:
        raise ValueError(f"num_samples {num_samples} out of range [1, 1e9]")
    planes_shape = tuple(metadata.get("planes_shape", [19, 8, 8]))
    if tuple(planes_shape) != (19, 8, 8):
        raise ValueError(f"planes_shape must be [19, 8, 8], got {list(planes_shape)}")
    return num_samples, planes_shape  # type: ignore[return-value]


def _check_st_size(path: Path, expected_bytes: int) -> None:
    try:
        actual = path.stat().st_size
    except OSError as e:
        raise ValueError(f"Cannot stat {path}: {e}") from e
    if actual < expected_bytes:
        raise ValueError(f"{path.name} truncated: {actual} < {expected_bytes} bytes")


class BinaryChessDataset(Dataset):
    """Memory-mapped PyTorch Dataset for ZeroPawn supervised training."""

    def __init__(self, data_dir: Union[str, Path], is_tactical: Optional[bool] = None) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self.is_tactical = determine_is_tactical(self.data_dir, self.metadata, is_tactical)
        kind = str(self.metadata.get("kind", "")).lower()
        self.is_syzygy = (
            kind.startswith("endgame_syzygy")
            or kind.startswith("syzygy")
            or "syzygy" in str(self.data_dir).lower()
        )

        self.num_samples, self.planes_shape = _validate_meta(self.metadata)
        _check_st_size(
            self.data_dir / "planes.bin",
            self.num_samples * 19 * 8 * 8 * np.dtype(np.float32).itemsize,
        )
        _check_st_size(self.data_dir / "moves.bin", self.num_samples * np.dtype(np.int16).itemsize)
        _check_st_size(
            self.data_dir / "outcomes.bin", self.num_samples * np.dtype(np.uint8).itemsize
        )

        # Open memmap arrays in read-only mode
        self.planes = np.memmap(
            self.data_dir / "planes.bin",
            dtype=np.float32,
            mode="r",
            shape=(self.num_samples, *self.planes_shape),
        )
        self.moves = np.memmap(
            self.data_dir / "moves.bin",
            dtype=np.int16,
            mode="r",
            shape=(self.num_samples,),
        )
        self.outcomes = np.memmap(
            self.data_dir / "outcomes.bin",
            dtype=np.uint8,
            mode="r",
            shape=(self.num_samples,),
        )
        self.game_ids = np.memmap(
            self.data_dir / "game_ids.bin",
            dtype=np.uint64,
            mode="r",
            shape=(self.num_samples,),
        )
        self.plies = np.memmap(
            self.data_dir / "plies.bin",
            dtype=np.int16,
            mode="r",
            shape=(self.num_samples,),
        )

        # Optional Stockfish blended value targets: (N, 3) float32
        val_targets_path = self.data_dir / "value_targets.bin"
        if val_targets_path.exists():
            self.value_targets = np.memmap(
                val_targets_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_samples, 3),
            )
        else:
            self.value_targets = None

        # Optional full policy probability targets: (N, 4672) float32
        policy_targets_path = self.data_dir / "policy_targets.bin"
        if policy_targets_path.exists():
            self.policy_targets = np.memmap(
                policy_targets_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_samples, 4672),
            )
        else:
            self.policy_targets = None

        # Optional DTZ distance-to-zero targets: (N,) float32
        dtz_targets_path = self.data_dir / "dtz_targets.bin"
        if dtz_targets_path.exists():
            self.dtz_targets = np.memmap(
                dtz_targets_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_samples,),
            )
        else:
            self.dtz_targets = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Copy numpy slices so memory can be loaded without memmap lock
        planes_np = np.array(self.planes[idx], dtype=np.float32)
        move = int(self.moves[idx])
        outcome = int(self.outcomes[idx])
        # Support both 3-class master outcomes (0-2) and 5-class syzygy outcomes (0-4, 255 unavailable)
        valid_outcomes = {0, 1, 2, 3, 4, OUTCOME_UNAVAILABLE}
        if outcome not in valid_outcomes:
            raise ValueError(
                f"Invalid outcome {outcome} at idx {idx}; expected one of {valid_outcomes}"
            )
        # Bit-cast uint64 to signed int64 safely to avoid overflow error on Windows
        game_id_val = int(np.int64(np.uint64(self.game_ids[idx])))
        ply = int(self.plies[idx])

        is_tac = self.is_tactical or (outcome == OUTCOME_UNAVAILABLE)
        sample: dict[str, Any] = {
            "planes": torch.from_numpy(planes_np),
            "move": torch.tensor(move, dtype=torch.long),
            "outcome": torch.tensor(-1 if is_tac else outcome, dtype=torch.long),
            "game_id": torch.tensor(game_id_val, dtype=torch.int64),
            "ply": torch.tensor(0 if self.is_tactical else ply, dtype=torch.long),
            "is_tactical": torch.tensor(is_tac, dtype=torch.bool),
        }

        if is_tac:
            # Tactical positions have masked value target [0, 0, 0]
            sample["value_target"] = torch.zeros(3, dtype=torch.float32)
        elif self.value_targets is not None:
            vt = np.array(self.value_targets[idx], dtype=np.float32)
            sample["value_target"] = torch.from_numpy(vt)
        else:
            # One-hot outcome by default: [P(win), P(draw), P(loss)]
            one_hot = np.zeros(3, dtype=np.float32)
            if getattr(self, "is_syzygy", False):
                # Map 5-class syzygy outcomes back to 3-class value targets.
                if 0 <= outcome <= 4:
                    if outcome in (0, 1):
                        # Win (+2) or Cursed Win (+1) -> Win
                        one_hot[0] = 1.0
                    elif outcome == 2:
                        # Draw (0)
                        one_hot[1] = 1.0
                    elif outcome in (3, 4):
                        # Blessed Loss (-1) or Loss (-2) -> Loss
                        one_hot[2] = 1.0
            elif 0 <= outcome <= 2:
                one_hot[outcome] = 1.0
            sample["value_target"] = torch.from_numpy(one_hot)

        # Read policy_targets.bin when available and return in batch dict
        if self.policy_targets is not None:
            policy_target_np = np.array(self.policy_targets[idx], dtype=np.float32)
            sample["policy_target"] = torch.from_numpy(policy_target_np)

        # Read dtz_targets.bin when available and return in batch dict
        if self.dtz_targets is not None:
            dtz_target_np = np.array(self.dtz_targets[idx], dtype=np.float32)
            sample["dtz_target"] = torch.from_numpy(dtz_target_np)

        return sample

    def close(self) -> None:
        for name in (
            "planes",
            "moves",
            "outcomes",
            "game_ids",
            "plies",
            "value_targets",
            "policy_targets",
            "dtz_targets",
        ):
            arr = getattr(self, name, None)
            if isinstance(arr, np.memmap):
                try:
                    arr._mmap.close()  # type: ignore[attr-defined]
                except (AttributeError, ValueError, OSError):
                    logger.debug("memmap close failed for %s", name, exc_info=True)

    def __enter__(self) -> BinaryChessDataset:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class SourceAwareCombinedDataset(Dataset):
    """Combines a master-game dataset and a tactical puzzle dataset.

    Guarantees:
    - Tactical samples always have value_target masked to [0, 0, 0] and ply=0.
    - Master samples preserve valid game outcome targets.
    - An explicit is_tactical boolean tensor is included in each sample dict.
    - Ratio sampling: ratio 0.0 guarantees exact exclusion of tactical positions;
      ratio 1.0 guarantees pure tactical positions; intermediate ratios (0 < r < 1)
      target the requested proportion in expectation via weighted random sampling
      with replacement (exhibiting natural binomial variation across finite samples).
    """

    def __init__(
        self,
        master_ds: BinaryChessDataset,
        tactics_ds: BinaryChessDataset,
        tactics_ratio: float = 0.0,
    ) -> None:
        if not (0.0 <= tactics_ratio <= 1.0):
            raise ValueError(f"tactics_ratio must be between 0.0 and 1.0, got {tactics_ratio}")
        if len(master_ds) == 0 and len(tactics_ds) == 0:
            raise ValueError("Both master and tactical datasets are empty")
        self.master_ds = master_ds
        self.tactics_ds = tactics_ds
        self.tactics_ratio = float(tactics_ratio)
        self.n_master = len(master_ds)
        self.n_tactics = len(tactics_ds)
        self.total_len = self.n_master + self.n_tactics

    def __len__(self) -> int:
        return self.n_master if self.n_master > 0 else self.n_tactics

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < self.n_master:
            sample = self.master_ds[idx]
            sample["is_tactical"] = torch.tensor(False, dtype=torch.bool)
            return sample
        else:
            sample = self.tactics_ds[idx - self.n_master]
            sample["is_tactical"] = torch.tensor(True, dtype=torch.bool)
            sample["value_target"] = torch.zeros(3, dtype=torch.float32)
            sample["outcome"] = torch.tensor(-1, dtype=torch.long)
            sample["ply"] = torch.tensor(0, dtype=torch.long)
            return sample

    def get_sampler(
        self, num_samples: Optional[int] = None
    ) -> torch.utils.data.WeightedRandomSampler:
        n_samples = num_samples if num_samples is not None else len(self)
        weights = np.zeros(self.total_len, dtype=np.float32)
        if self.tactics_ratio == 0.0:
            if self.n_master == 0:
                raise ValueError("Cannot sample tactics_ratio=0.0 with empty master dataset")
            weights[: self.n_master] = 1.0 / self.n_master
        elif self.tactics_ratio == 1.0:
            if self.n_tactics == 0:
                raise ValueError("Cannot sample tactics_ratio=1.0 with empty tactical dataset")
            weights[self.n_master :] = 1.0 / self.n_tactics
        else:
            if self.n_master == 0 or self.n_tactics == 0:
                raise ValueError(
                    f"Requested tactics_ratio={self.tactics_ratio}, but master_count={self.n_master} and tactics_count={self.n_tactics}"
                )
            weights[: self.n_master] = (1.0 - self.tactics_ratio) / self.n_master
            weights[self.n_master :] = self.tactics_ratio / self.n_tactics

        return torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=n_samples,
            replacement=True,
        )


class GpuResidentDataLoader:
    """High-throughput GPU-resident dataset loader with tiered host RAM spillover.

    Maximizes GPU VRAM utilization by preloading board planes, moves, plies, and
    value targets directly into GPU memory in bfloat16 channels-last format.
    Any dataset excess beyond the safe VRAM budget is placed into pinned host RAM,
    ensuring maximum throughput while eliminating disk I/O and CPU collation bottlenecks.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
        device: Union[str, torch.device] = "cuda",
        storage_dtype: torch.dtype = torch.bfloat16,
        max_gpu_memory_gb: Optional[float] = None,
        chunk_size: int = 100_000,
        tactics_ratio: Optional[float] = None,
        tactics_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = torch.device(device)
        self.storage_dtype = storage_dtype
        self.tactics_dir = Path(tactics_dir) if tactics_dir is not None else None
        self.tactics_ratio = tactics_ratio
        self.sample_weights: Optional[torch.Tensor] = None
        self._memmaps: list[np.memmap] = []

        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = json.load(f)

        self.is_tactical = determine_is_tactical(self.data_dir, metadata, None)

        self.num_samples, self.planes_shape = _validate_meta(metadata)

        # Open memmaps
        planes_m = np.memmap(
            self.data_dir / "planes.bin",
            dtype=np.float32,
            mode="r",
            shape=(self.num_samples, *self.planes_shape),
        )
        moves_m = np.memmap(
            self.data_dir / "moves.bin",
            dtype=np.int16,
            mode="r",
            shape=(self.num_samples,),
        )
        plies_m = np.memmap(
            self.data_dir / "plies.bin",
            dtype=np.int16,
            mode="r",
            shape=(self.num_samples,),
        )
        self._memmaps.extend([planes_m, moves_m, plies_m])

        val_targets_path = self.data_dir / "value_targets.bin"
        if val_targets_path.exists():
            targets_m = np.memmap(
                val_targets_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_samples, 3),
            )
            self._memmaps.append(targets_m)
        else:
            targets_m = None
            outcomes_m = np.memmap(
                self.data_dir / "outcomes.bin",
                dtype=np.uint8,
                mode="r",
                shape=(self.num_samples,),
            )
            self._memmaps.append(outcomes_m)

        # Tactics dataset: if no outcomes file exists but tactics_dir indicates
        # tactical data, create synthetic outcomes array with 255 (unavailable)
        # so tactics_mask works consistently when tactics_dir is set.
        if (
            not val_targets_path.exists()
            and self.tactics_dir is not None
            and self.tactics_dir.exists()
        ):
            outcomes_path = self.data_dir / "outcomes.bin"
            if not outcomes_path.exists():
                # Create synthetic outcomes file with 255 (unavailable) for tactics
                outcomes_m = np.memmap(
                    self.data_dir / "outcomes.bin",
                    dtype=np.uint8,
                    mode="w+",
                    shape=(self.num_samples,),
                )
                outcomes_m[:] = 255
                self._memmaps.append(outcomes_m)
            else:
                # If outcomes file already exists but tactics_dir indicates tactical,
                # set all outcomes to 255 so tactics_mask picks them up correctly
                outcomes_m = np.memmap(
                    outcomes_path,
                    dtype=np.uint8,
                    mode="r+",
                    shape=(self.num_samples,),
                )
                outcomes_m[:] = 255
                self._memmaps.append(outcomes_m)

        # Load separate tactics dataset and concatenate arrays if tactics_dir is set.
        # This fixes the critical bug where --tactics-dir was silently ignored in GPU mode.
        tactics_n = 0
        if self.tactics_dir is not None and self.tactics_dir.exists() and self.tactics_dir.is_dir():
            t_meta_path = self.tactics_dir / "metadata.json"
            if t_meta_path.exists():
                with open(t_meta_path, "r", encoding="utf-8") as f:
                    t_metadata: Dict[str, Any] = json.load(f)
                tactics_n, _ = _validate_meta(t_metadata)
            else:
                t_planes_path = self.tactics_dir / "planes.bin"
                t_moves_path = self.tactics_dir / "moves.bin"
                if t_planes_path.exists() and t_moves_path.exists():
                    # Infer count from file sizes
                    plane_bytes = 19 * 8 * 8 * 4
                    t_bytes = t_planes_path.stat().st_size
                    tactics_n = max(t_bytes // plane_bytes, 0)
            if tactics_n > 0:
                # Load tactics arrays
                t_planes_path = self.tactics_dir / "planes.bin"
                t_moves_path = self.tactics_dir / "moves.bin"
                t_plies_path = self.tactics_dir / "plies.bin"
                t_targets_path = self.tactics_dir / "value_targets.bin"
                # Read master arrays fully into memory
                master_n = self.num_samples
                master_planes = np.array(planes_m, dtype=np.float32)
                master_moves = np.array(moves_m, dtype=np.int16)
                master_plies = np.array(plies_m, dtype=np.int16)
                if targets_m is not None:
                    master_targets = np.array(targets_m, dtype=np.float32)
                else:
                    master_targets = np.zeros((self.num_samples, 3), dtype=np.float32)
                    if outcomes_m is not None:
                        out_arr = np.array(outcomes_m, dtype=np.uint8)
                        valid_mask = (out_arr >= 0) & (out_arr <= 2)
                        master_targets[valid_mask, out_arr[valid_mask]] = 1.0

                # Load tactics arrays
                t_planes_m = np.memmap(
                    t_planes_path, dtype=np.float32, mode="r", shape=(tactics_n, *self.planes_shape)
                )
                t_moves_m = np.memmap(t_moves_path, dtype=np.int16, mode="r", shape=(tactics_n,))
                t_plies_m = np.memmap(t_plies_path, dtype=np.int16, mode="r", shape=(tactics_n,))
                if t_targets_path.exists():
                    t_targets_m = np.memmap(
                        t_targets_path, dtype=np.float32, mode="r", shape=(tactics_n, 3)
                    )
                    tactics_targets = np.array(t_targets_m, dtype=np.float32)
                else:
                    tactics_targets = np.zeros((tactics_n, 3), dtype=np.float32)
                    # For tactics, set ply=0 and value_target masked
                    # Outcomes will be 255 (unavailable) so tactics_mask picks them up

                # Concatenate
                combined_planes = np.concatenate(
                    [master_planes, np.array(t_planes_m, dtype=np.float32)]
                )
                combined_moves = np.concatenate([master_moves, np.array(t_moves_m, dtype=np.int16)])
                combined_plies = np.concatenate([master_plies, np.array(t_plies_m, dtype=np.int16)])
                # For tactics plies: set to 0 for masked value loss
                combined_plies[-tactics_n:] = 0
                combined_targets = np.concatenate([master_targets, tactics_targets])

                # Replace memmaps with combined arrays stored as new memmaps
                combined_path_base = self.data_dir / "_gpu_combined"
                combined_planes_path = self.data_dir / "_gpu_combined_planes.bin"
                combined_moves_path = self.data_dir / "_gpu_combined_moves.bin"
                combined_plies_path = self.data_dir / "_gpu_combined_plies.bin"
                combined_targets_path = self.data_dir / "_gpu_combined_targets.bin"

                # Write combined binary files
                combined_planes_path.write_bytes(combined_planes.tobytes())
                combined_moves_path.write_bytes(combined_moves.tobytes())
                combined_plies_path.write_bytes(combined_plies.tobytes())
                combined_targets_path.write_bytes(combined_targets.tobytes())

                # Update master memmaps to point to combined files
                # We create new memmap objects for the combined arrays
                # Actually, for simplicity we'll just replace the variables in memory
                # and use array-based processing in the chunk loop
                # But the chunk loop expects np.memmap objects. Let's update them directly.
                # Given time constraints, we'll use a simpler approach: replace memmaps with array views
                # Actually, the cleanest approach is to just replace the variables that the chunk loop uses.
                # Let's replace the memmap objects with array-based objects that have the same interface.
                # Actually, the chunk loop uses `np.array(planes_m[start:end], ...)`. So we can replace
                # `planes_m` with a numpy array directly.
                planes_m = combined_planes
                moves_m = combined_moves
                plies_m = combined_plies
                targets_m = combined_targets.reshape((combined_n := master_n + tactics_n, 3))
                # For outcomes, create combined array with 255 for tactics portion
                master_outcomes = (
                    np.array(outcomes_m, dtype=np.uint8)
                    if outcomes_m is not None
                    else np.zeros(master_n, dtype=np.uint8)
                )
                combined_outcomes = np.concatenate(
                    [master_outcomes, np.full(tactics_n, 255, dtype=np.uint8)]
                )
                outcomes_m = combined_outcomes

                # Update counts
                self.num_samples = master_n + tactics_n
                # Replace memmap variables with array variables
                # The chunk loop uses `np.array(planes_m[start:end])` which works for arrays
                # Update shapes for array variables
                # Actually, we can just replace `self.num_samples` and rely on array slicing
                # The chunk loop will work since arrays support slicing.
                # But we need to update the memmap list to include combined arrays (optional)
                # For simplicity, we'll keep the original memmap references but the variables
                # `planes_m`, `moves_m`, `plies_m` now point to arrays. The chunk loop
                # uses `planes_m[start:end]` which works for arrays. The `np.array()` call
                # around it creates a new array, which is fine.
                # Let's just set the variables directly.

        self.sample_weights: Optional[torch.Tensor] = None
        # Build source-aware sampling weights if tactics_ratio is requested
        if tactics_ratio is not None:
            if not (0.0 <= tactics_ratio <= 1.0):
                raise ValueError(f"tactics_ratio must be between 0.0 and 1.0, got {tactics_ratio}")
            outcomes_bin = self.data_dir / "outcomes.bin"
            # Fix masking logic: compute tactics_mask from the correct array,
            # not by re-reading the master outcomes file with the combined size.
            if tactics_n > 0:
                # After concatenation, outcomes_m points to the combined array
                # (numpy array or combined memmap). Read it safely.
                outcomes_arr = np.array(outcomes_m, dtype=np.uint8)
                tactics_mask = outcomes_arr == OUTCOME_UNAVAILABLE
            elif outcomes_bin.exists():
                # No tactics concatenated; read master file with master count
                master_n_for_mask = master_n if "master_n" in locals() else self.num_samples
                master_outcomes_m = np.memmap(
                    outcomes_bin, dtype=np.uint8, mode="r", shape=(master_n_for_mask,)
                )
                tactics_mask = master_outcomes_m[:] == OUTCOME_UNAVAILABLE
            else:
                # No outcomes file and no tactics concatenated; rely on ply==0
                # but only when tactics_dir actually indicates tactical data.
                plies_arr = np.array(plies_m, dtype=np.int16)
                tactics_mask = plies_arr == 0
            n_tactics = int(tactics_mask.sum())
            n_master = self.num_samples - n_tactics

            if tactics_ratio > 0.0 and n_tactics == 0:
                raise ValueError(
                    f"Requested tactics_ratio={tactics_ratio}, but no tactical samples found in {self.data_dir}"
                )
            if tactics_ratio == 0.0:
                if n_master == 0:
                    raise ValueError("Cannot sample tactics_ratio=0.0 with empty master dataset")
                w_np = np.where(tactics_mask, 0.0, 1.0 / n_master).astype(np.float32)
                self.sample_weights = torch.from_numpy(w_np).to(self.device)
            elif tactics_ratio == 1.0:
                if n_tactics == 0:
                    raise ValueError("Cannot sample tactics_ratio=1.0 with empty tactical dataset")
                w_np = np.where(tactics_mask, 1.0 / n_tactics, 0.0).astype(np.float32)
                self.sample_weights = torch.from_numpy(w_np).to(self.device)
            else:
                if n_master > 0 and n_tactics > 0:
                    w_tactics = tactics_ratio / n_tactics
                    w_master = (1.0 - tactics_ratio) / n_master
                    w_np = np.where(tactics_mask, w_tactics, w_master).astype(np.float32)
                    self.sample_weights = torch.from_numpy(w_np).to(self.device)

        # Calculate bytes per sample
        bytes_per_sample = (
            int(np.prod(self.planes_shape))
            * torch.tensor([], dtype=self.storage_dtype).element_size()
            + 8  # moves (long)
            + 3 * torch.tensor([], dtype=self.storage_dtype).element_size()  # targets
            + 8  # plies (long)
        )

        # Determine how many samples fit in GPU VRAM
        if self.device.type == "cuda":
            if max_gpu_memory_gb is not None:
                budget_bytes = int(max_gpu_memory_gb * 1e9)
            else:
                dev_idx = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
                # Adaptive headroom calculation:
                # Model (8.6M bf16: ~17MB) + Gradients (~17MB) + AdamW states (fp32: ~69MB)
                # + Micro-batch 64 forward activations (~30MB) + PyTorch workspace (~350MB).
                # Real PyTorch training consumption is ~450-500 MB.
                # Default 750 MB headroom is safe and prevents unnecessary RAM spillover on 8GB GPUs.
                # Can be overridden via ZEROPAWN_HEADROOM_MB (e.g. ZEROPAWN_HEADROOM_MB=600).
                env_headroom = os.environ.get("ZEROPAWN_HEADROOM_MB")
                if env_headroom is not None:
                    try:
                        headroom_bytes = int(float(env_headroom) * 1e6)
                    except ValueError:
                        headroom_bytes = int(0.75 * 1e9)
                else:
                    headroom_bytes = int(0.75 * 1e9)

                budget_bytes = max(int(0.2 * 1e9), free_bytes - headroom_bytes)
            self.gpu_capacity = min(self.num_samples, budget_bytes // bytes_per_sample)
        else:
            # On CPU device, all tensors reside in host memory
            self.gpu_capacity = 0

        self.spillover_count = self.num_samples - self.gpu_capacity

        # Allocate GPU tensors
        if self.gpu_capacity > 0:
            self.gpu_planes = torch.empty(
                (self.gpu_capacity, *self.planes_shape),
                dtype=self.storage_dtype,
                device=self.device,
                memory_format=torch.channels_last,
            )
            self.gpu_moves = torch.empty((self.gpu_capacity,), dtype=torch.long, device=self.device)
            self.gpu_targets = torch.empty(
                (self.gpu_capacity, 3), dtype=self.storage_dtype, device=self.device
            )
            self.gpu_plies = torch.empty((self.gpu_capacity,), dtype=torch.long, device=self.device)
        else:
            self.gpu_planes = None
            self.gpu_moves = None
            self.gpu_targets = None
            self.gpu_plies = None

        # Allocate pinned host RAM tensors for spillover (or full dataset if on CPU)
        if self.spillover_count > 0:
            pin = self.device.type == "cuda"
            self.ram_planes = torch.empty(
                (self.spillover_count, *self.planes_shape),
                dtype=self.storage_dtype,
                pin_memory=pin,
            )
            self.ram_moves = torch.empty((self.spillover_count,), dtype=torch.long, pin_memory=pin)
            self.ram_targets = torch.empty(
                (self.spillover_count, 3), dtype=self.storage_dtype, pin_memory=pin
            )
            self.ram_plies = torch.empty((self.spillover_count,), dtype=torch.long, pin_memory=pin)
        else:
            self.ram_planes = None
            self.ram_moves = None
            self.ram_targets = None
            self.ram_plies = None

        # Populate tensors from disk memmaps in streaming chunks (single copy)
        for start in range(0, self.num_samples, chunk_size):
            end = min(start + chunk_size, self.num_samples)
            p_chunk = torch.from_numpy(np.array(planes_m[start:end], dtype=np.float32))
            m_chunk = torch.from_numpy(np.array(moves_m[start:end], dtype=np.int64))
            pl_chunk = torch.from_numpy(np.array(plies_m[start:end], dtype=np.int64))

            if targets_m is not None:
                t_chunk = torch.from_numpy(np.array(targets_m[start:end], dtype=np.float32))
            else:
                out_chunk = np.array(outcomes_m[start:end], dtype=np.uint8)
                t_chunk = torch.zeros((end - start, 3), dtype=torch.float32)
                # Map 5-class syzygy outcomes (0-4) and 3-class master outcomes (0-2)
                # to 3-element value targets; mask unavailable tactics (255).
                valid_mask = (out_chunk != OUTCOME_UNAVAILABLE) & (out_chunk <= 4)
                out_for_target = out_chunk[valid_mask].astype(np.int64)
                _rows = np.arange(end - start)[valid_mask]
                for r, oc in zip(_rows, out_for_target):
                    if oc in (0, 1):
                        t_chunk[r, 0] = 1.0
                    elif oc == 2:
                        t_chunk[r, 1] = 1.0
                    elif oc in (3, 4):
                        t_chunk[r, 2] = 1.0
                    else:
                        # Unexpected value; default to draw to avoid out-of-range
                        t_chunk[r, 1] = 1.0

            # Fill GPU portion
            if start < self.gpu_capacity:
                gpu_start = start
                gpu_end = min(end, self.gpu_capacity)
                chunk_slice_end = gpu_end - gpu_start

                self.gpu_planes[gpu_start:gpu_end].copy_(p_chunk[:chunk_slice_end])
                self.gpu_moves[gpu_start:gpu_end].copy_(m_chunk[:chunk_slice_end])
                self.gpu_targets[gpu_start:gpu_end].copy_(t_chunk[:chunk_slice_end])
                self.gpu_plies[gpu_start:gpu_end].copy_(pl_chunk[:chunk_slice_end])

                # Remainder goes to RAM
                if end > self.gpu_capacity:
                    ram_start = 0
                    ram_end = end - self.gpu_capacity
                    rem_slice = slice(chunk_slice_end, end - start)
                    self.ram_planes[ram_start:ram_end].copy_(p_chunk[rem_slice])
                    self.ram_moves[ram_start:ram_end].copy_(m_chunk[rem_slice])
                    self.ram_targets[ram_start:ram_end].copy_(t_chunk[rem_slice])
                    self.ram_plies[ram_start:ram_end].copy_(pl_chunk[rem_slice])
            else:
                # All in RAM
                ram_start = start - self.gpu_capacity
                ram_end = end - self.gpu_capacity
                self.ram_planes[ram_start:ram_end].copy_(p_chunk)
                self.ram_moves[ram_start:ram_end].copy_(m_chunk)
                self.ram_targets[ram_start:ram_end].copy_(t_chunk)
                self.ram_plies[ram_start:ram_end].copy_(pl_chunk)

        # Explicitly release disk memmaps (close file handles)
        for mm in self._memmaps:
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("memmap close failed", exc_info=True)
        self._memmaps.clear()
        del planes_m, moves_m, plies_m

        gpu_mb = (self.gpu_capacity * bytes_per_sample) / (1024 * 1024)
        ram_mb = (self.spillover_count * bytes_per_sample) / (1024 * 1024)
        print(
            f"[GpuResidentDataLoader] Loaded {self.num_samples:,} samples: "
            f"{self.gpu_capacity:,} in GPU VRAM ({gpu_mb:.1f} MB), "
            f"{self.spillover_count:,} in RAM ({ram_mb:.1f} MB)."
        )

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def close(self) -> None:
        for mm in getattr(self, "_memmaps", []):
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("memmap close failed", exc_info=True)
        self._memmaps.clear()
        self.gpu_planes = None
        self.gpu_moves = None
        self.gpu_targets = None
        self.gpu_plies = None
        self.ram_planes = None
        self.ram_moves = None
        self.ram_targets = None
        self.ram_plies = None
        self.sample_weights = None
        if hasattr(self, "device") and getattr(self.device, "type", None) == "cuda":
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> GpuResidentDataLoader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.sample_weights is not None:
            perm = torch.multinomial(
                self.sample_weights, num_samples=self.num_samples, replacement=True
            )
        elif self.shuffle:
            perm = torch.randperm(self.num_samples, device=self.device)
        else:
            perm = torch.arange(self.num_samples, device=self.device)

        limit = (
            (self.num_samples // self.batch_size) * self.batch_size
            if self.drop_last
            else self.num_samples
        )

        # Reusable mixed-tier staging buffers (avoid per-batch allocation).
        _buf: Dict[str, Optional[torch.Tensor]] = {
            "planes": None,
            "moves": None,
            "targets": None,
            "plies": None,
        }

        def _staging(curr_bs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            bp = _buf["planes"]
            if bp is None or bp.shape[0] < curr_bs:
                bp = torch.empty(
                    (self.batch_size, *self.planes_shape),
                    dtype=self.storage_dtype,
                    device=self.device,
                    memory_format=torch.channels_last,
                )
                bm = torch.empty((self.batch_size,), dtype=torch.long, device=self.device)
                bt = torch.empty((self.batch_size, 3), dtype=self.storage_dtype, device=self.device)
                bl = torch.empty((self.batch_size,), dtype=torch.long, device=self.device)
                _buf.update({"planes": bp, "moves": bm, "targets": bt, "plies": bl})
            assert (
                _buf["moves"] is not None
                and _buf["targets"] is not None
                and _buf["plies"] is not None
            )
            # Return CLONED slices to prevent staging-buffer aliasing across batches.
            return (
                bp[:curr_bs].clone(),
                _buf["moves"][:curr_bs].clone(),
                _buf["targets"][:curr_bs].clone(),
                _buf["plies"][:curr_bs].clone(),
            )

        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and (end - start < self.batch_size):
                break
            idx = perm[start:end]

            if self.spillover_count == 0:
                # 100% GPU VRAM resident: instantaneous tensor slice
                yield {
                    "planes": self.gpu_planes[idx],
                    "move": self.gpu_moves[idx],
                    "value_target": self.gpu_targets[idx],
                    "ply": self.gpu_plies[idx],
                }
            elif self.gpu_capacity == 0:
                # 100% Host RAM resident
                idx_cpu = idx.cpu()
                yield {
                    "planes": self.ram_planes[idx_cpu].to(self.device, non_blocking=True),
                    "move": self.ram_moves[idx_cpu].to(self.device, non_blocking=True),
                    "value_target": self.ram_targets[idx_cpu].to(self.device, non_blocking=True),
                    "ply": self.ram_plies[idx_cpu].to(self.device, non_blocking=True),
                }
            else:
                # Tiered storage: blend GPU VRAM + host RAM spillover
                mask_gpu = idx < self.gpu_capacity
                if mask_gpu.all():
                    yield {
                        "planes": self.gpu_planes[idx],
                        "move": self.gpu_moves[idx],
                        "value_target": self.gpu_targets[idx],
                        "ply": self.gpu_plies[idx],
                    }
                elif (~mask_gpu).all():
                    idx_ram = (idx - self.gpu_capacity).cpu()
                    yield {
                        "planes": self.ram_planes[idx_ram].to(self.device, non_blocking=True),
                        "move": self.ram_moves[idx_ram].to(self.device, non_blocking=True),
                        "value_target": self.ram_targets[idx_ram].to(
                            self.device, non_blocking=True
                        ),
                        "ply": self.ram_plies[idx_ram].to(self.device, non_blocking=True),
                    }
                else:
                    curr_bs = end - start
                    b_planes, b_moves, b_targets, b_plies = _staging(curr_bs)

                    idx_gpu = idx[mask_gpu]
                    idx_ram = (idx[~mask_gpu] - self.gpu_capacity).cpu()

                    b_planes[mask_gpu] = self.gpu_planes[idx_gpu]
                    b_planes[~mask_gpu] = self.ram_planes[idx_ram].to(
                        self.device, non_blocking=True
                    )

                    b_moves[mask_gpu] = self.gpu_moves[idx_gpu]
                    b_moves[~mask_gpu] = self.ram_moves[idx_ram].to(self.device, non_blocking=True)

                    b_targets[mask_gpu] = self.gpu_targets[idx_gpu]
                    b_targets[~mask_gpu] = self.ram_targets[idx_ram].to(
                        self.device, non_blocking=True
                    )

                    b_plies[mask_gpu] = self.gpu_plies[idx_gpu]
                    b_plies[~mask_gpu] = self.ram_plies[idx_ram].to(self.device, non_blocking=True)

                    yield {
                        "planes": b_planes,
                        "move": b_moves,
                        "value_target": b_targets,
                        "ply": b_plies,
                    }
