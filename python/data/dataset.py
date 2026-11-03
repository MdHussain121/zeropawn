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
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from python.constants import NUM_ACTIONS, PLANES_SHAPE
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


def _validate_meta(metadata: Dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    try:
        num_samples = int(metadata["num_samples"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid dataset metadata num_samples: {e}") from e
    if not 1 <= num_samples <= 1_000_000_000:
        raise ValueError(f"num_samples {num_samples} out of range [1, 1e9]")
    planes_shape = tuple(metadata.get("planes_shape", list(PLANES_SHAPE)))
    if tuple(planes_shape) != PLANES_SHAPE:
        raise ValueError(f"planes_shape must be {list(PLANES_SHAPE)}, got {list(planes_shape)}")
    return num_samples, planes_shape  # type: ignore[return-value]


def _check_st_size(path: Path, expected_bytes: int) -> None:
    try:
        actual = path.stat().st_size
    except OSError as e:
        raise ValueError(f"Cannot stat {path}: {e}") from e
    if actual < expected_bytes:
        raise ValueError(f"{path.name} truncated: {actual} < {expected_bytes} bytes")


def _outcome_to_value_target(outcomes: np.ndarray, is_syzygy: bool) -> np.ndarray:
    """Map stored outcome labels to (N, 3) one-hot value targets.

    3-class master labels are mover-relative 0=W, 1=D, 2=L. 5-class Syzygy
    labels map 0 (win) -> W, 1/2/3 (cursed win/draw/blessed loss) -> D, 4 (loss) -> L
    under FIDE rule 50 (SY-02). OUTCOME_UNAVAILABLE (255) rows are left all-zero so the
    WDL loss masks them. This is the single source of truth for both loaders;
    the GPU chunk loop previously applied the Syzygy mapping unconditionally,
    decoding 3-class draws as wins (mover-relative semantics, Stage A §3.3).
    """
    outcomes = np.asarray(outcomes)
    targets = np.zeros((outcomes.shape[0], 3), dtype=np.float32)
    if is_syzygy:
        valid = (outcomes != OUTCOME_UNAVAILABLE) & (outcomes <= 4)
        cls = np.where(outcomes[valid] == 0, 0, np.where(outcomes[valid] == 4, 2, 1))
    else:
        valid = (outcomes != OUTCOME_UNAVAILABLE) & (outcomes <= 2)
        cls = outcomes[valid].astype(np.int64)
    targets[np.nonzero(valid)[0], cls] = 1.0
    return targets


# Bit shift for big-endian (np.packbits-default) unpacking: bit i of a byte is
# (byte >> shift[i]) & 1 with shift = [7, 6, ..., 0].
_PACK_SHIFT = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8)

MASK_PACKED_BYTES = (NUM_ACTIONS + 7) // 8  # 584 — NUM_ACTIONS actions bit-packed


def decode_packed_masks(packed: torch.Tensor) -> torch.Tensor:
    """Decode bit-packed legal masks to a bool tensor, identical to np.unpackbits.

    ``packed``: (B, 584) uint8 → returns (B, 4672) bool. 4672 = 584 × 8 exactly,
    so no padding bits exist. Runs on whatever device the tensor lives on.
    """
    shift = _PACK_SHIFT.to(packed.device)
    bits = (packed.unsqueeze(-1) >> shift) & 1
    return bits.reshape(packed.shape[0], -1).to(torch.bool)


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
        kind = str(self.metadata.get("kind") or self.metadata.get("source") or "").lower()
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
            _check_st_size(val_targets_path, self.num_samples * 3 * 4)
            self.value_targets = np.memmap(
                val_targets_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_samples, 3),
            )
            # Consistency check (DI-22): spot-check row consistency against outcomes.bin
            check_n = min(self.num_samples, 1000)
            if check_n > 0:
                sample_outcomes = np.array(self.outcomes[:check_n])
                sample_vt = np.array(self.value_targets[:check_n])
                if not np.all(np.isfinite(sample_vt)) or np.any(sample_vt < 0.0):
                    raise ValueError(
                        f"Non-finite or negative values in value_targets.bin in {self.data_dir}"
                    )
                valid_outcomes = (sample_outcomes != OUTCOME_UNAVAILABLE) & (sample_outcomes <= 2)
                if np.any(valid_outcomes):
                    vt_sums = sample_vt[valid_outcomes].sum(axis=-1)
                    if np.any(vt_sums <= 0.0):
                        raise ValueError(
                            f"value_targets.bin row sum is 0 for unmasked outcomes in {self.data_dir}"
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
                shape=(self.num_samples, NUM_ACTIONS),
            )
        else:
            self.policy_targets = None

        # Optional DTZ distance-to-zero targets: (N,) float32
        dtz_targets_path = self.data_dir / "dtz_targets.bin"
        if dtz_targets_path.exists():
            dtz_mm = np.memmap(
                dtz_targets_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_samples,),
            )
            # Guard against legacy all-NaN DTZ files (SY-04): ignore if entirely NaN
            check_len = min(1024, self.num_samples)
            if np.isnan(dtz_mm[:check_len]).all() and np.isnan(dtz_mm).all():
                logger.debug(
                    "dtz_targets.bin in %s contains all-NaN entries; ignoring unsupported DTZ targets",
                    self.data_dir,
                )
                self.dtz_targets = None
            else:
                self.dtz_targets = dtz_mm
        else:
            self.dtz_targets = None

        # Optional packed legal-action masks: (N, 584) uint8, bit-packed over
        # 4672 actions (schema_version 2 datasets). A size mismatch is a hard
        # error — silently ignoring a corrupt mask stream would corrupt the
        # masked loss in subtle ways.
        self.masks = None
        masks_path = self.data_dir / "masks.bin"
        packed_bytes = (NUM_ACTIONS + 7) // 8
        if masks_path.exists():
            _check_st_size(masks_path, self.num_samples * packed_bytes)
            self.masks = np.memmap(
                masks_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.num_samples, packed_bytes),
            )

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
            # Stored ply passes through for every row (SLQ-06): supervision is
            # decided by the WDL loss mask (ply >= 10 and target sum > 0) and the
            # stored label, never by the dataset kind. Mate-labelled tactical
            # rows (ply=10, outcome=0) therefore reach the value head, while
            # OUTCOME_UNAVAILABLE rows stay masked by their zero target.
            "ply": torch.tensor(ply, dtype=torch.long),
            "is_tactical": torch.tensor(is_tac, dtype=torch.bool),
        }

        if self.masks is not None:
            unpacked = np.unpackbits(np.array(self.masks[idx]), count=NUM_ACTIONS)
            sample["legal_mask"] = torch.from_numpy(unpacked.astype(np.bool_))

        if outcome == OUTCOME_UNAVAILABLE:
            # Masked value target — the only contract for unavailable outcomes
            # (audit SLQ-06: never decode a masked row into a real class).
            sample["value_target"] = torch.zeros(3, dtype=torch.float32)
        elif self.value_targets is not None:
            vt = np.array(self.value_targets[idx], dtype=np.float32)
            sample["value_target"] = torch.from_numpy(vt)
        else:
            # One-hot outcome by default: [P(win), P(draw), P(loss)]
            one_hot = np.zeros(3, dtype=np.float32)
            if getattr(self, "is_syzygy", False):
                # Map 5-class syzygy outcomes back to 3-class value targets under FIDE rule 50 (SY-02).
                if 0 <= outcome <= 4:
                    if outcome == 0:
                        # Win (+2) -> Win
                        one_hot[0] = 1.0
                    elif outcome in (1, 2, 3):
                        # Cursed Win (+1), Draw (0), Blessed Loss (-1) -> Draw
                        one_hot[1] = 1.0
                    elif outcome == 4:
                        # Loss (-2) -> Loss
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
    - Each sample's ply and value target pass through from its source dataset
      (audit SLQ-06): mate-labelled tactical rows (ply=10, outcome=0) keep
      their value supervision; OUTCOME_UNAVAILABLE rows stay masked by their
      zero target. Supervision is decided by the trainer's WDL loss mask.
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
        # Total rows across both sources (audit SLQ-19): direct iteration must
        # see every sample. Epoch length under ratio sampling is chosen
        # explicitly via get_sampler(num_samples=...) by the trainer.
        return self.total_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < self.n_master:
            sample = self.master_ds[idx]
            sample["is_tactical"] = torch.tensor(False, dtype=torch.bool)
            return sample
        # Tactical rows pass through the underlying dataset's stored ply and
        # value labels (audit SLQ-06): mate-labelled rows supervise the value
        # head, OUTCOME_UNAVAILABLE rows stay masked. Only the is_tactical
        # marker is enforced here.
        sample = self.tactics_ds[idx - self.n_master]
        is_syzygy = getattr(self.tactics_ds, "is_syzygy", False)
        sample["is_tactical"] = torch.tensor(not is_syzygy, dtype=torch.bool)
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

    Schema-2 datasets' packed legal-action masks (``masks.bin``) are loaded with
    the same tiering and decoded per batch, so the legal-masked policy loss
    behaves identically on the GPU path and the disk-backed path (audit SLQ-05).
    Combined tactics shards are read segment-wise; no multi-gigabyte array is
    ever materialized in host RAM (audit SLQ-12).
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
        kind = str(metadata.get("kind") or metadata.get("source") or "").lower()
        self.is_syzygy = (
            kind.startswith("endgame_syzygy")
            or kind.startswith("syzygy")
            or "syzygy" in str(self.data_dir).lower()
        )

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

        # Packed legal-action masks (schema-2 datasets). The GPU path MUST
        # consume the same mask stream as BinaryChessDataset (audit SLQ-05):
        # without it the legal-masked policy loss can never activate on the
        # production CUDA default path.
        masks_m: Optional[np.memmap] = None
        masks_path = self.data_dir / "masks.bin"
        if masks_path.exists():
            _check_st_size(masks_path, self.num_samples * MASK_PACKED_BYTES)
            masks_m = np.memmap(
                masks_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.num_samples, MASK_PACKED_BYTES),
            )

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
            outcomes_path = self.data_dir / "outcomes.bin"
            if outcomes_path.exists():
                outcomes_m = np.memmap(
                    outcomes_path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(self.num_samples,),
                )
            elif self.is_tactical:
                # Tactical shards may legitimately ship without outcome labels;
                # treat every row as value-unavailable in memory. The source
                # tree is never written (audit V-1, data-integrity F-series).
                outcomes_m = np.full(self.num_samples, OUTCOME_UNAVAILABLE, dtype=np.uint8)
            else:
                raise FileNotFoundError(
                    f"Dataset is missing outcomes.bin and metadata does not declare it "
                    f"tactical: {self.data_dir}. "
                    "The loader will not synthesize labels or write to the dataset."
                )
            if isinstance(outcomes_m, np.memmap):
                self._memmaps.append(outcomes_m)

        # Value-target availability is decided in memory only (audit V-1): the
        # former branch reopened the master outcomes.bin in "r+" mode and
        # overwrote every stored label with the 255 sentinel whenever
        # tactics_dir was set, destroying the canonical master-game training
        # shard on disk. The loader now never mutates the source dataset tree:
        # BinaryChessDataset.__getitem__ and the GPU chunk loop below mask
        # OUTCOME_UNAVAILABLE (255) rows in memory, and rows that carry valid
        # W/D/L labels keep them.

        # Load a separate tactics dataset and concatenate it with the master
        # rows when tactics_dir is set (the --tactics-dir flag must never be
        # silently ignored). Planes and masks — the two multi-gigabyte arrays
        # — are kept as two segments and read segment-aware in the populate
        # loop below, so neither is ever fully materialized in host RAM
        # (audit SLQ-12: the previous implementation built ~3–4 full fp32
        # copies ≈ 36–48 GB for the 2.49M-sample corpus, guaranteed OOM on the
        # 16 GB host). Only the small per-row arrays are concatenated.
        tactics_n = 0
        master_n = self.num_samples
        t_planes_m: Optional[np.memmap] = None
        t_masks_m: Optional[np.memmap] = None
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
                t_planes_m = np.memmap(
                    self.tactics_dir / "planes.bin",
                    dtype=np.float32,
                    mode="r",
                    shape=(tactics_n, *self.planes_shape),
                )
                t_moves_m = np.memmap(
                    self.tactics_dir / "moves.bin", dtype=np.int16, mode="r", shape=(tactics_n,)
                )
                t_plies_m = np.memmap(
                    self.tactics_dir / "plies.bin", dtype=np.int16, mode="r", shape=(tactics_n,)
                )
                # Value targets: prefer an explicit stream; otherwise decode
                # from the shard's own outcomes so mate-labelled rows keep
                # their labels (SLQ-06) and 255 rows stay masked.
                t_targets_path = self.tactics_dir / "value_targets.bin"
                t_outcomes_m: Optional[np.memmap] = None
                if (self.tactics_dir / "outcomes.bin").exists():
                    t_outcomes_m = np.memmap(
                        self.tactics_dir / "outcomes.bin",
                        dtype=np.uint8,
                        mode="r",
                        shape=(tactics_n,),
                    )
                if t_targets_path.exists():
                    t_targets_arr = np.asarray(
                        np.memmap(t_targets_path, dtype=np.float32, mode="r", shape=(tactics_n, 3))
                    )
                elif t_outcomes_m is not None:
                    t_targets_arr = _outcome_to_value_target(np.asarray(t_outcomes_m), False)
                else:
                    t_targets_arr = np.zeros((tactics_n, 3), dtype=np.float32)

                # Small arrays: concatenate in RAM (a few MB each).
                moves_m = np.concatenate(
                    [np.asarray(moves_m), np.asarray(t_moves_m, dtype=np.int16)]
                )
                plies_m = np.concatenate(
                    [np.asarray(plies_m), np.asarray(t_plies_m, dtype=np.int16)]
                )
                # Stored plies pass through for tactics rows (SLQ-06): mate
                # rows carry ply=10 from the builder and must not be reset to 0.
                if targets_m is not None:
                    master_targets = np.asarray(targets_m, dtype=np.float32)
                elif outcomes_m is not None:
                    master_targets = _outcome_to_value_target(
                        np.asarray(outcomes_m, dtype=np.uint8), self.is_syzygy
                    )
                else:
                    master_targets = np.zeros((master_n, 3), dtype=np.float32)
                targets_m = np.concatenate([master_targets, t_targets_arr])

                master_outcomes = (
                    np.asarray(outcomes_m, dtype=np.uint8)
                    if outcomes_m is not None
                    else np.zeros(master_n, dtype=np.uint8)
                )
                t_outcomes_arr = (
                    np.asarray(t_outcomes_m, dtype=np.uint8)
                    if t_outcomes_m is not None
                    else np.full(tactics_n, OUTCOME_UNAVAILABLE, dtype=np.uint8)
                )
                outcomes_m = np.concatenate([master_outcomes, t_outcomes_arr])

                t_masks_path = self.tactics_dir / "masks.bin"
                if t_masks_path.exists():
                    _check_st_size(t_masks_path, tactics_n * MASK_PACKED_BYTES)
                    t_masks_m = np.memmap(
                        t_masks_path,
                        dtype=np.uint8,
                        mode="r",
                        shape=(tactics_n, MASK_PACKED_BYTES),
                    )

                self.num_samples = master_n + tactics_n

        self.sample_weights: Optional[torch.Tensor] = None
        # Build source-aware sampling weights if tactics_ratio is requested
        if tactics_ratio is not None:
            if not (0.0 <= tactics_ratio <= 1.0):
                raise ValueError(f"tactics_ratio must be between 0.0 and 1.0, got {tactics_ratio}")
            if outcomes_m is not None:
                # Combined (after concat) or master outcomes, already in memory.
                outcomes_arr = np.asarray(outcomes_m, dtype=np.uint8)
                tactics_mask = outcomes_arr == OUTCOME_UNAVAILABLE
            else:
                # No outcomes file and no tactics concatenated; relying on ply==0
                # without explicit confirmation is unsafe (opening positions are at ply 0).
                raise ValueError(
                    f"Cannot infer tactical mask in {self.data_dir}: outcomes.bin is missing. "
                    f"Relying on ply==0 as a tactics proxy is unsafe without explicit outcomes."
                )
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

        # Segment layout for the (possibly combined) dataset. Planes and masks
        # are read segment-aware so no full-size copy is ever materialized in
        # host RAM (audit SLQ-12). A combined dataset carries a mask stream
        # only when BOTH sources have one — a partial stream would silently
        # change loss semantics for the uncovered rows.
        if t_planes_m is not None:
            planes_segments = [
                (planes_m, 0, master_n),
                (t_planes_m, master_n, master_n + tactics_n),
            ]
            if masks_m is not None and t_masks_m is not None:
                mask_segments = [
                    (masks_m, 0, master_n),
                    (t_masks_m, master_n, master_n + tactics_n),
                ]
            else:
                if masks_m is not None or t_masks_m is not None:
                    print(
                        "[GpuResidentDataLoader] WARNING: legal masks dropped for the combined "
                        f"dataset — masks.bin exists on only one of the two sources "
                        f"({self.data_dir} vs {self.tactics_dir}). Batches will carry no "
                        "legal_mask (full-vocabulary smoothing fallback in the loss)."
                    )
                mask_segments = None
        else:
            planes_segments = [(planes_m, 0, self.num_samples)]
            mask_segments = [(masks_m, 0, self.num_samples)] if masks_m is not None else None
        self.has_masks = mask_segments is not None

        def _read_segments(segments, start: int, end: int) -> np.ndarray:
            """Concatenated rows [start, end) across (array, global_begin, global_end) segments."""
            pieces = []
            for arr, seg_begin, seg_end in segments:
                lo = max(start, seg_begin)
                hi = min(end, seg_end)
                if lo < hi:
                    pieces.append(np.asarray(arr[lo - seg_begin : hi - seg_begin]))
            if not pieces:
                raise RuntimeError(f"no segment covers rows [{start}, {end})")
            return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

        # Calculate bytes per sample
        bytes_per_sample = (
            int(np.prod(self.planes_shape))
            * torch.tensor([], dtype=self.storage_dtype).element_size()
            + 8  # moves (long)
            + 3 * torch.tensor([], dtype=self.storage_dtype).element_size()  # targets
            + 8  # plies (long)
            + (MASK_PACKED_BYTES if mask_segments is not None else 0)  # packed legal masks
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
            self.gpu_masks = (
                torch.empty(
                    (self.gpu_capacity, MASK_PACKED_BYTES), dtype=torch.uint8, device=self.device
                )
                if mask_segments is not None
                else None
            )
        else:
            self.gpu_planes = None
            self.gpu_moves = None
            self.gpu_targets = None
            self.gpu_plies = None
            self.gpu_masks = None

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
            self.ram_masks = (
                torch.empty(
                    (self.spillover_count, MASK_PACKED_BYTES), dtype=torch.uint8, pin_memory=pin
                )
                if mask_segments is not None
                else None
            )
        else:
            self.ram_planes = None
            self.ram_moves = None
            self.ram_targets = None
            self.ram_plies = None
            self.ram_masks = None

        # Populate tensors from the source arrays in streaming chunks (single
        # copy). Planes and masks read segment-aware (master rows, then tactics
        # rows) so neither multi-GB array is ever fully materialized in RAM.
        for start in range(0, self.num_samples, chunk_size):
            end = min(start + chunk_size, self.num_samples)
            p_chunk = torch.from_numpy(
                np.array(_read_segments(planes_segments, start, end), dtype=np.float32)
            )
            m_chunk = torch.from_numpy(np.array(moves_m[start:end], dtype=np.int64))
            pl_chunk = torch.from_numpy(np.array(plies_m[start:end], dtype=np.int64))
            mk_chunk = (
                torch.from_numpy(np.array(_read_segments(mask_segments, start, end)))
                if mask_segments is not None
                else None
            )

            if targets_m is not None:
                t_chunk = torch.from_numpy(np.array(targets_m[start:end], dtype=np.float32))
            else:
                out_chunk = np.array(outcomes_m[start:end], dtype=np.uint8)
                # Single mapping source of truth (3-class master vs 5-class
                # syzygy), mirroring BinaryChessDataset.__getitem__.
                t_chunk = torch.from_numpy(_outcome_to_value_target(out_chunk, self.is_syzygy))

            # Fill GPU portion
            if start < self.gpu_capacity:
                gpu_start = start
                gpu_end = min(end, self.gpu_capacity)
                chunk_slice_end = gpu_end - gpu_start

                self.gpu_planes[gpu_start:gpu_end].copy_(p_chunk[:chunk_slice_end])
                self.gpu_moves[gpu_start:gpu_end].copy_(m_chunk[:chunk_slice_end])
                self.gpu_targets[gpu_start:gpu_end].copy_(t_chunk[:chunk_slice_end])
                self.gpu_plies[gpu_start:gpu_end].copy_(pl_chunk[:chunk_slice_end])
                if mk_chunk is not None:
                    self.gpu_masks[gpu_start:gpu_end].copy_(mk_chunk[:chunk_slice_end])

                # Remainder goes to RAM
                if end > self.gpu_capacity:
                    ram_start = 0
                    ram_end = end - self.gpu_capacity
                    rem_slice = slice(chunk_slice_end, end - start)
                    self.ram_planes[ram_start:ram_end].copy_(p_chunk[rem_slice])
                    self.ram_moves[ram_start:ram_end].copy_(m_chunk[rem_slice])
                    self.ram_targets[ram_start:ram_end].copy_(t_chunk[rem_slice])
                    self.ram_plies[ram_start:ram_end].copy_(pl_chunk[rem_slice])
                    if mk_chunk is not None:
                        self.ram_masks[ram_start:ram_end].copy_(mk_chunk[rem_slice])
            else:
                # All in RAM
                ram_start = start - self.gpu_capacity
                ram_end = end - self.gpu_capacity
                self.ram_planes[ram_start:ram_end].copy_(p_chunk)
                self.ram_moves[ram_start:ram_end].copy_(m_chunk)
                self.ram_targets[ram_start:ram_end].copy_(t_chunk)
                self.ram_plies[ram_start:ram_end].copy_(pl_chunk)
                if mk_chunk is not None:
                    self.ram_masks[ram_start:ram_end].copy_(mk_chunk)

        # Explicitly release disk memmaps (close file handles). Planes/masks
        # have been fully copied into the GPU/RAM buffers; segment references
        # (including the tactics shards, which are not tracked in _memmaps)
        # are dropped so every mapping can be released.
        for mm in self._memmaps:
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("memmap close failed", exc_info=True)
        self._memmaps.clear()
        del planes_m, moves_m, plies_m
        if t_planes_m is not None:
            try:
                t_planes_m._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("tactics planes memmap close failed", exc_info=True)
        if t_masks_m is not None:
            try:
                t_masks_m._mmap.close()  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                logger.debug("tactics masks memmap close failed", exc_info=True)
        planes_segments = None
        mask_segments = None

        gpu_mb = (self.gpu_capacity * bytes_per_sample) / (1024 * 1024)
        ram_mb = (self.spillover_count * bytes_per_sample) / (1024 * 1024)
        print(
            f"[GpuResidentDataLoader] Loaded {self.num_samples:,} samples: "
            f"{self.gpu_capacity:,} in GPU VRAM ({gpu_mb:.1f} MB), "
            f"{self.spillover_count:,} in RAM ({ram_mb:.1f} MB). "
            f"legal masks: {'yes' if self.has_masks else 'NO (schema-1 dataset)'}"
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
        self.gpu_masks = None
        self.ram_planes = None
        self.ram_moves = None
        self.ram_targets = None
        self.ram_plies = None
        self.ram_masks = None
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
            "masks": None,
        }

        def _staging(curr_bs: int) -> tuple[torch.Tensor, ...]:
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
                bmk = (
                    torch.empty(
                        (self.batch_size, MASK_PACKED_BYTES), dtype=torch.uint8, device=self.device
                    )
                    if self.has_masks
                    else None
                )
                _buf.update({"planes": bp, "moves": bm, "targets": bt, "plies": bl, "masks": bmk})
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
                _buf["masks"][:curr_bs].clone() if _buf["masks"] is not None else None,
            )

        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and (end - start < self.batch_size):
                break
            idx = perm[start:end]

            if self.spillover_count == 0:
                # 100% GPU VRAM resident: instantaneous tensor slice
                batch = {
                    "planes": self.gpu_planes[idx],
                    "move": self.gpu_moves[idx],
                    "value_target": self.gpu_targets[idx],
                    "ply": self.gpu_plies[idx],
                }
                if self.has_masks:
                    batch["legal_mask"] = decode_packed_masks(self.gpu_masks[idx])
                yield batch
            elif self.gpu_capacity == 0:
                # 100% Host RAM resident
                idx_cpu = idx.cpu()
                batch = {
                    "planes": self.ram_planes[idx_cpu].to(self.device, non_blocking=True),
                    "move": self.ram_moves[idx_cpu].to(self.device, non_blocking=True),
                    "value_target": self.ram_targets[idx_cpu].to(self.device, non_blocking=True),
                    "ply": self.ram_plies[idx_cpu].to(self.device, non_blocking=True),
                }
                if self.has_masks:
                    batch["legal_mask"] = decode_packed_masks(
                        self.ram_masks[idx_cpu].to(self.device, non_blocking=True)
                    )
                yield batch
            else:
                # Tiered storage: blend GPU VRAM + host RAM spillover
                mask_gpu = idx < self.gpu_capacity
                if mask_gpu.all():
                    batch = {
                        "planes": self.gpu_planes[idx],
                        "move": self.gpu_moves[idx],
                        "value_target": self.gpu_targets[idx],
                        "ply": self.gpu_plies[idx],
                    }
                    if self.has_masks:
                        batch["legal_mask"] = decode_packed_masks(self.gpu_masks[idx])
                    yield batch
                elif (~mask_gpu).all():
                    idx_ram = (idx - self.gpu_capacity).cpu()
                    batch = {
                        "planes": self.ram_planes[idx_ram].to(self.device, non_blocking=True),
                        "move": self.ram_moves[idx_ram].to(self.device, non_blocking=True),
                        "value_target": self.ram_targets[idx_ram].to(
                            self.device, non_blocking=True
                        ),
                        "ply": self.ram_plies[idx_ram].to(self.device, non_blocking=True),
                    }
                    if self.has_masks:
                        batch["legal_mask"] = decode_packed_masks(
                            self.ram_masks[idx_ram].to(self.device, non_blocking=True)
                        )
                    yield batch
                else:
                    curr_bs = end - start
                    b_planes, b_moves, b_targets, b_plies, b_masks = _staging(curr_bs)

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

                    batch = {
                        "planes": b_planes,
                        "move": b_moves,
                        "value_target": b_targets,
                        "ply": b_plies,
                    }
                    if self.has_masks and b_masks is not None:
                        b_masks[mask_gpu] = self.gpu_masks[idx_gpu]
                        b_masks[~mask_gpu] = self.ram_masks[idx_ram].to(
                            self.device, non_blocking=True
                        )
                        batch["legal_mask"] = decode_packed_masks(b_masks)
                    yield batch
