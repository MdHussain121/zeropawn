"""Memory-Mapped Replay Buffer with Human Data Mixing and Taper Schedule.

Contract Specifications:
- Ring buffer of 500k-1M positions, memory-mapped on disk.
- Uniform random sampling across active buffer positions.
- Configurable human-data mix fraction with linear taper schedule:
  starts at 25%, tapers to 5% over configurable step count.
- Logs and reports the actual mix ratio per mini-batch.
- Resumable: state (size, write_head, step) persists across restarts.

Record schema contract (self-play ring, audit SR-07/SR-10/SR-15):

| shard          | shape                 | dtype   | content                                    |
|----------------|-----------------------|---------|--------------------------------------------|
| planes.bin     | (capacity, 19, 8, 8)  | float32 | side-to-move-relative encoding, values {0,1} |
| visits.bin     | (capacity, 4672)      | float32 | normalized root visit distribution (sum~=1)  |
| outcomes.bin   | (capacity,)           | int64   | 0=Win, 1=Draw, 2=Loss (mover-relative)       |
| game_ids.bin   | (capacity,)           | uint64  | producing game id (> 0)                      |
| versions.bin   | (capacity,)           | int32   | model version at record time (>= 0)          |
| full_search.bin| (capacity,)           | bool_   | playout-cap randomization flag               |

``meta.json`` carries ``schema_version`` (see ``SCHEMA_VERSION``), a ``shards``
byte-layout contract, the ``generation`` of the physical shard set, a
``buffer_id`` identifying the directory, and optional ``checksums``. Loading
validates metadata strictly, verifies exact shard byte sizes, and validates
the scalar domains of every live record; it fails closed (raises
``BufferIntegrityError``) rather than loading a state it cannot validate.
Insertion validates every record before writing (``add_positions`` raises
``ReplayRecordValidationError`` and commits nothing).

Randomness contract (audit SR-12 scope): buffer SAMPLING uses the
process-global ``numpy``/``torch`` RNG (uniform draws, batch permutation). The
orchestrator seeds those globals once when ``--seed`` is given, so sampling is
reproducible run-scoped (not batch-scoped); self-play GAME randomness is
per-game seeded (``master_seed + game_id``) and never touches the globals. The
cache/ring split and the staleness filter are deterministic given the sampled
indices.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

import numpy as np
import psutil
import torch

from python.selfplay.worker import SavedPosition

try:
    import zeropawn_cpp
except ImportError:
    try:
        import python  # noqa: F401
        import zeropawn_cpp
    except ImportError:
        zeropawn_cpp = None

logger = logging.getLogger(__name__)

# Version stamped on GPU-cache rows whose model version is genuinely unknown
# (added without version info, e.g. pre-population from a restored buffer
# before any weight push). Negative so it can never collide with a real
# monotonic version counter starting at 1; the staleness sampler treats
# unknown provenance as untrusted rather than as fresh data.
UNKNOWN_MODEL_VERSION = -1

# Game-id sentinel for sampled rows that have no producing game (human dataset
# and tactical-pool rows) or whose provenance was genuinely unavailable.
# Self-play records always carry game_id > 0 (enforced at insertion).
NO_GAME_ID = 0

# ---------------------------------------------------------------------------
# Schema / integrity / validation contract (audit SR-05..SR-13)
# ---------------------------------------------------------------------------

# 1 = legacy layout (six shards + meta without schema_version/shards/checksums);
# 2 = current: meta carries schema_version, shards layout, generation, buffer_id
#     and optional checksums; insertion validates records; scalar domains are
#     verified at load. Version 1 buffers load unchanged (missing field = 1).
SCHEMA_VERSION = 2

# Weight pushes a row may lag the freshest row before the staleness filter
# stops drawing it. Relative to the current max version (dormant for a
# single-version buffer, active once weights start moving). One constant is
# shared by the ring sampler and the GPU cache so both paths apply identical
# eligibility semantics (audit SR-05).
STALENESS_WINDOW = 20

# Tolerance for "the visit vector is a normalized distribution, not raw counts"
# (audit SR-15). Stored rows are empirically exact (sum == 1.000000).
POLICY_SUM_TOLERANCE = 1e-3

# Planes are binary indicators; allow a hair of float error without letting a
# genuinely different encoder change through silently.
PLANES_RANGE_TOLERANCE = 1e-6


class ReplayRecordValidationError(ValueError):
    """A record failed insertion-time validation; nothing was written (SR-07)."""


class BufferIntegrityError(ValueError):
    """Buffer metadata or shards violate the schema contract; load fails closed."""


def _shard_nbytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape)) * int(np.dtype(dtype).itemsize)


def validate_records_arrays(
    planes: np.ndarray,
    visits: np.ndarray,
    outcomes: np.ndarray,
    full_search: np.ndarray,
    versions: np.ndarray,
    game_ids: np.ndarray,
    *,
    in_channels: int = 19,
    height: int = 8,
    width: int = 8,
    num_actions: int = 4672,
) -> None:
    """Vectorized insertion-time validation of one flush (audit SR-07).

    Raises :class:`ReplayRecordValidationError` with a precise diagnostic when
    any record is malformed. The checks run on the stacked arrays (one pass per
    field, no per-record Python loop) and use min/max/sum rather than
    ``np.isfinite``: NaN and +/-Inf propagate into min/max/sum, so a
    comparison-based check rejects non-finite values as well — a non-finite
    value can never satisfy both a min and a max bound.

    Contract enforced (see module docstring for the full schema):

    - shapes ``(n, C, H, W)`` / ``(n, A)`` / ``(n,)`` and matching lengths;
    - planes finite, within [0, 1];
    - visits finite, non-negative, each row summing to 1 within
      ``POLICY_SUM_TOLERANCE`` (they are normalized distributions, not counts);
    - outcomes in {0, 1, 2} (mover-relative WDL; the 255 sentinel belongs to
      human/tactical pools and must never enter the self-play ring);
    - versions >= 0; game_ids > 0 (uint64 producing-game identity).
    """
    n = int(planes.shape[0]) if getattr(planes, "ndim", 0) else 0
    expected_shapes = [
        ("planes", planes, (n, in_channels, height, width)),
        ("visits", visits, (n, num_actions)),
        ("outcomes", outcomes, (n,)),
        ("full_search", full_search, (n,)),
        ("versions", versions, (n,)),
        ("game_ids", game_ids, (n,)),
    ]
    for name, arr, want in expected_shapes:
        if tuple(getattr(arr, "shape", ())) != want:
            raise ReplayRecordValidationError(
                f"record field {name!r} has shape {tuple(getattr(arr, 'shape', ()))}; expected {want}"
            )
    if n == 0:
        return

    # planes: finite and in [0, 1] (NaN/Inf fail both bounds)
    if not (float(np.min(planes)) >= -PLANES_RANGE_TOLERANCE):
        raise ReplayRecordValidationError(
            "planes contain values below 0 (or non-finite values); the encoder emits indicators in [0, 1]"
        )
    if not (float(np.max(planes)) <= 1.0 + PLANES_RANGE_TOLERANCE):
        raise ReplayRecordValidationError(
            "planes contain values above 1 (or non-finite values); the encoder emits indicators in [0, 1]"
        )

    # visits: finite, non-negative, normalized distribution per row
    if not (float(np.min(visits)) >= 0.0):
        raise ReplayRecordValidationError(
            "visits contain negative (or non-finite) entries; visits must be a probability distribution"
        )
    row_sums = visits.sum(axis=1, dtype=np.float64)
    bad = np.nonzero(
        (row_sums < 1.0 - POLICY_SUM_TOLERANCE) | (row_sums > 1.0 + POLICY_SUM_TOLERANCE)
    )[0]
    if bad.size:
        raise ReplayRecordValidationError(
            f"visits row {int(bad[0])} sums to {float(row_sums[bad[0]]):.6f}; expected a normalized "
            f"distribution (1.0 +/- {POLICY_SUM_TOLERANCE}). Raw visit counts are not valid targets."
        )

    # outcomes: {0,1,2}; NaN can never equal an integer class
    unique_out = np.unique(outcomes)
    invalid_out = unique_out[~np.isin(unique_out, (0, 1, 2))]
    if invalid_out.size:
        raise ReplayRecordValidationError(
            f"outcomes contain invalid class value(s) {invalid_out[:5].tolist()}; expected 0=Win, "
            "1=Draw, 2=Loss (255 is an unavailable-target sentinel valid only in human/tactical pools)"
        )

    # versions: >= 0 for ring records (negative versions are cache-only markers)
    if not (float(np.min(versions)) >= 0.0):
        raise ReplayRecordValidationError(
            "versions contain negative values; ring records must carry a real model version (>= 0)"
        )

    # game_ids: > 0 (producing game identity; 0 is the "no game" sentinel)
    if not (float(np.min(game_ids)) > 0.0):
        raise ReplayRecordValidationError(
            "game_ids contain 0 or non-finite values; self-play records must carry the producing game id (> 0)"
        )


def validate_position_record(
    pos: Any,
    *,
    in_channels: int = 19,
    height: int = 8,
    width: int = 8,
    num_actions: int = 4672,
) -> None:
    """:func:`validate_records_arrays` for a single :class:`SavedPosition`."""
    validate_records_arrays(
        np.expand_dims(np.asarray(pos.planes, dtype=np.float32), 0),
        np.expand_dims(np.asarray(pos.visits, dtype=np.float32), 0),
        np.asarray([pos.outcome], dtype=np.int64),
        np.asarray([pos.is_full_search], dtype=np.bool_),
        np.asarray([pos.model_version], dtype=np.int32),
        np.asarray([pos.game_id], dtype=np.uint64),
        in_channels=in_channels,
        height=height,
        width=width,
        num_actions=num_actions,
    )


def _mint_buffer_generation() -> str:
    """New generation id for a freshly created replay buffer."""
    import uuid
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _buffer_artifact_lock(target: Path, *, owner: str, timeout: float):
    """Cross-process lock for replay-buffer artifact writes (audit SUS-02).

    Two processes sharing one buffer directory must not interleave meta/array
    flushes. The lock waits briefly (buffer writes are frequent and short) and
    raises loudly if the other holder never releases.
    """
    from python.common.artifacts import artifact_lock

    return artifact_lock(target, timeout=timeout, owner=owner)


class ReplayBatchDict(TypedDict):
    planes: torch.Tensor
    visits: torch.Tensor
    outcomes: torch.Tensor
    is_full_search: torch.Tensor


class ReplayMetaDict(TypedDict):
    capacity: int
    in_channels: int
    height: int
    width: int
    num_actions: int
    write_head: int
    size: int
    total_added: int
    # Identity of this buffer instance, written once at creation and preserved
    # across flushes. A checkpoint carries the generation it was saved against
    # so a resume can never splice two buffer generations together (audit
    # CKP-05 restoration contract). ``None`` for legacy buffers written before
    # the marker existed.
    buffer_generation: Optional[str]


class GpuResidentReplayCache:
    """High-throughput circular GPU tensor cache in VRAM for zero-PCIe sampling.

    Coverage contract (audit SR-06): the cache is a velocity layer whose
    contents are the newest ``capacity`` ring rows (pre-population takes the
    newest rows; live inserts append). When the disk ring holds more records
    than the cache, the cache alone would only map the newest
    ``cache_capacity / ring_size`` fraction of the eligible population — a
    sampling bias. ``ReplayBuffer`` therefore splits each self-play request
    between the cache and the ring proportionally to that coverage
    (``selfplay_sampling_policy="ring_uniform"``, the default) so the combined
    distribution is uniform over the whole ring; ``"cache_uniform"`` restores
    the historical newest-rows-only behavior for comparison.

    Staleness semantics are shared with the ring path via
    ``STALENESS_WINDOW``: rows more than that many weight pushes behind the
    freshest row, and rows of unknown provenance (version < 0), are excluded
    while enough fresh rows exist; if not enough fresh rows exist the sampler
    falls back to sampling all rows (documented, counted in
    ``staleness_fallback_count``) rather than returning fewer rows than asked.
    """

    # Weight pushes a row may lag the freshest row before the sampler stops
    # drawing it. Relative to the current max version, so it is armed from the
    # first push (the old absolute `max_ver > 20` gate let the early run train
    # on arbitrarily old restored rows). Shared constant (audit SR-05).
    STALENESS_WINDOW = STALENESS_WINDOW

    def __init__(
        self,
        capacity: int,
        in_channels: int = 19,
        height: int = 8,
        width: int = 8,
        num_actions: int = 4672,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.capacity = capacity
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_actions = num_actions
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._write_head = 0
        self.size = 0
        self.total_added = 0
        # Writers (add_positions) and samplers race on the ring state and the
        # four tensors; both sides take this lock (short critical sections).
        self._lock = threading.Lock()

        # Preallocate GPU resident tensors
        self.gpu_planes = torch.empty(
            (self.capacity, self.in_channels, self.height, self.width),
            dtype=torch.float32,
            device=self.device,
            memory_format=torch.channels_last,
        )
        self.gpu_visits = torch.empty(
            (self.capacity, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.gpu_outcomes = torch.empty(
            (self.capacity,),
            dtype=torch.long,
            device=self.device,
        )
        self.gpu_full_search = torch.empty(
            (self.capacity,),
            dtype=torch.bool,
            device=self.device,
        )
        self.gpu_versions = torch.zeros(
            (self.capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        # Producing-game identity travels with the row (audit SR-13). Rows added
        # without an explicit id are stamped NO_GAME_ID (0 = provenance
        # genuinely unavailable, e.g. a cache restored before ids existed) and
        # can always be told apart from real ids, which are > 0.
        self.gpu_game_ids = torch.zeros(
            (self.capacity,),
            dtype=torch.int64,
            device=self.device,
        )
        # Rows more than the staleness window behind the freshest row, or of
        # unknown provenance, that had to be included because too few fresh
        # rows existed (telemetry for the documented fallback).
        self.staleness_fallback_count = 0

    @property
    def write_head(self) -> int:
        return self._write_head

    @write_head.setter
    def write_head(self, val: int) -> None:
        val = int(val)
        if getattr(self, "capacity", 0) > 0:
            self._write_head = val % self.capacity
        else:
            self._write_head = val

    def add_positions(
        self,
        planes: Union[np.ndarray, torch.Tensor],
        visits: Union[np.ndarray, torch.Tensor],
        outcomes: Union[np.ndarray, torch.Tensor],
        full_search: Union[np.ndarray, torch.Tensor],
        versions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        game_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Add batch of positions directly to GPU cache.

        Rows added without version information are stamped
        ``UNKNOWN_MODEL_VERSION`` (-1): the staleness sampler treats unknown
        provenance as untrusted (excluded while known rows are available)
        instead of as fresh data. Rows added without game ids are stamped
        ``NO_GAME_ID`` (0); real self-play ids are always > 0 (audit SR-13).
        """
        n = len(planes)
        if n == 0 or self.capacity == 0:
            return

        if isinstance(planes, np.ndarray):
            p_t = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).to(
                self.device, non_blocking=True
            )
            v_t = torch.from_numpy(np.ascontiguousarray(visits, dtype=np.float32)).to(
                self.device, non_blocking=True
            )
            o_t = torch.from_numpy(np.ascontiguousarray(outcomes, dtype=np.int64)).to(
                self.device, non_blocking=True
            )
            f_t = torch.from_numpy(np.ascontiguousarray(full_search, dtype=np.bool_)).to(
                self.device, non_blocking=True
            )
            ver_t = (
                torch.from_numpy(np.ascontiguousarray(versions, dtype=np.int32)).to(
                    self.device, non_blocking=True
                )
                if versions is not None
                else torch.full((n,), UNKNOWN_MODEL_VERSION, dtype=torch.int32).to(
                    self.device, non_blocking=True
                )
            )
            gid_t = (
                torch.from_numpy(np.ascontiguousarray(game_ids, dtype=np.int64)).to(
                    self.device, non_blocking=True
                )
                if game_ids is not None
                else torch.full((n,), NO_GAME_ID, dtype=torch.int64).to(
                    self.device, non_blocking=True
                )
            )
        else:
            p_t = planes.to(dtype=torch.float32, device=self.device, non_blocking=True)
            v_t = visits.to(dtype=torch.float32, device=self.device, non_blocking=True)
            o_t = outcomes.to(dtype=torch.long, device=self.device, non_blocking=True)
            f_t = full_search.to(dtype=torch.bool, device=self.device, non_blocking=True)
            ver_t = (
                versions.to(dtype=torch.int32, device=self.device, non_blocking=True)
                if versions is not None
                else torch.full((n,), UNKNOWN_MODEL_VERSION, dtype=torch.int32, device=self.device)
            )
            gid_t = (
                game_ids.to(dtype=torch.int64, device=self.device, non_blocking=True)
                if game_ids is not None
                else torch.full((n,), NO_GAME_ID, dtype=torch.int64, device=self.device)
            )

        if n >= self.capacity:
            # Audit RL-12: the full-overwrite fast path mutates the ring
            # counters and every cache tensor, so it must hold the same lock
            # as the incremental path and the sampler — otherwise a concurrent
            # sample_batch can observe a torn overwrite (new counters with
            # half-copied rows).
            with self._lock:
                self.gpu_planes.copy_(p_t[-self.capacity :])
                self.gpu_visits.copy_(v_t[-self.capacity :])
                self.gpu_outcomes.copy_(o_t[-self.capacity :])
                self.gpu_full_search.copy_(f_t[-self.capacity :])
                self.gpu_versions.copy_(ver_t[-self.capacity :])
                self.gpu_game_ids.copy_(gid_t[-self.capacity :])
                self.write_head = 0
                self.size = self.capacity
                self.total_added += n
            return

        start = self.write_head
        end = start + n
        with self._lock:
            if end <= self.capacity:
                self.gpu_planes[start:end].copy_(p_t)
                self.gpu_visits[start:end].copy_(v_t)
                self.gpu_outcomes[start:end].copy_(o_t)
                self.gpu_full_search[start:end].copy_(f_t)
                self.gpu_versions[start:end].copy_(ver_t)
                self.gpu_game_ids[start:end].copy_(gid_t)
            else:
                part1 = self.capacity - start
                part2 = n - part1
                self.gpu_planes[start : self.capacity].copy_(p_t[:part1])
                self.gpu_visits[start : self.capacity].copy_(v_t[:part1])
                self.gpu_outcomes[start : self.capacity].copy_(o_t[:part1])
                self.gpu_full_search[start : self.capacity].copy_(f_t[:part1])
                self.gpu_versions[start : self.capacity].copy_(ver_t[:part1])
                self.gpu_game_ids[start : self.capacity].copy_(gid_t[:part1])

                self.gpu_planes[:part2].copy_(p_t[part1:])
                self.gpu_visits[:part2].copy_(v_t[part1:])
                self.gpu_outcomes[:part2].copy_(o_t[part1:])
                self.gpu_full_search[:part2].copy_(f_t[part1:])
                self.gpu_versions[:part2].copy_(ver_t[part1:])
                self.gpu_game_ids[:part2].copy_(gid_t[part1:])

            self.write_head = (start + n) % self.capacity
            self.size = min(self.capacity, self.size + n)
            self.total_added += n

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample mini-batch directly on GPU with zero PCIe overhead.

        Staleness filter (always armed, identical to the ring path): exclude
        rows more than ``STALENESS_WINDOW`` weight pushes behind the freshest
        row and rows of unknown provenance (version < 0). For a fresh monotonic
        run this is a no-op — it only bites restored/stale/unknown rows. If too
        few fresh rows exist the sampler falls back to sampling every row
        (documented; counted in ``staleness_fallback_count``) rather than
        returning fewer rows than requested. Sampling is uniform with
        replacement. ``game_ids`` travels with every returned row (SR-13).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        with self._lock:
            if self.size == 0:
                raise ValueError("GPU cache is empty.")
            cur_versions = self.gpu_versions[: self.size]
            max_ver = int(cur_versions.max().item()) if self.size > 0 else 0
            valid_mask = (cur_versions >= (max_ver - self.STALENESS_WINDOW)) & (cur_versions >= 0)
            valid_count = int(valid_mask.sum().item())
            if valid_count >= batch_size:
                valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                rand_sub = torch.randint(
                    0, len(valid_indices), (batch_size,), device=self.device
                )
                indices = valid_indices[rand_sub]
            else:
                # Not enough known-fresh rows (e.g. a cache restored without
                # version info): fall back to sampling everything, but never
                # pretend unknown rows were fresh.
                self.staleness_fallback_count += 1
                indices = torch.randint(0, self.size, (batch_size,), device=self.device)
            out = {
                "planes": self.gpu_planes[indices].clone(),
                "visits": self.gpu_visits[indices].clone(),
                "outcomes": self.gpu_outcomes[indices].clone(),
                "is_full_search": self.gpu_full_search[indices].clone(),
                "game_ids": self.gpu_game_ids[indices].clone(),
            }
        return out

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        for attr in ("gpu_planes", "gpu_visits", "gpu_outcomes", "gpu_full_search", "gpu_versions"):
            t = getattr(self, attr, None)
            if isinstance(t, torch.Tensor):
                setattr(self, attr, None)

    def __enter__(self) -> GpuResidentReplayCache:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class TacticalPositionPool:
    """In-memory pool of pre-encoded tactical positions for anchor replay."""

    def __init__(
        self,
        planes: np.ndarray,
        moves: np.ndarray,
        outcomes: np.ndarray,
        source_path: Optional[Path] = None,
    ) -> None:
        self.planes = planes
        self.moves = moves
        self.outcomes = outcomes
        self.source_path = source_path

    def __len__(self) -> int:
        return len(self.moves)


def load_tactical_positions_from_csv(
    csv_path: Union[str, Path],
    max_positions: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and encode tactical positions from a CSV file.

    Validates legality of opponent setup moves and solution moves; skips
    positions with illegal moves and logs the count.
    """
    path = Path(csv_path)
    if not path.is_file():
        logger.warning("Tactical CSV not found at %s", path)
        return (
            np.zeros((0, 19, 8, 8), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    planes_list: List[np.ndarray] = []
    moves_list: List[int] = []
    outcomes_list: List[int] = []
    illegal_count = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            logger.warning("Tactical CSV at %s is empty", path)
            return (
                np.zeros((0, 19, 8, 8), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )

        fen_idx = 1
        moves_idx = 2
        for idx, col in enumerate(header):
            col_lower = col.strip().lower()
            if col_lower in ("fen", "board_fen"):
                fen_idx = idx
            elif col_lower in ("moves", "move", "solution", "solution_move"):
                moves_idx = idx

        for row in reader:
            if not row or len(row) <= max(fen_idx, moves_idx):
                continue

            fen = row[fen_idx].strip()
            moves_str = row[moves_idx].strip()
            if not fen or not moves_str:
                continue

            tokens = moves_str.split()
            if not tokens:
                illegal_count += 1
                continue

            try:
                g = zeropawn_cpp.ChessGame(fen) if zeropawn_cpp is not None else None
                if g is None:
                    break
            except Exception:
                illegal_count += 1
                continue

            if len(tokens) >= 2:
                opp_move = tokens[0]
                sol_move = tokens[1]
                if g.is_legal(opp_move):
                    g.make_move_uci(opp_move)
                elif g.is_legal(tokens[0]):
                    sol_move = tokens[0]
                else:
                    illegal_count += 1
                    continue
            else:
                sol_move = tokens[0]

            if not g.is_legal(sol_move):
                illegal_count += 1
                continue

            try:
                move_idx = zeropawn_cpp.move_to_index(g, sol_move)
            except Exception:
                illegal_count += 1
                continue

            if not (0 <= move_idx < 4672):
                illegal_count += 1
                continue

            planes = zeropawn_cpp.encode_state(g)
            planes_list.append(planes)
            moves_list.append(move_idx)
            outcomes_list.append(255)  # OUTCOME_UNAVAILABLE

            if max_positions is not None and len(moves_list) >= max_positions:
                break

    if illegal_count > 0:
        logger.warning(
            "Skipped %d tactical positions with illegal or unmappable solution moves from %s",
            illegal_count,
            path,
        )

    if not moves_list:
        return (
            np.zeros((0, 19, 8, 8), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.ascontiguousarray(np.stack(planes_list, axis=0), dtype=np.float32),
        np.ascontiguousarray(moves_list, dtype=np.int64),
        np.ascontiguousarray(outcomes_list, dtype=np.int64),
    )


class ReplayBuffer:
    """Ring buffer of self-play positions with memory-mapped disk or host RAM storage, GPU caching, and human-data mixing."""

    def __init__(
        self,
        buffer_dir: Union[str, Path],
        capacity: int = 500_000,
        in_channels: int = 19,
        height: int = 8,
        width: int = 8,
        num_actions: int = 4672,
        initial_human_mix: float = 0.25,
        final_human_mix: float = 0.05,
        taper_steps: int = 50_000,
        human_data_dir: Optional[Union[str, Path]] = None,
        create_new: bool = False,
        in_memory: bool = False,
        gpu_cache_capacity: int = 0,
        gpu_device: Optional[Union[str, torch.device]] = None,
        human_cache_size: int = 20_000,
        tactical_csv_path: Optional[Union[str, Path]] = None,
        tactical_mix: float = 0.12,
        tactical_cache_size: int = 20_000,
        data_role: Optional[str] = None,
        allow_uncertified_data: bool = False,
        selfplay_sampling_policy: str = "ring_uniform",
        cache_rng_seed: Optional[int] = None,
    ) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        # Audit RL-26: capacity=0 previously crashed deep in file creation
        # (seek to a negative offset); refuse it with an actionable message.
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(
                f"ReplayBuffer capacity must be > 0 (got {capacity}). The CLI enforces "
                "10,000..2,000,000; tests and tools should pass a small positive capacity."
            )
        self.capacity = capacity
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_actions = num_actions
        self.in_memory = in_memory

        self.initial_human_mix = initial_human_mix
        self.final_human_mix = final_human_mix
        self.taper_steps = taper_steps
        self.human_cache_size = human_cache_size
        self.data_role = data_role
        self.allow_uncertified_data = allow_uncertified_data
        self.gpu_device: Optional[torch.device] = (
            torch.device(gpu_device) if gpu_device is not None else None
        )

        if selfplay_sampling_policy not in ("ring_uniform", "cache_uniform"):
            raise ValueError(
                f"Unknown selfplay_sampling_policy {selfplay_sampling_policy!r}; expected "
                "'ring_uniform' (unbiased; default) or 'cache_uniform' (historical newest-rows bias)."
            )
        # Sampling policy (audit SR-06). "ring_uniform" splits each self-play
        # request between the GPU cache and the ring proportionally to the
        # cache's coverage of the ring, which makes the combined distribution
        # uniform over the whole ring. "cache_uniform" keeps the historical
        # behavior (whenever the cache is large enough, every self-play row
        # comes from it — i.e. uniformly over the newest `cache_capacity` rows
        # only) and is retained for comparison runs.
        self.selfplay_sampling_policy = selfplay_sampling_policy
        # Telemetry for the unified staleness fallback (rows older than the
        # window, or of unknown provenance, sampled because too few fresh rows
        # existed) and for the last self-play sampling shape.
        self.staleness_fallback_count = 0
        self.last_selfplay_rows = 0
        # Audit RL-25: cache composition (which rows a human/tactical cache
        # preloads) previously used a fresh unseeded generator on every load,
        # so the cache was irreproducible even under --seed. When the caller
        # provides a seed, one persistent seeded generator drives every cache
        # load and refresh; unseeded callers keep the entropy-seeded behavior.
        self._cache_rng: Optional[np.random.Generator] = (
            np.random.default_rng(int(cache_rng_seed)) if cache_rng_seed is not None else None
        )
        self.last_selfplay_source: Optional[str] = None
        # Records rejected by insertion validation (telemetry; a nonzero value
        # means a producer bug was caught before it could enter the buffer).
        self.rejected_records = 0

        self._write_head = 0
        self.size = 0
        self.total_added = 0
        # Generation identity of this buffer instance: minted on creation and
        # persisted in meta.json. Checkpoints record the generation they were
        # saved against; a resume refuses to splice different generations
        # together (CKP-05 restoration contract).
        self.buffer_generation: Optional[str] = None
        self._lock = threading.Lock()
        self._last_meta_save = 0.0
        self._games_since_meta = 0
        self._meta_throttle_games = 10
        self._meta_throttle_secs = 30.0

        # Schema/provenance bookkeeping (audit SR-04, SR-07, SR-10).
        self.schema_version = SCHEMA_VERSION
        # Generation of the physical shard set: incremented when a buffer
        # directory is re-created (--reset-buffer) so a checkpoint that
        # references an older generation can never be paired silently.
        self.generation = 1
        self._shard_checksums: Dict[str, str] = {}
        # Stable identity of this buffer directory (used by the
        # checkpoint<->buffer consistency contract).
        self.buffer_id = self._compute_buffer_id()

        # Human dataset cache (resides in GPU VRAM if CUDA available, or host RAM as fallback)
        self._human_cache_size = 0
        self._human_cache_device: Optional[torch.device] = None
        self._human_gpu_planes: Optional[torch.Tensor] = None
        self._human_gpu_moves: Optional[torch.Tensor] = None
        self._human_gpu_outcomes: Optional[torch.Tensor] = None
        self._human_cache_planes: Optional[np.ndarray] = None
        self._human_cache_moves: Optional[np.ndarray] = None
        self._human_cache_outcomes: Optional[np.ndarray] = None

        # Tactical dataset pool & cache (anchor replay: 10-15% of every batch)
        self.tactical_pool: Optional[TacticalPositionPool] = None
        self.tactical_mix = float(tactical_mix)
        self.tactical_cache_size = int(tactical_cache_size)
        self.tactical_pool_size = 0
        self._tactical_cache_size = 0
        self._tactical_cache_device: Optional[torch.device] = None
        self._tactical_gpu_planes: Optional[torch.Tensor] = None
        self._tactical_gpu_moves: Optional[torch.Tensor] = None
        self._tactical_gpu_outcomes: Optional[torch.Tensor] = None
        self._tactical_cache_planes: Optional[np.ndarray] = None
        self._tactical_cache_moves: Optional[np.ndarray] = None
        self._tactical_cache_outcomes: Optional[np.ndarray] = None
        self.last_tactical_mix = 0.0
        self.last_human_mix = 0.0

        self.gpu_cache: Optional[GpuResidentReplayCache] = None
        self.human_dataset = None

        # Memory-mapped binary file paths
        self.paths = {
            "planes": self.buffer_dir / "planes.bin",
            "visits": self.buffer_dir / "visits.bin",
            "outcomes": self.buffer_dir / "outcomes.bin",
            "full_search": self.buffer_dir / "full_search.bin",
            "versions": self.buffer_dir / "versions.bin",
            "game_ids": self.buffer_dir / "game_ids.bin",
            "meta": self.buffer_dir / "meta.json",
        }

        # Initialize or load metadata. A directory whose SHARD files exist must
        # go through _load_meta even when meta.json is missing: that path now
        # recovers from a valid scratch file or fails closed for a populated
        # buffer (REM-06) instead of silently starting a fresh size-0 view over
        # existing records.
        _shard_keys = ("planes", "visits", "outcomes", "full_search", "versions", "game_ids")
        _shards_exist = any(self.paths[key].exists() for key in _shard_keys)
        if not create_new and (self.paths["meta"].exists() or _shards_exist):
            self._load_meta()
        else:
            if create_new:
                # A reset starts a NEW generation of the physical shard set.
                # Stale bytes beyond `size` may remain in the preallocated files
                # (invisible by construction: bounds are [0, size) and every
                # write starts at `write_head`), but any checkpoint that
                # references the previous generation must fail consistency
                # verification rather than pair silently (audit SR-04).
                self.generation = self._read_existing_generation() + 1
            self._create_empty_files()
            self.buffer_generation = _mint_buffer_generation()
            self._save_meta()

        if self.in_memory:
            # Check host RAM headroom before allocating full ring buffer in RAM
            needed_bytes = self.capacity * 23573
            try:
                available_ram = psutil.virtual_memory().available
            except Exception:
                available_ram = 16 * 1024**3
            if needed_bytes > available_ram or (
                needed_bytes > 50 * 1024**2
                and needed_bytes > max(0, available_ram - int(2.0 * 1024**3))
            ):
                logger.warning(
                    "Requested in_memory replay buffer capacity %d requires %.2f GB RAM, "
                    "but available RAM is only %.2f GB. Falling back to disk memmap to prevent desktop freeze.",
                    self.capacity,
                    needed_bytes / (1024**3),
                    available_ram / (1024**3),
                )
                self.in_memory = False

        if self.in_memory:
            # Allocate directly in host RAM
            self.planes = np.zeros(
                (self.capacity, self.in_channels, self.height, self.width),
                dtype=np.float32,
            )
            self.visits = np.zeros(
                (self.capacity, self.num_actions),
                dtype=np.float32,
            )
            self.outcomes = np.zeros(self.capacity, dtype=np.int64)
            self.full_search = np.zeros(self.capacity, dtype=np.bool_)
            self.versions = np.zeros(self.capacity, dtype=np.int32)
            self.game_ids = np.zeros(self.capacity, dtype=np.uint64)
            if not create_new and self.paths["planes"].exists():
                self._load_from_disk_to_ram()
        else:
            # Open memmaps
            self.planes = np.memmap(
                self.paths["planes"],
                dtype=np.float32,
                mode="r+",
                shape=(self.capacity, self.in_channels, self.height, self.width),
            )
            self.visits = np.memmap(
                self.paths["visits"],
                dtype=np.float32,
                mode="r+",
                shape=(self.capacity, self.num_actions),
            )
            self.outcomes = np.memmap(
                self.paths["outcomes"],
                dtype=np.int64,
                mode="r+",
                shape=(self.capacity,),
            )
            self.full_search = np.memmap(
                self.paths["full_search"],
                dtype=np.bool_,
                mode="r+",
                shape=(self.capacity,),
            )
            self.versions = np.memmap(
                self.paths["versions"],
                dtype=np.int32,
                mode="r+",
                shape=(self.capacity,),
            )
            self.game_ids = np.memmap(
                self.paths["game_ids"],
                dtype=np.uint64,
                mode="r+",
                shape=(self.capacity,),
            )

        # Human dataset initialization if provided
        self.human_dataset = None
        if human_data_dir is not None:
            self._init_human_dataset(
                human_data_dir, cache_size=self.human_cache_size, device=self.gpu_device
            )

        # Tactical dataset initialization if provided
        if tactical_csv_path is not None:
            self._init_tactical_dataset(
                tactical_csv_path, cache_size=self.tactical_cache_size, device=self.gpu_device
            )

        # Tier 1 GPU cache in VRAM
        self.gpu_cache: Optional[GpuResidentReplayCache] = None
        if gpu_cache_capacity > 0:
            self.enable_gpu_cache(capacity=gpu_cache_capacity, device=self.gpu_device)

    @property
    def write_head(self) -> int:
        return self._write_head

    @write_head.setter
    def write_head(self, val: int) -> None:
        val = int(val)
        if getattr(self, "capacity", 0) > 0:
            self._write_head = val % self.capacity
        else:
            self._write_head = val

    def _create_empty_files(self) -> None:
        """Create or initialize empty files for memmaps."""
        for key, path in self.paths.items():
            if key == "meta":
                continue
            if not path.exists():
                shape = self._get_shape(key)
                dtype = self._get_dtype(key)
                size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
                with open(path, "wb") as f:
                    f.seek(size_bytes - 1)
                    f.write(b"\0")

    def _get_shape(self, key: str) -> Tuple[int, ...]:
        if key == "planes":
            return (self.capacity, self.in_channels, self.height, self.width)
        elif key == "visits":
            return (self.capacity, self.num_actions)
        else:
            return (self.capacity,)

    def _get_dtype(self, key: str) -> np.dtype:
        if key in ("planes", "visits"):
            return np.float32
        elif key == "outcomes":
            return np.int64
        elif key == "full_search":
            return np.bool_
        elif key == "versions":
            return np.int32
        elif key == "game_ids":
            return np.uint64
        raise ValueError(f"Unknown key {key}")

    def _save_meta(self, force: bool = True) -> None:
        """Save ring buffer state atomically (fsync + tmp + os.replace).

        Takes the buffer lock: concurrent writers (add_position(s) from game
        threads) otherwise race on the tmp file and snapshot torn counters
        (Windows raises access-denied on colliding replaces). No caller holds
        the lock when calling here (all call sites are outside it).
        """
        with self._lock:
            self._save_meta_locked(force=force)

    def _save_meta_locked(self, force: bool = True) -> None:
        """_save_meta body; caller must hold self._lock."""
        if not force:
            now = time.monotonic()
            self._games_since_meta += 1
            if (
                self._games_since_meta < self._meta_throttle_games
                and (now - self._last_meta_save) < self._meta_throttle_secs
            ):
                return
            self._games_since_meta = 0
            self._last_meta_save = now
        else:
            self._games_since_meta = 0
            self._last_meta_save = time.monotonic()
        if self.buffer_generation is None:
            # A buffer written before the generation marker existed adopts one
            # now, so subsequent checkpoints can pin their snapshot identity.
            self.buffer_generation = _mint_buffer_generation()
        meta = {
            "capacity": self.capacity,
            "in_channels": self.in_channels,
            "height": self.height,
            "width": self.width,
            "num_actions": self.num_actions,
            "write_head": self.write_head,
            "size": self.size,
            "total_added": self.total_added,
            "tactical_pool_size": getattr(self, "tactical_pool_size", 0),
            # Schema/integrity contract (audit SR-07, SR-10, SR-04).
            "schema_version": int(getattr(self, "schema_version", SCHEMA_VERSION)),
            "generation": int(getattr(self, "generation", 1)),
            "buffer_id": getattr(self, "buffer_id", None),
            "shards": self._expected_shard_layout(),
            # Checkpoint-resume identity (finding CKP-05): a distinct save-time
            # generation id consumed by the trainer's buffer restoration
            # contract. ``generation`` above is the directory re-creation
            # counter (SR-04); the two are different identities and both are
            # load-bearing.
            "buffer_generation": self.buffer_generation,
        }
        checksums = getattr(self, "_shard_checksums", None)
        if checksums:
            meta["checksums"] = dict(checksums)
        meta_path = self.paths["meta"]
        tmp_path = meta_path.with_suffix(".tmp")
        with _buffer_artifact_lock(meta_path, owner="replay-meta", timeout=10.0):
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    logger.debug("fsync meta tmp failed", exc_info=True)
            try:
                os.replace(str(tmp_path), str(meta_path))
            except OSError:
                logger.warning("os.replace for meta failed; writing in place", exc_info=True)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        logger.debug("fsync meta failed", exc_info=True)

    def _load_meta(self) -> None:
        """Load ring buffer state with strict validation (raises, never silent reset)."""
        meta_path = self.paths["meta"]
        tmp_path = meta_path.with_suffix(".tmp")
        if not meta_path.exists() or meta_path.stat().st_size == 0:
            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                try:
                    os.replace(str(tmp_path), str(meta_path))
                except OSError:
                    pass
            elif self._looks_populated():
                # REM-06 (same defect family as SR-10): meta.json is gone or
                # empty while the shards still contain records. Silently
                # writing a fresh size=0 meta would make the existing records
                # invisible (and later inserts would overwrite them from slot
                # 0). Refuse, with a diagnostic naming the recovery options.
                raise BufferIntegrityError(
                    f"Replay meta {meta_path} is missing/empty but the shard files at "
                    f"{self.buffer_dir} contain records. Refusing to silently reset a "
                    "populated buffer. Restore meta.json (or a valid meta scratch file), "
                    "or pass create_new=True / --reset-buffer to explicitly start a fresh "
                    "generation."
                )
            else:
                self._save_meta(force=True)
                return

        def _backup_corrupt() -> None:
            try:
                ts = int(time.time())
                backup = meta_path.with_name(f"meta.corrupt.{ts}.json")
                if meta_path.exists() and meta_path.stat().st_size > 0:
                    backup.write_bytes(meta_path.read_bytes())
                elif tmp_path.exists() and tmp_path.stat().st_size > 0:
                    backup.write_bytes(tmp_path.read_bytes())
                logger.warning("Backed up corrupt meta to %s", backup)
            except OSError:
                logger.warning("Failed to back up corrupt meta", exc_info=True)

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            _backup_corrupt()
            raise ValueError(f"Corrupt replay meta at {meta_path}: {e}") from e
        try:
            capacity = int(meta["capacity"])
            write_head = int(meta.get("write_head", 0))
            size = int(meta.get("size", 0))
            total_added = int(meta.get("total_added", size))
        except (KeyError, TypeError, ValueError) as e:
            _backup_corrupt()
            raise ValueError(f"Corrupt replay meta fields at {meta_path}: {e}") from e
        if capacity != self.capacity:
            raise ValueError(f"Meta capacity {capacity} != arg {self.capacity}")
        if not (0 <= write_head < capacity):
            raise ValueError(f"Meta write_head {write_head} out of range [0, {capacity})")
        if not (0 <= size <= capacity):
            raise ValueError(f"Meta size {size} out of range [0, {capacity}]")
        if total_added < 0:
            raise ValueError(f"Meta total_added {total_added} < 0")

        # ── Schema contract (audit SR-07/SR-10): fail closed on anything the
        # metadata describes but cannot validate. A missing schema_version
        # means the legacy format 1 buffer (e.g. the 2026-10-26 96,067-record
        # artifact); a version NEWER than this code understands is refused
        # rather than misinterpreted. ──
        schema_version = int(meta.get("schema_version", 1))
        if schema_version > SCHEMA_VERSION:
            raise BufferIntegrityError(
                f"Replay meta at {meta_path} declares schema_version {schema_version}, but this "
                f"code understands up to {SCHEMA_VERSION}. Refusing to load a newer format "
                "(no silent downgrade)."
            )
        self.schema_version = schema_version
        self.generation = int(meta.get("generation", 1))
        declared_buffer_id = meta.get("buffer_id")
        if declared_buffer_id is not None and declared_buffer_id != self.buffer_id:
            # Same directory should hash to the same id; a difference means the
            # meta was copied from another location. Not fatal (the records are
            # still this directory's records), but the checkpoint contract must
            # know: report it and record the mismatch.
            logger.warning(
                "[BUFFER] Meta buffer_id %s does not match this directory's id %s "
                "(meta copied between directories?); using the on-disk id",
                declared_buffer_id,
                self.buffer_id,
            )
        checksums = meta.get("checksums")
        if isinstance(checksums, dict):
            self._shard_checksums = {str(k): str(v) for k, v in checksums.items()}

        # Exact per-shard byte-size verification (audit SR-10): undersized
        # shards would raise at memmap open, but OVERSIZED ones open silently
        # with the extra bytes ignored, and a wrong-size file means the shard
        # does not belong to this meta. Both directions fail closed here.
        self._verify_shard_sizes(meta_path)

        # Scalar-domain verification of every live record (audit SR-10): a
        # right-sized but foreign/zero-filled/garbled scalar shard alongside a
        # valid meta must not load silently.
        self._verify_scalar_contract(meta_path, size=size)

        self.write_head = write_head
        self.size = size
        self.total_added = total_added
        generation = meta.get("buffer_generation")
        # Legacy meta (pre-generation) leaves this None; the resume contract
        # treats "unknown" conservatively rather than inventing an identity.
        self.buffer_generation = str(generation) if generation else None
        if "tactical_pool_size" in meta:
            self.saved_tactical_pool_size = int(meta["tactical_pool_size"])
            self.tactical_pool_size = self.saved_tactical_pool_size

    # ── Schema / integrity helpers (audit SR-04, SR-07, SR-10) ─────────────

    def _compute_buffer_id(self) -> str:
        """Stable identity of this buffer directory (checkpoint pairing)."""
        import hashlib

        try:
            resolved = os.path.normcase(str(self.buffer_dir.resolve()))
        except OSError:
            resolved = os.path.normcase(str(self.buffer_dir))
        return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]

    def _expected_shard_layout(self) -> Dict[str, Dict[str, Any]]:
        """The byte-layout contract written into meta and verified on load."""
        layout: Dict[str, Dict[str, Any]] = {}
        for key in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
            shape = self._get_shape(key)
            dtype = self._get_dtype(key)
            layout[key] = {
                "shape": [int(s) for s in shape],
                "dtype": str(np.dtype(dtype)),
                "bytes": _shard_nbytes(shape, np.dtype(dtype)),
            }
        return layout

    def _looks_populated(self) -> bool:
        """True when shard files exist and carry record data (REM-06 guard).

        Cheap proxy over the four scalar shards (a populated buffer always has
        ``game_ids > 0``); used only on the missing/empty-meta path so the cost
        (~10 MB worst case at capacity 500k, tens of ms) never touches the
        normal load path.
        """
        for key, dtype, shape in (
            ("game_ids", np.uint64, (self.capacity,)),
            ("outcomes", np.int64, (self.capacity,)),
            ("versions", np.int32, (self.capacity,)),
            ("full_search", np.bool_, (self.capacity,)),
        ):
            path = self.paths[key]
            if not path.exists():
                continue
            try:
                m = np.memmap(path, dtype=dtype, mode="r", shape=shape)
                try:
                    if bool(np.asarray(m).any()):
                        return True
                finally:
                    try:
                        m._mmap.close()  # type: ignore[attr-defined]
                    except (AttributeError, ValueError, OSError):
                        pass
                    del m
            except (OSError, ValueError):
                # Unreadable shard on the recovery path: let normal load fail
                # closed with its own diagnostic rather than guessing here.
                continue
        return False

    def _read_existing_generation(self) -> int:
        """Generation recorded in an existing meta (0 when none/unreadable)."""
        try:
            if self.paths["meta"].exists():
                with open(self.paths["meta"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                return int(data.get("generation", 1))
        except (OSError, ValueError, TypeError):
            pass
        return 0

    def _verify_shard_sizes(self, meta_path: Path) -> None:
        """Verify each shard file exists and has EXACTLY the expected bytes."""
        expected = self._expected_shard_layout()
        for key, spec in expected.items():
            path = self.paths[key]
            if not path.exists():
                raise BufferIntegrityError(
                    f"Replay shard {path} is missing but {meta_path} describes a populated "
                    "buffer. Refusing to load (no silent re-creation)."
                )
            actual = path.stat().st_size
            if actual != spec["bytes"]:
                raise BufferIntegrityError(
                    f"Replay shard {path} has {actual} bytes; expected exactly {spec['bytes']} "
                    f"({spec['shape']} {spec['dtype']}). The shard does not match the metadata "
                    "contract (oversized or truncated); refusing to load."
                )

    def _verify_scalar_contract(self, meta_path: Path, size: Optional[int] = None) -> None:
        """Validate the scalar domains of every live record at load.

        Cheap (O(size) over four scalar shards; ~a few MB at 500k records),
        read-only, and fail-closed: outcomes must be {0,1,2}, versions >= 0,
        game_ids > 0 for every record in [0, size). This also detects a
        zero-filled or foreign scalar shard that otherwise matches the size
        contract. ``size`` is passed explicitly because this runs during
        ``_load_meta`` before ``self.size`` is assigned from the metadata.
        """
        size = int(self.size if size is None else size)
        if size <= 0:
            return
        checks = (
            ("outcomes", np.int64, lambda a: np.isin(a, (0, 1, 2)).all(),
             "values outside {0=Win, 1=Draw, 2=Loss}"),
            ("versions", np.int32, lambda a: bool((a >= 0).all()),
             "negative model versions"),
            ("game_ids", np.uint64, lambda a: bool((a > 0).all()),
             "zero game ids"),
        )
        for key, dtype, predicate, description in checks:
            path = self.paths[key]
            m = np.memmap(path, dtype=dtype, mode="r", shape=(self.capacity,))
            try:
                window = np.array(m[:size])
            finally:
                try:
                    m._mmap.close()  # type: ignore[attr-defined]
                except (AttributeError, ValueError, OSError):
                    pass
                del m
            if not predicate(window):
                bad = np.nonzero(~np.isin(window, (0, 1, 2)))[0] if key == "outcomes" else None
                detail = f"first offending index {int(bad[0])}" if bad is not None and bad.size else ""
                raise BufferIntegrityError(
                    f"Replay shard {key} contains {description} within [0, {size}) "
                    f"({detail}); {meta_path} cannot be validated against its data. "
                    "Refusing to load (fail closed)."
                )

    def snapshot(self) -> Dict[str, Any]:
        """Point-in-time state descriptor used by the checkpoint contract.

        A checkpoint stores this dict; a resume compares it against the LIVE
        buffer via :meth:`checkpoint_consistency` before any training state is
        adopted (audit SR-04).
        """
        return {
            "buffer_id": self.buffer_id,
            "capacity": int(self.capacity),
            "schema_version": int(self.schema_version),
            "generation": int(self.generation),
            "size": int(self.size),
            "write_head": int(self.write_head),
            "total_added": int(self.total_added),
        }

    @property
    def gpu_cache_coverage(self) -> float:
        """Fraction of the live ring the GPU cache can serve (live-computed).

        Recomputed on access so it stays truthful as records are appended after
        cache construction: ``min(1, cache_size / ring_size)`` (audit SR-06).
        """
        if self.gpu_cache is None or self.size <= 0:
            return 0.0
        return min(1.0, float(len(self.gpu_cache)) / float(self.size))

    def checkpoint_consistency(self, snapshot: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """Classify a checkpoint's stored buffer snapshot against the live buffer.

        Returns ``(verdict, detail)`` where verdict is one of:

        - ``"match"``   — the checkpoint describes exactly the live state;
        - ``"ahead"``   — the live buffer verifiably contains the checkpoint
          state plus a consistent, monotonically appended suffix (the normal
          crash-recovery case: inserts continued after the checkpoint was
          written). The live state is authoritative and is NOT rewound;
        - ``"mismatch"``— any other combination: different buffer directory,
          capacity, schema, generation, or pointer/count arithmetic that does
          not correspond to appends. The caller must fail closed unless the
          operator explicitly overrides.

        The buffer pointer is NEVER rolled backward from a checkpoint: meta is
        persisted atomically after every insert, so the live state is always at
        least as fresh as any checkpoint written from it.
        """
        if not snapshot:
            return "mismatch", "checkpoint carries no buffer snapshot (legacy format)"

        def _get(key: str) -> Optional[int]:
            val = snapshot.get(key)
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        snap_id = snapshot.get("buffer_id")
        if snap_id is not None and str(snap_id) != self.buffer_id:
            return (
                "mismatch",
                f"different buffer directory (checkpoint {snap_id} != live {self.buffer_id})",
            )
        snap_cap = _get("capacity")
        if snap_cap is not None and snap_cap != self.capacity:
            return "mismatch", f"capacity {snap_cap} != live {self.capacity}"
        snap_schema = _get("schema_version")
        if snap_schema is not None and snap_schema != self.schema_version:
            return "mismatch", f"schema_version {snap_schema} != live {self.schema_version}"
        # Integrated contract: a unified checkpoint ``buffer_state`` carries the
        # CKP-05 save generation under ``generation`` (a string id) and this
        # directory-generation counter under ``dir_generation``. Older
        # SR-04-era snapshots carried the int directly under ``generation``.
        if "dir_generation" in snapshot:
            snap_gen = _get("dir_generation")
        else:
            snap_gen = _get("generation")
        if snap_gen is not None and snap_gen != self.generation:
            return (
                "mismatch",
                f"generation {snap_gen} != live {self.generation} (the buffer was re-created)",
            )

        snap_size, snap_head, snap_total = (
            _get("size"),
            _get("write_head"),
            _get("total_added"),
        )
        live = (self.size, self.write_head, self.total_added)

        if snap_total is None:
            # Legacy checkpoint (only buffer_size / buffer_write_head existed).
            if (snap_size, snap_head) == (live[0], live[1]):
                return "match", f"legacy snapshot matches live state (size={live[0]}, head={live[1]})"
            return (
                "mismatch",
                f"legacy snapshot (size={snap_size}, write_head={snap_head}) does not match live "
                f"(size={live[0]}, write_head={live[1]}); total_added was never recorded, so "
                "'ahead' cannot be verified",
            )

        if (snap_size, snap_head, snap_total) == live:
            return "match", f"size={live[0]}, write_head={live[1]}, total_added={live[2]}"

        appended = live[2] - snap_total
        if appended > 0 and snap_size is not None and snap_head is not None:
            expected_head = (snap_head + appended) % self.capacity
            if live[1] == expected_head and live[0] == min(self.capacity, snap_size + appended):
                return (
                    "ahead",
                    f"live buffer contains the checkpoint state plus {appended} appended record(s) "
                    f"(size {snap_size}->{live[0]}, head {snap_head}->{live[1]})",
                )
        return (
            "mismatch",
            f"snapshot (size={snap_size}, write_head={snap_head}, total_added={snap_total}) is "
            f"inconsistent with live (size={live[0]}, write_head={live[1]}, total_added={live[2]})",
        )

    def compute_shard_checksums(self, chunk_bytes: int = 8 * 1024 * 1024) -> Dict[str, str]:
        """Stream-compute SHA-256 for each shard file.

        O(buffer bytes): ~11.5 GB of reads at capacity 500k. This is a
        deliberate, explicit operation (data-freeze verification, forensic
        checks) — it is never run on the hot path and never on load.
        """
        import hashlib

        if not self.in_memory:
            for name in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
                arr = getattr(self, name, None)
                if isinstance(arr, np.memmap):
                    try:
                        arr.flush()
                    except (ValueError, OSError):
                        logger.debug("checksum pre-flush failed for %s", name, exc_info=True)
        sums: Dict[str, str] = {}
        for key in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
            path = self.paths[key]
            if not path.exists():
                continue
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_bytes), b""):
                    h.update(chunk)
            sums[key] = h.hexdigest()
        return sums

    def write_shard_checksums(self) -> Dict[str, str]:
        """Compute and persist shard checksums into meta (atomic)."""
        sums = self.compute_shard_checksums()
        self._shard_checksums = sums
        self._save_meta(force=True)
        logger.info("[BUFFER] Persisted shard checksums for %d shard(s)", len(sums))
        return sums

    def verify_shards(self, deep: bool = False) -> Dict[str, Any]:
        """Integrity verification report for the buffer (audit SR-10).

        Cheap checks (always): shard existence + exact byte sizes, scalar
        domains of live records, and (report-only) that slots at/after ``size``
        still carry the zero-fill from preallocation in the current generation.

        Deep checks (``deep=True``): additionally recompute every shard's
        SHA-256 and compare with the stored ``meta.checksums`` (written by
        :meth:`write_shard_checksums`). O(buffer bytes).

        Returns ``{"ok": bool, "errors": [...], "warnings": [...],
        "checks": {...}}``; never raises for data conditions.
        """
        report: Dict[str, Any] = {"ok": True, "errors": [], "warnings": [], "checks": {}}
        # 1. Sizes
        expected = self._expected_shard_layout()
        for key, spec in expected.items():
            path = self.paths[key]
            if not path.exists():
                report["ok"] = False
                report["errors"].append(f"missing shard {key} ({path})")
                continue
            actual = path.stat().st_size
            report["checks"].setdefault("shard_bytes", {})[key] = actual
            if actual != spec["bytes"]:
                report["ok"] = False
                report["errors"].append(
                    f"shard {key} has {actual} bytes; expected {spec['bytes']}"
                )
        # 2. Scalar domains (live records only)
        if self.size:
            try:
                self._verify_scalar_contract(self.paths["meta"])
                report["checks"]["scalar_domains"] = "ok"
            except BufferIntegrityError as exc:
                report["ok"] = False
                report["errors"].append(str(exc))
        # 3. Report-only: unused tail must be untouched since preallocation.
        if self.size < self.capacity and not self.in_memory:
            tail_hits = []
            for key in ("outcomes", "versions", "game_ids", "full_search"):
                arr = getattr(self, key, None)
                if arr is None:
                    continue
                tail = np.asarray(arr[self.size : self.capacity])
                if tail.size and bool(tail.any()):
                    tail_hits.append(key)
            if tail_hits:
                report["warnings"].append(
                    "slots at/after size are non-zero in "
                    f"{tail_hits} (expected after a --reset-buffer over existing files; "
                    "never sampled because bounds are [0, size))"
                )
            report["checks"]["tail_zero"] = "ok" if not tail_hits else "non-zero-stale"
        # 4. Deep: checksum comparison
        if deep:
            stored = dict(self._shard_checksums or {})
            computed = self.compute_shard_checksums()
            report["checks"]["checksums_computed"] = True
            if not stored:
                report["warnings"].append(
                    "no stored checksums in meta (write_shard_checksums() was never run); "
                    "reporting computed hashes without comparison"
                )
                report["checks"]["computed_sha256"] = computed
            else:
                mismatches = [
                    key
                    for key, digest in stored.items()
                    if key in computed and computed[key] != digest
                ]
                missing = [key for key in stored if key not in computed]
                if mismatches:
                    report["ok"] = False
                    report["errors"].append(f"checksum mismatch for shard(s): {sorted(mismatches)}")
                if missing:
                    report["ok"] = False
                    report["errors"].append(f"stores checksums for missing shard(s): {sorted(missing)}")
                report["checks"]["checksums_verified"] = not (mismatches or missing)
        return report

    def buffer_state(self) -> Dict[str, Any]:
        """The (write_head, size, total_added, capacity, generation) snapshot.

        This is the exact state a checkpoint must persist to be restorable
        against a ring buffer without mixing generations (audit CKP-05).
        """
        return {
            "write_head": int(self.write_head),
            "size": int(self.size),
            "total_added": int(self.total_added),
            "capacity": int(self.capacity),
            "generation": self.buffer_generation,
        }

    def _load_from_disk_to_ram(self) -> None:
        """Load existing disk memmap files into host RAM arrays."""
        for key in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
            path = self.paths[key]
            if path.exists():
                shape = self._get_shape(key)
                dtype = self._get_dtype(key)
                m = np.memmap(path, dtype=dtype, mode="r", shape=shape)
                try:
                    arr = getattr(self, key)
                    arr[:] = m[:]
                finally:
                    try:
                        m._mmap.close()  # type: ignore[attr-defined]
                    except (AttributeError, ValueError, OSError):
                        logger.debug("memmap close failed for %s", key, exc_info=True)
                    del m

    def flush_to_disk(self) -> None:
        """Persist buffer state and arrays to disk files."""
        if not self.in_memory:
            for arr_name in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
                arr = getattr(self, arr_name, None)
                if isinstance(arr, np.memmap):
                    arr.flush()
            self._save_meta(force=True)
            return

        with _buffer_artifact_lock(
            self.buffer_dir / "flush.lock", owner="replay-flush", timeout=30.0
        ):
            for key in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
                path = self.paths[key]
                tmp = path.with_name(path.name + ".tmp")
                arr = getattr(self, key)
                shape = self._get_shape(key)
                dtype = self._get_dtype(key)
                m = np.memmap(tmp, dtype=dtype, mode="w+", shape=shape)
                try:
                    m[:] = arr[:]
                    m.flush()
                finally:
                    try:
                        m._mmap.close()  # type: ignore[attr-defined]
                    except (AttributeError, ValueError, OSError):
                        logger.debug("transient memmap close failed for %s", key, exc_info=True)
                    del m
                os.replace(str(tmp), str(path))
            self._save_meta(force=True)

    def enable_gpu_cache(
        self,
        capacity: int = 20_000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> GpuResidentReplayCache:
        """Enable Tier 1 GPU-resident cache in VRAM for zero-PCIe training sampling."""
        dev = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gpu_device = dev

        # Adaptive VRAM safety clamp: preserve at least 2.0 GB headroom for activations/OS
        if dev.type == "cuda" and torch.cuda.is_available():
            try:
                free_vram, total_vram = torch.cuda.mem_get_info(dev)
                safe_budget = max(0, free_vram - int(2.0 * 1024**3))
                # Bytes per position: 4864 + 18688 + 8 + 1 = 23,561 bytes
                max_safe_cap = int(safe_budget // 23561)
                if capacity > max_safe_cap:
                    clamped = max(1000, max_safe_cap)
                    logger.warning(
                        "Clamping requested GPU cache capacity %d to safe budget %d based on free VRAM (%.2f GB free / %.2f GB total)",
                        capacity,
                        clamped,
                        free_vram / (1024**3),
                        total_vram / (1024**3),
                    )
                    capacity = clamped
            except Exception as e:
                logger.debug("Failed to query CUDA VRAM info: %s", e)

        if self.gpu_cache is not None:
            if self.gpu_cache.device == dev and self.gpu_cache.capacity == capacity:
                return self.gpu_cache
            try:
                self.gpu_cache.close()
            except (RuntimeError, ValueError) as e:
                logger.debug("old gpu cache close failed: %s", e)
            self.gpu_cache = None
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.gpu_cache = GpuResidentReplayCache(
            capacity=capacity,
            in_channels=self.in_channels,
            height=self.height,
            width=self.width,
            num_actions=self.num_actions,
            device=dev,
        )

        # Pre-populate GPU cache with recent positions from buffer if available.
        # Real model versions are carried over so restored rows keep their
        # provenance (a wholesale version 0 would mark every restored row
        # "fresh" / or "unknown" and corrupt staleness filtering).
        if self.size > 0:
            fill_n = min(self.size, capacity)
            if self.size < self.capacity:
                start_idx = max(0, self.write_head - fill_n)
                end_idx = self.write_head
                self.gpu_cache.add_positions(
                    planes=np.array(self.planes[start_idx:end_idx]),
                    visits=np.array(self.visits[start_idx:end_idx]),
                    outcomes=np.array(self.outcomes[start_idx:end_idx]),
                    full_search=np.array(self.full_search[start_idx:end_idx]),
                    versions=np.array(self.versions[start_idx:end_idx]),
                    game_ids=np.array(self.game_ids[start_idx:end_idx], dtype=np.int64),
                )
            else:
                recent_indices = [
                    (self.write_head - fill_n + i) % self.capacity for i in range(fill_n)
                ]
                self.gpu_cache.add_positions(
                    planes=np.array(self.planes[recent_indices]),
                    visits=np.array(self.visits[recent_indices]),
                    outcomes=np.array(self.outcomes[recent_indices]),
                    full_search=np.array(self.full_search[recent_indices]),
                    versions=np.array(self.versions[recent_indices]),
                    game_ids=np.array(self.game_ids[recent_indices], dtype=np.int64),
                )

        # Coverage telemetry (audit SR-06): how much of the live ring this
        # cache can serve, and therefore how much of each self-play request is
        # split off to the ring under the default "ring_uniform" policy.
        logger.info(
            "Enabled GPU-resident replay cache with capacity %d on %s "
            "(preloaded %d row(s); covers %s of the %d live ring records; self-play sampling "
            "policy=%s)",
            capacity,
            dev,
            len(self.gpu_cache),
            f"{self.gpu_cache_coverage * 100:.1f}%",
            self.size,
            self.selfplay_sampling_policy,
        )

        # Also migrate human dataset active cache to GPU VRAM if loaded
        if getattr(self, "human_dataset", None) is not None:
            if self._human_gpu_planes is None or self._human_cache_device != dev:
                target_human_size = max(
                    getattr(self, "_human_cache_size", 0), getattr(self, "human_cache_size", 20_000)
                )
                self._load_human_cache(cache_size=target_human_size, device=dev)

        # Also migrate tactical dataset active cache to GPU VRAM if loaded
        if getattr(self, "tactical_pool", None) is not None:
            if self._tactical_gpu_planes is None or self._tactical_cache_device != dev:
                target_tactical_size = max(
                    getattr(self, "_tactical_cache_size", 0),
                    getattr(self, "tactical_cache_size", 20_000),
                )
                self._load_tactical_cache(cache_size=target_tactical_size, device=dev)

        return self.gpu_cache

    def _init_human_dataset(
        self,
        data_dir: Union[str, Path],
        cache_size: int = 20_000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Load human dataset from data directory with in-memory or GPU-resident active cache."""
        from python.data.dataset import BinaryChessDataset

        p = Path(data_dir)
        if not p.is_dir() and not p.is_absolute():
            try:
                from python.cli.common import REPO_ROOT

                if (REPO_ROOT / p).is_dir():
                    p = REPO_ROOT / p
            except Exception:
                pass
        if (p / "metadata.json").exists():
            # Consumption gate (audit F-04/F-01): the human replay data must
            # satisfy its declared role and match a passing validation receipt
            # before a single position is ingested.
            from python.data.dataset_validation import require_validated_dataset

            gate = require_validated_dataset(
                p,
                expected_role=self.data_role,
                allow_uncertified=self.allow_uncertified_data,
                deep=True,
                consumer="replay-buffer[human]",
            )
            role_name = gate.role.name if gate.role else "unclassified"
            logger.info(
                "Human data gate: role=%s certified=%s masked_fraction=%.4f samples=%d (%s)",
                role_name,
                gate.certified,
                gate.masked_fraction,
                gate.num_samples,
                p,
            )
            for warning in gate.warnings:
                logger.warning(warning)
            self.human_dataset = BinaryChessDataset(p)
            logger.info(f"Loaded human dataset with {len(self.human_dataset)} positions from {p}")
            self._load_human_cache(cache_size=cache_size, device=device)
        else:
            logger.warning(f"Human dataset not found at {p}. Proceeding without human mix.")

    def _cache_sample_rng(self) -> np.random.Generator:
        """Audit RL-25: the generator that picks cache composition.

        Seeded runs get one persistent seeded generator (same --seed → same
        cache composition across restarts); unseeded runs keep entropy-seeded
        draws. Refreshes advance the same generator, so the sequence stays
        deterministic for a seeded run.
        """
        if self._cache_rng is not None:
            return self._cache_rng
        return np.random.default_rng()

    def _load_human_cache(
        self,
        cache_size: int = 20_000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Cache an active sample of human positions in GPU VRAM (or RAM fallback) to eliminate disk page thrashing."""
        if self.human_dataset is None or len(self.human_dataset) == 0:
            return
        total = len(self.human_dataset)
        n = min(cache_size, total)

        target_dev = None
        if device is not None:
            target_dev = torch.device(device)
        elif getattr(self, "gpu_cache", None) is not None:
            target_dev = self.gpu_cache.device
        elif getattr(self, "gpu_device", None) is not None:
            target_dev = self.gpu_device
        elif torch.cuda.is_available():
            target_dev = torch.device("cuda")

        if target_dev is not None and target_dev.type == "cuda" and torch.cuda.is_available():
            try:
                free_vram, _ = torch.cuda.mem_get_info(target_dev)
                safe_budget = max(0, free_vram - int(1.5 * 1024**3))
                max_safe_n = int(safe_budget // 4880)
                if n > max_safe_n:
                    clamped_n = max(500, max_safe_n)
                    logger.warning(
                        "Clamping human GPU cache from %d to %d due to limited free VRAM (%.2f GB)",
                        n,
                        clamped_n,
                        free_vram / (1024**3),
                    )
                    n = min(n, clamped_n)
            except Exception as e:
                logger.debug("mem_get_info check in human cache failed: %s", e)

        try:
            # O(n) sampling without replacement (Floyd's algorithm). The legacy
            # np.random.choice(replace=False) path permutes the whole dataset, so its
            # cost grows with the dataset size (millions of entries) and it stalls the
            # training thread that calls refresh_human_cache().
            if n >= total:
                indices = np.arange(total)
            else:
                indices = self._cache_sample_rng().choice(total, size=n, replace=False)
            planes_np = np.ascontiguousarray(self.human_dataset.planes[indices], dtype=np.float32)
            moves_np = np.ascontiguousarray(self.human_dataset.moves[indices], dtype=np.int64)
            outcomes_np = np.ascontiguousarray(self.human_dataset.outcomes[indices], dtype=np.int64)

            if target_dev is not None and target_dev.type == "cuda" and torch.cuda.is_available():
                # Direct placement in GPU VRAM - zero host RAM used!
                self._human_gpu_planes = torch.from_numpy(planes_np).to(
                    target_dev, non_blocking=True
                )
                self._human_gpu_moves = torch.from_numpy(moves_np).to(target_dev, non_blocking=True)
                self._human_gpu_outcomes = torch.from_numpy(outcomes_np).to(
                    target_dev, non_blocking=True
                )
                self._human_cache_size = n
                self._human_cache_device = target_dev

                # Clear CPU references to keep Host RAM at 0 bytes
                self._human_cache_planes = None
                self._human_cache_moves = None
                self._human_cache_outcomes = None

                mem_mb = (
                    self._human_gpu_planes.nbytes
                    + self._human_gpu_moves.nbytes
                    + self._human_gpu_outcomes.nbytes
                ) / (1024.0 * 1024.0)
                logger.info(
                    f"Preloaded {n} human positions directly into GPU VRAM ({mem_mb:.1f} MB) on {target_dev} (Host RAM freed)."
                )
            else:
                # CPU RAM fallback
                self._human_cache_planes = planes_np
                self._human_cache_moves = moves_np
                self._human_cache_outcomes = outcomes_np
                self._human_cache_size = n
                self._human_cache_device = torch.device("cpu")
                self._human_gpu_planes = None
                self._human_gpu_moves = None
                self._human_gpu_outcomes = None

                mem_mb = (
                    self._human_cache_planes.nbytes
                    + self._human_cache_moves.nbytes
                    + self._human_cache_outcomes.nbytes
                ) / (1024.0 * 1024.0)
                logger.info(
                    f"Preloaded {n} human positions in host memory ({mem_mb:.1f} MB) on CPU."
                )
        except Exception as e:
            logger.warning(f"Failed to preload human cache ({e}); falling back to disk memmap.")
            self._human_cache_size = 0

    def refresh_human_cache(self, cache_size: Optional[int] = None) -> None:
        """Periodically refresh the cached human positions with fresh data."""
        target_size = (
            cache_size
            if cache_size is not None
            else max(getattr(self, "_human_cache_size", 80_000), self.human_cache_size)
        )
        self._load_human_cache(cache_size=target_size, device=self._human_cache_device)

    def _init_tactical_dataset(
        self,
        csv_path: Union[str, Path],
        cache_size: int = 20_000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize tactical position pool from CSV file."""
        p = Path(csv_path)
        if not p.is_file() and not p.is_absolute():
            try:
                from python.cli.common import REPO_ROOT

                for cand in [
                    REPO_ROOT / p,
                    REPO_ROOT / "data/tactics_train_300k.csv",
                    REPO_ROOT / "data/puzzles/tactics_train_300k.csv",
                    Path("data/tactics_train_300k.csv"),
                ]:
                    if cand.is_file():
                        p = cand
                        break
            except Exception:
                pass
        if not p.is_file():
            logger.warning("Tactical dataset not found at %s. Proceeding with tactical_mix=0.", p)
            self.tactical_pool = None
            self.tactical_mix = 0.0
            self.tactical_pool_size = 0
            self._tactical_cache_size = 0
            return

        try:
            planes, moves, outcomes = load_tactical_positions_from_csv(p, max_positions=cache_size)
            if len(moves) == 0:
                logger.warning(
                    "Tactical dataset at %s contains 0 valid positions. Proceeding with tactical_mix=0.",
                    p,
                )
                self.tactical_pool = None
                self.tactical_mix = 0.0
                self.tactical_pool_size = 0
                self._tactical_cache_size = 0
                return

            self.tactical_pool = TacticalPositionPool(planes, moves, outcomes, source_path=p)
            self.tactical_pool_size = len(self.tactical_pool)
            logger.info("Loaded %d tactical anchor positions from %s", self.tactical_pool_size, p)
            self._load_tactical_cache(cache_size=cache_size, device=device)
            self._save_meta(force=True)
        except Exception as e:
            logger.warning(
                "Failed to initialize tactical dataset from %s: %s; proceeding with tactical_mix=0.",
                p,
                e,
            )
            self.tactical_pool = None
            self.tactical_mix = 0.0
            self.tactical_pool_size = 0
            self._tactical_cache_size = 0

    def _load_tactical_cache(
        self,
        cache_size: int = 20_000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Cache an active sample of tactical positions in GPU VRAM (or RAM fallback)."""
        if self.tactical_pool is None or len(self.tactical_pool) == 0:
            return
        total = len(self.tactical_pool)
        n = min(cache_size, total)

        target_dev = None
        if device is not None:
            target_dev = torch.device(device)
        elif getattr(self, "gpu_cache", None) is not None:
            target_dev = self.gpu_cache.device
        elif getattr(self, "gpu_device", None) is not None:
            target_dev = self.gpu_device
        elif torch.cuda.is_available():
            target_dev = torch.device("cuda")

        if target_dev is not None and target_dev.type == "cuda" and torch.cuda.is_available():
            try:
                free_vram, _ = torch.cuda.mem_get_info(target_dev)
                safe_budget = max(0, free_vram - int(1.5 * 1024**3))
                max_safe_n = int(safe_budget // 4880)
                if n > max_safe_n:
                    clamped_n = max(500, max_safe_n)
                    logger.warning(
                        "Clamping tactical GPU cache from %d to %d due to limited free VRAM (%.2f GB)",
                        n,
                        clamped_n,
                        free_vram / (1024**3),
                    )
                    n = min(n, clamped_n)
            except Exception as e:
                logger.debug("mem_get_info check in tactical cache failed: %s", e)

        try:
            if n == total:
                planes_np = np.ascontiguousarray(self.tactical_pool.planes, dtype=np.float32)
                moves_np = np.ascontiguousarray(self.tactical_pool.moves, dtype=np.int64)
                outcomes_np = np.ascontiguousarray(self.tactical_pool.outcomes, dtype=np.int64)
            else:
                # Same O(n) sampler as the human cache: the legacy replace=False path
                # permutes the entire pool, which scales with dataset size.
                indices = self._cache_sample_rng().choice(total, size=n, replace=False)
                planes_np = np.ascontiguousarray(
                    self.tactical_pool.planes[indices], dtype=np.float32
                )
                moves_np = np.ascontiguousarray(self.tactical_pool.moves[indices], dtype=np.int64)
                outcomes_np = np.ascontiguousarray(
                    self.tactical_pool.outcomes[indices], dtype=np.int64
                )

            if target_dev is not None and target_dev.type == "cuda" and torch.cuda.is_available():
                self._tactical_gpu_planes = torch.from_numpy(planes_np).to(
                    target_dev, non_blocking=True
                )
                self._tactical_gpu_moves = torch.from_numpy(moves_np).to(
                    target_dev, non_blocking=True
                )
                self._tactical_gpu_outcomes = torch.from_numpy(outcomes_np).to(
                    target_dev, non_blocking=True
                )
                self._tactical_cache_size = n
                self._tactical_cache_device = target_dev

                self._tactical_cache_planes = None
                self._tactical_cache_moves = None
                self._tactical_cache_outcomes = None

                mem_mb = (
                    self._tactical_gpu_planes.nbytes
                    + self._tactical_gpu_moves.nbytes
                    + self._tactical_gpu_outcomes.nbytes
                ) / (1024.0 * 1024.0)
                logger.info(
                    f"Preloaded {n} tactical positions directly into GPU VRAM ({mem_mb:.1f} MB) on {target_dev}."
                )
            else:
                self._tactical_cache_planes = planes_np
                self._tactical_cache_moves = moves_np
                self._tactical_cache_outcomes = outcomes_np
                self._tactical_cache_size = n
                self._tactical_cache_device = torch.device("cpu")
                self._tactical_gpu_planes = None
                self._tactical_gpu_moves = None
                self._tactical_gpu_outcomes = None

                mem_mb = (
                    self._tactical_cache_planes.nbytes
                    + self._tactical_cache_moves.nbytes
                    + self._tactical_cache_outcomes.nbytes
                ) / (1024.0 * 1024.0)
                logger.info(
                    f"Preloaded {n} tactical positions in host memory ({mem_mb:.1f} MB) on CPU."
                )
        except Exception as e:
            logger.warning(f"Failed to preload tactical cache ({e}); tactical mix disabled.")
            self._tactical_cache_size = 0

    def refresh_tactical_cache(self, cache_size: Optional[int] = None) -> None:
        """Periodically refresh the cached tactical positions with fresh data."""
        target_size = (
            cache_size
            if cache_size is not None
            else max(getattr(self, "_tactical_cache_size", 20_000), self.tactical_cache_size)
        )
        self._load_tactical_cache(cache_size=target_size, device=self._tactical_cache_device)

    def add_position(self, pos: SavedPosition) -> None:
        """Insert a single position into the ring buffer.

        The record is validated before anything is written (audit SR-07); an
        invalid record raises ``ReplayRecordValidationError`` and leaves the
        ring, meta and GPU cache untouched.
        """
        try:
            validate_position_record(
                pos,
                in_channels=self.in_channels,
                height=self.height,
                width=self.width,
                num_actions=self.num_actions,
            )
        except ReplayRecordValidationError:
            self.rejected_records += 1
            raise
        with self._lock:
            idx = self.write_head
            self.planes[idx] = pos.planes
            self.visits[idx] = pos.visits
            self.outcomes[idx] = pos.outcome
            self.full_search[idx] = pos.is_full_search
            self.versions[idx] = pos.model_version
            self.game_ids[idx] = pos.game_id

            self.write_head = (self.write_head + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)
            self.total_added += 1
        self._save_meta(force=True)

        if self.gpu_cache is not None:
            self.gpu_cache.add_positions(
                planes=np.expand_dims(pos.planes, 0),
                visits=np.expand_dims(pos.visits, 0),
                outcomes=np.array([pos.outcome], dtype=np.int64),
                full_search=np.array([pos.is_full_search], dtype=np.bool_),
                versions=np.array([pos.model_version], dtype=np.int32),
                game_ids=np.array([pos.game_id], dtype=np.int64),
            )

    def add_positions(self, positions: List[SavedPosition]) -> None:
        """Insert a batch of positions into the ring buffer (tail-take if n > capacity).

        Validation is transactional (audit SR-07): the whole flush is validated
        BEFORE the lock is taken and any byte is written, so a malformed record
        rejects the entire flush and never leaves a partially committed record
        or a desynchronized shard. ``total_added`` counts every record offered
        to the ring, including records dropped by an oversized-flush truncation
        (audit SR-11), so resetting/pacing arithmetic cannot understate
        production.
        """
        if not positions:
            return

        n_offered = len(positions)
        # Shape pre-check before stacking: a record with an inconsistent
        # plane/visit shape would otherwise raise a bare numpy stacking error
        # instead of the validation contract's diagnostic (and must reject the
        # whole flush, not crash halfway through array assembly).
        expected_planes = (self.in_channels, self.height, self.width)
        expected_visits = (self.num_actions,)
        for idx, p in enumerate(positions):
            if tuple(np.shape(p.planes)) != expected_planes or tuple(np.shape(p.visits)) != expected_visits:
                self.rejected_records += n_offered
                raise ReplayRecordValidationError(
                    f"record {idx} (game_id={getattr(p, 'game_id', '?')}) has planes shape "
                    f"{tuple(np.shape(p.planes))} / visits shape {tuple(np.shape(p.visits))}; "
                    f"expected {expected_planes} / {expected_visits}. Whole flush rejected; "
                    "no bytes written."
                )
        # Vectorized assembly of the full flush (before truncation, so a
        # malformed record anywhere in the game's experience rejects it).
        planes_arr = np.stack([np.asarray(p.planes, dtype=np.float32) for p in positions], axis=0)
        visits_arr = np.stack([np.asarray(p.visits, dtype=np.float32) for p in positions], axis=0)
        outcomes_arr = np.array([p.outcome for p in positions], dtype=np.int64)
        full_search_arr = np.array([p.is_full_search for p in positions], dtype=np.bool_)
        versions_arr = np.array([p.model_version for p in positions], dtype=np.int32)
        game_ids_arr = np.array([p.game_id for p in positions], dtype=np.uint64)

        try:
            validate_records_arrays(
                planes_arr,
                visits_arr,
                outcomes_arr,
                full_search_arr,
                versions_arr,
                game_ids_arr,
                in_channels=self.in_channels,
                height=self.height,
                width=self.width,
                num_actions=self.num_actions,
            )
        except ReplayRecordValidationError:
            self.rejected_records += n_offered
            logger.error(
                "[BUFFER] Rejected a flush of %d record(s): insertion validation failed; "
                "no bytes written",
                n_offered,
            )
            raise

        if n_offered > self.capacity:
            # Keep only the newest `capacity` entries (consistent with GPU cache).
            logger.warning(
                "[BUFFER] Oversized flush of %d record(s) exceeds capacity %d; "
                "keeping the newest %d (total_added still counts all %d)",
                n_offered,
                self.capacity,
                self.capacity,
                n_offered,
            )
            positions = positions[-self.capacity :]
            planes_arr = planes_arr[-self.capacity :]
            visits_arr = visits_arr[-self.capacity :]
            outcomes_arr = outcomes_arr[-self.capacity :]
            full_search_arr = full_search_arr[-self.capacity :]
            versions_arr = versions_arr[-self.capacity :]
            game_ids_arr = game_ids_arr[-self.capacity :]
        n = len(positions)

        with self._lock:
            # Handle potential ring buffer wrap-around
            start = self.write_head % self.capacity
            end = start + n
            if end <= self.capacity:
                self.planes[start:end] = planes_arr
                self.visits[start:end] = visits_arr
                self.outcomes[start:end] = outcomes_arr
                self.full_search[start:end] = full_search_arr
                self.versions[start:end] = versions_arr
                self.game_ids[start:end] = game_ids_arr
            else:
                # Split into two slices: head to end, then 0 to remainder
                part1 = self.capacity - start
                part2 = n - part1

                self.planes[start : self.capacity] = planes_arr[:part1]
                self.visits[start : self.capacity] = visits_arr[:part1]
                self.outcomes[start : self.capacity] = outcomes_arr[:part1]
                self.full_search[start : self.capacity] = full_search_arr[:part1]
                self.versions[start : self.capacity] = versions_arr[:part1]
                self.game_ids[start : self.capacity] = game_ids_arr[:part1]

                self.planes[:part2] = planes_arr[part1:]
                self.visits[:part2] = visits_arr[part1:]
                self.outcomes[:part2] = outcomes_arr[part1:]
                self.full_search[:part2] = full_search_arr[part1:]
                self.versions[:part2] = versions_arr[part1:]
                self.game_ids[:part2] = game_ids_arr[part1:]

            self.write_head = (start + n) % self.capacity
            self.size = min(self.capacity, self.size + n)
            # Honest accounting (SR-11): records truncated away by an oversized
            # flush were still produced by self-play and must be counted.
            self.total_added += n_offered
        self._save_meta(force=True)

        if self.gpu_cache is not None:
            self.gpu_cache.add_positions(
                planes=planes_arr,
                visits=visits_arr,
                outcomes=outcomes_arr,
                full_search=full_search_arr,
                versions=versions_arr,
                game_ids=game_ids_arr,
            )

    def get_target_human_mix(self, step: int) -> float:
        """Compute human data mix fraction at current training step via taper schedule."""
        if self.human_dataset is None:
            return 0.0
        if self.taper_steps <= 0:
            return self.final_human_mix

        progress = min(1.0, max(0.0, step / float(self.taper_steps)))
        mix = self.initial_human_mix - progress * (self.initial_human_mix - self.final_human_mix)
        return float(mix)

    def _sample_ring_indices_locked(self, n: int) -> np.ndarray:
        """Uniform-with-replacement ring indices under the unified staleness contract.

        Caller must hold ``self._lock`` (indices and payload are then read from
        one consistent snapshot). Single eligibility rule shared with the GPU
        cache (audit SR-05): a row is eligible while its model version is no
        more than ``STALENESS_WINDOW`` pushes behind the freshest row in the
        live ring, and unknown-provenance rows (version < 0) are never
        eligible. Rows are drawn uniformly with replacement (identical
        replacement semantics to the cache path). If fewer than ``n`` eligible
        rows exist the request falls back to sampling every live row —
        documented, counted in ``staleness_fallback_count``, and rate-limit
        warned — instead of returning fewer rows or raising.
        """
        if self.size <= 0:
            raise ValueError("Cannot sample self-play batch: replay buffer is empty.")
        versions_slice = np.array(self.versions[: self.size], dtype=np.int32)
        if versions_slice.size == 0:
            raise ValueError("Cannot sample self-play batch: replay buffer is empty.")
        max_ver = int(versions_slice.max())
        eligible = np.nonzero(
            (versions_slice >= (max_ver - STALENESS_WINDOW)) & (versions_slice >= 0)
        )[0]
        if eligible.size >= n:
            return eligible[np.random.randint(0, eligible.size, size=n)]
        if eligible.size < versions_slice.size:
            self.staleness_fallback_count += 1
            now = time.monotonic()
            if now - getattr(self, "_last_staleness_warn", 0.0) >= 60.0:
                self._last_staleness_warn = now
                logger.warning(
                    "[BUFFER] Staleness fallback: only %d of %d live rows are within the "
                    "%d-version window; sampling all rows for this request",
                    int(eligible.size),
                    int(versions_slice.size),
                    STALENESS_WINDOW,
                )
        return np.random.randint(0, self.size, size=n)

    def _selfplay_source_split(self, num_selfplay: int) -> Tuple[int, int, float]:
        """Split a self-play request between GPU cache and ring (audit SR-06).

        Returns ``(cache_rows, ring_rows, coverage)`` where ``coverage`` is the
        cache's share of the live ring. With the default
        ``selfplay_sampling_policy="ring_uniform"`` the request is split in
        proportion to coverage, which makes the combined distribution uniform
        over the whole ring instead of uniform over only the newest
        ``cache_capacity`` rows. ``"cache_uniform"`` restores the historical
        behavior (every row from the cache whenever it is large enough).
        """
        if self.gpu_cache is None or len(self.gpu_cache) == 0 or self.size <= 0:
            return 0, num_selfplay, 0.0
        coverage = min(1.0, float(len(self.gpu_cache)) / float(self.size))
        if self.selfplay_sampling_policy == "cache_uniform":
            if len(self.gpu_cache) >= num_selfplay:
                return num_selfplay, 0, coverage
            return 0, num_selfplay, coverage
        cache_rows = int(round(num_selfplay * coverage))
        cache_rows = max(0, min(num_selfplay, cache_rows))
        return cache_rows, num_selfplay - cache_rows, coverage

    def sample_batch(
        self,
        batch_size: int,
        step: int = 0,
        return_tactical_mix: bool = False,
    ) -> Tuple[Any, ...]:
        """Sample a training batch mixing self-play positions, human data, and tactical anchor positions.

        Args:
            batch_size: Total batch size.
            step: Current RL training step (controls human taper schedule).
            return_tactical_mix: If True, returns (batch, actual_human_mix, actual_tactical_mix).

        Returns:
            Tuple of:
                batch: Dictionary containing planes, visits, outcomes, is_full_search.
                actual_human_mix: float ratio of human positions measured on the non-tactical portion.
                (optional) actual_tactical_mix: float ratio of tactical positions in the total batch.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        # Audit RL-13: tactical demand must come from an actually-loaded
        # tactical cache, never from a restored meta counter alone. A stale
        # ``tactical_pool_size`` with no pool/cache used to under-fill the
        # batch and crash the shuffle permutation with an IndexError; the
        # saved value remains metadata only.
        tactical_available = getattr(self, "_tactical_cache_size", 0) > 0
        if tactical_available and self.tactical_mix > 0.0:
            num_tactical = int(round(batch_size * self.tactical_mix))
        else:
            num_tactical = 0

        remaining_size = batch_size - num_tactical
        target_mix = self.get_target_human_mix(step)

        # Determine counts for non-tactical portion
        if self.human_dataset is not None:
            if self.size == 0:
                # Buffer empty: 100% human data for non-tactical portion
                num_human = remaining_size
                num_selfplay = 0
            else:
                num_human = int(round(remaining_size * target_mix))
                num_selfplay = remaining_size - num_human
        else:
            num_human = 0
            num_selfplay = remaining_size

        if num_selfplay > 0 and self.size == 0 and num_human == 0 and num_tactical > 0:
            num_tactical = batch_size
            num_selfplay = 0
            remaining_size = 0

        batch_planes = []
        batch_visits = []
        batch_outcomes = []
        batch_full_search = []
        batch_game_ids = []

        # 1. Sample from self-play ring buffer (under lock: writers mutate
        # these arrays concurrently; copy indices + payload atomically). The
        # staleness rule is the shared unified contract (audit SR-05): this
        # path samples uniformly over the ring — the same eligible population
        # the corrected cache/ring split realizes on the tensor path.
        if num_selfplay > 0 and self.size > 0:
            with self._lock:
                sp_indices = self._sample_ring_indices_locked(num_selfplay)
                sp_planes = np.array(self.planes[sp_indices], dtype=np.float32)
                sp_visits = np.array(self.visits[sp_indices], dtype=np.float32)
                sp_outcomes = np.array(self.outcomes[sp_indices], dtype=np.int64)
                sp_full = np.array(self.full_search[sp_indices], dtype=np.bool_)
                sp_game_ids = np.array(self.game_ids[sp_indices], dtype=np.int64)
            self.last_selfplay_source = "ring"
            self.last_selfplay_rows = num_selfplay
            batch_planes.append(sp_planes)
            batch_visits.append(sp_visits)
            batch_outcomes.append(sp_outcomes)
            batch_full_search.append(sp_full)
            batch_game_ids.append(sp_game_ids)

        # 2. Sample from human dataset
        if num_human > 0 and self.human_dataset is not None:
            if getattr(self, "_human_cache_size", 0) > 0 and self._human_gpu_planes is not None:
                h_dev = self._human_gpu_planes.device
                h_indices = torch.randint(0, self._human_cache_size, (num_human,), device=h_dev)
                h_planes = self._human_gpu_planes[h_indices].cpu().numpy()
                h_moves = self._human_gpu_moves[h_indices].cpu().numpy()
                h_outcomes = self._human_gpu_outcomes[h_indices].cpu().numpy()
            elif getattr(self, "_human_cache_size", 0) > 0 and self._human_cache_planes is not None:
                h_indices = np.random.randint(0, self._human_cache_size, size=num_human)
                h_planes = self._human_cache_planes[h_indices]
                h_moves = self._human_cache_moves[h_indices]
                h_outcomes = self._human_cache_outcomes[h_indices]
            else:
                human_len = len(self.human_dataset)
                h_indices = np.random.randint(0, human_len, size=num_human)
                h_planes = np.array(self.human_dataset.planes[h_indices], dtype=np.float32)
                h_moves = np.array(self.human_dataset.moves[h_indices], dtype=np.int64)
                h_outcomes = np.array(self.human_dataset.outcomes[h_indices], dtype=np.int64)

            # Create one-hot visit distribution for human moves (vectorized)
            h_visits = np.zeros((num_human, self.num_actions), dtype=np.float32)
            _valid = (h_moves >= 0) & (h_moves < self.num_actions)
            _rows = np.arange(num_human)[_valid]
            h_visits[_rows, h_moves[_valid]] = 1.0

            # Human moves are treated as full-search policy targets
            h_full_search = np.ones(num_human, dtype=np.bool_)

            batch_planes.append(h_planes)
            batch_visits.append(h_visits)
            batch_outcomes.append(h_outcomes)
            batch_full_search.append(h_full_search)
            batch_game_ids.append(np.full((num_human,), NO_GAME_ID, dtype=np.int64))

        # 3. Sample from tactical pool
        if num_tactical > 0 and getattr(self, "_tactical_cache_size", 0) > 0:
            if self._tactical_gpu_planes is not None:
                t_dev = self._tactical_gpu_planes.device
                t_indices = torch.randint(
                    0, self._tactical_cache_size, (num_tactical,), device=t_dev
                )
                t_planes = self._tactical_gpu_planes[t_indices].cpu().numpy()
                t_moves = self._tactical_gpu_moves[t_indices].cpu().numpy()
                t_outcomes = self._tactical_gpu_outcomes[t_indices].cpu().numpy()
            elif self._tactical_cache_planes is not None:
                t_indices = np.random.randint(0, self._tactical_cache_size, size=num_tactical)
                t_planes = self._tactical_cache_planes[t_indices]
                t_moves = self._tactical_cache_moves[t_indices]
                t_outcomes = self._tactical_cache_outcomes[t_indices]
            else:
                t_planes = np.zeros(
                    (0, self.in_channels, self.height, self.width), dtype=np.float32
                )
                t_moves = np.zeros((0,), dtype=np.int64)
                t_outcomes = np.zeros((0,), dtype=np.int64)

            t_visits = np.zeros((num_tactical, self.num_actions), dtype=np.float32)
            _valid_t = (t_moves >= 0) & (t_moves < self.num_actions)
            _rows_t = np.arange(num_tactical)[_valid_t]
            t_visits[_rows_t, t_moves[_valid_t]] = 1.0

            t_full_search = np.ones(num_tactical, dtype=np.bool_)

            batch_planes.append(t_planes)
            batch_visits.append(t_visits)
            batch_outcomes.append(t_outcomes)
            batch_full_search.append(t_full_search)
            batch_game_ids.append(np.full((num_tactical,), NO_GAME_ID, dtype=np.int64))

        if not batch_planes:
            raise ValueError(
                "Cannot sample batch: buffer, human dataset, and tactical pool are all empty."
            )

        actual_human = num_human
        actual_human_mix = (
            float(actual_human) / float(remaining_size) if remaining_size > 0 else 0.0
        )
        actual_tactical_mix = float(num_tactical) / float(batch_size)
        self.last_human_mix = actual_human_mix
        self.last_tactical_mix = actual_tactical_mix

        planes = np.concatenate(batch_planes, axis=0) if len(batch_planes) > 1 else batch_planes[0]
        visits = np.concatenate(batch_visits, axis=0) if len(batch_visits) > 1 else batch_visits[0]
        outcomes = (
            np.concatenate(batch_outcomes, axis=0) if len(batch_outcomes) > 1 else batch_outcomes[0]
        )
        full_search = (
            np.concatenate(batch_full_search, axis=0)
            if len(batch_full_search) > 1
            else batch_full_search[0]
        )
        game_ids = (
            np.concatenate(batch_game_ids, axis=0) if len(batch_game_ids) > 1 else batch_game_ids[0]
        )

        # Shuffle mini-batch to interleave human, tactical, and self-play samples
        # (audit RL-40: over the ASSEMBLED length — never index past it).
        perm = np.random.permutation(len(planes))
        batch = {
            "planes": planes[perm],
            "visits": visits[perm],
            "outcomes": outcomes[perm],
            "is_full_search": full_search[perm],
            "game_ids": game_ids[perm],
        }

        if return_tactical_mix:
            return batch, actual_human_mix, actual_tactical_mix
        return batch, actual_human_mix

    def sample_batch_tensors(
        self,
        batch_size: int,
        step: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        return_tactical_mix: bool = False,
    ) -> Tuple[Any, ...]:
        """Sample mini-batch directly as PyTorch tensors on target device.

        Leverages Tier 1 GPU cache for zero-PCIe sampling.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        dev = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Audit RL-13: tactical demand must come from an actually-loaded
        # tactical cache, never from a restored meta counter alone. A stale
        # ``tactical_pool_size`` with no pool/cache used to under-fill the
        # batch and crash the shuffle permutation with an IndexError; the
        # saved value remains metadata only.
        tactical_available = getattr(self, "_tactical_cache_size", 0) > 0
        if tactical_available and self.tactical_mix > 0.0:
            num_tactical = int(round(batch_size * self.tactical_mix))
        else:
            num_tactical = 0

        remaining_size = batch_size - num_tactical
        target_mix = self.get_target_human_mix(step)

        # Determine counts for non-tactical portion
        if self.human_dataset is not None:
            if self.size == 0 and (self.gpu_cache is None or len(self.gpu_cache) == 0):
                num_human = remaining_size
                num_selfplay = 0
            else:
                num_human = int(round(remaining_size * target_mix))
                num_selfplay = remaining_size - num_human
        else:
            num_human = 0
            num_selfplay = remaining_size

        if (
            num_selfplay > 0
            and self.size == 0
            and (self.gpu_cache is None or len(self.gpu_cache) == 0)
            and num_human == 0
            and num_tactical > 0
        ):
            num_tactical = batch_size
            num_selfplay = 0
            remaining_size = 0

        batch_planes = []
        batch_visits = []
        batch_outcomes = []
        batch_full_search = []
        # Producing-game identity travels with every sampled row (audit
        # SR-13): self-play rows carry their real game id (> 0); human and
        # tactical rows carry NO_GAME_ID (0).
        batch_game_ids: List[torch.Tensor] = []

        # 1. Sample from self-play (GPU cache + ring, split per the documented
        #    coverage policy — audit SR-05/SR-06)
        if num_selfplay > 0:
            cache_rows, ring_rows, coverage = self._selfplay_source_split(num_selfplay)

            def _to_dev(t: torch.Tensor) -> torch.Tensor:
                return t if t.device == dev else t.to(dev, non_blocking=True)

            if cache_rows > 0 and self.gpu_cache is not None and self.gpu_cache.device == dev:
                sp_batch = self.gpu_cache.sample_batch(cache_rows)
                batch_planes.append(sp_batch["planes"])
                batch_visits.append(sp_batch["visits"])
                batch_outcomes.append(sp_batch["outcomes"])
                batch_full_search.append(sp_batch["is_full_search"])
                batch_game_ids.append(sp_batch["game_ids"])
            elif cache_rows > 0:
                # Cache exists but on another device: the ring path serves the
                # whole request (uniform over the ring, still unbiased).
                ring_rows += cache_rows
                cache_rows = 0

            if ring_rows > 0:
                if self.size <= 0:
                    if cache_rows > 0:
                        # Cache-only state (e.g. restored rows): serve the rest
                        # from the cache as well.
                        sp_batch = self.gpu_cache.sample_batch(ring_rows)
                        batch_planes.append(sp_batch["planes"])
                        batch_visits.append(sp_batch["visits"])
                        batch_outcomes.append(sp_batch["outcomes"])
                        batch_full_search.append(sp_batch["is_full_search"])
                        batch_game_ids.append(sp_batch["game_ids"])
                        ring_rows = 0
                        self.last_selfplay_source = "gpu_cache"
                        self.last_selfplay_rows = num_selfplay
                    else:
                        raise ValueError("Cannot sample self-play batch: replay buffer is empty.")
                else:
                    with self._lock:
                        sp_indices = self._sample_ring_indices_locked(ring_rows)
                        sp_planes = np.array(self.planes[sp_indices], dtype=np.float32)
                        sp_visits = np.array(self.visits[sp_indices], dtype=np.float32)
                        sp_outcomes = np.array(self.outcomes[sp_indices], dtype=np.int64)
                        sp_full = np.array(self.full_search[sp_indices], dtype=np.bool_)
                        sp_game_ids = np.array(self.game_ids[sp_indices], dtype=np.int64)

                    batch_planes.append(_to_dev(torch.from_numpy(sp_planes)))
                    batch_visits.append(_to_dev(torch.from_numpy(sp_visits)))
                    batch_outcomes.append(_to_dev(torch.from_numpy(sp_outcomes)))
                    batch_full_search.append(_to_dev(torch.from_numpy(sp_full)))
                    batch_game_ids.append(_to_dev(torch.from_numpy(sp_game_ids)))

            if ring_rows == 0 and cache_rows == 0:
                self.last_selfplay_source = "gpu_cache"
                self.last_selfplay_rows = num_selfplay
            elif cache_rows == 0:
                self.last_selfplay_source = "ring"
                self.last_selfplay_rows = num_selfplay
            elif ring_rows == 0:
                self.last_selfplay_source = "gpu_cache"
                self.last_selfplay_rows = num_selfplay
            else:
                self.last_selfplay_source = "ring+gpu_cache"
                self.last_selfplay_rows = num_selfplay

        # 2. Sample from human dataset
        if num_human > 0 and self.human_dataset is not None:
            if (
                getattr(self, "_human_cache_size", 0) > 0
                and self._human_gpu_planes is not None
                and self._human_gpu_planes.device == dev
            ):
                h_indices = torch.randint(0, self._human_cache_size, (num_human,), device=dev)
                h_planes = self._human_gpu_planes[h_indices]
                h_moves = self._human_gpu_moves[h_indices]
                h_outcomes = self._human_gpu_outcomes[h_indices]

                h_visits = torch.zeros(
                    (num_human, self.num_actions), dtype=torch.float32, device=dev
                )
                valid_h = (h_moves >= 0) & (h_moves < self.num_actions)
                if valid_h.any():
                    valid_rows = torch.nonzero(valid_h, as_tuple=True)[0]
                    h_visits[valid_rows, h_moves[valid_h]] = 1.0

                h_full_search = torch.ones(num_human, dtype=torch.bool, device=dev)

                batch_planes.append(h_planes)
                batch_visits.append(h_visits)
                batch_outcomes.append(h_outcomes)
                batch_full_search.append(h_full_search)
                batch_game_ids.append(
                    torch.full((num_human,), NO_GAME_ID, dtype=torch.int64, device=dev)
                )
            else:
                if (
                    getattr(self, "_human_cache_size", 0) > 0
                    and self._human_cache_planes is not None
                ):
                    h_indices = np.random.randint(0, self._human_cache_size, size=num_human)
                    h_planes_np = self._human_cache_planes[h_indices]
                    h_moves_np = self._human_cache_moves[h_indices]
                    h_outcomes_np = self._human_cache_outcomes[h_indices]
                elif (
                    getattr(self, "_human_cache_size", 0) > 0 and self._human_gpu_planes is not None
                ):
                    h_indices = torch.randint(
                        0,
                        self._human_cache_size,
                        (num_human,),
                        device=self._human_gpu_planes.device,
                    )
                    h_planes_np = self._human_gpu_planes[h_indices].cpu().numpy()
                    h_moves_np = self._human_gpu_moves[h_indices].cpu().numpy()
                    h_outcomes_np = self._human_gpu_outcomes[h_indices].cpu().numpy()
                else:
                    # Audit RL-10: draw the human contribution uniformly from
                    # the whole dataset, exactly like every other human path
                    # (GPU cache, CPU cache, numpy sampler). Outcome-255 rows
                    # are consumed as policy-only supervision and masked out
                    # of the WDL loss by ignore_index=255; filtering them on
                    # this path alone made the disk fallback disagree with
                    # the cache paths, and its empty/partial-result branch
                    # could shrink or crash the batch (audit RL-40). Value
                    # supervision accounting is reported by the trainer from
                    # the batch's outcome/game-id provenance instead.
                    human_len = len(self.human_dataset)
                    h_indices = np.random.randint(0, human_len, size=num_human)
                    h_planes_np = np.array(
                        self.human_dataset.planes[h_indices], dtype=np.float32
                    )
                    h_moves_np = np.array(self.human_dataset.moves[h_indices], dtype=np.int64)
                    h_outcomes_np = np.array(
                        self.human_dataset.outcomes[h_indices], dtype=np.int64
                    )

                h_visits_np = np.zeros((num_human, self.num_actions), dtype=np.float32)
                _valid_h = (h_moves_np >= 0) & (h_moves_np < self.num_actions)
                _rows_h = np.arange(num_human)[_valid_h]
                h_visits_np[_rows_h, h_moves_np[_valid_h]] = 1.0

                h_full_search_np = np.ones(num_human, dtype=np.bool_)

                def _to_dev2(t: torch.Tensor) -> torch.Tensor:
                    return t if t.device == dev else t.to(dev, non_blocking=True)

                batch_planes.append(_to_dev2(torch.from_numpy(h_planes_np)))
                batch_visits.append(_to_dev2(torch.from_numpy(h_visits_np)))
                batch_outcomes.append(_to_dev2(torch.from_numpy(h_outcomes_np)))
                batch_full_search.append(_to_dev2(torch.from_numpy(h_full_search_np)))
                batch_game_ids.append(
                    torch.full((int(h_full_search_np.shape[0]),), NO_GAME_ID, dtype=torch.int64).to(
                        dev, non_blocking=True
                    )
                )

        # 3. Sample from tactical pool
        if num_tactical > 0 and getattr(self, "_tactical_cache_size", 0) > 0:
            if self._tactical_gpu_planes is not None and self._tactical_gpu_planes.device == dev:
                t_indices = torch.randint(0, self._tactical_cache_size, (num_tactical,), device=dev)
                t_planes = self._tactical_gpu_planes[t_indices]
                t_moves = self._tactical_gpu_moves[t_indices]
                t_outcomes = self._tactical_gpu_outcomes[t_indices]

                t_visits = torch.zeros(
                    (num_tactical, self.num_actions), dtype=torch.float32, device=dev
                )
                valid_t = (t_moves >= 0) & (t_moves < self.num_actions)
                if valid_t.any():
                    valid_rows = torch.nonzero(valid_t, as_tuple=True)[0]
                    t_visits[valid_rows, t_moves[valid_t]] = 1.0

                t_full_search = torch.ones(num_tactical, dtype=torch.bool, device=dev)

                batch_planes.append(t_planes)
                batch_visits.append(t_visits)
                batch_outcomes.append(t_outcomes)
                batch_full_search.append(t_full_search)
                batch_game_ids.append(
                    torch.full((num_tactical,), NO_GAME_ID, dtype=torch.int64, device=dev)
                )
            else:
                if self._tactical_cache_planes is not None:
                    t_indices = np.random.randint(0, self._tactical_cache_size, size=num_tactical)
                    t_planes_np = self._tactical_cache_planes[t_indices]
                    t_moves_np = self._tactical_cache_moves[t_indices]
                    t_outcomes_np = self._tactical_cache_outcomes[t_indices]
                elif self._tactical_gpu_planes is not None:
                    t_indices = torch.randint(
                        0,
                        self._tactical_cache_size,
                        (num_tactical,),
                        device=self._tactical_gpu_planes.device,
                    )
                    t_planes_np = self._tactical_gpu_planes[t_indices].cpu().numpy()
                    t_moves_np = self._tactical_gpu_moves[t_indices].cpu().numpy()
                    t_outcomes_np = self._tactical_gpu_outcomes[t_indices].cpu().numpy()
                else:
                    t_planes_np = np.zeros(
                        (0, self.in_channels, self.height, self.width), dtype=np.float32
                    )
                    t_moves_np = np.zeros((0,), dtype=np.int64)
                    t_outcomes_np = np.zeros((0,), dtype=np.int64)

                t_visits_np = np.zeros((num_tactical, self.num_actions), dtype=np.float32)
                _valid_t = (t_moves_np >= 0) & (t_moves_np < self.num_actions)
                _rows_t = np.arange(num_tactical)[_valid_t]
                t_visits_np[_rows_t, t_moves_np[_valid_t]] = 1.0

                t_full_search_np = np.ones(num_tactical, dtype=np.bool_)

                def _to_dev_t(t: torch.Tensor) -> torch.Tensor:
                    return t if t.device == dev else t.to(dev, non_blocking=True)

                batch_planes.append(_to_dev_t(torch.from_numpy(t_planes_np)))
                batch_visits.append(_to_dev_t(torch.from_numpy(t_visits_np)))
                batch_outcomes.append(_to_dev_t(torch.from_numpy(t_outcomes_np)))
                batch_full_search.append(_to_dev_t(torch.from_numpy(t_full_search_np)))
                batch_game_ids.append(
                    torch.full((int(t_full_search_np.shape[0]),), NO_GAME_ID, dtype=torch.int64).to(
                        dev, non_blocking=True
                    )
                )

        if not batch_planes:
            raise ValueError(
                "Cannot sample batch: buffer, human dataset, and tactical pool are all empty."
            )

        actual_human = num_human
        actual_human_mix = (
            float(actual_human) / float(remaining_size) if remaining_size > 0 else 0.0
        )
        actual_tactical_mix = float(num_tactical) / float(batch_size)
        self.last_human_mix = actual_human_mix
        self.last_tactical_mix = actual_tactical_mix

        planes = torch.cat(batch_planes, dim=0) if len(batch_planes) > 1 else batch_planes[0]
        visits = torch.cat(batch_visits, dim=0) if len(batch_visits) > 1 else batch_visits[0]
        outcomes = (
            torch.cat(batch_outcomes, dim=0) if len(batch_outcomes) > 1 else batch_outcomes[0]
        )
        full_search = (
            torch.cat(batch_full_search, dim=0)
            if len(batch_full_search) > 1
            else batch_full_search[0]
        )
        game_ids = (
            torch.cat(batch_game_ids, dim=0) if len(batch_game_ids) > 1 else batch_game_ids[0]
        )

        if len(batch_planes) > 1:
            # Audit RL-40: shuffle over the ASSEMBLED batch length. Every
            # requested source now delivers exactly its requested rows, but
            # the permutation must never index past the assembled batch even
            # if a source under-delivers (the historical stale-tactical crash).
            perm = torch.randperm(planes.shape[0], device=dev)
            planes = planes[perm]
            visits = visits[perm]
            outcomes = outcomes[perm]
            full_search = full_search[perm]
            game_ids = game_ids[perm]

        batch = {
            "planes": planes,
            "visits": visits,
            "outcomes": outcomes,
            "is_full_search": full_search,
            "game_ids": game_ids,
        }
        if return_tactical_mix:
            return batch, actual_human_mix, actual_tactical_mix
        return batch, actual_human_mix

    def close(self) -> None:
        self._save_meta(force=True)
        for name in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
            arr = getattr(self, name, None)
            if isinstance(arr, np.memmap):
                try:
                    arr.flush()
                except (ValueError, OSError):
                    logger.debug("flush failed for %s", name, exc_info=True)
                try:
                    arr._mmap.close()  # type: ignore[attr-defined]
                except (AttributeError, ValueError, OSError):
                    logger.debug("memmap close failed for %s", name, exc_info=True)
        if self.human_dataset is not None:
            close_fn = getattr(self.human_dataset, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except (ValueError, OSError):
                    logger.debug("human dataset close failed", exc_info=True)
        if self.gpu_cache is not None:
            try:
                self.gpu_cache.close()
            except (RuntimeError, ValueError):
                logger.debug("gpu cache close failed", exc_info=True)
            self.gpu_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self._human_gpu_planes is not None:
            self._human_gpu_planes = None
            self._human_gpu_moves = None
            self._human_gpu_outcomes = None
            self._human_cache_size = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self._tactical_gpu_planes is not None:
            self._tactical_gpu_planes = None
            self._tactical_gpu_moves = None
            self._tactical_gpu_outcomes = None
            self._tactical_cache_size = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def destroy_on_disk(self) -> dict[str, int]:
        """Close memmaps then delete preallocated buffer files.

        For post-run cleanup when the buffer is no longer needed
        (final checkpoint already saved). Locked files count as
        failed, never raise. Returns {"deleted","failed","freed_bytes"}.
        """
        self.close()
        # Drop memmap references so Windows releases file locks.
        for name in ("planes", "visits", "outcomes", "full_search", "versions", "game_ids"):
            try:
                setattr(self, name, None)
            except (AttributeError, ValueError):
                logger.debug("destroy: could not drop %s", name, exc_info=True)
        from python.common.cleanup import cleanup_buffer

        return cleanup_buffer(self.buffer_dir)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> ReplayBuffer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self.size
