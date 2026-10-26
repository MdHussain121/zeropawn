"""Memory-Mapped Replay Buffer with Human Data Mixing and Taper Schedule.

Contract Specifications:
- Ring buffer of 500k-1M positions, memory-mapped on disk.
- Uniform random sampling across active buffer positions.
- Configurable human-data mix fraction with linear taper schedule:
  starts at 25%, tapers to 5% over configurable step count.
- Logs and reports the actual mix ratio per mini-batch.
- Resumable: state (size, write_head, step) persists across restarts.
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


class GpuResidentReplayCache:
    """High-throughput circular GPU tensor cache in VRAM for zero-PCIe sampling."""

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
    ) -> None:
        """Add batch of positions directly to GPU cache."""
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
                else None
            )
        else:
            p_t = planes.to(dtype=torch.float32, device=self.device, non_blocking=True)
            v_t = visits.to(dtype=torch.float32, device=self.device, non_blocking=True)
            o_t = outcomes.to(dtype=torch.long, device=self.device, non_blocking=True)
            f_t = full_search.to(dtype=torch.bool, device=self.device, non_blocking=True)
            ver_t = (
                versions.to(dtype=torch.int32, device=self.device, non_blocking=True)
                if versions is not None
                else None
            )

        if n >= self.capacity:
            self.gpu_planes.copy_(p_t[-self.capacity :])
            self.gpu_visits.copy_(v_t[-self.capacity :])
            self.gpu_outcomes.copy_(o_t[-self.capacity :])
            self.gpu_full_search.copy_(f_t[-self.capacity :])
            if ver_t is not None:
                self.gpu_versions.copy_(ver_t[-self.capacity :])
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
                if ver_t is not None:
                    self.gpu_versions[start:end].copy_(ver_t)
            else:
                part1 = self.capacity - start
                part2 = n - part1
                self.gpu_planes[start : self.capacity].copy_(p_t[:part1])
                self.gpu_visits[start : self.capacity].copy_(v_t[:part1])
                self.gpu_outcomes[start : self.capacity].copy_(o_t[:part1])
                self.gpu_full_search[start : self.capacity].copy_(f_t[:part1])
                if ver_t is not None:
                    self.gpu_versions[start : self.capacity].copy_(ver_t[:part1])

                self.gpu_planes[:part2].copy_(p_t[part1:])
                self.gpu_visits[:part2].copy_(v_t[part1:])
                self.gpu_outcomes[:part2].copy_(o_t[part1:])
                self.gpu_full_search[:part2].copy_(f_t[part1:])
                if ver_t is not None:
                    self.gpu_versions[:part2].copy_(ver_t[part1:])

            self.write_head = (start + n) % self.capacity
            self.size = min(self.capacity, self.size + n)
            self.total_added += n

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample mini-batch directly on GPU with zero PCIe overhead."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        with self._lock:
            if self.size == 0:
                raise ValueError("GPU cache is empty.")
            # Staleness filter: exclude versions > 20 pushes behind current max
            cur_versions = self.gpu_versions[: self.size]
            max_ver = int(cur_versions.max().item()) if self.size > 0 else 0
            if max_ver > 20:
                valid_mask = cur_versions >= (max_ver - 20)
                valid_count = int(valid_mask.sum().item())
                if valid_count >= batch_size:
                    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                    rand_sub = torch.randint(
                        0, len(valid_indices), (batch_size,), device=self.device
                    )
                    indices = valid_indices[rand_sub]
                else:
                    indices = torch.randint(0, self.size, (batch_size,), device=self.device)
            else:
                indices = torch.randint(0, self.size, (batch_size,), device=self.device)
            out = {
                "planes": self.gpu_planes[indices].clone(),
                "visits": self.gpu_visits[indices].clone(),
                "outcomes": self.gpu_outcomes[indices].clone(),
                "is_full_search": self.gpu_full_search[indices].clone(),
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
    ) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
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
        self.gpu_device: Optional[torch.device] = (
            torch.device(gpu_device) if gpu_device is not None else None
        )

        self._write_head = 0
        self.size = 0
        self.total_added = 0
        self._lock = threading.Lock()
        self._last_meta_save = 0.0
        self._games_since_meta = 0
        self._meta_throttle_games = 10
        self._meta_throttle_secs = 30.0

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

        # Initialize or load metadata
        if not create_new and self.paths["meta"].exists():
            self._load_meta()
        else:
            self._create_empty_files()
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
        }
        meta_path = self.paths["meta"]
        tmp_path = meta_path.with_suffix(".tmp")
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
        self.write_head = write_head
        self.size = size
        self.total_added = total_added
        if "tactical_pool_size" in meta:
            self.saved_tactical_pool_size = int(meta["tactical_pool_size"])
            self.tactical_pool_size = self.saved_tactical_pool_size

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

        # Pre-populate GPU cache with recent positions from buffer if available
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
                )

        logger.info(f"Enabled GPU-resident replay cache with capacity {capacity} on {dev}")

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
            self.human_dataset = BinaryChessDataset(p)
            logger.info(f"Loaded human dataset with {len(self.human_dataset)} positions from {p}")
            self._load_human_cache(cache_size=cache_size, device=device)
        else:
            logger.warning(f"Human dataset not found at {p}. Proceeding without human mix.")

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
                indices = np.random.default_rng().choice(total, size=n, replace=False)
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
                indices = np.random.default_rng().choice(total, size=n, replace=False)
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
        """Insert a single position into the ring buffer."""
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
            )

    def add_positions(self, positions: List[SavedPosition]) -> None:
        """Insert a batch of positions into the ring buffer (tail-take if n > capacity)."""
        if not positions:
            return

        if len(positions) > self.capacity:
            # Keep only the newest `capacity` entries (consistent with GPU cache).
            positions = positions[-self.capacity :]
        n = len(positions)
        # Vectorized batch insertion
        planes_arr = np.stack([p.planes for p in positions], axis=0)
        visits_arr = np.stack([p.visits for p in positions], axis=0)
        outcomes_arr = np.array([p.outcome for p in positions], dtype=np.int64)
        full_search_arr = np.array([p.is_full_search for p in positions], dtype=np.bool_)
        versions_arr = np.array([p.model_version for p in positions], dtype=np.int32)
        game_ids_arr = np.array([p.game_id for p in positions], dtype=np.uint64)

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
            self.total_added += n
        self._save_meta(force=True)

        if self.gpu_cache is not None:
            self.gpu_cache.add_positions(
                planes=planes_arr,
                visits=visits_arr,
                outcomes=outcomes_arr,
                full_search=full_search_arr,
                versions=versions_arr,
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

        tactical_available = (
            getattr(self, "tactical_pool_size", 0) > 0
            or getattr(self, "_tactical_cache_size", 0) > 0
        )
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

        # 1. Sample from self-play ring buffer (under lock: writers mutate
        # these arrays concurrently; copy indices + payload atomically).
        if num_selfplay > 0 and self.size > 0:
            with self._lock:
                sp_indices = np.random.randint(0, self.size, size=num_selfplay)
                sp_planes = np.array(self.planes[sp_indices], dtype=np.float32)
                sp_visits = np.array(self.visits[sp_indices], dtype=np.float32)
                sp_outcomes = np.array(self.outcomes[sp_indices], dtype=np.int64)
                sp_full = np.array(self.full_search[sp_indices], dtype=np.bool_)
            batch_planes.append(sp_planes)
            batch_visits.append(sp_visits)
            batch_outcomes.append(sp_outcomes)
            batch_full_search.append(sp_full)

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

        # Shuffle mini-batch to interleave human, tactical, and self-play samples
        perm = np.random.permutation(batch_size)
        batch = {
            "planes": planes[perm],
            "visits": visits[perm],
            "outcomes": outcomes[perm],
            "is_full_search": full_search[perm],
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

        tactical_available = (
            getattr(self, "tactical_pool_size", 0) > 0
            or getattr(self, "_tactical_cache_size", 0) > 0
        )
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

        # 1. Sample from self-play (GPU cache fast-path or RAM/mmap buffer)
        if num_selfplay > 0:
            if (
                self.gpu_cache is not None
                and len(self.gpu_cache) >= num_selfplay
                and self.gpu_cache.device == dev
            ):
                sp_batch = self.gpu_cache.sample_batch(num_selfplay)
                batch_planes.append(sp_batch["planes"])
                batch_visits.append(sp_batch["visits"])
                batch_outcomes.append(sp_batch["outcomes"])
                batch_full_search.append(sp_batch["is_full_search"])
            elif self.size > 0:
                with self._lock:
                    versions_slice = np.array(self.versions, dtype=np.int32)[: self.capacity]
                    max_ver = int(versions_slice.max()) if len(versions_slice) > 0 else 0
                    if max_ver > 20:
                        valid_mask = versions_slice >= (max_ver - 20)
                        valid_count = int(valid_mask.sum())
                        if valid_count >= num_selfplay:
                            valid_indices = np.nonzero(valid_mask)[0]
                            sp_indices = np.random.choice(
                                valid_indices, size=num_selfplay, replace=False
                            )
                        else:
                            sp_indices = np.random.randint(0, self.size, size=num_selfplay)
                    else:
                        sp_indices = np.random.randint(0, self.size, size=num_selfplay)
                    sp_planes = np.array(self.planes[sp_indices], dtype=np.float32)
                    sp_visits = np.array(self.visits[sp_indices], dtype=np.float32)
                    sp_outcomes = np.array(self.outcomes[sp_indices], dtype=np.int64)
                    sp_full = np.array(self.full_search[sp_indices], dtype=np.bool_)

                def _to_dev(t: torch.Tensor) -> torch.Tensor:
                    return t if t.device == dev else t.to(dev, non_blocking=True)

                batch_planes.append(_to_dev(torch.from_numpy(sp_planes)))
                batch_visits.append(_to_dev(torch.from_numpy(sp_visits)))
                batch_outcomes.append(_to_dev(torch.from_numpy(sp_outcomes)))
                batch_full_search.append(_to_dev(torch.from_numpy(sp_full)))
            else:
                raise ValueError("Cannot sample self-play batch: replay buffer is empty.")

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
                    valid_mask = np.array(self.human_dataset.outcomes) != 255
                    valid_indices = np.nonzero(valid_mask)[0]
                    if len(valid_indices) == 0:
                        h_indices = np.array([], dtype=np.int64)
                        h_planes_np = np.zeros((0, 19, 8, 8), dtype=np.float32)
                        h_moves_np = np.zeros((0,), dtype=np.int64)
                        h_outcomes_np = np.zeros((0,), dtype=np.int64)
                        num_human = 0
                    else:
                        h_indices = np.random.choice(
                            valid_indices, size=min(num_human, len(valid_indices)), replace=False
                        )
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

        if len(batch_planes) > 1:
            perm = torch.randperm(batch_size, device=dev)
            planes = planes[perm]
            visits = visits[perm]
            outcomes = outcomes[perm]
            full_search = full_search[perm]

        batch = {
            "planes": planes,
            "visits": visits,
            "outcomes": outcomes,
            "is_full_search": full_search,
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
