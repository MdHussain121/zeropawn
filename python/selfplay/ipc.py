"""Inter-Process Communication (IPC) via Shared Memory for Self-Play.

This module provides high-performance zero-copy shared memory tensor buffers
between the GPU inference server and multiple self-play worker processes on Windows/Linux.
"""

from __future__ import annotations

import atexit
import logging
from multiprocessing import shared_memory
import numpy as np

logger = logging.getLogger(__name__)


class SharedMemoryPool:
    """Manages shared memory segments for leaf state inputs and network evaluation outputs."""

    def __init__(
        self,
        name_prefix: str,
        capacity: int = 4096,
        in_channels: int = 19,
        height: int = 8,
        width: int = 8,
        num_actions: int = 4672,
        create: bool = False,
    ) -> None:
        self.name_prefix = name_prefix
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if in_channels <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"input dims must be positive, got {(in_channels, height, width)}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        self.capacity = capacity
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_actions = num_actions
        self.is_creator = create

        self.shm_names = {
            "inputs": f"{name_prefix}_inputs",
            "policy": f"{name_prefix}_policy",
            "wdl": f"{name_prefix}_wdl",
            "versions": f"{name_prefix}_versions",
        }

        self.shapes = {
            "inputs": (capacity, in_channels, height, width),
            "policy": (capacity, num_actions),
            "wdl": (capacity, 3),
            "versions": (capacity,),
        }

        self.dtypes = {
            "inputs": np.float32,
            "policy": np.float32,
            "wdl": np.float32,
            "versions": np.int32,
        }

        self.shm_blocks: dict[str, shared_memory.SharedMemory] = {}
        self.arrays: dict[str, np.ndarray] = {}

        try:
            for key, name in self.shm_names.items():
                shape = self.shapes[key]
                dtype = self.dtypes[key]
                size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

                if create:
                    # If existing segment with same name exists, unlink it first
                    try:
                        old_shm = shared_memory.SharedMemory(name=name)
                        old_shm.close()
                        try:
                            old_shm.unlink()
                        except (PermissionError, FileNotFoundError, OSError) as e:
                            logger.warning("Failed to unlink stale segment %s: %s", name, e)
                    except FileNotFoundError:
                        pass
                    shm = shared_memory.SharedMemory(name=name, create=True, size=size_bytes)
                else:
                    shm = shared_memory.SharedMemory(name=name, create=False)
                    if shm.size != size_bytes:
                        shm.close()
                        raise ValueError(
                            f"Shared segment {name} size {shm.size} != expected {size_bytes}"
                        )

                self.shm_blocks[key] = shm
                self.arrays[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

            if create:
                atexit.register(self.cleanup)

        except (OSError, ValueError, TypeError) as e:
            logger.error("Failed to initialize shared memory pool: %s", e)
            self.cleanup()
            raise

    @property
    def inputs(self) -> np.ndarray:
        return self.arrays["inputs"]

    @property
    def policy(self) -> np.ndarray:
        return self.arrays["policy"]

    @property
    def wdl(self) -> np.ndarray:
        return self.arrays["wdl"]

    @property
    def versions(self) -> np.ndarray:
        return self.arrays["versions"]

    def close(self) -> None:
        """Close shared memory handles."""
        for shm in self.shm_blocks.values():
            try:
                shm.close()
            except (OSError, ValueError) as e:
                logger.debug("shm close failed: %s", e)
        self.shm_blocks.clear()
        self.arrays.clear()

    def cleanup(self) -> None:
        """Close and unlink shared memory blocks if creator."""
        for shm in list(self.shm_blocks.values()):
            try:
                shm.close()
            except (OSError, ValueError) as e:
                logger.debug("shm close failed: %s", e)
            if self.is_creator:
                try:
                    shm.unlink()
                except (OSError, ValueError) as e:
                    logger.debug("shm unlink failed: %s", e)
        self.shm_blocks.clear()
        self.arrays.clear()

    def __enter__(self) -> "SharedMemoryPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.is_creator:
            self.cleanup()
        else:
            self.close()

    def __del__(self) -> None:
        try:
            if getattr(self, "is_creator", False):
                self.cleanup()
            else:
                self.close()
        except Exception:
            pass


class WorkerSlotAllocator:
    """Partitions the SharedMemoryPool slots among worker processes."""

    def __init__(self, num_workers: int, capacity: int = 4096) -> None:
        self.num_workers = num_workers
        self.capacity = capacity
        self.slots_per_worker = capacity // num_workers

    def get_worker_slots(self, worker_id: int) -> range:
        if not 0 <= worker_id < self.num_workers:
            raise ValueError(f"worker_id {worker_id} out of range [0, {self.num_workers})")
        start = worker_id * self.slots_per_worker
        end = start + self.slots_per_worker
        return range(start, end)
