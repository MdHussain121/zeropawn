"""Inference clients used by self-play workers to request neural evaluations.

Provides both SharedMemory-based IPC client and Direct (in-process) client.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Tuple
import numpy as np

from python.selfplay.ipc import SharedMemoryPool

logger = logging.getLogger(__name__)


class ShmInferenceClient:
    """Client for worker processes communicating with InferenceServer via SharedMemory."""

    def __init__(
        self,
        worker_id: int,
        shm_pool: SharedMemoryPool,
        request_queue: Any,
        completion_event: Any,
        slot_range: range,
    ) -> None:
        self.worker_id = worker_id
        self.shm_pool = shm_pool
        self.request_queue = request_queue
        self.completion_event = completion_event
        self.slot_range = slot_range
        self.slots = list(slot_range)
        self.req_counter = 0
        self._lock = threading.Lock()

    def evaluate_leaves(
        self,
        states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Submit leaf states to inference server and return (policy, wdl, model_version).

        Args:
            states: (N, C, H, W) numpy array of states.

        Returns:
            Tuple of:
                policy_logits: (N, num_actions)
                wdl_logits: (N, 3)
                model_version: int
        """
        num_states = states.shape[0]
        if num_states > len(self.slots):
            raise ValueError(
                f"Batch size {num_states} exceeds allocated worker slot capacity {len(self.slots)}"
            )

        assigned_slots = self.slots[:num_states]
        capacity = int(self.shm_pool.capacity)
        for s in assigned_slots:
            if not 0 <= int(s) < capacity:
                raise ValueError(f"Slot index {s} out of bounds [0, {capacity})")

        with self._lock:
            # 1. Zero-copy write states into worker's shared memory input slice
            self.shm_pool.inputs[assigned_slots] = states

            # 2. Put request to server
            self.req_counter += 1
            self.completion_event.clear()
            try:
                self.request_queue.put(
                    (self.worker_id, assigned_slots, self.req_counter), timeout=5
                )
            except queue.Full as e:
                raise TimeoutError("request_queue.put timed out after 5s") from e

            # 3. Wait for inference server completion event
            if not self.completion_event.wait(timeout=60):
                raise TimeoutError("Timed out waiting for inference completion")

            # 4. Read outputs from shared memory
            policy_logits = np.copy(self.shm_pool.policy[assigned_slots])
            wdl_logits = np.copy(self.shm_pool.wdl[assigned_slots])
            model_version = int(self.shm_pool.versions[assigned_slots[0]])

        return policy_logits, wdl_logits, model_version


class DirectInferenceClient:
    """In-process client directly wrapping an InferenceServer instance."""

    def __init__(self, server: Any) -> None:
        self.server = server

    def evaluate_leaves(
        self,
        states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Directly call server predict_batch."""
        return self.server.predict_batch(states)
