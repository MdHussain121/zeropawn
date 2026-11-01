"""High-Throughput GPU Inference Server for AlphaZero-style Self-Play.

Contract Specifications:
- Python, PyTorch, bfloat16 mixed precision.
- Channels-last memory format for Tensor Core optimization.
- Batches 256-512 leaves per GPU call with a short max-wait timeout.
- Dynamic checkpoint hot-reloading without server or worker restarts.
- Every response tagged with current monotonic model version.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from python.model.chess_net import ChessNet
from python.selfplay.ipc import SharedMemoryPool

logger = logging.getLogger(__name__)


class InferenceServer:
    """GPU-accelerated batched inference server for MCTS self-play."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        min_batch_size: int = 256,
        max_batch_size: int = 512,
        max_wait_ms: float = 3.0,
        use_bfloat16: bool = True,
        channels_last: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_ms / 1000.0
        self.use_bfloat16 = use_bfloat16 and self.device.type == "cuda"
        self.channels_last = channels_last
        self.model_version = 0

        # High-priority CUDA stream to avoid stalling self-play during background training
        if self.device.type == "cuda":
            try:
                self.stream = torch.cuda.Stream(device=self.device, priority=-1)
            except Exception:
                self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None

        # Hot-reload lock: load_checkpoint() swaps weights in place while
        # predict_batch() forwards may be running. Takers: predict grabs the
        # module reference under this lock and forwards outside it (lock-free
        # steady state); load holds it for the whole swap (predicts stall
        # only during a reload, ~1s for 8.6M params). No torn reads.
        # Created before any load_checkpoint() call in __init__ and it is the
        # ONLY lock instance (audit L-2: a second allocation used to replace
        # this one after construction; one lock, created before first use).
        self._weights_lock = threading.RLock()

        # Hot-reload outcome tracking (audit L-3): a failed candidate never
        # touches the active model, but the failure must be observable —
        # callers compare model_version and/or read these fields instead of
        # scraping logs.  ``active_checkpoint_path`` is the identity of the
        # weights currently being served (None = in-memory/unbacked).
        self.last_reload_error: Optional[str] = None
        self.reload_failure_count: int = 0
        self.active_checkpoint_path: Optional[str] = None

        # Load or create model
        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path:
            # An empty-string path means "no checkpoint requested" (same as
            # None); a REAL path that fails to load raises — see
            # _create_model_from_checkpoint's fail-closed contract.
            self.model = self._create_model_from_checkpoint(checkpoint_path).to(self.device)
            self.load_checkpoint(checkpoint_path)
        else:
            # Explicit no-weights construction: caller opted into an (untrained)
            # random-init ChessNet.
            self.model = ChessNet(num_blocks=12, num_filters=128).to(self.device)

        if self.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model.eval()

        # Telemetry metrics
        self.total_evaluations = 0
        self.total_batches = 0
        self.total_batch_size_sum = 0
        self.total_inference_time_sec = 0.0
        self._window_lock = threading.Lock()
        self._window_evaluations = 0
        self._window_batches = 0
        self._window_time_sec = 0.0

    def _create_model_from_checkpoint(self, path: str) -> nn.Module:
        """Inspect checkpoint and construct appropriate network.

        Fails closed (Stage A §5): an unreadable or structurally invalid
        checkpoint raises instead of silently constructing a random-init
        ChessNet — a silent fallback here previously let runs proceed with
        random weights while logging only an exception.
        """
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint '{path}' is unreadable or invalid: {e}. "
                "Refusing to construct a random-init model (fail-closed)."
            ) from e
        state_dict = (
            ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        )
        num_blocks = ckpt.get("num_blocks", 12) if isinstance(ckpt, dict) else 12
        num_filters = ckpt.get("num_filters", 128) if isinstance(ckpt, dict) else 128
        if isinstance(state_dict, dict):
            block_indices = [
                int(k.split(".")[1])
                for k in state_dict
                if k.startswith("blocks.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
            ]
            if block_indices:
                num_blocks = max(block_indices) + 1
            # Infer filters from weight shapes when the header lacks them
            # (bare {"model": ...} checkpoints): stem conv out-channels.
            stem = state_dict.get("stem_conv.weight", None)
            if stem is not None and hasattr(stem, "shape") and len(stem.shape) == 4:
                num_filters = int(stem.shape[0])
        return ChessNet(num_blocks=num_blocks, num_filters=num_filters)

    def _record_reload_failure(self, path: str, what: str, reason: str) -> int:
        """Record a rejected hot-reload candidate; return the unchanged version.

        The active model is untouched by construction (the candidate is only
        ever a fresh off-side module).  This helper makes the failure
        observable: an ERROR log naming the retained weights, a machine-
        readable ``last_reload_error``, and a failure counter.
        """
        self.last_reload_error = f"{path}: {what}: {reason}"
        self.reload_failure_count += 1
        logger.error(
            "HOT-RELOAD FAILED (%s): %s — the checkpoint was NOT activated; keeping "
            "active weights %s (model version %d). Reason: %s",
            what,
            path,
            self.active_checkpoint_path or "<in-memory>",
            self.model_version,
            reason,
        )
        return self.model_version

    def load_checkpoint(self, path: str, sync_event: Optional[Any] = None) -> int:
        """Hot-reload model weights without restarting server.

        Double-buffered swap: the new weights are loaded into a FRESH module
        off to the side, then published under the hot-reload lock by swapping
        the reference. In-flight forwards keep the old module alive by
        refcount and are never torn. Increments model_version monotonically
        and tags all future responses.

        Failure contract (audit L-3): a missing, unreadable, structurally
        invalid, or architecture-incompatible candidate never touches the
        active model.  The returned version then equals the pre-call version
        (success always increments) and ``last_reload_error`` /
        ``reload_failure_count`` record why.  ``_hot_reload_and_report`` is
        the loud wrapper used by the serving loop.
        """
        if sync_event is not None and self.stream is not None:
            try:
                self.stream.wait_event(sync_event)
            except Exception as e:
                logger.debug("wait_event failed on stream: %s", e)

        logger.info(f"Loading checkpoint from {path}...")
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            logger.exception("Failed to load checkpoint %s; keeping old weights", path)
            return self._record_reload_failure(path, "unreadable checkpoint", str(e))

        # Determine state dict key
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Build the replacement module off to the side (no lock needed).
        # Audit RL-39: the .to()/copy_ kernels below are enqueued on the
        # CALLER's current stream; record a completion event for them so the
        # serving stream can be ordered behind the copies before the new
        # weights are published (the pre-push sync_event only orders work
        # submitted before this method was entered).
        copy_done_event = None
        try:
            new_model = self._create_model_from_checkpoint(path).to(self.device)
            new_model.load_state_dict(state_dict)
            new_model.eval()
            if self.channels_last and self.device.type == "cuda":
                new_model = new_model.to(memory_format=torch.channels_last)
            if self.device.type == "cuda" and torch.cuda.is_available():
                copy_done_event = torch.cuda.Event()
                copy_done_event.record()
        except Exception as e:
            logger.exception("Invalid checkpoint %s; keeping old weights", path)
            return self._record_reload_failure(path, "invalid checkpoint", str(e))
        if copy_done_event is not None and self.stream is not None:
            try:
                self.stream.wait_event(copy_done_event)
            except Exception as e:
                logger.debug("wait_event failed on stream: %s", e)
        with self._weights_lock:
            self.model = new_model
            self.model_version += 1
            new_version = self.model_version
            self.active_checkpoint_path = str(path)
        self.last_reload_error = None
        logger.info(
            "Successfully reloaded checkpoint %s. New model version: %d", path, new_version
        )
        return new_version

    def load_state_dict_weights(
        self,
        state_dict: Dict[str, Any],
        sync_event: Optional[Any] = None,
    ) -> int:
        """Hot-reload model weights directly from state_dict without hitting disk.

        Safe across streams and devices: waits on sync_event if provided,
        constructs new model instance off to the side, transfers tensors to
        self.device, and publishes under the hot-reload lock.

        Same failure contract as :meth:`load_checkpoint` (audit L-3): on
        rejection the version is unchanged and ``last_reload_error`` records
        the reason; the caller (RLTrainer.push_weights) fail-closes on the
        version delta so trainer and server weights never diverge.
        """
        if sync_event is not None and self.stream is not None:
            try:
                self.stream.wait_event(sync_event)
            except Exception as e:
                logger.debug("wait_event failed on stream: %s", e)

        # Audit RL-39: the dev_state transfer, load_state_dict copy_ kernels
        # and the channels_last conversion above run on the caller's current
        # stream (the trainer's train stream during a weight push). The
        # pre-push sync_event only orders work submitted before this call, so
        # record a completion event for the copies themselves and make the
        # serving stream wait on it BEFORE the swap publishes new_model —
        # otherwise a concurrent forward pass could read parameters while
        # train-stream copy kernels are still writing them.
        copy_done_event = None
        try:
            cur_blocks = getattr(self.model, "num_blocks", 12)
            cur_filters = getattr(self.model, "num_filters", 128)
            new_model = ChessNet(num_blocks=cur_blocks, num_filters=cur_filters).to(self.device)
            dev_state = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in state_dict.items()
            }
            new_model.load_state_dict(dev_state)
            new_model.eval()
            if self.channels_last and self.device.type == "cuda":
                new_model = new_model.to(memory_format=torch.channels_last)
            if self.device.type == "cuda" and torch.cuda.is_available():
                copy_done_event = torch.cuda.Event()
                copy_done_event.record()
        except Exception as e:
            logger.exception("Failed to load state_dict weights; keeping old weights")
            return self._record_reload_failure(
                "<state_dict push>", "invalid state dict", str(e)
            )

        if copy_done_event is not None and self.stream is not None:
            try:
                self.stream.wait_event(copy_done_event)
            except Exception as e:
                logger.debug("wait_event failed on stream: %s", e)

        with self._weights_lock:
            self.model = new_model
            self.model_version += 1
            new_version = self.model_version
            self.active_checkpoint_path = "<state_dict push>"
        self.last_reload_error = None
        logger.info(f"Successfully reloaded weights from state_dict. New version: {new_version}")
        return new_version

    @property
    def reload_ok(self) -> bool:
        """True when the most recent reload attempt (if any) succeeded."""
        return self.last_reload_error is None

    @torch.no_grad()
    def predict_batch(
        self,
        states: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Perform batched forward inference on input states.

        Args:
            states: Input tensor/array of shape (B, C, H, W).

        Returns:
            Tuple of (policy_logits (B, num_actions), wdl_logits (B, 3), model_version).
        """
        stream_ctx = (
            torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
        )
        with stream_ctx:
            if isinstance(states, np.ndarray):
                states_t = torch.from_numpy(states).to(self.device, non_blocking=True)
            else:
                states_t = states.to(self.device, non_blocking=True)

            if self.channels_last and self.device.type == "cuda":
                states_t = states_t.to(memory_format=torch.channels_last)

            start_time = time.perf_counter()
            # Grab the module reference under the hot-reload lock; the forward
            # below then runs lock-free on a module no reload can mutate.
            with self._weights_lock:
                model = self.model
                current_version = self.model_version

            with torch.inference_mode():
                # Forward pass with bfloat16 mixed precision on CUDA
                if self.use_bfloat16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        p_logits, w_logits = model(states_t)
                else:
                    p_logits, w_logits = model(states_t)

                policy_np = p_logits.float().cpu().numpy()
                wdl_np = w_logits.float().cpu().numpy()

            elapsed = time.perf_counter() - start_time
            batch_size = states_t.shape[0]

            # Update telemetry
            self.total_evaluations += batch_size
            self.total_batches += 1
            self.total_batch_size_sum += batch_size
            self.total_inference_time_sec += elapsed
            with self._window_lock:
                self._window_evaluations += batch_size
                self._window_batches += 1
                self._window_time_sec += elapsed

        return policy_np, wdl_np, current_version

    @property
    def avg_batch_fill_rate(self) -> float:
        """Average batch fill relative to max_batch_size."""
        if self.total_batches == 0:
            return 0.0
        return (self.total_batch_size_sum / self.total_batches) / self.max_batch_size

    @property
    def current_fill_rate(self) -> float:
        """Non-destructive current batch fill rate from accumulated totals."""
        if self.total_batches == 0:
            return 0.0
        return self.avg_batch_fill_rate

    def get_window_stats(self) -> Tuple[float, float, float]:
        """Returns and resets (avg_batch_size, avg_latency_ms, batch_fill_rate) since last call."""
        with self._window_lock:
            batches = self._window_batches
            evals = self._window_evaluations
            time_sec = self._window_time_sec
            self._window_batches = 0
            self._window_evaluations = 0
            self._window_time_sec = 0.0
        if batches == 0:
            return 0.0, 0.0, 0.0
        avg_batch_size = evals / batches
        avg_latency_ms = (time_sec / batches) * 1000.0
        fill_rate = avg_batch_size / max(1, self.max_batch_size)
        return avg_batch_size, avg_latency_ms, fill_rate


def _hot_reload_and_report(server: InferenceServer, checkpoint_path: str) -> int:
    """Apply one hot-reload request and make non-activation loud (audit L-3).

    ``load_checkpoint`` keeps the active weights on failure; this wrapper
    turns the silent-stale outcome into an explicit, orchestrator-visible
    ERROR naming the retained weights and the recorded failure reason.
    Success is exactly "the returned version increased".
    """
    version_before = server.model_version
    new_version = server.load_checkpoint(checkpoint_path)
    if new_version == version_before:
        logger.error(
            "HOT-RELOAD REQUEST DID NOT ACTIVATE: %s (still model version %d, active "
            "weights %s). Continuing with the existing weights. Reason: %s",
            checkpoint_path,
            version_before,
            getattr(server, "active_checkpoint_path", None) or "<in-memory>",
            getattr(server, "last_reload_error", None) or "unknown",
        )
    return new_version


def run_inference_server_process(
    shm_pool_prefix: str,
    capacity: int,
    in_channels: int,
    height: int,
    width: int,
    num_actions: int,
    request_queue: Any,
    worker_events: Dict[int, Any],
    stop_event: Any,
    reload_queue: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    min_batch_size: int = 256,
    max_batch_size: int = 512,
    max_wait_ms: float = 3.0,
    device: Optional[str] = None,
) -> None:
    """LEGACY multiprocessing entry point for the Shared-Memory Inference Server.

    Status (audit SR-17): not production-reachable. The active runtime uses the
    in-process ``InferenceServer`` + ``DirectInferenceClient`` pair; nothing in
    ``scripts/`` or ``python/cli/`` starts this process, and its worker-side
    counterpart (``worker.run_worker_process``) is equally unused. Retained
    deliberately together with ``python/selfplay/ipc.py``; reviving the path
    requires the concurrency tests in
    ``tests/test_selfplay_replay_remediation.py`` to cover it first.
    """
    logger.info("Initializing SharedMemoryPool inside inference server process...")
    shm_pool = SharedMemoryPool(
        name_prefix=shm_pool_prefix,
        capacity=capacity,
        in_channels=in_channels,
        height=height,
        width=width,
        num_actions=num_actions,
        create=False,
    )

    server = InferenceServer(
        checkpoint_path=checkpoint_path,
        device=device,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )

    max_wait_sec = max_wait_ms / 1000.0

    try:
        pending_requests: List[Tuple[int, List[int], int]] = []  # (worker_id, slot_indices, req_id)
        pending_slots: List[int] = []
        first_arrival_time: Optional[float] = None

        while not stop_event.is_set():
            # Check for hot-reload requests (coalesce to latest path only)
            if reload_queue is not None:
                try:
                    latest_ckpt = None
                    while True:
                        new_ckpt = reload_queue.get_nowait()
                        if new_ckpt == "STOP":
                            return
                        latest_ckpt = new_ckpt
                except queue.Empty:
                    pass
                if latest_ckpt is not None:
                    try:
                        _hot_reload_and_report(server, latest_ckpt)
                    except (RuntimeError, OSError, KeyError, ValueError) as e:
                        logger.error("Failed to hot-reload checkpoint %s: %s", latest_ckpt, e)

            # Compute remaining wait timeout
            timeout = None
            if pending_slots:
                elapsed = time.monotonic() - first_arrival_time
                timeout = max(0.0, max_wait_sec - elapsed)
            else:
                timeout = 0.005  # Sleep briefly if nothing pending

            try:
                msg = request_queue.get(timeout=timeout)
                if msg is None or msg == "STOP":
                    break
                worker_id, slot_indices, req_id = msg
                if worker_id not in worker_events:
                    logger.warning("Unknown worker_id %s; ignoring request", worker_id)
                    continue
                slot_list = list(slot_indices)
                if any(not 0 <= int(s) < capacity for s in slot_list):
                    logger.warning("Slot indices out of bounds [0, %s); ignoring", capacity)
                    continue
                if not pending_slots:
                    first_arrival_time = time.monotonic()
                pending_requests.append((worker_id, slot_list, req_id))
                pending_slots.extend(slot_list)
                if len(pending_slots) > max_batch_size:
                    logger.warning(
                        "pending_slots %d exceeds max_batch_size %d; forcing flush",
                        len(pending_slots),
                        max_batch_size,
                    )
            except queue.Empty:
                pass

            # Determine whether to execute batch
            should_run = False
            if pending_slots:
                elapsed = time.monotonic() - first_arrival_time
                if len(pending_slots) >= min_batch_size:
                    should_run = True
                elif elapsed >= max_wait_sec:
                    should_run = True

            if should_run and pending_slots:
                # Slice up to max_batch_size slots (cap batch at max_batch_size)
                current_batch_slots = pending_slots[:max_batch_size]
                batch_size = len(current_batch_slots)

                # Extract input tensor slice from shared memory
                try:
                    batch_inputs = np.copy(shm_pool.inputs[current_batch_slots])
                    p_np, w_np, version = server.predict_batch(batch_inputs)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    logger.exception("CUDA OOM in predict_batch; emptying cache")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Error sentinel so workers do not hang
                    shm_pool.versions[current_batch_slots] = -1
                    for worker_id, _, _ in pending_requests:
                        if worker_id in worker_events:
                            worker_events[worker_id].set()
                    pending_requests = []
                    pending_slots = []
                    first_arrival_time = None
                    continue

                # Write results back to shared memory
                shm_pool.policy[current_batch_slots] = p_np
                shm_pool.wdl[current_batch_slots] = w_np
                shm_pool.versions[current_batch_slots] = version

                # Determine which worker requests are fulfilled
                completed_workers = set()
                remaining_requests = []
                remaining_slots = []
                slots_handled = 0

                for worker_id, req_slots, req_id in pending_requests:
                    # Check if all slots for this request were evaluated
                    if slots_handled + len(req_slots) <= batch_size:
                        completed_workers.add(worker_id)
                        slots_handled += len(req_slots)
                    else:
                        remaining_requests.append((worker_id, req_slots, req_id))
                        remaining_slots.extend(req_slots)

                pending_requests = remaining_requests
                pending_slots = remaining_slots
                first_arrival_time = time.monotonic() if pending_slots else None

                # Signal completed workers
                for worker_id in completed_workers:
                    worker_events[worker_id].set()

    finally:
        if shm_pool.is_creator:
            shm_pool.cleanup()
        else:
            shm_pool.close()
        logger.info("Inference server process terminated cleanly.")
