"""Tests for Task 1: Inference Server (Python, PyTorch, bfloat16, hot-reloading)."""

import os
import tempfile
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pytest
import torch

from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer, run_inference_server_process
from python.selfplay.ipc import SharedMemoryPool, WorkerSlotAllocator
from python.selfplay.client import DirectInferenceClient, ShmInferenceClient


def _get_test_ckpt(tmp_path: Path | None = None) -> str:
    default_p = Path("models/checkpoints_scaled/best_model.pt")
    if default_p.is_file():
        return str(default_p)
    tmp_dir = Path(tempfile.gettempdir()) / "zeropawn_test" if tmp_path is None else tmp_path
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dummy_p = tmp_dir / "dummy_test_best_model.pt"
    if not dummy_p.exists():
        model = ChessNet(num_blocks=12, num_filters=128)
        torch.save(
            {"model": model.state_dict(), "num_blocks": 12, "num_filters": 128, "step": 1}, dummy_p
        )
    return str(dummy_p)


def test_inference_server_bfloat16_shapes(tmp_path):
    """Verify inference server output shapes, bfloat16 autocast, and version tagging."""
    ckpt_path = _get_test_ckpt(tmp_path)
    server = InferenceServer(checkpoint_path=ckpt_path)

    batch_size = 256
    dummy_states = np.random.randn(batch_size, 19, 8, 8).astype(np.float32)

    policy_logits, wdl_logits, version = server.predict_batch(dummy_states)

    assert policy_logits.shape == (batch_size, 4672)
    assert wdl_logits.shape == (batch_size, 3)
    assert version == 1
    assert server.total_evaluations == batch_size
    assert server.total_batches == 1


def test_inference_server_checkpoint_hot_reload(tmp_path):
    """Verify that loading a new checkpoint updates weights and increments version without restart."""
    ckpt_path = _get_test_ckpt(tmp_path)
    server = InferenceServer(checkpoint_path=ckpt_path)
    initial_version = server.model_version

    # Run inference before reload
    dummy_states = np.random.randn(10, 19, 8, 8).astype(np.float32)
    p1, w1, v1 = server.predict_batch(dummy_states)
    assert v1 == initial_version

    # Create and save a new checkpoint with slightly perturbed weights
    new_ckpt_path = str(tmp_path / "updated_model.pt")
    new_state_dict = {k: v + 0.01 for k, v in server.model.state_dict().items()}
    torch.save({"model": new_state_dict, "num_blocks": 12, "num_filters": 128}, new_ckpt_path)

    # Hot-reload new checkpoint
    new_version = server.load_checkpoint(new_ckpt_path)
    assert new_version == initial_version + 1

    # Run inference after reload
    p2, w2, v2 = server.predict_batch(dummy_states)
    assert v2 == new_version
    # Outputs should change due to modified weights
    assert not np.allclose(p1, p2)


def test_direct_inference_client(tmp_path):
    """Verify DirectInferenceClient wrapper returns expected results."""
    ckpt_path = _get_test_ckpt(tmp_path)
    server = InferenceServer(checkpoint_path=ckpt_path)
    client = DirectInferenceClient(server)

    dummy_states = np.random.randn(16, 19, 8, 8).astype(np.float32)
    policy, wdl, version = client.evaluate_leaves(dummy_states)

    assert policy.shape == (16, 4672)
    assert wdl.shape == (16, 3)
    assert version == server.model_version


def _worker_subroutine(
    worker_id, shm_prefix, capacity, req_q, comp_event, slot_range, num_evals, results_q
):
    """Subroutine for multiprocessing test."""
    try:
        shm_pool = SharedMemoryPool(
            name_prefix=shm_prefix,
            capacity=capacity,
            create=False,
        )
        client = ShmInferenceClient(
            worker_id=worker_id,
            shm_pool=shm_pool,
            request_queue=req_q,
            completion_event=comp_event,
            slot_range=slot_range,
        )

        for _ in range(num_evals):
            # Batch of 32 leaves per request
            states = np.random.randn(32, 19, 8, 8).astype(np.float32)
            p, w, v = client.evaluate_leaves(states)
            assert p.shape == (32, 4672)
            assert w.shape == (32, 3)
            assert v >= 1

        shm_pool.close()
        results_q.put((worker_id, "SUCCESS"))
    except Exception as e:
        results_q.put((worker_id, f"ERROR: {e}"))


def test_multiprocess_shm_inference_ipc(tmp_path):
    """Verify that multiple worker processes concurrently query the server via SharedMemory IPC."""
    num_workers = 4
    capacity = 1024
    shm_prefix = f"zeropawn_test_{os.getpid()}"

    # Create shared memory pool in main process
    master_shm = SharedMemoryPool(
        name_prefix=shm_prefix,
        capacity=capacity,
        in_channels=19,
        height=8,
        width=8,
        num_actions=4672,
        create=True,
    )

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    reload_queue = ctx.Queue()
    stop_event = ctx.Event()
    results_queue = ctx.Queue()

    worker_events = {w: ctx.Event() for w in range(num_workers)}
    allocator = WorkerSlotAllocator(num_workers=num_workers, capacity=capacity)

    ckpt_path = _get_test_ckpt(tmp_path)

    # Start inference server process
    server_proc = ctx.Process(
        target=run_inference_server_process,
        kwargs=dict(
            shm_pool_prefix=shm_prefix,
            capacity=capacity,
            in_channels=19,
            height=8,
            width=8,
            num_actions=4672,
            request_queue=request_queue,
            worker_events=worker_events,
            stop_event=stop_event,
            reload_queue=reload_queue,
            checkpoint_path=ckpt_path,
            min_batch_size=64,
            max_batch_size=256,
            max_wait_ms=3.0,
        ),
    )
    server_proc.start()

    time.sleep(1.0)  # Wait for server to load model and be ready

    # Start worker processes
    worker_procs = []
    for w in range(num_workers):
        slot_range = allocator.get_worker_slots(w)
        p = ctx.Process(
            target=_worker_subroutine,
            args=(
                w,
                shm_prefix,
                capacity,
                request_queue,
                worker_events[w],
                slot_range,
                5,
                results_queue,
            ),
        )
        p.start()
        worker_procs.append(p)

    # Collect worker results
    for _ in range(num_workers):
        w_id, status = results_queue.get(timeout=30)
        assert status == "SUCCESS", f"Worker {w_id} failed: {status}"

    for p in worker_procs:
        p.join(timeout=5)

    # Stop server
    stop_event.set()
    request_queue.put("STOP")
    server_proc.join(timeout=5)

    # Cleanup master shared memory
    master_shm.cleanup()
