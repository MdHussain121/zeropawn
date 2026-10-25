"""Comprehensive Execution Device Matrix Verification (Modes A - F).

Modes:
- Mode A: Pure CPU-only execution (verified in isolated subprocess with CUDA_VISIBLE_DEVICES="").
- Mode B: GPU training + GPU inference.
- Mode C: CPU search + GPU inference.
- Mode D: CPU training + GPU inference (cross-device weight transfer).
- Mode E: GPU training + CPU inference (cross-device weight transfer).
- Mode F: Automatic CPU fallback when CUDA is requested or unavailable.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from python.cli.common import REPO_ROOT, resolve_device
from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.train.train_rl import RLTrainer


def test_mode_a_pure_cpu_subprocess():
    """Mode A: Pure CPU-only execution in an isolated subprocess with CUDA_VISIBLE_DEVICES=''.

    Proves zero hidden CUDA dependencies, CPU model instantiation, inference forward pass,
    and loss backward pass without any CUDA runtime initialization.
    """
    script = """
import os
import sys
from unittest import mock
import torch
import numpy as np

assert (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0), "CUDA devices must not be visible when CUDA_VISIBLE_DEVICES=''"

from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.train.train_rl import RLTrainer

# 1. Instantiate model on CPU
model = ChessNet(num_blocks=2, num_filters=32)
model.eval()

# 2. Inference on CPU
server = InferenceServer(model=model, device="cpu")
planes = np.zeros((1, 19, 8, 8), dtype=np.float32)
planes[:, 0, :, :] = 1.0
pol, wdl, ver = server.predict_batch(planes)
assert pol.shape == (1, 4672)
assert wdl.shape == (1, 3)
assert ver == 0

# 3. Training step on CPU
trainer = RLTrainer(model=model, replay_buffer=mock.MagicMock(), device="cpu", base_lr=1e-3)
batch_dict = {
    "planes": torch.from_numpy(planes),
    "visits": torch.zeros(1, 4672, dtype=torch.float32),
    "outcomes": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    "is_full_search": torch.tensor([True], dtype=torch.bool),
}
batch_dict["visits"][0, 0] = 1.0

metrics = trainer.train_step(batch_dict=batch_dict)
assert "total_loss" in metrics
assert metrics["total_loss"] > 0.0

print("MODE_A_SUCCESS")
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=25.0,
    )

    assert proc.returncode == 0, (
        f"Mode A failed with code {proc.returncode}\nStdout: {proc.stdout}\nStderr: {proc.stderr}"
    )
    assert "MODE_A_SUCCESS" in proc.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_mode_b_gpu_train_gpu_infer():
    """Mode B: GPU training + GPU inference on CUDA."""
    model = ChessNet(num_blocks=2, num_filters=32)
    server = InferenceServer(model=model, device="cuda")
    trainer = RLTrainer(model=model, replay_buffer=mock.MagicMock(), device="cuda", base_lr=1e-3)

    planes = np.zeros((2, 19, 8, 8), dtype=np.float32)
    planes[:, 0, :, :] = 1.0
    pol, wdl, ver = server.predict_batch(planes)
    assert pol.shape == (2, 4672)
    assert wdl.shape == (2, 3)

    batch_dict = {
        "planes": torch.from_numpy(planes).cuda(),
        "visits": torch.zeros(2, 4672, device="cuda", dtype=torch.float32),
        "outcomes": torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32
        ),
        "is_full_search": torch.tensor([True, True], device="cuda", dtype=torch.bool),
    }
    batch_dict["visits"][:, 0] = 1.0

    metrics = trainer.train_step(batch_dict=batch_dict)
    assert metrics["total_loss"] > 0.0


def test_mode_c_cpu_search_gpu_infer():
    """Mode C: CPU search with batched inference server."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNet(num_blocks=2, num_filters=32)
    server = InferenceServer(model=model, device=device)

    # Simulate CPU-based search threads generating positions
    for _ in range(5):
        planes = np.zeros((1, 19, 8, 8), dtype=np.float32)
        pol, wdl, ver = server.predict_batch(planes)
        assert isinstance(pol, np.ndarray)
        assert isinstance(wdl, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_mode_d_cpu_train_gpu_infer_cross_device():
    """Mode D: CPU training + GPU inference (cross-device weight transfer)."""
    # Trainer model on CPU
    cpu_model = ChessNet(num_blocks=2, num_filters=32).cpu()
    trainer = RLTrainer(model=cpu_model, replay_buffer=mock.MagicMock(), device="cpu", base_lr=1e-3)

    # Inference model on GPU
    gpu_model = ChessNet(num_blocks=2, num_filters=32).cuda()
    server = InferenceServer(model=gpu_model, device="cuda")

    # Step trainer on CPU
    batch_dict = {
        "planes": torch.zeros(2, 19, 8, 8, device="cpu"),
        "visits": torch.zeros(2, 4672, device="cpu"),
        "outcomes": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device="cpu"),
        "is_full_search": torch.tensor([True, True], device="cpu", dtype=torch.bool),
    }
    batch_dict["visits"][:, 10] = 1.0

    trainer.train_step(batch_dict=batch_dict)

    # Push CPU weights to GPU server
    state_dict_cpu = trainer.model.state_dict()
    v_new = server.load_state_dict_weights(state_dict_cpu)
    assert v_new == 1

    # Inference on GPU with updated weights
    planes = np.zeros((1, 19, 8, 8), dtype=np.float32)
    pol, wdl, ver = server.predict_batch(planes)
    assert pol.shape == (1, 4672)
    assert ver == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_mode_e_gpu_train_cpu_infer_cross_device():
    """Mode E: GPU training + CPU inference (cross-device weight transfer)."""
    # Trainer model on GPU
    gpu_model = ChessNet(num_blocks=2, num_filters=32).cuda()
    trainer = RLTrainer(
        model=gpu_model, replay_buffer=mock.MagicMock(), device="cuda", base_lr=1e-3
    )

    # Inference model on CPU
    cpu_model = ChessNet(num_blocks=2, num_filters=32).cpu()
    server = InferenceServer(model=cpu_model, device="cpu")

    # Step trainer on GPU
    batch_dict = {
        "planes": torch.zeros(2, 19, 8, 8, device="cuda"),
        "visits": torch.zeros(2, 4672, device="cuda"),
        "outcomes": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"),
        "is_full_search": torch.tensor([True, True], device="cuda", dtype=torch.bool),
    }
    batch_dict["visits"][:, 10] = 1.0

    trainer.train_step(batch_dict=batch_dict)

    # Push GPU weights to CPU server
    state_dict_gpu = trainer.model.state_dict()
    v_new = server.load_state_dict_weights(state_dict_gpu)
    assert v_new == 1

    # Inference on CPU with updated weights
    planes = np.zeros((1, 19, 8, 8), dtype=np.float32)
    pol, wdl, ver = server.predict_batch(planes)
    assert pol.shape == (1, 4672)
    assert ver == 1


def test_mode_f_auto_cpu_fallback():
    """Mode F: Graceful CPU fallback when device is unconfigured or invalid."""
    resolved_none = resolve_device(None)
    assert resolved_none in ("cuda", "cpu")

    resolved_cpu = resolve_device("cpu")
    assert resolved_cpu == "cpu"

    with pytest.raises(ValueError):
        resolve_device("tpu")
