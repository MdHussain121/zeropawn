"""Profiling script for Task 2 wall-clock analysis.

Collects microsecond-resolution timing traces across:
1. C++ MCTS search vs Python callback
2. State encoding (encode_state)
3. Tablebase adjudication & FEN extraction
4. Worker batch aggregation & queue lock waits
5. Inference Server Host-to-Device transfer
6. GPU forward inference pass
7. Inference Server Device-to-Host transfer & tolist unboxing
8. Training step breakdown: replay sampling, H2D copy, forward, backward, optimizer step
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Path setup
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_build_dir = _repo_root / "build" / "cpp"
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))
_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_winlibs_bin)
    except OSError:
        pass

import zeropawn_cpp
import numpy as np
import torch

from python.model.chess_net import ChessNet
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import TablebaseAdjudicator
from python.train.train_rl import RLTrainer


def profile_isolated_components(num_iterations: int = 1000) -> Dict[str, float]:
    """Measure isolated execution time of each hot-path operation in microseconds."""
    timings = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. C++ ChessGame state operations
    game = zeropawn_cpp.ChessGame()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        actions = game.legal_actions()
    t_actions = (time.perf_counter() - t0) / num_iterations * 1e6
    timings["cpp_legal_actions_us"] = t_actions

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        planes = zeropawn_cpp.encode_state(game)
    t_encode = (time.perf_counter() - t0) / num_iterations * 1e6
    timings["cpp_encode_state_us"] = t_encode

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        fen = game.get_fen()
    t_fen = (time.perf_counter() - t0) / num_iterations * 1e6
    timings["cpp_get_fen_us"] = t_fen

    # 2. Tablebase probing (with 5-piece position)
    adjudicator = TablebaseAdjudicator.get_instance("data/syzygy", max_pieces=5)
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = adjudicator.probe("8/8/8/8/8/5K2/4Q3/R6k b - - 0 1")
    t_tb = (time.perf_counter() - t0) / num_iterations * 1e6
    timings["tablebase_probe_hit_us"] = t_tb

    # 3. Model forward pass and GPU memory transfers
    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    model.eval()

    for batch_size in [4, 16, 64, 128, 256]:
        dummy_np = np.random.randn(batch_size, 19, 8, 8).astype(np.float32)

        # HtoD transfer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            t = torch.from_numpy(dummy_np).to(device, non_blocking=True)
            torch.cuda.synchronize()
        t_h2d = (time.perf_counter() - t0) / 100 * 1e6
        timings[f"h2d_copy_batch_{batch_size}_us"] = t_h2d

        # Forward pass (bfloat16)
        t_input = torch.from_numpy(dummy_np).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(100):
                    p, w = model(t_input)
                    torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t0) / 100 * 1e6
        timings[f"gpu_forward_batch_{batch_size}_us"] = t_fwd

        # DtoH transfer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            p_cpu = p.float().cpu().numpy()
            w_cpu = w.float().cpu().numpy()
            torch.cuda.synchronize()
        t_d2h = (time.perf_counter() - t0) / 100 * 1e6
        timings[f"d2h_copy_batch_{batch_size}_us"] = t_d2h

        # Python float boxing: .tolist() on 4672 elements
        t0 = time.perf_counter()
        for _ in range(100):
            for i in range(batch_size):
                _ = p_cpu[i].tolist()
        t_tolist = (time.perf_counter() - t0) / 100 * 1e6
        timings[f"python_tolist_batch_{batch_size}_us"] = t_tolist

    # 4. Training step breakdown
    buf_dir = Path("data/tmp_profile_buf")
    buf_dir.mkdir(parents=True, exist_ok=True)
    buf = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=10000,
        human_data_dir="data/dataset_scaled/train",
        gpu_device=device,
        create_new=True,
    )
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        device=device,
        batch_size=256,
        use_bfloat16=True,
    )

    # Warmup
    for _ in range(10):
        trainer.train_step()
    torch.cuda.synchronize()

    # Measure sample_batch
    t0 = time.perf_counter()
    for _ in range(50):
        batch, _ = buf.sample_batch(256)
    t_sample = (time.perf_counter() - t0) / 50 * 1e6
    timings["train_sample_batch_256_us"] = t_sample

    # Measure train_step overall
    t0 = time.perf_counter()
    for _ in range(50):
        trainer.train_step()
        torch.cuda.synchronize()
    t_step = (time.perf_counter() - t0) / 50 * 1e6
    timings["train_step_total_us"] = t_step

    # Cleanup
    import shutil

    shutil.rmtree(buf_dir, ignore_errors=True)

    return timings


if __name__ == "__main__":
    results = profile_isolated_components()
    print("\n--- ISOLATED COMPONENT TIMINGS (MICROSECONDS) ---")
    for k, v in results.items():
        print(f"  {k:<35}: {v:>10.2f} us ({v / 1000.0:>6.3f} ms)")
