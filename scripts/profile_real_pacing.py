"""Follow-up 4: Measure Combined Mode Under Real Pacing (max_reuse_ratio=8)."""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

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

from python.model.chess_net import ChessNet
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import SelfPlayWorker
from python.train.train_rl import RLTrainer


def run_paced_combined(concurrency: int = 6, duration_sec: float = 60.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt",
        device=device,
        min_batch_size=min(48, concurrency * 2),
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    buf_dir = Path("data/tmp_paced_buf")
    buf_dir.mkdir(parents=True, exist_ok=True)
    buffer = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=100_000,
        human_data_dir="data/dataset_scaled/train",
        gpu_device=device,
        create_new=True,
    )

    positions_added = [0]

    def sink(positions, summary):
        buffer.add_positions(positions)
        positions_added[0] += len(positions)

    worker = SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=concurrency,
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        allow_resignation=False,
        tablebase_path="data/syzygy",
        tablebase_max_pieces=5,
        position_sink=sink,
    )

    trainer_model = ChessNet(num_blocks=12, num_filters=128).to(device)
    trainer_model.load_state_dict(server.model.state_dict())
    trainer = RLTrainer(
        model=trainer_model,
        replay_buffer=buffer,
        device=device,
        batch_size=256,
        direct_server=server,
        use_bfloat16=True,
    )

    # Pre-seed buffer with 50,000 human positions to simulate having reached min_buffer_fill 50000
    print("Pre-seeding buffer with 50,000 positions to simulate reaching min_buffer_fill...")
    dummy_positions = []
    # Set size to 50,000
    buffer.size = 50000
    buffer.total_added = 50000

    # Start self-play
    worker.start()

    # Track pacing
    max_reuse_ratio = 8
    batch_size = 256
    train_active_time = 0.0
    total_train_steps = 0
    t_start = time.time()
    last_step_num = 0

    print(f"Running paced combined benchmark for {duration_sec}s with max_reuse_ratio=8...")
    while (time.time() - t_start) < duration_sec:
        # Pacing condition from Phase4Orchestrator
        target_max_step = max(500, int((buffer.total_added * max_reuse_ratio) / batch_size))
        if trainer.step_num >= target_max_step:
            time.sleep(0.02)
            continue

        t0 = time.perf_counter()
        trainer.train_step()
        train_active_time += time.perf_counter() - t0
        total_train_steps += 1

    worker.stop()
    elapsed = time.time() - t_start

    sims_per_sec = worker.live_simulations / elapsed
    train_steps_per_sec = total_train_steps / elapsed
    train_samples_per_sec = train_steps_per_sec * batch_size
    train_time_fraction = (train_active_time / elapsed) * 100.0

    avg_batch_sz, avg_latency_ms, fill_rate = server.get_window_stats()

    import shutil

    shutil.rmtree(buf_dir, ignore_errors=True)

    print("\n--- COMBINED MODE UNDER REAL PACING (REUSE RATIO 8, 50k BUFFER) ---")
    print(f"Self-play Sims/sec: {sims_per_sec:.1f}")
    print(f"Inference Latency: {avg_latency_ms:.2f} ms")
    print(f"Training Steps/sec: {train_steps_per_sec:.2f}")
    print(f"Training Samples/sec: {train_samples_per_sec:.1f}")
    print(f"Total Time Trainer Was Executing on GPU: {train_active_time:.2f}s / {elapsed:.1f}s")
    print(f"Fraction of Time Trainer Is Actually Running: {train_time_fraction:.2f}%")


if __name__ == "__main__":
    run_paced_combined(concurrency=6, duration_sec=40.0)
