"""Measure Model-Version Lag over 30-minute self-play run with real trainer pacing.

Protocol:
- Concurrency: 24 concurrent games
- Settings: 400 full / 80 fast simulations, full_search_prob=0.35, tablebase=5pc Syzygy, resignation=False
- Trainer Pacing: max_reuse_ratio=8.0, min_buffer_fill=50,000, batch_size=256, push_freq=200
- Duration: 30 minutes (1,800 seconds)

Reports:
- Distribution of (current version - generating version) in pushes and in minutes
- Real reuse ratio achieved
- push_freq used
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import psutil
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

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import SelfPlayWorker
from python.train.train_rl import RLTrainer


def run_model_lag_measurement(
    duration_sec: float = 1800.0, max_reuse_ratio: float = 8.0, push_freq: int = 200
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== MEASURED MODEL LAG EXPERIMENT ===")
    print(f"Duration: {duration_sec}s ({duration_sec / 60.0:.1f} minutes)")
    print(f"Concurrency: 24, Reuse Ratio: {max_reuse_ratio}, Push Freq: {push_freq}")
    print(f"Simulations: 400 full / 80 fast (p=0.35), Syzygy 5pc, Resignation=False")

    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt",
        device=device,
        min_batch_size=24,
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    buf_dir = Path("data/tmp_lag_buf")
    shutil.rmtree(buf_dir, ignore_errors=True)
    buf_dir.mkdir(parents=True, exist_ok=True)

    buffer = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=500_000,
        human_data_dir="data/dataset_scaled/train",
        gpu_device=device,
        create_new=True,
    )

    # Pre-seed buffer to 50,000 to represent having reached min_buffer_fill 50,000
    print("Pre-seeding replay buffer to min_buffer_fill = 50,000 positions...")
    buffer.size = 50000
    buffer.total_added = 50000

    trainer_model = ChessNet(num_blocks=12, num_filters=128).to(device)
    trainer_model.load_state_dict(server.model.state_dict())

    trainer = RLTrainer(
        model=trainer_model,
        replay_buffer=buffer,
        device=device,
        batch_size=256,
        push_freq=push_freq,
        direct_server=server,
        use_bfloat16=True,
    )

    # Position tracking structures
    position_records = []  # list of (generating_version, created_timestamp)
    push_events = [(0, time.time())]  # list of (version, timestamp)
    last_known_server_version = server.model_version

    def sink(positions, summary):
        now = time.time()
        for p in positions:
            gen_v = getattr(p, "model_version", server.model_version)
            position_records.append((gen_v, now))
        buffer.add_positions(positions)

    worker = SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=24,
        game_type="chess",
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        allow_resignation=False,
        tablebase_path="data/syzygy",
        tablebase_max_pieces=5,
        position_sink=sink,
    )

    print(f"Starting self-play workers at concurrency 24...")
    worker.start()
    t_start = time.time()
    batch_size = 256

    print(f"Starting paced training loop for {duration_sec}s...")
    last_log_time = t_start

    while (time.time() - t_start) < duration_sec:
        # Pacing formula
        target_max_step = max(500, int((buffer.total_added * max_reuse_ratio) / batch_size))
        if trainer.step_num < target_max_step:
            trainer.train_step()
            # Check for weight push
            if server.model_version > last_known_server_version:
                last_known_server_version = server.model_version
                push_events.append((last_known_server_version, time.time()))
        else:
            time.sleep(0.01)

        now = time.time()
        if (now - last_log_time) >= 60.0:
            elapsed_min = (now - t_start) / 60.0
            print(
                f"[{elapsed_min:.1f}m / {duration_sec / 60.0:.1f}m] "
                f"Positions: {len(position_records)} | Steps: {trainer.step_num} | "
                f"Model Version: {server.model_version} | Pushes: {len(push_events) - 1}"
            )
            last_log_time = now

    worker.stop()
    t_end = time.time()
    total_elapsed_sec = t_end - t_start
    total_new_positions = len(position_records)
    final_version = server.model_version
    total_training_steps = trainer.step_num
    total_samples_trained = total_training_steps * batch_size

    # Compute actual reuse ratio
    actual_reuse_ratio = total_samples_trained / max(1, total_new_positions)

    # Compute lag distribution across all newly generated positions
    lags_in_pushes = []
    lags_in_minutes = []

    for gen_v, t_created in position_records:
        # Age in pushes at time of experiment end
        lag_pushes = final_version - gen_v
        # Age in minutes at time of experiment end
        lag_mins = (t_end - t_created) / 60.0
        lags_in_pushes.append(lag_pushes)
        lags_in_minutes.append(lag_mins)

    # Statistics
    push_arr = np.array(lags_in_pushes, dtype=np.float64)
    min_arr = np.array(lags_in_minutes, dtype=np.float64)

    results = {
        "duration_sec": total_elapsed_sec,
        "duration_minutes": round(total_elapsed_sec / 60.0, 2),
        "concurrency": 24,
        "push_freq_steps": push_freq,
        "target_reuse_ratio": max_reuse_ratio,
        "actual_reuse_ratio": round(actual_reuse_ratio, 2),
        "total_new_positions_generated": total_new_positions,
        "total_training_steps": total_training_steps,
        "total_samples_trained": total_samples_trained,
        "final_model_version": final_version,
        "total_pushes": len(push_events) - 1,
        "lag_in_pushes": {
            "mean": round(float(np.mean(push_arr)), 2),
            "median": round(float(np.median(push_arr)), 2),
            "std": round(float(np.std(push_arr)), 2),
            "p90": round(float(np.percentile(push_arr, 90)), 2),
            "p95": round(float(np.percentile(push_arr, 95)), 2),
            "max": int(np.max(push_arr)),
            "min": int(np.min(push_arr)),
        },
        "lag_in_minutes": {
            "mean": round(float(np.mean(min_arr)), 2),
            "median": round(float(np.median(min_arr)), 2),
            "std": round(float(np.std(min_arr)), 2),
            "p90": round(float(np.percentile(min_arr, 90)), 2),
            "p95": round(float(np.percentile(min_arr, 95)), 2),
            "max": round(float(np.max(min_arr)), 2),
            "min": round(float(np.min(min_arr)), 2),
        },
    }

    out_file = Path("docs/perf/measured_model_lag_30min.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    shutil.rmtree(buf_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("MEASURED MODEL-VERSION LAG RESULTS (30-MINUTE PACED RUN AT CONCURRENCY 24)")
    print("=" * 80)
    print(f"Elapsed Time:                  {results['duration_minutes']} minutes")
    print(f"Positions Generated:           {results['total_new_positions_generated']:,}")
    print(f"Training Steps Executed:       {results['total_training_steps']:,}")
    print(f"Total Samples Trained:         {results['total_samples_trained']:,}")
    print(f"Push Frequency (push_freq):    {results['push_freq_steps']} steps")
    print(f"Target Max Reuse Ratio:        {results['target_reuse_ratio']}")
    print(f"Actual Measured Reuse Ratio:   {results['actual_reuse_ratio']}")
    print(
        f"Total Weight Pushes:           {results['total_pushes']} pushes (Version 0 -> {results['final_model_version']})"
    )
    print("-" * 80)
    print("DISTRIBUTION OF (CURRENT MODEL VERSION - GENERATING VERSION):")
    print(
        f"  Mean Lag:                    {results['lag_in_pushes']['mean']} pushes ({results['lag_in_minutes']['mean']} min)"
    )
    print(
        f"  Median Lag (p50):            {results['lag_in_pushes']['median']} pushes ({results['lag_in_minutes']['median']} min)"
    )
    print(
        f"  p90 Lag:                     {results['lag_in_pushes']['p90']} pushes ({results['lag_in_minutes']['p90']} min)"
    )
    print(
        f"  p95 Lag:                     {results['lag_in_pushes']['p95']} pushes ({results['lag_in_minutes']['p95']} min)"
    )
    print(
        f"  Max Lag:                     {results['lag_in_pushes']['max']} pushes ({results['lag_in_minutes']['max']} min)"
    )
    print(
        f"  Min Lag:                     {results['lag_in_pushes']['min']} pushes ({results['lag_in_minutes']['min']} min)"
    )
    print("=" * 80)


if __name__ == "__main__":
    run_model_lag_measurement(duration_sec=1800.0, max_reuse_ratio=8.0, push_freq=200)
