"""Item 1(a): Memory Reproduction Experiment.

Compares:
1. Buffer capacity 100,000 (reproducing the ~7.0 GB RSS environment)
2. Buffer capacity 20,000

Reports:
- Process RSS
- Process USS (Unique Set Size / Private Bytes)
- Memory-mapped file sizes (visits.bin, planes.bin, tablebases)
- Telemetry every 10s up to 120s (2 minutes, strictly < 5 min rule)
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import shutil
import psutil
import torch
import numpy as np

from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import SelfPlayWorker


def get_mapped_file_sizes(buf_dir: Path, tb_dir: Path):
    visits_path = buf_dir / "visits.bin"
    planes_path = buf_dir / "planes.bin"
    v_sz = visits_path.stat().st_size if visits_path.exists() else 0
    p_sz = planes_path.stat().st_size if planes_path.exists() else 0
    tb_sz = sum(f.stat().st_size for f in tb_dir.glob("*.rtbw")) if tb_dir.exists() else 0
    return {
        "visits_bin_mb": v_sz / (1024.0 * 1024.0),
        "planes_bin_mb": p_sz / (1024.0 * 1024.0),
        "tablebases_mb": tb_sz / (1024.0 * 1024.0),
        "total_mapped_mb": (v_sz + p_sz + tb_sz) / (1024.0 * 1024.0),
    }


def run_trial_memory(capacity: int, duration_sec: float = 120.0):
    buf_dir = REPO_ROOT / f"tmp_mem_exp_{capacity}"
    if buf_dir.exists():
        shutil.rmtree(buf_dir, ignore_errors=True)
    buf_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = REPO_ROOT / "data" / "syzygy"

    replay_buf = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=capacity,
        gpu_device="cuda" if torch.cuda.is_available() else "cpu",
        create_new=True,
    )
    mapped_sizes = get_mapped_file_sizes(buf_dir, tb_dir)

    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_batch_size=48,
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    def sink(positions, summary):
        replay_buf.add_positions(positions)

    worker = SelfPlayWorker(
        worker_id=1,
        client=client,
        num_concurrent_games=24,
        game_type="chess",
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        allow_resignation=False,
        tablebase_path=str(tb_dir),
        tablebase_max_pieces=5,
        position_sink=sink,
        max_wait_ms=1.5,
    )

    worker.start()
    t0 = time.time()
    samples = []
    watchdog_fired = None

    try:
        while time.time() - t0 < duration_sec:
            time.sleep(10.0)
            elapsed = time.time() - t0
            proc = psutil.Process()
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / (1024.0 * 1024.0)

            try:
                full_info = proc.memory_full_info()
                uss_mb = getattr(full_info, "uss", full_info.rss) / (1024.0 * 1024.0)
            except Exception:
                uss_mb = rss_mb

            free_ram_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
            cache_stats = worker.get_live_cache_stats()

            sample = {
                "elapsed_sec": round(elapsed, 1),
                "rss_mb": round(rss_mb, 1),
                "uss_mb": round(uss_mb, 1),
                "free_ram_mb": round(free_ram_mb, 1),
                "total_cache_entries": cache_stats["total_cache_entries"],
                "est_cache_mb": round(cache_stats["est_cache_mb"], 1),
            }
            samples.append(sample)
            print(
                f"[{elapsed:5.1f}s] Capacity {capacity}: RSS={rss_mb:6.1f} MB | USS={uss_mb:6.1f} MB | Cache={sample['est_cache_mb']:6.1f} MB ({sample['total_cache_entries']} entries) | Free RAM={free_ram_mb:6.1f} MB",
                flush=True,
            )

            # WATCHDOG
            if rss_mb > 6000.0:
                watchdog_fired = f"WATCHDOG: Process RSS={rss_mb:.1f} MB exceeded 6.0 GB ceiling!"
                print(watchdog_fired, flush=True)
                break
            if free_ram_mb < 2000.0:
                watchdog_fired = f"WATCHDOG: System Free RAM={free_ram_mb:.1f} MB dropped below 2.0 GB safety floor!"
                print(watchdog_fired, flush=True)
                break

    finally:
        worker.stop()
        replay_buf = None
        shutil.rmtree(buf_dir, ignore_errors=True)

    return {
        "capacity": capacity,
        "mapped_sizes": mapped_sizes,
        "samples": samples,
        "watchdog_fired": watchdog_fired,
        "peak_rss_mb": max(s["rss_mb"] for s in samples) if samples else 0.0,
        "peak_uss_mb": max(s["uss_mb"] for s in samples) if samples else 0.0,
        "final_rss_mb": samples[-1]["rss_mb"] if samples else 0.0,
        "final_uss_mb": samples[-1]["uss_mb"] if samples else 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=20_000)
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()

    print(f"=== STARTING MEMORY REPRODUCTION EXPERIMENT: Capacity {args.capacity} ===")
    res = run_trial_memory(args.capacity, duration_sec=args.duration)

    print("\n" + "=" * 80)
    print(f"TRIAL RESULT (Capacity {args.capacity}):")
    print("=" * 80)
    print(
        f"  Mapped files: visits.bin={res['mapped_sizes']['visits_bin_mb']:.1f} MB, planes.bin={res['mapped_sizes']['planes_bin_mb']:.1f} MB, tablebases={res['mapped_sizes']['tablebases_mb']:.1f} MB (Total={res['mapped_sizes']['total_mapped_mb']:.1f} MB)"
    )
    print(f"  Peak RSS:     {res['peak_rss_mb']:.1f} MB | Peak USS: {res['peak_uss_mb']:.1f} MB")
    print(f"  Final RSS:    {res['final_rss_mb']:.1f} MB | Final USS: {res['final_uss_mb']:.1f} MB")
    print(f"  Watchdog:     {res['watchdog_fired']}")
    print("=" * 80)
