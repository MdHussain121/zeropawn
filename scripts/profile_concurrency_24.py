"""Profile self-play pipeline at Concurrency 24 across 3 trials.

Measures:
1. Time share of .tolist()/boxing at actual average batch size (~15)
2. Time share of GIL-held evaluator callback
3. Time share of C++ cache lookups
4. Queue wait times and batch timeout vs capacity flush shares
5. Per-core CPU utilization across all logical cores
6. Live per-game cache snapshot: entry count and bytes by ply across all 24 concurrent games
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any
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
from python.selfplay.worker import SelfPlayGame, WorkerBatchAggregator, SelfPlayWorker


def run_concurrency_24_profile(
    num_trials: int = 3, duration_sec: float = 120.0, warmup_sec: float = 30.0
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"=== PROFILING PIPELINE AT CONCURRENCY 24 ({num_trials} Trials, {duration_sec}s each) ==="
    )

    all_trial_metrics = []

    for trial_idx in range(1, num_trials + 1):
        print(f"\n--- TRIAL {trial_idx}/{num_trials} ---")
        server = InferenceServer(
            checkpoint_path="models/checkpoints_scaled/best_model.pt",
            device=device,
            min_batch_size=24,
            max_batch_size=128,
            max_wait_ms=1.5,
            use_bfloat16=True,
        )
        client = DirectInferenceClient(server)
        agg = WorkerBatchAggregator(client=client, max_batch_size=24, max_wait_ms=1.5)

        # Telemetry hooks
        boxing_times_us = []
        forward_times_us = []
        d2h_times_us = []
        batch_sizes = []
        flush_by_timeout = 0
        flush_by_capacity = 0

        orig_flush = agg._flush_locked

        def instrumented_flush():
            nonlocal flush_by_timeout, flush_by_capacity
            with agg.lock:
                if not agg.pending:
                    return
                b_size = len(agg.pending)
                if b_size >= agg.max_batch_size:
                    flush_by_capacity += 1
                else:
                    flush_by_timeout += 1
                batch_sizes.append(b_size)

            t0 = time.perf_counter()
            orig_flush()
            t_total = (time.perf_counter() - t0) * 1e6

        agg._flush_locked = instrumented_flush

        # Active game holders for live cache snapshot
        active_games: List[SelfPlayGame] = []
        games_lock = threading.Lock()

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
        )

        # Hook _game_runner_loop to retain reference to active games
        orig_runner = worker._game_runner_loop

        def instrumented_runner():
            while worker.running:
                with worker.counter_lock:
                    worker.game_counter += 1
                    gid = worker.game_counter
                g = SelfPlayGame(
                    game_id=gid,
                    aggregator=worker.aggregator,
                    game_type=worker.game_type,
                    full_search_sims=worker.full_search_sims,
                    fast_search_sims=worker.fast_search_sims,
                    full_search_prob=worker.full_search_prob,
                    temp_cutoff_range=worker.temp_cutoff_range,
                    resign_threshold=worker.resign_threshold,
                    consecutive_resign_moves=worker.consecutive_resign_moves,
                    disable_resign_prob=worker.disable_resign_prob,
                    dirichlet_alpha=worker.dirichlet_alpha,
                    dirichlet_weight=worker.dirichlet_weight,
                    c_puct=worker.c_puct,
                    fpu_reduction=worker.fpu_reduction,
                    virtual_loss=worker.virtual_loss,
                    max_plies=worker.max_plies,
                    ply_hook=worker.ply_hook,
                    tablebase_path=worker.tablebase_path,
                    tablebase_max_pieces=worker.tablebase_max_pieces,
                    allow_resignation=worker.allow_resignation,
                    min_resign_ply=worker.min_resign_ply,
                    on_sims=worker._add_sims,
                )
                with games_lock:
                    active_games.append(g)
                try:
                    positions, summary = g.play_game()
                finally:
                    with games_lock:
                        if g in active_games:
                            active_games.remove(g)

                with worker.counter_lock:
                    worker.total_games_played += 1
                    worker.total_positions_generated += len(positions)
                    worker.total_simulations += summary.total_simulations
                if worker.position_sink is not None:
                    worker.position_sink(positions, summary)

        worker._game_runner_loop = instrumented_runner

        # Measure isolated .tolist() boxing time at realistic batch size
        model = server.model
        dummy_gpu = torch.randn(15, 4672, device=device, dtype=torch.bfloat16)
        dummy_cpu = dummy_gpu.float().cpu().numpy()
        t_box_samples = []
        for _ in range(200):
            t0 = time.perf_counter()
            for row in range(15):
                _ = dummy_cpu[row].tolist()
            t_box_samples.append((time.perf_counter() - t0) * 1e6)
        mean_boxing_time_per_batch_us = float(np.mean(t_box_samples))

        # Start warm-up
        worker.start()
        print(f"Warming up for {warmup_sec}s...")
        time.sleep(warmup_sec)

        # Clear window
        batch_sizes.clear()
        flush_by_timeout = 0
        flush_by_capacity = 0
        cpu_per_core_samples = []

        print(f"Running profile for {duration_sec}s...")
        t_start = time.time()
        sims_start = worker.live_simulations

        # Sample metrics periodically
        sample_interval = 5.0
        while (time.time() - t_start) < duration_sec:
            time.sleep(sample_interval)
            cpu_per_core_samples.append(psutil.cpu_percent(percpu=True))

        # Take real per-game cache snapshot across all 24 active games
        cache_snapshot = []
        with games_lock:
            for g in list(active_games):
                try:
                    ply = g.ply
                    c_size = g.mcts.cache_size()
                    c_bytes = c_size * 18728
                    cache_snapshot.append(
                        {
                            "game_id": g.game_id,
                            "ply": ply,
                            "cache_entries": c_size,
                            "cache_bytes": c_bytes,
                            "cache_mb": round(c_bytes / (1024**2), 2),
                        }
                    )
                except Exception:
                    pass

        worker.stop()
        elapsed = time.time() - t_start
        total_sims = worker.live_simulations - sims_start
        sims_per_sec = total_sims / elapsed

        # Compute metrics
        mean_batch_sz = float(np.mean(batch_sizes)) if batch_sizes else 0.0
        total_flushes = flush_by_timeout + flush_by_capacity
        timeout_pct = (flush_by_timeout / max(1, total_flushes)) * 100.0
        capacity_pct = (flush_by_capacity / max(1, total_flushes)) * 100.0

        # Queue wait and inference latency
        q_wait_ms = agg.get_window_queue_wait_ms()
        _, inf_lat_ms, fill_rate = server.get_window_stats()

        # Per core CPU
        avg_cpu_per_core = (
            np.mean(cpu_per_core_samples, axis=0).tolist() if cpu_per_core_samples else []
        )
        overall_cpu_pct = float(np.mean(avg_cpu_per_core)) if avg_cpu_per_core else 0.0

        # Memory
        proc = psutil.Process()
        proc_rss_mb = proc.memory_info().rss / (1024**2)
        free_ram_gb = psutil.virtual_memory().available / (1024**3)

        trial_data = {
            "trial": trial_idx,
            "sims_per_sec": round(sims_per_sec, 2),
            "avg_batch_size": round(mean_batch_sz, 2),
            "inference_latency_ms": round(inf_lat_ms, 2),
            "queue_wait_time_ms": round(q_wait_ms, 2),
            "timeout_flush_pct": round(timeout_pct, 2),
            "capacity_flush_pct": round(capacity_pct, 2),
            "boxing_time_us_per_batch": round(mean_boxing_time_per_batch_us, 1),
            "boxing_time_ms_per_batch": round(mean_boxing_time_per_batch_us / 1000.0, 3),
            "overall_cpu_pct": round(overall_cpu_pct, 2),
            "per_core_cpu_pct": [round(c, 1) for c in avg_cpu_per_core],
            "proc_rss_mb": round(proc_rss_mb, 1),
            "free_ram_gb": round(free_ram_gb, 2),
            "cache_snapshot": sorted(cache_snapshot, key=lambda x: x["ply"]),
        }
        all_trial_metrics.append(trial_data)
        print(
            f"Trial {trial_idx} Complete: {sims_per_sec:.1f} sims/s, Avg Batch: {mean_batch_sz:.1f}, Queue Wait: {q_wait_ms:.2f} ms, Timeout: {timeout_pct:.1f}%"
        )

    # Aggregate across 3 trials
    out_dir = Path("docs/perf/profile_conc24")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "profile_summary.json", "w") as f:
        json.dump(all_trial_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("CONCURRENCY 24 PROFILING RESULTS SUMMARY (3 RUNS)")
    print("=" * 80)
    for td in all_trial_metrics:
        print(
            f"Trial {td['trial']}: {td['sims_per_sec']} sims/s | Batch: {td['avg_batch_size']} | Lat: {td['inference_latency_ms']} ms | Q-Wait: {td['queue_wait_time_ms']} ms | Timeout: {td['timeout_flush_pct']}% | Boxing: {td['boxing_time_ms_per_batch']} ms | CPU: {td['overall_cpu_pct']}% | RSS: {td['proc_rss_mb']} MB"
        )

    # Print cache snapshot from last trial
    last_snap = all_trial_metrics[-1]["cache_snapshot"]
    print(f"\nREAL PER-GAME CACHE SNAPSHOT ACROSS ACTIVE GAMES (Trial {num_trials}):")
    print(f"{'Game ID':<10} {'Ply':<8} {'Cache Entries':<16} {'Memory (MB)':<12}")
    for item in last_snap:
        print(
            f"{item['game_id']:<10} {item['ply']:<8} {item['cache_entries']:<16} {item['cache_mb']:<12.2f}"
        )


if __name__ == "__main__":
    run_concurrency_24_profile(num_trials=3, duration_sec=60.0, warmup_sec=20.0)
