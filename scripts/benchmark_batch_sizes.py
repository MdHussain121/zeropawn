"""Automated Batch Size & Timeout Benchmark for Self-Play Inference.

Evaluates max_batch_size = [128, 256, 512] with a short timeout (2 to 5 ms).
Measures:
  1. Simulations per second (aggregate leaf evaluations / sec)
  2. Batch fill rate (average batch size / max_batch_size * 100%)
  3. GPU utilization (%)

Applies the project decision rule:
  - Choose the setting with the highest simulations per second.
  - If 512 only fills about 40% of the time, stay at 256.
  - If 256 is always full and the GPU isn't saturated, try 512.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_build_cpp = _repo_root / "build" / "cpp"
if _build_cpp.is_dir() and str(_build_cpp) not in sys.path:
    sys.path.insert(0, str(_build_cpp))

_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    try:
        os.add_dll_directory(_winlibs_bin)
    except (AttributeError, OSError):
        pass

import zeropawn_cpp
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import SavedPosition, GameSummary, SelfPlayWorker
from python.selfplay.metrics import query_gpu_metrics
from python.cli.common import setup_logging
from python.common.progress import (
    banner,
    emit_error,
    emit_report,
    error_block,
    header_block,
    parameters_table,
    run_context,
    summary_block,
)


def benchmark_single_config(
    batch_size: int,
    timeout_ms: float = 3.0,
    duration_sec: float = 300.0,
    concurrent_games: int = 128,
    checkpoint_path: str = "models/checkpoints_scaled/best_model.pt",
) -> Dict[str, Any]:
    """Run a single self-play configuration and record throughput metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 72}")
    print(f"BENCHMARKING: Max Batch Size = {batch_size} | Timeout = {timeout_ms:.1f}ms")
    print(
        f"Duration: {duration_sec:.0f}s ({duration_sec / 60:.1f} min) | Concurrent Games: {concurrent_games} | Device: {device}"
    )
    print(f"{'=' * 72}", flush=True)

    server = InferenceServer(
        checkpoint_path=checkpoint_path,
        device=device,
        min_batch_size=min(48, batch_size // 2),
        max_batch_size=batch_size,
        max_wait_ms=timeout_ms,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    total_positions = 0
    total_games = 0
    gpu_util_samples: List[float] = []

    def sink(positions: List[SavedPosition], summary: GameSummary):
        nonlocal total_positions, total_games
        total_positions += len(positions)
        total_games += 1

    worker = SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=concurrent_games,
        game_type="chess",
        full_search_sims=800,
        fast_search_sims=150,
        full_search_prob=0.25,
        temp_cutoff_range=(15, 30),
        resign_threshold=-0.9,
        consecutive_resign_moves=4,
        disable_resign_prob=0.10,
        position_sink=sink,
    )
    # Set aggregator batch size to match server max_batch_size
    worker.aggregator.max_batch_size = batch_size
    worker.aggregator.max_wait_sec = timeout_ms / 1000.0

    print("Starting worker threads...", flush=True)
    worker.start()

    start_time = time.perf_counter()
    next_sample_time = start_time + 1.0

    try:
        while (time.perf_counter() - start_time) < duration_sec:
            time.sleep(0.5)
            now = time.perf_counter()
            if now >= next_sample_time:
                gpu_util, _, _ = query_gpu_metrics()
                gpu_util_samples.append(gpu_util)
                next_sample_time = now + 1.0

                elapsed = now - start_time
                current_evals = server.total_evaluations
                current_sims_per_sec = current_evals / max(1e-5, elapsed)
                avg_b = server.total_batch_size_sum / max(1, server.total_batches)
                fill_pct = (avg_b / batch_size) * 100.0
                print(
                    f"  [{elapsed:4.0f}s/{duration_sec:4.0f}s] "
                    f"Sims/sec: {current_sims_per_sec:7.1f} | "
                    f"Avg Batch: {avg_b:5.1f}/{batch_size} ({fill_pct:4.1f}%) | "
                    f"GPU Util: {gpu_util:4.1f}%",
                    flush=True,
                )
    finally:
        print("Stopping workers...", flush=True)
        worker.stop()

    elapsed_total = time.perf_counter() - start_time
    total_simulations = server.total_evaluations
    sims_per_sec = total_simulations / max(1e-5, elapsed_total)
    avg_batch_size = server.total_batch_size_sum / max(1, server.total_batches)
    batch_fill_rate = avg_batch_size / batch_size
    avg_gpu_util = float(np.mean(gpu_util_samples)) if gpu_util_samples else 0.0

    return {
        "max_batch_size": batch_size,
        "timeout_ms": timeout_ms,
        "duration_sec": elapsed_total,
        "total_simulations": total_simulations,
        "sims_per_sec": sims_per_sec,
        "avg_batch_size": avg_batch_size,
        "batch_fill_rate": batch_fill_rate,
        "avg_gpu_util": avg_gpu_util,
    }


def main() -> None:
    setup_logging(log_dir="logs", log_filename="benchmark_batch_sizes.log")
    parser = argparse.ArgumentParser(description="Benchmark batch sizes (128, 256, 512)")
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Duration in seconds per configuration (default: 300s = 5 min)",
    )
    parser.add_argument(
        "--timeout", type=float, default=3.0, help="Max wait timeout in ms (default: 3.0ms)"
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=96,
        help="Concurrent games for load generation (default: 96)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints_scaled/best_model.pt",
        help="Checkpoint path",
    )
    args = parser.parse_args()

    ctx = run_context("benchmark-batch")
    t0 = time.time()
    emit_report(banner("benchmark-batch", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            args.checkpoint,
            "auto",
            "n/a",
        )
    )
    emit_report(
        parameters_table(
            [
                ("duration", args.duration, "user" if args.duration != 300.0 else "default"),
                ("timeout", args.timeout, "user" if args.timeout != 3.0 else "default"),
                (
                    "concurrent_games",
                    args.concurrent_games,
                    "user" if args.concurrent_games != 96 else "default",
                ),
                (
                    "checkpoint",
                    args.checkpoint,
                    "user"
                    if args.checkpoint != "models/checkpoints_scaled/best_model.pt"
                    else "default",
                ),
            ]
        )
    )

    configs = [128, 256, 512]
    results: List[Dict[str, Any]] = []

    print("=" * 72)
    print("      ZEROPAWN BATCH SIZE SELECTION BENCHMARK (128 vs 256 vs 512)")
    print(
        f"Testing {len(configs)} configurations for {args.duration:.0f}s each (Total ~{len(configs) * args.duration / 60:.1f} mins)"
    )
    print("=" * 72, flush=True)

    for b in configs:
        res = benchmark_single_config(
            batch_size=b,
            timeout_ms=args.timeout,
            duration_sec=args.duration,
            concurrent_games=args.concurrent_games,
            checkpoint_path=args.checkpoint,
        )
        results.append(res)

    print("\n" + "=" * 78)
    print("                       FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 78)
    print(
        f"{'Max Batch':>10} | {'Sims/sec':>12} | {'Avg Batch':>11} | {'Fill Rate':>11} | {'GPU Util':>10}"
    )
    print("-" * 78)

    best_config = None
    highest_sims = -1.0

    for r in results:
        b = r["max_batch_size"]
        sims = r["sims_per_sec"]
        avg_b = r["avg_batch_size"]
        fill_pct = r["batch_fill_rate"] * 100.0
        gpu_u = r["avg_gpu_util"]
        print(f"{b:10d} | {sims:11.1f}/s | {avg_b:11.1f} | {fill_pct:10.1f}% | {gpu_u:9.1f}%")

    print("=" * 78)

    # Apply decision rule:
    # 1. Start with the setting that has highest sims/sec
    # 2. If 512 only fills about 40% of the time, stay at 256.
    # 3. If 256 is always full and the GPU isn't saturated, try 512.
    res_128 = next((r for r in results if r["max_batch_size"] == 128), None)
    res_256 = next((r for r in results if r["max_batch_size"] == 256), None)
    res_512 = next((r for r in results if r["max_batch_size"] == 512), None)

    decision_rationale = []
    chosen_batch = 256  # default baseline

    if res_512 and res_512["batch_fill_rate"] < 0.40:
        decision_rationale.append(
            f"Max batch 512 only filled {res_512['batch_fill_rate'] * 100:.1f}% (< 40%). Per rule: staying at 256."
        )
        chosen_batch = 256
    elif (
        res_256 and res_256["batch_fill_rate"] > 0.85 and res_256["avg_gpu_util"] < 80.0 and res_512
    ):
        decision_rationale.append(
            "Max batch 256 was highly filled (>85%) while GPU was not saturated. Per rule: selecting 512."
        )
        chosen_batch = 512
    else:
        # Pick highest throughput
        chosen_r = max(results, key=lambda x: x["sims_per_sec"])
        chosen_batch = chosen_r["max_batch_size"]
        decision_rationale.append(
            f"Selected max batch {chosen_batch} because it achieved the highest simulations per second ({chosen_r['sims_per_sec']:.1f} sims/s)."
        )

    print("\nDECISION & RECOMMENDATION:")
    for line in decision_rationale:
        print(f"  * {line}")
    print(f"\n>>> FINAL RECOMMENDED MAX BATCH SIZE: {chosen_batch} <<<\n")
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))


if __name__ == "__main__":
    _t0 = time.time()
    try:
        main()
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - _t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"benchmark-batch failed: {e}",
                "OOM/watchdog trip, bad checkpoint, or missing dataset",
                "check free RAM, --checkpoint, and dataset, then see LOG",
                "logs/benchmark_batch_sizes.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - _t0, 1, 1))
        sys.exit(1)
