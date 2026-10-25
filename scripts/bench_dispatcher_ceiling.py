"""Item 5: Dispatcher Ceiling Benchmark.

Measures the maximum throughput and latency characteristics of the batch aggregator
and inference server under synthetic burst workloads from 24 stub games.

Tests bursts of K in {1, 2, 4, 8} requests per game thread.
Reports:
- Throughput (requests/sec)
- Average batch size dispatched to GPU
- p50, p90, p95, p99 latency (ms)
- CPU unboxing vs GPU compute time breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import WorkerBatchAggregator
from python.cli.common import setup_logging
from python.common.progress import emit_report, event_line, format_kv_line

# Single logging setup: rotating file for everything, WARNING+ on stderr.
# (No logging.basicConfig: one setup path, no duplicate lines.)
setup_logging(log_dir="logs", log_filename="bench_dispatcher_ceiling.log")
logger = logging.getLogger("bench_dispatcher")


class StubGameWorker:
    """Simulates a game thread issuing bursts of K leaf requests."""

    def __init__(
        self,
        thread_id: int,
        aggregator: WorkerBatchAggregator,
        burst_k: int,
        dummy_state: np.ndarray,
        duration_sec: float,
    ) -> None:
        self.thread_id = thread_id
        self.aggregator = aggregator
        self.burst_k = burst_k
        self.dummy_state = dummy_state
        self.duration_sec = duration_sec
        self.latencies_ms: List[float] = []
        self.total_requests = 0
        self.running = True

    def run(self) -> None:
        end_time = time.time() + self.duration_sec
        k = self.burst_k

        while self.running and time.time() < end_time and self.aggregator.running:
            # Issue K requests
            t0 = time.perf_counter()
            boxes = [[None, None] for _ in range(k)]
            events = [threading.Event() for _ in range(k)]

            with self.aggregator.lock:
                for i in range(k):
                    self.aggregator.pending.append((self.dummy_state, boxes[i], events[i]))
                need_flush = len(self.aggregator.pending) >= self.aggregator.max_batch_size
                if not need_flush:
                    self.aggregator.cond.notify()
            if need_flush:
                with self.aggregator._window_lock:
                    self.aggregator.flush_capacity_count += 1
                self.aggregator._flush_locked()

            # Wait for all K results
            for evt in events:
                evt.wait(timeout=10.0)

            t1 = time.perf_counter()
            burst_latency_ms = (t1 - t0) * 1000.0
            per_req_latency_ms = burst_latency_ms / k
            for _ in range(k):
                self.latencies_ms.append(per_req_latency_ms)
            self.total_requests += k


def run_burst_trial(
    burst_k: int,
    num_games: int = 24,
    duration_sec: float = 30.0,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    logger.info(
        "=== Benchmarking Dispatcher Burst K=%d (Threads=%d, Duration=%.0fs) ===",
        burst_k,
        num_games,
        duration_sec,
    )
    emit_report(
        event_line("TRIAL", f"burst K={burst_k} threads={num_games} duration={duration_sec:.0f}s")
    )
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / "models/checkpoints_scaled/best_model.pt"

    server = InferenceServer(
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
        device=device,
        min_batch_size=min(48, num_games * burst_k),
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)
    aggregator = WorkerBatchAggregator(
        client=client,
        max_batch_size=min(128, num_games * burst_k),
        max_wait_ms=1.5,
    )

    disp_thread = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp_thread.start()

    dummy_state = np.zeros((19, 8, 8), dtype=np.float32)
    workers = [
        StubGameWorker(
            thread_id=i,
            aggregator=aggregator,
            burst_k=burst_k,
            dummy_state=dummy_state,
            duration_sec=duration_sec,
        )
        for i in range(num_games)
    ]

    t_start = time.time()
    threads = [threading.Thread(target=w.run, daemon=True) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration_sec + 5.0)

    elapsed = time.time() - t_start
    aggregator.stop()

    all_latencies = []
    total_reqs = 0
    for w in workers:
        all_latencies.extend(w.latencies_ms)
        total_reqs += w.total_requests

    reqs_per_sec = total_reqs / max(1e-6, elapsed)
    p50 = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
    p90 = float(np.percentile(all_latencies, 90)) if all_latencies else 0.0
    p95 = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    p99 = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0
    mean_lat = float(np.mean(all_latencies)) if all_latencies else 0.0

    # Server stats
    if hasattr(server, "get_window_stats"):
        avg_batch_size, avg_inference_latency_ms, _ = server.get_window_stats()
    else:
        avg_batch_size, avg_inference_latency_ms = 0.0, 0.0

    # Estimate unboxing time vs GPU execution time
    # Round-trip latency = queue_wait + gpu_forward + cpu_unboxing
    # cpu_unboxing is the CPU overhead of converting tensor -> numpy / python list and setting events
    est_unbox_ms = max(0.0, mean_lat - avg_inference_latency_ms)

    res = {
        "burst_k": burst_k,
        "num_games": num_games,
        "duration_sec": round(elapsed, 2),
        "total_requests": total_reqs,
        "throughput_reqs_per_sec": round(reqs_per_sec, 1),
        "avg_batch_size": round(avg_batch_size, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "gpu_inference_latency_ms": round(avg_inference_latency_ms, 2),
        "estimated_unbox_wait_ms": round(est_unbox_ms, 2),
        "timeout_flushes": aggregator.flush_timeout_count,
        "capacity_flushes": aggregator.flush_capacity_count,
    }
    logger.info(
        "K=%d | Req/s: %.1f | Avg Batch: %.2f | p95 Latency: %.2f ms | GPU Latency: %.2f ms | Unbox/Wait: %.2f ms",
        burst_k,
        reqs_per_sec,
        avg_batch_size,
        p95,
        avg_inference_latency_ms,
        est_unbox_ms,
    )
    emit_report(
        format_kv_line(
            [
                ("burst_k", burst_k),
                ("req_sec", f"{reqs_per_sec:.1f}"),
                ("batch_avg", f"{avg_batch_size:.2f}"),
                ("p95_ms", f"{p95:.2f}"),
            ]
        )
    )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0, help="Duration per K burst trial")
    parser.add_argument("--games", type=int, default=24, help="Number of stub game threads")
    args = parser.parse_args()

    results = []
    for k in [1, 2, 4, 8]:
        r = run_burst_trial(burst_k=k, num_games=args.games, duration_sec=args.duration)
        results.append(r)

    out_file = REPO_ROOT / "docs/perf/dispatcher_ceiling_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved dispatcher ceiling results to %s", out_file)


if __name__ == "__main__":
    main()
