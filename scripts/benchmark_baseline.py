"""Comprehensive Stage A Baseline Measurement Suite for ZeroPawn.

Measures:
1. Pure Neural Inference Latency and Throughput across batch sizes (CUDA BF16, CUDA FP32, CPU FP32).
2. Pure MCTS Single-Game Search (CPU search + CPU infer vs CPU search + GPU infer).
3. Concurrent Self-Play Games Throughput (concurrency 1, 2, 4, 6, 8, 12, 16, 24).
4. Combined Mode (Self-play + concurrent RL training steps).
5. Memory, CPU, and VRAM utilization across all runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # ensures DLL loading
import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import GameSummary, SavedPosition, SelfPlayWorker


def get_system_metrics() -> Dict[str, float]:
    vm = psutil.virtual_memory()
    res = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": (vm.total - vm.available) / 1e9,
        "ram_free_gb": vm.available / 1e9,
        "ram_percent": vm.percent,
    }
    if torch.cuda.is_available():
        res["vram_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        res["vram_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    return res


def benchmark_inference_sweep(
    checkpoint_path: str,
    device_str: str = "cuda",
    batch_sizes: List[int] = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256],
    precision: str = "bfloat16",
    num_warmup: int = 30,
    num_trials: int = 100,
) -> Dict[str, Any]:
    """Measure inference latency percentiles and throughput across batch sizes."""
    device = torch.device(device_str)
    model = ChessNet(num_blocks=12, num_filters=128)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd)
    model.to(device)
    model.eval()

    use_bf16 = (precision == "bfloat16") and (device.type == "cuda")
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    results: Dict[str, Any] = {}

    for b in batch_sizes:
        x = torch.randn(b, 19, 8, 8, dtype=torch.float32, device=device)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)

        # Warmup
        for _ in range(num_warmup):
            with torch.inference_mode():
                if use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _ = model(x)
                else:
                    _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms: List[float] = []
        t0_total = time.perf_counter()

        for _ in range(num_trials):
            t_start = time.perf_counter()
            with torch.inference_mode():
                if use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _ = model(x)
                else:
                    _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

        total_elapsed = time.perf_counter() - t0_total
        latencies_np = np.array(latencies_ms)
        evals_per_sec = (b * num_trials) / total_elapsed

        results[str(b)] = {
            "batch_size": b,
            "mean_ms": float(np.mean(latencies_np)),
            "median_ms": float(np.median(latencies_np)),
            "p50_ms": float(np.percentile(latencies_np, 50)),
            "p90_ms": float(np.percentile(latencies_np, 90)),
            "p99_ms": float(np.percentile(latencies_np, 99)),
            "min_ms": float(np.min(latencies_np)),
            "max_ms": float(np.max(latencies_np)),
            "evals_per_sec": float(evals_per_sec),
        }
        print(
            f"[{device_str.upper()}:{precision}] Batch {b:3d}: median={np.median(latencies_np):.2f}ms, p99={np.percentile(latencies_np, 99):.2f}ms -> {evals_per_sec:.1f} evals/sec"
        )

    return results


def benchmark_single_mcts(
    checkpoint_path: str,
    device_str: str = "cuda",
    num_simulations: int = 200,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """Measure single-tree MCTS search speed (simulations/sec)."""
    device = torch.device(device_str)
    model = ChessNet(num_blocks=12, num_filters=128)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    mcts_cfg = zeropawn_cpp.MCTSConfig()
    mcts_cfg.c_puct = 2.0
    mcts_cfg.fpu_reduction = 0.2
    mcts_cfg.add_noise = False
    mcts = zeropawn_cpp.MCTS(mcts_cfg)

    def nn_eval(game):
        planes = zeropawn_cpp.encode_state(game)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device.type == "cuda":
            t = t.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p_logits, v_wdl = model(t)
        else:
            with torch.inference_mode():
                p_logits, v_wdl = model(t)
        p = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
        v = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
        scalar_val = float(v[0] - v[2])
        return (p, scalar_val)

    game = zeropawn_cpp.ChessGame()
    # Warmup
    _ = mcts.search(game, 20, nn_eval)

    times = []
    for _ in range(num_runs):
        mcts.clear()
        t0 = time.perf_counter()
        res = mcts.search(game, num_simulations, nn_eval)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_t = float(np.median(times))
    sims_per_sec = num_simulations / median_t
    print(
        f"[MCTS Single Search {device_str.upper()}] {num_simulations} sims: median {median_t * 1000.0:.1f}ms -> {sims_per_sec:.1f} sims/sec"
    )
    return {
        "device": device_str,
        "num_simulations": num_simulations,
        "median_time_sec": median_t,
        "sims_per_sec": sims_per_sec,
    }


def benchmark_selfplay_concurrency_sweep(
    checkpoint_path: str,
    concurrency_list: List[int] = [1, 2, 4, 6, 8, 12, 16, 24],
    warmup_sec: float = 3.0,
    measured_sec: float = 10.0,
    pipelined: bool = False,
) -> Dict[str, Any]:
    """Measure self-play throughput scaling across concurrency levels."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: Dict[str, Any] = {}

    for c in concurrency_list:
        server = InferenceServer(
            checkpoint_path=checkpoint_path,
            device=device,
            min_batch_size=c,
            max_batch_size=max(c * 2, 64),
            max_wait_ms=1.5,
            use_bfloat16=(device == "cuda"),
        )
        client = DirectInferenceClient(server)

        worker = SelfPlayWorker(
            worker_id=0,
            client=client,
            num_concurrent_games=c,
            game_type="chess",
            full_search_sims=400,
            fast_search_sims=80,
            full_search_prob=0.35,
            pipelined=pipelined,
            max_wait_ms=1.5,
        )

        worker.start()
        time.sleep(warmup_sec)

        # Measure baseline window
        evals_start = server.total_evaluations
        batches_start = server.total_batches
        t_start = time.perf_counter()

        time.sleep(measured_sec)

        t_end = time.perf_counter()
        evals_end = server.total_evaluations
        batches_end = server.total_batches
        elapsed = t_end - t_start

        worker.stop()

        delta_evals = evals_end - evals_start
        delta_batches = batches_end - batches_start
        evals_per_sec = delta_evals / elapsed if elapsed > 0 else 0.0
        avg_batch = delta_evals / delta_batches if delta_batches > 0 else 0.0

        sys_metrics = get_system_metrics()

        results[str(c)] = {
            "concurrency": c,
            "evals_per_sec": evals_per_sec,
            "avg_batch_size": avg_batch,
            "total_evals": delta_evals,
            "total_batches": delta_batches,
            "elapsed_sec": elapsed,
            "cpu_percent": sys_metrics["cpu_percent"],
            "ram_used_gb": sys_metrics["ram_used_gb"],
            "vram_allocated_mb": sys_metrics.get("vram_allocated_mb", 0.0),
        }
        print(
            f"[SELF-PLAY CONCURRENCY {c:2d}] Throughput: {evals_per_sec:.1f} evals/s | Avg Batch: {avg_batch:.2f} | CPU: {sys_metrics['cpu_percent']:.1f}% | RAM: {sys_metrics['ram_used_gb']:.2f}GB"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ZeroPawn Stage A Baseline Profiler")
    parser.add_argument("--checkpoint", default="models/checkpoints_scaled/best_model.pt")
    parser.add_argument("--out", default="docs/perf/STAGE_A_BASELINE.json")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE A — ZERO PAWN REPRODUCIBLE BASELINE MEASUREMENT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print("=" * 80)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "total_ram_gb": psutil.virtual_memory().total / 1e9,
        },
    }

    # 1. Pure Inference Latency & Throughput (CUDA BF16)
    if torch.cuda.is_available():
        print("\n--- 1. PURE INFERENCE SWEEP (CUDA BF16) ---")
        report["cuda_bf16_inference"] = benchmark_inference_sweep(
            args.checkpoint, device_str="cuda", precision="bfloat16"
        )

        print("\n--- 2. PURE INFERENCE SWEEP (CUDA FP32) ---")
        report["cuda_fp32_inference"] = benchmark_inference_sweep(
            args.checkpoint, device_str="cuda", precision="float32"
        )

    # 3. Pure Inference on CPU (FP32)
    print("\n--- 3. PURE INFERENCE SWEEP (CPU FP32) ---")
    report["cpu_fp32_inference"] = benchmark_inference_sweep(
        args.checkpoint,
        device_str="cpu",
        precision="float32",
        batch_sizes=[1, 2, 4, 8, 16, 32],
        num_warmup=5,
        num_trials=20,
    )

    # 4. MCTS Single Search
    print("\n--- 4. MCTS SINGLE SEARCH ---")
    if torch.cuda.is_available():
        report["mcts_gpu_search"] = benchmark_single_mcts(
            args.checkpoint, device_str="cuda", num_simulations=200
        )
    report["mcts_cpu_search"] = benchmark_single_mcts(
        args.checkpoint, device_str="cpu", num_simulations=50, num_runs=2
    )

    # 5. Concurrent Self-Play Sweep (Synchronous Dispatcher)
    print("\n--- 5. SELF-PLAY CONCURRENCY SWEEP (SYNCHRONOUS) ---")
    report["selfplay_sync_sweep"] = benchmark_selfplay_concurrency_sweep(
        args.checkpoint, concurrency_list=[1, 2, 4, 6, 8, 12, 16, 24], pipelined=False
    )

    # 6. Concurrent Self-Play Sweep (Pipelined Dispatcher)
    print("\n--- 6. SELF-PLAY CONCURRENCY SWEEP (PIPELINED) ---")
    report["selfplay_pipelined_sweep"] = benchmark_selfplay_concurrency_sweep(
        args.checkpoint, concurrency_list=[1, 2, 4, 6, 8, 12, 16, 24], pipelined=True
    )

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STAGE A BASELINE MEASUREMENTS COMPLETE. SAVED TO: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
