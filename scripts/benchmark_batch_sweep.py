"""ZeroPawn Stage D.1 Extended Batch-Size Sweep (B=1 to 256).

Measures:
1. Pure Neural Inference Latency & Throughput across B=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256].
2. Single-Tree MCTS Search Throughput, Wall-clock time, Best Action, Root Value, and Completed Backups.
3. System telemetry: VRAM (allocated and reserved), RAM, and CPU utilization.

Outputs: docs/perf/BATCH_SWEEP_EXTENDED.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # ensures DLL loading
import zeropawn_cpp
from python.hardware.detector import detect_hardware
from python.model.chess_net import ChessNet


def get_system_metrics() -> Dict[str, float]:
    vm = psutil.virtual_memory()
    res = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((vm.total - vm.available) / 1e9, 3),
        "ram_free_gb": round(vm.available / 1e9, 3),
        "ram_percent": vm.percent,
    }
    if torch.cuda.is_available():
        res["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        res["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        res["vram_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return res


def benchmark_pure_inference(
    model: ChessNet,
    device: torch.device,
    batch_sizes: List[int],
    num_warmup: int = 25,
    num_trials: int = 80,
) -> Dict[str, Any]:
    print("\n--- Part 1: Pure CUDA Neural Inference Sweep ---")
    results = {}

    for b in batch_sizes:
        x = torch.randn(b, 19, 8, 8, dtype=torch.float32, device=device)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)

        # Warmup
        for _ in range(num_warmup):
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(num_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        med_ms = float(np.median(times_ms))
        p95_ms = float(np.percentile(times_ms, 95))
        evals_per_sec = float(b / (med_ms / 1000.0))
        print(
            f"Batch {b:3d}: Median {med_ms:6.2f} ms | P95 {p95_ms:6.2f} ms | {evals_per_sec:8.1f} evals/s"
        )

        results[f"B_{b}"] = {
            "batch_size": b,
            "median_latency_ms": round(med_ms, 3),
            "p95_latency_ms": round(p95_ms, 3),
            "throughput_evals_sec": round(evals_per_sec, 1),
        }

    return results


def benchmark_mcts_search_sweep(
    model: ChessNet,
    device: torch.device,
    batch_sizes: List[int],
    num_simulations: int = 400,
    repeats: int = 3,
) -> Dict[str, Any]:
    print(f"\n--- Part 2: Single-Tree MCTS Search Sweep ({num_simulations} simulations/search) ---")

    def serial_eval(game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(device, non_blocking=True)
        if device.type == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

    def batched_eval(batch_planes: np.ndarray):
        n = int(batch_planes.shape[0])
        if n == 0:
            return (np.zeros((0, 4672), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32))
        t = t.view(n, 19, 8, 8).to(device, non_blocking=True)
        if device.type == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p_logits, v_wdl = model(t)
            policy = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            wdl = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)
        return policy.astype(np.float32), values

    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    cfg.fpu_reduction = 0.2
    cfg.virtual_loss = 1
    mcts = zeropawn_cpp.MCTS(cfg)

    # Serial Baseline
    print("Measuring Serial Search Baseline...")
    serial_times = []
    serial_actions = []
    serial_values = []
    for rep in range(repeats):
        game = zeropawn_cpp.ChessGame()
        mcts.clear()
        mcts.set_seed(42 + rep)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = mcts.search(game, num_simulations, serial_eval)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        serial_times.append((t1 - t0) * 1000.0)
        serial_actions.append(game.action_to_string(res.best_action))
        serial_values.append(res.root_value)

    med_ser_ms = float(np.median(serial_times))
    ser_sims_per_sec = float(num_simulations / (med_ser_ms / 1000.0))
    print(
        f"Serial Baseline: {med_ser_ms:6.1f} ms | {ser_sims_per_sec:6.1f} sims/s | Action: {serial_actions[0]} | Val: {serial_values[0]:.4f}"
    )

    results = {
        "num_simulations": num_simulations,
        "serial_baseline": {
            "median_ms": round(med_ser_ms, 2),
            "sims_per_sec": round(ser_sims_per_sec, 1),
            "best_action": serial_actions[0],
            "root_value": round(float(serial_values[0]), 4),
        },
        "batched_sweep": {},
    }

    for b in batch_sizes:
        b_times = []
        b_actions = []
        b_values = []
        vram_start = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for rep in range(repeats):
            game = zeropawn_cpp.ChessGame()
            mcts.clear()
            mcts.set_seed(42 + rep)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = mcts.search_batched(game, num_simulations, b, batched_eval)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            b_times.append((t1 - t0) * 1000.0)
            b_actions.append(game.action_to_string(res.best_action))
            b_values.append(res.root_value)

        med_b_ms = float(np.median(b_times))
        sims_sec = float(num_simulations / (med_b_ms / 1000.0))
        speedup = sims_sec / max(1e-4, ser_sims_per_sec)
        peak_vram_mb = (
            (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
        )

        print(
            f"Batch {b:3d}: {med_b_ms:6.1f} ms | {sims_sec:7.1f} sims/s | Speedup: {speedup:5.2f}x | Action: {b_actions[0]} | Val: {b_values[0]:.4f} | Peak VRAM: {peak_vram_mb:5.1f} MB"
        )

        results["batched_sweep"][f"B_{b}"] = {
            "batch_size": b,
            "median_ms": round(med_b_ms, 2),
            "sims_per_sec": round(sims_sec, 1),
            "speedup_vs_serial": round(speedup, 2),
            "best_action": b_actions[0],
            "root_value": round(float(b_values[0]), 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="ZeroPawn Extended Batch Sweep")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,12,16,24,32,48,64,96,128,192,256",
        help="Comma-separated batch sizes to sweep",
    )
    parser.add_argument(
        "--sims", type=int, default=400, help="Simulations per tested batch size (default: 400)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints_scaled/best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="docs/perf/BATCH_SWEEP_EXTENDED.json")
    args = parser.parse_args()

    hw = detect_hardware()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"=== ZeroPawn Extended Batch Sweep on {device} ({hw.gpu_name if hw.cuda_available else 'CPU'}) ==="
    )
    print(
        f"Host: {hw.cpu_physical_cores} physical / {hw.cpu_logical_threads} logical threads, {hw.system_total_ram_gb:.1f} GB RAM"
    )

    model = ChessNet(num_blocks=12, num_filters=128)
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
            model.load_state_dict(sd)
            print(f"Loaded checkpoint from: {args.checkpoint}")
        except Exception as e:
            print(f"Warning: Failed loading checkpoint ({e}); using randomized weights")
    model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    try:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    except Exception:
        batch_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

    report: Dict[str, Any] = {
        "hardware": hw.to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tested_batch_sizes": batch_sizes,
        "system_metrics_start": get_system_metrics(),
    }

    # 1. Pure neural inference
    report["pure_inference"] = benchmark_pure_inference(model, device, batch_sizes)

    # 2. End-to-End MCTS Search
    report[f"mcts_search_{args.sims}sims"] = benchmark_mcts_search_sweep(
        model, device, batch_sizes, num_simulations=args.sims
    )

    report["system_metrics_end"] = get_system_metrics()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Extended Batch Sweep Complete. Report saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
