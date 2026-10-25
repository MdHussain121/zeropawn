"""Hardware pilot benchmarking script for ZeroPawn on RTX 4060 GPU + i7-12650H."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from python.model.chess_net import ChessNet
from python.data.dataset import BinaryChessDataset


def benchmark_inference(
    model: nn.Module, device: torch.device, use_bfloat16: bool = True
) -> dict[str, float]:
    model.eval()
    batch_1 = torch.randn(1, 19, 8, 8, device=device)
    batch_64 = torch.randn(64, 19, 8, 8, device=device)

    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                _ = model(batch_1)
                _ = model(batch_64)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Latency single item (batch size 1)
    iters = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                _ = model(batch_1)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t1 = time.perf_counter()
    latency_b1_ms = ((t1 - t0) / iters) * 1000.0

    # Batched throughput (batch size 64)
    iters_b64 = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters_b64):
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                _ = model(batch_64)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t1 = time.perf_counter()
    sps_b64 = (iters_b64 * 64) / (t1 - t0)

    return {
        "latency_b1_ms": latency_b1_ms,
        "throughput_b64_sps": sps_b64,
    }


def benchmark_training_step(
    micro_batch_size: int,
    accum_steps: int,
    use_bfloat16: bool = True,
    device_str: str = "cuda",
) -> dict[str, float]:
    device = torch.device(
        device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)

    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    # Synthetic batches
    planes = torch.randn(micro_batch_size, 19, 8, 8, device=device)
    moves = torch.randint(0, 4672, (micro_batch_size,), device=device)
    outcomes = torch.randint(0, 3, (micro_batch_size,), device=device)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        for _ in range(accum_steps):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda" and use_bfloat16),
            ):
                pol_logits, val_logits = model(planes)
                l_pol = F.cross_entropy(pol_logits, moves)
                l_val = F.cross_entropy(val_logits, outcomes)
                loss = (l_pol + 0.5 * l_val) / accum_steps
            loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measured steps
    num_effective_steps = 10
    t0 = time.perf_counter()
    for _ in range(num_effective_steps):
        optimizer.zero_grad()
        for _ in range(accum_steps):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda" and use_bfloat16),
            ):
                pol_logits, val_logits = model(planes)
                l_pol = F.cross_entropy(pol_logits, moves)
                l_val = F.cross_entropy(val_logits, outcomes)
                loss = (l_pol + 0.5 * l_val) / accum_steps
            loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    dur = t1 - t0
    total_samples = num_effective_steps * micro_batch_size * accum_steps
    sps = total_samples / dur

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "micro_batch_size": micro_batch_size,
        "accum_steps": accum_steps,
        "effective_batch_size": micro_batch_size * accum_steps,
        "precision": "bfloat16" if (use_bfloat16 and device.type == "cuda") else "float32",
        "device": device.type,
        "throughput_sps": sps,
        "step_time_ms": (dur / num_effective_steps) * 1000.0,
        "peak_vram_mb": peak_vram_mb,
    }


def run_pilot():
    print("=" * 70)
    print("ZeroPawn Hardware Pilot Benchmark (RTX 4060 8GB + i7-12650H)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672).to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count * 4 / (1024 * 1024):.2f} MB float32)")

    # 1. Inference benchmark
    print("\n--- 1. Inference Latency & Throughput ---")
    inf_bf16 = benchmark_inference(model, device, use_bfloat16=True)
    print(f"bfloat16 Batch=1 Latency:    {inf_bf16['latency_b1_ms']:.2f} ms")
    print(f"bfloat16 Batch=64 Throughput: {inf_bf16['throughput_b64_sps']:.1f} samples/sec")

    inf_fp32 = benchmark_inference(model, device, use_bfloat16=False)
    print(f"float32  Batch=1 Latency:    {inf_fp32['latency_b1_ms']:.2f} ms")
    print(f"float32  Batch=64 Throughput: {inf_fp32['throughput_b64_sps']:.1f} samples/sec")

    # 2. Training configurations
    print("\n--- 2. Training Configuration Sweep (Effective Batch Size = 512) ---")
    configs = [
        (64, 8, True),  # micro 64, accum 8, bfloat16
        (128, 4, True),  # micro 128, accum 4, bfloat16
        (256, 2, True),  # micro 256, accum 2, bfloat16
        (64, 8, False),  # micro 64, accum 8, float32
    ]

    results = []
    for micro_bs, accum, bf16 in configs:
        res = benchmark_training_step(micro_bs, accum, use_bfloat16=bf16, device_str="cuda")
        results.append(res)
        print(
            f"Config micro={micro_bs:3d}, accum={accum:2d}, {res['precision']:8s} -> "
            f"Throughput: {res['throughput_sps']:6.1f} sps | "
            f"Step: {res['step_time_ms']:6.1f} ms | "
            f"Peak VRAM: {res['peak_vram_mb']:6.1f} MB"
        )

    # 3. CPU Baseline check
    print("\n--- 3. CPU Training Step (micro=32, accum=2) ---")
    cpu_res = benchmark_training_step(32, 2, use_bfloat16=False, device_str="cpu")
    print(
        f"CPU Baseline -> Throughput: {cpu_res['throughput_sps']:.1f} sps | Step: {cpu_res['step_time_ms']:.1f} ms"
    )

    # Save benchmark report artifact
    os.makedirs("reports", exist_ok=True)
    out_path = Path("reports/hardware_pilot_benchmark.json")
    summary = {
        "system": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if device.type == "cuda"
            else 0.0,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        },
        "model": {
            "parameters": param_count,
            "architecture": "12 ResBlocks x 128 Filters",
        },
        "inference": {
            "bfloat16": inf_bf16,
            "float32": inf_fp32,
        },
        "training_sweep": results,
        "cpu_baseline": cpu_res,
        "recommended_config": {
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 512,
            "precision": "bfloat16",
            "vram_headroom_gb": (8.0 - (results[0]["peak_vram_mb"] / 1024.0)),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPilot benchmark report written to {out_path}")


if __name__ == "__main__":
    run_pilot()
