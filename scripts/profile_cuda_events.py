"""Follow-up 1: CUDA Event Timing for Forward Inference with Clocks and Power Logging."""

import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.model.chess_net import ChessNet


def query_gpu_status():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, encoding="utf-8", timeout=3
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "gpu_clock_mhz": float(parts[0]),
            "mem_clock_mhz": float(parts[1]),
            "power_w": float(parts[2]),
            "temp_c": float(parts[3]),
            "pstate": parts[4],
        }
    except Exception:
        return {
            "gpu_clock_mhz": 0.0,
            "mem_clock_mhz": 0.0,
            "power_w": 0.0,
            "temp_c": 0.0,
            "pstate": "N/A",
        }


def benchmark_forward_cuda_events():
    device = torch.device("cuda")
    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    model.to(memory_format=torch.channels_last)
    model.eval()

    batch_sizes = [1, 4, 16, 64, 128]
    warmup_iters = 50
    timed_iters = 500

    results = {}

    print(
        f"{'Batch':>6} | {'Mean (ms)':>10} | {'p95 (ms)':>10} | {'Clock (MHz)':>12} | {'Mem Clock (MHz)':>16} | {'Power (W)':>10} | {'P-State':>8}"
    )
    print("-" * 85)

    for b in batch_sizes:
        # Create input tensor in channels-last and bfloat16
        x = torch.randn(b, 19, 8, 8, device=device, dtype=torch.float32)
        x = x.to(memory_format=torch.channels_last)

        # Warm-up iterations
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(warmup_iters):
                    _ = model(x)
        torch.cuda.synchronize()

        times_ms = []
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]

        gpu_status_before = query_gpu_status()

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for i in range(timed_iters):
                    start_events[i].record()
                    _ = model(x)
                    end_events[i].record()

        torch.cuda.synchronize()
        gpu_status_after = query_gpu_status()

        for i in range(timed_iters):
            times_ms.append(start_events[i].elapsed_time(end_events[i]))

        times_arr = np.array(times_ms)
        mean_ms = float(np.mean(times_arr))
        p95_ms = float(np.percentile(times_arr, 95))
        clock_mhz = (gpu_status_before["gpu_clock_mhz"] + gpu_status_after["gpu_clock_mhz"]) / 2.0
        mem_clock_mhz = (
            gpu_status_before["mem_clock_mhz"] + gpu_status_after["mem_clock_mhz"]
        ) / 2.0
        power_w = (gpu_status_before["power_w"] + gpu_status_after["power_w"]) / 2.0
        pstate = gpu_status_after["pstate"]

        results[b] = {
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "per_item_ms": mean_ms / b,
            "gpu_clock_mhz": clock_mhz,
            "mem_clock_mhz": mem_clock_mhz,
            "power_w": power_w,
            "pstate": pstate,
        }

        print(
            f"{b:>6} | {mean_ms:>10.3f} | {p95_ms:>10.3f} | {clock_mhz:>12.0f} | {mem_clock_mhz:>16.0f} | {power_w:>10.1f} | {pstate:>8}"
        )

    return results


if __name__ == "__main__":
    benchmark_forward_cuda_events()
