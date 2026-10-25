"""Item 3: Dispatcher Timeout Sweep at Concurrency 24.

Screens timeout values: 0.25, 0.5, 1.0, 1.5, 3.0 ms with 1 run each.
Window protocol: 3m warm-up (180s) + 6m counted (360s) = 540s max.
Identifies top 2 configurations and re-runs each 3 times.
Reports:
- sims/s
- avg batch size
- queue wait time (ms)
- timeout share (%)
- GPU utilization (%)
- Process RSS (MB)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.cli.common import setup_logging
from python.common.progress import emit_report, event_line, format_kv_line

# Single logging setup: rotating file for everything, WARNING+ on stderr.
# (No logging.basicConfig: one setup path, no duplicate lines.)
setup_logging(log_dir="logs", log_filename="sweep_dispatcher_timeout.log")
logger = logging.getLogger("sweep_timeout")


def run_single_timeout_trial(
    timeout_ms: float,
    trial_idx: int,
    warmup_sec: float = 60.0,
    counted_sec: float = 180.0,
    output_base: Path = Path("docs/perf/timeout_sweep"),
) -> Dict[str, Any]:
    out_dir = output_base / f"t_{timeout_ms}ms_trial_{trial_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/bench_selfplay.py"),
        "--mode",
        "selfplay",
        "--concurrency",
        "24",
        "--warmup-sec",
        str(warmup_sec),
        "--counted-sec",
        str(counted_sec),
        "--trials",
        "1",
        "--max-wait-ms",
        str(timeout_ms),
        "--output-dir",
        str(out_dir),
    ]

    logger.info(
        ">>> Launching Trial: timeout=%.2f ms, idx=%d (Warmup: %.0fs, Counted: %.0fs) <<<",
        timeout_ms,
        trial_idx,
        warmup_sec,
        counted_sec,
    )
    emit_report(
        event_line(
            "TRIAL",
            f"timeout={timeout_ms:.2f}ms idx={trial_idx} "
            f"warmup={warmup_sec:.0f}s counted={counted_sec:.0f}s",
        )
    )
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0

    if res.returncode != 0:
        logger.error(
            "Trial failed with return code %d:\n%s",
            res.returncode,
            res.stderr[-1000:] if res.stderr else "",
        )
        emit_report(event_line("TRIAL_FAIL", f"timeout={timeout_ms:.2f}ms rc={res.returncode}"))
        raise RuntimeError(f"Trial failed for timeout={timeout_ms}ms")

    # Parse selfplay_summary.json
    summary_file = out_dir / "selfplay_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary not found at {summary_file}")

    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    agg = data.get("aggregated", {})
    t_sum = data.get("trial_summaries", [{}])[0]
    mem_end = t_sum.get("mem_end", {})
    gil_sum = t_sum.get("gil_summary", {})

    # Extract metrics
    metrics = {
        "timeout_ms": timeout_ms,
        "trial_idx": trial_idx,
        "elapsed_sec": round(elapsed, 1),
        "sims_per_sec": agg.get("sims_per_sec", {}).get("mean", 0.0),
        "avg_batch_size": agg.get("avg_batch_size", {}).get("mean", 0.0),
        "queue_wait_time_ms": agg.get("queue_wait_time_ms", {}).get("mean", 0.0),
        "gpu_util_pct": agg.get("gpu_util_pct", {}).get("mean", 0.0),
        "free_ram_mb": agg.get("free_ram_mb", {}).get("mean", 0.0),
        "proc_rss_mb": mem_end.get("proc_rss_mb", 0.0),
        "timeout_share_pct": gil_sum.get("timeout_share_pct", 0.0),
    }
    logger.info(
        "RESULT: timeout=%.2fms | sims/s=%.1f | batch_size=%.2f | queue_wait=%.2fms | timeout_share=%.1f%% | GPU=%.1f%% | RSS=%.1fMB",
        timeout_ms,
        metrics["sims_per_sec"],
        metrics["avg_batch_size"],
        metrics["queue_wait_time_ms"],
        metrics["timeout_share_pct"],
        metrics["gpu_util_pct"],
        metrics["proc_rss_mb"],
    )
    emit_report(
        format_kv_line(
            [
                ("trial", f"timeout={timeout_ms:.2f}ms idx={trial_idx}"),
                ("sims_sec", f"{metrics['sims_per_sec']:.1f}"),
                ("batch_avg", f"{metrics['avg_batch_size']:.2f}"),
                ("gpu", f"{metrics['gpu_util_pct']:.1f}%"),
            ]
        )
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--counted-sec", type=float, default=180.0)
    parser.add_argument("--screen-only", action="store_true", default=False)
    args = parser.parse_args()

    timeouts = [0.25, 0.5, 1.0, 1.5, 3.0]
    screening_results = []

    logger.info("=== PHASE 1: SCREENING 5 TIMEOUT CONFIGURATIONS (1 run each) ===")
    for t_ms in timeouts:
        r = run_single_timeout_trial(
            timeout_ms=t_ms,
            trial_idx=1,
            warmup_sec=args.warmup_sec,
            counted_sec=args.counted_sec,
        )
        screening_results.append(r)

    # Sort by sims/s descending
    screening_results.sort(key=lambda x: x["sims_per_sec"], reverse=True)
    top2 = [screening_results[0]["timeout_ms"], screening_results[1]["timeout_ms"]]
    logger.info(
        "Top 2 configurations: %.2f ms (%.1f sims/s) and %.2f ms (%.1f sims/s)",
        top2[0],
        screening_results[0]["sims_per_sec"],
        top2[1],
        screening_results[1]["sims_per_sec"],
    )

    print("\n" + "=" * 95)
    print("TIMEOUT SWEEP SCREENING RESULTS (CONCURRENCY 24)")
    print("=" * 95)
    print(
        f"{'Timeout (ms)':<14} | {'Sims/s':<10} | {'Batch Size':<12} | {'Queue Wait':<12} | {'Timeout %':<11} | {'GPU %':<8} | {'RSS (MB)':<10}"
    )
    print("-" * 95)
    for r in screening_results:
        print(
            f"{r['timeout_ms']:<14.2f} | {r['sims_per_sec']:<10.1f} | {r['avg_batch_size']:<12.2f} | {r['queue_wait_time_ms']:<12.2f} | {r['timeout_share_pct']:<11.1f} | {r['gpu_util_pct']:<8.1f} | {r['proc_rss_mb']:<10.1f}"
        )
    print("=" * 95 + "\n")

    out_file = REPO_ROOT / "docs/perf/timeout_sweep_report.json"
    if args.screen_only:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"screening": screening_results}, f, indent=2)
        logger.info("Saved screening report to %s", out_file)
        return

    logger.info("=== PHASE 2: CONFIRMING TOP 2 CONFIGURATIONS (3 runs each) ===")
    confirmation_results = {}
    for t_ms in top2:
        trials = []
        for idx in range(1, 4):
            r = run_single_timeout_trial(
                timeout_ms=t_ms,
                trial_idx=idx,
                warmup_sec=args.warmup_sec,
                counted_sec=args.counted_sec,
            )
            trials.append(r)
        confirmation_results[str(t_ms)] = trials

    final_report = {
        "screening": screening_results,
        "top2_confirmed": confirmation_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Saved final report to %s", out_file)


if __name__ == "__main__":
    main()
