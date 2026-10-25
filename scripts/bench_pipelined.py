"""Benchmark Pipelined vs Non-Pipelined Dispatcher at Concurrency 24 (Item 4).

Compares:
- Baseline: Non-pipelined (--pipelined OFF), 3 trials
- Candidate: Pipelined (--pipelined ON), 3 trials

Configuration:
- Concurrency 24, selfplay mode
- Warm-up: 30s, Counted: 60s per trial (total 90s per trial, well under 5-minute cap)
- Reports:
  * sims/s (mean, std, CV%)
  * avg batch size
  * queue wait time (ms)
  * GPU utilization (%)
  * Process RSS (MB)
  * Free RAM (MB)
- Evaluates Keep Rule: >= 5% gain beyond variance and no regression.
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
setup_logging(log_dir="logs", log_filename="bench_pipelined.log")
logger = logging.getLogger("bench_pipelined")


def run_pipelined_trial(
    pipelined: bool,
    trial_idx: int,
    warmup_sec: float = 30.0,
    counted_sec: float = 60.0,
    output_base: Path = Path("docs/perf/pipelined_benchmark"),
) -> Dict[str, Any]:
    mode_str = "pipelined_on" if pipelined else "pipelined_off"
    out_dir = output_base / f"{mode_str}_trial_{trial_idx}"
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
        "1.5",
        "--output-dir",
        str(out_dir),
    ]
    if pipelined:
        cmd.append("--pipelined")

    logger.info(
        ">>> Launching %s Trial %d (Warmup: %.0fs, Counted: %.0fs) <<<",
        mode_str.upper(),
        trial_idx,
        warmup_sec,
        counted_sec,
    )
    emit_report(
        event_line(
            "TRIAL", f"{mode_str} {trial_idx} warmup={warmup_sec:.0f}s counted={counted_sec:.0f}s"
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
        emit_report(event_line("TRIAL_FAIL", f"{mode_str} {trial_idx} rc={res.returncode}"))
        raise RuntimeError(f"Trial failed for {mode_str} trial {trial_idx}")

    summary_file = out_dir / "selfplay_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary not found at {summary_file}")

    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    agg = data.get("aggregated", {})
    t_sum = data.get("trial_summaries", [{}])[0]
    mem_end = t_sum.get("mem_end", {})
    gil_sum = t_sum.get("gil_summary", {})

    metrics = {
        "pipelined": pipelined,
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
        "RESULT (%s T%d): sims/s=%.1f | batch_size=%.2f | queue_wait=%.2fms | timeout_share=%.1f%% | GPU=%.1f%% | RSS=%.1fMB",
        mode_str,
        trial_idx,
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
                ("trial", f"{mode_str} {trial_idx}"),
                ("sims_sec", f"{metrics['sims_per_sec']:.1f}"),
                ("batch_avg", f"{metrics['avg_batch_size']:.2f}"),
                ("gpu", f"{metrics['gpu_util_pct']:.1f}%"),
            ]
        )
    )
    return metrics


def summarize_series(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    sims = [t["sims_per_sec"] for t in trials]
    batches = [t["avg_batch_size"] for t in trials]
    waits = [t["queue_wait_time_ms"] for t in trials]
    gpus = [t["gpu_util_pct"] for t in trials]
    rsss = [t["proc_rss_mb"] for t in trials]
    rams = [t["free_ram_mb"] for t in trials]

    return {
        "sims_mean": float(np.mean(sims)),
        "sims_std": float(np.std(sims)),
        "sims_cv_pct": float(np.std(sims) / np.mean(sims) * 100.0) if np.mean(sims) > 0 else 0.0,
        "batch_mean": float(np.mean(batches)),
        "wait_mean": float(np.mean(waits)),
        "gpu_mean": float(np.mean(gpus)),
        "rss_mean": float(np.mean(rsss)),
        "ram_mean": float(np.mean(rams)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--counted-sec", type=float, default=60.0)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print("=== BENCHMARKING PIPELINED DISPATCHER AT CONCURRENCY 24 (ITEM 4) ===")

    # 1. Baseline: OFF (3 trials)
    print("\n--- Running Baseline: Pipelined OFF (3 trials) ---")
    off_trials = []
    for idx in range(1, args.trials + 1):
        off_trials.append(
            run_pipelined_trial(
                pipelined=False,
                trial_idx=idx,
                warmup_sec=args.warmup_sec,
                counted_sec=args.counted_sec,
            )
        )

    # 2. Candidate: ON (3 trials)
    print("\n--- Running Candidate: Pipelined ON (3 trials) ---")
    on_trials = []
    for idx in range(1, args.trials + 1):
        on_trials.append(
            run_pipelined_trial(
                pipelined=True,
                trial_idx=idx,
                warmup_sec=args.warmup_sec,
                counted_sec=args.counted_sec,
            )
        )

    s_off = summarize_series(off_trials)
    s_on = summarize_series(on_trials)

    gain_pct = (s_on["sims_mean"] - s_off["sims_mean"]) / s_off["sims_mean"] * 100.0
    pooled_noise = (
        np.sqrt(s_off["sims_std"] ** 2 + s_on["sims_std"] ** 2) / s_off["sims_mean"] * 100.0
    )
    keep_rule_passed = gain_pct >= 5.0 and (gain_pct > pooled_noise)

    print("\n" + "=" * 95)
    print("PIPELINED DISPATCHER BENCHMARK COMPARISON TABLE (ITEM 4)")
    print("=" * 95)
    print(
        f"{'Metric':<25} | {'Pipelined OFF (Baseline)':<22} | {'Pipelined ON (Candidate)':<22} | {'Delta':<16}"
    )
    print("-" * 95)
    print(
        f"{'Sims/s':<25} | {s_off['sims_mean']:>10.1f} +/- {s_off['sims_std']:<6.1f} | {s_on['sims_mean']:>10.1f} +/- {s_on['sims_std']:<6.1f} | {gain_pct:>+6.2f}%"
    )
    print(
        f"{'Run-to-run CV':<25} | {s_off['sims_cv_pct']:>18.2f}% | {s_on['sims_cv_pct']:>18.2f}% | -"
    )
    print(
        f"{'Avg Batch Size':<25} | {s_off['batch_mean']:>22.2f} | {s_on['batch_mean']:>22.2f} | {s_on['batch_mean'] - s_off['batch_mean']:>+6.2f}"
    )
    print(
        f"{'Queue Wait Time (ms)':<25} | {s_off['wait_mean']:>20.2f}ms | {s_on['wait_mean']:>20.2f}ms | {s_on['wait_mean'] - s_off['wait_mean']:>+6.2f}ms"
    )
    print(
        f"{'GPU Utilization (%)':<25} | {s_off['gpu_mean']:>21.1f}% | {s_on['gpu_mean']:>21.1f}% | {s_on['gpu_mean'] - s_off['gpu_mean']:>+6.1f}%"
    )
    print(
        f"{'Process RSS (MB)':<25} | {s_off['rss_mean']:>19.1f} MB | {s_on['rss_mean']:>19.1f} MB | {s_on['rss_mean'] - s_off['rss_mean']:>+6.1f} MB"
    )
    print(
        f"{'Free RAM (MB)':<25} | {s_off['ram_mean']:>19.1f} MB | {s_on['ram_mean']:>19.1f} MB | {s_on['ram_mean'] - s_off['ram_mean']:>+6.1f} MB"
    )
    print("-" * 95)
    print(f"KEEP RULE (>= 5% gain beyond variance): {'PASSED' if keep_rule_passed else 'NOT MET'}")
    print(f"Gain: {gain_pct:+.2f}%, Combined Uncertainty: +/-{pooled_noise:.2f}%")
    print("=" * 95 + "\n")

    report = {
        "baseline_off": {"trials": off_trials, "summary": s_off},
        "candidate_on": {"trials": on_trials, "summary": s_on},
        "comparison": {
            "gain_pct": gain_pct,
            "pooled_noise_pct": pooled_noise,
            "keep_rule_passed": keep_rule_passed,
        },
    }
    out_file = REPO_ROOT / "docs/perf/pipelined_benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved pipelined benchmark report to %s", out_file)


if __name__ == "__main__":
    main()
