"""Paired interleaved A/B benchmark: sync vs pipelined self-play dispatch.

Implements the audit's §5.4 measurement rule: A-B-A-B pairs (A = sync), each
window warmup + counted seconds, telemetry from both arms, and a GPU pre-flight
that aborts when a foreign compute app shares the device (the 2026-10-27
contaminated-trial incident). Reads each window's {mode}_summary.json written
by scripts/bench_selfplay.py.

Usage:
    python scripts/bench_pipelined_paired.py --pairs 3 --warmup-sec 60 --counted-sec 90
"""

from __future__ import annotations

import argparse
import json
import subprocess
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = r"C:/Users/<user>/Music/chess/chess-rl/models/checkpoints_scaled/latest_checkpoint.pt"


def gpu_busy_with_foreign_process() -> bool:
    """True when another process holds CUDA (our own windows run sequentially)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        pids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return bool(pids)
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_window(args, arm: str, pair_idx: int, root: Path) -> dict:
    out_dir = root / f"pair{pair_idx}_{arm}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bench_selfplay.py"),
        "--mode", "selfplay",
        "--warmup-sec", str(args.warmup_sec),
        "--counted-sec", str(args.counted_sec),
        "--trials", "1",
        "--concurrency", str(args.concurrency),
        "--full-sims", str(args.full_sims),
        "--fast-sims", str(args.fast_sims),
        "--full-search-prob", "0.35",
        "--max-wait-ms", "1.5",
        "--checkpoint-path", args.checkpoint_path,
        "--output-dir", str(out_dir),
        "--base-seed", str(args.base_seed + pair_idx),
    ]
    if arm == "pipelined":
        cmd.append("--pipelined")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    summary_path = out_dir / "selfplay_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"window {arm} pair{pair_idx} produced no summary (rc={proc.returncode});\n"
            f"stderr tail: {proc.stderr[-800:]}"
        )
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    trials = summary.get("trials") or [summary]
    return trials[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--counted-sec", type=float, default=90.0)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--full-sims", type=int, default=400)
    parser.add_argument("--fast-sims", type=int, default=80)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=str, default="tmp_bench/pipelined_paired")
    args = parser.parse_args()

    if gpu_busy_with_foreign_process():
        print("ABORT: GPU has an active compute app; paired windows would be contaminated.")
        return 2

    root = REPO_ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)

    sync_sps, pipe_sps = [], []
    rows = []
    print(
        f"paired A/B: {args.pairs} pairs x (warmup {args.warmup_sec:.0f}s + counted "
        f"{args.counted_sec:.0f}s), C={args.concurrency}, checkpoint={args.checkpoint_path}"
    )
    for i in range(1, args.pairs + 1):
        s = run_window(args, "sync", i, root)
        p = run_window(args, "pipelined", i, root)
        sync_sps.append(s["sims_per_sec"])
        pipe_sps.append(p["sims_per_sec"])
        rows.append((s, p))
        print(
            f"pair {i}: sync {s['sims_per_sec']:.0f} sims/s vs pipelined "
            f"{p['sims_per_sec']:.0f} sims/s "
            f"(batch {s.get('avg_batch_size', 0):.1f}/{p.get('avg_batch_size', 0):.1f}, "
            f"queue {s.get('queue_wait_time_ms', 0):.1f}/{p.get('queue_wait_time_ms', 0):.1f} ms, "
            f"GPU {s.get('gpu_util_pct', 0):.0f}%/{p.get('gpu_util_pct', 0):.0f}%)"
        )

    s_med, p_med = statistics.median(sync_sps), statistics.median(pipe_sps)
    deltas = [(p / s - 1) * 100 for s, p in zip(sync_sps, pipe_sps)]
    print(f"sync:      median {s_med:.0f} sims/s  runs {[f'{x:.0f}' for x in sync_sps]}")
    print(f"pipelined: median {p_med:.0f} sims/s  runs {[f'{x:.0f}' for x in pipe_sps]}")
    print(f"per-pair deltas: {[f'{d:+.1f}%' for d in deltas]}")
    print(f"median paired delta: {statistics.median(deltas):+.1f}%  (median-of-medians {(p_med/s_med-1)*100:+.1f}%)")

    report = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "pairs": args.pairs,
            "warmup_sec": args.warmup_sec,
            "counted_sec": args.counted_sec,
            "concurrency": args.concurrency,
            "checkpoint_sha256": None,
            "full_sims": args.full_sims,
            "fast_sims": args.fast_sims,
        },
        "sync_sims_per_sec": sync_sps,
        "pipelined_sims_per_sec": pipe_sps,
        "per_pair_delta_pct": deltas,
        "median_sync": s_med,
        "median_pipelined": p_med,
        "median_delta_pct": statistics.median(deltas),
    }
    out = root / "paired_ab_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"summary written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
