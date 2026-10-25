"""Item 4: benchmark discipline — one session, interleaved pairs + bracketing refs.

Session plan (concurrency 12, seed fixed):
  REF, [change x {A,B,A,B,A,B} for each change], REF
Changes (control -> variant):
  flush-s0: --flush-slack unset      -> --flush-slack 0   (null control)
  flush-s1: --flush-slack unset      -> --flush-slack 1
  flush-s2: --flush-slack unset      -> --flush-slack 2
  time-05:  --max-wait-ms 1.5        -> --max-wait-ms 0.5
  time-30:  --max-wait-ms 1.5        -> --max-wait-ms 3.0
Each run: 60 s warmup + 120 s counted (~3.5 min). Resume: completed run dirs are
skipped. Win rule: variant wins >= 2 of 3 pairs by more than the session
variance threshold = max(|ref_end - ref_start|, 2% of ref mean).
Usage: python scripts/ab_bench.py --out-dir docs/perf2/ab --chunk "0:8"  (etc.)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RUN_WARMUP = 60.0
RUN_COUNTED = 120.0
CONC = 12
SEED = 42

BASE_ARGS = [
    "--mode",
    "selfplay",
    "--warmup-sec",
    str(RUN_WARMUP),
    "--counted-sec",
    str(RUN_COUNTED),
    "--trials",
    "1",
    "--concurrency",
    str(CONC),
    "--stop-on-rss-6gb",
    "--base-seed",
    str(SEED),
]

CHANGES = [
    ("flush-s0", {}, {"flush-slack": "0"}),
    ("flush-s1", {}, {"flush-slack": "1"}),
    ("flush-s2", {}, {"flush-slack": "2"}),
    ("time-05", {"max-wait-ms": "1.5"}, {"max-wait-ms": "0.5"}),
    ("time-30", {"max-wait-ms": "1.5"}, {"max-wait-ms": "3.0"}),
]


def plan():
    seq = [("ref-start", {})]
    for name, actl, bctl in CHANGES:
        for i in range(3):
            seq.append((f"{name}-A{i + 1}", dict(actl)))
            seq.append((f"{name}-B{i + 1}", dict(bctl)))
    seq.append(("ref-end", {}))
    return seq


def run_one(out_dir: Path, tag: str, extra: dict) -> dict:
    rundir = out_dir / tag
    done_marker = rundir / "AB_DONE.json"
    if done_marker.exists():
        return json.loads(done_marker.read_text())
    rundir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/bench_selfplay.py", *BASE_ARGS, "--output-dir", str(rundir)]
    for k, v in extra.items():
        cmd += [f"--{k}", v]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = round(time.perf_counter() - t0, 1)
    rec = {
        "tag": tag,
        "extra": extra,
        "wall_s": wall,
        "returncode": cp.returncode,
        "tail": cp.stdout[-1500:] if cp.stdout else "",
        "stderr_tail": cp.stderr[-500:] if cp.stderr else "",
    }
    try:
        summ = json.loads((rundir / "selfplay_summary.json").read_text())
        ts = summ["trial_summaries"][0]
        rec["summary"] = {
            k: ts.get(k)
            for k in (
                "sims_per_sec",
                "avg_batch_size",
                "inference_latency_ms",
                "queue_wait_time_ms",
                "gpu_util_pct",
                "gpu_clock_mhz",
                "power_w",
                "temp_c",
                "games_per_hour",
                "positions_per_hour",
                "free_ram_mb",
            )
        }
        rows = list(__import__("csv").DictReader(open(rundir / "selfplay_trial_1.csv")))
        rec["counted_rows"] = sum(1 for r in rows if r["phase"] == "counted")
    except Exception as e:
        rec["summary_error"] = str(e)
    done_marker.write_text(json.dumps(rec, indent=1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/ab")
    ap.add_argument("--chunk", default="all", help="'start:end' slice of the 32-run plan, e.g. 0:8")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    seq = plan()
    if args.chunk != "all":
        a, b = (int(x) for x in args.chunk.split(":"))
        seq = seq[a:b]
    results = []
    for tag, extra in seq:
        print(f"[{time.strftime('%H:%M:%S')}] RUN {tag} {extra}", flush=True)
        try:
            rec = run_one(out_dir, tag, extra)
        except subprocess.TimeoutExpired:
            rec = {"tag": tag, "extra": extra, "timeout": True}
            (out_dir / tag).mkdir(parents=True, exist_ok=True)
            (out_dir / tag / "AB_DONE.json").write_text(json.dumps(rec))
        s = (rec.get("summary") or {}).get("sims_per_sec", "NA")
        print(f"[{time.strftime('%H:%M:%S')}] DONE {tag}: sims/s={s}", flush=True)
        results.append(rec)
    (out_dir / f"chunk_{args.chunk.replace(':', '-')}.json").write_text(
        json.dumps(results, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
