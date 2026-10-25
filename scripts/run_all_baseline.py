"""Overnight chain: 6 full 100-game calibrated baseline runs, sequential.

Launched via schtasks with pythonw (no console). Logs to runs/baseline/chain.log.
Continues past per-setting failures; writes CHAIN_DONE marker at the end.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "baseline" / "chain.log"
JOBS = [
    ("data/models/checkpoints_finetune/best_model.pt", 1320),
    ("data/models/checkpoints_finetune/best_model.pt", 1500),
    ("data/models/checkpoints_finetune/best_model.pt", 1700),
    ("data/models/checkpoints_scaled/best_model.pt", 1320),
    ("data/models/checkpoints_scaled/best_model.pt", 1500),
    ("data/models/checkpoints_scaled/best_model.pt", 1700),
]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("CHAIN START")
    for ckpt, elo in JOBS:
        out = ROOT / "runs" / "baseline"
        tag = f"{Path(ckpt).parent.name}_{Path(ckpt).stem}_elo{elo}_full.json"
        if (out / tag).exists():
            log(f"SKIP {tag} (already exists)")
            continue
        log(f"START ckpt={ckpt} elo={elo}")
        t0 = time.time()
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_calibrated_baseline.py",
                    "--ckpt",
                    ckpt,
                    "--elo",
                    str(elo),
                    "--outdir",
                    "runs/baseline",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=5 * 3600,
            )
            (out / f"job_{elo}_{Path(ckpt).parent.name}.stdout.log").write_text(
                r.stdout[-20000:], encoding="utf-8"
            )
            (out / f"job_{elo}_{Path(ckpt).parent.name}.stderr.log").write_text(
                r.stderr[-20000:], encoding="utf-8"
            )
            log(f"END ckpt={ckpt} elo={elo} rc={r.returncode} wall={time.time() - t0:.0f}s")
        except Exception as e:
            log(f"FAIL ckpt={ckpt} elo={elo}: {e!r}")
    log("CHAIN_DONE")


main()
