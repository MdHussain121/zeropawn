"""Full remediation validation of a rebuilt dataset (Layers A-D exhaustive).

Runs per-split structural/semantic validation with a full plane scan, the
exhaustive cross-split integrity checks (train/val, train/test, val/test),
label/mask/stm distribution capture, and writes the validation receipt.

Usage:
    python scripts/remediation_validate_rebuild.py --dataset data/dataset_training_rebuild_20261031
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.data.dataset_validation import (
    CheckStatus,
    check_split_integrity,
    validate_split,
    write_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset)
    t0 = time.time()
    summary: dict = {"dataset": str(root), "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    reports = {}
    all_pass = True
    for split in args.splits:
        print(f"[validate] {split}: full structural+semantic+plane scan...", flush=True)
        report = validate_split(root / split, scan_planes="full")
        reports[split] = report
        counts = {s.value: 0 for s in CheckStatus}
        for r in report.results:
            counts[r.status.value] += 1
            if r.status == CheckStatus.FAIL:
                print(f"  FAIL {r.name}: {r.detail}")
        summary[split] = {
            "overall": report.overall,
            "definitive": report.is_definitive,
            "check_counts": counts,
            "label_stats": report.label_stats.get(split, {}),
        }
        all_pass = all_pass and report.overall == "pass"
        print(f"  overall={report.overall} definitive={report.is_definitive} checks={counts}")
        print(f"  labels: {report.label_stats.get(split)}")

    if not args.skip_overlap and len(args.splits) >= 2:
        print(
            "[integrity] exhaustive cross-split scan (this re-hashes every plane row)...",
            flush=True,
        )
        test_dir = root / "test" if (root / "test").is_dir() else None
        results = check_split_integrity(
            root / "train", root / args.splits[1], exhaustive_states=True, test_dir=test_dir
        )
        integrity = []
        for r in results:
            print(f"  {r.status.value.upper()} {r.name}: {r.detail} [{r.coverage}]")
            integrity.append(
                {
                    "name": r.name,
                    "status": r.status.value,
                    "detail": r.detail,
                    "coverage": r.coverage,
                }
            )
            if r.status in (CheckStatus.FAIL,):
                all_pass = False
            if r.status in (CheckStatus.SKIPPED, CheckStatus.BOUNDED):
                all_pass = False
        summary["split_integrity"] = integrity

    overall = "pass" if all_pass else "FAIL"
    summary["overall"] = overall
    summary["duration_sec"] = round(time.time() - t0, 1)

    out_path = root / f"remediation_validation_{time.strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[receipt] summary -> {out_path}")

    if all_pass:
        receipt = write_receipt(root, reports, overall="pass")
        print(f"[receipt] validation receipt written -> {receipt}")
    else:
        print("[receipt] validation FAILED — no receipt written")
        return 1
    print(f"DONE overall={overall} in {summary['duration_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
