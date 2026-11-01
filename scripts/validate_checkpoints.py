"""Verify the checkpoint registry against the files actually on disk.

Re-computes every SHA-256 and re-reads metadata for every record whose file
exists; reports missing artifacts as missing (never as recovered). Fails with
a non-zero exit when a recorded hash no longer matches its file — that is the
signal that provenance is broken and must be fixed before any training or
evaluation run cites these artifacts.

Usage:
    python scripts/validate_checkpoints.py            # verify
    python scripts/validate_checkpoints.py --write    # (re)write registry
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
from python.model.checkpoint_registry import (
    REGISTRY_REL_PATH,
    compute_sha256,
    inspect_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="(re)write the registry")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(ROOT),
        help=(
            "Root containing the (gitignored) models/ artifacts. Useful from a "
            "worktree whose checkout has no models/ directory."
        ),
    )
    args = parser.parse_args()

    reg_path = ROOT / REGISTRY_REL_PATH
    artifact_root = Path(args.repo_root)
    if not reg_path.is_file():
        print(f"No registry at {reg_path}.", file=sys.stderr)
        return 2

    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    failures = []
    missing = []
    verified = 0
    for rec in registry.get("checkpoints", []):
        f = artifact_root / rec["file"]
        if not f.is_file():
            missing.append(rec["id"])
            if rec.get("status") == "verified":
                failures.append(f"{rec['id']}: registered 'verified' but file is GONE")
            continue
        sha = compute_sha256(f)
        if rec.get("status") != "missing" and sha != rec.get("sha256"):
            failures.append(
                f"{rec['id']}: hash mismatch registry={rec.get('sha256')[:16]}... disk={sha[:16]}..."
            )
            continue
        if rec.get("status") == "missing":
            # A 'missing' record whose file reappeared with the same hash gets promoted.
            if sha == rec.get("sha256") and len(rec.get("sha256", "")) == 64:
                rec["status"] = "verified"
                rec["recovery_status"] = "RECOVERED and re-verified on disk."
                verified += 1
                print(f"[RECOVERED] {rec['id']}: file reappeared with matching hash")
            else:
                print(f"[STILL MISSING] {rec['id']}: file absent")
            continue
        info = inspect_checkpoint(f)
        if info.step is not None and rec.get("step") is not None and info.step != rec["step"]:
            failures.append(f"{rec['id']}: step drift registry={rec['step']} disk={info.step}")
            continue
        verified += 1
        print(f"[OK] {rec['id']}: sha256={sha[:16]}... step={info.step}")

    print(
        f"\nRegistry: {verified} verified on disk, {len(missing)} missing, "
        f"{len(failures)} failures"
    )
    for f_ in failures:
        print(f"FAIL: {f_}", file=sys.stderr)
    if args.write:
        # Audit CKP-14: the historical writer rewrote the tracked registry with a
        # bare write_text (torn JSON on crash). Now atomic + locked.
        from python.model.checkpoint_registry import write_registry

        write_registry(registry, ROOT, stamp_verified=True)
        print(f"Registry written: {reg_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
