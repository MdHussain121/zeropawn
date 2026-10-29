"""Fixed evaluation-set certification verification (audit F-10).

A certified evaluation CSV is produced by ``scripts/check_puzzle_overlap.py``
with a ``.certified.json`` sidecar recording the training-source hashes, the
output CSV hash, and the measured overlap. This module re-verifies such a
certificate against the CURRENT bytes before an evaluation is allowed to
report itself as certified:

- the manifest must exist, parse, and assert ``certified: true``;
- the evaluated CSV's SHA-256 must still match the recorded ``output_sha256``;
- the recorded row count must match;
- every recorded training source must still exist with the recorded hash.

The verdict is bounded by the certificate's own scope: source-ID and
starting-FEN separation from the LISTED training CSVs. It is not a claim of
global or historical adaptive independence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from python.data.dataset_contract import sha256_file


def verify_eval_certification(csv_path: str | Path) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Re-verify an eval CSV's certification against current bytes.

    Returns:
        (certified, failure_reasons, manifest_or_none)
    """
    csv_path = Path(csv_path)
    manifest_path = Path(str(csv_path) + ".certified.json")
    if not manifest_path.is_file():
        return False, [f"no certification manifest at {manifest_path}"], None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"unreadable certification manifest: {exc}"], None
    if not isinstance(data, dict) or data.get("certified") is not True:
        return False, ["manifest does not assert certified=true"], data

    reasons: List[str] = []
    recorded_output = data.get("output_sha256")
    if not recorded_output:
        reasons.append("manifest does not record the output CSV sha256")
    elif not csv_path.is_file():
        reasons.append(f"eval CSV missing: {csv_path}")
    else:
        actual = sha256_file(csv_path)
        if actual != recorded_output:
            reasons.append(
                f"eval CSV bytes changed since certification ({actual[:12]}… != {str(recorded_output)[:12]}…)"
            )
    recorded_rows = data.get("output_row_count")
    if recorded_rows is not None and csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8") as f:
            actual_rows = max(0, sum(1 for _ in f) - 1)
        if actual_rows != recorded_rows:
            reasons.append(f"eval CSV row count {actual_rows} != certified {recorded_rows}")
    for source in data.get("train_sources", []):
        src_path = Path(source.get("path", ""))
        src_hash = source.get("sha256")
        if not src_path.is_file():
            reasons.append(f"training source missing: {src_path}")
        elif not src_hash:
            reasons.append(f"training source {src_path} recorded without a hash")
        elif sha256_file(src_path) != src_hash:
            reasons.append(f"training source {src_path} changed since certification")
    return (not reasons), reasons, data
