"""Layered, fail-closed dataset validation with tamper-evident receipts.

Validation is split into four layers (audit F-04):

- **Layer A — structural integrity**: required files, exact byte sizes
  (trailing bytes fail), record counts, schema/dtype declarations.
- **Layer B — semantic integrity**: outcome-label domains, move-index range,
  plane finiteness/range, value/policy target validity, and the role rule: a
  dataset consumed in a role that requires WDL labels must not consist solely
  of masked (``OUTCOME_UNAVAILABLE``) targets.
- **Layer C — provenance and artifact identity**: exact SHA-256 of every
  array, build-record agreement, manifest/split-count agreement, bound into a
  receipt that later consumers re-verify so the bytes loaded are the bytes
  validated.
- **Layer D — split and leakage integrity**: exhaustive source/group-ID
  disjointness and exhaustive-or-bounded encoded-state overlap with explicit
  coverage reporting. A check that cannot complete is reported as such and
  prevents any definitive "disjoint" claim.

Failure philosophy: every check reports PASS / FAIL / BOUNDED / SKIPPED with
detail; an overall result is ``pass`` only when every executed check passed
and none were skipped, and ``incomplete`` when required checks could not run.
Nothing defaults to PASS.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from python.data.dataset_contract import (
    BUILD_RECORD_NAME,
    OUTCOME_UNAVAILABLE,
    PLANES_SHAPE,
    RECEIPT_NAME,
    RECEIPT_VERSION,
    REQUIRED_ARRAYS,
    SCHEMA_VERSION,
    DatasetContractError,
    DatasetRole,
    classify_role,
    file_identity,
    identity_matches,
    role_by_name,
    sha256_file,
)

logger = logging.getLogger(__name__)

ARRAY_DTYPES: Dict[str, str] = {
    "planes.bin": "float32",
    "moves.bin": "int16",
    "outcomes.bin": "uint8",
    "game_ids.bin": "uint64",
    "plies.bin": "int16",
    "value_targets.bin": "float32",
    "policy_targets.bin": "float32",
    "dtz_targets.bin": "float32",
}


def _array_num_bytes(name: str, num_samples: int) -> int:
    itemsize = np.dtype(ARRAY_DTYPES[name]).itemsize
    if name == "planes.bin":
        return num_samples * PLANES_SHAPE[0] * PLANES_SHAPE[1] * PLANES_SHAPE[2] * itemsize
    if name == "value_targets.bin":
        return num_samples * 3 * itemsize
    if name == "policy_targets.bin":
        return num_samples * 4672 * itemsize
    return num_samples * itemsize


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BOUNDED = "bounded"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    coverage: Optional[str] = None  # e.g. "full (2490528 rows)" or "first 20000 rows"
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "status": self.status.value, "detail": self.detail}
        if self.coverage:
            out["coverage"] = self.coverage
        if self.data:
            out["data"] = self.data
        return out


@dataclass
class ValidationReport:
    dataset_dir: str
    role: Optional[str]
    results: List[CheckResult] = field(default_factory=list)
    file_identities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    label_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @property
    def has_fail(self) -> bool:
        return any(r.status == CheckStatus.FAIL for r in self.results)

    @property
    def has_skipped_required(self) -> bool:
        return any(r.status == CheckStatus.SKIPPED for r in self.results)

    @property
    def has_bounded(self) -> bool:
        return any(r.status == CheckStatus.BOUNDED for r in self.results)

    @property
    def overall(self) -> str:
        if self.has_fail:
            return "fail"
        if self.has_skipped_required:
            return "incomplete"
        return "pass"

    @property
    def is_definitive(self) -> bool:
        """True when every executed check completed (no bounded/skipped checks)."""
        return not (self.has_skipped_required or self.has_bounded)

    def summary_line(self) -> str:
        counts = {s.value: 0 for s in CheckStatus}
        for r in self.results:
            counts[r.status.value] += 1
        return (
            f"overall={self.overall} definitive={self.is_definitive} "
            f"checks={counts}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "role": self.role,
            "overall": self.overall,
            "definitive": self.is_definitive,
            "summary": self.summary_line(),
            "label_stats": self.label_stats,
            "checks": [r.to_dict() for r in self.results],
        }


# ────────────────────────────────────────────────────────────────────────────
# Layer A + B: per-split structural and semantic validation
# ────────────────────────────────────────────────────────────────────────────


def _read_metadata(split_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[CheckResult]]:
    meta_path = split_dir / "metadata.json"
    if not meta_path.is_file():
        return None, CheckResult(
            "metadata_present", CheckStatus.FAIL, f"missing {meta_path}"
        )
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return None, CheckResult("metadata_parse", CheckStatus.FAIL, f"malformed metadata.json: {e}")
    return metadata, None


def validate_split(
    split_dir: Path,
    role: Optional[DatasetRole] = None,
    scan_planes: str = "bounded",
    plane_scan_rows: int = 20000,
    compute_hashes: bool = True,
    check_build_record: bool = True,
) -> ValidationReport:
    """Run Layers A-C against one split directory.

    Args:
        scan_planes: "off" (skip plane value checks), "bounded" (first
            ``plane_scan_rows`` rows), or "full" (every row).
        compute_hashes: compute exact SHA-256 for every array file (required
            for receipts; disable only for quick structural feedback).
    """
    split_dir = Path(split_dir)
    report = ValidationReport(dataset_dir=str(split_dir), role=role.name if role else None)

    metadata, meta_err = _read_metadata(split_dir)
    if meta_err is not None:
        report.add(meta_err)
        return report
    assert metadata is not None

    # Role resolution (Layer B contract, role never inferred from path names)
    resolved_role = role or classify_role(metadata)
    report.add(
        CheckResult(
            "role_declared",
            CheckStatus.PASS if resolved_role else CheckStatus.SKIPPED,
            detail=(
                f"role={resolved_role.name}" if resolved_role else "metadata declares no known role"
            ),
        )
    )

    # ── Layer A: structural ──────────────────────────────────────────────
    try:
        num_samples = int(metadata["num_samples"])
        if num_samples < 1:
            raise ValueError(f"num_samples {num_samples} out of range")
    except (KeyError, TypeError, ValueError) as e:
        report.add(
            CheckResult("num_samples_valid", CheckStatus.FAIL, f"invalid num_samples: {e}")
        )
        return report

    shape = metadata.get("planes_shape", list(PLANES_SHAPE))
    if tuple(shape) != PLANES_SHAPE:
        report.add(
            CheckResult(
                "schema_planes_shape",
                CheckStatus.FAIL,
                f"planes_shape {shape} != {list(PLANES_SHAPE)}",
            )
        )

    declared_schema = metadata.get("schema_version")
    if declared_schema is not None and str(declared_schema) != SCHEMA_VERSION:
        report.add(
            CheckResult(
                "schema_version",
                CheckStatus.FAIL,
                f"metadata schema_version {declared_schema!r} != current {SCHEMA_VERSION!r}",
            )
        )

    present_files: Dict[str, int] = {}
    for name in REQUIRED_ARRAYS + tuple(
        n for n in ARRAY_DTYPES if n not in REQUIRED_ARRAYS
    ):
        path = split_dir / name
        if not path.is_file():
            if name in REQUIRED_ARRAYS:
                report.add(
                    CheckResult(
                        f"file_present:{name}",
                        CheckStatus.FAIL,
                        f"required file {name} is missing from {split_dir}",
                    )
                )
            continue
        expected = _array_num_bytes(name, num_samples)
        actual = path.stat().st_size
        present_files[name] = actual
        if actual != expected:
            kind = "truncated or trailing data" if actual != expected else "ok"
            report.add(
                CheckResult(
                    f"exact_size:{name}",
                    CheckStatus.FAIL,
                    f"expected {expected} bytes for {num_samples} samples, found {actual} ({kind})",
                )
            )
        else:
            report.add(
                CheckResult(
                    f"exact_size:{name}",
                    CheckStatus.PASS,
                    f"{actual} bytes",
                )
            )

    if report.has_fail:
        # Structural failures invalidate every later layer.
        return report

    # ── Layer C (part 1): artifact identity ──────────────────────────────
    if compute_hashes:
        for name, size in present_files.items():
            path = split_dir / name
            report.file_identities[name] = file_identity(path)
        report.add(
            CheckResult(
                "artifact_hashes",
                CheckStatus.PASS,
                f"sha256 computed for {len(report.file_identities)} files",
            )
        )

    # ── Layer B: semantic ────────────────────────────────────────────────
    outcomes = np.fromfile(split_dir / "outcomes.bin", dtype=np.uint8)
    moves = np.fromfile(split_dir / "moves.bin", dtype=np.int16)
    plies = np.fromfile(split_dir / "plies.bin", dtype=np.int16)
    game_ids = np.fromfile(split_dir / "game_ids.bin", dtype=np.uint64)

    valid_outcomes = resolved_role.valid_outcomes if resolved_role else tuple(range(0, 256))
    bad_outcomes = np.sort(np.unique(outcomes[~np.isin(outcomes, valid_outcomes)]))
    report.add(
        CheckResult(
            "outcome_domain",
            CheckStatus.PASS if bad_outcomes.size == 0 else CheckStatus.FAIL,
            (
                "all outcomes within role domain"
                if bad_outcomes.size == 0
                else f"invalid outcome values {bad_outcomes.tolist()} not in role domain {valid_outcomes}"
            ),
            data={"histogram": _value_histogram(outcomes)},
        )
    )

    out_of_range_moves = int(((moves < 0) | (moves >= 4672)).sum())
    report.add(
        CheckResult(
            "move_index_range",
            CheckStatus.PASS if out_of_range_moves == 0 else CheckStatus.FAIL,
            (
                f"all {moves.size} move indices within [0, 4672)"
                if out_of_range_moves == 0
                else f"{out_of_range_moves} move indices outside [0, 4672)"
            ),
        )
    )

    negative_plies = int((plies < 0).sum())
    report.add(
        CheckResult(
            "plies_nonnegative",
            CheckStatus.PASS if negative_plies == 0 else CheckStatus.FAIL,
            f"{negative_plies} negative ply values",
        )
    )

    # Role-vs-labels rule: WDL-required roles cannot consist only of masked rows.
    n_masked = int((outcomes == OUTCOME_UNAVAILABLE).sum())
    n_wdl = int(outcomes.size - n_masked)
    masked_frac = n_masked / max(1, outcomes.size)
    report.label_stats[split_dir.name] = {
        "rows": int(outcomes.size),
        "masked": n_masked,
        "wdl_labelled": n_wdl,
        "masked_fraction": round(masked_frac, 6),
    }
    if resolved_role is not None and resolved_role.requires_wdl_labels and n_wdl == 0:
        report.add(
            CheckResult(
                "role_label_availability",
                CheckStatus.FAIL,
                (
                    f"role {resolved_role.name} requires WDL labels but all "
                    f"{outcomes.size} outcomes are masked (255); this artifact provides no "
                    "value supervision and must not be consumed as WDL-supervised data"
                ),
            )
        )
    else:
        report.add(
            CheckResult(
                "role_label_availability",
                CheckStatus.PASS,
                f"masked_fraction={masked_frac:.4f} ({n_masked} masked / {n_wdl} WDL-labelled)",
            )
        )

    # Optional target files: exact size already checked; validate content.
    vt_path = split_dir / "value_targets.bin"
    if vt_path.is_file():
        vt = np.memmap(vt_path, dtype=np.float32, mode="r", shape=(num_samples, 3))
        report.add(_check_value_targets(vt))
    pt_path = split_dir / "policy_targets.bin"
    if pt_path.is_file():
        pt = np.memmap(pt_path, dtype=np.float32, mode="r", shape=(num_samples, 4672))
        report.add(_check_policy_targets(pt, num_samples))

    # Planes: finiteness and [0, 1] range with explicit coverage.
    if scan_planes != "off":
        report.add(
            _check_planes(
                split_dir / "planes.bin",
                num_samples,
                scan_mode=scan_planes,
                scan_rows=plane_scan_rows,
            )
        )
    else:
        report.add(
            CheckResult(
                "planes_numeric_range",
                CheckStatus.SKIPPED,
                "plane value scan disabled by caller",
            )
        )

    # ── Layer C (part 2): build-record agreement ─────────────────────────
    if check_build_record:
        report.add(_check_build_record(split_dir, report.file_identities, num_samples))

    return report


def _value_histogram(arr: np.ndarray, max_values: int = 16) -> Dict[str, int]:
    values, counts = np.unique(arr, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values[:max_values], counts[:max_values])}


def _check_planes(planes_path: Path, num_samples: int, scan_mode: str, scan_rows: int) -> CheckResult:
    n_check = num_samples if scan_mode == "full" else min(scan_rows, num_samples)
    mm = np.memmap(planes_path, dtype=np.float32, mode="r", shape=(num_samples, *PLANES_SHAPE))
    bad_finite = 0
    bad_range = 0
    try:
        chunk = max(1, min(4096, n_check))
        for start in range(0, n_check, chunk):
            block = np.asarray(mm[start : min(start + chunk, n_check)], dtype=np.float32)
            if not np.isfinite(block).all():
                bad_finite += int((~np.isfinite(block)).sum())
            bad_range += int(((block < 0.0) | (block > 1.0)).sum())
    finally:
        try:
            mm._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    coverage = f"full ({n_check} rows)" if n_check == num_samples else f"first {n_check} of {num_samples} rows"
    status = CheckStatus.PASS if (bad_finite == 0 and bad_range == 0) else CheckStatus.FAIL
    if scan_mode == "bounded":
        status = CheckStatus.BOUNDED if status == CheckStatus.PASS else status
    return CheckResult(
        "planes_numeric_range",
        status,
        f"non-finite values: {bad_finite}, out-of-range [0,1] values: {bad_range}",
        coverage=coverage,
    )


def _check_value_targets(vt: np.memmap, tol: float = 1e-3) -> CheckResult:
    block_size = 65536
    bad_finite = 0
    bad_rows = 0
    n = vt.shape[0]
    for start in range(0, n, block_size):
        block = np.asarray(vt[start : min(start + block_size, n)], dtype=np.float32)
        if not np.isfinite(block).all():
            bad_finite += int((~np.isfinite(block)).sum())
        row_sum = block.sum(axis=1)
        is_zero = np.all(block == 0.0, axis=1)
        bad_rows += int((~is_zero & (np.abs(row_sum - 1.0) > tol)).sum())
    ok = bad_finite == 0 and bad_rows == 0
    return CheckResult(
        "value_targets_valid",
        CheckStatus.PASS if ok else CheckStatus.FAIL,
        f"non-finite: {bad_finite}, rows neither masked-zero nor summing to 1: {bad_rows}",
        coverage=f"full ({n} rows)",
    )


def _check_policy_targets(pt: np.memmap, num_samples: int, tol: float = 1e-3) -> CheckResult:
    block_size = 4096
    bad_finite = 0
    bad_rows = 0
    n = pt.shape[0]
    for start in range(0, n, block_size):
        block = np.asarray(pt[start : min(start + block_size, n)], dtype=np.float32)
        if not np.isfinite(block).all():
            bad_finite += int((~np.isfinite(block)).sum())
        row_sum = block.sum(axis=1)
        is_zero = np.all(block == 0.0, axis=1)
        bad_rows += int((~is_zero & (np.abs(row_sum - 1.0) > tol)).sum())
    ok = bad_finite == 0 and bad_rows == 0
    return CheckResult(
        "policy_targets_valid",
        CheckStatus.PASS if ok else CheckStatus.FAIL,
        f"non-finite: {bad_finite}, rows neither masked-zero nor summing to 1: {bad_rows}",
        coverage=f"full ({n} rows)",
    )


def _check_build_record(
    split_dir: Path, file_identities: Dict[str, Dict[str, Any]], num_samples: int
) -> CheckResult:
    """Layer C: verify the build record's output hashes match the current bytes."""
    root = split_dir.parent if split_dir.name in ("train", "val", "test") else split_dir
    record_path = root / BUILD_RECORD_NAME
    if not record_path.is_file():
        # Legacy artifact without a build record: reported, not fatal here —
        # the consumption gate decides whether provenance is mandatory.
        return CheckResult(
            "build_record_agreement",
            CheckStatus.SKIPPED,
            f"no {BUILD_RECORD_NAME} at {root}; artifact provenance is not reproducible",
        )
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        return CheckResult("build_record_agreement", CheckStatus.FAIL, f"malformed build record: {e}")

    outputs = record.get("outputs", {})
    split_key = split_dir.name if split_dir.name in ("train", "val", "test") else ""
    split_outputs = outputs.get(split_key, {}) if split_key else outputs
    if not isinstance(split_outputs, dict) or not split_outputs:
        return CheckResult(
            "build_record_agreement",
            CheckStatus.FAIL,
            f"build record has no output hashes for split '{split_key or 'root'}'",
        )
    mismatches = []
    for name, ident in split_outputs.items():
        path = split_dir / name
        expected_hash = ident.get("sha256") if isinstance(ident, dict) else ident
        if not path.is_file():
            mismatches.append(f"{name}: file missing")
            continue
        actual = file_identities.get(name, {}).get("sha256") or sha256_file(path)
        if expected_hash and actual != expected_hash:
            mismatches.append(f"{name}: sha256 {actual[:12]}… != build record {str(expected_hash)[:12]}…")
    if mismatches:
        return CheckResult(
            "build_record_agreement",
            CheckStatus.FAIL,
            "current bytes differ from build record: " + "; ".join(mismatches[:8]),
        )
    return CheckResult(
        "build_record_agreement",
        CheckStatus.PASS,
        f"all {len(split_outputs)} recorded output hashes match current bytes",
    )


# ────────────────────────────────────────────────────────────────────────────
# Layer D: split integrity
# ────────────────────────────────────────────────────────────────────────────


def _plane_row_hashes(planes_path: Path, num_samples: int, limit: Optional[int]) -> Tuple[Dict[bytes, List[int]], int]:
    """Row-wise MD5 of serialized plane records. Returns (hash -> row indices, rows hashed)."""
    n = num_samples if limit is None else min(limit, num_samples)
    mm = np.memmap(planes_path, dtype=np.float32, mode="r", shape=(num_samples, *PLANES_SHAPE))
    hashes: Dict[bytes, List[int]] = {}
    chunk = 2048
    try:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = np.ascontiguousarray(mm[start:end])
            for offset in range(end - start):
                digest = hashlib.md5(block[offset].tobytes()).digest()
                hashes.setdefault(digest, []).append(start + offset)
    finally:
        try:
            mm._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    return hashes, n


def check_split_integrity(
    train_dir: Path,
    val_dir: Path,
    exhaustive_states: bool = True,
    max_bounded_rows: int = 2000,
) -> List[CheckResult]:
    """Layer D between two splits: group-ID disjointness + encoded-state overlap.

    Returns check results only; callers decide exit codes. State overlap is
    classified into same-target and conflicting-target duplicates. When
    ``exhaustive_states`` is False only the first ``max_bounded_rows`` rows of
    each split are compared and the result is BOUNDED, never "disjoint".
    """
    results: List[CheckResult] = []
    train_dir, val_dir = Path(train_dir), Path(val_dir)

    ids_ok = True
    id_sets: List[Tuple[str, set[int]]] = []
    for label, d in (("train", train_dir), ("val", val_dir)):
        p = d / "game_ids.bin"
        if not p.is_file():
            results.append(
                CheckResult(
                    "group_ids_present",
                    CheckStatus.FAIL,
                    f"{label} split is missing game_ids.bin; source/group disjointness cannot be checked",
                )
            )
            ids_ok = False
        else:
            ids = np.fromfile(p, dtype=np.uint64)
            id_sets.append((label, set(int(x) for x in np.unique(ids))))
    if ids_ok:
        train_ids, val_ids = id_sets[0][1], id_sets[1][1]
        shared = train_ids & val_ids
        results.append(
            CheckResult(
                "group_id_disjointness",
                CheckStatus.FAIL if shared else CheckStatus.PASS,
                (
                    f"{len(shared)} group IDs shared across splits"
                    if shared
                    else f"0 shared group IDs (train {len(train_ids):,}, val {len(val_ids):,} unique)"
                ),
                coverage="exhaustive (all rows)",
            )
        )

    tr_meta, err_tr = _read_metadata(train_dir)
    va_meta, err_va = _read_metadata(val_dir)
    if err_tr or err_va or tr_meta is None or va_meta is None:
        results.append(
            CheckResult(
                "encoded_state_overlap",
                CheckStatus.FAIL,
                "metadata unreadable; state overlap check cannot run",
            )
        )
        return results

    tr_n = int(tr_meta["num_samples"])
    va_n = int(va_meta["num_samples"])
    limit = None if exhaustive_states else max_bounded_rows
    train_hashes, n_tr = _plane_row_hashes(train_dir / "planes.bin", tr_n, limit)
    val_hashes, n_va = _plane_row_hashes(val_dir / "planes.bin", va_n, limit)

    shared_keys = set(train_hashes) & set(val_hashes)
    same_target = 0
    conflicts: List[Dict[str, Any]] = []
    tr_moves = np.fromfile(train_dir / "moves.bin", dtype=np.int16)
    va_moves = np.fromfile(val_dir / "moves.bin", dtype=np.int16)
    for key in shared_keys:
        tr_rows = train_hashes[key]
        va_rows = val_hashes[key]
        if int(tr_moves[tr_rows[0]]) == int(va_moves[va_rows[0]]):
            same_target += 1
        else:
            conflicts.append(
                {
                    "state_md5": key.hex(),
                    "train_rows": tr_rows[:4],
                    "val_rows": va_rows[:4],
                    "train_move": int(tr_moves[tr_rows[0]]),
                    "val_move": int(va_moves[va_rows[0]]),
                }
            )
    within_train_dupes = sum(len(v) - 1 for v in train_hashes.values() if len(v) > 1)
    within_val_dupes = sum(len(v) - 1 for v in val_hashes.values() if len(v) > 1)

    if exhaustive_states:
        status = CheckStatus.FAIL if shared_keys else CheckStatus.PASS
        coverage = f"exhaustive (train {n_tr:,} rows, val {n_va:,} rows)"
        detail = (
            f"{len(shared_keys)} encoded states cross the split boundary "
            f"(same-target: {same_target}, conflicting-target: {len(conflicts)}); "
            f"within-split duplicate rows: train {within_train_dupes:,}, val {within_val_dupes:,}"
            if shared_keys
            else (
                f"0 cross-split encoded states; within-split duplicate rows: "
                f"train {within_train_dupes:,}, val {within_val_dupes:,}"
            )
        )
    else:
        status = CheckStatus.BOUNDED if not shared_keys else CheckStatus.FAIL
        coverage = f"first {n_tr:,} of {tr_n:,} train rows, first {n_va:,} of {va_n:,} val rows"
        detail = (
            f"{len(shared_keys)} encoded states overlap in the bounded prefix "
            f"(same-target: {same_target}, conflicting-target: {len(conflicts)}); "
            "NOT an exhaustive result — a disjoint claim cannot be made"
        )
    result = CheckResult("encoded_state_overlap", status, detail, coverage=coverage)
    if conflicts:
        result.data["conflicts"] = conflicts[:20]
    results.append(result)
    return results


# ────────────────────────────────────────────────────────────────────────────
# Receipts: bind a validation result to exact bytes
# ────────────────────────────────────────────────────────────────────────────


def write_receipt(
    dataset_dir: Path,
    reports: Dict[str, ValidationReport],
    overall: str,
    out_path: Optional[Path] = None,
) -> Path:
    """Write a validation receipt binding results to exact file identities."""
    from datetime import datetime, timezone

    dataset_dir = Path(dataset_dir)
    out_path = out_path or (dataset_dir / RECEIPT_NAME)
    files: Dict[str, Dict[str, Any]] = {}
    for split, report in reports.items():
        for name, ident in report.file_identities.items():
            files[f"{split}/{name}"] = ident
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(dataset_dir),
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validator": "python.data.dataset_validation",
        "overall": overall,
        "definitive": all(r.is_definitive for r in reports.values()),
        "role": next(iter(reports.values())).role if reports else None,
        "files": files,
        "splits": {split: report.to_dict() for split, report in reports.items()},
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(out_path)
    return out_path


def read_receipt(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(dataset_dir) / RECEIPT_NAME
    if not path.is_file():
        # Split-level callers fall back to the dataset root receipt.
        p = Path(dataset_dir)
        if p.name in ("train", "val", "test") and (p.parent / RECEIPT_NAME).is_file():
            path = p.parent / RECEIPT_NAME
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def verify_receipt(
    dataset_dir: Path,
    deep: bool = True,
    receipt: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Verify that the current bytes match the receipt's recorded identities.

    Args:
        deep: re-hash every file (cryptographic identity) instead of the cheap
            size+mtime_ns staleness check.
    Returns:
        (ok, failure_reasons, receipt_or_none)
    """
    dataset_dir = Path(dataset_dir)
    receipt = receipt or read_receipt(dataset_dir)
    if receipt is None:
        return False, [f"no validation receipt at {dataset_dir / RECEIPT_NAME}"], None
    # Resolve file paths against the receipt's recorded dataset root so a
    # split-level lookup still verifies the whole recorded artifact.
    recorded_dir = receipt.get("dataset_dir")
    base = Path(recorded_dir) if recorded_dir and Path(recorded_dir).is_dir() else dataset_dir
    reasons: List[str] = []
    if str(receipt.get("receipt_version", "")) != RECEIPT_VERSION:
        reasons.append(f"unknown receipt version {receipt.get('receipt_version')!r}")
    if receipt.get("overall") != "pass":
        reasons.append(f"recorded validation overall={receipt.get('overall')!r}, not 'pass'")
    files: Dict[str, Dict[str, Any]] = receipt.get("files", {})
    if not files:
        reasons.append("receipt records no file identities")
    for rel, ident in sorted(files.items()):
        path = base / rel
        if not path.is_file():
            reasons.append(f"{rel}: file missing")
            continue
        if deep:
            actual = sha256_file(path)
            expected = ident.get("sha256")
            if not expected or actual != expected:
                reasons.append(
                    f"{rel}: sha256 {actual[:12]}… != receipt {str(expected)[:12]}…"
                )
        else:
            if not identity_matches(path, ident):
                reasons.append(f"{rel}: size/mtime differ from receipt (stale receipt)")
    return (not reasons), reasons, receipt


# ────────────────────────────────────────────────────────────────────────────
# Consumption gate
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    data_dir: str
    role: Optional[DatasetRole]
    certified: bool
    receipt: Optional[Dict[str, Any]]
    masked_fraction: float
    num_samples: int
    warnings: List[str] = field(default_factory=list)


def require_validated_dataset(
    data_dir: Path | str,
    expected_role: Optional[str] = None,
    allow_uncertified: bool = False,
    deep: bool = True,
    consumer: str = "consumer",
) -> GateResult:
    """Fail-closed gate for any consumer about to read a dataset split.

    Enforces, before any batch is drawn:
    1. The split's role is known — either the consumer declares it via
       ``expected_role`` or the metadata declares it. Role is never inferred
       from directory names.
    2. The role-vs-labels invariant: a role requiring WDL labels fails when
       every outcome is masked. This invariant cannot be waived by
       ``allow_uncertified``; declare a policy-only role explicitly instead.
    3. A valid validation receipt binds the loaded bytes to a passing
       validation. Missing, stale, mismatched, or incomplete receipts fail
       unless ``allow_uncertified`` is set (which logs a loud, explicit
       UNCERTIFIED warning and returns ``certified=False``).

    Raises:
        DatasetContractError: when a required invariant is violated.
        FileNotFoundError: when metadata or outcomes.bin is missing.
    """
    from python.data.dataset_contract import classify_role as _classify

    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    role: Optional[DatasetRole]
    if expected_role:
        role = role_by_name(expected_role)
        if role is None:
            raise DatasetContractError(
                f"{consumer}: unknown expected role {expected_role!r}; "
                f"valid roles: games_wdl, policy_tactics, endgame_syzygy, unified"
            )
    else:
        role = _classify(metadata)

    num_samples = int(metadata.get("num_samples", 0))
    outcomes_path = data_dir / "outcomes.bin"
    if not outcomes_path.is_file():
        raise FileNotFoundError(f"outcomes.bin missing in {data_dir}")
    outcomes = np.fromfile(outcomes_path, dtype=np.uint8)
    if outcomes.size not in (0, num_samples):
        raise DatasetContractError(
            f"{consumer}: outcomes.bin has {outcomes.size} rows but metadata declares "
            f"{num_samples} samples in {data_dir}"
        )
    n_masked = int((outcomes == OUTCOME_UNAVAILABLE).sum())
    masked_fraction = n_masked / max(1, outcomes.size)
    n_wdl = outcomes.size - n_masked

    if role is None:
        raise DatasetContractError(
            f"{consumer}: dataset at {data_dir} declares no recognizable role "
            f"(metadata kind={metadata.get('kind')!r}). Pass expected_role explicitly "
            "(games_wdl, policy_tactics, endgame_syzygy, unified) or fix the dataset metadata."
        )
    if role.requires_wdl_labels and outcomes.size > 0 and n_wdl == 0:
        raise DatasetContractError(
            f"{consumer}: dataset at {data_dir} plays role '{role.name}' which requires WDL "
            f"labels, but all {outcomes.size} outcomes are masked (255). It cannot be consumed "
            f"as {role.name} data. Rebuild the dataset with real outcome labels or explicitly "
            "declare a policy-only role if that is the intent."
        )

    warnings: List[str] = []
    ok, reasons, receipt = verify_receipt(data_dir, deep=deep)
    certified = False
    if ok:
        certified = True
    elif allow_uncertified:
        warnings.append(
            f"UNCERTIFIED DATA: {consumer} is consuming {data_dir} without a valid validation "
            f"receipt ({'; '.join(reasons[:4])}). This consumption is not covered by any "
            "validation contract."
        )
        logger.warning(warnings[-1])
    else:
        raise DatasetContractError(
            f"{consumer}: dataset at {data_dir} failed validation-receipt verification: "
            + "; ".join(reasons[:6])
            + ". Run 'zeropawn data validate --dir <dataset> --write-receipt' or pass "
            "allow_uncertified=True explicitly to bypass (not recommended for reported results)."
        )

    return GateResult(
        data_dir=str(data_dir),
        role=role,
        certified=certified,
        receipt=receipt if certified else None,
        masked_fraction=masked_fraction,
        num_samples=num_samples,
        warnings=warnings,
    )
