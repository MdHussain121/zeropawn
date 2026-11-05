"""Regression suite for the 2026-11-05 pre-push audit SA-2 remediation.

SA2-01 (Low): masks.bin must be inside the receipt-binding flow. The packed
legal-action mask stream is consumed by every schema-2 loader, so validation
must size-check and hash it, the receipt must cryptographically bind it, and
receipt verification must reject a receipt that does not cover a masks stream
present on disk (stale receipt) or a masks.bin replaced after validation.

SA2-02 (Low): ``zeropawn data build`` must fail closed without an explicit
output target. There is deliberately no default dataset directory; an
invocation that omits --output-dir must not build into persistent data, and an
existing dataset target must refuse to be replaced without --force.

All fixtures are hermetic (tmp_path): no test in this module reads or writes
anything under the repository's data/ or models/ directories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest

from python.cli.commands_data import run_build_dataset
from python.data.dataset_contract import (
    MASK_PACKED_BYTES,
    RECEIPT_NAME,
    RECEIPT_VERSION,
    ROLE_GAMES_WDL,
    DatasetContractError,
    sha256_file,
)
from python.data.dataset_validation import (
    CheckStatus,
    require_validated_dataset,
    validate_split,
    verify_receipt,
    write_receipt,
)

# ──────────────────────────── helpers ────────────────────────────


def make_split_with_masks(split_dir: Path, n: int, masks: np.ndarray | None) -> Path:
    """Hermetic split fixture: required arrays + optional masks.bin + metadata.

    The metadata mirrors what the production builders write for schema-2
    datasets (dtype_masks/masks_shape) so the declared-stream check is
    exercised.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    np.zeros((n, 19, 8, 8), dtype=np.float32).tofile(split_dir / "planes.bin")
    np.asarray([10] * n, dtype=np.int16).tofile(split_dir / "moves.bin")
    np.asarray([0, 1, 2, 0] * (n // 4), dtype=np.uint8).tofile(split_dir / "outcomes.bin")
    np.asarray([1000 + i for i in range(n)], dtype=np.uint64).tofile(split_dir / "game_ids.bin")
    np.asarray([10 + i for i in range(n)], dtype=np.int16).tofile(split_dir / "plies.bin")
    if masks is not None:
        masks.tofile(split_dir / "masks.bin")
    meta = {
        "split": split_dir.name,
        "num_samples": n,
        "planes_shape": [19, 8, 8],
        "role": "games_wdl",
        "kind": "games",
        "dtype_masks": "uint8-bitpacked-4672",
        "masks_shape": [n, MASK_PACKED_BYTES],
    }
    (split_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return split_dir


def legal_masks(n: int) -> np.ndarray:
    """Deterministic valid packed masks: bit 0 set, rest clear."""
    masks = np.zeros((n, MASK_PACKED_BYTES), dtype=np.uint8)
    masks[:, 0] = 1
    return masks


# ──────────────────── SA2-01: masks.bin receipt binding ────────────────────


def test_sa2_masks_bin_is_hashed_and_bound_into_receipt(tmp_path):
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="off", compute_hashes=True, check_build_record=False
    )
    assert not report.has_fail, [r.detail for r in report.results if r.status == CheckStatus.FAIL]

    # Exact-size check present and passing for the mask stream.
    size_checks = [r for r in report.results if r.name == "exact_size:masks.bin"]
    assert size_checks and all(r.status == CheckStatus.PASS for r in size_checks)

    # The mask stream carries a cryptographic identity in the report...
    assert "masks.bin" in report.file_identities
    assert report.file_identities["masks.bin"]["sha256"] == sha256_file(d / "masks.bin")

    # ...and that identity is bound into the receipt.
    write_receipt(tmp_path, {"train": report}, overall="pass")
    receipt = json.loads((tmp_path / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert "train/masks.bin" in receipt["files"]
    assert receipt["files"]["train/masks.bin"]["sha256"] == sha256_file(d / "masks.bin")
    ok, reasons, _ = verify_receipt(tmp_path, deep=True)
    assert ok, reasons


def test_sa2_masks_bin_same_size_replacement_detected(tmp_path):
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="full", compute_hashes=True, check_build_record=False
    )
    write_receipt(tmp_path, {"train": report}, overall="pass")
    ok, reasons, _ = verify_receipt(tmp_path, deep=True)
    assert ok, reasons

    # Replace the mask stream with the same number of different bytes (every
    # legal mask flipped to 0xFF) — the receipt must reject the swapped file.
    np.ones((4, MASK_PACKED_BYTES), dtype=np.uint8).tofile(d / "masks.bin")
    ok, reasons, _ = verify_receipt(tmp_path, deep=True)
    assert not ok, "same-size masks.bin replacement must not pass receipt verification"
    assert any("masks.bin" in r for r in reasons)


def test_sa2_masks_bin_removal_after_receipt_detected(tmp_path):
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="full", compute_hashes=True, check_build_record=False
    )
    write_receipt(tmp_path, {"train": report}, overall="pass")
    (d / "masks.bin").unlink()
    ok, reasons, _ = verify_receipt(tmp_path, deep=True)
    assert not ok
    assert any("masks.bin" in r and "missing" in r for r in reasons)


def test_sa2_masks_bin_wrong_size_fails_validation(tmp_path):
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    # Truncate the mask stream by one byte per row (stale/corrupt layout).
    bad = np.zeros((4, MASK_PACKED_BYTES - 1), dtype=np.uint8)
    bad.tofile(d / "masks.bin")
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="off", compute_hashes=True, check_build_record=False
    )
    assert report.has_fail
    size_fail = [r for r in report.results if r.name == "exact_size:masks.bin"]
    assert size_fail and size_fail[0].status == CheckStatus.FAIL


def test_sa2_stale_receipt_without_mask_binding_rejected(tmp_path):
    """A receipt that predates masks.bin binding must not certify the dataset.

    Simulates every receipt written before the SA2-01 remediation: the other
    arrays are bound but the mask stream is absent from the receipt files.
    """
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="full", compute_hashes=True, check_build_record=False
    )
    write_receipt(tmp_path, {"train": report}, overall="pass")

    receipt_path = tmp_path / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "train/masks.bin" in receipt["files"]
    del receipt["files"]["train/masks.bin"]
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok, reasons, _ = verify_receipt(tmp_path, deep=True)
    assert not ok, "receipt not covering a present masks.bin must be rejected"
    assert any("not bound by this receipt" in r for r in reasons)

    # The cheap (size/mtime) verification path must reject it too.
    ok_cheap, reasons_cheap, _ = verify_receipt(tmp_path, deep=False)
    assert not ok_cheap
    assert any("not bound by this receipt" in r for r in reasons_cheap)


def test_sa2_declared_masks_with_missing_file_fails_validation(tmp_path):
    """Metadata declaring a mask stream but no masks.bin on disk must fail.

    The loaders consume masks.bin by existence, so a deleted stream would
    otherwise silently train without legal-action masking.
    """
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    (d / "masks.bin").unlink()
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="off", compute_hashes=True, check_build_record=False
    )
    assert report.has_fail
    check = [r for r in report.results if r.name == "masks_stream_present"]
    assert check and check[0].status == CheckStatus.FAIL


def test_sa2_consumption_gate_rejects_swapped_masks(tmp_path):
    """End-to-end: the fail-closed gate refuses masks.bin swapped post-receipt."""
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="full", compute_hashes=True, check_build_record=False
    )
    assert report.overall == "pass", [r.detail for r in report.results]
    write_receipt(tmp_path, {"train": report}, overall="pass")

    gate = require_validated_dataset(tmp_path / "train", expected_role="games_wdl")
    assert gate.certified

    np.ones((4, MASK_PACKED_BYTES), dtype=np.uint8).tofile(d / "masks.bin")
    with pytest.raises(DatasetContractError) as excinfo:
        require_validated_dataset(tmp_path / "train", expected_role="games_wdl")
    assert "masks.bin" in str(excinfo.value)


# ──────────────── SA2-02: fail-closed explicit build target ────────────────


def test_sa2_data_build_parser_has_no_default_target():
    from python.cli.main import build_parser

    args = build_parser().parse_args(["data", "build"])
    assert args.output_dir is None, "data build must not default to a production directory"
    assert args.force is False


def test_sa2_operations_registry_build_has_no_default_target():
    from python.cli.operations import OPERATIONS_REGISTRY

    op = next(op for op in OPERATIONS_REGISTRY if op.id == "data_build")
    out_arg = next(a for a in op.arguments if a.name == "output_dir")
    assert out_arg.default is None


def _build_ns(output_dir, force=False, pgn="nonexistent_input.pgn"):
    return argparse.Namespace(pgn=pgn, output_dir=output_dir, max_games=10, force=force)


def test_sa2_data_build_missing_target_fails_closed(tmp_path):
    assert run_build_dataset(_build_ns(None)) == 1
    # Blank/whitespace targets are treated as missing, never as cwd.
    assert run_build_dataset(_build_ns("   ")) == 1
    # Nothing may have been created anywhere by the failed invocations.
    assert list(tmp_path.iterdir()) == []


def test_sa2_data_build_refuses_existing_dataset_target(tmp_path):
    target = tmp_path / "existing_ds"
    (target / "train").mkdir(parents=True)
    (target / "train" / "metadata.json").write_text("{}", encoding="utf-8")
    (target / "train" / "planes.bin").write_bytes(b"\x00" * 64)
    (target / RECEIPT_NAME).write_text(
        json.dumps({"receipt_version": RECEIPT_VERSION, "overall": "pass"}), encoding="utf-8"
    )

    before = {p: p.read_bytes() for p in sorted(target.rglob("*")) if p.is_file()}
    assert run_build_dataset(_build_ns(str(target))) == 1

    after = {p: p.read_bytes() for p in sorted(target.rglob("*")) if p.is_file()}
    assert before == after, "refused build must not modify the existing dataset"
    # In particular the builder must not have started writing split arrays.
    assert not (target / "train" / "planes.bin.part").exists()
    assert not (target / "train" / "moves.bin").exists()


def test_sa2_data_build_refuses_nonempty_and_file_targets(tmp_path):
    # Any non-empty directory (even without dataset artifacts) is refused.
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "notes.txt").write_text("keep me", encoding="utf-8")
    assert run_build_dataset(_build_ns(str(nonempty))) == 1
    assert (nonempty / "notes.txt").read_text(encoding="utf-8") == "keep me"

    # A target that exists but is a file is refused outright.
    a_file = tmp_path / "a_file.bin"
    a_file.write_bytes(b"\x00")
    assert run_build_dataset(_build_ns(str(a_file))) == 1


def test_sa2_data_build_explicit_fresh_target_proceeds(tmp_path, monkeypatch):
    captured = {}

    def fake_executor(script_rel_path, args, op_name, description, parameters=None, **kwargs):
        captured["script"] = script_rel_path
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr("python.cli.commands_data.execute_script_with_monitoring", fake_executor)
    pgn = tmp_path / "input.pgn"
    pgn.write_text('[Event "?"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n', encoding="utf-8")
    fresh = tmp_path / "fresh_target"

    rc = run_build_dataset(_build_ns(str(fresh), pgn=str(pgn)))
    assert rc == 0
    assert captured["script"] == "python/data/build_dataset.py"
    assert captured["args"] == [
        "--pgn",
        str(pgn),
        "--output-dir",
        str(fresh),
        "--max-games",
        "10",
    ]
    # The guard itself must not have created anything; the builder owns that.
    assert not fresh.exists()


def test_sa2_data_build_refuses_repository_critical_targets(tmp_path, monkeypatch):
    """Repo root, data/, and models/ are never valid build targets, even with --force."""
    from python.cli.common import REPO_ROOT

    captured = {}

    def fake_executor(script_rel_path, args, op_name, description, parameters=None, **kwargs):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr("python.cli.commands_data.execute_script_with_monitoring", fake_executor)
    for critical in (REPO_ROOT, REPO_ROOT / "data", REPO_ROOT / "models"):
        assert run_build_dataset(_build_ns(str(critical), force=True)) == 1
        assert run_build_dataset(_build_ns(str(critical))) == 1
    assert "args" not in captured, "no build may reach the executor for critical targets"


def test_sa2_data_build_force_overrides_guard_and_plumbs_flag(tmp_path, monkeypatch):
    captured = {}

    def fake_executor(script_rel_path, args, op_name, description, parameters=None, **kwargs):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr("python.cli.commands_data.execute_script_with_monitoring", fake_executor)
    target = tmp_path / "existing_ds"
    target.mkdir()
    (target / "planes.bin").write_bytes(b"\x00" * 64)

    rc = run_build_dataset(_build_ns(str(target), force=True))
    assert rc == 0
    assert captured["args"][-1] == "--force"


def test_sa2_masks_replacement_preserving_mtime_still_detected_by_deep_verify(tmp_path):
    """Crypto binding catches a swap even when size and mtime are restored."""
    d = make_split_with_masks(tmp_path / "train", 4, legal_masks(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="full", compute_hashes=True, check_build_record=False
    )
    write_receipt(tmp_path, {"train": report}, overall="pass")

    st = (d / "masks.bin").stat()
    np.ones((4, MASK_PACKED_BYTES), dtype=np.uint8).tofile(d / "masks.bin")
    os.utime(d / "masks.bin", ns=(st.st_atime_ns, st.st_mtime_ns))

    # The cryptographic identity must catch the swap even though the cheap
    # size/mtime signature was perfectly restored.
    ok_deep, reasons, _ = verify_receipt(tmp_path, deep=True)
    assert not ok_deep
    assert any("masks.bin" in r for r in reasons)
