"""Regression suite for the data-integrity remediation (audit F-01..F-10).

Covers the twelve required scenarios from the remediation mission:
WDL label semantics on a known PGN build, role-vs-label gates (fail closed),
policy-only role acceptance, collision-resistant PGN identity, deterministic
duplicate handling, position-family split integrity for tactics, conflict
reporting, semantic validation failures, receipt tamper-evidence and
incompleteness, download completion enforcement, evaluation-certificate
tamper detection, and trainer consumption gates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from python.data.build_dataset import build_binary_dataset, process_pgn_game
from python.data.dataset_contract import (
    OUTCOME_UNAVAILABLE,
    ROLE_GAMES_WDL,
    ROLE_POLICY_TACTICS,
    DatasetContractError,
)
from python.data.dataset_validation import (
    CheckStatus,
    check_split_integrity,
    validate_split,
    verify_receipt,
    write_receipt,
    require_validated_dataset,
)

# ──────────────────────────── helpers ────────────────────────────

GAME_WHITE_LOSES = """[Event "?"]
[Site "?"]
[Date "2026.01.01"]
[White "PlayerW"]
[Black "PlayerB"]
[Result "0-1"]

1. e4 e5 0-1
"""

GAME_WHITE_WINS = """[Event "?"]
[Site "?"]
[Date "2026.01.01"]
[White "PlayerW"]
[Black "PlayerB"]
[Result "1-0"]

1. d4 d5 1-0
"""

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_split(
    split_dir: Path,
    n: int,
    outcomes: list[int] | np.ndarray,
    moves: list[int] | np.ndarray | None = None,
    gids: list[int] | np.ndarray | None = None,
    plies: list[int] | np.ndarray | None = None,
    metadata: dict | None = None,
    planes_nan_row: int | None = None,
    value_targets: np.ndarray | None = None,
) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    planes = np.zeros((n, 19, 8, 8), dtype=np.float32)
    for i in range(n):
        planes[i, 0, 0, 0] = (i % 11) / 10.0
    if planes_nan_row is not None:
        planes[planes_nan_row, 0, 0, 0] = np.nan
    planes.tofile(split_dir / "planes.bin")
    moves_arr = np.asarray(moves if moves is not None else [10] * n, dtype=np.int16)
    moves_arr.tofile(split_dir / "moves.bin")
    out_arr = np.asarray(outcomes, dtype=np.uint8)
    out_arr.tofile(split_dir / "outcomes.bin")
    gids_arr = np.asarray(gids if gids is not None else [1000 + i for i in range(n)], dtype=np.uint64)
    gids_arr.tofile(split_dir / "game_ids.bin")
    plies_arr = np.asarray(plies if plies is not None else [10 + i for i in range(n)], dtype=np.int16)
    plies_arr.tofile(split_dir / "plies.bin")
    if value_targets is not None:
        value_targets.tofile(split_dir / "value_targets.bin")
    meta = {
        "split": split_dir.name,
        "num_samples": n,
        "planes_shape": [19, 8, 8],
    }
    if metadata:
        meta.update(metadata)
    (split_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return split_dir


def games_wdl_meta(n: int) -> dict:
    return {
        "num_samples": n,
        "role": "games_wdl",
        "kind": "games",
        "label_semantics": {"role": "games_wdl", "wdl_supervised": True},
    }


# ──────────────── 1. tiny known PGN build yields mover WDL labels ────────────────


def test_pgn_build_wdl_labels(tmp_path):
    out_dir = tmp_path / "ds"
    n_train, n_val, removed = build_binary_dataset(io.StringIO(GAME_WHITE_LOSES), out_dir)
    assert n_train + n_val == 2  # one 2-ply game
    split = out_dir / ("train" if n_train == 2 else "val")
    outcomes = np.fromfile(split / "outcomes.bin", dtype=np.uint8)
    plies = np.fromfile(split / "plies.bin", dtype=np.int16)
    # Result 0-1 (Black wins): ply0 White to move -> Loss(2); ply1 Black -> Win(0)
    assert outcomes.tolist() == [2, 0]
    assert plies.tolist() == [0, 1]


def test_pgn_build_wdl_labels_white_win(tmp_path):
    out_dir = tmp_path / "ds"
    n_train, n_val, _ = build_binary_dataset(io.StringIO(GAME_WHITE_WINS), out_dir)
    split = out_dir / ("train" if n_train == 2 else "val")
    outcomes = np.fromfile(split / "outcomes.bin", dtype=np.uint8)
    # Result 1-0 (White wins): ply0 White -> Win(0); ply1 Black -> Loss(2)
    assert outcomes.tolist() == [0, 2]


# ──────────────── 2/3. role-vs-label gates ────────────────


def test_wdl_required_all_masked_fails(tmp_path):
    d = make_split(
        tmp_path / "train",
        4,
        outcomes=[OUTCOME_UNAVAILABLE] * 4,
        metadata=games_wdl_meta(4),
    )
    report = validate_split(d, role=ROLE_GAMES_WDL, scan_planes="off")
    assert report.has_fail
    role_checks = [r for r in report.results if r.name == "role_label_availability"]
    assert role_checks and role_checks[0].status == CheckStatus.FAIL

    with pytest.raises(DatasetContractError, match="requires WDL"):
        require_validated_dataset(d, expected_role="games_wdl", allow_uncertified=True)

    # Explicit policy-only declaration is the documented escape hatch.
    gate = require_validated_dataset(
        d, expected_role="policy_tactics", allow_uncertified=True
    )
    assert gate.certified is False  # no receipt
    assert gate.masked_fraction == 1.0


def test_policy_only_declared_masked_passes(tmp_path):
    d = make_split(
        tmp_path / "train",
        4,
        outcomes=[OUTCOME_UNAVAILABLE] * 4,
        metadata={
            "num_samples": 4,
            "role": "policy_tactics",
            "kind": "tactics-fullline",
            "label_semantics": {"role": "policy_tactics", "wdl_supervised": False},
        },
    )
    report = validate_split(d, scan_planes="off")
    assert not report.has_fail
    gate = require_validated_dataset(
        d, expected_role="policy_tactics", allow_uncertified=True
    )
    assert gate.role is not None and gate.role.name == "policy_tactics"

    # With a full validation receipt the same dataset certifies cleanly.
    full_report = validate_split(d, scan_planes="full", compute_hashes=True)
    write_receipt(d.parent, {"train": full_report}, overall="pass")
    ok, reasons, _ = verify_receipt(d.parent, deep=True)
    assert ok, reasons
    gate2 = require_validated_dataset(d, expected_role="policy_tactics")
    assert gate2.certified is True


# ──────────────── 4/5. PGN identity (F-03) ────────────────


def _read_game(pgn_text: str):
    import chess.pgn

    return chess.pgn.read_game(io.StringIO(pgn_text))


SAME_HEADERS_1 = GAME_WHITE_LOSES
SAME_HEADERS_2 = GAME_WHITE_WINS  # same Date/White/Black, different mainline + result


def test_fallback_identity_collision_fixed():
    g1 = _read_game(SAME_HEADERS_1)
    g2 = _read_game(SAME_HEADERS_2)
    id1, _, pos1 = process_pgn_game(g1)
    id2, _, pos2 = process_pgn_game(g2)
    assert pos1 and pos2
    assert id1 != id2, "distinct same-date/player games must not share a group identity"

    id1_again, _, _ = process_pgn_game(_read_game(SAME_HEADERS_1))
    assert id1_again == id1, "identity must be stable across replays"


def test_exact_duplicate_games_deterministic(tmp_path):
    single = tmp_path / "single"
    duplicated = tmp_path / "dup"
    pgn = GAME_WHITE_LOSES
    n1, v1, _ = build_binary_dataset(io.StringIO(pgn), single)
    n2, v2, _ = build_binary_dataset(io.StringIO(pgn + "\n\n" + pgn), duplicated)
    assert (n1, v1) == (n2, v2), "exact duplicate games must not double the serialized rows"


# ──────────────── 6/7. tactics position-family integrity (F-02/F-06) ────────────────


def _find_pid_with_hash(prefix: str, want_val: bool):
    from python.data.build_dataset import hash_game_id

    for i in range(100000):
        pid = f"{prefix}{i:05d}"
        is_val = hash_game_id(f"tactics_{pid}") % 100 < 2
        if is_val == want_val:
            return pid
    raise AssertionError("no pid found")


def test_tactics_position_family_and_conflicts(tmp_path):
    from python.data.build_tactics import build_tactics_dataset

    train_pid = _find_pid_with_hash("T", want_val=False)
    val_pid = _find_pid_with_hash("V", want_val=True)

    rows = [
        [train_pid, START_FEN, "e2e4 e7e5", "1500"],
        [val_pid, START_FEN, "e2e4 e7e5", "1500"],  # same state, other split candidate
        [val_pid + "x", START_FEN, "e2e4 e7e6", "1500"],  # same state, conflicting target
    ]
    csv_path = tmp_path / "puzzles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
        w.writerows(rows)

    out = tmp_path / "ds"
    n_train, n_val = build_tactics_dataset(str(csv_path), str(out), max_solver_plies=2)
    assert n_train >= 1
    # All rows of the shared-state family must be in ONE split.
    assert n_val == 0 or n_train == 0, "family spanning both splits is forbidden"

    record = json.loads((out / "build_record.json").read_text(encoding="utf-8"))
    verification = record["notes"]["serialized_output_verification"]
    assert verification["passed"] is True
    assert verification["cross_split_overlap"] == 0
    total_conflicts = (
        record["counts"]["train_target_conflicts"] + record["counts"]["val_target_conflicts"]
    )
    assert total_conflicts >= 1, "the conflicting-target case must be detected and reported"
    assert record["source"]["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()


# ──────────────── 8. semantic validation failures ────────────────


@pytest.mark.parametrize(
    ("overrides", "check_name"),
    [
        ({"moves": [5000, 10, 10, 10]}, "move_index_range"),
        ({"outcomes": [0, 1, 2, 7]}, "outcome_domain"),
        ({"planes_nan_row": 2}, "planes_numeric_range"),
    ],
)
def test_correct_size_invalid_data_fails(tmp_path, overrides, check_name):
    merged = {"outcomes": [0, 1, 2, 0]}
    merged.update(overrides)
    d = make_split(tmp_path / "train", 4, metadata=games_wdl_meta(4), **merged)
    report = validate_split(d, role=ROLE_GAMES_WDL, scan_planes="full")
    names = {r.name: r for r in report.results}
    assert check_name in names
    assert names[check_name].status == CheckStatus.FAIL
    assert report.overall == "fail"


def test_invalid_value_targets_fail(tmp_path):
    vt = np.zeros((4, 3), dtype=np.float32)
    vt[1] = [0.5, 0.2, 0.1]  # sums to 0.8: neither masked nor normalized
    d = make_split(
        tmp_path / "train",
        4,
        outcomes=[0, 1, 2, 0],
        metadata=games_wdl_meta(4),
        value_targets=vt,
    )
    report = validate_split(d, role=ROLE_GAMES_WDL, scan_planes="off")
    names = {r.name: r for r in report.results}
    assert names["value_targets_valid"].status == CheckStatus.FAIL


# ──────────────── 9. receipts never certify incomplete/stale state ────────────────


def test_missing_required_file_fails_validation(tmp_path):
    d = make_split(tmp_path / "train", 4, outcomes=[0, 1, 2, 0], metadata=games_wdl_meta(4))
    (d / "game_ids.bin").unlink()
    report = validate_split(d, role=ROLE_GAMES_WDL, scan_planes="off")
    assert report.has_fail
    assert any("game_ids.bin" in r.detail for r in report.results if r.status == CheckStatus.FAIL)


def test_receipt_tamper_and_staleness(tmp_path):
    d = make_split(tmp_path / "train", 4, outcomes=[0, 1, 2, 0], metadata=games_wdl_meta(4))
    root = tmp_path
    report = validate_split(d, role=ROLE_GAMES_WDL, scan_planes="full", compute_hashes=True)
    write_receipt(root, {"train": report}, overall="pass")
    ok, reasons, _ = verify_receipt(root, deep=True)
    assert ok, reasons

    # Tamper with a validated byte.
    data = bytearray((d / "moves.bin").read_bytes())
    data[-1] = (data[-1] + 1) % 256
    (d / "moves.bin").write_bytes(bytes(data))
    ok, reasons, _ = verify_receipt(root, deep=True)
    assert not ok
    assert any("moves.bin" in r for r in reasons)

    # Restore bytes but change mtime: stat-level verification must fail.
    import os

    (d / "moves.bin").write_bytes(bytes(bytearray((d / "moves.bin").read_bytes())))
    st = (d / "moves.bin").stat()
    os.utime((d / "moves.bin"), ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    ok, reasons, _ = verify_receipt(root, deep=False)
    assert not ok
    assert any("stale" in r or "mtime" in r for r in reasons)


def test_incomplete_validation_never_certifies(tmp_path):
    d = make_split(tmp_path / "train", 4, outcomes=[0, 1, 2, 0], metadata=games_wdl_meta(4))
    report = validate_split(
        d, role=ROLE_GAMES_WDL, scan_planes="bounded", compute_hashes=True, check_build_record=False
    )
    assert report.overall == "incomplete"
    assert not report.is_definitive  # bounded plane scan
    write_receipt(d.parent, {"train": report}, overall="incomplete")
    ok, reasons, _ = verify_receipt(d.parent, deep=True)
    assert not ok, "an incomplete validation receipt must never satisfy the gate"
    assert any("not 'pass'" in r for r in reasons)


def test_split_integrity_checks_fail_closed(tmp_path):
    tr = make_split(
        tmp_path / "train",
        3,
        outcomes=[0, 1, 2],
        gids=[7, 8, 9],
        metadata=games_wdl_meta(3),
    )
    va = make_split(
        tmp_path / "val",
        2,
        outcomes=[0, 1],
        gids=[9, 10],  # gid 9 shared with train
        metadata=games_wdl_meta(2),
    )
    results = {r.name: r for r in check_split_integrity(tr, va, exhaustive_states=True)}
    assert results["group_id_disjointness"].status == CheckStatus.FAIL

    # Same planes content in both splits -> encoded-state overlap.
    va2 = make_split(
        tmp_path / "val2",
        2,
        outcomes=[0, 1],
        gids=[11, 12],
        metadata=games_wdl_meta(2),
    )
    # make val2 planes identical to train's first two rows
    planes = np.zeros((3, 19, 8, 8), dtype=np.float32)
    for i in range(3):
        planes[i, 0, 0, 0] = (i % 11) / 10.0
    planes[:2].tofile(va2 / "planes.bin")
    results = {r.name: r for r in check_split_integrity(tr, va2, exhaustive_states=True)}
    assert results["encoded_state_overlap"].status == CheckStatus.FAIL

    # Bounded scan reports BOUNDED, never a definitive disjoint claim.
    results = {r.name: r for r in check_split_integrity(tr, va2, exhaustive_states=False)}
    assert results["encoded_state_overlap"].status == CheckStatus.FAIL  # overlap in prefix


# ──────────────── 10. downloader completion enforcement (F-07) ────────────────


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._body = body
        self._pos = 0
        self.status = status
        self.code = status
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        if self._pos >= len(self._body):
            return b""
        end = len(self._body) if n < 0 else min(self._pos + n, len(self._body))
        chunk = self._body[self._pos : end]
        self._pos = end
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_download(
    monkeypatch, body: bytes, status: int = 200, headers: dict | None = None, remote_size: int | None = None
):
    import python.data.download as dl

    calls = {"range": None}

    def fake_urlopen(req, timeout=None):
        calls["range"] = req.headers.get("Range")
        return FakeResponse(body, status=status, headers=headers)

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dl, "get_remote_file_size", lambda url, timeout=10.0: remote_size if remote_size is not None else len(body))
    return calls


def test_truncated_download_never_published(tmp_path, monkeypatch):
    import python.data.download as dl

    body = b"x" * 40  # server body ends early; HEAD reported 100 bytes
    calls = _patch_download(monkeypatch, body, remote_size=100)
    dest = tmp_path / "out" / "file.bin"
    with pytest.raises(IOError, match="Incomplete download"):
        dl.download_file("http://example.test/f.bin", dest)
    assert not dest.exists(), "a truncated body must never be published"
    assert Path(str(dest) + ".part").exists(), "partial data kept for resume"
    assert calls["range"] is None


def test_invalid_206_content_range_aborts(tmp_path, monkeypatch):
    import python.data.download as dl

    dest = tmp_path / "out" / "file.bin"
    dest.parent.mkdir(parents=True)
    part = Path(str(dest) + ".part")
    part.write_bytes(b"1234567890")  # 10 bytes on disk
    body = b"1234567890" + b"x" * 90
    _patch_download(
        monkeypatch,
        body,
        status=206,
        headers={"Content-Range": "bytes 0-99/100"},  # wrong resume offset (0 != 10)
    )
    with pytest.raises(ValueError, match="Content-Range"):
        dl.download_file("http://example.test/f.bin", dest)
    assert not dest.exists()


def test_ignored_range_restarts_from_zero(tmp_path, monkeypatch):
    import python.data.download as dl

    dest = tmp_path / "out" / "file.bin"
    dest.parent.mkdir(parents=True)
    Path(str(dest) + ".part").write_bytes(b"STALE-HEADER-DATA-xx")  # 20 stale bytes
    body = b"y" * 100
    calls = _patch_download(monkeypatch, body, status=200)
    dl.download_file("http://example.test/f.bin", dest)
    assert dest.read_bytes() == body, "HTTP 200 on resume must restart, not append"
    assert Path(str(dest) + ".sha256").is_file()
    prov = json.loads(Path(str(dest) + ".provenance.json").read_text(encoding="utf-8"))
    assert prov["sha256"] == hashlib.sha256(body).hexdigest()
    assert calls["range"] is not None


def test_same_size_corrupt_file_with_sidecar_redownloads(tmp_path, monkeypatch):
    import python.data.download as dl

    body = b"z" * 50
    dest = tmp_path / "out" / "file.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"CORRUPT" + b"!" * 43)  # same size, wrong content
    Path(str(dest) + ".sha256").write_text(
        f"{hashlib.sha256(body).hexdigest()}  file.bin\n", encoding="utf-8"
    )
    _patch_download(monkeypatch, body)
    dl.download_file("http://example.test/f.bin", dest)
    assert dest.read_bytes() == body


# ──────────────── 11. evaluation certificate tamper detection (F-10) ────────────────


def test_eval_certificate_binds_output_and_sources(tmp_path):
    import sys

    import scripts.check_puzzle_overlap as cpo

    train_csv = tmp_path / "tactics_train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
        w.writerow(["tp1", START_FEN, "e2e4 e7e5", "1500"])
        w.writerow(["tp2", START_FEN, "e2e4 e7e6", "1500"])
    eval_csv = tmp_path / "pool.csv"
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
        w.writerow(["tp1", START_FEN, "e2e4 e7e5", "1500"])  # contaminated, dropped
        w.writerow(["ep1", "8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2 b1b2", "1500"])
        w.writerow(["ep2", "8/8/8/8/8/8/8/K6k w - - 0 1", "a1b1 b1b2", "1500"])
    out_csv = tmp_path / "certified.csv"

    rc = cpo.main(
        [
            "--eval-csv",
            str(eval_csv),
            "--train-csv",
            str(train_csv),
            "--output",
            str(out_csv),
            "--max-puzzles",
            "2",
        ]
    )
    assert rc == 0
    manifest = json.loads(Path(str(out_csv) + ".certified.json").read_text(encoding="utf-8"))
    assert manifest["certified"] is True
    assert manifest["output_sha256"] == hashlib.sha256(out_csv.read_bytes()).hexdigest()
    assert manifest["output_row_count"] == 2
    assert manifest["overlap_in_output"] == 0

    from python.data.eval_certification import verify_eval_certification

    ok, reasons, _ = verify_eval_certification(out_csv)
    assert ok, reasons

    # Tamper with the output bytes -> certificate must fail.
    with open(out_csv, "a", encoding="utf-8") as f:
        f.write("ep3,8/8/8/8/8/8/8/K6k w - - 0 1,a1a3 b1b2,1500\n")
    ok, reasons, _ = verify_eval_certification(out_csv)
    assert not ok
    assert any("changed" in r or "row count" in r for r in reasons)

    # Tamper with a training source -> certificate must fail.
    ok, reasons, _ = verify_eval_certification(out_csv)
    with open(train_csv, "a", encoding="utf-8") as f:
        f.write("tp3,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4 d7d5,1500\n")
    ok, reasons, _ = verify_eval_certification(out_csv)
    assert not ok
    assert any("training source" in r for r in reasons)


# ──────────────── 12. trainer refuses invalid data before consuming ────────────────


def test_training_gate_refuses_masked_wdl_data(tmp_path):
    from python.train.train_supervised import train_supervised

    tr = make_split(
        tmp_path / "train",
        4,
        outcomes=[OUTCOME_UNAVAILABLE] * 4,
        metadata=games_wdl_meta(4),
    )
    va = make_split(
        tmp_path / "val",
        2,
        outcomes=[0, 1],
        metadata=games_wdl_meta(2),
    )
    with pytest.raises(DatasetContractError, match="requires WDL"):
        train_supervised(
            train_dir=tr,
            val_dir=va,
            epochs=1,
            max_steps=1,
            checkpoint_dir=tmp_path / "ckpt",
            log_dir=tmp_path / "runs",
        )


def test_trainer_gate_accepts_validated_data(tmp_path):
    """A declared policy-only dataset with a passing receipt passes the gate."""
    from python.train.train_supervised import train_supervised

    meta = {
        "num_samples": 2,
        "role": "policy_tactics",
        "kind": "tactics-fullline",
        "label_semantics": {"role": "policy_tactics", "wdl_supervised": False},
    }
    tr = make_split(tmp_path / "train", 2, outcomes=[255, 255], metadata=meta)
    va = make_split(tmp_path / "val", 2, outcomes=[255, 255], metadata=meta)
    report = validate_split(tr, scan_planes="full", compute_hashes=True)
    report_val = validate_split(va, scan_planes="full", compute_hashes=True)
    write_receipt(tmp_path, {"train": report, "val": report_val}, overall="pass")
    gate = require_validated_dataset(tr, expected_role="policy_tactics", deep=True)
    assert gate.certified is True


# ──────────────── regression: loaders must never rewrite source artifacts ────────────────


def test_gpu_loader_does_not_mutate_outcomes_file(tmp_path):
    """The pre-remediation GpuResidentDataLoader overwrote outcomes.bin in place
    with 255 when tactics_dir was set (an F-01-shaped data destroyer)."""
    import hashlib

    from python.data.dataset import GpuResidentDataLoader

    master = make_split(
        tmp_path / "master",
        3,
        outcomes=[0, 1, 2],
        metadata={**games_wdl_meta(3), "kind": "games"},
    )
    tactics = make_split(
        tmp_path / "tactics",
        2,
        outcomes=[255, 255],
        metadata={
            "num_samples": 2,
            "role": "policy_tactics",
            "kind": "tactics-fullline",
        },
    )
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in master.glob("*.bin")
    }
    loader = GpuResidentDataLoader(
        master,
        batch_size=2,
        device="cpu",
        tactics_dir=tactics,
    )
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in master.glob("*.bin")
        if p.name in before
    }
    assert before == after, "GpuResidentDataLoader must never rewrite dataset arrays on disk"
    loader.close()
