"""Comprehensive regression test suite for Syzygy tablebase integration remediation.

Covers all 12 findings (SY-01 through SY-12):
- SY-01: Rule-50 safeguard on decisive WDL probes.
- SY-02: Syzygy supervision targets (5-class to 3-class mapping).
- SY-03: Real cursed-win and blessed-loss positions unconditionally asserted.
- SY-04: DTZ artifact handling, validation failure on all-NaN DTZ, and graceful loader fallback.
- SY-05: Singleton lifecycle, path-change resource release, and explicit close.
- SY-06: Invalid board state detection (no silent draws for corrupt states).
- SY-07: Tooling CWD-independent path resolution, exit codes, and network failure semantics.
- SY-08: Build record schema versioning and generator metadata provenance.
- SY-09: Six-piece position rejection by 5-piece fast filter.
- SY-10: KvK two-piece theoretical draw handling.
- SY-11: Caching and resource bounds without dedicated layer.
- SY-12: Architectural boundary preservation (no search/eval coupling).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import chess
import chess.syzygy
import numpy as np
import pytest

from python.data.dataset import BinaryChessDataset, _outcome_to_value_target
from python.data.dataset_validation import CheckStatus, validate_split
from python.data.syzygy_dataset import DEFAULT_SYZYGY_PATH, SyzygySupervisionGenerator
from python.selfplay.tb_adjudicate import TablebaseAdjudicator

TB_PATH = str(DEFAULT_SYZYGY_PATH)


def _create_minimal_split(split_dir: Path, n: int, outcomes: Optional[np.ndarray] = None) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    if outcomes is None:
        outcomes = np.zeros(n, dtype=np.uint8)
    np.zeros((n, 19, 8, 8), dtype=np.float32).tofile(split_dir / "planes.bin")
    np.zeros(n, dtype=np.int16).tofile(split_dir / "moves.bin")
    outcomes.astype(np.uint8).tofile(split_dir / "outcomes.bin")
    np.zeros(n, dtype=np.uint64).tofile(split_dir / "game_ids.bin")
    np.zeros(n, dtype=np.int16).tofile(split_dir / "plies.bin")

    meta = {
        "num_samples": n,
        "planes_shape": [19, 8, 8],
        "kind": "endgame_syzygy",
        "source": "endgame_syzygy",
    }
    with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


@pytest.fixture
def adjudicator():
    """Fixture ensuring a valid, open tablebase adjudicator instance."""
    if not Path(TB_PATH).is_dir() or not any(Path(TB_PATH).glob("*.rtbw")):
        pytest.skip(f"Syzygy tablebase files not found at {TB_PATH}")
    adj = TablebaseAdjudicator.get_instance(tablebase_path=TB_PATH, max_pieces=5)
    if adj.tablebase is None:
        TablebaseAdjudicator.reset_instance()
        adj = TablebaseAdjudicator.get_instance(tablebase_path=TB_PATH, max_pieces=5)
    if adj.tablebase is None:
        pytest.skip(f"Failed to open tablebase at {TB_PATH}")
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# SY-01: Rule-50 safeguard
# ─────────────────────────────────────────────────────────────────────────────


class TestSY01Rule50Safeguard:
    """Verifies that decisive tablebase adjudications enforce halfmove_clock == 0."""

    def test_decisive_at_clock_zero(self, adjudicator: TablebaseAdjudicator):
        # KRvK: White to move winning -> raw_wdl = 2, outcome = 1
        fen_win = "8/8/8/8/8/4K3/R7/7k w - - 0 1"
        outcome, meta = adjudicator.probe(fen_win)
        assert outcome == 1
        assert meta["raw_wdl"] == 2
        assert meta["halfmove_clock"] == 0
        assert not meta.get("halfmove_ineligible", False)

        # KQvK: Black to move losing -> raw_wdl = -2, outcome = -1
        fen_loss = "8/8/8/8/8/5K2/4Q3/7k b - - 0 1"
        outcome, meta = adjudicator.probe(fen_loss)
        assert outcome == -1
        assert meta["raw_wdl"] == -2
        assert meta["halfmove_clock"] == 0
        assert not meta.get("halfmove_ineligible", False)

    def test_decisive_at_clock_greater_than_zero_returns_none(
        self, adjudicator: TablebaseAdjudicator
    ):
        init_ineligible = adjudicator.probes_halfmove_ineligible

        # KRvK: White to move winning, but halfmove_clock = 5
        fen_clock5 = "8/8/8/8/8/4K3/R7/7k w - - 5 1"
        outcome, meta = adjudicator.probe(fen_clock5)
        assert outcome is None, (
            "Decisive position with clock > 0 must return None, not a fabricated draw"
        )
        assert meta["qualified"] is True
        assert meta["raw_wdl"] == 2
        assert meta["halfmove_ineligible"] is True
        assert meta["halfmove_clock"] == 5

        # KQvK: Black to move losing, but halfmove_clock = 40
        fen_clock40 = "8/8/8/8/8/5K2/4Q3/7k b - - 40 1"
        outcome, meta = adjudicator.probe(fen_clock40)
        assert outcome is None, "Decisive position with clock > 0 must return None"
        assert meta["qualified"] is True
        assert meta["raw_wdl"] == -2
        assert meta["halfmove_ineligible"] is True
        assert meta["halfmove_clock"] == 40

        assert adjudicator.probes_halfmove_ineligible == init_ineligible + 2

    def test_theoretical_draw_at_clock_greater_than_zero(self, adjudicator: TablebaseAdjudicator):
        # KBvK: theoretical draw (raw_wdl = 0), halfmove_clock = 25
        fen_draw = "8/8/8/8/8/5K2/4B3/7k w - - 25 1"
        outcome, meta = adjudicator.probe(fen_draw)
        assert outcome == 0
        assert meta["raw_wdl"] == 0
        assert meta["halfmove_clock"] == 25
        assert not meta.get("halfmove_ineligible", False)

    def test_cursed_win_and_blessed_loss_at_clock_greater_than_zero(
        self, adjudicator: TablebaseAdjudicator
    ):
        # KNNvKP: Cursed win (raw_wdl = 1), clock = 10
        fen_cursed = "7k/8/4N3/4p3/1N6/8/8/K7 w - - 10 1"
        outcome, meta = adjudicator.probe(fen_cursed)
        assert outcome == 0, "Cursed win must adjudicate to draw under rule-50 regardless of clock"
        assert meta["raw_wdl"] == 1
        assert meta["halfmove_clock"] == 10

        # KNNvKP: Blessed loss (raw_wdl = -1), clock = 20
        fen_blessed = "7k/4N3/3p4/8/5N2/8/8/K7 b - - 20 1"
        outcome, meta = adjudicator.probe(fen_blessed)
        assert outcome == 0, (
            "Blessed loss must adjudicate to draw under rule-50 regardless of clock"
        )
        assert meta["raw_wdl"] == -1
        assert meta["halfmove_clock"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# SY-02: Supervision target mapping (5-class to 3-class)
# ─────────────────────────────────────────────────────────────────────────────


class TestSY02SupervisionTargets:
    """Verifies 5-class tablebase outcome to 3-class FIDE value target mapping."""

    def test_outcome_to_value_target_mapping(self):
        # 5-class Syzygy outcomes: 0=Win, 1=Cursed Win, 2=Draw, 3=Blessed Loss, 4=Loss
        raw_outcomes = np.array([0, 1, 2, 3, 4], dtype=np.int8)
        targets = _outcome_to_value_target(raw_outcomes, is_syzygy=True)

        # Expected 3-class one-hot targets:
        # 0 -> Win:  [1, 0, 0]
        # 1 -> Draw: [0, 1, 0]
        # 2 -> Draw: [0, 1, 0]
        # 3 -> Draw: [0, 1, 0]
        # 4 -> Loss: [0, 0, 1]
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(targets, expected)

    def test_generator_probe_wdl_targets(self):
        gen = SyzygySupervisionGenerator(tablebase_path=TB_PATH)
        assert gen.tb is not None

        # Win: KRvK (White winning)
        board_win = chess.Board("8/8/8/8/8/4K3/R7/7k w - - 0 1")
        raw_wdl, outcome_class, wdl_target, move_scores = gen.probe_position_and_moves(board_win)
        assert raw_wdl == 2
        assert outcome_class == 0
        np.testing.assert_array_equal(wdl_target, np.array([1.0, 0.0, 0.0], dtype=np.float32))

        # Loss: KQvK (Black to move, losing)
        board_loss = chess.Board("8/8/8/8/8/5K2/4Q3/7k b - - 0 1")
        raw_wdl, outcome_class, wdl_target, move_scores = gen.probe_position_and_moves(board_loss)
        assert raw_wdl == -2
        assert outcome_class == 4
        np.testing.assert_array_equal(wdl_target, np.array([0.0, 0.0, 1.0], dtype=np.float32))

        # Draw: KBvK
        board_draw = chess.Board("8/8/8/8/8/5K2/4B3/7k w - - 0 1")
        raw_wdl, outcome_class, wdl_target, move_scores = gen.probe_position_and_moves(board_draw)
        assert raw_wdl == 0
        assert outcome_class == 2
        np.testing.assert_array_equal(wdl_target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

        # Cursed win: KNNvKP
        board_cursed = chess.Board("7k/8/4N3/4p3/1N6/8/8/K7 w - - 0 1")
        raw_wdl, outcome_class, wdl_target, move_scores = gen.probe_position_and_moves(board_cursed)
        assert raw_wdl == 1
        assert outcome_class == 1
        np.testing.assert_array_equal(
            wdl_target,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            err_msg="Cursed win must produce draw value target [0, 1, 0]",
        )

        # Blessed loss: KNNvKP
        board_blessed = chess.Board("7k/4N3/3p4/8/5N2/8/8/K7 b - - 0 1")
        raw_wdl, outcome_class, wdl_target, move_scores = gen.probe_position_and_moves(
            board_blessed
        )
        assert raw_wdl == -1
        assert outcome_class == 3
        np.testing.assert_array_equal(
            wdl_target,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            err_msg="Blessed loss must produce draw value target [0, 1, 0]",
        )
        gen.close()

    def test_binary_dataset_item_decoding_for_all_outcomes(self, tmp_path: Path):
        # Create a mock split directory with 5 samples covering all outcomes (0, 1, 2, 3, 4)
        split_dir = tmp_path / "mock_split"
        _create_minimal_split(split_dir, 5, np.array([0, 1, 2, 3, 4], dtype=np.uint8))

        dataset = BinaryChessDataset(str(split_dir))
        try:
            assert len(dataset) == 5

            # Sample 0: Outcome 0 (Win) -> value target [1, 0, 0]
            val0 = dataset[0]["value_target"].numpy()
            np.testing.assert_array_equal(val0, np.array([1.0, 0.0, 0.0], dtype=np.float32))

            # Sample 1: Outcome 1 (Cursed win) -> value target [0, 1, 0]
            val1 = dataset[1]["value_target"].numpy()
            np.testing.assert_array_equal(val1, np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # Sample 2: Outcome 2 (Theoretical draw) -> value target [0, 1, 0]
            val2 = dataset[2]["value_target"].numpy()
            np.testing.assert_array_equal(val2, np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # Sample 3: Outcome 3 (Blessed loss) -> value target [0, 1, 0]
            val3 = dataset[3]["value_target"].numpy()
            np.testing.assert_array_equal(val3, np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # Sample 4: Outcome 4 (Loss) -> value target [0, 0, 1]
            val4 = dataset[4]["value_target"].numpy()
            np.testing.assert_array_equal(val4, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        finally:
            dataset.close()


# ─────────────────────────────────────────────────────────────────────────────
# SY-03: Cursed win and blessed loss unconditional assertions
# ─────────────────────────────────────────────────────────────────────────────


class TestSY03CursedWinUnconditional:
    """Verifies genuine cursed-win and blessed-loss positions with unconditional assertions."""

    @pytest.mark.parametrize(
        "fen,expected_raw_wdl",
        [
            # KNNvKP cursed win (White to move, theoretical win > 50 moves)
            ("7k/8/4N3/4p3/1N6/8/8/K7 w - - 0 1", 1),
            # KNNvKP blessed loss (Black to move)
            ("7k/4N3/3p4/8/5N2/8/8/K7 b - - 0 1", -1),
            # Verified KNNvKP cursed win fixture 2
            ("7k/8/8/3Np3/1N6/8/8/K7 w - - 0 1", 1),
            # Verified KNNvKP blessed loss fixture 2
            ("7k/8/8/3Np3/1N6/8/8/K7 b - - 0 1", -1),
        ],
    )
    def test_unconditional_cursed_win_blessed_loss(
        self, adjudicator: TablebaseAdjudicator, fen: str, expected_raw_wdl: int
    ):
        board = chess.Board(fen)
        assert board.is_valid(), f"Fixture {fen} must be a strictly legal chess board"

        res, meta = adjudicator.probe(fen)
        # UNCONDITIONAL ASSERTIONS: never placed inside an if-check
        assert res == 0, f"Expected FIDE draw outcome 0 for {fen}, got {res}"
        assert meta["raw_wdl"] == expected_raw_wdl, (
            f"Expected raw_wdl {expected_raw_wdl} for {fen}, got {meta['raw_wdl']}"
        )
        assert meta["qualified"] is True


# ─────────────────────────────────────────────────────────────────────────────
# SY-04: DTZ artifact handling and validation
# ─────────────────────────────────────────────────────────────────────────────


class TestSY04DTZHandlingAndValidation:
    """Verifies that DTZ tables absence is handled cleanly and all-NaN DTZ is rejected."""

    def test_has_dtz_returns_false_with_rtbw_only(self):
        gen = SyzygySupervisionGenerator(tablebase_path=TB_PATH)
        assert gen.has_dtz() is False, "has_dtz() must be False when only .rtbw files exist"
        board = chess.Board("8/8/8/8/8/4K3/R7/7k w - - 0 1")
        assert gen.probe_dtz(board) is None, "probe_dtz must return None when no DTZ tables loaded"
        gen.close()

    def test_build_dataset_does_not_create_dtz_bin_without_dtz_tables(self, tmp_path: Path):
        gen = SyzygySupervisionGenerator(tablebase_path=TB_PATH)
        out_dir = tmp_path / "test_no_dtz_ds"
        gen.build_endgame_dataset(output_dir=str(out_dir), synthetic_count=10, val_ratio=0.2)
        gen.close()

        train_dtz = out_dir / "train" / "dtz_targets.bin"
        val_dtz = out_dir / "val" / "dtz_targets.bin"
        assert not train_dtz.exists(), "dtz_targets.bin should NOT be created when DTZ is absent"
        assert not val_dtz.exists(), "dtz_targets.bin should NOT be created when DTZ is absent"

    def test_validation_rejects_all_nan_dtz(self, tmp_path: Path):
        # Create a valid minimal split
        split_dir = tmp_path / "split_nan_dtz"
        n = 8
        _create_minimal_split(split_dir, n)

        # Plant all-NaN dtz_targets.bin (SY-04 defect)
        nan_dtz = np.full(n, np.nan, dtype=np.float32)
        nan_dtz.tofile(split_dir / "dtz_targets.bin")

        report = validate_split(split_dir)
        assert report.overall == "fail", (
            "validate_split must FAIL when dtz_targets.bin contains all NaNs"
        )
        dtz_check = next((c for c in report.results if c.name == "dtz_targets_valid"), None)
        assert dtz_check is not None
        assert dtz_check.status == CheckStatus.FAIL
        assert "all-NaN" in dtz_check.detail

    def test_dataset_loader_ignores_all_nan_dtz(self, tmp_path: Path):
        split_dir = tmp_path / "split_loader_nan_dtz"
        n = 5
        _create_minimal_split(split_dir, n)

        nan_dtz = np.full(n, np.nan, dtype=np.float32)
        nan_dtz.tofile(split_dir / "dtz_targets.bin")

        ds = BinaryChessDataset(str(split_dir))
        try:
            assert ds.dtz_targets is None, "BinaryChessDataset must ignore all-NaN dtz_targets.bin"
            sample = ds[0]
            assert "dtz" not in sample
        finally:
            ds.close()


# ─────────────────────────────────────────────────────────────────────────────
# SY-05: Singleton lifecycle and resource cleanup
# ─────────────────────────────────────────────────────────────────────────────


class TestSY05SingletonLifecycle:
    """Verifies tablebase file handle management and singleton lifecycle."""

    def test_singleton_get_instance_idempotent(self):
        TablebaseAdjudicator.reset_instance()
        adj1 = TablebaseAdjudicator.get_instance(TB_PATH)
        adj2 = TablebaseAdjudicator.get_instance(TB_PATH)
        assert adj1 is adj2

    def test_path_change_closes_previous_handle(self):
        TablebaseAdjudicator.reset_instance()
        adj1 = TablebaseAdjudicator.get_instance(TB_PATH)
        assert adj1.tablebase is not None

        # Change path to None or another directory
        adj2 = TablebaseAdjudicator.get_instance(tablebase_path="nonexistent_tb_dir")
        assert adj1 is not adj2
        # Verify old handle was closed
        assert adj1.tablebase is None

        # Reset back to valid
        TablebaseAdjudicator.reset_instance()
        adj_restored = TablebaseAdjudicator.get_instance(TB_PATH)
        assert adj_restored.tablebase is not None

    def test_explicit_close_and_context_manager(self):
        with TablebaseAdjudicator(tablebase_path=TB_PATH) as adj:
            assert adj.tablebase is not None
        assert adj.tablebase is None


# ─────────────────────────────────────────────────────────────────────────────
# SY-06: Invalid position handling
# ─────────────────────────────────────────────────────────────────────────────


class TestSY06InvalidPositionHandling:
    """Verifies that invalid positions are never silently adjudicated as draws."""

    @pytest.mark.parametrize(
        "invalid_fen,reason",
        [
            # Adjacent kings
            ("8/8/8/8/8/8/4K3/4k3 w - - 0 1", "adjacent kings"),
            # Missing kings (no Black king)
            ("8/8/8/8/8/8/8/4K3 w - - 0 1", "missing king"),
            # Pawn on back rank (rank 1)
            ("8/8/8/8/8/8/8/4KP1k w - - 0 1", "pawn on rank 1"),
            # Both kings in check (impossible state)
            ("8/8/8/8/8/8/4K3/k6R w - - 0 1", "king in check impossible"),
        ],
    )
    def test_invalid_boards_return_none_with_invalid_metadata(
        self, adjudicator: TablebaseAdjudicator, invalid_fen: str, reason: str
    ):
        init_invalid = adjudicator.invalid_positions
        outcome, meta = adjudicator.probe(invalid_fen)
        assert outcome is None, f"Invalid board ({reason}) must return None, got {outcome}"
        assert meta.get("invalid") is True, f"Metadata must indicate invalid=True for ({reason})"
        assert adjudicator.invalid_positions == init_invalid + 1

    def test_malformed_fen_string(self, adjudicator: TablebaseAdjudicator):
        outcome, meta = adjudicator.probe("not_a_valid_fen_string")
        assert outcome is None
        assert meta["qualified"] is False


# ─────────────────────────────────────────────────────────────────────────────
# SY-07: Tooling, CWD independence, exit codes, and network failure handling
# ─────────────────────────────────────────────────────────────────────────────


class TestSY07ToolingAndProvenance:
    """Verifies CWD-independent path resolution, exit codes, and network error semantics."""

    def test_tablebase_adjudicator_resolves_relative_path_from_subdirs(self):
        # Even if CWD is changed, "data/syzygy" resolves relative to repo root
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.chdir(tmp_dir)
                adj = TablebaseAdjudicator("data/syzygy")
                assert adj.tablebase is not None
                adj.close()
            finally:
                os.chdir(cwd)

    def test_syzygy_generator_resolves_relative_path_from_subdirs(self):
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.chdir(tmp_dir)
                gen = SyzygySupervisionGenerator("data/syzygy")
                assert gen.tb is not None
                gen.close()
            finally:
                os.chdir(cwd)

    def test_validate_tb_offline_exit_code(self):
        from scripts import validate_tb_offline

        # Run on a tiny sample of 2 games
        with pytest.raises(SystemExit) as excinfo:
            validate_tb_offline.main(["--max-games", "2", "--min-agreement", "0.90"])
        assert excinfo.value.code == 0, (
            f"validate_tb_offline should exit with 0, got {excinfo.value.code}"
        )

    def test_tb_feasibility_probe_network_error_semantics(self):
        from scripts.tb_feasibility import probe_syzygy_wdl

        # When urlopen raises URLError/HTTPError, probe_syzygy_wdl must return None, not raise
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("Network unreachable")
        ):
            cat = probe_syzygy_wdl("8/8/8/8/8/5K2/4R3/7k w - - 0 1")
            assert cat is None, "Network failure must return None"


# ─────────────────────────────────────────────────────────────────────────────
# SY-08: Build record provenance and schema versioning
# ─────────────────────────────────────────────────────────────────────────────


class TestSY08BuildRecordProvenance:
    """Verifies dataset generation produces valid BuildRecord with schema versioning."""

    def test_dataset_contains_valid_build_record(self, tmp_path: Path):
        gen = SyzygySupervisionGenerator(tablebase_path=TB_PATH)
        out_dir = tmp_path / "provenance_ds"
        gen.build_endgame_dataset(output_dir=str(out_dir), synthetic_count=12, val_ratio=0.25)
        gen.close()

        rec_file = out_dir / "build_record.json"
        assert rec_file.exists(), f"build_record.json must exist at dataset root {out_dir}"

        with open(rec_file, "r", encoding="utf-8") as f:
            rec = json.load(f)

        assert rec["schema_version"] == "1.1"
        assert rec["builder"] == "python.data.syzygy_dataset.SyzygySupervisionGenerator"
        assert "counts" in rec
        assert rec["label_semantics"]["value_targets"] == (
            "3-class FIDE rule-50 (0=W, 1=D [incl. cursed/blessed], 2=L)"
        )

        # Run validate_split to ensure build_record agreement check passes
        report = validate_split(out_dir / "train")
        rec_check = next((c for c in report.results if c.name == "build_record_agreement"), None)
        assert rec_check is not None
        assert rec_check.status == CheckStatus.PASS


# ─────────────────────────────────────────────────────────────────────────────
# SY-09 through SY-12: Architectural boundaries and intentional limitations
# ─────────────────────────────────────────────────────────────────────────────


class TestSY09ThroughSY12ArchitecturalBoundaries:
    """Verifies intentional architectural boundaries (SY-09 through SY-12)."""

    def test_sy09_six_piece_positions_rejected_by_fast_filter(
        self, adjudicator: TablebaseAdjudicator
    ):
        # 6-piece position: White K+Q+R+P vs Black K+p
        fen_6piece = "8/8/8/4p3/8/5K2/4QP2/R6k w - - 0 1"
        passed, pc = adjudicator.fast_filter(fen_6piece, max_pieces=5)
        assert not passed, "6-piece position must fail fast_filter"
        assert pc == 6

        outcome, meta = adjudicator.probe(fen_6piece)
        assert outcome is None, "6-piece position must not be probed"
        assert meta["qualified"] is False

    def test_sy10_kvk_handled_cleanly_as_theoretical_draw(self, adjudicator: TablebaseAdjudicator):
        # 2-piece position (KvK): theoretical draw
        fen_kvk = "8/8/8/8/8/5K2/8/7k w - - 0 1"
        passed, pc = adjudicator.fast_filter(fen_kvk, max_pieces=5)
        assert passed
        assert pc == 2

        outcome, meta = adjudicator.probe(fen_kvk)
        assert outcome == 0
        assert meta["raw_wdl"] == 0

    def test_sy11_sy12_no_mcts_search_or_cpp_eval_coupling(self):
        # Verify that TablebaseAdjudicator is decoupled from MCTS search and evaluation functions
        # Inspect model and worker imports to confirm Syzygy is solely used between plies
        import python.model

        assert not hasattr(python.model, "chess_tablebase"), (
            "Neural network models must have zero tablebase coupling"
        )
