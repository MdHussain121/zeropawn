"""Tests for Syzygy Tablebase Supervision Pipeline and Perspective Semantics."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from python.data.syzygy_dataset import SyzygySupervisionGenerator


@pytest.fixture
def syzygy_gen() -> SyzygySupervisionGenerator:
    tb_p = Path("data/syzygy")
    if not tb_p.exists() or not any(tb_p.glob("*.rtbw")):
        pytest.skip("data/syzygy tablebases not present in worktree")
    return SyzygySupervisionGenerator(tablebase_path="data/syzygy", max_pieces=5)


def test_tablebase_loaded(syzygy_gen: SyzygySupervisionGenerator):
    """Verify tablebase opened properly with local 3-4-5 tables."""
    assert syzygy_gen.tb is not None


def test_syzygy_eligibility_filter(syzygy_gen: SyzygySupervisionGenerator):
    """Verify eligibility conditions: piece count <= 5 and castling == '-'."""
    # 1. Startpos: 32 pieces, castling 'KQkq' -> ineligible
    start = chess.Board()
    el, pc = syzygy_gen.is_eligible(start)
    assert el is False
    assert pc == 32

    # 2. 6-piece position -> ineligible
    b6 = chess.Board("8/8/4k3/8/8/3K4/2Q1RN2/8 w - - 0 1")
    # pieces: k (1), K (1), Q (1), R (1), N (1) = wait, that is 5
    # Let's make it 6 pieces: k, K, Q, R, B, N:
    b6 = chess.Board("8/8/4k3/8/8/3K4/1BQ1RN2/8 w - - 0 1")
    el6, pc6 = syzygy_gen.is_eligible(b6)
    assert el6 is False
    assert pc6 == 6

    # 3. 3-piece position (KQvK) with no castling -> eligible!
    b3 = chess.Board("8/8/4k3/8/8/3K4/2Q5/8 w - - 0 1")
    el3, pc3 = syzygy_gen.is_eligible(b3)
    assert el3 is True
    assert pc3 == 3

    # 4. Position with castling rights -> ineligible
    b_castling = chess.Board("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")
    el_c, _ = syzygy_gen.is_eligible(b_castling)
    assert el_c is False


def test_syzygy_wdl_perspective_white_and_black(syzygy_gen: SyzygySupervisionGenerator):
    """Verify WDL labels and 3-class value targets across all 5 outcome classes (SY-02)."""
    # 1. White KQvK (White to move, decisive win for White)
    # White side to move: raw_wdl must be +2 (win for White), outcome class = 0 (Win)
    w_win = chess.Board("8/8/4k3/8/8/3K4/2Q5/8 w - - 0 1")
    res_w = syzygy_gen.probe_position_and_moves(w_win)
    assert res_w is not None
    raw_wdl_w, out_w, wdl_target_w, moves_w = res_w
    assert raw_wdl_w == 2
    assert out_w == 0  # 0 = Win for side to move
    assert np.array_equal(wdl_target_w, [1.0, 0.0, 0.0])

    # 2. Black to move in identical losing KQvK position
    # Black side to move: raw_wdl must be -2 (loss for Black), outcome class = 4 (Loss in 5-class)
    b_loss = chess.Board("8/8/4k3/8/8/3K4/2Q5/8 b - - 0 1")
    res_b = syzygy_gen.probe_position_and_moves(b_loss)
    assert res_b is not None
    raw_wdl_b, out_b, wdl_target_b, moves_b = res_b
    assert raw_wdl_b == -2
    assert out_b == 4  # 4 = Loss for side to move (5-class)
    assert np.array_equal(wdl_target_b, [0.0, 0.0, 1.0])

    # 3. Known draw position (KBvK, White to move) -> raw 0, class 2, target [0, 1, 0]
    draw_pos = chess.Board("8/8/8/8/8/5K2/4B3/7k w - - 0 1")
    res_d = syzygy_gen.probe_position_and_moves(draw_pos)
    assert res_d is not None
    raw_wdl_d, out_d, wdl_target_d, moves_d = res_d
    assert raw_wdl_d == 0
    assert out_d == 2  # 2 = Draw
    assert np.array_equal(wdl_target_d, [0.0, 1.0, 0.0])

    # 4. Verified Cursed Win position (KNNvKP, White to move) -> raw +1, class 1, target [0, 1, 0] (SY-02)
    # Under FIDE rule 50, cursed win cannot be won within 50 moves -> theoretical Draw
    cursed_pos = chess.Board("7k/8/4N3/4p3/1N6/8/8/K7 w - - 0 1")
    res_c = syzygy_gen.probe_position_and_moves(cursed_pos)
    assert res_c is not None
    raw_wdl_c, out_c, wdl_target_c, moves_c = res_c
    assert raw_wdl_c == 1
    assert out_c == 1  # 1 = Cursed Win (raw 5-class metadata)
    assert np.array_equal(wdl_target_c, [0.0, 1.0, 0.0]), f"Cursed win value target must be Draw [0, 1, 0], got {wdl_target_c}"

    # 5. Verified Blessed Loss position (KNNvKP, Black to move) -> raw -1, class 3, target [0, 1, 0] (SY-02)
    # Under FIDE rule 50, blessed loss is saved by 50-move rule -> theoretical Draw
    blessed_pos = chess.Board("7k/4N3/3p4/8/5N2/8/8/K7 b - - 0 1")
    res_bl = syzygy_gen.probe_position_and_moves(blessed_pos)
    assert res_bl is not None
    raw_wdl_bl, out_bl, wdl_target_bl, moves_bl = res_bl
    assert raw_wdl_bl == -1
    assert out_bl == 3  # 3 = Blessed Loss (raw 5-class metadata)
    assert np.array_equal(wdl_target_bl, [0.0, 1.0, 0.0]), f"Blessed loss value target must be Draw [0, 1, 0], got {wdl_target_bl}"


def test_syzygy_training_sample_creation(syzygy_gen: SyzygySupervisionGenerator):
    """Verify training sample creation, plane encoding, and optimal move set."""
    board = chess.Board("8/8/4k3/8/8/3K4/2Q5/8 w - - 0 1")
    sample = syzygy_gen.create_training_sample(board, game_id=123)
    assert sample is not None
    assert sample.planes.shape == (19, 8, 8)
    assert 0 <= sample.best_move_idx < 4672
    assert sample.outcome_class == 0
    assert len(sample.optimal_action_indices) >= 1
    assert sample.optimal_policy_target is not None
    assert np.isclose(sample.optimal_policy_target.sum(), 1.0)


def test_build_endgame_dataset(syzygy_gen: SyzygySupervisionGenerator, tmp_path: Path):
    """Verify building a small endgame dataset with binary memmaps."""
    out_dir = tmp_path / "dataset_tb"
    # Use 5 controlled positions
    positions = [
        chess.Board("8/8/4k3/8/8/3K4/2Q5/8 w - - 0 1"),
        chess.Board("8/8/4k3/8/8/3K4/2Q5/8 b - - 0 1"),
        chess.Board("8/8/4k3/8/8/3K4/2R5/8 w - - 0 1"),
        chess.Board("8/8/4k3/8/8/3K4/4P3/8 w - - 0 1"),
        chess.Board("8/8/4k3/8/8/3K4/2B5/8 w - - 0 1"),
    ]
    summary = syzygy_gen.build_endgame_dataset(out_dir, positions=positions, val_ratio=0.2)
    assert summary["total_samples"] == 5
    assert (out_dir / "train" / "metadata.json").is_file()
    assert (out_dir / "val" / "metadata.json").is_file()
    assert (out_dir / "train" / "planes.bin").stat().st_size > 0
    assert (out_dir / "train" / "value_targets.bin").stat().st_size > 0
