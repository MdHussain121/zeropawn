"""Unit test verifying worker tablebase adjudication integration (Task 3).

Covers:
(a) Stop-and-label: game ends at first applicable tablebase position with adjudicated=True.
(b) Cut position has side NOT to move winning:
    - Black to move, White winning (outcome: White=0 win, Black=2 loss).
    - White to move, Black winning (outcome: White=2 loss, Black=0 win).
(c) Multi-ply trajectories: print stored outcomes for several positions from both sides.
(d) Cut position has no sample generated in saved_positions.
(e) Tablebase path OFF (None) preserves standard un-adjudicated behavior.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pytest
import chess

from python.selfplay.worker import SelfPlayGame, GameSummary, WorkerBatchAggregator, SavedPosition


class MockInferenceClient:
    """Mock client returning deterministic uniform policy and neutral value for testing."""

    def __init__(self, action_size: int = 4672) -> None:
        self.action_size = action_size

    def evaluate_leaves(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        batch_size = states.shape[0]
        p_logits = np.zeros((batch_size, self.action_size), dtype=np.float32)
        w_logits = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (batch_size, 1))
        return p_logits, w_logits, 0


def test_adjudication_side_not_to_move_winning():
    """Verify outcome labeling when side NOT to move is winning at cut (Black to move, White winning)."""
    tb_p = Path("data/syzygy")
    if not tb_p.exists() or not any(tb_p.glob("*.rtbw")):
        pytest.skip("data/syzygy tablebases not present in worktree")
    random.seed(0)
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)

    # 6 pieces initially: White K+Q+R vs Black K+r+p.
    # Games plays multiple moves until capture drops pieces to <= 5.
    # At cut: Black to move, White is winning.
    game_a = SelfPlayGame(
        game_id=1,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=5,
        fast_search_sims=5,
        temp_cutoff_range=(0, 0),
        seed=0,
        tablebase_path="data/syzygy",
        tablebase_max_pieces=5,
        max_plies=40,
    )
    game_a.game.set_fen("8/8/8/3p4/8/5K2/4Q3/R5rk w - - 0 1")

    saved_positions, summary = game_a.play_game()

    print("\n--- Adjudication Test: Multi-Ply Trajectory, White Winning ---")
    print(f"Adjudicated: {summary.adjudicated}")
    print(f"Result: {summary.result}")
    print(f"Termination Reason: {summary.termination_reason}")
    print(f"Cut Ply: {summary.cut_ply}")
    print(f"Cut Piece Count: {summary.cut_piece_count}")
    print(f"Raw WDL: {summary.raw_wdl}")
    print(f"Saved Positions Count: {len(saved_positions)}")
    print(f"Moves Played: {summary.moves}")

    assert summary.adjudicated is True
    assert summary.termination_reason == "tablebase_adjudication"
    assert summary.result == "1-0"  # White won
    assert summary.cut_piece_count <= 5

    # No sample created for cut position itself
    assert len(saved_positions) == summary.cut_ply

    # Inspect positions from both sides
    print("Stored outcomes by ply:")
    for idx, pos in enumerate(saved_positions):
        side = "White" if (idx % 2 == 0) else "Black"
        outcome_str = (
            "Win (0)" if pos.outcome == 0 else ("Loss (2)" if pos.outcome == 2 else "Draw (1)")
        )
        print(f"  Ply {idx:2d} ({side:>5} to move): outcome = {pos.outcome} [{outcome_str}]")
        if side == "White":
            assert pos.outcome == 0  # White won -> 0
        else:
            assert pos.outcome == 2  # Black lost -> 2


def test_adjudication_black_winning_white_not_to_move():
    """Verify outcome labeling when White to move at cut, Black winning."""
    tb_p = Path("data/syzygy")
    if not tb_p.exists() or not any(tb_p.glob("*.rtbw")):
        pytest.skip("data/syzygy tablebases not present in worktree")
    random.seed(0)
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)

    # 6 pieces initially: Black K+q+r vs White K+R+P. Black to move.
    # At cut: White to move, Black is winning.
    game_b = SelfPlayGame(
        game_id=2,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=5,
        fast_search_sims=5,
        temp_cutoff_range=(0, 0),
        seed=0,
        tablebase_path="data/syzygy",
        tablebase_max_pieces=5,
        max_plies=40,
    )
    game_b.game.set_fen("r5RK/4q3/5k2/8/3P4/8/8/8 b - - 0 1")

    saved_positions, summary = game_b.play_game()

    print("\n--- Adjudication Test: Multi-Ply Trajectory, Black Winning ---")
    print(f"Adjudicated: {summary.adjudicated}")
    print(f"Result: {summary.result}")
    print(f"Termination Reason: {summary.termination_reason}")
    print(f"Cut Ply: {summary.cut_ply}")
    print(f"Cut Piece Count: {summary.cut_piece_count}")
    print(f"Raw WDL: {summary.raw_wdl}")
    print(f"Saved Positions Count: {len(saved_positions)}")
    print(f"Moves Played: {summary.moves}")

    assert summary.adjudicated is True
    assert summary.termination_reason == "tablebase_adjudication"
    assert summary.result == "0-1"  # Black won
    assert summary.cut_piece_count <= 5

    # No sample created for cut position itself
    assert len(saved_positions) == summary.cut_ply

    # Inspect positions from both sides
    print("Stored outcomes by ply:")
    for idx, pos in enumerate(saved_positions):
        side = "Black" if (idx % 2 == 0) else "White"  # Started with Black to move
        outcome_str = (
            "Win (0)" if pos.outcome == 0 else ("Loss (2)" if pos.outcome == 2 else "Draw (1)")
        )
        print(f"  Ply {idx:2d} ({side:>5} to move): outcome = {pos.outcome} [{outcome_str}]")
        if side == "Black":
            assert pos.outcome == 0  # Black won -> 0
        else:
            assert pos.outcome == 2  # White lost -> 2


def test_adjudication_disabled_parity():
    """Verify that when tablebase_path is None, adjudication never triggers."""
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)

    game_c = SelfPlayGame(
        game_id=3,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=5,
        fast_search_sims=5,
        tablebase_path=None,  # OFF
        tablebase_max_pieces=5,
        max_plies=10,
    )
    # Set to 4-piece endgame
    game_c.game.set_fen("8/8/8/8/8/5K2/4Q3/7k w - - 0 1")

    saved_positions, summary = game_c.play_game()
    assert summary.adjudicated is False
    assert summary.cut_ply is None
    assert summary.termination_reason != "tablebase_adjudication"
