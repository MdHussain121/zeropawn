"""Unit test verifying that worker diagnostic hooks (ply_hook and termination_reason)
are passive and produce 100% identical moves and behavior when active vs inactive.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from python.selfplay.worker import SelfPlayGame, GameSummary, WorkerBatchAggregator


class MockInferenceClient:
    """Mock client returning deterministic uniform policy and neutral value for testing."""

    def __init__(self, action_size: int = 4672) -> None:
        self.action_size = action_size

    def evaluate_leaves(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        batch_size = states.shape[0]
        # Uniform policy logits
        p_logits = np.zeros((batch_size, self.action_size), dtype=np.float32)
        # Neutral WDL logits [win, draw, loss] -> equal draw preference
        w_logits = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (batch_size, 1))
        return p_logits, w_logits, 0


def run_seeded_game(
    game_id: int, seed: int, hook_fn=None
) -> tuple[List[str], GameSummary, List[Dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)

    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)

    hook_records = []

    def wrapped_hook(rec):
        hook_records.append(rec)
        if hook_fn:
            hook_fn(rec)

    game = SelfPlayGame(
        game_id=game_id,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=10,
        fast_search_sims=5,
        full_search_prob=0.35,
        consecutive_resign_moves=4,
        disable_resign_prob=1.0,  # Resignation disabled
        max_plies=20,  # Short games for parity testing
        ply_hook=wrapped_hook if hook_fn is not None else None,
        use_book=False,
    )

    saved_positions, summary = game.play_game()
    return summary.moves, summary, hook_records


def test_worker_hook_parity_3_games():
    """Verify that 3 seeded games produce 100% identical moves with ply_hook ON vs OFF."""
    seeds = [1001, 1002, 1003]

    for game_id, seed in enumerate(seeds):
        moves_off, summary_off, _ = run_seeded_game(game_id=game_id, seed=seed, hook_fn=None)
        moves_on, summary_on, records_on = run_seeded_game(
            game_id=game_id, seed=seed, hook_fn=lambda r: None
        )

        # Move sequences must be identical move for move
        assert moves_off == moves_on, f"Game {game_id} (seed {seed}) diverged with hook ON!"
        assert summary_off.result == summary_on.result
        assert summary_off.termination_reason == summary_on.termination_reason
        assert len(records_on) == len(moves_on)

        # Verify hook record fields
        for idx, rec in enumerate(records_on):
            assert rec["game_id"] == game_id
            assert rec["ply"] == idx
            assert rec["side_to_move"] in ("white", "black")
            assert isinstance(rec["root_value"], float)
            assert rec["search_type"] in ("full", "fast")
            assert rec["uci_move"] == moves_on[idx]


def test_game_summary_termination_reason_default():
    """Verify GameSummary termination_reason defaults to 'unknown'."""
    summary = GameSummary(
        game_id=0,
        length_plies=10,
        result="1/2-1/2",
        resigned=False,
        resigned_side_white=None,
        would_have_resigned=False,
        is_false_resign=False,
        duration_sec=1.0,
        total_simulations=100,
        model_version=0,
    )
    assert summary.termination_reason == "unknown"
