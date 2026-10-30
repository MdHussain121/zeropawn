"""Resignation semantics and false-resign telemetry tests (RL-02/E1).

The production feature switch (allow_resignation=True) must keep the per-game
disable_resign_prob calibration sampling alive — under the pre-remediation
logic the feature switch short-circuited the sampling, so would-have-resigned
instrumentation was dead code while the banner claimed resignation was off.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
from python.selfplay.metrics import MetricsTracker
from python.selfplay.worker import GameSummary, SelfPlayGame, WorkerBatchAggregator


class WinningLeafClient:
    """Uniform policy, every leaf evaluates to +1 for the leaf's mover.

    The MCTS backup negates values per ply, so every parent (including the
    root) reads ≈ -1 from its own perspective — forcing the resign condition
    (< -0.95) for the side to move at every search.
    """

    def __init__(self, action_size: int = 4672) -> None:
        self.action_size = action_size
        self.version = 1

    def evaluate_leaves(self, states: np.ndarray):
        n = states.shape[0]
        policy = np.full((n, self.action_size), 1.0 / self.action_size, dtype=np.float32)
        # The aggregator softmaxes WDL logits, so return large logits for a
        # ~[1, 0, 0] distribution: leaf value ≈ +1 -> parent/root ≈ -1.
        wdl = np.tile(np.array([40.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        return policy, wdl, self.version


def _make_game(game_id: int, **kwargs) -> SelfPlayGame:
    aggregator = WorkerBatchAggregator(
        client=WinningLeafClient(), max_batch_size=1, max_wait_ms=1.0
    )
    defaults = dict(
        game_id=game_id,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=6,
        fast_search_sims=3,
        full_search_prob=1.0,
        # Sample by visit distribution for the WHOLE game: greedy play from a
        # uniform 6-sim search repeats positions and ends games by threefold
        # long before the ply-30 resign gate can be observed.
        temp_cutoff_range=(40, 41),
        resign_threshold=-0.95,
        consecutive_resign_moves=4,
        min_resign_ply=30,
        max_plies=40,
        use_book=False,
        seed=game_id,
    )
    defaults.update(kwargs)
    return SelfPlayGame(**defaults)


def test_feature_on_with_disable_prob_one_is_pure_diagnostic():
    """disable_resign_prob=1.0 with the feature ON: no game may resign, and
    every game that reaches the condition logs would-have-resigned."""
    random.seed(7)
    np.random.seed(7)
    game = _make_game(1, allow_resignation=True, disable_resign_prob=1.0)
    assert game.resign_feature_enabled is True
    assert game.allow_resignation is False  # sampled into calibration
    _, summary = game.play_game()
    assert summary.resigned is False
    assert summary.would_have_resigned is True
    assert summary.would_have_resigned_side_white is True


def test_feature_on_with_disable_prob_zero_resigns():
    """disable_resign_prob=0.0: resignation fires at the configured threshold."""
    random.seed(7)
    np.random.seed(7)
    game = _make_game(2, allow_resignation=True, disable_resign_prob=0.0)
    assert game.allow_resignation is True
    _, summary = game.play_game()
    assert summary.resigned is True
    # The losing side to move at ply 30 (white) resigns.
    assert summary.resigned_side_white is True
    assert summary.would_have_resigned is False


def test_feature_off_logs_would_have_resign_and_never_resigns():
    """Legacy diagnostic mode: resignation never fires; condition logged."""
    random.seed(7)
    np.random.seed(7)
    game = _make_game(3, allow_resignation=False, disable_resign_prob=0.0)
    assert game.resign_feature_enabled is False
    assert game.allow_resignation is False
    _, summary = game.play_game()
    assert summary.resigned is False
    assert summary.would_have_resigned is True


def test_sampling_mixture_over_games():
    """disable_resign_prob=0.5 with the feature ON: consecutive games must see
    BOTH outcomes (the pre-remediation code produced resignation in 100%)."""
    outcomes = []
    for i in range(24):
        random.seed(1000 + i)
        game = _make_game(i, allow_resignation=True, disable_resign_prob=0.5)
        outcomes.append(game.allow_resignation)
    assert any(outcomes), "no game sampled with resignation enabled"
    assert not all(outcomes), "no game sampled into the calibration fraction"


def test_metrics_counters_consistent_with_summaries(tmp_path):
    """would-have-resigned counts only calibration games; false-resign rate is
    computed from their actual outcomes."""
    tracker = MetricsTracker(log_dir=tmp_path, csv_path=tmp_path / "m.csv")
    # 3 calibration games (would-have, all drawing) + 1 resigned game.
    for i in range(3):
        tracker.record_game(
            GameSummary(
                game_id=i,
                length_plies=60,
                result="1/2-1/2",
                resigned=False,
                resigned_side_white=None,
                would_have_resigned=True,
                is_false_resign=True,
                duration_sec=1.0,
                total_simulations=100,
                model_version=1,
            )
        )
    tracker.record_game(
        GameSummary(
            game_id=3,
            length_plies=31,
            result="0-1",
            resigned=True,
            resigned_side_white=True,
            would_have_resigned=False,
            is_false_resign=False,
            duration_sec=1.0,
            total_simulations=200,
            model_version=1,
        )
    )
    assert tracker.total_would_have_resigned == 3
    assert tracker.total_false_resigns == 3
    metrics = tracker.compute_metrics()
    assert metrics["false_resign_rate"] == pytest.approx(1.0)
    assert metrics["draw_rate"] == pytest.approx(0.75)


def test_is_false_resign_derived_from_outcome():
    """A calibration game whose 'lost' side went on to draw or win is a false
    resignation; one that actually lost is a true positive."""
    random.seed(11)
    np.random.seed(11)
    # WinningLeafClient makes every node read 'lost' at the root; the game therefore draws via
    # max_plies when resignation is off — the would-have-resigned side (white)
    # drew => false resignation.
    game = _make_game(5, allow_resignation=True, disable_resign_prob=1.0, max_plies=34)
    _, summary = game.play_game()
    assert summary.would_have_resigned is True
    assert summary.is_false_resign is True  # drew instead of losing
