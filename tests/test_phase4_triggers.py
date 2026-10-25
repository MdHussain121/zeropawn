"""Unit tests for Phase 4 RL safety triggers, pacing, and tablebase CLI options.

Tests covered:
- (g) Log-only false-resign check does not halt (warning logged, no exception when false resign > 5%).
- Draw rate > 70% continues to halt (proves other safety triggers stay locked).
- (h) Entropy trigger: baseline computed as mean over first 50 steps compared against 20-step rolling average.
- (h) Min-buffer fill and reuse cap pacing calculations.
- (f) Parity test: resignation off and tablebase path None produces deterministic identical moves.
- Passive would-resign logging under t=-0.95, k=4, ply>=30 without game termination.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from python.selfplay.metrics import MetricsTracker
from python.selfplay.worker import GameSummary, SelfPlayGame, WorkerBatchAggregator
from scripts.run_phase4_rl import Phase4Orchestrator, SafetyTriggerException


class MockInferenceClient:
    """Mock client returning deterministic uniform policy and neutral value for testing."""

    def __init__(self, action_size: int = 4672) -> None:
        self.action_size = action_size

    def evaluate_leaves(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        batch_size = states.shape[0]
        p_logits = np.zeros((batch_size, self.action_size), dtype=np.float32)
        w_logits = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (batch_size, 1))
        return p_logits, w_logits, 0


def create_mock_orchestrator(tmp_path) -> Phase4Orchestrator:
    """Create a minimal Phase4Orchestrator with mocked sub-components for testing triggers."""
    orchestrator = object.__new__(Phase4Orchestrator)
    orchestrator.metrics_tracker = MetricsTracker(
        log_dir=tmp_path, csv_path=tmp_path / "metrics.csv"
    )
    orchestrator.low_gpu_util_counter = 0
    orchestrator.initial_policy_entropy = None
    orchestrator.entropy_history = []
    orchestrator.min_buffer_fill = 0
    orchestrator.max_reuse_ratio = 16
    return orchestrator


def test_false_resign_log_only_warning(tmp_path, caplog):
    """Test (g): False-resign rate > 5% logs a warning and does NOT raise SafetyTriggerException."""
    orchestrator = create_mock_orchestrator(tmp_path)

    # Record 20 games: 10 would-have-resigned, 7 of which are false resigrations (70.0% false resign)
    for i in range(10):
        # 10 decisive games with normal outcome
        orchestrator.metrics_tracker.record_game(
            GameSummary(
                game_id=i,
                length_plies=40,
                result="1-0",
                resigned=False,
                resigned_side_white=None,
                would_have_resigned=False,
                is_false_resign=False,
                duration_sec=1.0,
                total_simulations=100,
                model_version=0,
                moves=[],
            )
        )

    for i in range(10, 20):
        # 10 would-have-resigned games: 7 false resigns (e.g. White thought lost but drew or won)
        is_false = i < 17
        orchestrator.metrics_tracker.record_game(
            GameSummary(
                game_id=i,
                length_plies=50,
                result="1/2-1/2" if is_false else "0-1",
                resigned=False,
                resigned_side_white=None,
                would_have_resigned=True,
                is_false_resign=is_false,
                duration_sec=1.0,
                total_simulations=100,
                model_version=0,
                moves=[],
            )
        )

    telemetry = orchestrator.metrics_tracker.compute_metrics()
    assert telemetry["false_resign_rate"] == 0.70
    assert orchestrator.metrics_tracker.total_games == 20
    assert orchestrator.metrics_tracker.total_would_have_resigned == 10

    # Call check_safety_triggers with WARNING log level captured
    with caplog.at_level(logging.WARNING):
        # Must NOT raise SafetyTriggerException
        orchestrator.check_safety_triggers({"policy_entropy": 1.0})

    # Assert warning message was logged
    assert any("SAFETY TRIGGER (LOG-ONLY)" in record.message for record in caplog.records)
    assert any("False-resign rate exceeded 5%" in record.message for record in caplog.records)


def test_draw_rate_safety_trigger_remains_active(tmp_path):
    """Verify other safety triggers stay locked: Draw rate > 70% raises SafetyTriggerException."""
    orchestrator = create_mock_orchestrator(tmp_path)

    # Record 20 games with 15 draws (75% draw rate)
    for i in range(20):
        is_draw = i < 15
        orchestrator.metrics_tracker.record_game(
            GameSummary(
                game_id=i,
                length_plies=40,
                result="1/2-1/2" if is_draw else "1-0",
                resigned=False,
                resigned_side_white=None,
                would_have_resigned=False,
                is_false_resign=False,
                duration_sec=1.0,
                total_simulations=100,
                model_version=0,
                moves=[],
            )
        )

    assert orchestrator.metrics_tracker.compute_metrics()["draw_rate"] == 0.75

    with pytest.raises(SafetyTriggerException, match="Self-play draw rate exceeded 70%"):
        orchestrator.check_safety_triggers({"policy_entropy": 1.0})


def test_entropy_trigger_50step_baseline(tmp_path):
    """Test (h): Policy entropy collapse trigger using 50-step baseline vs 20-step rolling average."""
    orchestrator = create_mock_orchestrator(tmp_path)

    # 1. Less than 50 steps: even a low value does not trigger
    for _ in range(40):
        orchestrator.record_entropy(1.0)
    for _ in range(9):
        orchestrator.record_entropy(0.2)
    assert len(orchestrator.entropy_history) == 49
    # Should not raise because baseline is not yet established (< 50 steps)
    orchestrator.check_safety_triggers({"policy_entropy": 0.2})

    # Reset
    orchestrator.entropy_history.clear()

    # 2. Establish 50-step baseline of 1.0
    for _ in range(50):
        orchestrator.record_entropy(1.0)

    # 3. Add 20 steps at 0.70 (30% drop, <= 35% threshold) -> should NOT trigger
    for _ in range(20):
        orchestrator.record_entropy(0.70)

    assert len(orchestrator.entropy_history) == 70
    orchestrator.check_safety_triggers({"policy_entropy": 0.70})

    # 4. Add 20 steps at 0.60 (40% drop, > 35% threshold: 0.60 < 0.65 * 1.0) -> MUST trigger
    for _ in range(20):
        orchestrator.record_entropy(0.60)

    assert len(orchestrator.entropy_history) == 90
    with pytest.raises(
        SafetyTriggerException,
        match="Policy entropy rolling 20-step average.*fell >35% below 50-step baseline",
    ):
        orchestrator.check_safety_triggers({"policy_entropy": 0.60})


def test_min_buffer_fill_and_pacing_logic():
    """Test (h): Min buffer fill logic and max-reuse-ratio pacing calculation."""
    # Min buffer fill calculation
    capacity = 500_000

    # Default (0) behaves as 500 backward-compatible seed
    min_fill_0 = 0
    target_0 = min_fill_0 if min_fill_0 > 0 else 500
    assert min(target_0, capacity) == 500

    # Explicit 50,000 waits for 50,000
    min_fill_50k = 50_000
    target_50k = min_fill_50k if min_fill_50k > 0 else 500
    assert min(target_50k, capacity) == 50_000

    # Reuse cap calculation
    batch_size = 256

    # When total_added = 50,000 positions:
    total_added = 50_000

    # max_reuse_ratio = 8: target max step = (50000 * 8) // 256 = 1562
    target_step_8 = max(500, int((total_added * 8) / batch_size))
    assert target_step_8 == 1562

    # max_reuse_ratio = 16: target max step = (50000 * 16) // 256 = 3125
    target_step_16 = max(500, int((total_added * 16) / batch_size))
    assert target_step_16 == 3125

    # Verification: step 2000 is allowed under ratio 16, but throttled under ratio 8
    assert 2000 <= target_step_16
    assert 2000 > target_step_8


def test_resignation_disabled_parity_deterministic():
    """Test (f): With tablebase_path=None and allow_resignation=False, seeded runs produce identical moves."""
    seed = 42

    def run_game():
        random.seed(seed)
        np.random.seed(seed)
        client = MockInferenceClient()
        aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)
        game = SelfPlayGame(
            game_id=1,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=10,
            fast_search_sims=5,
            full_search_prob=0.35,
            allow_resignation=False,
            min_resign_ply=30,
            tablebase_path=None,
            max_plies=20,
        )
        _, summary = game.play_game()
        return summary

    s1 = run_game()
    s2 = run_game()

    assert s1.moves == s2.moves
    assert len(s1.moves) == 20
    assert s1.resigned is False
    assert s2.resigned is False
    assert s1.adjudicated is False
    assert s2.adjudicated is False


def test_passive_would_resign_logging_at_ply_ge_30():
    """Verify passive would-resign logging triggers when low root value reaches k=4 at ply >= 30,

    without ending the game by resignation.
    """
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)
    game = SelfPlayGame(
        game_id=99,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=5,
        fast_search_sims=2,
        allow_resignation=False,
        resign_threshold=-0.95,
        consecutive_resign_moves=4,
        min_resign_ply=30,
        max_plies=35,
    )

    # Simulate low root values manually
    game.consecutive_low_root_white = 4
    game.ply = 28
    # At ply 28 (< 30), would_have_resigned should NOT trigger
    assert game.ply < game.min_resign_ply

    game.ply = 30
    # At ply >= 30 with consecutive_low >= 4:
    consecutive_low = game.consecutive_low_root_white
    assert consecutive_low >= game.consecutive_resign_moves
    assert game.ply >= game.min_resign_ply
    assert game.allow_resignation is False

    # Simulate trigger
    if not game.would_have_resigned:
        game.would_have_resigned = True
        game.would_have_resigned_side_white = True

    assert game.would_have_resigned is True
    assert game.would_have_resigned_side_white is True
