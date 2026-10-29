"""Regression tests for batched self-play (H-1) and inference-failure handling (M-3).

H-1: ``SelfPlayGame.play_game`` used to call ``search_batched(game, sims, evaluator)``
     with three arguments while the pybind11 binding requires four
     ``(game, sims, batch_size, batch_evaluator)`` — every ``search_mode="batched"``
     self-play run crashed with TypeError at the first search. The aggregator now
     exposes a raw batch evaluator (planes in, ``(policy, values)`` out) and the
     call passes the batch size.

M-3: a genuine inference failure used to be converted into uniform policies and
     0.0 values *inside a running search*, biasing values toward draw. Failures
     now propagate as ``InferenceError`` and abort the affected search; shutdown
     wake-ups discard the partial game instead of recording fabricated evals.
"""

import threading

import numpy as np
import pytest

from python.selfplay.worker import InferenceError, SelfPlayGame, WorkerBatchAggregator


class MockInferenceClient:
    """Deterministic client: uniform policy logits, draw-leaning WDL."""

    def __init__(self, action_size: int = 4672) -> None:
        self.action_size = action_size
        self.calls = 0

    def evaluate_leaves(self, states: np.ndarray):
        self.calls += 1
        batch_size = states.shape[0]
        p_logits = np.zeros((batch_size, self.action_size), dtype=np.float32)
        w_logits = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (batch_size, 1))
        return p_logits, w_logits, 0


class FailingInferenceClient:
    """Client that simulates a genuine inference failure (e.g. transient CUDA error)."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_leaves(self, states: np.ndarray):
        self.calls += 1
        raise RuntimeError("simulated inference backend failure")


# ==============================================================================
# H-1: aggregator raw batch evaluator contract
# ==============================================================================


def test_aggregator_evaluate_planes_batch_contract():
    """(n, state_size) planes in -> (policy (n, 4672), values (n,)) float32 out."""
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=8, max_wait_ms=1.0)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        planes = np.random.default_rng(0).standard_normal((5, 19 * 8 * 8)).astype(np.float32)
        policy, values = aggregator.evaluate_planes_batch(planes)
        assert policy.shape == (5, 4672)
        assert values.shape == (5,)
        assert policy.dtype == np.float32
        assert values.dtype == np.float32
        # Uniform softmax policy, draw value 0.0 from the mock client.
        assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(values, 0.0, atol=1e-6)
        assert client.calls >= 1
    finally:
        aggregator.stop()


def test_batched_selfplay_smoke():
    """Full batched-mode self-play game must complete (regression for H-1)."""
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=4, max_wait_ms=1.0)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        game = SelfPlayGame(
            game_id=1,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=6,
            fast_search_sims=6,
            full_search_prob=1.0,
            temp_cutoff_range=(0, 0),
            max_plies=6,
            seed=3,
            search_mode="batched",
            use_book=False,
        )
        saved_positions, summary = game.play_game()
        assert summary.length_plies > 0
        assert len(saved_positions) == summary.length_plies
        for pos in saved_positions:
            assert pos.planes.shape == (19, 8, 8)
            assert pos.visits.shape == (4672,)
        assert summary.total_simulations == 6 * summary.length_plies
        assert client.calls >= 1
    finally:
        aggregator.stop()


def test_serial_selfplay_still_works():
    """Control: the default serial self-play contract is unaffected."""
    client = MockInferenceClient()
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        game = SelfPlayGame(
            game_id=2,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=6,
            fast_search_sims=6,
            full_search_prob=1.0,
            temp_cutoff_range=(0, 0),
            max_plies=4,
            seed=5,
            search_mode="serial",
            use_book=False,
        )
        saved_positions, summary = game.play_game()
        assert summary.length_plies > 0
        assert len(saved_positions) == summary.length_plies
    finally:
        aggregator.stop()


# ==============================================================================
# M-3: genuine inference failures must not masquerade as evaluations
# ==============================================================================


def test_inference_failure_aborts_serial_search():
    """A failing backend raises InferenceError out of search instead of
    silently feeding uniform policies into the running tree."""
    aggregator = WorkerBatchAggregator(client=FailingInferenceClient(), max_batch_size=4, max_wait_ms=1.0)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        game = SelfPlayGame(
            game_id=3,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=4,
            fast_search_sims=4,
            full_search_prob=1.0,
            temp_cutoff_range=(0, 0),
            max_plies=2,
            seed=7,
            search_mode="serial",
            use_book=False,
        )
        with pytest.raises(InferenceError):
            game.play_game()
        assert aggregator.inference_failure_count >= 1
    finally:
        aggregator.stop()


def test_inference_failure_aborts_batched_search():
    """Same contract through the legacy batched entry point."""
    aggregator = WorkerBatchAggregator(client=FailingInferenceClient(), max_batch_size=4, max_wait_ms=1.0)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        game = SelfPlayGame(
            game_id=4,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=4,
            fast_search_sims=4,
            full_search_prob=1.0,
            temp_cutoff_range=(0, 0),
            max_plies=2,
            seed=9,
            search_mode="batched",
            use_book=False,
        )
        with pytest.raises(InferenceError):
            game.play_game()
        assert aggregator.inference_failure_count >= 1
    finally:
        aggregator.stop()


def test_stop_wakes_waiters_with_error_not_uniform():
    """Shutdown wake-up must be distinguishable from a real evaluation."""
    aggregator = WorkerBatchAggregator(client=MockInferenceClient(), max_batch_size=8, max_wait_ms=50.0)
    results = []

    def waiter():
        try:
            planes = np.zeros((1, 19 * 8 * 8), dtype=np.float32)
            aggregator.evaluate_planes_batch(planes)
            results.append("returned")
        except InferenceError:
            results.append("inference_error")

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    # Give the waiter time to enqueue, then stop the aggregator.
    for _ in range(100):
        if aggregator.pending:
            break
        threading.Event().wait(0.01)
    aggregator.stop()
    t.join(timeout=5.0)
    assert results == ["inference_error"]
