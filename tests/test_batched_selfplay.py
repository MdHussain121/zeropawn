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
import time

import numpy as np
import pytest
import torch

import zeropawn_cpp

from python.model.chess_net import ChessNet
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import (
    InferenceError,
    SelfPlayGame,
    SelfPlayWorker,
    WorkerBatchAggregator,
)


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


@pytest.fixture
def real_inference_client(tmp_path):
    """Real-model client (tiny ChessNet) for tests that need position-dependent outputs."""
    ckpt = tmp_path / "dummy_batched_model.pt"
    if not ckpt.exists():
        model = ChessNet(num_blocks=2, num_filters=32)
        torch.save(
            {"model": model.state_dict(), "num_blocks": 2, "num_filters": 32, "step": 1},
            ckpt,
        )
    server = InferenceServer(checkpoint_path=str(ckpt))
    return DirectInferenceClient(server)


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
    aggregator = WorkerBatchAggregator(
        client=FailingInferenceClient(), max_batch_size=4, max_wait_ms=1.0
    )
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
    aggregator = WorkerBatchAggregator(
        client=FailingInferenceClient(), max_batch_size=4, max_wait_ms=1.0
    )
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
    aggregator = WorkerBatchAggregator(
        client=MockInferenceClient(), max_batch_size=8, max_wait_ms=50.0
    )
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


def test_batched_results_order_correctness(real_inference_client):
    """evaluate_planes_batch returns each position's own (policy, value), in row order.

    Distinct positions get distinct policies from the dummy net; each batched
    row must equal the serial single-leaf result for the same position,
    proving results are not shifted or mixed across requests (N-10/H-1).
    """
    aggregator = WorkerBatchAggregator(real_inference_client, max_batch_size=4)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        moves = []
        games = []
        for action_index in range(4):
            game = zeropawn_cpp.ChessGame()
            game.apply_action(game.legal_actions()[action_index])
            moves.append(zeropawn_cpp.encode_state(game))
            games.append(game)

        # Serial reference for the same positions, via fresh equivalent games.
        serial_results = []
        for action_index in range(4):
            twin = zeropawn_cpp.ChessGame()
            twin.apply_action(twin.legal_actions()[action_index])
            serial_results.append(aggregator.evaluate_game_state(twin))

        policy, values = aggregator.evaluate_planes_batch(
            np.stack(moves).reshape(4, -1).astype(np.float32)
        )
        policy = np.asarray(policy)
        assert policy.shape[0] == 4 and np.asarray(values).shape[0] == 4
        for i, (p_s, v_s) in enumerate(serial_results):
            np.testing.assert_allclose(policy[i], np.asarray(p_s), atol=1e-6)
            assert abs(float(values[i]) - float(v_s)) < 1e-6
    finally:
        aggregator.stop()


def test_batched_selfplay_end_to_end(real_inference_client):
    """A full self-play game with search_mode='batched' completes through the real path.

    Before the fix this path raised TypeError (3-arg call vs 4-arg binding);
    the repaired 4-arg call must produce a complete game with finite visit
    distributions and recorded positions.
    """
    collected_positions = []
    collected_summaries = []

    def sink(positions, summary):
        collected_positions.extend(positions)
        collected_summaries.append(summary)

    worker = SelfPlayWorker(
        worker_id=7,
        client=real_inference_client,
        num_concurrent_games=2,
        game_type="chess",
        full_search_sims=16,
        fast_search_sims=8,
        full_search_prob=0.5,
        search_mode="batched",
        position_sink=sink,
    )

    worker.start()
    deadline = time.time() + 120.0
    while len(collected_summaries) < 2 and time.time() < deadline:
        time.sleep(0.5)
    worker.stop()

    assert len(collected_summaries) >= 2, "batched self-play produced no completed games"
    assert len(collected_positions) > 0
    for pos in collected_positions:
        assert pos.planes.shape == (19, 8, 8)
        assert pos.visits.shape == (4672,)
        assert np.isfinite(pos.visits).all()
        assert pos.visits.sum() > 0


def test_batched_game_visit_distribution_sums_to_one(real_inference_client):
    """Root visit distribution from the batched path is normalized (MCTS.search contract)."""
    aggregator = WorkerBatchAggregator(real_inference_client, max_batch_size=4)
    disp = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    disp.start()
    try:
        game = SelfPlayGame(
            game_id=101,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=24,
            fast_search_sims=12,
            full_search_prob=1.0,
            temp_cutoff_range=(10, 15),
            max_plies=6,
            use_book=False,
            search_mode="batched",
        )
        positions, summary = game.play_game()
        assert summary.length_plies > 0
        assert len(positions) > 0
        for pos in positions:
            assert abs(float(pos.visits.sum()) - 1.0) < 1e-3
    finally:
        aggregator.stop()
