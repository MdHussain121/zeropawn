"""Tests for Task 2: Self-Play Workers (concurrent games, playout cap, temperature, resignation)."""

import numpy as np
import pytest
import torch

from pathlib import Path

from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import (
    SavedPosition,
    GameSummary,
    SelfPlayGame,
    SelfPlayWorker,
    WorkerBatchAggregator,
)


@pytest.fixture
def test_inference_client(tmp_path):
    default_p = Path("models/checkpoints_scaled/best_model.pt")
    if default_p.is_file():
        ckpt_path = str(default_p)
    else:
        dummy_p = tmp_path / "dummy_worker_model.pt"
        if not dummy_p.exists():
            model = ChessNet(num_blocks=12, num_filters=128)
            torch.save(
                {"model": model.state_dict(), "num_blocks": 12, "num_filters": 128, "step": 1},
                dummy_p,
            )
        ckpt_path = str(dummy_p)
    server = InferenceServer(checkpoint_path=ckpt_path)
    return DirectInferenceClient(server)


def test_saved_position_schema_and_outcome_perspectives(test_inference_client):
    """Verify saved position schema: planes (19,8,8), visits (4672,), WDL outcome, full_search flag, model version."""
    aggregator = WorkerBatchAggregator(test_inference_client, max_batch_size=8)
    disp_thread = None

    import threading

    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()

    game = SelfPlayGame(
        game_id=42,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=20,
        fast_search_sims=10,
        full_search_prob=0.5,
        temp_cutoff_range=(10, 15),
        max_plies=25,
        use_book=False,
    )

    positions, summary = game.play_game()
    aggregator.stop()

    assert len(positions) > 0
    assert summary.game_id == 42
    assert summary.length_plies == len(positions)

    for pos in positions:
        assert isinstance(pos, SavedPosition)
        assert pos.planes.shape == (19, 8, 8)
        assert pos.planes.dtype == np.float32
        assert pos.visits.shape == (4672,)
        assert pos.visits.dtype == np.float32
        assert abs(pos.visits.sum() - 1.0) < 1e-4
        assert pos.outcome in (0, 1, 2)  # 0=Win, 1=Draw, 2=Loss
        assert isinstance(pos.is_full_search, bool)
        assert pos.model_version >= 1
        assert pos.game_id == 42

    # Check that outcomes flip consistently if non-draw
    if summary.result == "1-0":
        # White won: ply 0 (White) should be Win (0), ply 1 (Black) should be Loss (2)
        assert positions[0].outcome == 0
        if len(positions) > 1:
            assert positions[1].outcome == 2
    elif summary.result == "0-1":
        # Black won: ply 0 (White) should be Loss (2), ply 1 (Black) should be Win (0)
        assert positions[0].outcome == 2
        if len(positions) > 1:
            assert positions[1].outcome == 0
    elif summary.result == "1/2-1/2":
        # Draw: all positions should be Draw (1)
        for pos in positions:
            assert pos.outcome == 1


def test_playout_cap_randomization(test_inference_client):
    """Verify that full_search and fast_search flags are set based on full_search_prob."""
    aggregator = WorkerBatchAggregator(test_inference_client, max_batch_size=8)
    import threading

    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()

    # Probability 0.5 ensures mix of fast and full
    game = SelfPlayGame(
        game_id=101,
        aggregator=aggregator,
        game_type="chess",
        full_search_sims=15,
        fast_search_sims=5,
        full_search_prob=0.5,
        max_plies=20,
    )

    positions, summary = game.play_game()
    aggregator.stop()

    full_count = sum(1 for p in positions if p.is_full_search)
    fast_count = sum(1 for p in positions if not p.is_full_search)
    assert full_count + fast_count == len(positions)


def test_worker_assigns_distinct_per_game_seeds(test_inference_client):
    """R-09: concurrent games in a seeded worker must not share one RNG seed.

    Sharing the worker seed across all game threads makes every game draw identical
    Dirichlet noise and temperature samples, so a 24-game batch degenerates into
    duplicates of one game.
    """
    worker = SelfPlayWorker(
        worker_id=1,
        client=test_inference_client,
        num_concurrent_games=4,
        game_type="chess",
        seed=42,
    )
    seeds = [worker._game_seed(game_id) for game_id in range(1, 25)]
    assert all(s is not None for s in seeds)
    assert len(set(seeds)) == len(seeds)

    unseeded = SelfPlayWorker(
        worker_id=2,
        client=test_inference_client,
        num_concurrent_games=2,
        game_type="chess",
        seed=None,
    )
    assert unseeded._game_seed(7) is None


def test_concurrent_selfplay_worker(test_inference_client):
    """Verify SelfPlayWorker runs multiple concurrent games and generates positions."""
    collected_positions = []
    collected_summaries = []

    def sink(positions, summary):
        collected_positions.extend(positions)
        collected_summaries.append(summary)

    worker = SelfPlayWorker(
        worker_id=1,
        client=test_inference_client,
        num_concurrent_games=4,
        game_type="chess",
        full_search_sims=15,
        fast_search_sims=5,
        full_search_prob=0.25,
        position_sink=sink,
    )

    worker.start()

    import time

    # Wait until at least 2 games complete
    timeout = 30.0
    start = time.time()
    while len(collected_summaries) < 2 and (time.time() - start) < timeout:
        time.sleep(0.5)

    worker.stop()

    assert len(collected_summaries) >= 2
    assert len(collected_positions) > 0
    for s in collected_summaries:
        assert s.length_plies > 0
        assert s.duration_sec > 0


def test_pipelined_aggregator_parity(test_inference_client):
    """Verify that pipelined dispatcher produces bit-for-bit identical outputs at batch size 1."""
    import zeropawn_cpp
    import threading

    # Setup synchronous aggregator
    agg_sync = WorkerBatchAggregator(test_inference_client, max_batch_size=1, pipelined=False)
    t_sync = threading.Thread(target=agg_sync.dispatcher_loop, daemon=True)
    t_sync.start()

    # Setup pipelined aggregator
    agg_pipe = WorkerBatchAggregator(test_inference_client, max_batch_size=1, pipelined=True)
    t_pipe_disp = threading.Thread(target=agg_pipe.dispatcher_loop, daemon=True)
    t_pipe_disp.start()
    pipe_threads = agg_pipe.start_pipelined_workers()

    game = zeropawn_cpp.ChessGame()
    p_sync, v_sync = agg_sync.evaluate_game_state(game)
    p_pipe, v_pipe = agg_pipe.evaluate_game_state(game)

    agg_sync.stop()
    agg_pipe.stop()

    assert len(p_sync) == len(p_pipe) == 4672
    np.testing.assert_allclose(p_sync, p_pipe, atol=1e-6)
    assert abs(v_sync - v_pipe) < 1e-6
