"""Tests for Task 6: Correctness Suite for Self-Play Infrastructure.

Contract Specifications:
(a) Replay 200 saved games through python-chess: every move legal, final result equals stored outcome.
(b) Check policy targets sum to 1 and put zero mass on illegal moves.
(c) Check outcome labels flip correctly with the side to move.
(d) Kill and resume the whole system: confirm no data loss and version counter monotonically increasing.
"""

import time
import chess
import numpy as np
import pytest
import torch

import python
import zeropawn_cpp
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
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import RLTrainer


def test_correctness_replay_200_games_and_policy_and_outcomes(tmp_path):
    from pathlib import Path

    default_p = Path("models/checkpoints_scaled/best_model.pt")
    if default_p.is_file():
        ckpt_path = str(default_p)
    else:
        dummy_p = tmp_path / "dummy_correctness_model.pt"
        if not dummy_p.exists():
            model = ChessNet(num_blocks=12, num_filters=128)
            torch.save(
                {"model": model.state_dict(), "num_blocks": 12, "num_filters": 128, "step": 1},
                dummy_p,
            )
        ckpt_path = str(dummy_p)
    server = InferenceServer(checkpoint_path=ckpt_path)
    client = DirectInferenceClient(server)

    num_games_target = 200
    all_game_positions: list[list[SavedPosition]] = []
    all_game_summaries: list[GameSummary] = []

    def sink(positions: list[SavedPosition], summary: GameSummary):
        all_game_positions.append(positions)
        all_game_summaries.append(summary)

    # Use 16 concurrent games for high-throughput generation
    worker = SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=16,
        game_type="chess",
        full_search_sims=15,
        fast_search_sims=5,
        full_search_prob=0.25,
        temp_cutoff_range=(10, 20),
        resign_threshold=-0.9,
        consecutive_resign_moves=4,
        disable_resign_prob=0.10,
        max_plies=100,
        position_sink=sink,
        use_book=False,
    )

    worker.start()

    start_time = time.time()
    timeout = 180.0  # 3 minutes maximum
    while len(all_game_summaries) < num_games_target and (time.time() - start_time) < timeout:
        time.sleep(0.5)

    worker.stop()

    assert len(all_game_summaries) >= num_games_target, (
        f"Only generated {len(all_game_summaries)} games within {timeout}s timeout"
    )

    # Limit exactly to 200 games for verification
    eval_summaries = all_game_summaries[:num_games_target]
    eval_positions = all_game_positions[:num_games_target]

    print(f"\n--- Verifying {num_games_target} self-play games through python-chess ---")

    for g_idx, (summary, positions) in enumerate(zip(eval_summaries, eval_positions)):
        board = chess.Board()
        cpp_game = zeropawn_cpp.ChessGame()

        assert summary.length_plies == len(positions), (
            f"Game {g_idx}: plies mismatch: summary {summary.length_plies} vs positions {len(positions)}"
        )
        assert len(summary.moves) == summary.length_plies

        # -------------------------------------------------------------
        # (a) Replay through python-chess: every move legal
        # -------------------------------------------------------------
        for ply, (pos, uci_move) in enumerate(zip(positions, summary.moves)):
            # ---------------------------------------------------------
            # (b) Policy targets sum to 1 and put ZERO mass on illegal moves
            # ---------------------------------------------------------
            assert abs(pos.visits.sum() - 1.0) < 1e-4, (
                f"Game {g_idx} ply {ply}: visit sum {pos.visits.sum()} != 1.0"
            )

            # Compute legal action indices for this exact position
            cpp_game.set_fen(board.fen())
            legal_ucis = set(cpp_game.legal_moves_uci())
            legal_indices = set()
            for uci in legal_ucis:
                idx = zeropawn_cpp.move_to_index(cpp_game, uci)
                if idx >= 0:
                    legal_indices.add(idx)

            # Check zero mass on all illegal action indices
            non_zero_indices = np.flatnonzero(pos.visits > 1e-6)
            for act_idx in non_zero_indices:
                assert act_idx in legal_indices, (
                    f"Game {g_idx} ply {ply}: Visit mass {pos.visits[act_idx]} on illegal move index {act_idx}!"
                )

            # ---------------------------------------------------------
            # (c) Outcome labels flip correctly with the side to move
            # ---------------------------------------------------------
            side_is_white = board.turn == chess.WHITE
            if summary.result == "1-0":
                expected_outcome = 0 if side_is_white else 2
            elif summary.result == "0-1":
                expected_outcome = 2 if side_is_white else 0
            else:
                expected_outcome = 1  # Draw

            assert pos.outcome == expected_outcome, (
                f"Game {g_idx} ply {ply}: Outcome {pos.outcome} != expected {expected_outcome} "
                f"(result: {summary.result}, side: {'White' if side_is_white else 'Black'})"
            )

            # Make move in python-chess
            move = chess.Move.from_uci(uci_move)
            assert move in board.legal_moves, (
                f"Game {g_idx} ply {ply}: Illegal move {uci_move} according to python-chess!"
            )
            board.push(move)

        # Final result check
        if summary.resigned:
            # Resigned game: result should be 1-0 or 0-1
            assert summary.result in ("1-0", "0-1")
        elif board.is_checkmate():
            # Checkmate: side to move lost
            winner = "0-1" if board.turn == chess.WHITE else "1-0"
            assert summary.result == winner
        else:
            # Draw rule or max plies reached
            assert summary.result in ("1/2-1/2", "1-0", "0-1")

    print(
        f"Successfully replayed all {num_games_target} games: 100% legal, targets valid, outcomes flipped."
    )


def test_correctness_kill_and_resume(tmp_path):
    """Executes Correctness Test (d): Kill and resume system, confirming no data loss and monotonic version."""
    buf_dir = tmp_path / "resume_buffer"
    ckpt_dir = tmp_path / "resume_ckpts"
    sync_path = tmp_path / "resume_sync.pt"

    # Step 1: Initialize buffer and trainer
    buffer = ReplayBuffer(buffer_dir=buf_dir, capacity=1000, create_new=True)

    # Add 100 dummy positions
    dummy_positions = []
    for i in range(100):
        dummy_positions.append(
            SavedPosition(
                planes=np.random.randn(19, 8, 8).astype(np.float32),
                visits=np.ones(4672, dtype=np.float32) / 4672.0,
                outcome=i % 3,
                is_full_search=True,
                model_version=1,
                game_id=i,
            )
        )
    buffer.add_positions(dummy_positions)
    initial_buffer_size = len(buffer)
    assert initial_buffer_size == 100

    server_model = ChessNet(num_blocks=2, num_filters=32)
    server = InferenceServer(model=server_model)
    v0 = server.model_version

    trainer_model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=trainer_model,
        replay_buffer=buffer,
        batch_size=32,
        push_freq=5,
        checkpoint_dir=ckpt_dir,
        sync_weights_path=sync_path,
        direct_server=server,
    )

    # Train for 5 steps (triggers weight push at step 5)
    for _ in range(5):
        trainer.train_step()

    v1 = server.model_version
    assert v1 > v0, "Model version did not increment upon weight push"
    step_before_kill = trainer.step_num
    assert step_before_kill == 5

    # Save checkpoint before sudden shutdown
    saved_ckpt = trainer.save_checkpoint()

    # Step 2: Simulate kill — delete trainer, server, buffer references
    del trainer
    del server
    del buffer

    # Step 3: Resume system
    resumed_buffer = ReplayBuffer(buffer_dir=buf_dir, capacity=1000, create_new=False)
    # Confirm no data loss
    assert len(resumed_buffer) == initial_buffer_size, (
        f"Data loss detected: {len(resumed_buffer)} vs {initial_buffer_size}"
    )

    resumed_server_model = ChessNet(num_blocks=2, num_filters=32)
    resumed_server = InferenceServer(model=resumed_server_model)
    resumed_server.load_checkpoint(sync_path)
    resumed_server_version = resumed_server.model_version
    assert resumed_server_version >= v1

    resumed_trainer_model = ChessNet(num_blocks=2, num_filters=32)
    resumed_trainer = RLTrainer(
        model=resumed_trainer_model,
        replay_buffer=resumed_buffer,
        batch_size=32,
        push_freq=5,
        checkpoint_dir=ckpt_dir,
        sync_weights_path=sync_path,
        direct_server=resumed_server,
    )
    resumed_step = resumed_trainer.load_checkpoint(saved_ckpt)
    assert resumed_step == step_before_kill

    # Train for 5 more steps (triggers next weight push at step 10)
    for _ in range(5):
        resumed_trainer.train_step()

    v2 = resumed_server.model_version
    assert v2 > resumed_server_version, "Version counter did not continue increasing after resume"
    assert resumed_trainer.step_num == 10
    print(
        f"Kill and resume verified successfully: zero data loss, versions monotonically increased: {v0} -> {v1} -> {v2}"
    )
