"""Pipeline Smoke Test on Connect 4 using the Phase 3 Self-Play & RL Infrastructure.

Contract Specifications:
- Uses the SAME generic Game infrastructure:
  InferenceServer, ReplayBuffer, SelfPlayWorker, RLTrainer.
- Runs for at most 1 hour.
- Reports that it learns (win rate vs random player rising).
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python
import zeropawn_cpp
from python.model.connect4_net import Connect4Net
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import SavedPosition, GameSummary, SelfPlayWorker
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import RLTrainer


def evaluate_connect4_vs_random(model: Connect4Net, num_games: int = 50, sims: int = 35) -> float:
    """Evaluate Connect4Net against a uniform random player."""
    device = next(model.parameters()).device
    model.eval()

    def eval_fn(g):
        c4 = g
        state = c4.encode_state()
        state_t = torch.from_numpy(state).float()
        return model.predict_single(state_t, device=device)

    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    mcts = zeropawn_cpp.MCTS(cfg)

    wins = 0
    draws = 0
    losses = 0

    for g in range(1, num_games + 1):
        agent_player = 1 if (g % 2 != 0) else 2
        game = zeropawn_cpp.Connect4()

        while not game.is_terminal():
            legal = game.legal_actions()
            if game.current_player() == agent_player:
                res = mcts.search(game, sims, eval_fn)
                action = res.best_action
            else:
                action = random.choice(legal)

            game.apply_action(action)

        w = game.winner()
        if w == agent_player:
            wins += 1
        elif w == -1:
            draws += 1
        else:
            losses += 1

    win_rate = float(wins) / float(num_games)
    print(
        f"Eval ({num_games} games): Wins: {wins} ({win_rate * 100:.1f}%), Draws: {draws}, Losses: {losses}"
    )
    return win_rate


def run_connect4_smoke_test(
    train_duration_sec: float = 300.0,  # 5 minutes (max allowed: 1 hour)
    eval_games: int = 50,
) -> Tuple[float, float]:
    """Run full pipeline smoke test on Connect 4 with generic Game infrastructure."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"======================================================================")
    print(f"TASK 7: CONNECT 4 PIPELINE SMOKE TEST ON GENERIC GAME INFRASTRUCTURE")
    print(f"Device: {device} | Max Duration: {train_duration_sec}s")
    print(f"======================================================================")

    # 1. Initialize untrained Connect4Net model
    model = Connect4Net(in_channels=3, num_blocks=3, num_filters=64, num_actions=7).to(device)

    print("\n--- 1. Baseline Evaluation (Untrained Model vs Random Player) ---")
    baseline_win_rate = evaluate_connect4_vs_random(model, num_games=eval_games, sims=25)

    # 2. Setup Phase 3 Self-Play Infrastructure
    tmp_buffer_dir = Path("data/smoke_test_c4_buffer")
    if tmp_buffer_dir.exists():
        import shutil

        shutil.rmtree(tmp_buffer_dir, ignore_errors=True)

    buffer = ReplayBuffer(
        buffer_dir=tmp_buffer_dir,
        capacity=20_000,
        in_channels=3,
        height=6,
        width=7,
        num_actions=7,
        create_new=True,
    )

    server = InferenceServer(
        model=model,
        device=device,
        min_batch_size=16,
        max_batch_size=64,
        max_wait_ms=2.0,
        use_bfloat16=False,  # small network runs fast in fp32
    )
    client = DirectInferenceClient(server)

    def position_sink(positions: list[SavedPosition], summary: GameSummary):
        buffer.add_positions(positions)

    worker = SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=8,
        game_type="connect4",
        full_search_sims=35,
        fast_search_sims=15,
        full_search_prob=0.3,
        dirichlet_alpha=0.8,
        dirichlet_weight=0.25,
        position_sink=position_sink,
    )

    trainer_model = Connect4Net(in_channels=3, num_blocks=3, num_filters=64, num_actions=7).to(
        device
    )
    trainer_model.load_state_dict(model.state_dict())

    trainer = RLTrainer(
        model=trainer_model,
        replay_buffer=buffer,
        device=device,
        base_lr=0.02,
        momentum=0.9,
        weight_decay=1e-4,
        warmup_steps=50,
        total_steps=1000,
        batch_size=64,
        push_freq=15,
        checkpoint_dir="models/checkpoints_c4_smoke",
        sync_weights_path="models/checkpoints_c4_smoke/live_c4.pt",
        direct_server=server,
        use_bfloat16=False,
    )

    print("\n--- 2. Starting Self-Play Generation & RL Training Loop ---")
    worker.start()

    start_train_time = time.time()
    step_count = 0

    # Wait until buffer has initial seed positions
    print("Seeding initial buffer positions from self-play...")
    while len(buffer) < 128 and (time.time() - start_train_time) < train_duration_sec:
        time.sleep(0.5)

    print(f"Initial seed ready ({len(buffer)} positions). Training started...")

    while (time.time() - start_train_time) < train_duration_sec:
        metrics = trainer.train_step()
        step_count = metrics["step"]

        if step_count % 25 == 0:
            elapsed = time.time() - start_train_time
            print(
                f"Step {step_count:4d} | Buffer: {len(buffer):5d} | "
                f"Loss: {metrics['total_loss']:.4f} (Pol: {metrics['policy_loss']:.4f}, WDL: {metrics['wdl_loss']:.4f}) | "
                f"Model Ver: {server.model_version} | Elapsed: {elapsed:.1f}s"
            )

        time.sleep(0.01)

    worker.stop()
    print(f"\nTraining completed: {step_count} steps, {len(buffer)} positions generated.")

    # 3. Post-Training Evaluation
    print("\n--- 3. Post-Training Evaluation (Trained Model vs Random Player) ---")
    # Evaluate with trained model weights
    final_win_rate = evaluate_connect4_vs_random(trainer.model, num_games=eval_games, sims=25)

    print("\n======================================================================")
    print("CONNECT 4 SMOKE TEST SUMMARY")
    print("======================================================================")
    print(f"Baseline Win Rate (Untrained):  {baseline_win_rate * 100:.1f}%")
    print(f"Final Win Rate (After RL):      {final_win_rate * 100:.1f}%")
    delta = (final_win_rate - baseline_win_rate) * 100
    print(f"Win Rate Delta:                 {delta:+.1f}%")
    print(
        f"Result:                         {'PASS (Learns successfully)' if final_win_rate > baseline_win_rate else 'FAIL'}"
    )
    print("======================================================================")

    assert final_win_rate > baseline_win_rate, (
        f"Model did not learn: final win rate {final_win_rate:.2f} <= baseline {baseline_win_rate:.2f}"
    )

    return baseline_win_rate, final_win_rate


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0  # 3 minutes default
    run_connect4_smoke_test(train_duration_sec=duration, eval_games=40)
