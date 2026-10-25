#!/usr/bin/env python3
"""Self-play reinforcement learning training script for Connect 4."""

from __future__ import annotations

import logging
import os
import sys
import time
import random
from collections import deque
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# Windows DLL directory fix
_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    os.add_dll_directory(_winlibs_bin)

_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "cpp"))
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import zeropawn_cpp
from python.model.connect4_net import Connect4Net


def play_selfplay_game(
    model: Connect4Net,
    mcts_sims: int = 30,
    temp: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int]:
    dev = torch.device(device)
    model.eval()
    game = zeropawn_cpp.Connect4()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = True
    cfg.dirichlet_alpha = 0.8  # Higher for Connect 4 (7 actions)
    cfg.dirichlet_weight = 0.25
    cfg.seed = random.randint(1, 1000000)
    mcts = zeropawn_cpp.MCTS(cfg)

    def eval_fn(g: Any) -> tuple[list[float], float]:
        c4 = g
        state = c4.encode_state()
        state_t = torch.from_numpy(state).float()
        return model.predict_single(state_t, device=dev)

    trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []
    step = 0

    while not game.is_terminal():
        current_p = game.current_player()
        state_arr = game.encode_state()

        res = mcts.search(game, mcts_sims, eval_fn)
        visits = np.array(res.action_visits, dtype=np.float32)

        trajectory.append((state_arr, visits, current_p))

        # Sample or pick best action
        legal = game.legal_actions()
        if step < 10 and temp > 0:
            probs = visits[legal]
            if probs.sum() > 1e-6:
                probs = probs / probs.sum()
                action = np.random.choice(legal, p=probs)
            else:
                action = random.choice(legal)
        else:
            action = res.best_action

        game.apply_action(action)
        step += 1

    winner = game.winner()

    # Create training samples
    samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    for state_arr, policy_target, p in trajectory:
        if winner == -1:
            wdl_target = 1  # Draw
        elif winner == p:
            wdl_target = 0  # Win
        else:
            wdl_target = 2  # Loss
        samples.append((state_arr, policy_target, wdl_target))

    return samples, winner


def train_connect4(
    num_games: int = 120,
    sims: int = 35,
    epochs: int = 3,
    batch_size: int = 32,
    save_path: str = "models/connect4_model.pt",
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Connect 4 on {device}...")

    model = Connect4Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    buffer: deque[tuple[np.ndarray, np.ndarray, int]] = deque(maxlen=5000)
    start_time = time.time()

    for g in range(1, num_games + 1):
        samples, winner = play_selfplay_game(model, mcts_sims=sims, device=device)
        buffer.extend(samples)

        if len(buffer) >= batch_size and g % 5 == 0:
            model.train()
            batch = random.sample(buffer, min(len(buffer), batch_size * 2))
            states = torch.tensor(np.array([s[0] for s in batch]), dtype=torch.float32).to(device)
            target_p = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.float32).to(device)
            target_wdl = torch.tensor([s[2] for s in batch], dtype=torch.long).to(device)

            # Normalize raw visit counts into a target distribution.
            target_p = target_p / target_p.sum(dim=1, keepdim=True).clamp_min(1e-8)

            p_logits, wdl_logits = model(states)

            # Losses
            p_loss = F.cross_entropy(p_logits, target_p)
            v_loss = F.cross_entropy(wdl_logits, target_wdl)
            loss = p_loss + 0.5 * v_loss

            if not torch.isfinite(loss.detach()):
                logger.warning("Non-finite loss at game %d; skipping optimizer step", g)
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if g % 20 == 0 or g == num_games:
            elapsed = time.time() - start_time
            print(f"Game {g:3d}/{num_games} | Buffer: {len(buffer):4d} | Elapsed: {elapsed:.1f}s")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return save_path


if __name__ == "__main__":
    train_connect4()
