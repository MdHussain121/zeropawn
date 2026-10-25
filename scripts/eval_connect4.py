#!/usr/bin/env python3
"""Evaluate Connect 4 agent over 100 games against a random player."""

import os
import sys
import random
import torch
import numpy as np

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

_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "cpp"))
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import zeropawn_cpp
from python.model.connect4_net import Connect4Net


def evaluate_vs_random(model_path="models/connect4_model.pt", num_games=100, sims=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Connect4Net().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained weights from {model_path}")
    else:
        print("Model checkpoint not found, running with initial weights")

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
    losses = 0
    draws = 0

    print(f"Playing {num_games} games vs Random Player (sims={sims})...")

    for g in range(1, num_games + 1):
        # Alternate sides: Agent is Player 1 on odd games, Player 2 on even games
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

        if g % 20 == 0 or g == num_games:
            print(
                f"  Game {g:3d}/{num_games} | Wins: {wins:3d} | Draws: {draws:2d} | Losses: {losses:2d} | WinRate: {wins / g * 100:.1f}%"
            )

    print("\n--- Final Results ---")
    print(f"Total Games: {num_games}")
    print(f"Agent Wins:  {wins} ({wins / num_games * 100:.1f}%)")
    print(f"Draws:       {draws} ({draws / num_games * 100:.1f}%)")
    print(f"Losses:      {losses} ({losses / num_games * 100:.1f}%)")
    return wins, draws, losses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Connect 4 agent against random player.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model_path",
        default="models/connect4_model.pt",
        help="Path to Connect 4 model.",
    )
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move.")
    args = parser.parse_args()
    evaluate_vs_random(model_path=args.model_path, num_games=args.games, sims=args.sims)
