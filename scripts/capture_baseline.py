"""Capture all Part 0 baseline snapshots before renaming to ZeroPawn.

Saves:
- Perft values
- C++ golden visits (20 fixed FENs with stub and best_model.pt)
- Checkpoint loading integrity & forward outputs for all models/
- Scripted UCI session transcript
- 5-step supervised training run
- 2-game self-play run
- 10-puzzle evaluation
- Complete --help output of CLI commands
All saved under docs/rename/before/
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import python
import chessrl_cpp
from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import WorkerBatchAggregator, SelfPlayGame
from python.eval.puzzles import load_puzzles, PuzzleEvaluator

OUTPUT_DIR = REPO_ROOT / "docs" / "rename" / "before"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
    "rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
    "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
    "8/8/8/8/8/5K2/4Q3/7k w - - 0 1",
    "7k/5K2/8/8/8/8/8/R7 w - - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/5n2/3p4/3P4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "r3k2r/pppq1ppp/2n1pn2/2pp4/2PP4/2N1PN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "8/5pk1/5p1p/8/8/1BK5/8/8 w - - 0 1",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "7K/5P1p/6k1/8/8/8/8/8 w - - 0 1",
    "8/2K5/8/8/8/8/6p1/6k1 b - - 0 1",
    "2r1k2r/ppp2ppp/2n1pn2/2bq4/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w k - 0 1",
    "rnbq1rk1/ppp1ppbp/5np1/3p4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQ - 0 1",
    "r1bq1rk1/pp1n1ppp/2p1pn2/3p4/3P1B2/3BPN2/PPP2PPP/RN1Q1RK1 w - - 0 1",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_perft() -> None:
    print("[1/8] Capturing Perft baseline...")
    results = {}

    # 1. Startpos
    game = chessrl_cpp.ChessGame()
    results["startpos"] = [game.perft(d) for d in range(1, 6)]

    # 2. Kiwipete
    game_k = chessrl_cpp.ChessGame(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    results["kiwipete"] = [game_k.perft(d) for d in range(1, 5)]

    # 3. Position 3
    game_3 = chessrl_cpp.ChessGame("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    results["pos3"] = [game_3.perft(d) for d in range(1, 5)]

    out_file = OUTPUT_DIR / "perft_values.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  -> Saved to {out_file}")


def capture_cpp_golden() -> None:
    print("[2/8] Capturing C++ Golden MCTS visits (stub & warm-start model)...")
    model_path = REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt"
    srv = InferenceServer(
        checkpoint_path=str(model_path),
        min_batch_size=1,
        max_batch_size=1,
        max_wait_ms=0.0,
        device="cpu",
    )

    def stub_evaluator(game):
        n = game.num_actions()
        return [1.0 / n] * n, 0.0

    def model_evaluator(game):
        state = chessrl_cpp.encode_state(game)
        p_logits, w_logits, _v = srv.predict_batch(state.reshape(1, 19, 8, 8))
        w_max = np.max(w_logits, axis=-1, keepdims=True)
        w_exp = np.exp(w_logits - w_max)
        w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)
        p_max = np.max(p_logits, axis=-1, keepdims=True)
        p_exp = np.exp(p_logits - p_max)
        p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)
        return p_probs[0].tolist(), float(w_probs[0, 0] - w_probs[0, 2])

    golden_results = {"stub": [], "model": []}

    def run_eval(fens, eval_fn, key, sims=100, seed=42):
        for idx, fen in enumerate(fens):
            game = chessrl_cpp.ChessGame(fen)
            cfg = chessrl_cpp.MCTSConfig()
            cfg.c_puct = 2.0
            cfg.add_noise = False
            cfg.seed = seed
            mcts = chessrl_cpp.MCTS(cfg)
            res = mcts.search(game, sims, eval_fn)
            visits = np.asarray(res.action_visits, dtype=np.float64)
            nz = {
                game.action_to_string(i): int(round(v * sims))
                for i, v in enumerate(visits)
                if v > 0
            }
            best_uci = game.action_to_string(res.best_action) if res.best_action >= 0 else None
            golden_results[key].append(
                {
                    "index": idx,
                    "fen": fen,
                    "best_action": int(res.best_action),
                    "best_uci": best_uci,
                    "root_value": float(res.root_value),
                    "visits": nz,
                    "cache_size": int(mcts.cache_size()),
                }
            )

    run_eval(FIXED_FENS, stub_evaluator, "stub")
    run_eval(FIXED_FENS, model_evaluator, "model")

    out_file = OUTPUT_DIR / "cpp_golden.json"
    out_file.write_text(json.dumps(golden_results, indent=2), encoding="utf-8")
    print(f"  -> Saved {len(FIXED_FENS)} golden positions to {out_file}")


def capture_checkpoints_info() -> None:
    print("[3/8] Capturing Checkpoint loading and forward pass hashes...")
    models_dir = REPO_ROOT / "models"
    dummy_input = torch.zeros(1, 19, 8, 8, dtype=torch.float32)

    ckpt_info = {}
    for pt in sorted(models_dir.rglob("*.pt")):
        rel = str(pt.relative_to(REPO_ROOT)).replace("\\", "/")
        h = sha256_file(pt)
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)

        info = {
            "path": rel,
            "sha256": h,
            "size_bytes": pt.stat().st_size,
        }

        if isinstance(ckpt, dict) and "model" in ckpt:
            info["type"] = "training_checkpoint"
            info["keys"] = list(ckpt.keys())
            info["step"] = ckpt.get("step")
            info["epoch"] = ckpt.get("epoch")
            info["num_blocks"] = ckpt.get("num_blocks")
            info["num_filters"] = ckpt.get("num_filters")

            # Model forward
            net = ChessNet(
                num_blocks=ckpt.get("num_blocks", 12), num_filters=ckpt.get("num_filters", 128)
            )
            net.load_state_dict(ckpt["model"])
            net.eval()
            with torch.no_grad():
                p_logits, v_logits = net(dummy_input)
            info["param_count"] = sum(p.numel() for p in net.parameters())
            info["forward_p_mean"] = float(p_logits.mean().item())
            info["forward_p_std"] = float(p_logits.std().item())
            info["forward_v_mean"] = float(v_logits.mean().item())
            info["forward_v_std"] = float(v_logits.std().item())
        elif isinstance(ckpt, dict):
            info["type"] = "state_dict_or_weights"
            info["keys"] = list(ckpt.keys())[:10]  # sample
            info["total_tensors"] = len(ckpt)
            info["param_count"] = sum(
                v.numel() for v in ckpt.values() if isinstance(v, torch.Tensor)
            )
        ckpt_info[rel] = info

    out_file = OUTPUT_DIR / "checkpoints_info.json"
    out_file.write_text(json.dumps(ckpt_info, indent=2), encoding="utf-8")
    print(f"  -> Saved info for {len(ckpt_info)} checkpoints to {out_file}")


def capture_uci_transcript() -> None:
    print("[4/8] Capturing scripted UCI session transcript...")
    commands = "uci\nisready\nucinewgame\nposition startpos moves e2e4 e7e5\ngo depth 1\nquit\n"
    proc = subprocess.run(
        [sys.executable, "cli.py", "uci"],
        input=commands.encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    stdout_text = proc.stdout.decode("utf-8", errors="replace")
    out_file = OUTPUT_DIR / "uci_transcript.txt"
    out_file.write_text(stdout_text, encoding="utf-8")
    print(f"  -> Saved UCI transcript to {out_file} (exit code: {proc.returncode})")


def capture_training_5step() -> None:
    print("[5/8] Capturing controlled 5-step supervised training run...")
    cmd = [
        sys.executable,
        "cli.py",
        "supervised",
        "--verify",
    ]
    proc = subprocess.run(cmd, capture_output=True, cwd=str(REPO_ROOT))
    stdout_text = proc.stdout.decode("utf-8", errors="replace")
    stderr_text = proc.stderr.decode("utf-8", errors="replace")

    out_file = OUTPUT_DIR / "training_5step.json"
    out_file.write_text(
        json.dumps(
            {
                "exit_code": proc.returncode,
                "stdout": stdout_text[-2000:],
                "stderr": stderr_text[-2000:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  -> Saved training verify result to {out_file} (exit code: {proc.returncode})")


def capture_selfplay_2games() -> None:
    print("[6/8] Capturing 2-game self-play with fixed seed...")
    ckpt_path = REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt"
    server = InferenceServer(checkpoint_path=str(ckpt_path), device="cpu")
    client = DirectInferenceClient(server)
    aggregator = WorkerBatchAggregator(client, max_batch_size=8)

    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()

    games_data = []
    master_seed = 42
    try:
        for gid in [101, 102]:
            seed = master_seed + gid
            game = SelfPlayGame(
                game_id=gid,
                aggregator=aggregator,
                game_type="chess",
                full_search_sims=15,
                fast_search_sims=5,
                full_search_prob=0.5,
                max_plies=15,
                seed=seed,
            )
            positions, summary = game.play_game()
            games_data.append(
                {
                    "game_id": gid,
                    "plies": summary.length_plies,
                    "result": summary.result,
                    "num_positions": len(positions),
                    "full_searches": sum(1 for p in positions if p.is_full_search),
                    "fast_searches": sum(1 for p in positions if not p.is_full_search),
                }
            )
    finally:
        aggregator.stop()

    out_file = OUTPUT_DIR / "selfplay_2games.json"
    out_file.write_text(json.dumps(games_data, indent=2), encoding="utf-8")
    print(f"  -> Saved 2-game selfplay to {out_file}")


def capture_puzzles_10eval() -> None:
    print("[7/8] Capturing 10-puzzle evaluation...")
    model_path = REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt"
    net = ChessNet(num_blocks=12, num_filters=128)
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])

    puzzles = load_puzzles()[:10]
    evaluator = PuzzleEvaluator(net, device="cpu", puzzles=puzzles)
    results = evaluator.evaluate_policy()

    out_file = OUTPUT_DIR / "puzzles_10eval.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  -> Saved 10-puzzle eval to {out_file}: {results}")


def capture_help_texts() -> None:
    print("[8/8] Capturing --help texts of all CLI commands...")
    subcommands = [
        [],
        ["train"],
        ["supervised"],
        ["finetune"],
        ["data"],
        ["data", "download"],
        ["data", "filter"],
        ["data", "build"],
        ["models"],
        ["match"],
        ["match", "run"],
        ["eval"],
        ["bench"],
        ["puzzles"],
        ["diagnose"],
        ["status"],
        ["tablebase"],
        ["config"],
        ["presets"],
        ["build"],
    ]
    help_dict = {}
    for sub in subcommands:
        cmd_key = " ".join(["chess"] + sub)
        cmd = [sys.executable, "cli.py"] + sub + ["--help"]
        proc = subprocess.run(cmd, capture_output=True, cwd=str(REPO_ROOT))
        stdout_text = proc.stdout.decode("utf-8", errors="replace")
        help_dict[cmd_key] = {
            "exit_code": proc.returncode,
            "stdout": stdout_text,
        }
    out_file = OUTPUT_DIR / "help_texts.json"
    out_file.write_text(json.dumps(help_dict, indent=2), encoding="utf-8")
    print(f"  -> Saved help texts for {len(help_dict)} commands to {out_file}")


if __name__ == "__main__":
    capture_perft()
    capture_cpp_golden()
    capture_checkpoints_info()
    capture_uci_transcript()
    capture_training_5step()
    capture_selfplay_2games()
    capture_puzzles_10eval()
    capture_help_texts()
    print("\nAll Part 0 baseline snapshots successfully captured in docs/rename/before/!")
