"""F-2 measurement: end-to-end serial search cost of the old vs new UCI
evaluation callback on identical conditions.

Old path (pre-remediation): ChessGame(game.get_fen()) round-trip + policy
returned as a 4,672-element Python list via .tolist().
New path: direct encode_state(game) + contiguous float32 NumPy array.

Both callbacks run the same model, device, position set, and simulation
budget through the real MCTS binding.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import python  # noqa: F401
import zeropawn_cpp
from python.model.chess_net import ChessNet

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/1bp1bppp/p1np4/1p2p3/3PP3/1BN1PN2/PPP3PP/R2Q1RK1 w - - 0 12",
    "8/5pk1/4p1p1/7p/3R3P/4PKP1/5P2/2r5 b - - 6 38",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def build_model(ckpt: str, device: str):
    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model") or ck.get("model_state_dict")
    model.load_state_dict(state)
    model.eval()
    return model


def make_old_callback(model, device, lock):
    """Pre-remediation UCI serial callback: FEN round-trip + .tolist()."""

    def old_eval(game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with lock:
                if device == "cuda":
                    with torch.amp.autocast(device, dtype=torch.bfloat16):
                        p_logits, v_wdl = model(t)
                else:
                    p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

    return old_eval


def make_new_callback(model, device, lock):
    """Post-remediation UCI serial callback: direct encode + NumPy float32."""

    def new_eval(game):
        planes = zeropawn_cpp.encode_state(game)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with lock:
                if device == "cuda":
                    with torch.amp.autocast(device, dtype=torch.bfloat16):
                        p_logits, v_wdl = model(t)
                else:
                    p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (np.ascontiguousarray(p_probs, dtype=np.float32), scalar_val)

    return new_eval


def run_condition(eval_fn, sims: int, reps: int) -> dict:
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.fpu_reduction = 0.2
    cfg.add_noise = False
    times = []
    moves = []
    for rep in range(reps):
        for fen in POSITIONS:
            cfg.seed = 42 + rep
            mcts = zeropawn_cpp.MCTS(cfg)
            g = zeropawn_cpp.ChessGame(fen)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            res = mcts.search(g, sims, eval_fn)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - t0)
            moves.append(g.action_to_string(res.best_action))
            mcts.clear()
    return {
        "median_ms_per_search": round(statistics.median(times) * 1000, 1),
        "mean_sims_per_sec": round(sims / statistics.mean(times), 1),
        "best_moves": sorted(set(moves)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="docs/perf/EVAL_CALLBACK_REMEDIATION_20261026.json")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model(args.checkpoint, device)
    import threading

    lock = threading.Lock()

    results = {
        "date": "2026-10-26",
        "purpose": "F-2 callback fast path, end-to-end serial search cost",
        "checkpoint": args.checkpoint,
        "device": device,
        "sims_per_search": args.sims,
        "positions": len(POSITIONS),
        "reps": args.reps,
    }

    # Warmup (CUDA context, autotune) on both paths before timing either.
    _ = run_condition(make_new_callback(model, device, lock), 16, 1)
    _ = run_condition(make_old_callback(model, device, lock), 16, 1)

    results["old_callback_fen_tolist"] = run_condition(
        make_old_callback(model, device, lock), args.sims, args.reps
    )
    results["new_callback_numpy_direct"] = run_condition(
        make_new_callback(model, device, lock), args.sims, args.reps
    )

    old = results["old_callback_fen_tolist"]["mean_sims_per_sec"]
    new = results["new_callback_numpy_direct"]["mean_sims_per_sec"]
    results["speedup_end_to_end"] = round(new / old, 3)

    out = _repo_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
