"""PERF3 Stage 0c — Golden reference: 500 positions x (stub + real model).

20 fixed FENs + 480 sampled from real-model games (fixed seed).
Each position: one MCTS search, sims=SIMS, noise off, fixed seed, fresh MCTS
instance per position (empty cache at start; cache_size recorded after).

Contract: records git commit + deployed .pyd SHA-256 + model checkpoint
SHA-256 in the output header. Checkpointed every 25 positions (resumable).
Watchdog: aborts if free RAM < 2 GB or own RSS > 6 GB (ctypes, no new deps).

Usage:
  python scripts/perf3_golden.py [--sims 400] [--seed 1234]
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import python  # noqa: F401,E402  (adds build/cpp to sys.path)
import zeropawn_cpp  # noqa: E402
from python.selfplay.inference_server import InferenceServer  # noqa: E402

GIT_COMMIT = "f2c8eb4717919f66f8730662d833adb5eb3b4972"
DEPLOYED_PYD_SHA256 = "107A0C891AA24A02BB865819AEC454898FBC636500B1B8248AC9578417C27925"
MODEL_PATH = os.path.join(REPO, "models", "checkpoints_rl", "latest_checkpoint.pt")

FIXED_FENS = [
    # startpos + perft trio
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    # mate-in-1 cases (also used by tests/test_mcts.py)
    "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
    "rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
    "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
    "8/8/8/8/8/5K2/4Q3/7k w - - 0 1",
    "7k/5K2/8/8/8/8/8/R7 w - - 0 1",
    # middlegames / structures / special moves
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/5n2/3p4/3P4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "r3k2r/pppq1ppp/2n1pn2/2pp4/2PP4/2N1PN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    # endgames
    "8/5pk1/5p1p/8/8/1BK5/8/8 w - - 0 1",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "7K/5P1p/6k1/8/8/8/8/8 w - - 0 1",
    "8/2K5/8/8/8/8/6p1/6k1 b - - 0 1",
    # opposite-side castling / open tactical
    "2r1k2r/ppp2ppp/2n1pn2/2bq4/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w k - 0 1",
    "rnbq1rk1/ppp1ppbp/5np1/3p4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQ - 0 1",
    "r1bq1rk1/pp1n1ppp/2p1pn2/3p4/3P1B2/3BPN2/PPP2PPP/RN1Q1RK1 w - - 0 1",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Mem(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_ram_gb() -> float:
    st = Mem()
    st.dwLength = ctypes.sizeof(Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / (1024**3)


def own_rss_gb() -> float:
    class PM(ctypes.Structure):
        _fields_ = (
            [("cb", ctypes.c_ulong)]
            + [(f"f{i}", ctypes.c_ulonglong) for i in range(9)]
            + [("WorkingSetSize", ctypes.c_size_t)]
        )

    # PROCESS_QUERY_LIMITED_INFORMATION handle via GetCurrentProcess (pseudo, no close needed)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    counters = PM()
    counters.cb = ctypes.sizeof(PM)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb)
    if not ok:
        return 0.0
    return counters.WorkingSetSize / (1024**3)


def check_watchdog(tag: str) -> None:
    free = free_ram_gb()
    rss = own_rss_gb()
    if free < 2.0:
        raise RuntimeError(f"WATCHDOG at {tag}: free RAM {free:.2f} GB < 2 GB — aborting")
    if rss > 6.0:
        raise RuntimeError(f"WATCHDOG at {tag}: own RSS {rss:.2f} GB > 6 GB — aborting")


def make_evaluators(srv: InferenceServer, fixed: bool = False):
    def model_evaluator_raw(game):
        state = zeropawn_cpp.encode_state(game)
        p_logits, w_logits, _v = srv.predict_batch(state.reshape(1, 19, 8, 8))
        w_max = np.max(w_logits, axis=-1, keepdims=True)
        w_exp = np.exp(w_logits - w_max)
        w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)
        return p_logits[0].tolist(), float(w_probs[0, 0] - w_probs[0, 2])

    def model_evaluator_fixed(game):
        # == perf3-softmax worker patch (full-4672 float32 softmax, max-subtracted)
        state = zeropawn_cpp.encode_state(game)
        p_logits, w_logits, _v = srv.predict_batch(state.reshape(1, 19, 8, 8))
        w_max = np.max(w_logits, axis=-1, keepdims=True)
        w_exp = np.exp(w_logits - w_max)
        w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)
        p_max = np.max(p_logits, axis=-1, keepdims=True)
        p_exp = np.exp(p_logits - p_max)
        p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)
        return p_probs[0].tolist(), float(w_probs[0, 0] - w_probs[0, 2])

    def stub_evaluator(game):
        n = game.num_actions()
        return [1.0 / n] * n, 0.0

    return stub_evaluator, (model_evaluator_fixed if fixed else model_evaluator_raw)


def run_search(fen: str, sims: int, seed: int, evaluator) -> dict:
    game = zeropawn_cpp.ChessGame(fen)
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    cfg.seed = seed
    mcts = zeropawn_cpp.MCTS(cfg)
    res = mcts.search(game, sims, evaluator)
    visits = np.asarray(res.action_visits, dtype=np.float64)
    total = int(round(visits.sum() * sims)) if visits.sum() > 0 else 0
    # sparse legal-only visit map (illegal entries are always exactly 0)
    nz = {
        game.action_to_string(i): int(round(v * sims)) if visits.sum() > 0 else 0
        for i, v in enumerate(visits)
        if v > 0
    }
    best_uci = game.action_to_string(res.best_action) if res.best_action >= 0 else None
    return {
        "fen": fen,
        "n_legal": len(game.legal_moves_uci()),
        "best_action": int(res.best_action),
        "best_uci": best_uci,
        "root_q": float(res.root_value),
        "visit_total": total,
        "visits": nz,
        "cache_size": int(mcts.cache_size()),
    }


def sample_game_positions(model_ev, seed: int, n_games: int, max_plies: int = 120) -> list:
    """Play real-model games (low sims, temperature sampling early) and collect FENs."""
    rng = np.random.default_rng(seed)
    fens: list = []
    for gi in range(n_games):
        game = zeropawn_cpp.ChessGame()
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.c_puct = 2.0
        cfg.add_noise = False
        cfg.seed = seed + gi
        mcts = zeropawn_cpp.MCTS(cfg)
        for ply in range(max_plies):
            if game.is_terminal():
                break
            legal = game.legal_moves_uci()
            if not legal:
                break
            fens.append(game.get_fen())
            res = mcts.search(game, 30, model_ev)
            visits = np.asarray(res.action_visits, dtype=np.float64)
            if ply < 20:
                probs = visits / visits.sum()
                choice = int(rng.choice(len(probs), p=probs))
                uci = game.action_to_string(choice)
                if uci is None:
                    uci = game.action_to_string(res.best_action)
            else:
                uci = game.action_to_string(res.best_action)
            if not game.make_move_uci(uci):
                break
    # dedupe, preserve order
    seen, out = set(), []
    for f in fens:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--fixed", action="store_true", help="fixed (softmax) model path, == perf3-softmax worker"
    )
    ap.add_argument(
        "--reuse-positions",
        type=str,
        default=None,
        help="JSON golden file whose positions to reuse (skips game sampling)",
    )
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "perf3", "golden_500.json"))
    ap.add_argument(
        "--ckpt",
        default=os.path.join(tempfile.gettempdir(), "golden_ckpt.json"),
    )
    args = ap.parse_args()
    ckpt_path = os.path.abspath(args.ckpt)

    print(f"cuda={torch.cuda.is_available()}", flush=True)
    srv = InferenceServer(checkpoint_path=MODEL_PATH)
    stub_ev, model_ev = make_evaluators(srv, fixed=args.fixed)
    model_sha = sha256_file(MODEL_PATH)

    # positions: 20 fixed + sampled from real-model games
    if args.reuse_positions:
        with open(args.reuse_positions) as f:
            old = json.load(f)
        positions = [r["fen"] for r in old["results"]]
        print(f"reusing {len(positions)} positions from {args.reuse_positions}", flush=True)
    else:
        pool = sample_game_positions(model_ev, args.seed, n_games=10)
        print(f"sampled_pool={len(pool)} unique game positions", flush=True)
        positions = list(FIXED_FENS) + [f for f in pool if f not in FIXED_FENS][:480]
    print(f"positions={len(positions)} (fixed=20 sampled={len(positions) - 20})", flush=True)
    assert len(positions) == 500, f"need 500 positions, have {len(positions)}"

    done: dict = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            done = json.load(f).get("results", {})
        print(f"resuming: {len(done)} positions already done", flush=True)

    t_start = time.perf_counter()
    for idx, fen in enumerate(positions):
        if fen in done:
            continue
        check_watchdog(f"pos {idx}")
        t0 = time.perf_counter()
        s = run_search(fen, args.sims, args.seed, stub_ev)
        m = run_search(fen, args.sims, args.seed, model_ev)
        dt = time.perf_counter() - t0
        done[fen] = {"stub": s, "model": m, "sec": round(dt, 2)}
        if (idx + 1) % 25 == 0 or (idx + 1) == len(positions):
            with open(ckpt_path, "w") as f:
                json.dump({"results": done}, f)
            el = time.perf_counter() - t_start
            rate = (idx + 1) / el
            eta = (len(positions) - (idx + 1)) / rate if rate > 0 else -1
            print(
                f"[{idx + 1}/{len(positions)}] elapsed={el / 60:.1f}min eta={eta / 60:.1f}min "
                f"free_ram={free_ram_gb():.1f}GB",
                flush=True,
            )

    out = {
        "header": {
            "git_commit": GIT_COMMIT,
            "deployed_pyd_sha256": DEPLOYED_PYD_SHA256,
            "model_checkpoint": os.path.basename(MODEL_PATH),
            "model_sha256": model_sha,
            "sims": args.sims,
            "seed": args.seed,
            "noise": False,
            "c_puct": 2.0,
            "n_positions": len(positions),
            "n_fixed": 20,
            "fresh_mcts_per_position": True,
            "policy_path": (
                "fixed-softmax (== perf3-softmax aca945e / master worker FIX 2026-10-12)"
                if args.fixed
                else "raw-logits (buggy, pre-fix worker)"
            ),
            "fix_commit": "aca945ef42566d6002b420c399ea864ddd086220" if args.fixed else None,
            "evaluators": {
                "stub": "uniform [1/4672]*4672, value 0.0 (== tests/test_mcts.py uniform_evaluator)",
                "model": "InferenceServer.predict_batch + WDL softmax P(win)-P(loss), raw policy logits "
                "(== worker.py:219-229 semantics)",
            },
        },
        "results": [dict(done[f], fen=f) for f in positions],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(
        f"WROTE {args.out} bytes={os.path.getsize(args.out)} elapsed_min="
        f"{(time.perf_counter() - t_start) / 60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
