"""PERF3 Stage 0d — Cache-key staleness probe (READ-ONLY, no source changes).

Question: positions sharing one cache key (Zobrist of pieces/side/castling/ep)
but differing in halfmove-clock plane (17) or repetition plane (18) hit the
same entry. How far is the stored value from a fresh evaluation?

Method: build PAIRS (A, B) with identical get_hash() but different plane 17
(halfmove clock) or plane 18 (repetition flag). Evaluate both with the real
model in big batches. Report mean |vA - vB| and fraction > 0.05 over the
cached scalar value. Policy max-abs diff reported as extra diagnostic.

This measures network sensitivity to the unkeyed planes (upper bound on
staleness error); true hit-rate-weighted production staleness needs live
traffic (Stage C4).

Usage: python scripts/perf3_staleness.py [--pairs 2000] [--seed 7]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import python  # noqa: F401,E402
import zeropawn_cpp  # noqa: E402
from python.selfplay.inference_server import InferenceServer  # noqa: E402

GIT_COMMIT = "f2c8eb4717919f66f8730662d833adb5eb3b4972"
DEPLOYED_PYD_SHA256 = "107A0C891AA24A02BB865819AEC454898FBC636500B1B8248AC9578417C27925"
MODEL_PATH = os.path.join(REPO, "models", "checkpoints_rl", "latest_checkpoint.pt")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_halfmove(fen: str, hm: int) -> str:
    parts = fen.split(" ")
    parts[4] = str(hm)
    return " ".join(parts)


def collect_pool(model_ev, seed: int) -> list:
    """Diverse real-game positions via short model games (also feeds pair roots)."""
    rng = np.random.default_rng(seed)
    fens: list = []
    for gi in range(12):
        game = zeropawn_cpp.ChessGame()
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.c_puct = 2.0
        cfg.add_noise = False
        cfg.seed = seed + 100 + gi
        mcts = zeropawn_cpp.MCTS(cfg)
        for ply in range(100):
            if game.is_terminal():
                break
            if not game.legal_moves_uci():
                break
            fens.append(game.get_fen())
            res = mcts.search(game, 20, model_ev)
            visits = np.asarray(res.action_visits, dtype=np.float64)
            if visits.sum() <= 0:
                break
            if ply < 14:
                choice = int(rng.choice(len(visits), p=visits / visits.sum()))
                uci = game.action_to_string(choice) or game.action_to_string(res.best_action)
            else:
                uci = game.action_to_string(res.best_action)
            if not game.make_move_uci(uci):
                break
    seen, out = set(), []
    for f in fens:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def halfmove_pairs(pool: list, n: int, rng) -> list:
    pairs = []
    for fen in pool:
        if len(pairs) >= n:
            break
        try:
            hm0 = int(fen.split(" ")[4])
        except Exception:
            continue
        for delta in (20, 40, 60, 80):
            if len(pairs) >= n:
                break
            hm1 = (hm0 + delta) % 100
            if hm1 == hm0:
                continue
            b_fen = set_halfmove(fen, hm1)
            a, b = zeropawn_cpp.ChessGame(fen), zeropawn_cpp.ChessGame(b_fen)
            if a.get_hash() != b.get_hash():
                continue  # must share the key by construction; skip if not
            ea = np.asarray(zeropawn_cpp.encode_state(a), dtype=np.float32).reshape(19, 8, 8)
            eb = np.asarray(zeropawn_cpp.encode_state(b), dtype=np.float32).reshape(19, 8, 8)
            if not np.array_equal(ea[17], eb[17]):
                pairs.append((fen, b_fen))
    return pairs


def plane18_of(game) -> float:
    e = np.asarray(zeropawn_cpp.encode_state(game), dtype=np.float32).reshape(19, 8, 8)
    return float(e[18, 0, 0])


def repetition_pairs(pool: list, n: int) -> list:
    """Roots where a 4-ply out-and-back (m1, m2, reverse(m1), reverse(m2))
    returns to the same hash with a different repetition plane.
    Pair = (fen, seq). Acceptance is direct: hash equal AND plane 18 differs.
    Cost per root ~ (#moves)^2 make/unmake quadruples (C++, milliseconds)."""
    pairs = []
    for fen in pool:
        if len(pairs) >= n:
            break
        g = zeropawn_cpp.ChessGame(fen)
        if g.is_terminal():
            continue
        h0 = g.get_hash()
        p0 = plane18_of(g)
        found = None
        for m1 in g.legal_moves_uci():
            if not g.make_move_uci(m1):
                continue
            for m2 in g.legal_moves_uci():
                if not g.make_move_uci(m2):
                    continue
                m3 = m1[2:4] + m1[0:2]
                m4 = m2[2:4] + m2[0:2]
                ok3 = g.make_move_uci(m3)
                ok4 = g.make_move_uci(m4) if ok3 else False
                if ok3 and ok4 and g.get_hash() == h0 and plane18_of(g) != p0:
                    found = (m1, m2, m3, m4)
                if ok4:
                    g.unmake_move()
                if ok3:
                    g.unmake_move()
                g.unmake_move()  # m2
                if found:
                    break
            g.unmake_move()  # m1
            if found:
                break
        if found:
            pairs.append((fen, found))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "perf3", "staleness_0d.json"))
    args = ap.parse_args()

    srv = InferenceServer(checkpoint_path=MODEL_PATH)

    def model_ev(game):
        state = zeropawn_cpp.encode_state(game)
        p_logits, w_logits, _v = srv.predict_batch(state.reshape(1, 19, 8, 8))
        w_max = np.max(w_logits, axis=-1, keepdims=True)
        w_exp = np.exp(w_logits - w_max)
        w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)
        return p_logits[0], float(w_probs[0, 0] - w_probs[0, 2])

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    pool = collect_pool(model_ev, args.seed)
    print(f"pool={len(pool)} positions in {time.perf_counter() - t0:.0f}s", flush=True)

    hm = halfmove_pairs(pool, args.pairs, rng)
    print(f"halfmove_pairs={len(hm)}", flush=True)
    rep = repetition_pairs(pool, max(0, args.pairs - len(hm)))
    print(f"repetition_pairs={len(rep)}", flush=True)

    # build evaluation lists: (fenA, fenB or (fen, seq))
    eval_fens_a: list = []
    eval_fens_b: list = []
    kinds: list = []
    for a, b in hm[: args.pairs]:
        eval_fens_a.append(a)
        eval_fens_b.append(b)
        kinds.append("halfmove")
    for fen, seq in rep:
        if len(kinds) >= args.pairs:
            break
        g = zeropawn_cpp.ChessGame(fen)
        for m in seq:
            assert g.make_move_uci(m), (fen, seq)
        assert g.get_hash() == zeropawn_cpp.ChessGame(fen).get_hash()
        assert plane18_of(g) != plane18_of(zeropawn_cpp.ChessGame(fen))
        eval_fens_a.append(g.get_fen())
        eval_fens_b.append(fen)
        kinds.append("repetition")

    n = len(kinds)
    print(f"total_pairs={n}", flush=True)

    def batch_values(fens: list):
        vals, pols = [], []
        B = 256
        for i in range(0, len(fens), B):
            chunk = fens[i : i + B]
            states = np.stack(
                [
                    np.asarray(
                        zeropawn_cpp.encode_state(zeropawn_cpp.ChessGame(f)), dtype=np.float32
                    ).reshape(19, 8, 8)
                    for f in chunk
                ]
            )
            p_logits, w_logits, _v = srv.predict_batch(states)
            w_max = np.max(w_logits, axis=-1, keepdims=True)
            w_exp = np.exp(w_logits - w_max)
            w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)
            vals.extend((w_probs[:, 0] - w_probs[:, 2]).tolist())
            pols.append(p_logits)
        return np.array(vals), np.concatenate(pols, axis=0) if pols else np.zeros((0, 4672))

    va, pa = batch_values(eval_fens_a)
    vb, pb = batch_values(eval_fens_b)
    adiff = np.abs(va - vb)
    pmax = np.max(np.abs(pa - pb), axis=1) if n else np.array([])

    def stats(mask):
        d = adiff[mask]
        return {
            "n": int(mask.sum()),
            "mean_abs_diff": float(d.mean()) if len(d) else 0.0,
            "frac_above_0.05": float((d > 0.05).mean()) if len(d) else 0.0,
            "max_abs_diff": float(d.max()) if len(d) else 0.0,
        }

    kinds_arr = np.array(kinds)
    report = {
        "header": {
            "git_commit": GIT_COMMIT,
            "deployed_pyd_sha256": DEPLOYED_PYD_SHA256,
            "model_checkpoint": os.path.basename(MODEL_PATH),
            "model_sha256": sha256_file(MODEL_PATH),
            "note": "read-only probe; pairs share get_hash() but differ in plane 17/18. "
            "A=value as stored (variant A), B=fresh eval (variant B).",
        },
        "overall": stats(np.ones(n, dtype=bool)),
        "halfmove": stats(kinds_arr == "halfmove"),
        "repetition": stats(kinds_arr == "repetition"),
        "policy_raw_logit_max_abs_diff_mean": float(pmax.mean()) if n else 0.0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
