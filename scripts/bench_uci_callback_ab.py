"""Honest paired A/B of the UCI serial leaf callback: slow (FEN round-trip + .tolist())
vs fast (direct encode + contiguous f32), interleaved in one process, fresh tree+MCTS
per search. Usage: python tmp_bench/bench_uci_callback_ab.py
"""

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from python.uci.uci_engine import UCIEngine
import zeropawn_cpp

CKPT = r"C:/Users/<user>/Music/chess/chess-rl/models/checkpoints_scaled/latest_checkpoint.pt"
SIMS = 400
PAIRS = 6


def slow_callback_factory(engine):
    """Replica of the pre-N-07 callback: FEN round-trip + .tolist()."""

    def _slow(game):
        chess_g = engine.zeropawn_cpp.ChessGame(game.get_fen())
        planes = engine.zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(engine.device)
        if engine.device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with engine._lock:
                if engine.device == "cuda":
                    with torch.amp.autocast(engine.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = engine.model(t)
                else:
                    p_logits, v_wdl = engine.model(t)
            p_probs = (
                torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            )
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

    return _slow


def run_once(engine, callback):
    game = zeropawn_cpp.ChessGame()
    mcts = zeropawn_cpp.MCTS(engine.mcts_config)
    t0 = time.perf_counter()
    res = mcts.search(game, SIMS, callback, engine._eval_batch_callback)
    dt = time.perf_counter() - t0
    return dt, res.best_action


def main():
    engine = UCIEngine(checkpoint_path=CKPT, use_book=False)
    fast_cb = engine._eval_callback
    slow_cb = slow_callback_factory(engine)
    print(f"device={engine.device} sims={SIMS} pairs={PAIRS} (interleaved A-B)")

    for _ in range(2):
        run_once(engine, slow_cb)
        run_once(engine, fast_cb)

    slow, fast = [], []
    slow_moves, fast_moves = [], []
    for i in range(PAIRS):
        dt, mv = run_once(engine, slow_cb)
        slow.append(dt)
        slow_moves.append(mv)
        dt, mv = run_once(engine, fast_cb)
        fast.append(dt)
        fast_moves.append(mv)
        print(f"pair {i+1}: slow={slow[-1]*1000:.1f}ms fast={fast[-1]*1000:.1f}ms")

    s_med, f_med = statistics.median(slow), statistics.median(fast)
    print(f"slow: median {s_med*1000:.1f} ms ({SIMS/s_med:.1f} sims/s) min {min(slow)*1000:.1f} max {max(slow)*1000:.1f} moves={set(slow_moves)}")
    print(f"fast: median {f_med*1000:.1f} ms ({SIMS/f_med:.1f} sims/s) min {min(fast)*1000:.1f} max {max(fast)*1000:.1f} moves={set(fast_moves)}")
    print(f"delta: {(s_med - f_med) / s_med * 100:.2f}% wall time saved (fast vs slow, paired)")


if __name__ == "__main__":
    main()
