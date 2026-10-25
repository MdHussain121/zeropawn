"""Batched NN evaluator for MCTS search.

Replaces the one-forward-per-simulation callback with a single forward pass
per mini-batch of leaves. Same math (softmax policy, WDL win-minus-loss
value), far fewer Python/CUDA round-trips.

Protocol with C++ ``MCTS.search_batched``:
    input:  float32 numpy array of shape (N, state_size)  [1216 for chess]
    output: (policy (N, 4672) float32, values (N,) float32)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

NUM_PLANES = 19
BOARD_HW = 8
STATE_SIZE = NUM_PLANES * BOARD_HW * BOARD_HW  # 1216
NUM_ACTIONS = 4672


def batched_enabled() -> bool:
    """Opt-in gate for the UNAPPROVED round-1 batched path (default OFF).

    Requires ZEROPAWN_ENABLE_BATCHED_MCTS=1 AND an extension build that
    exports search_batched (callers must check both).
    """
    from python.common.env import get_env

    return get_env("ENABLE_BATCHED_MCTS", "0") == "1"


class BatchedEvaluator:
    """Callable batch evaluator: (N, S) planes -> (policy, values)."""

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        if self.device == "cuda":
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except (RuntimeError, ValueError):
                pass
        self._warmup()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros((8, STATE_SIZE), dtype=np.float32)
            self(dummy)
            if self.device == "cuda":
                torch.cuda.synchronize()
        except (RuntimeError, ValueError, OSError):
            pass

    @torch.no_grad()
    def __call__(self, batch_planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(batch_planes.shape[0])
        if n == 0:
            return (
                np.zeros((0, NUM_ACTIONS), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32))
        t = t.view(n, NUM_PLANES, BOARD_HW, BOARD_HW).to(self.device, non_blocking=True)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                    p_logits, v_wdl = self.model(t)
        else:
            with torch.inference_mode():
                p_logits, v_wdl = self.model(t)
        policy = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
        wdl = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
        values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)
        return policy.astype(np.float32), values


def make_single_eval_fn(model: torch.nn.Module, device: str):
    """Fast single-position fallback (no FEN round-trip).

    Uses the ``encode_game`` binding which encodes the passed game object
    directly instead of serialize-to-FEN + re-parse.
    """
    import zeropawn_cpp

    has_direct = hasattr(zeropawn_cpp, "encode_game")

    def eval_fn(game) -> tuple:
        if has_direct:
            planes = zeropawn_cpp.encode_game(game).reshape(NUM_PLANES, BOARD_HW, BOARD_HW)
        else:  # pragma: no cover - legacy fallback
            chess_g = zeropawn_cpp.ChessGame(game.get_fen())
            planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device, dtype=torch.bfloat16):
                    p_logits, v_wdl = model(t)
            else:
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

    return eval_fn
