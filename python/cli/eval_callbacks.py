"""Shared serial MCTS evaluator factory for the CLI play paths.

``match live``, ``match human`` and ``puzzles solve`` evaluate positions
through the same contract as every other inference consumer:

* encode the *passed* C++ game object (19x8x8, side-to-move perspective),
* one forward pass, no softmax inside the model,
* return ``(policy_probs, scalar_value)`` with exactly one softmax per head
  and ``scalar_value = P(win) - P(loss)`` relative to the side to move
  (WDL class order ``[win, draw, loss]``).

The encoder call MUST receive the game object itself.  The game carries the
real move history, so history-derived planes (18 = one-fold repetition) and
fields (halfmove clock) match the training encoder.  Rebuilding the position
with ``ChessGame(game.get_fen())`` and encoding that instead silently drops
the history and zeroes plane 18 (audit M-1, 2026-10-31) — the same defect
class as H-2, which was fixed in the library paths but missed these three
inline CLI callbacks.  A single shared factory keeps the contract in one
place so a future fix cannot reach only some of the play paths.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from python.model.chess_net import ChessNet


def make_serial_eval_fn(
    model: Any,
    device: str,
    cpp_module: Optional[Any] = None,
) -> Callable[[Any], Tuple[list, float]]:
    """Build the serial evaluator used by the CLI play commands.

    Args:
        model: ChessNet-like callable returning ``(policy_logits, wdl_logits)``.
        device: ``"cpu"`` or ``"cuda"`` (channels-last input on CUDA, matching
            the other CLI/UCI inference paths).
        cpp_module: Optional injected ``zeropawn_cpp`` module (tests).  The
            real extension is imported lazily here so CLI startup keeps the
            per-command "extension not compiled" handling.

    Returns:
        ``eval_fn(game) -> (policy_probs: list[float], value: float)``
    """
    if cpp_module is None:
        import zeropawn_cpp as cpp_module

    def eval_fn(game: Any) -> Tuple[list, float]:
        # Direct game-state encoding: ``game`` carries the real move history
        # (plane 18 repetition flag, halfmove clock).  Never FEN round-trip.
        planes = cpp_module.encode_state(game)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            # Exactly one softmax; [win, draw, loss] relative to the side to move.
            value = float(ChessNet.wdl_logits_to_value(v_wdl.float()).squeeze(0))
        return (p_probs, value)

    return eval_fn
