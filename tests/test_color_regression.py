"""Color-inversion regression tests (Stage 1, A4).

A position with a known mate for White and one for Black. Pins:
(a) MCTS search finds the mating move for the side to move (both colors);
(b) self-play labeling stores the result and per-position WDL correctly;
(c) match result strings are board-relative for mates by either color.
Any re-introduction of `== 0 / == 1` on the bool side_to_move() API breaks
these (verified: (c) fails on the pre-fix mapping).
"""

from __future__ import annotations

import threading

import chess
import numpy as np
import pytest

import zeropawn_cpp
from python.selfplay.worker import SelfPlayGame, WorkerBatchAggregator
from python.eval.match_runner import engine_outcome

WHITE_MATES_IN_1 = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
WHITE_MATE_MOVE = "h5f7"
BLACK_MATES_IN_1 = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
BLACK_MATE_MOVE = "d8h4"


class _MockClient:
    def evaluate_leaves(self, states):
        n = states.shape[0]
        return (
            np.zeros((n, 4672), np.float32),
            np.tile(np.array([0.0, 1.0, 0.0], np.float32), (n, 1)),
            0,
        )


class _MateMockClient:
    """Concentrates policy mass on one action index; positive mover value."""

    def __init__(self, mate_idx: int) -> None:
        self.mate_idx = mate_idx

    def evaluate_leaves(self, states):
        n = states.shape[0]
        p = np.full((n, 4672), -5.0, np.float32)
        p[:, self.mate_idx] = 5.0
        return (p, np.tile(np.array([3.0, 0.0, -3.0], np.float32), (n, 1)), 0)


def _search_best(fen: str, sims: int = 300) -> str:
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 7
    game = zeropawn_cpp.ChessGame(fen)
    res = zeropawn_cpp.MCTS(cfg).search(game, sims, lambda g: ([1.0 / 4672] * 4672, 0.0))
    return game.action_to_string(res.best_action)


@pytest.mark.parametrize(
    ("fen", "mate_move", "mover_is_white"),
    [(WHITE_MATES_IN_1, WHITE_MATE_MOVE, True), (BLACK_MATES_IN_1, BLACK_MATE_MOVE, False)],
)
def test_search_finds_mate_for_side_to_move(fen, mate_move, mover_is_white):
    board = chess.Board(fen)
    assert (board.turn == chess.WHITE) == mover_is_white
    found = _search_best(fen)
    assert found == mate_move, f"search missed mate: {found} != {mate_move}"
    board.push_uci(found)
    assert board.is_checkmate()
    # Winner is the mover (side NOT to move after mate).
    assert (board.turn == chess.BLACK) == mover_is_white


def _play_from_fen(fen: str, mate_move: str, seed: int):
    mate_idx = zeropawn_cpp.move_to_index(zeropawn_cpp.ChessGame(fen), mate_move)
    assert mate_idx >= 0
    agg = WorkerBatchAggregator(client=_MateMockClient(mate_idx), max_batch_size=8, max_wait_ms=1.0)
    agg.running = True
    disp = threading.Thread(target=agg.dispatcher_loop, daemon=True)
    disp.start()
    try:
        game = SelfPlayGame(
            game_id=seed,
            aggregator=agg,
            game_type="chess",
            full_search_sims=200,
            fast_search_sims=50,
            temp_cutoff_range=(0, 0),
            allow_resignation=False,
            tablebase_path=None,
            max_plies=10,
            seed=seed,
        )
        game.game.set_fen(fen)
        return game.play_game()
    finally:
        agg.running = False


@pytest.mark.parametrize(
    ("fen", "mate_move", "result", "n_positions", "wdl"),
    [
        (WHITE_MATES_IN_1, WHITE_MATE_MOVE, "1-0", 1, 0),
        (BLACK_MATES_IN_1, BLACK_MATE_MOVE, "0-1", 1, 0),
    ],
)
def test_selfplay_labels_mates_from_mover_perspective(fen, mate_move, result, n_positions, wdl):
    saved, summary = _play_from_fen(fen, mate_move, seed=31337)
    assert summary.result == result
    assert summary.termination_reason == "checkmate"
    assert len(saved) == n_positions
    # Stored WDL is from the position's side-to-move perspective (mover wins).
    assert all(p.outcome == wdl for p in saved)
    assert engine_outcome(result, True) == ("win" if result == "1-0" else "loss")
    assert engine_outcome(result, False) == ("win" if result == "0-1" else "loss")
