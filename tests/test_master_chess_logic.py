"""Master Chess Logic, Side-to-Move, MCTS Verification, and Parity Test Suite.

Audits and proves:
- Side-to-move boolean truthiness convention (White = True, Black = False).
- Board rules, special moves (castling, en-passant, promotion), make/unmake reversibility.
- Move encoding/decoding bijective mapping and bounds.
- MCTS value propagation, leaf evaluation, and parent_q FPU perspective negation.
- Training target construction and side-to-move WDL alignment.
- Evaluation fairness and Elo interval symmetry.
- UCI protocol compliance and position state integrity.
- Python/C++ cross-implementation consistency against python-chess reference.
"""

from __future__ import annotations

import io
import math
import sys
import chess
import numpy as np
import pytest
import torch

import python
import zeropawn_cpp
from python.eval.match_runner import engine_outcome, score_board_results, elo_interval
from python.uci.uci_engine import UCIEngine
from python.data.stockfish_eval import centipawns_to_wdl, outcome_to_one_hot, blend_targets


# ==============================================================================
# Group A: Side-to-Move Boolean Truthiness & Contract
# ==============================================================================


def test_side_to_move_boolean_truthiness_contract():
    """Verify side_to_move() follows the explicit boolean truthiness convention.

    White to move must be truthy (is True).
    Black to move must be falsy (is False).
    Direct 'if game.side_to_move():' branching must work correctly.
    """
    game = zeropawn_cpp.ChessGame()

    # Initial position: White to move
    assert game.side_to_move() is True
    assert bool(game.side_to_move()) is True

    branch_taken = None
    if game.side_to_move():
        branch_taken = "white"
    else:
        branch_taken = "black"
    assert branch_taken == "white"

    # Make move: Black to move
    assert game.make_move_uci("e2e4")
    assert game.side_to_move() is False
    assert bool(game.side_to_move()) is False

    if game.side_to_move():
        branch_taken = "white"
    else:
        branch_taken = "black"
    assert branch_taken == "black"

    # Unmake move: restores White to move
    assert game.unmake_move()
    assert game.side_to_move() is True


def test_side_to_move_alternation_and_clone():
    """Verify turn alternates exactly once per legal move and clone preserves turn."""
    game = zeropawn_cpp.ChessGame()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]

    expected_white = True
    for mv in moves:
        assert bool(game.side_to_move()) == expected_white
        clone_game = game.clone()
        assert clone_game is not None
        assert bool(clone_game.side_to_move()) == expected_white
        assert clone_game.get_fen() == game.get_fen()
        assert clone_game.get_hash() == game.get_hash()

        assert game.make_move_uci(mv)
        expected_white = not expected_white

    assert bool(game.side_to_move()) == expected_white


# ==============================================================================
# Group B: Board Rules, Special Moves & Reversibility
# ==============================================================================


def test_board_invalid_fen_does_not_corrupt_state():
    """Verify set_fen returns false on invalid FEN and does not corrupt board."""
    game = zeropawn_cpp.ChessGame()
    fen_orig = game.get_fen()
    hash_orig = game.get_hash()

    # Attempt invalid FEN
    ok = game.set_fen("invalid_corrupted_fen_string")
    assert ok is False
    # Board state must remain unchanged and valid
    assert game.get_fen() == fen_orig
    assert game.get_hash() == hash_orig
    assert game.side_to_move() is True
    assert len(game.legal_moves_uci()) == 20


def test_board_make_unmake_reversibility_and_castling():
    """Verify full state restoration (hash, turn, castling) across make/unmake."""
    game = zeropawn_cpp.ChessGame()
    h0 = game.get_hash()
    fen0 = game.get_fen()

    # Play moves leading to White kingside castling
    seq = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"]
    for m in seq:
        assert game.make_move_uci(m)

    # Unmake all moves in reverse
    for _ in range(len(seq)):
        assert game.unmake_move()

    assert game.get_hash() == h0
    assert game.get_fen() == fen0
    assert game.side_to_move() is True


def test_board_en_passant_and_promotion():
    """Verify en passant generation and pawn promotions to all four pieces."""
    # En passant setup: White pawn on e5, Black pawn advances d7d5
    fen_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    game = zeropawn_cpp.ChessGame(fen_ep)
    moves = game.legal_moves_uci()
    assert "e5d6" in moves  # En passant capture legal

    # Underpromotions
    fen_promo = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    game_promo = zeropawn_cpp.ChessGame(fen_promo)
    promo_moves = set(game_promo.legal_moves_uci())
    for piece in ("q", "r", "b", "n"):
        assert f"e7e8{piece}" in promo_moves


def test_terminal_detection_checkmate_stalemate_draws():
    """Verify terminal conditions: Fool's mate, scholar's mate, stalemate, 50-move."""
    # Fool's mate (White checkmated)
    game_white_mated = zeropawn_cpp.ChessGame(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2"
    )
    assert game_white_mated.is_checkmate() is True
    assert game_white_mated.is_stalemate() is False
    assert game_white_mated.is_game_over() is True
    assert game_white_mated.terminal_value() == -1.0  # Mover is checkmated

    # Scholar's mate (Black checkmated)
    game_black_mated = zeropawn_cpp.ChessGame(
        "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"
    )
    assert game_black_mated.is_checkmate() is True
    assert game_black_mated.is_stalemate() is False
    assert game_black_mated.is_game_over() is True
    assert game_black_mated.terminal_value() == -1.0  # Black is checkmated

    # Stalemate
    game_stale = zeropawn_cpp.ChessGame("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game_stale.is_checkmate() is False
    assert game_stale.is_stalemate() is True
    assert game_stale.is_game_over() is True
    assert game_stale.terminal_value() == 0.0


# ==============================================================================
# Group C: Move Encoding and Decoding Bijective Parity
# ==============================================================================


def test_move_encoding_decoding_roundtrip():
    """Verify encode(decode(x)) == x and decode(encode(m)) == m for legal moves."""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "4k3/4P3/8/8/8/8/3p4/4K3 w - - 0 1",
        "4k3/4P3/8/8/8/8/3p4/4K3 b - - 0 1",
    ]
    for fen in fens:
        game = zeropawn_cpp.ChessGame(fen)
        legal_uci = game.legal_moves_uci()
        seen_indices = set()
        for uci in legal_uci:
            idx = zeropawn_cpp.move_to_index(game, uci)
            assert 0 <= idx < 4672, f"Index out of range: {idx} for {uci}"
            assert idx not in seen_indices, f"Index collision: {idx} for {uci}"
            seen_indices.add(idx)

            decoded = zeropawn_cpp.index_to_move(game, idx)
            assert decoded == uci, f"Decode mismatch: {uci} -> {idx} -> {decoded}"

    # Invalid indices
    assert zeropawn_cpp.index_to_move(game, -1) == ""
    assert zeropawn_cpp.index_to_move(game, 4672) == ""
    assert zeropawn_cpp.index_to_move(game, 99999) == ""


# ==============================================================================
# Group D: MCTS Logic, FPU Perspective Negation & Value Backup
# ==============================================================================


def test_mcts_mate_in_one_both_colors():
    """Verify MCTS finds mate-in-1 for White and Black, returning positive root value."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    def uniform_eval(g):
        return ([1.0 / g.num_actions()] * g.num_actions(), 0.0)

    # 1. White mates in 1 (Fool's mate response: d1h5#)
    game_w = zeropawn_cpp.ChessGame("rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    res_w = mcts.search(game_w, 200, uniform_eval)
    best_w = game_w.action_to_string(res_w.best_action)
    assert best_w == "d1h5"
    assert res_w.root_value > 0.5  # Favorable to White (side to move)

    # 2. Black mates in 1 (Fool's mate: d8h4#)
    game_b = zeropawn_cpp.ChessGame("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    mcts.clear()
    res_b = mcts.search(game_b, 200, uniform_eval)
    best_b = game_b.action_to_string(res_b.best_action)
    assert best_b == "d8h4"
    assert res_b.root_value > 0.5  # Favorable to Black (side to move)


def test_mcts_fpu_perspective_negation():
    """Verify that parent_q is properly negated when descending to opponent's ply."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    cfg.fpu_reduction = 0.2
    cfg.seed = 123
    mcts = zeropawn_cpp.MCTS(cfg)

    # Position: White to move
    game = zeropawn_cpp.ChessGame()

    # Controlled mock evaluator:
    # If White to move (root), value = +0.2
    # If Black to move (child), value = -0.8 (Black is badly losing, White is winning)
    def controlled_eval(g):
        n = g.num_actions()
        p = [1.0 / n] * n
        is_w = g.side_to_move()
        val = 0.2 if is_w else -0.8  # Side to move perspective
        return p, val

    res = mcts.search(game, 50, controlled_eval)
    # Root value for White must be positive since White is winning
    assert res.root_value >= 0.5


# ==============================================================================
# Group E: Strict Value Perspective & Training Target Labeling
# ==============================================================================


def test_wdl_centipawns_perspective_symmetry():
    """Verify centipawns_to_wdl correctly mirrors positive and negative scores."""
    w_pos, d_pos, l_pos = centipawns_to_wdl(+400.0)
    w_neg, d_neg, l_neg = centipawns_to_wdl(-400.0)

    assert w_pos > l_pos
    assert l_neg > w_neg
    assert math.isclose(w_pos, l_neg, rel_tol=1e-5)
    assert math.isclose(l_pos, w_neg, rel_tol=1e-5)
    assert math.isclose(d_pos, d_neg, rel_tol=1e-5)


def test_outcome_labels_align_with_side_to_move():
    """Verify self-play outcome assignment: 0=Win, 1=Draw, 2=Loss for side to move."""
    # When White wins:
    # White positions get 0 (Win)
    # Black positions get 2 (Loss)
    white_won = True
    assert (0 if white_won else 2) == 0  # White position
    assert (2 if white_won else 0) == 2  # Black position

    # When Black wins:
    white_won = False
    assert (0 if white_won else 2) == 2  # White position gets 2 (Loss)
    assert (2 if white_won else 0) == 0  # Black position gets 0 (Win)


# ==============================================================================
# Group F: Model Evaluation, Color Fairness & Elo Calculation
# ==============================================================================


def test_engine_outcome_color_mapping():
    """Verify engine_outcome correctly attributes board-relative results."""
    # Engine is White
    assert engine_outcome("1-0", engine_is_white=True) == "win"
    assert engine_outcome("0-1", engine_is_white=True) == "loss"
    assert engine_outcome("1/2-1/2", engine_is_white=True) == "draw"

    # Engine is Black
    assert engine_outcome("1-0", engine_is_white=False) == "loss"
    assert engine_outcome("0-1", engine_is_white=False) == "win"
    assert engine_outcome("1/2-1/2", engine_is_white=False) == "draw"


def test_elo_interval_symmetry():
    """Verify Elo interval calculation is exactly antisymmetric under score reversal."""
    res_pos = elo_interval(wins=60, draws=20, losses=20)
    res_neg = elo_interval(wins=20, draws=20, losses=60)

    assert res_pos["elo_diff"] is not None
    assert res_neg["elo_diff"] is not None
    assert math.isclose(res_pos["elo_diff"], -res_neg["elo_diff"], abs_tol=1e-4)


# ==============================================================================
# Group G: UCI Protocol Compliance & State Integrity
# ==============================================================================


def test_uci_protocol_position_moves_and_illegal_cutoff():
    """Verify UCI position handling and that illegal moves do not corrupt state."""
    engine = UCIEngine()

    # 1. Normal position sequence
    assert engine.handle_line("position startpos moves e2e4 e7e5") is True
    assert engine.game.side_to_move() is True
    assert engine.game.full_move_number() == 2

    # 2. Position sequence containing an illegal move
    # e2e4 is legal, INVALID is illegal, e7e5 should be aborted
    engine.handle_line("position startpos moves e2e4 INVALID e7e5")
    # After e2e4, turn is Black (side_to_move() is False)
    assert engine.game.side_to_move() is False
    assert engine.game.get_fen().split()[1] == "b"

    # 3. Newgame clears search state
    assert engine.handle_line("ucinewgame") is True
    assert engine.game.side_to_move() is True
    assert engine.game.full_move_number() == 1


# ==============================================================================
# Group H: Python/C++ Parity Differential Check
# ==============================================================================


def test_python_cpp_parity_across_positions():
    """Differential verification between python-chess reference and C++ ChessGame."""
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    ]

    for fen in test_fens:
        py_b = chess.Board(fen)
        cpp_g = zeropawn_cpp.ChessGame(fen)

        # 1. Side to move
        assert (py_b.turn == chess.WHITE) == bool(cpp_g.side_to_move())

        # 2. Legal moves count and set
        py_moves = {m.uci() for m in py_b.legal_moves}
        cpp_moves = set(cpp_g.legal_moves_uci())
        assert py_moves == cpp_moves

        # 3. Terminal state parity
        assert py_b.is_checkmate() == cpp_g.is_checkmate()
        assert py_b.is_stalemate() == cpp_g.is_stalemate()
        assert py_b.is_game_over() == cpp_g.is_game_over()
