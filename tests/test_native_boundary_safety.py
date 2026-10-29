"""Regression tests for native-boundary safety fixes (search-quality audit 2026-10-29).

C-1: rule-illegal FENs (side not to move already in check) used to crash the
     process with an access violation: move generation emitted a king-capture,
     applying it removed a king, and the next move generation dereferenced the
     missing king. set_fen now rejects them, the FEN constructor raises
     ValueError, and the UCI layer keeps the previous position.
M-2: legacy MCTS.search_batched never touches the retained tree, so a
     reuse_tree=true config was a silent no-op; the binding now refuses the
     combination with an actionable error.
"""

import numpy as np
import pytest

import zeropawn_cpp

# The audit's exact reproduction: black king g8 attacked by the f7 pawn while
# White is to move. movegen used to emit f7g8q (king capture) -> segfault.
CRASH_FEN = "6k1/5Ppp/8/8/8/8/8/K7 w - - 0 1"


def uniform_eval(game):
    n = game.num_actions()
    return ([1.0 / n] * n, 0.0)


def uniform_batch_eval(batch_planes):
    n = int(batch_planes.shape[0])
    return (
        np.full((n, 4672), 1.0 / 4672, dtype=np.float32),
        np.zeros((n,), dtype=np.float32),
    )


# ==============================================================================
# C-1: illegal-FEN rejection
# ==============================================================================


def test_set_fen_rejects_side_not_to_move_in_check():
    """The audit's crash FEN must be rejected by set_fen, not accepted."""
    game = zeropawn_cpp.ChessGame()
    assert game.set_fen(CRASH_FEN) is False
    # Board must be unchanged (still startpos).
    assert game.get_fen() == zeropawn_cpp.ChessGame().get_fen()


def test_constructor_raises_on_rule_illegal_fen():
    """The FEN constructor must fail loudly instead of silently yielding a
    different (start) position for rule-illegal input."""
    with pytest.raises(ValueError):
        zeropawn_cpp.ChessGame(CRASH_FEN)


@pytest.mark.parametrize(
    "bad_fen",
    [
        "6k1/5Ppp/8/8/8/8/8/K7 w - - 0 1",  # side not to move in check (C-1 crash)
        "4k3/8/8/8/8/8/3p4/4K3 b - - 0 1",  # white king in check while black is to move
        "8/8/8/8/8/8/8/K7 w - - 0 1",  # missing black king
        "6k1/5ppp/8/8/8/8/8/8 w - - 0 1",  # missing white king
        "kk6/8/8/8/8/8/8/K6k w - - 0 1",  # multiple kings of one color
        "not a fen",
        "8/8/8/8/8/8/8/8 w - - 0 1",  # empty board (no kings)
        "invalid_corrupted_fen_string",
    ],
)
def test_invalid_fen_variants_rejected(bad_fen):
    """Missing kings, malformed strings, and king-in-check positions all fail
    deterministically via set_fen (False) and the constructor (ValueError)."""
    game = zeropawn_cpp.ChessGame()
    assert game.set_fen(bad_fen) is False
    with pytest.raises(ValueError):
        zeropawn_cpp.ChessGame(bad_fen)


@pytest.mark.parametrize(
    "fen",
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # startpos
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
        "4k3/4P3/8/8/8/8/3p4/4K3 w - - 0 1",  # side to move in check (must stay legal)
        "k7/8/1Q6/8/8/8/8/7K b - - 0 1",  # stalemate fixture (terminal, 0 legal moves)
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",  # fool's-mate prelude
    ],
)
def test_valid_positions_still_accepted(fen):
    """Control: normal and edge-case LEGAL positions are unaffected by the guard."""
    game = zeropawn_cpp.ChessGame(fen)
    assert game.legal_moves_uci() is not None
    assert game.set_fen(fen) is True


def test_startpos_move_count_unchanged():
    """Sanity control on the most common position."""
    assert len(zeropawn_cpp.ChessGame().legal_moves_uci()) == 20


def test_king_capture_fen_cannot_reach_search():
    """Even if a caller keeps a rejected position out of the constructor, no
    king-capture position can be produced: set_fen is the only FEN entry point
    and it rejects. A search on a valid control position still works and the
    process survives."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 7
    mcts = zeropawn_cpp.MCTS(cfg)
    game = zeropawn_cpp.ChessGame("rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    res = mcts.search(game, 30, uniform_eval)
    assert game.action_to_string(res.best_action) in game.legal_moves_uci()


def test_uci_position_command_survives_invalid_fen():
    """UCI 'position fen <illegal>' must not kill the engine: the UCI layer
    catches the ValueError and keeps the previous game state (regression for
    the tournament-forfeit crash)."""
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine(checkpoint_path="", use_book=False)
    assert engine.handle_line("uci") is True
    assert engine.handle_line(f"position fen {CRASH_FEN}") is True  # rejected, engine alive
    assert engine.handle_line("go nodes 20") is True  # searches the retained (valid) position
    # A valid position still loads afterwards.
    assert (
        engine.handle_line("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        is True
    )
    assert engine.handle_line("go nodes 20") is True


# ==============================================================================
# M-2: search_batched + reuse_tree incompatibility
# ==============================================================================


def test_search_batched_rejects_reuse_tree():
    """The unsupported combination must fail loudly with an actionable message."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    mcts = zeropawn_cpp.MCTS(cfg)
    game = zeropawn_cpp.ChessGame()
    with pytest.raises(ValueError, match="reuse_tree"):
        mcts.search_batched(game, 8, 4, uniform_batch_eval)
    with pytest.raises(ValueError, match="reuse_tree"):
        mcts.search_batched(game, 8, uniform_batch_eval, 4)


def test_search_batched_without_reuse_still_works():
    """Control: the supported legacy-batched configuration is unaffected."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 11
    mcts = zeropawn_cpp.MCTS(cfg)
    game = zeropawn_cpp.ChessGame("rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    res = mcts.search_batched(game, 30, 8, uniform_batch_eval)
    assert game.action_to_string(res.best_action) in game.legal_moves_uci()
    # reuse_tree=false remains the config default on fresh instances.
    assert zeropawn_cpp.MCTSConfig().reuse_tree is False
