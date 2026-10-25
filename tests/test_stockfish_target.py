import math
import numpy as np
import pytest
import chess
import python  # ensures zeropawn_cpp
import zeropawn_cpp
from python.data.stockfish_eval import (
    centipawns_to_wdl,
    blend_targets,
    outcome_to_one_hot,
    DEFAULT_STOCKFISH_PATH,
    StockfishEvaluator,
)
from python.data.fen_utils import decode_planes_to_fen


def test_centipawns_to_wdl_properties():
    # Test zero centipawns
    w0, d0, l0 = centipawns_to_wdl(0.0)
    assert math.isclose(w0 + d0 + l0, 1.0, rel_tol=1e-5)
    assert math.isclose(w0, l0, rel_tol=1e-5)
    assert math.isclose(w0, 1.0 / 3.0, rel_tol=1e-2)

    # Test symmetry for positive and negative cp
    w_pos, d_pos, l_pos = centipawns_to_wdl(300.0)
    w_neg, d_neg, l_neg = centipawns_to_wdl(-300.0)

    assert math.isclose(w_pos + d_pos + l_pos, 1.0, rel_tol=1e-5)
    assert math.isclose(w_neg + d_neg + l_neg, 1.0, rel_tol=1e-5)
    assert math.isclose(w_pos, l_neg, rel_tol=1e-5)
    assert math.isclose(l_pos, w_neg, rel_tol=1e-5)
    assert math.isclose(d_pos, d_neg, rel_tol=1e-5)
    assert w_pos > l_pos


def test_blend_targets():
    sf_wdl = (0.7, 0.2, 0.1)
    # Win outcome: [1.0, 0.0, 0.0]
    blended_win = blend_targets(sf_wdl, outcome=0)
    assert np.allclose(blended_win, [0.5 * 0.7 + 0.5 * 1.0, 0.5 * 0.2, 0.5 * 0.1])
    assert math.isclose(float(blended_win.sum()), 1.0, rel_tol=1e-5)

    # Draw outcome: [0.0, 1.0, 0.0]
    blended_draw = blend_targets(sf_wdl, outcome=1)
    assert np.allclose(blended_draw, [0.5 * 0.7, 0.5 * 0.2 + 0.5 * 1.0, 0.5 * 0.1])
    assert math.isclose(float(blended_draw.sum()), 1.0, rel_tol=1e-5)


def test_decode_planes_to_fen_roundtrip():
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
        "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 4",
    ]

    for fen in test_fens:
        g = zeropawn_cpp.ChessGame(fen)
        planes = zeropawn_cpp.encode_state(g)
        is_white = g.side_to_move()
        decoded_fen = decode_planes_to_fen(planes, is_white_to_move=is_white)

        board_orig = chess.Board(fen)
        board_dec = chess.Board(decoded_fen)

        # Confirm piece map matches exactly
        assert board_orig.piece_map() == board_dec.piece_map()
        assert board_orig.turn == board_dec.turn
        assert board_orig.castling_rights == board_dec.castling_rights


@pytest.mark.skipif(not DEFAULT_STOCKFISH_PATH.exists(), reason="Stockfish binary not found")
def test_stockfish_evaluator_live():
    with StockfishEvaluator(DEFAULT_STOCKFISH_PATH, threads=1) as sf:
        cp = sf.evaluate_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", nodes=1000
        )
        assert isinstance(cp, float)
        assert -500.0 < cp < 500.0
        w, d, l = centipawns_to_wdl(cp)
        assert math.isclose(w + d + l, 1.0, rel_tol=1e-5)
