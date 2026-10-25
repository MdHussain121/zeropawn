import pytest
import numpy as np
import zeropawn_cpp


def test_encode_state_shape_and_planes():
    game = zeropawn_cpp.ChessGame()
    planes = zeropawn_cpp.encode_state(game)
    assert isinstance(planes, np.ndarray)
    assert planes.shape == (19, 8, 8)
    assert planes.dtype == np.float32

    # Plane 0: own pawns at row 1
    assert planes[0, 1, :].sum() == 8.0
    # Plane 5: own king at e1 (row 0, col 4)
    assert planes[5, 0, 4] == 1.0
    # Castling planes
    assert (planes[12] == 1.0).all()
    assert (planes[13] == 1.0).all()
    assert (planes[14] == 1.0).all()
    assert (planes[15] == 1.0).all()
    # Halfmove clock
    assert (planes[17] == 0.0).all()
    # Repetition flag
    assert (planes[18] == 0.0).all()


def test_move_roundtrip_startpos():
    game = zeropawn_cpp.ChessGame()
    for m in game.legal_moves_uci():
        idx = zeropawn_cpp.move_to_index(game, m)
        assert 0 <= idx < 4672
        back = zeropawn_cpp.index_to_move(game, idx)
        assert back == m


def test_move_roundtrip_black():
    game = zeropawn_cpp.ChessGame()
    game.make_move_uci("e2e4")
    for m in game.legal_moves_uci():
        idx = zeropawn_cpp.move_to_index(game, m)
        assert 0 <= idx < 4672
        back = zeropawn_cpp.index_to_move(game, idx)
        assert back == m


def test_underpromotions_roundtrip():
    # Position with White pawn on e7 and Black pawn on d2
    fen = "4k3/4P3/8/8/8/8/3p4/4K3 w - - 0 1"
    game = zeropawn_cpp.ChessGame(fen)
    moves = game.legal_moves_uci()
    # promotions from e7 to e8: e7e8q, e7e8r, e7e8b, e7e8n
    for m in moves:
        if m.startswith("e7e8"):
            idx = zeropawn_cpp.move_to_index(game, m)
            assert 0 <= idx < 4672
            back = zeropawn_cpp.index_to_move(game, idx)
            assert back == m

    # Black's perspective
    game.make_move_uci("e1d1")  # White king moves
    black_moves = game.legal_moves_uci()
    for m in black_moves:
        if m.startswith("d2d1") or m.startswith("d2e1"):
            idx = zeropawn_cpp.move_to_index(game, m)
            assert 0 <= idx < 4672
            back = zeropawn_cpp.index_to_move(game, idx)
            assert back == m
