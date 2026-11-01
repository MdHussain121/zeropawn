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


def test_ep_plane_written_only_for_capturable_en_passant():
    """Plane 16 contract (audit I-2): the ep square is written only when an
    en-passant capture is actually legal — the chess library normalizes
    non-capturable ep fields away, so external tooling must NOT assume
    raw-FEN ep semantics (FEN keeps 'e3' after 1.e4; the plane does not)."""
    # 1.e4: FEN carries e3, but no black pawn can capture -> plane empty.
    game = zeropawn_cpp.ChessGame()
    game.make_move_uci("e2e4")
    assert game.get_fen().split()[3] == "-"  # library already normalized it
    planes = zeropawn_cpp.encode_state(game)
    assert planes[16].sum() == 0.0

    # 1.e4 c5 2.e5 d5: white e5 pawn can capture d6 en passant -> written.
    game = zeropawn_cpp.ChessGame()
    for mv in ("e2e4", "c7c5", "e4e5", "d7d5"):
        assert game.make_move_uci(mv)
    planes = zeropawn_cpp.encode_state(game)
    assert planes[16].sum() == 1.0
    # White to move: d6 square appears at rank row 5, file col 3.
    assert planes[16, 5, 3] == 1.0

    # Same position one ply later (knight shuffle): the ep opportunity is
    # gone (not a capturable ep) -> plane empty again.
    for mv in ("g1f3", "b8c6"):
        game.make_move_uci(mv)
    planes = zeropawn_cpp.encode_state(game)
    assert planes[16].sum() == 0.0


def test_black_to_move_planes_are_the_mirrored_side_to_move_view():
    """Coverage gap closed (audit §16.4): the encoder's black-to-move plane
    content is pinned, not just white's.  Black-to-move encoding equals the
    color-mirrored position's white-to-move encoding (groups 0-5/6-11 swapped
    plus rank flip — expressed here as: black's own pieces occupy planes
    0-5 starting at rank row 0, and pawns advance in +r)."""
    game = zeropawn_cpp.ChessGame()
    assert game.make_move_uci("e2e4")  # Black to move
    planes = zeropawn_cpp.encode_state(game)
    assert not game.side_to_move()

    # Black is the side to move: black pieces are "own" (planes 0-5) with the
    # rank flip (black's home rank 7 -> own row 1; e8 king -> own row 0).
    assert planes[0, 1, :].sum() == 8.0  # black pawns -> own row 1
    assert planes[5, 0, 4] == 1.0  # black king e8 -> own row 0, col 4
    # White (opponent) pieces are planes 6-11; white's advanced e-pawn is a
    # white pawn (plane 6) far from black's home row (e4 -> row 4), and the
    # untouched white pawns sit on row 6.
    assert planes[6, 4, 4] == 1.0
    assert planes[6, 6, :].sum() == 7.0
    # Castling rights: black's (own) rights fill planes 12/13, white's 14/15.
    assert (planes[12] == 1.0).all() and (planes[13] == 1.0).all()
    assert (planes[14] == 1.0).all() and (planes[15] == 1.0).all()
