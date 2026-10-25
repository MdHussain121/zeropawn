import pytest
import zeropawn_cpp


def test_chess_game_init():
    print("\n  [Op 1/2] Initializing default starting chess position in C++ engine...")
    game = zeropawn_cpp.ChessGame()
    assert game.side_to_move() is True
    assert game.full_move_number() == 1
    assert game.half_move_clock() == 0
    assert not game.in_check()
    print("  [Op 2/2] Generating legal moves from starting position...")
    moves = game.legal_moves_uci()
    assert len(moves) == 20
    print(f"    [OK] Engine verified: 20 legal moves, White to move, full_move=1, half_clock=0.")


def test_make_unmake_move():
    print("\n  [Op 1/3] Initializing game and calculating initial Zobrist hash...")
    game = zeropawn_cpp.ChessGame()
    h0 = game.get_hash()
    print("  [Op 2/3] Making move 'e2e4' and checking turn transition...")
    assert game.make_move_uci("e2e4")
    assert game.side_to_move() is False
    assert game.get_hash() != h0
    print("  [Op 3/3] Unmaking move and verifying state & hash restoration...")
    assert game.unmake_move()
    assert game.side_to_move() is True
    assert game.get_hash() == h0
    print("    [OK] Move make/unmake and hash consistency verified.")


def test_fen_and_kiwipete():
    print("\n  [Op 1/2] Loading complex Kiwipete tactical position via FEN...")
    game = zeropawn_cpp.ChessGame()
    kiwi = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    game.set_fen(kiwi)
    print("  [Op 2/2] Generating and counting legal moves in Kiwipete...")
    moves = game.legal_moves_uci()
    assert len(moves) == 48
    print(f"    [OK] Kiwipete position verified: exactly 48 legal moves generated.")
