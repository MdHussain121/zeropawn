import pytest
import zeropawn_cpp


def test_checkmate():
    print("\n  [Op 1/1] Evaluating Fool's Mate terminal state (checkmate)...")
    game = zeropawn_cpp.ChessGame("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2")
    assert game.in_check() is True
    assert game.is_checkmate() is True
    assert game.is_stalemate() is False
    assert game.is_game_over() is True
    print("    [OK] Checkmate confirmed: in_check=True, is_checkmate=True, is_game_over=True.")


def test_stalemate():
    print("\n  [Op 1/1] Evaluating Stalemate terminal state...")
    game = zeropawn_cpp.ChessGame("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert game.in_check() is False
    assert game.is_checkmate() is False
    assert game.is_stalemate() is True
    assert game.is_game_over() is True
    print("    [OK] Stalemate confirmed: in_check=False, is_stalemate=True, is_game_over=True.")


def test_50_move_rule():
    print("\n  [Op 1/1] Evaluating 50-move draw rule detection...")
    game = zeropawn_cpp.ChessGame("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 50")
    assert game.is_50_move_rule() is True
    assert game.is_game_over() is True

    game99 = zeropawn_cpp.ChessGame("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 50")
    assert game99.is_50_move_rule() is False
    print("    [OK] 50-move rule correctly triggered at 100 half-moves and rejected at 99.")


def test_threefold_repetition():
    print("\n  [Op 1/1] Evaluating threefold repetition detection over repeated knight moves...")
    game = zeropawn_cpp.ChessGame()
    assert not game.is_repetition(2)
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    for m in moves:
        assert game.make_move_uci(m)
    assert game.is_repetition(2) is True
    assert game.is_game_over() is True
    print("    [OK] Threefold repetition successfully detected after repetition cycle.")


def test_insufficient_material():
    print(
        "\n  [Op 1/1] Evaluating insufficient material rules (K vs K, K+B vs K, K+N vs K, K+P vs K)..."
    )
    kvk = zeropawn_cpp.ChessGame("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    assert kvk.is_insufficient_material() is True
    assert kvk.is_game_over() is True

    kvkb = zeropawn_cpp.ChessGame("8/8/8/4k3/8/8/4KB2/8 w - - 0 1")
    assert kvkb.is_insufficient_material() is True

    kvkn = zeropawn_cpp.ChessGame("8/8/8/4k3/8/8/4KN2/8 w - - 0 1")
    assert kvkn.is_insufficient_material() is True

    kvkp = zeropawn_cpp.ChessGame("8/8/8/4k3/8/8/4KP2/8 w - - 0 1")
    assert kvkp.is_insufficient_material() is False
    print(
        "    [OK] Insufficient material accurately distinguished across all piece configurations."
    )
