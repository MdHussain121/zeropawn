import pytest
import zeropawn_cpp


def test_perft_startpos():
    print("\n  [Op 1/1] Running move generation Perft depth 1-5 from start position...")
    game = zeropawn_cpp.ChessGame()
    expected = [20, 400, 8902, 197281, 4865609]
    for d, exp in enumerate(expected, 1):
        cnt = game.perft(d)
        assert cnt == exp
    print(
        f"    [OK] Depth 1-5 nodes: 20 -> 400 -> 8,902 -> 197,281 -> 4,865,609 verified exact match."
    )


def test_perft_kiwipete():
    # Canonical Kiwipete from Chess Programming Wiki (PPPBBPPP)
    print("\n  [Op 1/1] Running Perft depth 1-4 on complex Kiwipete position...")
    game = zeropawn_cpp.ChessGame(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    expected = [48, 2039, 97862, 4085603]
    for d, exp in enumerate(expected, 1):
        cnt = game.perft(d)
        assert cnt == exp
    print(
        f"    [OK] Kiwipete depth 1-4 nodes: 48 -> 2,039 -> 97,862 -> 4,085,603 verified exact match."
    )


def test_perft_pos3():
    print("\n  [Op 1/1] Running Perft depth 1-4 on endgame Position 3...")
    game = zeropawn_cpp.ChessGame("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    expected = [14, 191, 2812, 43238]
    for d, exp in enumerate(expected, 1):
        cnt = game.perft(d)
        assert cnt == exp
    print(
        f"    [OK] Position 3 depth 1-4 nodes: 14 -> 191 -> 2,812 -> 43,238 verified exact match."
    )
