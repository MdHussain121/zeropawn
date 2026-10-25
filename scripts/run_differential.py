#!/usr/bin/env python3
"""Differential test vs python-chess over 10,000 random games and forced underpromotions."""

import os
import sys
import time
import random

# Windows DLL directory fix
_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    os.add_dll_directory(_winlibs_bin)

_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "cpp"))
if os.path.isdir(_build_dir):
    sys.path.insert(0, _build_dir)

import chess
import zeropawn_cpp


def run_random_games(num_games=10000, max_plies=120, seed=42):
    random.seed(seed)
    total_plies = 0
    total_moves_checked = 0
    underpromo_count = 0
    start_time = time.time()

    print(f"Starting differential testing vs python-chess ({num_games} games)...")

    for g in range(1, num_games + 1):
        b = chess.Board()
        eng = zeropawn_cpp.ChessGame()

        ply = 0
        while not b.is_game_over() and ply < max_plies:
            py_moves = set(m.uci() for m in b.legal_moves)
            eng_moves = set(eng.legal_moves_uci())

            assert py_moves == eng_moves, (
                f"Move mismatch at game {g}, ply {ply}, fen: {b.fen()}\nPy: {py_moves}\nEng: {eng_moves}"
            )

            # Verify move round trip for every legal move
            for m_uci in py_moves:
                idx = zeropawn_cpp.move_to_index(eng, m_uci)
                assert 0 <= idx < 4672, f"Index out of bounds {idx} for {m_uci}"
                back = zeropawn_cpp.index_to_move(eng, idx)
                assert back == m_uci, f"Round trip mismatch: {m_uci} -> {idx} -> {back}"
                total_moves_checked += 1
                if len(m_uci) == 5 and m_uci[4] in ("n", "b", "r"):
                    underpromo_count += 1

            # Pick a move (bias slightly towards promotions when available to stress-test underpromotions)
            legal_list = list(b.legal_moves)
            promo_moves = [m for m in legal_list if m.promotion is not None]
            if promo_moves and random.random() < 0.6:
                m = random.choice(promo_moves)
            else:
                m = random.choice(legal_list)

            b.push(m)
            assert eng.make_move_uci(m.uci()), f"Engine failed to make legal move {m.uci()}"
            ply += 1
            total_plies += 1

        if g % 2000 == 0 or g == num_games:
            elapsed = time.time() - start_time
            rate = total_plies / elapsed if elapsed > 0 else 0
            print(
                f"  [{g:5d}/{num_games}] Plies: {total_plies:7d} | Moves checked: {total_moves_checked:8d} | Underpromotions: {underpromo_count:4d} | {rate:.0f} plies/s"
            )

    elapsed = time.time() - start_time
    print(f"Completed {num_games} games in {elapsed:.2f}s ({total_plies / elapsed:.0f} plies/s).")
    print(f"Total plies verified: {total_plies}")
    print(f"Total individual moves round-tripped: {total_moves_checked}")
    print(f"Total underpromotions checked: {underpromo_count}")
    return total_plies, total_moves_checked, underpromo_count


def test_forced_underpromotions():
    print("Testing forced underpromotions across all files and colors...")
    tested = 0
    for side in (chess.WHITE, chess.BLACK):
        for from_file in range(8):
            for promo_piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                # Setup pawn on 7th rank (rank 7 for white, rank 2 for black)
                b = chess.Board(None)
                if side == chess.WHITE:
                    b.set_piece_at(chess.square(from_file, 6), chess.Piece(chess.PAWN, chess.WHITE))
                    b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
                    b.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
                    b.turn = chess.WHITE
                else:
                    b.set_piece_at(chess.square(from_file, 1), chess.Piece(chess.PAWN, chess.BLACK))
                    b.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
                    b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
                    b.turn = chess.BLACK

                eng = zeropawn_cpp.ChessGame(b.fen())
                py_moves = set(m.uci() for m in b.legal_moves)
                eng_moves = set(eng.legal_moves_uci())
                assert py_moves == eng_moves

                for m_uci in py_moves:
                    idx = zeropawn_cpp.move_to_index(eng, m_uci)
                    assert 0 <= idx < 4672
                    back = zeropawn_cpp.index_to_move(eng, idx)
                    assert back == m_uci
                    tested += 1

    print(f"Forced underpromotions tested: {tested} moves verified.")


if __name__ == "__main__":
    test_forced_underpromotions()
    num_games = 10000
    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])
    run_random_games(num_games=num_games)
