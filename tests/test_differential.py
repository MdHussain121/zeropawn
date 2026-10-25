import random
import chess
import zeropawn_cpp


def test_forced_underpromotions():
    for side in (chess.WHITE, chess.BLACK):
        for from_file in range(8):
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


def test_differential_random_games():
    random.seed(1337)
    num_games = 200
    for g in range(num_games):
        b = chess.Board()
        eng = zeropawn_cpp.ChessGame()
        ply = 0
        while not b.is_game_over() and ply < 80:
            py_moves = set(m.uci() for m in b.legal_moves)
            eng_moves = set(eng.legal_moves_uci())
            assert py_moves == eng_moves

            for m_uci in py_moves:
                idx = zeropawn_cpp.move_to_index(eng, m_uci)
                assert 0 <= idx < 4672
                back = zeropawn_cpp.index_to_move(eng, idx)
                assert back == m_uci

            m = random.choice(list(b.legal_moves))
            b.push(m)
            assert eng.make_move_uci(m.uci())
            ply += 1
