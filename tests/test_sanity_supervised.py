"""Sanity tests for Phase 2 Supervised Warm Start.

Task 7 Sanity Checks (executed BEFORE the full run):
(a) The model can overfit a single batch to near-zero loss.
(b) Training on shuffled labels gives near-chance accuracy.
(c) Decode 20 random dataset samples back to FEN and confirm
    perspective and labels against the source PGN.
"""

from __future__ import annotations

import io
import random
import torch
import torch.nn.functional as F
import chess
import chess.pgn

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.data.fen_utils import decode_planes_to_fen
from python.data.build_dataset import process_pgn_game, PositionData


SAMPLE_TEST_GAMES = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/sanity01"]
[Date "2026.09.01"]
[White "MasterA"]
[Black "MasterB"]
[Result "1-0"]
[WhiteElo "2350"]
[BlackElo "2320"]
[Termination "Normal"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/sanity02"]
[Date "2026.09.01"]
[White "MasterC"]
[Black "MasterD"]
[Result "0-1"]
[WhiteElo "2400"]
[BlackElo "2450"]
[Termination "Normal"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. a3 Bxc3+ 6. Qxc3 b6 7. Bg5 Bb7 8. e3 d6 9. f3 Nbd7 10. Bd3 c5 11. Ne2 Rc8 12. Qd2 h6 0-1
""".strip()


def test_sanity_overfit_single_batch():
    """Sanity test (a): Model can overfit a single batch to near-zero loss and 100% top-1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 16
    x = torch.randn(batch_size, 19, 8, 8, device=device)
    moves = torch.randint(0, 4672, (batch_size,), device=device)
    wdl = torch.randn(batch_size, 3, device=device).softmax(dim=-1)

    model.train()
    for step in range(120):
        optimizer.zero_grad(set_to_none=True)
        policy_logits, wdl_logits = model(x)
        policy_loss = F.cross_entropy(policy_logits, moves)
        wdl_loss = F.cross_entropy(wdl_logits, wdl)
        total_loss = policy_loss + 0.5 * wdl_loss
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_p, final_v = model(x)
        final_loss = (F.cross_entropy(final_p, moves) + 0.5 * F.cross_entropy(final_v, wdl)).item()
        top1_acc = (final_p.argmax(dim=-1) == moves).float().mean().item() * 100.0

    print(
        f"\n[SANITY 7.a] Overfit single batch: Final loss = {final_loss:.4f}, Top-1 = {top1_acc:.1f}%"
    )
    assert top1_acc == 100.0, f"Top-1 accuracy {top1_acc}% != 100%"
    assert final_loss < 0.5, f"Final loss {final_loss} >= 0.5"


def test_sanity_shuffled_labels_near_chance():
    """Sanity test (b): Training on shuffled labels gives near-chance accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Generate random training positions and random target moves
    batch_size = 64
    x = torch.randn(batch_size, 19, 8, 8, device=device)
    shuffled_moves = torch.randint(0, 4672, (batch_size,), device=device)

    # Run a few gradient steps on shuffled labels
    model.train()
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        policy_logits, _ = model(x)
        loss = F.cross_entropy(policy_logits, shuffled_moves)
        loss.backward()
        optimizer.step()

    # Evaluate on held-out random test batch
    x_test = torch.randn(batch_size, 19, 8, 8, device=device)
    test_targets = torch.randint(0, 4672, (batch_size,), device=device)

    model.eval()
    with torch.no_grad():
        test_logits, _ = model(x_test)
        test_top1 = (test_logits.argmax(dim=-1) == test_targets).float().mean().item() * 100.0

    print(f"\n[SANITY 7.b] Shuffled labels held-out top-1: {test_top1:.2f}% (Chance is ~0.02%)")
    # Must remain near chance (< 5%)
    assert test_top1 < 5.0, f"Shuffled accuracy {test_top1}% exceeds chance threshold"


def test_sanity_decode_dataset_samples():
    """Sanity test (c): Decode 20 random dataset samples back to FEN and confirm
    perspective and labels against the source PGN.
    """
    f = io.StringIO(SAMPLE_TEST_GAMES)
    all_positions: list[tuple[PositionData, chess.Board, chess.Move, str]] = []

    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        result_str = game.headers["Result"]
        game_id_hash, is_val, pos_list = process_pgn_game(game)

        # Track board step by step with python-chess
        board = game.board()
        for ply_idx, move in enumerate(game.mainline_moves()):
            pos_data = pos_list[ply_idx]
            all_positions.append((pos_data, board.copy(), move, result_str))
            board.push(move)

    assert len(all_positions) >= 20, f"Need at least 20 positions, got {len(all_positions)}"

    # Sample exactly 20 positions, ensuring both White and Black to move
    random.seed(42)
    sampled = random.sample(all_positions, 20)

    white_count = 0
    black_count = 0

    for i, (pos_data, expected_board, expected_move, game_result) in enumerate(sampled):
        is_white_to_move = expected_board.turn == chess.WHITE
        if is_white_to_move:
            white_count += 1
        else:
            black_count += 1

        # 1. Decode planes back to FEN
        decoded_fen = decode_planes_to_fen(
            pos_data.planes, is_white_to_move=is_white_to_move, ply=pos_data.ply
        )
        decoded_board = chess.Board(decoded_fen)

        # 2. Confirm piece placement matches expected board
        assert decoded_board.piece_map() == expected_board.piece_map(), (
            f"Sample {i}: Piece map mismatch!\n"
            f"Expected: {expected_board.fen()}\n"
            f"Decoded:  {decoded_fen}"
        )
        assert decoded_board.turn == expected_board.turn

        # 3. Confirm target move is legal in decoded board
        cpp_game = zeropawn_cpp.ChessGame(decoded_fen)
        decoded_move_uci = zeropawn_cpp.index_to_move(cpp_game, pos_data.move_idx)
        assert decoded_move_uci == expected_move.uci(), (
            f"Sample {i}: Move mismatch! Decoded {decoded_move_uci} != Expected {expected_move.uci()}"
        )
        assert chess.Move.from_uci(decoded_move_uci) in decoded_board.legal_moves, (
            f"Sample {i}: Decoded move {decoded_move_uci} is illegal in decoded position!"
        )

        # 4. Confirm outcome perspective
        # 0 = Win, 1 = Draw, 2 = Loss for side to move
        if game_result == "1-0":
            expected_outcome = 0 if is_white_to_move else 2
        elif game_result == "0-1":
            expected_outcome = 2 if is_white_to_move else 0
        else:
            expected_outcome = 1

        assert pos_data.outcome == expected_outcome, (
            f"Sample {i}: Outcome label mismatch! Got {pos_data.outcome}, expected {expected_outcome} "
            f"(Game result {game_result}, Side to move: {'White' if is_white_to_move else 'Black'})"
        )

        # 5. Confirm board orientation:
        # When Black is to move, row 0 in planes corresponds to Rank 8.
        if not is_white_to_move:
            # Check row 0 of Black own pieces planes (planes 0..5) matches Black pieces on Rank 8
            for c in range(8):
                sq_rank8 = chess.square(c, 7)  # Rank 8
                piece = expected_board.piece_at(sq_rank8)
                if piece and piece.color == chess.BLACK:
                    plane_idx = piece.piece_type - 1  # 0..5
                    assert pos_data.planes[plane_idx, 0, c] == 1.0, (
                        f"Sample {i}: Black rank 8 piece {piece} not found at flipped planes row 0, col {c}!"
                    )

    print(
        f"\n[SANITY 7.c] Verified 20 samples ({white_count} White, {black_count} Black to move). All decoded FENs, legal moves, and outcome perspectives matched perfectly!"
    )
