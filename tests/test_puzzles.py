"""Unit tests for puzzle benchmark dataset and PuzzleEvaluator.

The benchmark dataset is built *in the test* (a deterministic CSV in
``tmp_path``) instead of being read from ``data/puzzles/``:

* the bundled ``data/puzzles/lichess_puzzles_1000_1200_1600.csv`` does not
  exist in this tree, so the three tests below were hard failures
  (PUZ-TEST-003);
* a dataset that may be deleted must never decide whether a test passes.

The assertions are unchanged - the fixture simply provides the dataset they
describe.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import chess
import pytest
import torch
import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.eval.puzzles import Puzzle, load_puzzles, PuzzleEvaluator

OPENING_LINE = [
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1c4",
    "g8f6",
    "d2d3",
    "f6e4",
    "b1c3",
    "f8c5",
    "a2a3",
    "d7d6",
]
"""A fixed opening line: the positions of the fixture are its replay."""

CSV_HEADER = [
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
    "DailyDate",
]
"""The column order of the Lichess puzzle export."""

PUZZLE_COUNT = 1000
"""How many puzzles the benchmark fixture holds."""

MIN_RATING = 1200
"""The lowest rating of the benchmark bracket."""

MAX_RATING = 1600
"""The highest rating of the benchmark bracket."""


def _positions() -> list[str]:
    """The positions of the fixture: the start position and its replay."""
    board = chess.Board()
    positions = [board.fen()]
    for uci in OPENING_LINE:
        board.push_uci(uci)
        positions.append(board.fen())
    return positions


def _triples() -> list[tuple[str, str, str]]:
    """The ``(fen, opponent move, solution move)`` triples of the fixture.

    Only pawn and knight moves are used: they are unambiguous in both engines
    (no castling, no en-passant edge case).
    """
    triples: list[tuple[str, str, str]] = []
    for fen in _positions():
        board = chess.Board(fen)
        moves = sorted(
            move.uci()
            for move in board.legal_moves
            if board.piece_type_at(move.from_square) in (chess.PAWN, chess.KNIGHT)
        )
        for uci in moves[:4]:
            after = board.copy()
            after.push_uci(uci)
            replies = sorted(
                move.uci()
                for move in after.legal_moves
                if after.piece_type_at(move.from_square) in (chess.PAWN, chess.KNIGHT)
            )
            if not replies:
                continue
            triples.append((fen, uci, replies[0]))
    assert triples, "the fixture must not be empty"
    return triples


def _write_dataset(target: Path, count: int = PUZZLE_COUNT, offset: int = 0) -> Path:
    """Write the deterministic benchmark CSV and return its path."""
    triples = _triples()
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for index in range(count):
            fen, opponent_move, solution_move = triples[(index + offset) % len(triples)]
            rating = MIN_RATING + (index % (MAX_RATING - MIN_RATING + 1))
            writer.writerow(
                [
                    f"SYN{index + offset:04d}",
                    fen,
                    f"{opponent_move} {solution_move}",
                    rating,
                    50,
                    80,
                    10,
                    "endgame",
                    f"https://lichess.org/training/SYN{index + offset:04d}",
                    "Test Opening",
                    "2024-01-01",
                ]
            )
    return target


@pytest.fixture(scope="module")
def puzzle_csv(tmp_path_factory) -> Path:
    """The deterministic benchmark dataset (1000 puzzles, rating 1200-1600)."""
    target = tmp_path_factory.mktemp("puzzle_benchmark") / "lichess_puzzles_1000_1200_1600.csv"
    return _write_dataset(target)


@pytest.fixture(scope="module")
def puzzle_dataset_dir(tmp_path_factory) -> Path:
    """A directory with two datasets - the loader must combine them."""
    directory = tmp_path_factory.mktemp("puzzle_dataset_dir")
    _write_dataset(directory / "part_one.csv", count=600, offset=0)
    _write_dataset(directory / "part_two.csv", count=400, offset=600)
    return directory


def test_load_puzzles_real_set(puzzle_csv: Path):
    csv_file = str(puzzle_csv)
    print(f"\n  [Op 1/3] Loading dataset from '{csv_file}'...")
    puzzles = load_puzzles(csv_file)
    assert len(puzzles) == 1000
    ratings = [p.rating for p in puzzles]
    print(
        f"    [OK] Loaded {len(puzzles):,} puzzles | Rating range: [{min(ratings)} - {max(ratings)}]"
    )

    print(f"  [Op 2/3] Validating puzzle metadata & rating boundaries (sample: 20)...")
    for p in puzzles[:20]:
        assert 1200 <= p.rating <= 1600
        assert len(p.fen) > 10
        assert len(p.opponent_move) in (4, 5)
        assert len(p.solution_move) in (4, 5)
    print(
        f"    [OK] Validated 20 sample positions: all ratings within [1200, 1600], FENs and moves well-formed."
    )

    print(f"  [Op 3/3] Validating chess move legality in C++ engine...")
    for p in puzzles[:20]:
        g = zeropawn_cpp.ChessGame(p.fen)
        assert g.is_legal(p.opponent_move)
    print(f"    [OK] Validated 20 positions in C++ ChessGame: all opponent moves confirmed legal.")


def test_puzzle_evaluator_dummy_model(puzzle_csv: Path):
    print(f"\n  [Op 1/3] Initializing PuzzleEvaluator with ChessNet dummy model...")
    puzzles = load_puzzles(str(puzzle_csv))[:10]
    model = ChessNet(num_blocks=2, num_filters=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = PuzzleEvaluator(model=model, device=device, puzzles=puzzles)
    print(
        f"    [OK] Loaded {len(puzzles)} puzzles | Model: 2 blocks, 32 filters | Device: {device}"
    )

    # Test policy evaluation
    print(f"  [Op 2/3] Evaluating Policy Head (batch_size=5, total=10)...")
    t0 = time.perf_counter()
    res_policy = evaluator.evaluate_policy(batch_size=5)
    t_pol = time.perf_counter() - t0
    assert "solve_rate" in res_policy
    assert res_policy["total"] == 10
    assert 0.0 <= res_policy["solve_rate"] <= 1.0
    print(
        f"    [OK] Policy Head: Solved {res_policy['solved']}/{res_policy['total']} ({res_policy['solve_rate'] * 100:.1f}%) in {t_pol:.3f}s"
    )

    # Test MCTS evaluation on 2 puzzles
    print(f"  [Op 3/3] Evaluating MCTS Search (simulations=10, max_puzzles=2)...")
    t0 = time.perf_counter()
    res_mcts = evaluator.evaluate_mcts(num_simulations=10, max_puzzles=2)
    t_mcts = time.perf_counter() - t0
    assert "solve_rate" in res_mcts
    assert res_mcts["total"] == 2
    assert 0.0 <= res_mcts["solve_rate"] <= 1.0
    print(
        f"    [OK] MCTS Search: Solved {res_mcts['solved']}/{res_mcts['total']} ({res_mcts['solve_rate'] * 100:.1f}%) in {t_mcts:.3f}s"
    )


def test_load_puzzles_directory(puzzle_dataset_dir: Path):
    """Verify load_puzzles can load and combine all puzzles from a directory."""
    puzzles = load_puzzles(str(puzzle_dataset_dir))
    assert len(puzzles) >= 1000
    assert all(isinstance(p, Puzzle) for p in puzzles)


def test_puzzle_evaluator_batched_search(puzzle_csv: Path):
    puzzles = load_puzzles(str(puzzle_csv))[:4]
    model = ChessNet(num_blocks=2, num_filters=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = PuzzleEvaluator(
        model=model, device=device, puzzles=puzzles, search_mode="batched", batch_size=8
    )
    res = evaluator.evaluate_mcts(num_simulations=10, max_puzzles=2)
    assert res["search_mode"] == "batched"
    assert res["batch_size"] == 8
    assert res["total"] == 2
    assert 0.0 <= res["solve_rate"] <= 1.0
