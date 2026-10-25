"""Tests for OpeningBook: Fast Lookup, Selection Policies, Legality, and Depth Cutoff."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from python.data.opening_book import OpeningBook
from python.uci.uci_engine import UCIEngine


@pytest.fixture
def sample_book_file(tmp_path: Path) -> Path:
    book_path = tmp_path / "test_book.json"
    book_data = {
        "summary": {"unique_canonical_positions": 2},
        "positions": {
            # Startpos
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": {
                "canonical_key": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "side_to_move": True,
                "ply": 0,
                "opening_name": "Starting Position",
                "moves": {
                    "e2e4": {
                        "uci": "e2e4",
                        "frequency": 100,
                        "stockfish_cp": 20.0,
                        "qualification": "both",
                    },
                    "d2d4": {
                        "uci": "d2d4",
                        "frequency": 80,
                        "stockfish_cp": 18.0,
                        "qualification": "both",
                    },
                    "c2c4": {
                        "uci": "c2c4",
                        "frequency": 40,
                        "stockfish_cp": 15.0,
                        "qualification": "both",
                    },
                    "g1f3": {
                        "uci": "g1f3",
                        "frequency": 30,
                        "stockfish_cp": 12.0,
                        "qualification": "both",
                    },
                    "a2a3": {
                        "uci": "a2a3",
                        "frequency": 1,
                        "stockfish_cp": -10.0,
                        "qualification": "stockfish",
                    },
                },
            },
            # 1. e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": {
                "canonical_key": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "side_to_move": True,
                "ply": 2,
                "opening_name": "King's Pawn Game",
                "moves": {
                    "g1f3": {
                        "uci": "g1f3",
                        "frequency": 150,
                        "stockfish_cp": 35.0,
                        "qualification": "both",
                    },
                    "f1c4": {
                        "uci": "f1c4",
                        "frequency": 30,
                        "stockfish_cp": 15.0,
                        "qualification": "both",
                    },
                },
            },
        },
    }
    book_path.write_text(json.dumps(book_data), encoding="utf-8")
    return book_path


def test_book_lookup_and_legality(sample_book_file: Path):
    """Verify position lookup and legality of returned moves."""
    book = OpeningBook(book_path=sample_book_file, policy="best")
    board = chess.Board()

    res = book.lookup(board)
    assert res is not None
    move_uci, meta = res
    assert move_uci in [m.uci() for m in board.legal_moves]
    # Under "best" policy, e2e4 has highest frequency (100)
    assert move_uci == "e2e4"
    assert meta["source"] == "opening_book"


def test_book_policies(sample_book_file: Path):
    """Verify different selection policies."""
    board = chess.Board()

    # Deterministic best
    b_best = OpeningBook(book_path=sample_book_file, policy="best", seed=42)
    m_best, _ = b_best.lookup(board)
    assert m_best == "e2e4"

    # Frequency sampling
    b_freq = OpeningBook(book_path=sample_book_file, policy="frequency", seed=42)
    m_freq, _ = b_freq.lookup(board)
    assert m_freq in ["e2e4", "d2d4", "c2c4", "g1f3", "a2a3"]

    # Score policy
    b_score = OpeningBook(book_path=sample_book_file, policy="score", seed=42)
    m_score, _ = b_score.lookup(board)
    assert m_score in ["e2e4", "d2d4", "c2c4", "g1f3", "a2a3"]


def test_book_depth_cutoff(sample_book_file: Path):
    """Verify lookup returns None when ply exceeds max_depth_plies."""
    book = OpeningBook(book_path=sample_book_file, max_depth_plies=2)
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    # Ply is now 2 >= max_depth_plies
    res = book.lookup(board)
    assert res is None


def test_book_fallback_when_disabled(sample_book_file: Path):
    """Verify book lookup returns None when disabled."""
    book = OpeningBook(book_path=sample_book_file, enabled=False)
    board = chess.Board()
    assert book.lookup(board) is None


def test_uci_engine_with_opening_book(sample_book_file: Path, monkeypatch, capsys):
    """Verify UCIEngine OwnBook option and protocol-clean output."""
    engine = UCIEngine(
        device="cpu",
        use_book=True,
        book_path=sample_book_file,
        book_policy="best",
    )

    # Test "uci" output contains OwnBook option
    engine.handle_line("uci")
    out = capsys.readouterr().out
    assert "option name OwnBook type check default true" in out
    assert "option name BookFile type string" in out
    assert "option name BookPolicy type combo" in out

    # Test bestmove from book
    engine.handle_line("ucinewgame")
    engine.handle_line("position startpos")
    bestmove = engine.search_and_respond(sims=50)
    out = capsys.readouterr().out

    # Must pick e2e4 from book
    assert bestmove == "e2e4"
    assert "info depth 1 nodes 1 score cp 0 pv e2e4\n" in out
    assert "bestmove e2e4\n" in out
