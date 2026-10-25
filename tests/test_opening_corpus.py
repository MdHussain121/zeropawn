"""Tests for OpeningCorpus: Ingestion, Validation, Deduplication, and Provenance."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from python.data.opening_corpus import OpeningCorpus, canonical_position_key


def test_canonical_position_key_transpositions():
    """Verify that transposed move orders reach identical canonical keys."""
    # 1. d4 Nf6 2. c4 e6
    b1 = chess.Board()
    b1.push_san("d4")
    b1.push_san("Nf6")
    b1.push_san("c4")
    b1.push_san("e6")

    # 1. c4 e6 2. d4 Nf6
    b2 = chess.Board()
    b2.push_san("c4")
    b2.push_san("e6")
    b2.push_san("d4")
    b2.push_san("Nf6")

    k1 = canonical_position_key(b1)
    k2 = canonical_position_key(b2)
    # Transpositions have identical canonical keys
    assert k1 == k2
    assert b1.piece_map() == b2.piece_map()
    assert b1.turn == b2.turn
    assert b1.castling_rights == b2.castling_rights


def test_side_to_move_differentiation():
    """Ensure positions with different sides to move are never conflated."""
    b_white = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    b_white_to_move = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1")

    assert canonical_position_key(b_white) != canonical_position_key(b_white_to_move)


def test_add_position_move_and_frequency_threshold():
    """Verify distinct game counting and frequency threshold."""
    corpus = OpeningCorpus(min_game_frequency=5)
    board = chess.Board()
    move_e4 = chess.Move.from_uci("e2e4")

    # Add 4 distinct games -> pending
    for i in range(4):
        ok = corpus.add_position_move(board, move_e4, game_id=f"game_{i}", source="test")
        assert ok is True

    counts = corpus.update_qualification_status()
    pos = corpus.get_or_create_position(board)
    assert len(pos.moves["e2e4"].distinct_game_ids) == 4
    assert pos.moves["e2e4"].qualification == "pending"

    # Add 5th game -> frequency qualified!
    corpus.add_position_move(board, move_e4, game_id="game_5", source="test")
    counts = corpus.update_qualification_status()
    assert pos.moves["e2e4"].qualification == "frequency"
    assert counts["frequency_only"] >= 1


def test_illegal_move_quarantine():
    """Verify illegal moves are safely quarantined without crashing."""
    corpus = OpeningCorpus()
    board = chess.Board()
    illegal_move = chess.Move.from_uci("e2e5")  # Pawn can't jump 3 squares

    ok = corpus.add_position_move(board, illegal_move, source="bad_pgn")
    assert ok is False
    assert corpus.records_rejected == 1
    assert len(corpus.quarantined_items) == 1
    assert "Illegal move" in corpus.quarantined_items[0]["reason"]


def test_ingest_epd(tmp_path: Path):
    """Test ingesting valid and malformed EPD lines."""
    epd_file = tmp_path / "test.epd"
    epd_content = (
        'r1bqk2r/ppppbppp/2nn4/1B2N3/8/8/PPPP1PPP/RNBQR1K1 w kq - 1 7 ; id "Berlin Wall";\n'
        'invalid_fen_string_here ; id "Broken";\n'
    )
    epd_file.write_text(epd_content, encoding="utf-8")

    corpus = OpeningCorpus()
    count = corpus.ingest_epd(epd_file, source_name="test_epd")
    assert count == 1
    assert corpus.records_rejected == 1
    assert len(corpus.positions) == 1


def test_ingest_json_catalog(tmp_path: Path):
    """Test ingesting JSON catalog format."""
    catalog_file = tmp_path / "catalog.json"
    catalog_data = [
        {
            "name": "Ruy Lopez",
            "eco": "C60",
            "count": 10,
            "uci": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
        }
    ]
    catalog_file.write_text(json.dumps(catalog_data), encoding="utf-8")

    corpus = OpeningCorpus()
    count = corpus.ingest_json_catalog(catalog_file, source_name="test_cat")
    assert count == 1

    startpos = chess.Board()
    start_key = canonical_position_key(startpos)
    assert start_key in corpus.positions
    assert "e2e4" in corpus.positions[start_key].moves
    assert corpus.positions[start_key].moves["e2e4"].frequency >= 5
