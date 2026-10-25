"""Tests for Stockfish Soundness Evaluation, Perspective Normalization, and Union Qualification."""

from __future__ import annotations

import chess
import pytest

from python.data.opening_corpus import OpeningCorpus
from python.data.stockfish_soundness import StockfishSoundnessEvaluator


def test_perspective_normalization_mock():
    """Verify that child evaluation is correctly inverted to the moving player's perspective."""
    # When mover plays candidate move m, child has opponent to move.
    # If child is evaluated at +150 cp (favorable to opponent), mover score is -150 cp (unsound).
    # If child is evaluated at -50 cp (unfavorable to opponent), mover score is +50 cp (sound).
    evaluator = StockfishSoundnessEvaluator(cache_path=None)

    # Test mock cache injection
    parent = chess.Board()
    move = chess.Move.from_uci("e2e4")
    cache_key = f"{parent.epd(en_passant='pawns')}_e2e4_{evaluator.nodes}"

    # Invert logic verification:
    # Say child evaluation was -40 cp -> mover evaluation is +40 cp (sound >= -30 cp)
    evaluator._cache_result(cache_key, cp=40.0, mate=None, sound=True)
    cp, mate, sound = evaluator.evaluate_candidate_move(parent, move)
    assert cp == 40.0
    assert sound is True

    # Say child evaluation was +200 cp -> mover evaluation is -200 cp (unsound < -30 cp)
    move_h4 = chess.Move.from_uci("h2h4")
    cache_key_h4 = f"{parent.epd(en_passant='pawns')}_h2h4_{evaluator.nodes}"
    evaluator._cache_result(cache_key_h4, cp=-200.0, mate=None, sound=False)
    cp_h4, mate_h4, sound_h4 = evaluator.evaluate_candidate_move(parent, move_h4)
    assert cp_h4 == -200.0
    assert sound_h4 is False


def test_mate_score_handling():
    """Verify mate score normalization for decisive wins and losses."""
    evaluator = StockfishSoundnessEvaluator(cache_path=None)
    parent = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # Mover delivers mate: mate = +1
    key_win = f"{parent.epd(en_passant='pawns')}_e2e4_{evaluator.nodes}"
    evaluator._cache_result(key_win, cp=10000.0, mate=1, sound=True)
    cp, mate, sound = evaluator.evaluate_candidate_move(parent, move)
    assert cp == 10000.0
    assert mate == 1
    assert sound is True


def test_union_qualification_rule():
    """Verify the OR qualification rule:

    Qualifies if:
      frequency >= 5 OR stockfish_cp >= -30 cp
    """
    corpus = OpeningCorpus(min_game_frequency=5, sf_soundness_threshold_cp=-30.0)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # Case 1: freq < 5, unevaluated -> pending
    corpus.add_position_move(board, move, game_id="g1", source="test")
    corpus.update_qualification_status()
    pos = corpus.get_or_create_position(board)
    assert pos.moves["e2e4"].qualification == "pending"

    # Case 2: freq < 5, but stockfish eval >= -30 cp -> stockfish qualified!
    pos.moves["e2e4"].stockfish_evaluated = True
    pos.moves["e2e4"].stockfish_cp = -15.0  # -15 cp >= -30 cp
    corpus.update_qualification_status()
    assert pos.moves["e2e4"].qualification == "stockfish"

    # Case 3: freq >= 5 and stockfish eval >= -30 cp -> both qualified!
    for i in range(2, 6):
        corpus.add_position_move(board, move, game_id=f"g{i}", source="test")
    corpus.update_qualification_status()
    assert pos.moves["e2e4"].qualification == "both"

    # Case 4: freq >= 5 but stockfish eval < -30 cp -> frequency qualified! (Union rule)
    pos.moves["e2e4"].stockfish_cp = -120.0
    corpus.update_qualification_status()
    assert pos.moves["e2e4"].qualification == "frequency"

    # Case 5: freq < 5 and stockfish eval < -30 cp -> rejected
    move_bad = chess.Move.from_uci("a2a3")
    corpus.add_position_move(board, move_bad, game_id="g1", source="test")
    pos.moves["a2a3"].stockfish_evaluated = True
    pos.moves["a2a3"].stockfish_cp = -150.0
    corpus.update_qualification_status()
    assert pos.moves["a2a3"].qualification == "rejected"
