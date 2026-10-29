"""Regression tests for the repetition input-plane train/serve skew (audit H-2).

Plane 18 (one-fold repetition flag) is computed from game move history during
self-play/training. The play-time eval callbacks used to rebuild a
``ChessGame`` from FEN before encoding — erasing history — so the plane was
always zero during UCI/match/puzzle evaluation. After the fix the callbacks
encode the passed C++ game directly, matching the training encoding contract.

The FEN-reconstruction case (position loaded from FEN with no move history)
is documented as legitimately producing a zero repetition plane: history that
does not exist must not be fabricated.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import zeropawn_cpp


@pytest.fixture()
def repetition_game():
    """ChessGame with knight out-and-back: position revisits its start."""
    game = zeropawn_cpp.ChessGame()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        assert game.make_move_uci(move)
    return game


class EncodeSpy:
    """Wraps zeropawn_cpp, recording every encode_state call and its input game."""

    def __init__(self, real_module):
        self._real = real_module
        self.calls = []  # (game, planes)

    def encode_state(self, game):
        planes = self._real.encode_state(game)
        self.calls.append((game, planes))
        return planes

    def __getattr__(self, name):
        return getattr(self._real, name)


def tiny_model():
    from python.model.chess_net import ChessNet

    return ChessNet(num_filters=8, num_blocks=1)


def uniform_eval_from_planes(planes, game):
    n = game.num_actions()
    return (np.full(n, 1.0 / n, dtype=np.float32), 0.0)


# ==============================================================================
# Encoding contract
# ==============================================================================


def test_repetition_plane_set_for_history_game(repetition_game):
    """Training-side contract: an in-game repetition sets plane 18."""
    planes = zeropawn_cpp.encode_state(repetition_game)
    assert planes.shape == (19, 8, 8)
    assert planes[18].max() == 1.0


def test_fen_reconstruction_has_no_history(repetition_game):
    """Documented unavailable-history case: a FEN-rooted game cannot know about
    prior repetitions, so plane 18 stays zero (no fabricated history)."""
    reborn = zeropawn_cpp.ChessGame(repetition_game.get_fen())
    assert reborn.history_len() == 0
    assert zeropawn_cpp.encode_state(reborn)[18].max() == 0.0


# ==============================================================================
# UCI eval callbacks
# ==============================================================================


def test_uci_eval_callback_encodes_history_game(repetition_game):
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine(checkpoint_path="", use_book=False)
    engine.device = "cpu"
    engine.model = tiny_model()
    spy = EncodeSpy(engine.zeropawn_cpp)
    engine.zeropawn_cpp = spy

    policy, value = engine._eval_callback(repetition_game)
    assert len(policy) == repetition_game.num_actions()
    assert spy.calls, "eval callback must encode the passed game"
    encoded_game, planes = spy.calls[0]
    # The encoded game must be the actual passed game (with history), not a
    # FEN round-trip reconstruction (history_len 0).
    assert encoded_game.history_len() == repetition_game.history_len()
    assert planes[18].max() == 1.0


def test_uci_batch_eval_callback_encodes_history_game(repetition_game):
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine(checkpoint_path="", use_book=False)
    engine.device = "cpu"
    engine.model = tiny_model()
    spy = EncodeSpy(engine.zeropawn_cpp)
    engine.zeropawn_cpp = spy

    results = engine._eval_batch_callback([repetition_game])
    assert len(results) == 1
    assert spy.calls, "batch eval callback must encode the passed game"
    encoded_game, planes = spy.calls[0]
    assert encoded_game.history_len() == repetition_game.history_len()
    assert planes[18].max() == 1.0


def test_uci_search_receives_history_game(repetition_game):
    """End-to-end: the native search passes the real (history-carrying) game
    object — including clones of the root — into the eval callback."""
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine(checkpoint_path="", use_book=False)
    engine.device = "cpu"
    engine.model = tiny_model()
    spy = EncodeSpy(engine.zeropawn_cpp)
    engine.zeropawn_cpp = spy

    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 3
    mcts = zeropawn_cpp.MCTS(cfg)
    mcts.search(repetition_game, 2, engine._eval_callback)
    assert spy.calls
    # The very first evaluation is the root — it carries the repetition.
    root_game, root_planes = spy.calls[0]
    assert root_game.history_len() == repetition_game.history_len()
    assert root_planes[18].max() == 1.0


def test_uci_batch_evaluator_via_search(repetition_game):
    """The internal-batched path (config.batch_size > 1) also sees history."""
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine(checkpoint_path="", use_book=False)
    engine.device = "cpu"
    engine.model = tiny_model()
    spy = EncodeSpy(engine.zeropawn_cpp)
    engine.zeropawn_cpp = spy

    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 4
    cfg.batch_size = 4
    mcts = zeropawn_cpp.MCTS(cfg)
    mcts.search(repetition_game, 8, engine._eval_callback, engine._eval_batch_callback)
    assert spy.calls
    assert max(planes[18].max() for _, planes in spy.calls) == 1.0


# ==============================================================================
# Match runner eval callbacks
# ==============================================================================


def test_match_runner_callbacks_encode_history_game(repetition_game):
    import python.eval.match_runner as match_runner

    spy = EncodeSpy(match_runner.zeropawn_cpp)
    stub = SimpleNamespace(device="cpu", model=tiny_model(), _lock=threading.Lock())

    match_runner.zeropawn_cpp = spy
    try:
        policy, value = match_runner.MatchRunner._eval_callback(stub, repetition_game)
        assert len(policy) == repetition_game.num_actions()
        assert spy.calls[0][0].history_len() == repetition_game.history_len()
        assert spy.calls[0][1][18].max() == 1.0

        results = match_runner.MatchRunner._eval_batch_callback(stub, [repetition_game])
        assert len(results) == 1
        assert spy.calls[-1][1][18].max() == 1.0
    finally:
        match_runner.zeropawn_cpp = spy._real


# ==============================================================================
# Puzzle evaluator
# ==============================================================================


def test_puzzle_eval_encodes_search_leaf_games_directly(repetition_game):
    """The puzzle eval callback must encode the search's own leaf games (which
    carry the root's history) rather than FEN reconstructions (history_len 0)."""
    import python.eval.puzzles as puzzles_mod

    from python.eval.puzzles import Puzzle, PuzzleEvaluator

    history_lens = []

    real_module = puzzles_mod.zeropawn_cpp

    class PuzzleSpy:
        def encode_state(self, game):
            history_lens.append(game.history_len())
            return real_module.encode_state(game)

        def __getattr__(self, name):
            return getattr(real_module, name)

    puzzle = Puzzle(
        puzzle_id="test",
        fen=zeropawn_cpp.ChessGame().get_fen(),
        opponent_move="e2e4",
        solution_move="e7e5",
        rating=1500,
    )
    evaluator = PuzzleEvaluator(
        model=tiny_model(), device="cpu", puzzles=[puzzle], search_mode="serial"
    )
    puzzles_mod.zeropawn_cpp = PuzzleSpy()
    try:
        result = evaluator.evaluate_mcts(num_simulations=2, max_puzzles=1)
        assert result["total"] == 1
    finally:
        puzzles_mod.zeropawn_cpp = real_module

    # Leaf games evaluated during search are clones rooted at the puzzle
    # position with >= 1 applied move — history_len >= 1 means the callback
    # encoded the actual game object, not a FEN reconstruction (which would
    # report history_len == 0).
    assert history_lens, "puzzle eval must encode via the callback"
    assert max(history_lens) >= 1
