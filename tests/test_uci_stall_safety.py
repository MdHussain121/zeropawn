"""Deterministic concurrency tests for the UCI search-stall policy (F-03).

Audit scenario: the search thread runs ``mcts.search`` with the GIL released;
if the evaluator stalls inside a GIL-releasing native call, the in-search
abort predicate cannot fire, a ``stop`` request does not end the search, and
the main thread's 10-second join times out. The old implementation logged
"proceeding unsafely" and mutated ``mcts``/``game`` anyway — a use-after-free
window.

The remediation refuses every state-mutating command while the search thread
is alive past the stall timeout. These tests reproduce the stall
deterministically by blocking the evaluator on a ``threading.Event`` (which
releases the GIL — the same shape as a hung native call) and asserting that
``position``/``ucinewgame``/``setoption``/``go``/model-load are refused, that
recovery is automatic once the search finishes, and that repeated
stall/recovery cycles remain stable.
"""

import threading

import pytest

from python.uci.uci_engine import UCIEngine


@pytest.fixture()
def engine():
    # use_book=False keeps the stall scenario deterministic: an opening-book
    # answer for the start position would emit bestmove without ever reaching
    # the (blocked) evaluator.
    eng = UCIEngine(checkpoint_path="nonexistent-checkpoint.pt", device="cpu", use_book=False)
    eng._stall_timeout = 0.2
    eng._stop_event = threading.Event()
    eng._search_thread = None
    eng._search_stalled = False
    yield eng
    eng._stop_event.set()
    thread = eng._search_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=30)


def _start_stalled_search(engine, release: threading.Event) -> threading.Event:
    """Start a 'go infinite' search whose evaluator blocks until released."""
    entered = threading.Event()
    original_eval = engine._eval_callback

    def blocking_eval(game):
        entered.set()
        release.wait(timeout=60)
        return original_eval(game)

    engine._eval_callback = blocking_eval
    assert engine.handle_line("go infinite") is True
    assert entered.wait(timeout=30), "search never reached the evaluator"
    return entered


def _finish_stalled_search(engine, release: threading.Event) -> None:
    release.set()
    thread = engine._search_thread
    thread.join(timeout=60)
    assert not thread.is_alive(), "search thread did not finish after evaluator release"


def test_position_refused_while_search_stalled(engine):
    release = threading.Event()
    _start_stalled_search(engine, release)

    fen_before = engine.game.get_fen()
    assert engine.handle_line("position startpos moves e2e4 e7e5") is True
    # The command must be refused: game state untouched, stall flag set.
    assert engine.game.get_fen() == fen_before
    assert engine._search_stalled is True

    _finish_stalled_search(engine, release)

    # Recovery is automatic once the search thread finishes.
    assert engine.handle_line("position startpos moves e2e4 e7e5") is True
    assert engine.game.get_fen() != fen_before
    assert engine._search_stalled is False


def test_ucinewgame_refused_while_search_stalled(engine):
    release = threading.Event()
    _start_stalled_search(engine, release)

    game_before = engine.game
    assert engine.handle_line("ucinewgame") is True
    assert engine.game is game_before, "ucinewgame must not replace the game while stalled"
    assert engine._search_stalled is True

    _finish_stalled_search(engine, release)

    assert engine.handle_line("ucinewgame") is True
    assert engine.game is not game_before
    assert engine._search_stalled is False


def test_setoption_refused_while_search_stalled(engine):
    release = threading.Event()
    _start_stalled_search(engine, release)

    sims_before = engine.default_sims
    mcts_before = engine.mcts
    assert engine.handle_line("setoption name Simulations value 999") is True
    assert engine.default_sims == sims_before, "setoption must be refused while stalled"

    assert engine.handle_line("setoption name CP_UCT value 3.5") is True
    assert engine.mcts is mcts_before, "setoption must not rebuild the MCTS while stalled"

    _finish_stalled_search(engine, release)

    assert engine.handle_line("setoption name Simulations value 999") is True
    assert engine.default_sims == 999


def test_go_refused_while_search_stalled(engine):
    release = threading.Event()
    _start_stalled_search(engine, release)

    thread_before = engine._search_thread
    # A budgeted 'go' must not start a second (synchronous) search on top of
    # the stalled one, and must not replace the search thread.
    assert engine.handle_line("go movetime 50") is True
    assert engine._search_thread is thread_before

    # An async 'go infinite' must also be refused.
    assert engine.handle_line("go infinite") is True
    assert engine._search_thread is thread_before

    _finish_stalled_search(engine, release)


def test_model_load_refused_while_search_stalled(engine):
    release = threading.Event()
    _start_stalled_search(engine, release)

    assert engine.model is None
    engine.checkpoint_path = "also-nonexistent.pt"
    assert engine.handle_line("isready") is True  # still answers readyok
    assert engine.model is None, "isready must not trigger a model load while stalled"

    _finish_stalled_search(engine, release)


def test_repeated_stall_recovery_cycles_stable(engine):
    """The audit acceptance criterion: repeated stall/recovery iterations stay
    stable; commands refused during each stall, applied after each recovery."""
    move_pairs = ["e2e4 e7e5", "d2d4 d7d5", "c2c4 c7c5", "f2f4 f7f5", "g2g4 g7g5"]
    for cycle in range(10):
        moves = move_pairs[cycle % len(move_pairs)]
        release = threading.Event()
        _start_stalled_search(engine, release)

        fen_before = engine.game.get_fen()
        assert engine.handle_line(f"position startpos moves {moves}") is True
        assert engine._search_stalled is True
        assert engine.game.get_fen() == fen_before

        _finish_stalled_search(engine, release)

        assert engine.handle_line(f"position startpos moves {moves}") is True
        assert engine.game.get_fen() != fen_before
        assert engine._search_stalled is False


def test_normal_stop_finishes_without_stall(engine):
    """Sanity: without an evaluator stall, 'go infinite' + 'stop' completes
    within the timeout and no command is ever refused."""
    engine._stall_timeout = 30.0
    assert engine.handle_line("go infinite") is True
    assert engine.handle_line("stop") is True
    thread = engine._search_thread
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert engine._search_stalled is False

    # State mutation proceeds normally.
    assert engine.handle_line("position startpos moves e2e4") is True
    assert engine.game.get_fen() != engine.game.__class__().get_fen()


def test_search_still_answers_bestmove_after_stall(engine, capsys):
    release = threading.Event()
    _start_stalled_search(engine, release)

    engine.handle_line("position startpos moves e2e4 e7e5")  # refused, no crash
    _finish_stalled_search(engine, release)

    captured = capsys.readouterr().out
    assert "bestmove" in captured, "the interrupted search must still emit bestmove"
    assert "info string engine busy" in captured, "refusals must be reported on stdout"
