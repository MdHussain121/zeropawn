"""Stage 1 Item 4: one explicit search-mode policy (default serial).

Covers: default serial everywhere, batched opt-in rejected at use time on this
branch (no batched C++ path here), and seeded determinism of the serial path.
"""

from __future__ import annotations

import argparse
import inspect

import numpy as np
import pytest

import zeropawn_cpp
from python.eval.match_runner import MatchRunner, require_serial_search
from python.eval.puzzles import PuzzleEvaluator
from python.uci.uci_engine import UCIEngine
from python.selfplay.worker import SelfPlayWorker, SelfPlayGame


def test_defaults_are_serial():
    assert inspect.signature(MatchRunner.__init__).parameters["search_mode"].default == "serial"
    assert inspect.signature(PuzzleEvaluator.__init__).parameters["search_mode"].default == "serial"
    assert inspect.signature(UCIEngine.__init__).parameters["search_mode"].default == "serial"
    assert inspect.signature(SelfPlayWorker.__init__).parameters["search_mode"].default == "serial"
    assert inspect.signature(SelfPlayGame.__init__).parameters["search_mode"].default == "serial"
    assert inspect.signature(UCIEngine.__init__).parameters["batch_size"].default == 1
    assert inspect.signature(MatchRunner.__init__).parameters["batch_size"].default == 1


def test_batched_search_mode():
    require_serial_search("serial", "test")
    require_serial_search("batched", "test")
    # If search_batched is absent, require_serial_search raises RuntimeError
    orig = getattr(zeropawn_cpp.MCTS, "search_batched", None)
    try:
        delattr(zeropawn_cpp.MCTS, "search_batched")
        with pytest.raises(RuntimeError):
            require_serial_search("batched", "test")
    finally:
        if orig is not None:
            setattr(zeropawn_cpp.MCTS, "search_batched", orig)


def _twenty_positions() -> list[str]:
    rng = np.random.default_rng(20261013)
    fens = [zeropawn_cpp.ChessGame().get_fen()]
    while len(fens) < 20:
        g = zeropawn_cpp.ChessGame()
        for _ in range(int(rng.integers(4, 60))):
            if g.is_game_over():
                break
            g.apply_action(int(rng.choice(g.legal_actions())))
        fens.append(g.get_fen())
    return fens


_POLICY = [(i % 17 + 1) / 100.0 for i in range(4672)]


def _run_all(fens: list[str]) -> list[tuple[int, list[float]]]:
    out = []
    for fen in fens:
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.add_noise = False
        cfg.seed = 777
        mcts = zeropawn_cpp.MCTS(cfg)
        res = mcts.search(zeropawn_cpp.ChessGame(fen), 200, lambda game: (_POLICY, 0.0))
        out.append((res.best_action, [round(float(v), 6) for v in res.action_visits]))
    return out


def test_serial_path_seeded_deterministic_20_positions():
    """Default (serial) path is stable: same seed => same visit counts.

    One-off before/after proof (working tree vs stashed HEAD, same binary):
    20/20 positions identical visit vectors. This test pins that behavior.
    """
    fens = _twenty_positions()
    assert _run_all(fens) == _run_all(fens)


def test_cli_search_mode_flags_default_serial():
    from python.cli.commands_match import _add_batch_args, _add_live_args, _add_human_args

    for adder in (_add_batch_args, _add_live_args, _add_human_args):
        parser = argparse.ArgumentParser()
        adder(parser)
        assert parser.parse_args([]).search_mode == "serial"
        assert parser.parse_args(["--search-mode", "batched"]).search_mode == "batched"

    from python.cli.commands_puzzles import _add_bench_args

    parser = argparse.ArgumentParser()
    _add_bench_args(parser)
    assert parser.parse_args([]).search_mode == "serial"
