"""M-1 regression tests: the three CLI play paths must encode the live game.

The 2026-10-31 model-architecture/inference audit (finding M-1) confirmed
that ``match live``, ``match human`` and ``puzzles solve`` rebuilt the
position with ``ChessGame(game.get_fen())`` before encoding, silently zeroing
plane 18 (one-fold repetition) and dropping the move history.  All three now
go through the shared factory in ``python/cli/eval_callbacks.py``, which
encodes the passed game object directly (the H-2 contract already used by
UCI/match-runner/puzzles library paths).

Two layers of coverage:

1. Factory semantics (hermetic): direct-game encoding, repetition plane,
   both sides to move, ordinary positions, WDL value contract.
2. Command integration: the real ``run_live_game`` / ``run_human_game`` /
   ``run_solve_puzzles`` entry points are driven with a scripted policy
   model and fake Stockfish/inputs; every ``encode_state`` call the engine
   makes is spied and asserted to carry the live history.  Against the old
   inline callbacks every recorded game would have ``history_len() == 0``
   (FEN reconstruction), so these tests fail on the defective code.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import zeropawn_cpp  # noqa: E402
import python.cli.eval_callbacks as eval_callbacks  # noqa: E402
import python.cli.commands_match as commands_match  # noqa: E402
import python.cli.commands_puzzles as commands_puzzles  # noqa: E402
import python.eval.match_runner as match_runner  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KNIGHT_SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _play(game, moves) -> None:
    for uci in moves:
        assert game.make_move_uci(uci), f"illegal move in test sequence: {uci}"


def _white_repetition_game():
    """Start position seen twice, White to move (plane 18 must be 1.0)."""
    game = zeropawn_cpp.ChessGame()
    _play(game, KNIGHT_SHUFFLE)
    return game


def _black_repetition_game():
    """Position after 1.e4 seen twice, Black to move (plane 18 must be 1.0)."""
    game = zeropawn_cpp.ChessGame()
    _play(game, ["e2e4", "g8f6", "g1f3", "f6g8", "f3g1"])
    return game


class EncodeSpy:
    """Wraps zeropawn_cpp, recording every encode_state call and its input game.

    ``snapshots`` records the game's state (history length, repetition flag)
    *at call time*: command loops keep mutating the live game after each
    evaluation, so assertions must use the snapshot, not the later state.
    """

    def __init__(self, real_module):
        self._real = real_module
        self.calls = []  # (game, planes)
        self.snapshots = []  # dicts captured at call time

    def encode_state(self, game):
        planes = self._real.encode_state(game)
        self.calls.append((game, planes))
        self.snapshots.append(
            {
                "history_len": game.history_len(),
                "plane18": float(planes[18].max()),
                "is_rep": bool(game.is_repetition(1)),
            }
        )
        return planes

    def __getattr__(self, name):
        return getattr(self._real, name)


class ScriptedPolicyModel:
    """Deterministic callable model stub for driving the real CLI commands.

    Returns fixed WDL logits (default 0 -> value 0.0) and a policy whose mass
    is concentrated on a scripted move so the real MCTS deterministically
    plays the scripted continuation.  ``bias_fn`` inspects the encoded planes
    of the first row and returns the action index to favor.
    """

    def __init__(self, bias_fn=None, wdl_logits=(0.0, 0.0, 0.0)):
        self.bias_fn = bias_fn
        self.wdl_logits = tuple(float(v) for v in wdl_logits)
        self.calls = 0

    def __call__(self, t):
        self.calls += 1
        b = int(t.shape[0])
        p = torch.zeros((b, 4672), dtype=torch.float32)
        if self.bias_fn is not None:
            idx = self.bias_fn(t[0])
            if idx is not None:
                p[:, idx] = 100.0
        w = torch.tensor([self.wdl_logits], dtype=torch.float32).repeat(b, 1)
        return p, w


def _tiny_factory_wrapper(monkeypatch, command_module, spy):
    """Make a command's ``make_serial_eval_fn`` build the shared factory with
    the spy module injected (records the games handed to the encoder)."""
    real_factory = eval_callbacks.make_serial_eval_fn

    def wrapper(model, device):
        return real_factory(model, device, cpp_module=spy)

    monkeypatch.setattr(command_module, "make_serial_eval_fn", wrapper)


@pytest.fixture()
def dummy_ckpt(tmp_path):
    p = tmp_path / "dummy_model.pt"
    torch.save({"model": {"stem_conv.weight": torch.zeros(1)}, "step": 0}, p)
    return p


# ---------------------------------------------------------------------------
# 1. Factory semantics
# ---------------------------------------------------------------------------


def test_factory_encodes_the_passed_game_directly_white_repetition():
    game = _white_repetition_game()
    spy = EncodeSpy(zeropawn_cpp)
    model = ScriptedPolicyModel(wdl_logits=(0.0, 0.0, 0.0))
    eval_fn = eval_callbacks.make_serial_eval_fn(model, "cpu", cpp_module=spy)

    policy, value = eval_fn(game)

    assert len(spy.calls) == 1, "exactly one encode per evaluation"
    encoded_game, planes = spy.calls[0]
    # The encoded input is the *same* game object carrying real history —
    # not a FEN reconstruction (which has history_len 0).
    assert encoded_game is game
    assert encoded_game.history_len() == 4
    assert planes[18].max() == 1.0, "plane 18 (one-fold repetition) must be set"
    assert np.array_equal(planes, zeropawn_cpp.encode_state(game))
    # Sanity: the FEN reconstruction the old code used would have zeroed it.
    rebuilt = zeropawn_cpp.ChessGame(game.get_fen())
    assert zeropawn_cpp.encode_state(rebuilt)[18].max() == 0.0


def test_factory_encodes_the_passed_game_directly_black_repetition():
    game = _black_repetition_game()
    assert game.side_to_move() is False  # Black to move
    spy = EncodeSpy(zeropawn_cpp)
    eval_fn = eval_callbacks.make_serial_eval_fn(
        ScriptedPolicyModel(), "cpu", cpp_module=spy
    )

    eval_fn(game)

    encoded_game, planes = spy.calls[0]
    assert encoded_game is game
    assert planes[18].max() == 1.0


def test_factory_preserves_ordinary_positions():
    game = zeropawn_cpp.ChessGame()  # startpos, no history, White to move
    spy = EncodeSpy(zeropawn_cpp)
    eval_fn = eval_callbacks.make_serial_eval_fn(
        ScriptedPolicyModel(), "cpu", cpp_module=spy
    )

    policy, value = eval_fn(game)

    encoded_game, planes = spy.calls[0]
    assert encoded_game is game
    assert planes.shape == (19, 8, 8)
    assert planes[18].max() == 0.0
    assert np.array_equal(planes, zeropawn_cpp.encode_state(game))
    # Full-size, normalized policy over the canonical action space.
    assert len(policy) == 4672
    assert abs(float(np.sum(policy)) - 1.0) < 1e-5
    assert value == pytest.approx(0.0, abs=1e-6)


def test_factory_value_is_side_to_move_relative_win_minus_loss():
    """WDL ordering contract through the real CLI callback chain: logits
    [win, draw, loss] with win dominant must yield a positive value."""
    logits = (2.0, 0.0, -2.0)
    model = ScriptedPolicyModel(wdl_logits=logits)
    eval_fn = eval_callbacks.make_serial_eval_fn(model, "cpu", cpp_module=EncodeSpy(zeropawn_cpp))
    game = zeropawn_cpp.ChessGame()

    _, value = eval_fn(game)

    probs = np.exp(np.array(logits) - np.max(logits))
    probs = probs / probs.sum()
    expected = float(probs[0] - probs[2])
    assert expected > 0
    assert value == pytest.approx(expected, abs=1e-6)
    # Swapped logits (loss dominant) must flip the sign.
    model2 = ScriptedPolicyModel(wdl_logits=(-2.0, 0.0, 2.0))
    eval_fn2 = eval_callbacks.make_serial_eval_fn(
        model2, "cpu", cpp_module=EncodeSpy(zeropawn_cpp)
    )
    _, value2 = eval_fn2(game)
    assert value2 == pytest.approx(-expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Command integration — match live
# ---------------------------------------------------------------------------


class _ScriptedStockfish:
    """Deterministic Stockfish stand-in: plays a scripted move list."""

    def __init__(self, *args, **kwargs):
        self._moves = ["g8f6", "f6g8", "zzzz"]  # last one is illegal -> loop ends
        self._i = 0

    def get_move(self, fen: str) -> str:
        move = self._moves[min(self._i, len(self._moves) - 1)]
        self._i += 1
        return move

    def close(self) -> None:
        pass


def test_match_live_eval_callback_preserves_history_and_repetition(
    tmp_path, dummy_ckpt, monkeypatch
):
    spy = EncodeSpy(zeropawn_cpp)
    _tiny_factory_wrapper(monkeypatch, commands_match, spy)

    # Scripted engine policy (engine is White):
    #   startpos (white)        -> g1f3
    #   after g1f3 g8f6 (white) -> f3g1
    # Fake Stockfish mirrors: g8f6, then f6g8 -> start position repeats.
    idx_g1f3 = zeropawn_cpp.move_to_index(zeropawn_cpp.ChessGame(), "g1f3")
    g2 = zeropawn_cpp.ChessGame()
    _play(g2, ["g1f3", "g8f6"])
    idx_f3g1 = zeropawn_cpp.move_to_index(g2, "f3g1")

    def bias_fn(row):
        return idx_f3g1 if row[1, 2, 5].item() > 0.5 else idx_g1f3

    model = ScriptedPolicyModel(bias_fn=bias_fn)

    monkeypatch.setattr(
        commands_match, "load_engine_model", lambda path, device=None: (model, {"path": path})
    )
    monkeypatch.setattr(
        match_runner, "StockfishPlayer", _ScriptedStockfish
    )
    sf_dummy = tmp_path / "stockfish.exe"
    sf_dummy.write_bytes(b"x")
    monkeypatch.setattr(commands_match, "get_default_stockfish_path", lambda: sf_dummy)

    args = SimpleNamespace(
        checkpoint=str(dummy_ckpt),
        stockfish_path=None,
        device="cpu",
        fen=START_FEN,
        engine_color="white",
        mcts_sims=8,
        delay=0.0,
        step=False,
        batch_size="1",
        search_mode="serial",
        stockfish_nodes=10,
        reuse_tree=False,
    )
    rc = commands_match.run_live_game(args)
    assert rc == 0

    assert spy.snapshots, "engine search must have encoded at least one position"
    # THE M-1 PROOF: positions handed to the encoder carry the live move
    # history.  The old FEN round-trip always produced history_len() == 0 and
    # could never show a repetition.
    history_lens = [s["history_len"] for s in spy.snapshots]
    assert max(history_lens) >= 4, history_lens
    # The repetition was reached through the real game loop and the engine's
    # root evaluation saw it (plane 18 = 1.0 at encode time).
    repeated = [s for s in spy.snapshots if s["plane18"] == 1.0]
    assert repeated, "no encoded position carried the repetition plane"
    assert repeated[-1]["is_rep"] is True


# ---------------------------------------------------------------------------
# 2b. Command integration — match human
# ---------------------------------------------------------------------------


def test_match_human_eval_callback_preserves_history_and_repetition(
    tmp_path, dummy_ckpt, monkeypatch
):
    spy = EncodeSpy(zeropawn_cpp)
    _tiny_factory_wrapper(monkeypatch, commands_match, spy)

    idx_g1f3 = zeropawn_cpp.move_to_index(zeropawn_cpp.ChessGame(), "g1f3")
    g2 = zeropawn_cpp.ChessGame()
    _play(g2, ["g1f3", "g8f6"])
    idx_f3g1 = zeropawn_cpp.move_to_index(g2, "f3g1")

    def bias_fn(row):
        return idx_f3g1 if row[1, 2, 5].item() > 0.5 else idx_g1f3

    model = ScriptedPolicyModel(bias_fn=bias_fn)
    monkeypatch.setattr(
        commands_match, "load_engine_model", lambda path, device=None: (model, {"path": path})
    )

    # Human (Black) plays the mirror of the engine's knight shuffle, then quits.
    answers = iter(["g8f6", "f6g8", "quit"])
    monkeypatch.setattr(
        commands_match.console, "input", lambda *a, **k: next(answers)
    )

    args = SimpleNamespace(
        checkpoint=str(dummy_ckpt),
        device="cpu",
        human_color="black",
        mcts_sims=8,
        search_mode="serial",
    )
    rc = commands_match.run_human_game(args)
    assert rc == 0

    assert spy.snapshots, "engine search must have encoded at least one position"
    history_lens = [s["history_len"] for s in spy.snapshots]
    assert max(history_lens) >= 4, history_lens
    repeated = [s for s in spy.snapshots if s["plane18"] == 1.0]
    assert repeated, "no encoded position carried the repetition plane"
    assert repeated[-1]["is_rep"] is True


# ---------------------------------------------------------------------------
# 2c. Command integration — puzzles solve
# ---------------------------------------------------------------------------


class _NonTtyStdin(io.StringIO):
    def isatty(self) -> bool:  # pragma: no cover - trivial
        return False


def test_puzzles_solve_eval_callback_preserves_history(tmp_path, dummy_ckpt, monkeypatch):
    spy = EncodeSpy(zeropawn_cpp)
    _tiny_factory_wrapper(monkeypatch, commands_puzzles, spy)

    model = ScriptedPolicyModel()
    monkeypatch.setattr(
        commands_puzzles, "load_engine_model", lambda path, device=None: (model, {"path": path})
    )
    monkeypatch.setattr(commands_puzzles, "sys", SimpleNamespace(stdin=_NonTtyStdin()))

    csv_path = tmp_path / "puzzles.csv"
    csv_path.write_text(
        "puzzle_id,fen,moves,rating\n"
        f"test-1,{START_FEN},e2e4 e7e5,1500\n",
        encoding="utf-8",
    )

    args = SimpleNamespace(
        checkpoint=str(dummy_ckpt),
        device="cpu",
        puzzles_file=str(csv_path),
        start_index=0,
        human_guess=False,
        mcts_sims=4,
    )
    rc = commands_puzzles.run_solve_puzzles(args)
    assert rc == 0

    assert spy.snapshots, "puzzle search must have encoded at least one position"
    # The search games are the puzzle position with the opponent move applied:
    # their history must reach the callback intact (snapshotted at encode
    # time — search leaves are C++-owned and may not outlive the search).
    # The old FEN round-trip always encoded history-less reconstructions
    # (history_len == 0 at call time).
    history_lens = [s["history_len"] for s in spy.snapshots]
    assert max(history_lens) >= 1, history_lens
