"""Regression tests for the 2026-10-26 search-quality remediation (F-1..F-8, F-11/F-12, F-14/F-15).

Covers the audit's gap list:
1.  F-1: batched self-play completes through the real SelfPlayGame/worker
    interface without an arity or evaluator-contract error.
2.  F-2: UCI/match evaluation callbacks return contiguous float32 NumPy arrays.
3.  F-3: UCI time management — allocation policy, deadline enforcement, stop.
4.  F-4: ChessGame::apply_action raises on undecodable actions.
5.  F-6: MCTS NN-cache respects the configured per-instance cap.
6.  F-7: inference failures fail closed (InferenceError), games marked invalid,
    no samples emitted.
7.  F-8: serial vs batched decision parity on the 20-FEN golden battery with a
    deterministic evaluator.
8.  F-11/F-12: warm-start validation (missing / unregistered / designated).
9.  F-14: puzzle-decline band calibrated to certified-set sampling noise.
10. F-15: per-game match persistence callback + interrupted-match partial tally.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.selfplay.worker import (
    InferenceError,
    SelfPlayGame,
    SelfPlayWorker,
    WorkerBatchAggregator,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dummy_ckpt(tmp_path_factory):
    """A valid-architecture dummy checkpoint (12 blocks x 128 filters)."""
    p = tmp_path_factory.mktemp("ckpts") / "dummy_model.pt"
    if not p.exists():
        model = ChessNet(num_blocks=12, num_filters=128)
        torch.save(
            {"model": model.state_dict(), "num_blocks": 12, "num_filters": 128, "step": 1},
            p,
        )
    return p


class UniformClient:
    """Minimal InferenceClient stand-in: uniform policy, zero value, fast."""

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after
        self.version = 1

    def evaluate_leaves(self, states: np.ndarray):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated CUDA failure")
        n = states.shape[0]
        policy = np.full((n, 4672), 1.0 / 4672, dtype=np.float32)
        wdl = np.full((n, 3), 1.0 / 3, dtype=np.float32)
        return policy, wdl, self.version


class FlakyClient(UniformClient):
    """Fails every call after warmup, then recovers (transient-failure shape)."""

    def __init__(self, fail_calls: int = 1):
        super().__init__()
        self.fail_calls = fail_calls
        self._failed = 0

    def evaluate_leaves(self, states: np.ndarray):
        if self._failed < self.fail_calls:
            self._failed += 1
            raise RuntimeError("simulated transient CUDA failure")
        return super().evaluate_leaves(states)


# ---------------------------------------------------------------------------
# F-1: batched self-play through the real game interface
# ---------------------------------------------------------------------------


def test_f01_batched_selfplay_completes_without_typeerror():
    """The old code called search_batched(game, sims, single_leaf_eval) — 3 args
    against a 4-arg binding — raising TypeError on the first search. The
    remediation routes batched mode through the canonical search(batch_evaluator)
    contract; this test fails if that ever regresses."""
    aggregator = WorkerBatchAggregator(UniformClient(), max_batch_size=8)
    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()
    try:
        game = SelfPlayGame(
            game_id=1,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=12,
            fast_search_sims=6,
            full_search_prob=0.5,
            temp_cutoff_range=(3, 5),
            max_plies=12,
            use_book=False,
            search_mode="batched",
        )
        positions, summary = game.play_game()
        assert len(positions) > 0
        assert summary.game_id == 1
        for pos in positions:
            assert pos.visits.shape == (4672,)
            assert abs(pos.visits.sum() - 1.0) < 1e-4
    finally:
        aggregator.stop()


def test_f01_batched_and_serial_both_use_one_inference_pipeline():
    """Both modes must submit leaves through the same aggregator (single
    inference pipeline), i.e., the client sees the calls."""
    client = UniformClient()
    aggregator = WorkerBatchAggregator(client, max_batch_size=8)
    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()
    try:
        game = SelfPlayGame(
            game_id=2,
            aggregator=aggregator,
            search_mode="batched",
            full_search_sims=8,
            fast_search_sims=4,
            full_search_prob=1.0,
            temp_cutoff_range=(2, 3),
            max_plies=6,
            use_book=False,
        )
        game.play_game()
        assert client.calls > 0
        assert aggregator.failed_flushes == 0
    finally:
        aggregator.stop()


# ---------------------------------------------------------------------------
# F-4: invalid action hardening (C++)
# ---------------------------------------------------------------------------


def test_f04_apply_action_raises_on_undecodable_action():
    for bad in (-1, 9999, 4671):
        g = zeropawn_cpp.ChessGame()
        with pytest.raises(ValueError):
            g.apply_action(bad)
        # State unchanged and usable afterwards (no corruption).
        assert g.get_fen().startswith("rnbqkbnr/pppppppp")
        assert len(g.legal_actions()) == 20


def test_f04_search_survives_evaluator_exception_with_clean_state():
    """An exception escaping the evaluator (which is how apply_action errors
    surface) must unwind virtual loss cleanly and leave the MCTS reusable."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.virtual_loss = 1
    mcts = zeropawn_cpp.MCTS(cfg)
    g = zeropawn_cpp.ChessGame()

    def bad_eval(_game):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        mcts.search(g, 10, bad_eval)
    assert mcts.total_active_virtual_losses() == 0

    def good_eval(game):
        return (np.full(game.num_actions(), 1.0 / game.num_actions(), dtype=np.float32), 0.0)

    res = mcts.search(g, 8, good_eval)
    assert res.new_simulations == 8
    assert mcts.total_active_virtual_losses() == 0


# ---------------------------------------------------------------------------
# F-3 (C++ side): abort predicate + actual simulation reporting
# ---------------------------------------------------------------------------


def test_f03_abort_fn_stops_search_and_reports_actual_sims():
    g = zeropawn_cpp.ChessGame()

    def eval_fn(game):
        return (np.full(game.num_actions(), 1.0 / game.num_actions(), dtype=np.float32), 0.0)

    cfg = zeropawn_cpp.MCTSConfig()
    mcts = zeropawn_cpp.MCTS(cfg)

    polls = {"n": 0}

    def abort_after_two_polls():
        polls["n"] += 1
        return polls["n"] > 2

    res = mcts.search(g, 10_000, eval_fn, None, abort_after_two_polls)
    # First poll is at sim 0; second at sim 32; abort triggers on the third
    # poll → exactly 64 simulations were committed.
    assert res.new_simulations == 64
    assert res.best_action >= 0

    # Without abort the full budget is reported.
    res_full = mcts.search(g, 40, eval_fn)
    assert res_full.new_simulations == 40


def test_f03_batched_path_reports_batch_size_and_sims():
    g = zeropawn_cpp.ChessGame()
    state_size = g.state_size()
    num_actions = g.num_actions()

    def raw_eval(flat_planes):
        arr = np.asarray(flat_planes)
        n = arr.shape[0]
        policy = np.full((n, num_actions), 1.0 / num_actions, dtype=np.float32)
        values = np.zeros((n,), dtype=np.float32)
        return policy, values

    cfg = zeropawn_cpp.MCTSConfig()
    mcts = zeropawn_cpp.MCTS(cfg)
    res = mcts.search_batched(g, 20, raw_eval, 4)
    assert res.new_simulations == 20
    assert res.batch_size == 4


# ---------------------------------------------------------------------------
# F-6: per-instance NN-cache cap
# ---------------------------------------------------------------------------


def test_f06_nn_cache_respects_configured_cap():
    g = zeropawn_cpp.ChessGame()

    def eval_fn(game):
        return (np.full(game.num_actions(), 1.0 / game.num_actions(), dtype=np.float32), 0.0)

    # Move the game around so many distinct positions get evaluated.
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.nn_cache_cap = 5
    mcts = zeropawn_cpp.MCTS(cfg)
    for _ in range(6):
        res = mcts.search(g, 24, eval_fn)
        assert mcts.cache_size() <= 5
        act = res.best_action
        if act >= 0 and not g.is_game_over():
            try:
                g.apply_action(act)
            except ValueError:
                break
    assert mcts.cache_size() <= 5

    # Default cap unchanged (100,000).
    cfg_default = zeropawn_cpp.MCTSConfig()
    assert cfg_default.nn_cache_cap == 0  # 0 = use NN_CACHE_CAP


# ---------------------------------------------------------------------------
# F-7: fail-closed inference
# ---------------------------------------------------------------------------


def test_f07_inference_failure_marks_game_invalid_and_emits_nothing():
    """Permanent inference failure: no game can complete, so the sink must
    never receive samples and the position counter must stay at zero."""
    worker = SelfPlayWorker(
        worker_id=0,
        client=UniformClient(fail_after=0),  # every inference call fails
        num_concurrent_games=2,
        full_search_sims=6,
        fast_search_sims=3,
        full_search_prob=1.0,
        temp_cutoff_range=(2, 3),
        max_plies=8,
        position_sink=lambda pos, summary: pytest.fail(
            "sink must not receive samples from an inference-failed game"
        ),
        use_book=False,
    )
    worker.start()
    deadline = time.time() + 30
    while time.time() < deadline and worker.invalid_games < 2:
        time.sleep(0.05)
    worker.stop()
    assert worker.invalid_games >= 1
    assert worker.total_positions_generated == 0
    assert worker.total_games_played == 0


def test_f07_transient_failure_discards_game_and_keeps_worker_alive():
    """One transient failure: the affected game is marked invalid and its
    samples are discarded, while the worker recovers and keeps producing
    healthy games (documented fail-closed + keep-alive semantics)."""
    sink_calls: list[int] = []
    worker = SelfPlayWorker(
        worker_id=0,
        client=FlakyClient(fail_calls=1),
        num_concurrent_games=2,
        full_search_sims=6,
        fast_search_sims=3,
        full_search_prob=1.0,
        temp_cutoff_range=(2, 3),
        max_plies=8,
        position_sink=lambda pos, summary: sink_calls.append(len(pos)),
        use_book=False,
    )
    worker.start()
    deadline = time.time() + 30
    while time.time() < deadline and (worker.invalid_games < 1 or not sink_calls):
        time.sleep(0.05)
    worker.stop()
    assert worker.invalid_games >= 1
    # The worker recovered from the transient failure and produced healthy games.
    assert worker.total_games_played >= 1
    assert worker.total_positions_generated == sum(sink_calls)


def test_f07_waiter_raises_inference_error_not_uniform_policy():
    aggregator = WorkerBatchAggregator(UniformClient(fail_after=0), max_batch_size=4)
    t = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t.start()
    try:
        g = zeropawn_cpp.ChessGame()
        with pytest.raises(InferenceError):
            aggregator.evaluate_game_state(g)
        assert aggregator.failed_flushes >= 1
    finally:
        aggregator.stop()


# ---------------------------------------------------------------------------
# F-8: serial vs batched decision parity on the golden 20-FEN battery
# ---------------------------------------------------------------------------

GOLDEN_20_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 1 6",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
    "r1b2rk1/pp1p1pp1/1qn1pn1p/8/1b1NP3/2NBB3/PPP2PPP/R2Q1RK1 w - - 4 10",
    "r2q1rk1/1bp1bppp/p1np4/1p2p3/3PP3/1BN1PN2/PPP3PP/R2Q1RK1 w - - 0 12",
    "r2qkb1r/pp2pppp/2n1b3/2pn4/2B5/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 7",
    "8/5pk1/4p1p1/7p/3R3P/4PKP1/5P2/2r5 b - - 6 38",
    "6k1/5p2/p3p1p1/1p5p/1P1q4/P4Q1P/5PP1/6K1 w - - 0 34",
    "8/p4pk1/1p1b1n1p/3p1Bp1/3P4/PP3NPP/5PK1/8 w - - 1 31",
    "r1bqk2r/ppp1b1pp/2n1pn2/3p4/2PP4/2NBP3/PP3PPP/R1BQK1NR w KQkq - 0 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4r1k1/5ppp/1p2p3/2q5/6b1/2P5/PP1Q1PPP/2BR1RK1 b - - 0 22",
    "2rr2k1/pb1nqpp1/1p2pb1p/7p/3P4/2PNBNP1/PPB1QPBP/2RR2K1 w - - 6 18",
]


def _deterministic_evaluator():
    """Policy/value as a pure function of the position hash — deterministic
    across serial and batched evaluation orders."""

    def eval_fn(game):
        r = random.Random((game.get_hash() ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        n = game.num_actions()
        raw = [r.random() + 1e-6 for _ in range(n)]
        s = sum(raw)
        policy = np.array([x / s for x in raw], dtype=np.float32)
        return (np.ascontiguousarray(policy), float(r.uniform(-0.5, 0.5)))

    return eval_fn


@pytest.mark.parametrize("fen", GOLDEN_20_FENS)
def test_f08_serial_batched_decision_parity(fen):
    """At equal simulation counts, serial search and canonical intra-tree
    batched search must agree on the best move and (closely) on root value.
    Batched evaluation order differs, so visits are intentionally not compared."""
    sims = 48

    def make_mcts():
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.c_puct = 2.0
        cfg.fpu_reduction = 0.2
        cfg.virtual_loss = 1
        cfg.add_noise = False
        cfg.seed = 42
        return zeropawn_cpp.MCTS(cfg)

    eval_fn = _deterministic_evaluator()

    g_serial = zeropawn_cpp.ChessGame(fen)
    res_serial = make_mcts().search(g_serial, sims, eval_fn)

    def batch_eval_fn(games):
        return [eval_fn(g) for g in games]

    g_batched = zeropawn_cpp.ChessGame(fen)
    res_batched = make_mcts().search(g_batched, sims, eval_fn, batch_eval_fn)

    move_serial = g_serial.action_to_string(res_serial.best_action)
    move_batched = g_batched.action_to_string(res_batched.best_action)
    assert move_serial == move_batched, (
        f"serial picked {move_serial}, batched picked {move_batched} for {fen}"
    )
    assert abs(res_serial.root_value - res_batched.root_value) < 1e-4


# ---------------------------------------------------------------------------
# F-2: callback fast path returns NumPy float32
# ---------------------------------------------------------------------------


def test_f02_uci_eval_callback_returns_numpy_float32():
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine.__new__(UCIEngine)  # skip heavy __init__ (no model)
    engine.zeropawn_cpp = zeropawn_cpp
    engine.model = None
    engine.device = "cpu"
    engine._lock = threading.Lock()

    g = zeropawn_cpp.ChessGame()
    policy, value = engine._eval_callback(g)
    assert isinstance(policy, np.ndarray)
    assert policy.dtype == np.float32
    assert policy.flags["C_CONTIGUOUS"]
    assert policy.shape == (4672,)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert isinstance(value, float)

    batch = engine._eval_batch_callback([g])
    assert len(batch) == 1
    p2, v2 = batch[0]
    assert isinstance(p2, np.ndarray) and p2.dtype == np.float32
    assert p2.shape == (4672,)


# ---------------------------------------------------------------------------
# F-11/F-12: warm-start validation
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, records: list, designated_sha: str | None = None):
    reg = {
        "schema_version": 1,
        "designated_warm_start_sha256": designated_sha,
        "checkpoints": records,
    }
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "checkpoint_registry.json").write_text(json.dumps(reg), encoding="utf-8")


def test_f12_warm_start_rejects_missing_file(tmp_path):
    from python.model.checkpoint_registry import WarmStartValidationError, validate_warm_start

    _write_registry(tmp_path, [])
    with pytest.raises(WarmStartValidationError, match="not found"):
        validate_warm_start(tmp_path / "does_not_exist.pt", repo_root=tmp_path)


def test_f12_warm_start_rejects_unregistered_checkpoint(tmp_path, dummy_ckpt):
    from python.model.checkpoint_registry import WarmStartValidationError, validate_warm_start

    _write_registry(tmp_path, [])
    with pytest.raises(WarmStartValidationError, match="not in the checkpoint registry"):
        validate_warm_start(dummy_ckpt, repo_root=tmp_path)


def test_f12_warm_start_rejects_registered_but_not_designated(tmp_path, dummy_ckpt):
    from python.model.checkpoint_registry import (
        compute_sha256,
        validate_warm_start,
        WarmStartValidationError,
    )

    sha = compute_sha256(dummy_ckpt)
    _write_registry(
        tmp_path,
        [{"file": str(dummy_ckpt), "sha256": sha, "status": "verified", "role": "x"}],
    )
    with pytest.raises(WarmStartValidationError, match="NOT designated"):
        validate_warm_start(dummy_ckpt, repo_root=tmp_path)
    # Operator override is allowed but loud.
    info = validate_warm_start(dummy_ckpt, repo_root=tmp_path, allow_unverified=True)
    assert info["status"] == "verified"


def test_f12_warm_start_accepts_designated_verified(tmp_path, dummy_ckpt):
    from python.model.checkpoint_registry import compute_sha256, validate_warm_start

    sha = compute_sha256(dummy_ckpt)
    _write_registry(
        tmp_path,
        [
            {
                "file": str(dummy_ckpt),
                "sha256": sha,
                "status": "verified",
                "role": "baseline",
                "designated_warm_start": True,
            }
        ],
    )
    info = validate_warm_start(dummy_ckpt, repo_root=tmp_path)
    assert info["status"] == "verified"
    assert info["sha256"] == sha


def test_f12_warm_start_rejects_wrong_architecture(tmp_path):
    from python.model.checkpoint_registry import WarmStartValidationError, validate_warm_start

    p = tmp_path / "wrong_arch.pt"
    model = ChessNet(num_blocks=4, num_filters=32)
    torch.save({"model": model.state_dict(), "num_blocks": 4, "num_filters": 32, "step": 1}, p)
    _write_registry(tmp_path, [])
    with pytest.raises(WarmStartValidationError, match="architecture"):
        validate_warm_start(p, repo_root=tmp_path, allow_unverified=True)


# ---------------------------------------------------------------------------
# F-14: puzzle-decline band calibration
# ---------------------------------------------------------------------------


def test_f14_puzzle_decline_band_is_sampling_calibrated():
    from scripts.run_phase4_rl import puzzle_decline_trips

    # 1.5-point jitter (within certified-set noise of 1000 puzzles) must NOT trip.
    hist = [0.640, 0.625, 0.612, 0.598]
    assert not puzzle_decline_trips(hist, band=0.015)
    # Three consecutive drops beyond the band DO trip.
    hist_big = [0.640, 0.620, 0.600, 0.575]
    assert puzzle_decline_trips(hist_big, band=0.015)
    # Flat/jittered histories never trip; short histories never trip.
    assert not puzzle_decline_trips([0.640, 0.639, 0.638, 0.637], band=0.015)
    assert not puzzle_decline_trips([0.640, 0.600, 0.560], band=0.015)
    # Default band matches the documented calibration.
    import inspect

    default = inspect.signature(puzzle_decline_trips).parameters["band"].default
    assert abs(default - 0.015) < 1e-9


# ---------------------------------------------------------------------------
# F-15: per-game persistence + interrupted-match partial tally
# ---------------------------------------------------------------------------

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _bare_match_runner(openings, seed, num_workers=2, monkeypatch=None):
    """A MatchRunner with __init__ skipped; attributes set directly.

    MatchWorkerContext is stubbed so no Stockfish binary is needed — the games
    themselves are replaced below; only the runner loop is under test.
    """
    from python.eval import match_runner as mr

    class _DummyCtx:
        def close(self):
            pass

    if monkeypatch is not None:
        monkeypatch.setattr(mr, "MatchWorkerContext", lambda *a, **k: _DummyCtx())
    runner = mr.MatchRunner.__new__(mr.MatchRunner)
    runner.num_workers = num_workers
    runner.openings = openings
    runner.seed = seed
    runner.search_mode = "serial"
    runner.stockfish_path = "unused"
    runner.stockfish_nodes = 1000
    runner.stockfish_elo = None
    runner.mcts_sims = 8
    runner.c_puct = 2.0
    runner.reuse_tree = False
    runner.batch_size = 1
    return runner


def test_f15_on_game_result_persists_each_game(monkeypatch):
    persisted = []
    runner = _bare_match_runner([START_FEN] * 3, seed=42, monkeypatch=monkeypatch)

    results = ["1-0", "1/2-1/2", "0-1"]

    def instant_game(fen, engine_is_white, ctx, game_seed, move_log, info_log=None):
        return results[game_seed % 3]

    runner._play_game_worker = instant_game

    res = runner.run_match(
        num_openings=3,
        on_game_result=lambda g: persisted.append((g["opening_idx"], g["result"], g["seed"])),
    )
    assert res["total_games"] == 6
    assert res["interrupted"] is False
    # Every game reached the callback exactly once: 6 games, unique seeds.
    seeds = [s for _, _, s in persisted]
    assert len(persisted) == 6 and len(set(seeds)) == 6
    # Tally is consistent with the persisted results.
    wins = sum(1 for _, r, _ in persisted if r == "1-0")
    draws = sum(1 for _, r, _ in persisted if r == "1/2-1/2")
    losses = sum(1 for _, r, _ in persisted if r == "0-1")
    assert (res["engine_wins"], res["engine_draws"], res["engine_losses"]) == (wins, draws, losses)


def test_f15_interrupted_match_returns_partial_tally(monkeypatch):
    """A KeyboardInterrupt mid-match must keep completed games and mark interrupted."""
    runner = _bare_match_runner([START_FEN] * 4, seed=7, num_workers=1, monkeypatch=monkeypatch)

    persisted = []

    def slow_game(fen, engine_is_white, ctx, game_seed, move_log, info_log=None):
        time.sleep(0.05)
        return "1/2-1/2"

    runner._play_game_worker = slow_game

    import concurrent.futures as cf

    original_as_completed = cf.as_completed

    def interrupting_as_completed(fs):
        it = original_as_completed(fs)
        yield next(it)
        raise KeyboardInterrupt()

    monkeypatch.setattr(cf, "as_completed", interrupting_as_completed)

    res = runner.run_match(num_openings=4, on_game_result=lambda g: persisted.append(g))
    assert res["interrupted"] is True
    assert res["completed_games"] == 2  # first pair finished before the interrupt
    assert len(persisted) == 2
    assert res["engine_draws"] == 2


# ---------------------------------------------------------------------------
# F-3 (Python side): UCI clock allocation, deadline, and stop
# ---------------------------------------------------------------------------


def _uci_engine_no_model():
    """UCIEngine without loading any checkpoint (model None → uniform eval)."""
    from python.uci.uci_engine import UCIEngine

    engine = UCIEngine.__new__(UCIEngine)
    # Minimal state the go/stop paths touch.
    import zeropawn_cpp as zc

    engine.zeropawn_cpp = zc
    engine.model = None
    engine.device = "cpu"
    engine._lock = threading.Lock()
    engine._stop_event = threading.Event()
    engine._search_thread = None
    engine._sims_per_sec = 200.0
    engine.default_sims = 400
    engine.search_mode = "serial"
    from python.eval.match_runner import require_serial_search

    engine._require_serial_search = require_serial_search
    engine.mcts_config = zc.MCTSConfig()
    engine.mcts = zc.MCTS(engine.mcts_config)
    engine.game = zc.ChessGame()
    engine.use_book = False
    engine.opening_book = None
    engine.last_engine_action = None
    engine.last_engine_move = None
    engine.prev_moves_at_search = None
    engine.prev_moves = []
    engine.prev_fen = "startpos"
    return engine


def test_f03_allocate_time_movetime_wtime_nodes_infinite():
    engine = _uci_engine_no_model()
    # movetime 1000 ms at 200 sims/s, 5% margin → ~190 sims, ~950 ms budget.
    now = time.perf_counter()
    sims, deadline, infinite = engine.allocate_time({"movetime": "1000"}, side_is_white=True)
    assert not infinite
    assert 150 <= sims <= 210
    assert deadline is not None and 0.85 <= deadline - now <= 1.0  # absolute monotonic seconds

    # wtime/btime: 30000 ms clock + 0 inc → alloc = 1000 ms, clamped.
    sims, deadline, infinite = engine.allocate_time(
        {"wtime": "30000", "btime": "30000", "winc": "0", "binc": "0"},
        side_is_white=True,
    )
    assert not infinite
    assert 100 <= sims <= 250
    assert deadline is not None

    # Low clock: 500 ms remaining → alloc floor 100 ms.
    now = time.perf_counter()
    sims, deadline, _ = engine.allocate_time(
        {"wtime": "500", "btime": "500", "winc": "0", "binc": "0"}, side_is_white=True
    )
    assert deadline is not None and deadline - now <= 0.2

    # nodes: exact budget, no deadline.
    sims, deadline, infinite = engine.allocate_time({"nodes": "123"}, side_is_white=True)
    assert sims == 123 and deadline is None and not infinite

    # infinite: huge budget, no deadline.
    sims, deadline, infinite = engine.allocate_time({"infinite": True}, side_is_white=False)
    assert infinite and deadline is None and sims > 1_000_000


def test_f03_search_aborts_at_deadline_and_answers_legal_move():
    engine = _uci_engine_no_model()

    # Slow evaluator: 15 ms per leaf — a full 2000-sim search would take ~30 s.
    def slow_eval(game):
        time.sleep(0.015)
        n = game.num_actions()
        return (np.full(n, 1.0 / n, dtype=np.float32), 0.0)

    engine._eval_callback = slow_eval
    engine._eval_batch_callback = lambda games: [slow_eval(g) for g in games]

    start = time.perf_counter()
    # deadline 0.5 s from now (monotonic seconds), generous sim budget.
    best = engine.search_and_respond(2000, deadline_ms=start + 0.5)
    elapsed = time.perf_counter() - start
    assert best in engine.game.legal_moves_uci()
    # Deadline (0.5 s) + selection/output margin; without the abort this would
    # take ~30 s.
    assert elapsed < 5.0, f"deadline not enforced (took {elapsed:.2f}s)"


def test_f03_stop_event_aborts_running_search():
    engine = _uci_engine_no_model()

    started = threading.Event()

    def slow_eval(game):
        started.set()
        time.sleep(0.01)
        n = game.num_actions()
        return (np.full(n, 1.0 / n, dtype=np.float32), 0.0)

    engine._eval_callback = slow_eval
    engine._eval_batch_callback = lambda games: [slow_eval(g) for g in games]

    engine._stop_event = threading.Event()
    thread = threading.Thread(
        target=engine.search_and_respond, args=(500_000, None), daemon=True
    )
    start = time.perf_counter()
    thread.start()
    assert started.wait(timeout=10)
    time.sleep(0.05)
    engine._stop_event.set()  # the "stop" command path
    thread.join(timeout=15)
    elapsed = time.perf_counter() - start
    assert not thread.is_alive(), "search ignored stop"
    assert elapsed < 10.0, f"stop not honored promptly (took {elapsed:.2f}s)"
    assert engine.last_engine_move is not None


def test_f03_rate_estimate_updates_after_search():
    engine = _uci_engine_no_model()
    engine._eval_callback = lambda g: (
        np.full(g.num_actions(), 1.0 / g.num_actions(), dtype=np.float32),
        0.0,
    )
    engine._eval_batch_callback = None
    before = engine._sims_per_sec
    engine.search_and_respond(64)
    after = engine._sims_per_sec
    assert after > 0
    # Uniform-policy evals are extremely fast; measured rate should exceed the
    # conservative 200/s prior.
    assert after > before or after >= 200.0
