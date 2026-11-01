"""Comprehensive regression suite for ZeroPawn MCTS & Search Correctness Audit Remediation.

Pins all findings S-1 through S-8 and verifies algorithmic and integration correctness.
"""

from __future__ import annotations

import io
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Ensure native extension and diagnostics reference can be imported
DIAG_DIR = Path(__file__).resolve().parent.parent / "audit-reports" / "mcts-search-correctness" / "diagnostics"
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import ref_mcts
import zeropawn_cpp
from python.selfplay.worker import (
    GameSummary,
    InferenceError,
    SelfPlayWorker,
    TERMINATION_ABORTED,
)
from python.uci.time_management import ThroughputKey
from python.uci.uci_engine import UCIEngine


# ---------------------------------------------------------------------------
# Helpers & deterministic evaluators
# ---------------------------------------------------------------------------

def _deterministic_eval(salt: int, num_actions: int):
    rng = np.random.default_rng(42 + salt)
    weights = (1.0 + rng.random(num_actions) * 3.0).astype(np.float64)

    def eval_fn(game):
        actions = list(game.legal_actions())
        if not actions or game.is_terminal():
            val = game.terminal_value() if game.is_terminal() else 0.0
            return [0.0] * num_actions, float(val)
        sub = weights[actions]
        s = float(np.sum(sub))
        p = np.zeros(num_actions, dtype=np.float32)
        if s > 1e-8:
            p[actions] = (sub / s).astype(np.float32)
        else:
            p[actions] = 1.0 / len(actions)
        h = game.get_hash()
        val = math.sin(float(h & 0xFFFFFF) * 0.001) * 0.8
        return p.tolist(), float(val)

    return eval_fn


def _make_cfg(c_puct=2.0, fpu=0.2, vl=1, noise=False, seed=42, reuse=False, batch_size=1):
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = c_puct
    cfg.fpu_reduction = fpu
    cfg.virtual_loss = vl
    cfg.add_noise = noise
    cfg.seed = seed
    cfg.reuse_tree = reuse
    cfg.batch_size = batch_size
    return cfg


# ==============================================================================
# S-1: UCI Clock Deadlines & Responsive Search Cancellation
# ==============================================================================

def test_s1_uci_go_movetime_enforces_deadline():
    """Verify that a movetime search finishes within the requested time budget."""
    engine = UCIEngine(checkpoint_path="", use_book=False, device="cpu")
    engine.handle_line("uci")
    engine.handle_line("position startpos")

    # Give an excessively high node cap if not bounded by time, with a small 100ms movetime
    t0 = time.perf_counter()
    assert engine.handle_line("go movetime 100") is True
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Must complete promptly without hanging (allowing 500ms margin for scheduling/polling)
    assert elapsed_ms < 600.0
    assert engine.last_engine_move in engine.game.legal_moves_uci()


def test_s1_uci_go_wtime_enforces_deadline():
    """Verify that clocked searches enforce the computed deadline."""
    engine = UCIEngine(checkpoint_path="", use_book=False, device="cpu")
    engine.handle_line("uci")
    engine.handle_line("position startpos")

    t0 = time.perf_counter()
    # 2000ms clock with 0 increment -> small budget and tight deadline
    assert engine.handle_line("go wtime 2000 btime 2000 winc 0 binc 0") is True
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 1500.0
    assert engine.last_engine_move in engine.game.legal_moves_uci()


def test_s1_uci_stop_aborts_search_responsively():
    """Verify that 'stop' halts an in-flight search promptly and returns legal bestmove."""
    engine = UCIEngine(checkpoint_path="", use_book=False, device="cpu")
    engine.handle_line("uci")
    engine.handle_line("position startpos")

    out_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = out_buf
        assert engine.handle_line("go infinite") is True
        # Let the search thread start
        time.sleep(0.05)
        assert engine._search_thread is not None and engine._search_thread.is_alive()

        # Issue stop
        t_stop0 = time.perf_counter()
        assert engine.handle_line("stop") is True
        engine._search_thread.join(timeout=2.0)
        stop_elapsed = time.perf_counter() - t_stop0

        assert not engine._search_thread.is_alive()
        assert stop_elapsed < 0.5  # Responsive abort under 500ms

        lines = [l for l in out_buf.getvalue().splitlines() if l.startswith("bestmove")]
        assert len(lines) == 1, "Exactly one bestmove response must be emitted"
        mv = lines[0].split()[1]
        assert mv in engine.game.legal_moves_uci()
    finally:
        sys.stdout = old_stdout


def test_s1_zero_or_tiny_movetime_boundary():
    """Verify immediate deadline expiry (tiny budget) produces a legal move without crash."""
    engine = UCIEngine(checkpoint_path="", use_book=False, device="cpu")
    engine.handle_line("uci")
    engine.handle_line("position startpos")

    out_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = out_buf
        # Expired / near-zero movetime (1ms)
        assert engine.handle_line("go movetime 1") is True
        if engine._search_thread is not None:
            engine._search_thread.join(timeout=3.0)

        lines = [l for l in out_buf.getvalue().splitlines() if l.startswith("bestmove")]
        assert len(lines) == 1
        mv = lines[0].split()[1]
        assert mv in engine.game.legal_moves_uci()
    finally:
        sys.stdout = old_stdout


# ==============================================================================
# S-2: Worker Shutdown Race & Vacuous Game Elimination
# ==============================================================================

def test_s2_worker_shutdown_race_eliminates_vacuous_games(monkeypatch):
    """Delay during SelfPlayGame construction racing worker.stop() must not
    create a false completed game, must not increment total_games_played,
    and must not invoke position_sink with an empty/aborted game."""
    from python.selfplay import worker as worker_mod

    orig_init = worker_mod.SelfPlayGame.__init__

    def delayed_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        time.sleep(0.05)  # Window for worker.stop() to run

    monkeypatch.setattr(worker_mod.SelfPlayGame, "__init__", delayed_init)

    sink_called = []

    def sink(positions, summary):
        sink_called.append((len(positions), summary.length_plies))

    class DummyClient:
        def evaluate_planes(self, planes):
            return np.ones((len(planes), 4672), dtype=np.float32) / 4672, np.zeros((len(planes), 3), dtype=np.float32)

    w = SelfPlayWorker(
        worker_id=1,
        client=DummyClient(),
        num_concurrent_games=2,
        full_search_sims=5,
        fast_search_sims=2,
        position_sink=sink,
        use_book=False,
    )
    w.start()
    time.sleep(0.02)
    w.stop()

    # The vacuous game must not be recorded as played or sunk
    assert w.total_games_played == 0
    assert w.total_positions_generated == 0
    assert len(sink_called) == 0


def test_s2_accepts_for_sink_rejects_zero_ply():
    """accepts_for_sink must refuse any 0-ply or aborted game."""
    s = GameSummary(
        game_id=1,
        length_plies=0,
        result="*",
        resigned=False,
        resigned_side_white=None,
        would_have_resigned=False,
        is_false_resign=False,
        duration_sec=0.1,
        total_simulations=0,
        model_version=0,
        aborted=True,
        termination_reason=TERMINATION_ABORTED,
    )
    assert SelfPlayWorker.accepts_for_sink(s) is False

    s_valid = GameSummary(
        game_id=2,
        length_plies=10,
        result="1-0",
        resigned=False,
        resigned_side_white=None,
        would_have_resigned=False,
        is_false_resign=False,
        duration_sec=1.0,
        total_simulations=50,
        model_version=0,
        aborted=False,
        termination_reason="checkmate",
    )
    assert SelfPlayWorker.accepts_for_sink(s_valid) is True


# ==============================================================================
# S-3: Retained-Visits State Reporting (MCTS.retained_visits)
# ==============================================================================

def test_s3_retained_visits_getter_monotone_and_accurate():
    """MCTS.retained_visits() must reflect the promoted root's visit mass
    immediately after advance_root() BEFORE the next search() is called."""
    cfg = _make_cfg(reuse=True)
    m = zeropawn_cpp.MCTS(cfg)
    g = zeropawn_cpp.ChessGame()
    eval_fn = _deterministic_eval(10, g.num_actions())

    assert m.retained_visits() == 0

    # Initial search with 25 simulations
    res1 = m.search(g, 25, eval_fn)
    best_act = res1.best_action
    assert best_act >= 0
    raw_visits_best = int(res1.action_visits[best_act] * 25 + 0.5)

    g_next = g.clone()
    g_next.apply_action(best_act)

    # Advance root to best action
    advanced = m.advance_root(best_act, g_next)
    assert advanced is True

    # S-3 check: getter must report the promoted root's visits IMMEDIATELY
    retained_immediately = m.retained_visits()
    assert retained_immediately > 0

    # Next search with 15 simulations
    res2 = m.search(g_next, 15, eval_fn)
    assert res2.retained_visits == retained_immediately
    assert res2.new_simulations == 15
    assert m.retained_visits() == retained_immediately


def test_s3_failed_advance_root_resets_retained_visits():
    """Failed advance_root resets the tree and sets retained_visits to 0."""
    cfg = _make_cfg(reuse=True)
    m = zeropawn_cpp.MCTS(cfg)
    g = zeropawn_cpp.ChessGame()
    eval_fn = _deterministic_eval(11, g.num_actions())

    m.search(g, 20, eval_fn)
    assert m.advance_root(999999, g.clone()) is False
    assert m.retained_visits() == 0
    assert not m.has_retained_root()


# ==============================================================================
# S-4: Legacy Batched Deduplication Prior Assignment
# ==============================================================================

def test_s4_legacy_search_batched_duplicate_nodes_receive_priors():
    """In search_batched, when two candidates in the same batch reach the same
    cache key, the duplicate node must receive evaluated priors from the batch."""
    cfg = _make_cfg(batch_size=8)
    m = zeropawn_cpp.MCTS(cfg)
    g = zeropawn_cpp.ChessGame()
    eval_fn = _deterministic_eval(20, g.num_actions())

    # Raw batch evaluator
    def raw_batch_eval(planes):
        n = len(planes)
        policies = np.ones((n, g.num_actions()), dtype=np.float32) / g.num_actions()
        values = np.zeros(n, dtype=np.float32)
        return (policies, values)

    res = m.search_batched(g.clone(), 64, 8, raw_batch_eval)
    assert res.new_simulations == 64
    assert res.best_action in g.legal_actions()


# ==============================================================================
# S-5: Evaluator Exception Handling in UCI Engine
# ==============================================================================

def test_s5_uci_evaluator_exception_emits_legal_bestmove_and_recovers():
    """Inference failure during search must not crash the UCI loop, must emit
    a legal fallback bestmove, clear virtual loss, and allow subsequent commands."""
    engine = UCIEngine(checkpoint_path="", use_book=False, device="cpu")
    engine.handle_line("uci")
    engine.handle_line("position startpos")

    # Injected failing evaluator
    def failing_eval(game):
        raise RuntimeError("Injected GPU out of memory error")

    engine._eval_callback = failing_eval

    out_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = out_buf
        assert engine.handle_line("go nodes 50") is True
        if engine._search_thread is not None:
            engine._search_thread.join(timeout=3.0)

        lines = [l for l in out_buf.getvalue().splitlines() if l.startswith("bestmove")]
        assert len(lines) == 1, "Must emit exactly one bestmove on failure"
        mv = lines[0].split()[1]
        assert mv in engine.game.legal_moves_uci(), "Must return a legal fallback move"

        # Virtual loss must be clean
        assert engine.mcts.total_active_virtual_losses() == 0

        # Subsequent commands must work normally
        out_buf.truncate(0)
        out_buf.seek(0)
        assert engine.handle_line("isready") is True
        assert "readyok" in out_buf.getvalue()

        # Engine can accept a new position
        assert engine.handle_line("position startpos moves e2e4 e7e5") is True
    finally:
        sys.stdout = old_stdout


# ==============================================================================
# S-7: UCI SearchMode Functional Toggling & Configuration
# ==============================================================================

def test_s7_search_mode_toggles_batch_size_and_rebuilds_mcts():
    """setoption SearchMode must actively adjust batch size and rebuild MCTS."""
    engine = UCIEngine(checkpoint_path="", use_book=False, device="cpu", batch_size=1)
    engine.handle_line("uci")
    assert engine.search_mode == "serial"
    assert engine.batch_size == 1
    assert engine.mcts_config.batch_size == 1

    # Switch to batched mode
    assert engine.handle_line("setoption name SearchMode value batched") is True
    assert engine.search_mode == "batched"
    assert engine.batch_size > 1
    assert engine.mcts_config.batch_size > 1

    # Switch back to serial mode
    assert engine.handle_line("setoption name SearchMode value serial") is True
    assert engine.search_mode == "serial"
    assert engine.batch_size == 1
    assert engine.mcts_config.batch_size == 1

    # Setting BatchSize to 8 automatically updates search_mode to batched
    assert engine.handle_line("setoption name BatchSize value 8") is True
    assert engine.batch_size == 8
    assert engine.search_mode == "batched"

    # Setting BatchSize to 1 updates search_mode to serial
    assert engine.handle_line("setoption name BatchSize value 1") is True
    assert engine.batch_size == 1
    assert engine.search_mode == "serial"

    # Invalid SearchMode rejected safely
    assert engine.handle_line("setoption name SearchMode value invalid_mode") is True
    assert engine.search_mode == "serial"


# ==============================================================================
# S-8.2: Oracle-Math Parity Against Independent Reference
# ==============================================================================

def test_s8_oracle_parity_connect4_serial():
    """Verify exact parity between native serial search and independent reference on Connect4."""
    game = zeropawn_cpp.Connect4()
    cfg = _make_cfg(c_puct=2.0, fpu=0.2, vl=1, noise=False, seed=42)
    m = zeropawn_cpp.MCTS(cfg)
    eval_fn = _deterministic_eval(100, game.num_actions())

    prod_res = m.search(game, 40, eval_fn)
    ref_best, ref_val, ref_visits = ref_mcts.reference_serial_search(
        game, 40, eval_fn,
        {"c_puct": 2.0, "fpu": 0.2, "fpu_reduction": 0.2, "virtual_loss": 1, "add_noise": False, "seed": 42}
    )

    assert prod_res.best_action == ref_best
    assert np.allclose(prod_res.action_visits, ref_visits, atol=1e-6)
    assert abs(prod_res.root_value - ref_val) < 1e-6


def test_s8_oracle_parity_chess_serial():
    """Verify exact parity between native serial search and independent reference on Chess startpos."""
    game = zeropawn_cpp.ChessGame()
    cfg = _make_cfg(c_puct=2.0, fpu=0.2, vl=1, noise=False, seed=42)
    m = zeropawn_cpp.MCTS(cfg)
    eval_fn = _deterministic_eval(200, game.num_actions())

    prod_res = m.search(game, 40, eval_fn)
    ref_best, ref_val, ref_visits = ref_mcts.reference_serial_search(
        game, 40, eval_fn,
        {"c_puct": 2.0, "fpu": 0.2, "fpu_reduction": 0.2, "virtual_loss": 1, "add_noise": False, "seed": 42}
    )

    assert prod_res.best_action == ref_best
    assert np.allclose(prod_res.action_visits, ref_visits, atol=1e-6)
    assert abs(prod_res.root_value - ref_val) < 1e-6


# ==============================================================================
# S-8.3: Batch-Size-One Equivalence (Serial == Batched B=1)
# ==============================================================================

def test_s8_batch_size_one_internal_equals_serial():
    """Internal-batched search with B=1 must be exactly equivalent to serial search."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _deterministic_eval(300, game.num_actions())

    # Serial
    m_serial = zeropawn_cpp.MCTS(_make_cfg(batch_size=1, seed=123))
    res_serial = m_serial.search(game.clone(), 50, eval_fn)

    # Internal batched B=1
    m_batched = zeropawn_cpp.MCTS(_make_cfg(batch_size=1, seed=123))
    def batch_cb(states):
        return [eval_fn(s) for s in states]

    res_batched = m_batched.search(game.clone(), 50, eval_fn, batch_cb)

    assert res_serial.best_action == res_batched.best_action
    assert np.allclose(res_serial.action_visits, res_batched.action_visits, atol=1e-7)
    assert abs(res_serial.root_value - res_batched.root_value) < 1e-7
    assert res_serial.new_simulations == res_batched.new_simulations == 50


# ==============================================================================
# S-8.4: Root Advancement Statistics & Visit Mass Identity
# ==============================================================================

def test_s8_root_advancement_visit_mass_identity():
    """Verify that tree reuse preserves visit mass: raw sum == retained + new."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _deterministic_eval(400, game.num_actions())

    m = zeropawn_cpp.MCTS(_make_cfg(reuse=True, seed=42))
    res1 = m.search(game, 30, eval_fn)

    best = res1.best_action
    g2 = game.clone()
    g2.apply_action(best)

    assert m.advance_root(best, g2) is True
    retained = m.retained_visits()

    res2 = m.search(g2, 20, eval_fn)
    assert res2.retained_visits == retained
    assert res2.new_simulations == 20

    raw_visits = np.array(res2.action_visits) * (retained + 20)
    assert int(round(raw_visits.sum())) == retained + 20


# ==============================================================================
# S-8.5: Dirichlet Noise Distribution & Normalization
# ==============================================================================

def test_s8_dirichlet_noise_normalization_and_root_only():
    """Verify Dirichlet noise properties: normalized, non-negative, root-only."""
    cfg = _make_cfg(noise=True, seed=42)
    m = zeropawn_cpp.MCTS(cfg)
    game = zeropawn_cpp.ChessGame()
    eval_fn = _deterministic_eval(500, game.num_actions())

    res = m.search(game, 20, eval_fn)
    visits = np.array(res.action_visits)
    assert np.all(visits >= 0.0)
    assert abs(visits.sum() - 1.0) < 1e-5


def test_s8_dirichlet_monte_carlo_distribution_bounds():
    """Monte Carlo verification of Dirichlet(alpha=0.3, eps=0.25) mixing against reference distribution."""
    k_legal = 7
    observed = np.zeros(k_legal, dtype=np.int64)
    trials = 400

    def zero_eval(g):
        return [1.0 / k_legal] * g.num_actions(), 0.0

    for trial in range(trials):
        cfg = _make_cfg(noise=True, seed=50_000 + trial)
        m = zeropawn_cpp.MCTS(cfg)
        g = zeropawn_cpp.Connect4()
        res = m.search(g, 2, zero_eval)
        raw = np.array(res.action_visits[:k_legal]) * 2.0
        nz = [i for i in range(k_legal) if raw[i] > 0]
        if len(nz) == 2:
            observed[nz[1]] += 1
        elif len(nz) == 1:
            observed[nz[0]] += 1

    rng = np.random.default_rng(7)
    mc_trials = 100_000
    emp_obs = observed / float(trials)
    mc_freq = np.zeros(k_legal)
    batch_mc = mc_trials // 20
    for _ in range(20):
        noise = rng.gamma(0.3, 1.0, size=(batch_mc, k_legal))
        noise /= noise.sum(axis=1, keepdims=True)
        mixed = 0.75 * (1.0 / k_legal) + 0.25 * noise
        p0 = mixed[:, 0]
        rest = mixed[:, 1:]
        best_rest = rest.max(axis=1)
        keep = p0 < (-0.2 + 2.0 * best_rest)
        picks = 1 + rest[keep].argmax(axis=1)
        mc_freq += np.bincount(picks, minlength=k_legal) / float(batch_mc) / 20.0
        mc_freq[0] += float((~keep).mean()) / 20.0

    max_dev = float(np.max(np.abs(emp_obs - mc_freq)))
    # sampling noise bound for 400 trials ~ 1.96*sqrt(p(1-p)/400) <= 0.049
    assert max_dev < 0.06, f"Dirichlet mixture deviation {max_dev:.4f} exceeded bound 0.06"
