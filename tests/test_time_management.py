"""Tests for mode-aware UCI time management (audit H-3).

The old ``go`` handler translated wall-clock budgets with a hardcoded
400 sims/s; measured throughput differs by 2-6x across device/mode/batch-size.
The new ``SearchTimeManager`` estimates from per-configuration EWMA of
observed search throughput with a conservative cold-start default, and the
UCI ``go`` path allocates with explicit margins.
"""

from typing import Optional

import numpy as np
import pytest

from python.uci.time_management import (
    SearchTimeManager,
    ThroughputKey,
    allocate_clock_ms,
)


def key(device="cuda", mode="serial", batch=1):
    return ThroughputKey.from_engine(device, mode, batch)


# ==============================================================================
# SearchTimeManager unit behavior
# ==============================================================================


def test_cold_start_uses_conservative_default():
    tm = SearchTimeManager()
    assert tm.estimate_rate(key("cuda", "serial", 1)) == 100.0
    assert tm.estimate_rate(key("cuda", "batched", 16)) == 800.0
    assert tm.estimate_rate(key("cpu", "serial", 1)) == 40.0
    # Unknown configuration falls back to the safest default.
    assert tm.estimate_rate(key("tpu", "serial", 1)) == 50.0


def test_movetime_budget_uses_hard_deadline_margin():
    tm = SearchTimeManager()
    # Cold cuda/serial estimate 100 sims/s; movetime margin 0.9.
    sims = tm.sims_for_budget(key("cuda", "serial", 1), 1000.0, "movetime")
    assert sims == int(100.0 * 0.9)
    # Zero budget -> zero sims.
    assert tm.sims_for_budget(key("cuda", "serial", 1), 0.0, "movetime") == 0


def test_clock_budget_uses_soft_margin():
    tm = SearchTimeManager()
    sims = tm.sims_for_budget(key("cpu", "serial", 1), 1000.0, "clock")
    assert sims == int(40.0 * 0.8)


def test_observed_throughput_replaces_default():
    tm = SearchTimeManager()
    k = key("cuda", "serial", 1)
    tm.record(k, 400, 2.0)  # 200 sims/s observed
    assert tm.estimate_rate(k) == 200.0
    assert tm.sims_for_budget(k, 1000.0, "movetime") == int(200.0 * 0.9)


def test_ewma_smooths_changing_throughput():
    tm = SearchTimeManager()
    k = key("cuda", "batched", 16)
    for _ in range(20):
        tm.record(k, 1600, 1.0)  # stable 1600 sims/s
    assert abs(tm.estimate_rate(k) - 1600.0) < 1e-6
    # Hardware slows to 800 sims/s: the EWMA moves partway, not instantly.
    tm.record(k, 800, 1.0)
    est = tm.estimate_rate(k)
    assert 800.0 < est < 1600.0
    # After many slow samples the estimate converges to the new rate.
    for _ in range(50):
        tm.record(k, 800, 1.0)
    assert tm.estimate_rate(k) == pytest.approx(800.0, rel=1e-4)


def test_configurations_are_keyed_separately():
    """A stale measurement from another mode/device/batch-size is never reused:
    the conservative default applies until the new configuration is measured."""
    tm = SearchTimeManager()
    tm.record(key("cuda", "batched", 16), 3200, 1.0)
    assert tm.estimate_rate(key("cuda", "batched", 16)) == 3200.0
    assert tm.estimate_rate(key("cuda", "serial", 1)) == 100.0
    assert tm.estimate_rate(key("cpu", "serial", 1)) == 40.0
    assert tm.estimate_rate(key("cuda", "batched", 32)) == 800.0


def test_implausible_observations_ignored():
    """Cache-served (no inference) or sub-millisecond searches must not inflate
    the estimate into an overcommitted clock."""
    tm = SearchTimeManager()
    k = key("cuda", "serial", 1)
    tm.record(k, 0, 1.0)
    tm.record(k, 400, 0.0)
    tm.record(k, -5, 1.0)
    tm.record(k, 400, 0.0002)  # sub-ms noise
    tm.record(k, 100, 0.002)  # 50k sims/s: above the plausibility ceiling
    assert tm.sample_count(k) == 0
    assert tm.estimate_rate(k) == 100.0


def test_short_clock_gets_small_budget():
    """A nearly exhausted clock must not overcommit: 500ms left, no increment,
    no samples -> 100ms allocated, ~8 sims at the cold cuda/serial estimate."""
    allocated = allocate_clock_ms(500.0, 0.0)
    assert allocated == 100.0
    tm = SearchTimeManager()
    sims = tm.sims_for_budget(key("cuda", "serial", 1), allocated, "clock")
    assert sims <= 10


# ==============================================================================
# allocate_clock_ms shape (established allocation policy preserved)
# ==============================================================================


def test_allocate_clock_shapes():
    # 30s clock, no increment -> 1s (established total/30 shape)
    assert allocate_clock_ms(30_000.0, 0.0) == pytest.approx(1_000.0)
    # Increment extends the budget linearly
    assert allocate_clock_ms(30_000.0, 200.0) == pytest.approx(1_200.0)
    # Floor: never move instantly on a nearly empty clock
    assert allocate_clock_ms(300.0, 0.0) == 100.0
    # Ceiling: never sink the whole clock into one move
    assert allocate_clock_ms(600_000.0, 30_000.0) == 10_000.0
    # movestogo: spend 1/N of the remaining time
    assert allocate_clock_ms(60_000.0, 0.0, moves_to_go=10) == pytest.approx(6_000.0)
    assert allocate_clock_ms(60_000.0, 0.0, moves_to_go=100) == pytest.approx(
        60_000.0 / 40.0
    )  # clamped to 40
    assert allocate_clock_ms(0.0, 0.0) == 0.0


# ==============================================================================
# UCI go-handler integration (search mocked; no model needed)
# ==============================================================================


class _SearchProbe:
    """Replaces UCIEngine.search_and_respond, capturing the simulation budget.

    Emulates the real method's observation contract: ``_last_search_ran`` is
    True only when a search plausibly executed, and the elapsed time is
    realistic (sub-millisecond timings would be rejected by the estimator's
    noise guard).
    """

    def __init__(self, engine, elapsed: float = 0.05):
        self.engine = engine
        self.elapsed = elapsed
        self.budgets = []

    def __call__(self, sims: int, deadline_ms: Optional[float] = None) -> str:
        import time as _time

        self.budgets.append(sims)
        _time.sleep(self.elapsed)
        self.engine._last_search_ran = True
        return "e2e4"


@pytest.fixture()
def engine():
    from python.uci.uci_engine import UCIEngine

    eng = UCIEngine(checkpoint_path="", use_book=False, device="cpu")
    eng.handle_line("uci")
    eng.handle_line("position startpos")
    return eng


def test_go_movetime_respects_budget(engine):
    probe = _SearchProbe(engine, elapsed=0.0)
    engine.search_and_respond = probe
    engine.handle_line("go movetime 1000")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    # cpu/serial cold estimate 40 sims/s, margin 0.9, 1s budget -> 36 sims
    assert probe.budgets == [36]


def test_go_wtime_btime_uses_clock_policy(engine):
    probe = _SearchProbe(engine, elapsed=0.0)
    engine.search_and_respond = probe
    engine.handle_line("go wtime 30000 btime 30000 winc 0 binc 0")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    # 30s clock -> 1000ms allocated -> int(40 * 0.8) = 32 sims
    assert probe.budgets == [32]
    engine.handle_line("go wtime 30000 btime 30000 winc 200 binc 200")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    # +200ms increment -> 1200ms -> int(40 * 0.8 * 1.2) = 38
    assert probe.budgets[-1] == 38


def test_go_movestogo(engine):
    probe = _SearchProbe(engine, elapsed=0.0)
    engine.search_and_respond = probe
    engine.handle_line("go movestogo 10 wtime 30000 btime 30000 winc 0 binc 0")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    # 30s / 10 moves = 3000ms -> int(40 * 0.8 * 3.0) = 96
    assert probe.budgets == [96]


def test_go_nodes_passthrough(engine):
    probe = _SearchProbe(engine, elapsed=0.0)
    engine.search_and_respond = probe
    engine.handle_line("go nodes 777")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    assert probe.budgets == [777]


def test_throughput_observation_updates_next_budget(engine):
    """After one real search at a measured rate, the next budget uses the
    observation, not the conservative default."""
    engine.search_and_respond = _SearchProbe(engine, elapsed=1.0)  # 40 sims in 1.0s
    engine.handle_line("go nodes 40")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    k = ThroughputKey.from_engine(engine.device, engine.search_mode, engine.batch_size)
    assert engine.time_manager.observed_rate(k) == pytest.approx(40.0, rel=0.05)

    engine.search_and_respond = _SearchProbe(engine, elapsed=0.0)
    engine.handle_line("go wtime 30000 btime 30000 winc 0 binc 0")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    # 1000ms at the observed 40 sims/s with 0.8 margin -> 32 sims (same value
    # here by construction of the default; assert the observation is in use
    # via the estimator, and that a faster observation raises the budget).
    assert engine.time_manager.estimate_rate(k) == pytest.approx(40.0, rel=0.05)
    engine.time_manager.record(k, 400, 1.0)  # now measured 400 sims/s...
    for _ in range(30):  # ...converged by repeated observation (EWMA)
        engine.time_manager.record(k, 400, 1.0)
    engine.handle_line("go wtime 30000 btime 30000 winc 0 binc 0")
    if engine._search_thread is not None:
        engine._search_thread.join(timeout=5.0)
    est = engine.time_manager.estimate_rate(k)
    assert est == pytest.approx(400.0, rel=1e-3)
    assert engine.search_and_respond.budgets[-1] == int(est * 0.8 * 1.0)


def test_book_and_game_over_shortcuts_do_not_poison_estimator():
    """Game-over and opening-book short-circuits run no MCTS search; their
    near-zero elapsed time must not inflate the throughput estimate. This uses
    the REAL search_and_respond (model-less uniform-eval mode)."""
    from python.uci.uci_engine import UCIEngine

    # book_path points at a nonexistent file so no real book loads; the stub
    # below is injected for the book-short-circuit case.
    engine = UCIEngine(
        checkpoint_path="", use_book=True, book_path="nonexistent-book-for-test.json", device="cpu"
    )
    engine.handle_line("uci")
    k = ThroughputKey.from_engine(engine.device, engine.search_mode, engine.batch_size)

    # Game over: checkmated position -> bestmove 0000 without searching.
    engine.handle_line(
        "position fen r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"
    )
    assert engine.handle_line("go movetime 1000") is True
    assert engine.time_manager.observed_rate(k) is None
    assert engine._last_search_ran is False

    # Opening-book short-circuit.
    class _StubBook:
        enabled = True

        def lookup(self, game, **kwargs):
            return ("e2e4", {})

    engine.handle_line("position startpos")
    engine.opening_book = _StubBook()
    assert engine.handle_line("go movetime 1000") is True
    assert engine.time_manager.observed_rate(k) is None
    assert engine._last_search_ran is False
