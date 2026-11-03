"""S5-02 regression: the MCTS serial-search abort poll is wall-clock bounded.

Before the 2026-11-03 remediation the serial path polled the abort predicate
only every 32 fresh simulations. With a slow evaluator (contended CPU host,
~40 ms per leaf) that let `go movetime 500` overshoot by tens of seconds and
gave `stop` latency of up to 21 s. The poll is now wall-clock based (>=1 ms
interval in addition to the 32-sim quantum), so an abort request is honored
within milliseconds regardless of the evaluation rate.

The test drives the REAL pybind search with a deliberately slow evaluator and
a wall-clock abort deadline, then asserts the search stopped within a bound
that only the wall-clock poll can satisfy (the 32-sim quantum would need
32 full evaluations).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

zeropawn_cpp = pytest.importorskip("zeropawn_cpp")


def _slow_uniform_evaluator(delay_sec: float):
    """Uniform-policy evaluator that burns `delay_sec` per call (CPU sleep)."""
    def evaluate(game):
        time.sleep(delay_sec)
        policy = np.full(4672, 1.0 / 4672, dtype=np.float32)
        return policy, 0.0

    return evaluate


def _deadline_abort(deadline: float):
    def abort() -> bool:
        return time.perf_counter() >= deadline

    return abort


@pytest.mark.parametrize("sim_rate_delay", [0.04])  # 40 ms/leaf = contended-CPU class
def test_abort_is_honored_within_milliseconds_at_slow_sim_rates(sim_rate_delay):
    game = zeropawn_cpp.ChessGame()
    config = zeropawn_cpp.MCTSConfig()
    config.batch_size = 1  # serial path — the path the audit measured
    mcts = zeropawn_cpp.MCTS(config)

    start = time.perf_counter()
    abort_after = 0.2  # seconds of wall clock before aborting
    result = mcts.search(
        game,
        1000,  # far more sims than can complete in the window
        _slow_uniform_evaluator(sim_rate_delay),
        None,
        _deadline_abort(start + abort_after),
    )
    duration = time.perf_counter() - start

    # The search must stop within ~2x the abort window plus scheduling slop.
    # The 32-sim quantum at 40 ms/eval would take >= 1.28 s.
    assert duration < 0.8, (
        f"abort took {duration:.3f}s — the wall-clock poll is not bounding the "
        "32-simulation quantum at slow sim rates (S5-02 regression)"
    )
    assert result.best_action is not None or len(result.action_visits) > 0


def test_abort_poll_does_not_change_serial_result_semantics():
    """Without any abort, a small serial search still completes normally."""
    game = zeropawn_cpp.ChessGame()
    config = zeropawn_cpp.MCTSConfig()
    config.batch_size = 1
    mcts = zeropawn_cpp.MCTS(config)

    result = mcts.search(
        game,
        16,
        lambda g: (np.full(4672, 1.0 / 4672, dtype=np.float32), 0.0),
        None,
        None,
    )
    total = float(np.sum(result.action_visits))
    assert abs(total - 1.0) < 1e-5  # visits still normalized
