"""Unit tests for python/common/progress.py."""

import time
import pytest

from python.common.progress import format_progress_line, ProgressReporter


def test_format_progress_line_structure():
    line = format_progress_line(
        step=500,
        policy_loss=1.8765,
        wdl_loss=0.6543,
        buffer_size=20000,
        games_count=150,
        would_resign_rate=0.730,
        draw_rate=0.710,
        cap_rate=0.080,
        avg_plies=220.5,
        sims_per_sec=1250.0,
        gpu_util=82.5,
        model_version=5,
        timestamp=1700000000.0,
    )

    # Check required fields
    assert "step=500" in line
    assert "policy_loss=1.8765" in line
    assert "wdl_loss=0.6543" in line
    assert "buffer=20000" in line
    assert "games=150" in line
    assert "resign=off" in line
    assert "would_resign_rate=0.730" in line
    assert "draw_rate=0.710" in line
    assert "cap_rate=0.080" in line
    assert "avg_plies=220.5" in line
    assert "sims_per_sec=1250.0" in line
    assert "gpu=82.5%" in line
    assert "model_version=5" in line

    # Verify explicitly removed fields are absent
    assert "resign_mode" not in line
    assert "resign_t" not in line
    assert "false_resign_ub" not in line

    # Time stamp bracket prefix
    assert line.startswith("[")
    assert "] step=" in line


def test_progress_reporter_rate_limiting():
    emitted = []
    reporter = ProgressReporter(interval_sec=10.0, output_fn=emitted.append)

    t0 = 1000.0
    # First report
    res1 = reporter.maybe_report(
        step=10,
        policy_loss=2.0,
        wdl_loss=0.8,
        buffer_size=1000,
        games_count=10,
        would_resign_rate=0.5,
        draw_rate=0.5,
        cap_rate=0.0,
        avg_plies=100.0,
        total_sims=5000,
        gpu_util=70.0,
        model_version=1,
        current_time=t0,
    )
    assert res1 is not None
    assert len(emitted) == 1

    # Second call 3 seconds later: should be suppressed
    res2 = reporter.maybe_report(
        step=20,
        policy_loss=1.9,
        wdl_loss=0.7,
        buffer_size=2000,
        games_count=20,
        would_resign_rate=0.5,
        draw_rate=0.5,
        cap_rate=0.0,
        avg_plies=100.0,
        total_sims=10000,
        gpu_util=75.0,
        model_version=1,
        current_time=t0 + 3.0,
    )
    assert res2 is None
    assert len(emitted) == 1

    # Forced call: should emit even within 10s
    res3 = reporter.maybe_report(
        step=30,
        policy_loss=1.8,
        wdl_loss=0.6,
        buffer_size=3000,
        games_count=30,
        would_resign_rate=0.5,
        draw_rate=0.5,
        cap_rate=0.0,
        avg_plies=100.0,
        total_sims=15000,
        gpu_util=75.0,
        model_version=1,
        force=True,
        current_time=t0 + 5.0,
    )
    assert res3 is not None
    assert len(emitted) == 2

    # Call after 11 seconds from last emitted report (t0 + 5.0 + 11.0 = t0 + 16.0)
    res4 = reporter.maybe_report(
        step=40,
        policy_loss=1.7,
        wdl_loss=0.5,
        buffer_size=4000,
        games_count=40,
        would_resign_rate=0.5,
        draw_rate=0.5,
        cap_rate=0.0,
        avg_plies=100.0,
        total_sims=26000,
        gpu_util=80.0,
        model_version=2,
        current_time=t0 + 16.0,
    )
    assert res4 is not None
    assert len(emitted) == 3


def test_progress_reporter_sims_per_sec():
    reporter = ProgressReporter(interval_sec=10.0)

    # Initial call sets baseline
    s0 = reporter.compute_sims_per_sec(total_sims=10000, current_time=100.0)
    assert s0 == 0.0

    # 10 seconds later, 20,000 more sims (total 30,000)
    s1 = reporter.compute_sims_per_sec(total_sims=30000, current_time=110.0)
    assert pytest.approx(s1, rel=1e-3) == 2000.0
