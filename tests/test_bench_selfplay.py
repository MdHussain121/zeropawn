"""Tests for scripts/bench_selfplay.py benchmark protocol."""

import json
from pathlib import Path
import pytest

from scripts.bench_selfplay import (
    BenchmarkRunner,
    query_gpu_telemetry,
    query_system_telemetry,
    query_throttle_reasons,
)


def test_telemetry_query_functions():
    """Verify hardware telemetry query functions return non-empty data without crashing."""
    gpu = query_gpu_telemetry()
    assert isinstance(gpu, dict)
    assert "gpu_util_pct" in gpu
    assert "temp_c" in gpu

    sys_telemetry = query_system_telemetry()
    assert isinstance(sys_telemetry, dict)
    assert "avg_cpu_util_pct" in sys_telemetry
    assert "free_ram_mb" in sys_telemetry
    assert sys_telemetry["free_ram_mb"] > 0

    throttle = query_throttle_reasons()
    assert isinstance(throttle, str)


def test_benchmark_runner_dry_run(tmp_path):
    """Verify BenchmarkRunner executes warm-up + counted phases, produces CSV and JSON."""
    runner = BenchmarkRunner(
        mode="selfplay",
        warmup_sec=1.0,
        counted_sec=2.0,
        sample_interval=1.0,
        trials=1,
        concurrency=2,
        full_sims=5,
        fast_sims=5,
        full_search_prob=0.35,
        tablebase_path=None,
        output_dir=str(tmp_path),
        checkpoint_path="",
        watchdog_free_ram_mb=0.0,
    )

    result = runner.run_benchmark()

    assert result["mode"] == "selfplay"
    assert result["trials"] == 1
    assert "sims_per_sec" in result["aggregated"]

    csv_path = tmp_path / "selfplay_trial_1.csv"
    assert csv_path.is_file()

    json_path = tmp_path / "selfplay_summary.json"
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["mode"] == "selfplay"
