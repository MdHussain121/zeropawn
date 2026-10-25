"""Tests for Task 5: Metrics Dashboard (TensorBoard / CSV tracking all 12 key metrics)."""

import csv
from pathlib import Path
import pytest

from python.selfplay.worker import GameSummary
from python.selfplay.metrics import MetricsTracker, query_gpu_metrics, query_process_memory


def test_metrics_tracker_all_12_metrics(tmp_path):
    """Verify that MetricsTracker computes and logs all 12 required metrics."""
    log_dir = tmp_path / "tensorboard_logs"
    csv_file = tmp_path / "telemetry.csv"

    tracker = MetricsTracker(log_dir=log_dir, csv_path=csv_file)

    # Record completed games
    tracker.record_game(
        GameSummary(
            game_id=1,
            length_plies=40,
            result="1-0",
            resigned=True,
            resigned_side_white=False,
            would_have_resigned=False,
            is_false_resign=False,
            duration_sec=2.0,
            total_simulations=4000,
            model_version=2,
        )
    )
    tracker.record_game(
        GameSummary(
            game_id=2,
            length_plies=60,
            result="1/2-1/2",
            resigned=False,
            resigned_side_white=None,
            would_have_resigned=True,
            is_false_resign=True,  # Met resign threshold but drew! (False resign)
            duration_sec=3.0,
            total_simulations=6000,
            model_version=2,
        )
    )

    # Record batch inferences
    tracker.record_batch_inference(batch_size=256, max_batch_size=512)
    tracker.record_batch_inference(batch_size=384, max_batch_size=512)

    # Record training step
    tracker.record_train_step(
        {
            "policy_entropy": 2.35,
            "actual_human_mix": 0.20,
            "actual_tactical_mix": 0.12,
        }
    )

    # Flush metrics
    metrics = tracker.log_step(step=100, buffer_size=15000, model_version=2)

    required_keys = [
        "simulations_per_sec",
        "games_per_hour",
        "gpu_utilization",
        "batch_fill_rate",
        "draw_rate",
        "resign_rate",
        "false_resign_rate",
        "average_game_length",
        "policy_entropy",
        "buffer_size",
        "human_mix_ratio",
        "tactical_mix_ratio",
        "model_version",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"

    # Verify calculation accuracy
    assert metrics["draw_rate"] == 0.5  # 1 draw out of 2 games
    assert metrics["resign_rate"] == 0.5  # 1 resign out of 2 games
    assert metrics["false_resign_rate"] == 1.0  # 1 false resign out of 1 that met resign condition
    assert metrics["average_game_length"] == 50.0  # (40 + 60) / 2
    assert abs(metrics["batch_fill_rate"] - ((256 / 512 + 384 / 512) / 2)) < 1e-4
    assert metrics["buffer_size"] == 15000.0
    assert metrics["model_version"] == 2.0
    assert abs(metrics["human_mix_ratio"] - 0.20) < 1e-4
    assert abs(metrics["tactical_mix_ratio"] - 0.12) < 1e-4
    assert abs(metrics["policy_entropy"] - 2.35) < 1e-4

    tracker.close()

    # Verify CSV file
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 data row
        assert rows[0][1] == "step"
        assert rows[1][1] == "100"
        tactical_col = rows[0].index("tactical_mix_ratio")
        assert abs(float(rows[1][tactical_col]) - 0.12) < 1e-4


def test_hardware_queries():
    """Verify that system hardware queries succeed without throwing exceptions."""
    print("\n  [Op 1/2] Querying GPU utilization and VRAM status...")
    gpu_util, vram_used, vram_total = query_gpu_metrics()
    assert isinstance(gpu_util, float)
    assert isinstance(vram_used, float)
    assert isinstance(vram_total, float)
    print(f"    [OK] GPU Metrics: util={gpu_util}%, vram={vram_used:.1f}MB / {vram_total:.1f}MB")

    print("  [Op 2/2] Querying CPU process memory & system usage...")
    rss_mb, sys_percent = query_process_memory()
    assert rss_mb > 0.0
    assert 0.0 <= sys_percent <= 100.0
    print(f"    [OK] System Memory: RSS={rss_mb:.1f}MB, process_cpu={sys_percent:.1f}%")
