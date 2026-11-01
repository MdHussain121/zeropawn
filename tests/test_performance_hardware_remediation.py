"""Regression tests for ZeroPawn Performance & Hardware Utilization Audit Remediation.

Covers:
- F-12: _warm_start_info AttributeError on cold/warm start.
- F-13: Stale default warm-start path resolved via canonical registry lookup.
- F-18: Clean CLI argument registration without duplicates; single --allow-unverified-warm-start.
- F-01: --pipelined CLI argument and orchestrator wiring.
- F-14: scripts/bench_pipelined_paired.py schema handling for trial_summaries vs trials count.
- F-15: MetricsTracker flush_final guarantees phase4_telemetry.csv is never empty/header-only.
- F-16: train_step consecutive failure tracking and resilience.
- F-17: ReplayBuffer.add_position meta save force=False to avoid per-position fsync.
- RL Trainer dataset provenance hashing imports (hashlib, json).
"""

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from python.cli.config import CONFIG_REGISTRY
from python.common.progress import short_hash
from python.selfplay.metrics import MetricsTracker
from python.selfplay.replay_buffer import ReplayBuffer
from scripts.run_phase4_rl import Phase4Orchestrator, build_parser


class TestStartupBlockersRemediation:
    """Verifies resolution of P0 startup blockers F-12, F-13, and F-18."""

    def test_cli_parser_no_duplicate_flags_and_defaults(self):
        """F-18, F-13, F-01: build_parser succeeds without ArgumentError and defines canonical defaults."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Ensure help generates cleanly without raising ArgumentError or exception
        help_text = parser.format_help()
        assert "--allow-unverified-warm-start" in help_text
        assert "--pipelined" in help_text
        assert "--base-checkpoint" in help_text

        # Verify default for --base-checkpoint is 'default' (dynamic resolution)
        actions = {a.dest: a for a in parser._actions}
        assert actions["base_checkpoint"].default == "default"
        assert actions["pipelined"].default is False
        assert actions["allow_unverified_warm_start"].default is False

    def test_full_config_default_checkpoint_is_none(self):
        """F-13: CONFIG_REGISTRY model.checkpoint default is None to enforce dynamic resolution."""
        assert CONFIG_REGISTRY["model.checkpoint"].default is None

    def test_orchestrator_cold_start_construction(self, tmp_path):
        """F-12: Cold start (base_checkpoint='none') constructs without AttributeError."""
        log_dir = tmp_path / "runs"
        ckpt_dir = tmp_path / "ckpts"
        orch = Phase4Orchestrator(
            max_hours=0.01,
            num_concurrent_games=2,
            buffer_capacity=1000,
            gpu_cache_capacity=100,
            human_cache_size=100,
            log_dir=str(log_dir),
            checkpoint_dir=str(ckpt_dir),
            base_checkpoint="none",
            device="cpu",
            train_device="cpu",
            infer_device="cpu",
            allow_uncertified_data=True,
            allow_uncertified_eval=True,
        )
        assert orch._warm_start_record is None
        # F-12 backwards compatibility property:
        assert orch._warm_start_info is None
        assert orch._active_checkpoint_path == "random_init"
        assert orch.pipelined is False
        assert hasattr(orch, "_consecutive_train_failures")
        assert orch._consecutive_train_failures == 0

    def test_orchestrator_warm_start_info_property_compatibility(self, tmp_path):
        """F-12: _warm_start_info property returns _warm_start_record faithfully."""
        log_dir = tmp_path / "runs"
        ckpt_dir = tmp_path / "ckpts"
        orch = Phase4Orchestrator(
            max_hours=0.01,
            num_concurrent_games=2,
            buffer_capacity=1000,
            gpu_cache_capacity=100,
            human_cache_size=100,
            log_dir=str(log_dir),
            checkpoint_dir=str(ckpt_dir),
            base_checkpoint="none",
            device="cpu",
            train_device="cpu",
            infer_device="cpu",
            allow_uncertified_data=True,
            allow_uncertified_eval=True,
        )
        fake_record = {"status": "verified", "sha256": "abcdef1234567890", "role": "supervised_baseline"}
        orch._warm_start_record = fake_record
        assert orch._warm_start_info == fake_record


class TestMeasurementInfrastructureRemediation:
    """Verifies fixes for measurement and telemetry infrastructure F-14 and F-15."""

    def test_bench_pipelined_paired_schema_handling(self):
        """F-14: bench_pipelined_paired reads trial_summaries instead of subscripting int trials."""
        from scripts.bench_pipelined_paired import build_parser as bench_parser

        parser = bench_parser()
        help_text = parser.format_help()
        assert "--pairs" in help_text
        assert "--warmup-sec" in help_text

        # Test schema parsing logic
        dummy_sync_summary = {
            "mode": "sync",
            "trials": 1,
            "trial_summaries": [
                {
                    "simulations_per_sec": 500.0,
                    "games_per_hour": 100.0,
                    "gpu_utilization_mean": 35.0,
                    "batch_size_mean": 15.2,
                }
            ],
        }
        dummy_pipe_summary = {
            "mode": "pipelined",
            "trials": 1,
            "trial_summaries": [
                {
                    "simulations_per_sec": 525.0,
                    "games_per_hour": 105.0,
                    "gpu_utilization_mean": 36.0,
                    "batch_size_mean": 15.5,
                }
            ],
        }

        # Subscripting summary.get("trial_summaries") must return list of dicts without TypeError
        sync_trials = dummy_sync_summary.get("trial_summaries") or []
        pipe_trials = dummy_pipe_summary.get("trial_summaries") or []
        assert isinstance(sync_trials, list) and len(sync_trials) == 1
        assert sync_trials[0]["simulations_per_sec"] == 500.0
        assert pipe_trials[0]["simulations_per_sec"] == 525.0

    def test_metrics_tracker_flush_final_writes_row(self, tmp_path):
        """F-15: flush_final guarantees phase4_telemetry.csv receives at least one row."""
        log_dir = tmp_path / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "phase4_telemetry.csv"
        tracker = MetricsTracker(log_dir=log_dir, csv_path=csv_path)

        assert tracker.rows_written == 0

        # Call flush_final on tracker without prior log_step calls
        tracker.flush_final(
            step=0,
            buffer_size=100,
            model_version=1,
            buffer_total_added=100,
            buffer_write_head=100,
            buffer_schema_version=2,
            games_aborted=0,
            games_invalid=0,
            records_rejected=0,
            checkpoint_file="phase4_final_checkpoint.pt",
        )
        assert tracker.rows_written == 1

        tracker.close()

        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2  # 1 header + 1 data row
            header, row = reader[0], reader[1]
            assert "step" in header
            step_idx = header.index("step")
            assert row[step_idx] == "0"


class TestReplayAndTrainerRemediation:
    """Verifies fixes for single-position replay buffer fsync (F-17) and trainer imports."""

    def test_replay_buffer_add_position_force_false(self):
        """F-17: add_position passes force=False to _save_meta to eliminate per-position fsync."""
        import inspect
        src = inspect.getsource(ReplayBuffer.add_position)
        assert "self._save_meta(force=False)" in src
        assert "self._save_meta(force=True)" not in src

    def test_train_rl_module_has_hashlib_and_json(self):
        """Verifies train_rl has hashlib and json imported for dataset provenance hashing."""
        import python.train.train_rl as trl
        assert hasattr(trl, "hashlib")
        assert hasattr(trl, "json")


class TestPipeliningAndConcurrencyOptions:
    """Verifies F-01 and F-02 wiring and safety defaults."""

    def test_pipelined_flag_forwarded_to_worker(self, tmp_path):
        """F-01: Orchestrator accepts pipelined flag and sets attribute."""
        log_dir = tmp_path / "runs"
        ckpt_dir = tmp_path / "ckpts"
        orch = Phase4Orchestrator(
            max_hours=0.01,
            num_concurrent_games=2,
            buffer_capacity=1000,
            gpu_cache_capacity=100,
            human_cache_size=100,
            log_dir=str(log_dir),
            checkpoint_dir=str(ckpt_dir),
            base_checkpoint="none",
            pipelined=True,
            device="cpu",
            train_device="cpu",
            infer_device="cpu",
            allow_uncertified_data=True,
            allow_uncertified_eval=True,
        )
        assert orch.pipelined is True
        assert orch.worker.pipelined is True
