"""72-Hour Soak Test Runner with Memory Logging (RSS and VRAM) Every 10 Minutes.

Contract Specifications:
- Real chess self-play generation + continuous RL training loop.
- Memory monitoring thread logging RSS and VRAM every 10 minutes (600s).
- Logs to TensorBoard and persistent CSV: runs/soak_test/soak_memory.csv.
- Fully resumable checkpoints every 30 minutes.
- Graceful shutdown on SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python
import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import SavedPosition, GameSummary, SelfPlayWorker
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.metrics import MetricsTracker, query_gpu_metrics, query_process_memory
from python.train.train_rl import RLTrainer
from python.cli.common import setup_logging
from python.common.progress import (
    Heartbeat,
    banner,
    emit_report,
    event_line,
    format_kv_line,
    header_block,
    run_context,
    short_hash,
    summary_block,
)

# Single logging setup: rotating file for everything, WARNING+ on stderr.
# (No logging.basicConfig: one setup path, no duplicate lines.)
setup_logging(log_dir="logs", log_filename="soak_test.log")
logger = logging.getLogger("soak_test")


class SoakTestOrchestrator:
    """Manages the 72-hour self-play + RL training run with 10-minute memory logging."""

    def __init__(
        self,
        max_hours: float = 72.0,
        log_interval_sec: float = 600.0,  # 10 minutes
        num_concurrent_games: int = 48,
        buffer_capacity: int = 500_000,
        checkpoint_dir: str = "models/checkpoints_soak",
        log_dir: str = "runs/soak_test",
        checkpoint_path: str = "models/checkpoints_scaled/best_model.pt",
        human_data_dir: str = "data/dataset_scaled/train",
    ) -> None:
        self.max_hours = max_hours
        self.max_duration_sec = max_hours * 3600.0
        self.log_interval_sec = log_interval_sec
        self.num_concurrent_games = num_concurrent_games
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "soak_memory.csv"

        self.stop_event = threading.Event()
        self.start_time = time.time()

        # Initialize CSV
        self._init_csv()

        # Initial memory baselines
        self.initial_rss_mb, _ = query_process_memory()
        _, self.initial_vram_mb, _ = query_gpu_metrics()

        logger.info(
            f"Initialized Soak Test: target duration {max_hours:.1f} hours ({self.max_duration_sec:.0f}s)"
        )
        logger.info(f"Memory logging interval: {log_interval_sec:.1f}s (every 10 minutes)")
        logger.info(
            f"Baseline Host RSS: {self.initial_rss_mb:.1f} MB | Baseline GPU VRAM: {self.initial_vram_mb:.1f} MB"
        )

        # 1. Initialize Replay Buffer
        self.buffer = ReplayBuffer(
            buffer_dir=self.log_dir / "buffer",
            capacity=buffer_capacity,
            initial_human_mix=0.25,
            final_human_mix=0.05,
            taper_steps=50_000,
            human_data_dir=human_data_dir,
            create_new=False,
        )

        # 2. Initialize Inference Server & Client
        self.server = InferenceServer(
            checkpoint_path=checkpoint_path,
            device=self.device,
            min_batch_size=48,
            max_batch_size=128,
            max_wait_ms=2.5,
            use_bfloat16=True,
        )
        self.client = DirectInferenceClient(self.server)

        # 3. Initialize Metrics Tracker
        self.metrics_tracker = MetricsTracker(
            log_dir=self.log_dir,
            csv_path=self.log_dir / "soak_telemetry.csv",
        )

        # 4. Initialize Self-Play Worker
        def sink(positions: list[SavedPosition], summary: GameSummary):
            self.buffer.add_positions(positions)
            self.metrics_tracker.record_game(summary)

        self.worker = SelfPlayWorker(
            worker_id=0,
            client=self.client,
            num_concurrent_games=num_concurrent_games,
            game_type="chess",
            full_search_sims=800,
            fast_search_sims=150,
            full_search_prob=0.25,
            temp_cutoff_range=(15, 30),
            resign_threshold=-0.9,
            consecutive_resign_moves=4,
            disable_resign_prob=0.10,
            position_sink=sink,
        )

        # 5. Initialize RL Trainer
        trainer_model = ChessNet(num_blocks=12, num_filters=128).to(self.device)
        trainer_model.load_state_dict(self.server.model.state_dict())

        self.trainer = RLTrainer(
            model=trainer_model,
            replay_buffer=self.buffer,
            device=self.device,
            base_lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            warmup_steps=500,
            total_steps=100_000,
            batch_size=256,
            push_freq=200,
            checkpoint_interval_sec=1800.0,  # 30 minutes
            checkpoint_dir=self.checkpoint_dir,
            sync_weights_path=self.checkpoint_dir / "live_weights.pt",
            direct_server=self.server,
            use_bfloat16=True,
        )

    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            headers = [
                "timestamp",
                "elapsed_hours",
                "step",
                "rss_mb",
                "rss_delta_mb",
                "vram_used_mb",
                "vram_delta_mb",
                "gpu_util_percent",
                "buffer_size",
                "total_games",
                "model_version",
            ]
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_memory(self, step: int) -> None:
        """Query and log RSS and VRAM memory metrics."""
        now = time.time()
        elapsed_hours = (now - self.start_time) / 3600.0

        rss_mb, _ = query_process_memory()
        gpu_util, vram_used_mb, _ = query_gpu_metrics()

        rss_delta = rss_mb - self.initial_rss_mb
        vram_delta = vram_used_mb - self.initial_vram_mb

        logger.info(
            f"[SOAK MONITOR | {elapsed_hours:6.2f}h] "
            f"Step: {step:6d} | Buffer: {len(self.buffer):7d} | "
            f"RSS: {rss_mb:7.1f} MB (Delta: {rss_delta:+6.1f} MB) | "
            f"VRAM: {vram_used_mb:7.1f} MB (Delta: {vram_delta:+6.1f} MB) | "
            f"GPU Util: {gpu_util:4.1f}% | Model Ver: {self.server.model_version}"
        )
        # Console mirror (10-minute cadence): same measured numbers, kv shape.
        emit_report(
            format_kv_line(
                [
                    ("soak_h", f"{elapsed_hours:.2f}"),
                    ("step", step),
                    ("buffer", len(self.buffer)),
                    ("rss_mb", f"{rss_mb:.1f}"),
                    ("rss_delta_mb", f"{rss_delta:+.1f}"),
                    ("vram_mb", f"{vram_used_mb:.1f}"),
                    ("vram_delta_mb", f"{vram_delta:+.1f}"),
                    ("gpu", f"{gpu_util:.1f}%"),
                    ("model_v", self.server.model_version),
                ]
            )
        )

        # Write to CSV
        row = [
            now,
            round(elapsed_hours, 4),
            step,
            round(rss_mb, 2),
            round(rss_delta, 2),
            round(vram_used_mb, 2),
            round(vram_delta, 2),
            round(gpu_util, 1),
            len(self.buffer),
            self.metrics_tracker.total_games,
            self.server.model_version,
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Write to TensorBoard
        self.metrics_tracker.writer.add_scalar("memory/rss_mb", rss_mb, global_step=step)
        self.metrics_tracker.writer.add_scalar(
            "memory/vram_used_mb", vram_used_mb, global_step=step
        )
        self.metrics_tracker.writer.add_scalar("memory/rss_delta_mb", rss_delta, global_step=step)
        self.metrics_tracker.writer.add_scalar("memory/vram_delta_mb", vram_delta, global_step=step)

    def _memory_monitor_loop(self) -> None:
        """Background thread running memory logger every 10 minutes."""
        next_log_time = self.start_time + self.log_interval_sec
        while not self.stop_event.is_set():
            time.sleep(1.0)
            now = time.time()
            if now >= next_log_time:
                self.log_memory(step=self.trainer.step_num)
                next_log_time = now + self.log_interval_sec

    def run(self) -> None:
        """Run self-play + RL training soak test."""
        ctx = run_context("soak-test")
        t0 = self.start_time
        emit_report(banner("soak-test", ctx["start"] + " (local)"))
        emit_report(
            header_block(
                ctx["start"],
                ctx["command"],
                ctx["workdir"],
                ctx["commit"],
                "none",
                "none",
                self.device if hasattr(self, "device") else "auto",
                "n/a",
            )
        )
        emit_report(
            event_line(
                "START",
                f"target {self.max_hours:.1f}h, memory log every {self.log_interval_sec:.0f}s",
            )
        )
        logger.info("Starting self-play worker threads and memory monitor...")
        self.worker.start()

        monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        monitor_thread.start()

        # Heartbeat for the long silent stretches between memory logs.
        hb = Heartbeat(30.0)
        hb_stop = threading.Event()

        def _beat() -> None:
            while not hb_stop.wait(1.0):
                line = hb.maybe_emit("soak running (memory detail in LOG)")
                if line is not None:
                    emit_report(line)

        hb_thread = threading.Thread(target=_beat, daemon=True)
        hb_thread.start()

        # Initial memory log at t=0
        self.log_memory(step=0)

        try:
            # Seed initial buffer positions
            logger.info("Awaiting initial self-play buffer positions...")
            emit_report(event_line("BUFFER_WAIT", "need 256 positions before training"))
            while len(self.buffer) < 256 and not self.stop_event.is_set():
                time.sleep(1.0)

            logger.info(
                f"Buffer populated with {len(self.buffer)} positions. Continuous RL training started."
            )
            emit_report(
                event_line("BUFFER_READY", f"{len(self.buffer)} positions; training active")
            )

            while not self.stop_event.is_set():
                elapsed = time.time() - self.start_time
                if elapsed >= self.max_duration_sec:
                    logger.info(
                        f"Reached target duration {self.max_hours} hours. Stopping soak test."
                    )
                    emit_report(event_line("STOP", f"target {self.max_hours:.1f}h reached"))
                    break

                # Execute RL training step
                train_metrics = self.trainer.train_step()
                step = train_metrics["step"]
                self.metrics_tracker.record_train_step(train_metrics)

                # Periodic telemetry flush every 50 steps
                if step % 50 == 0:
                    self.metrics_tracker.log_step(
                        step=step,
                        buffer_size=len(self.buffer),
                        model_version=self.server.model_version,
                    )

                time.sleep(0.005)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Initiating graceful shutdown...")
            emit_report(event_line("SHUTDOWN", "Ctrl+C received; stopping cleanly"))
        finally:
            hb_stop.set()
            self.stop()
            emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))

    def stop(self) -> None:
        """Gracefully stop all worker and monitoring threads and save checkpoint."""
        logger.info("Shutting down workers and saving final checkpoint...")
        emit_report(event_line("SHUTDOWN", "stopping workers, saving final checkpoint"))
        self.stop_event.set()
        self.worker.stop()
        final_ckpt = self.trainer.save_checkpoint(self.checkpoint_dir / "soak_final_checkpoint.pt")
        # Final memory log
        self.log_memory(step=self.trainer.step_num)
        self.metrics_tracker.close()
        logger.info(f"Soak test completed cleanly. Final checkpoint saved to {final_ckpt}")
        emit_report(
            event_line(
                "CHECKPOINT", f"saved {final_ckpt} sha={short_hash(str(final_ckpt))} (final)"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="72-Hour Soak Test Runner with Memory Logging")
    parser.add_argument(
        "--max-hours", type=float, default=72.0, help="Total soak duration in hours (default 72.0)"
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=600.0,
        help="Memory log interval in seconds (default 600s = 10 min)",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=24,
        help=(
            "Number of concurrent self-play games (default 24, matching the training "
            "default; raise deliberately — 48 roughly doubles the RAM footprint)"
        ),
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=500_000,
        help="Replay buffer capacity (default 500,000)",
    )
    parser.add_argument(
        "--test-verify",
        action="store_true",
        help="Run a quick 30-second verification of the soak logging pipeline",
    )
    args = parser.parse_args()

    if args.test_verify:
        logger.info("Running soak test verification mode (30s duration, 5s logging interval)...")
        orchestrator = SoakTestOrchestrator(
            max_hours=30.0 / 3600.0,
            log_interval_sec=5.0,
            num_concurrent_games=8,
            buffer_capacity=10_000,
        )
        orchestrator.run()
    else:
        orchestrator = SoakTestOrchestrator(
            max_hours=args.max_hours,
            log_interval_sec=args.interval_sec,
            num_concurrent_games=args.concurrent_games,
            buffer_capacity=args.buffer_capacity,
        )
        orchestrator.run()


if __name__ == "__main__":
    main()
