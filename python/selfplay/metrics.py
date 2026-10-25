"""Telemetry and Metrics Dashboard for ZeroPawn Self-Play.

Contract Specifications:
Tracks and logs all 12 key metrics to TensorBoard and CSV:
1. simulations_per_sec
2. games_per_hour
3. gpu_utilization (%)
4. batch_fill_rate (0.0 to 1.0)
5. draw_rate (0.0 to 1.0)
6. resign_rate (0.0 to 1.0)
7. false_resign_rate (0.0 to 1.0)
8. average_game_length (plies)
9. policy_entropy (nats)
10. buffer_size (positions)
11. human_mix_ratio (0.0 to 1.0)
12. tactical_mix_ratio (0.0 to 1.0)
13. model_version (integer)
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from python.selfplay.worker import GameSummary

logger = logging.getLogger(__name__)

_GPU_CACHE: Tuple[float, Tuple[float, float, float]] = (0.0, (0.0, 0.0, 0.0))
_GPU_CACHE_TTL_SEC = 5.0


def query_gpu_metrics() -> Tuple[float, float, float]:
    """Query GPU utilization %, VRAM used (MB), and total VRAM (MB).

    Returns:
        Tuple of (utilization_percent, vram_used_mb, vram_total_mb).
    """
    global _GPU_CACHE
    now = time.monotonic()
    if now - _GPU_CACHE[0] < _GPU_CACHE_TTL_SEC:
        return _GPU_CACHE[1]
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    try:
        # Fast query via nvidia-smi
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            timeout=5,
        ).strip()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty output from nvidia-smi")
        device_idx = 0
        if torch.cuda.is_available():
            try:
                device_idx = torch.cuda.current_device()
            except (RuntimeError, ValueError):
                device_idx = 0
        target_line = lines[device_idx] if 0 <= device_idx < len(lines) else lines[0]
        parts = [float(x.strip()) for x in target_line.split(",")[:3]]
        if len(parts) < 3:
            raise ValueError(f"Expected at least 3 values from nvidia-smi, got {len(parts)}")
        result = (parts[0], parts[1], parts[2])
        _GPU_CACHE = (now, result)
        return result
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.debug("nvidia-smi query failed: %s", e)
        # Fallback to PyTorch VRAM query
        try:
            allocated_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            reserved_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            return 0.0, allocated_mb, reserved_mb
        except (RuntimeError, AttributeError) as e2:
            logger.debug("torch cuda fallback failed: %s", e2)
            return 0.0, 0.0, 0.0


def query_process_memory() -> Tuple[float, float]:
    """Query current process RSS memory (MB) and system RAM usage (%)."""
    try:
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
        sys_percent = psutil.virtual_memory().percent
        return rss_mb, sys_percent
    except (psutil.Error, OSError, ValueError) as e:
        logger.debug("process memory query failed: %s", e)
        return 0.0, 0.0


class MetricsTracker:
    """Central metrics aggregator and TensorBoard/CSV logger."""

    def __init__(
        self,
        log_dir: Union[str, Path] = "runs/selfplay",
        csv_path: Optional[Union[str, Path]] = None,
        window_size: int = 100,
        inference_server: Optional[Any] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.csv_path = Path(csv_path) if csv_path is not None else None
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_csv()

        self.window_size = window_size
        self.start_time = time.perf_counter()
        self.last_flush_time = self.start_time

        # Cumulative counters
        self.total_games = 0
        self.total_draws = 0
        self.total_resigns = 0
        self.total_would_have_resigned = 0
        self.total_false_resigns = 0
        self.total_plies = 0
        self.total_simulations = 0

        # Sliding window deques for rate estimation
        self.recent_game_durations = deque(maxlen=window_size)
        self.recent_game_plies = deque(maxlen=window_size)
        self.recent_simulations = deque(maxlen=window_size)
        self.recent_batch_fills = deque(maxlen=window_size)
        self.recent_policy_entropies = deque(maxlen=window_size)
        self.recent_target_entropies = deque(maxlen=window_size)
        self.recent_human_mixes = deque(maxlen=window_size)
        self.recent_tactical_mixes = deque(maxlen=window_size)

        # Latest values
        self.inference_server = inference_server
        self.latest_model_version = 0
        self.latest_buffer_size = 0
        self._has_target_entropy = True

    def _init_csv(self) -> None:
        """Initialize CSV file with header."""
        if self.csv_path is None:
            return
        if self.csv_path.exists():
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                self._has_target_entropy = ("target_entropy" in first_line) if first_line else False
                if first_line and "tactical_mix_ratio" not in first_line:
                    with open(self.csv_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    if lines:
                        old_hdr = lines[0].strip().split(",")
                        if "human_mix_ratio" in old_hdr:
                            idx = old_hdr.index("human_mix_ratio") + 1
                            old_hdr.insert(idx, "tactical_mix_ratio")
                            new_lines = [",".join(old_hdr) + "\n"]
                            for line in lines[1:]:
                                parts = line.strip().split(",")
                                if len(parts) == len(old_hdr) - 1:
                                    parts.insert(idx, "0.0")
                                    new_lines.append(",".join(parts) + "\n")
                                else:
                                    new_lines.append(line)
                            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                                f.writelines(new_lines)
            except Exception as e:
                logger.warning("Failed to check CSV header in %s: %s", self.csv_path, e)
            return
        self._has_target_entropy = True
        headers = [
            "timestamp",
            "step",
            "simulations_per_sec",
            "games_per_hour",
            "gpu_utilization",
            "batch_fill_rate",
            "draw_rate",
            "resign_rate",
            "false_resign_rate",
            "average_game_length",
            "policy_entropy",
            "target_entropy",
            "buffer_size",
            "human_mix_ratio",
            "tactical_mix_ratio",
            "model_version",
            "rss_mb",
            "vram_used_mb",
        ]
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except (OSError, csv.Error) as e:
            logger.warning("Failed to init CSV %s: %s", self.csv_path, e)

    def record_game(self, summary: GameSummary) -> None:
        """Record completed game summary."""
        self.total_games += 1
        self.total_plies += summary.length_plies
        self.total_simulations += summary.total_simulations

        if summary.result == "1/2-1/2":
            self.total_draws += 1
        if summary.resigned:
            self.total_resigns += 1
        if summary.would_have_resigned:
            self.total_would_have_resigned += 1
            if summary.is_false_resign:
                self.total_false_resigns += 1

        self.recent_game_durations.append(summary.duration_sec)
        self.recent_game_plies.append(summary.length_plies)
        self.recent_simulations.append(summary.total_simulations)
        self.latest_model_version = max(self.latest_model_version, summary.model_version)

    def record_batch_inference(self, batch_size: int, max_batch_size: int) -> None:
        """Record inference batch fill rate."""
        fill_rate = float(batch_size) / float(max(1, max_batch_size))
        self.recent_batch_fills.append(fill_rate)

    def record_train_step(self, metrics: Dict[str, float]) -> None:
        """Record RL training step metrics."""
        if "policy_entropy" in metrics:
            self.recent_policy_entropies.append(metrics["policy_entropy"])
        if "target_entropy" in metrics:
            self.recent_target_entropies.append(metrics["target_entropy"])
        if "actual_human_mix" in metrics:
            self.recent_human_mixes.append(metrics["actual_human_mix"])
        if "actual_tactical_mix" in metrics:
            self.recent_tactical_mixes.append(metrics["actual_tactical_mix"])
        elif "tactical_mix_ratio" in metrics:
            self.recent_tactical_mixes.append(metrics["tactical_mix_ratio"])

    def get_latest_metrics(self) -> Dict[str, float]:
        """Return the latest computed set of 12 metrics."""
        return self.compute_metrics()

    def compute_metrics(
        self,
        buffer_size: Optional[int] = None,
        model_version: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute the full set of 12 required metrics."""
        now = time.perf_counter()
        elapsed_total = max(1e-5, now - self.start_time)

        if buffer_size is not None:
            self.latest_buffer_size = buffer_size
        if model_version is not None:
            self.latest_model_version = model_version

        # 1. simulations_per_sec
        sims_per_sec = float(self.total_simulations) / elapsed_total

        # 2. games_per_hour
        games_per_hour = (float(self.total_games) / elapsed_total) * 3600.0

        # 3. gpu_utilization
        gpu_util, vram_used, _ = query_gpu_metrics()

        # 4. batch_fill_rate (instrumented from inference server, non-destructive)
        batch_fill = 0.0
        if self.inference_server is not None:
            try:
                batch_fill = float(self.inference_server.current_fill_rate)
            except Exception:
                pass
        if batch_fill == 0.0:
            batch_fill = (
                float(sum(self.recent_batch_fills) / len(self.recent_batch_fills))
                if self.recent_batch_fills
                else 0.0
            )

        # 5. draw_rate
        draw_rate = float(self.total_draws) / max(1, self.total_games)

        # 6. resign_rate
        resign_rate = float(self.total_resigns) / max(1, self.total_games)

        # 7. false_resign_rate (calibrated from games where resign was disabled)
        false_resign_rate = (
            float(self.total_false_resigns) / float(self.total_would_have_resigned)
            if self.total_would_have_resigned > 0
            else 0.0
        )

        # 8. average_game_length
        avg_game_length = (
            float(sum(self.recent_game_plies) / len(self.recent_game_plies))
            if self.recent_game_plies
            else (float(self.total_plies) / max(1, self.total_games))
        )

        # 9. policy_entropy
        policy_entropy = (
            float(sum(self.recent_policy_entropies) / len(self.recent_policy_entropies))
            if self.recent_policy_entropies
            else 0.0
        )

        # 10. target_entropy
        target_entropy = (
            float(sum(self.recent_target_entropies) / len(self.recent_target_entropies))
            if self.recent_target_entropies
            else 0.0
        )

        # 11. buffer_size
        buf_size = float(self.latest_buffer_size)

        # 12. human_mix_ratio
        human_mix = (
            float(sum(self.recent_human_mixes) / len(self.recent_human_mixes))
            if self.recent_human_mixes
            else 0.0
        )

        # 13. tactical_mix_ratio
        tactical_mix = (
            float(sum(self.recent_tactical_mixes) / len(self.recent_tactical_mixes))
            if self.recent_tactical_mixes
            else 0.0
        )

        # 14. model_version
        version = float(self.latest_model_version)

        return {
            "simulations_per_sec": sims_per_sec,
            "games_per_hour": games_per_hour,
            "gpu_utilization": gpu_util,
            "batch_fill_rate": batch_fill,
            "draw_rate": draw_rate,
            "resign_rate": resign_rate,
            "false_resign_rate": false_resign_rate,
            "average_game_length": avg_game_length,
            "policy_entropy": policy_entropy,
            "target_entropy": target_entropy,
            "buffer_size": buf_size,
            "human_mix_ratio": human_mix,
            "tactical_mix_ratio": tactical_mix,
            "model_version": version,
        }

    def log_step(
        self,
        step: int,
        buffer_size: Optional[int] = None,
        model_version: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute metrics, log to TensorBoard and CSV, and return dictionary."""
        metrics = self.compute_metrics(buffer_size=buffer_size, model_version=model_version)

        # Write to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f"selfplay/{name}", value, global_step=step)

        # Write to CSV
        if self.csv_path is not None:
            rss_mb, _ = query_process_memory()
            _, vram_used_mb, _ = query_gpu_metrics()
            row = [
                time.time(),
                step,
                metrics["simulations_per_sec"],
                metrics["games_per_hour"],
                metrics["gpu_utilization"],
                metrics["batch_fill_rate"],
                metrics["draw_rate"],
                metrics["resign_rate"],
                metrics["false_resign_rate"],
                metrics["average_game_length"],
                metrics["policy_entropy"],
            ]
            if getattr(self, "_has_target_entropy", True):
                row.append(metrics["target_entropy"])
            row.extend(
                [
                    metrics["buffer_size"],
                    metrics["human_mix_ratio"],
                    metrics["tactical_mix_ratio"],
                    int(metrics["model_version"]),
                    rss_mb,
                    vram_used_mb,
                ]
            )
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except (OSError, csv.Error) as e:
                logger.warning("Failed to append CSV %s: %s", self.csv_path, e)

        self.last_flush_time = time.perf_counter()
        return metrics

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
