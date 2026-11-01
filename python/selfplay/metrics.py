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

# Line terminator used when rewriting the telemetry CSV during a header
# upgrade (kept as a named constant so the rewrite is explicit).
CSV_LINE_END = chr(10)

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
        # Lifecycle counters (audit SR-08): aborted games are counted
        # separately and are NEVER folded into draws or completed games.
        self.total_aborted_games = 0
        self.total_invalid_games = 0
        # Extension columns beyond the historical fixed schema (append-only:
        # old files keep their columns, new columns are appended and old rows
        # padded); the CSV is assembled by column NAME so a row can never be
        # written wider/narrower than its header.
        self.run_id: Optional[str] = None
        self._extra_rows_last: Dict[str, Any] = {}
        # Audit F-15: track durable row writes to verify CSV is never left header-only.
        self.rows_written: int = 0

    def _default_headers(self) -> list:
        """Canonical telemetry header (append-only; never reorder/rename)."""
        return [
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
            # ── Provenance columns appended by the 2026-10-31 self-play /
            # replay-buffer remediation (audit SR-08). Append-only: existing
            # files keep their rows; new columns are appended to the header and
            # old rows are padded so widths always match the header.
            "run_id",
            "games_aborted",
            "games_invalid",
            "records_rejected",
            "buffer_total_added",
            "buffer_write_head",
            "buffer_schema_version",
            "checkpoint_file",
            "sample_source",
            "sampled_selfplay_rows",
            "sampled_unique_games",
            "staleness_fallbacks",
        ]

    def _init_csv(self) -> None:
        """Initialize/upgrade the telemetry CSV header and column contract."""
        if self.csv_path is None:
            return
        default = self._default_headers()
        if not self.csv_path.exists():
            self._has_target_entropy = True
            self._csv_columns = list(default)
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self._csv_columns)
            except (OSError, csv.Error) as e:
                logger.warning("Failed to init CSV %s: %s", self.csv_path, e)
            return
        try:
            # Universal-newline read: CRLF files must not leak a stray
            # carriage return into the final column name.
            with open(self.csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            existing = lines[0].strip().split(",") if lines else []
            self._has_target_entropy = ("target_entropy" in existing) if existing else False
            if not existing:
                self._csv_columns = list(default)
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self._csv_columns)
                return
            # Historical retrofit: the oldest format predates target_entropy and
            # tactical_mix_ratio; row data only existed after the header was
            # written in the same order, so insert placeholders positionally.
            if "tactical_mix_ratio" not in existing and "human_mix_ratio" in existing:
                idx = existing.index("human_mix_ratio") + 1
                existing.insert(idx, "tactical_mix_ratio")
                upgraded = [",".join(existing) + CSV_LINE_END]
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) == len(existing) - 1:
                        parts.insert(idx, "0.0")
                    upgraded.append(",".join(parts) + CSV_LINE_END)
                lines = upgraded
            # Append any new default columns, padding older rows with blanks.
            header = lines[0].strip().split(",") if lines else []
            missing = [c for c in default if c not in header]
            if missing:
                header = header + missing
                padded = [",".join(header) + CSV_LINE_END]
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) < len(header):
                        parts = parts + [""] * (len(header) - len(parts))
                    padded.append(",".join(parts) + CSV_LINE_END)
                lines = padded
                logger.info(
                    "[TELEMETRY] Upgraded %s header with %d provenance column(s)",
                    self.csv_path,
                    len(missing),
                )
            self._csv_columns = header
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                f.writelines(lines)
        except Exception as e:
            logger.warning("Failed to check/upgrade CSV header in %s: %s", self.csv_path, e)
            self._csv_columns = list(default)

    def record_game(self, summary: GameSummary) -> None:
        """Record completed game summary.

        An aborted game (shutdown mid-game; audit SR-01) has no result and is
        never counted as a played or drawn game — it increments the dedicated
        aborted counter and returns.
        """
        if getattr(summary, "aborted", False) or getattr(
            summary, "termination_reason", ""
        ) == "aborted":
            self.total_aborted_games += 1
            return
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
        *,
        buffer_total_added: Optional[int] = None,
        buffer_write_head: Optional[int] = None,
        buffer_schema_version: Optional[int] = None,
        games_aborted: Optional[int] = None,
        games_invalid: Optional[int] = None,
        records_rejected: Optional[int] = None,
        checkpoint_file: Optional[str] = None,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute metrics, log to TensorBoard and CSV, and return dictionary.

        The CSV row is assembled by column NAME against the file's own header
        contract, so rows and headers can never drift apart. Provenance columns
        (audit SR-08) carry game lifecycle counters, buffer insertion state,
        checkpoint association and sampling statistics; every one of them is
        optional so older callers keep working unchanged.
        """
        metrics = self.compute_metrics(buffer_size=buffer_size, model_version=model_version)

        # Write to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f"selfplay/{name}", value, global_step=step)

        # Write to CSV
        if self.csv_path is not None:
            rss_mb, _ = query_process_memory()
            _, vram_used_mb, _ = query_gpu_metrics()
            sampling = sampling or {}
            values: Dict[str, Any] = {
                "timestamp": time.time(),
                "step": step,
                **{name: value for name, value in metrics.items()},
                "model_version": int(metrics["model_version"]),
                "rss_mb": rss_mb,
                "vram_used_mb": vram_used_mb,
                "run_id": self.run_id or "",
                "games_aborted": (
                    games_aborted if games_aborted is not None else self.total_aborted_games
                ),
                "games_invalid": (
                    games_invalid if games_invalid is not None else self.total_invalid_games
                ),
                "records_rejected": records_rejected if records_rejected is not None else "",
                "buffer_total_added": (
                    buffer_total_added if buffer_total_added is not None else ""
                ),
                "buffer_write_head": buffer_write_head if buffer_write_head is not None else "",
                "buffer_schema_version": (
                    buffer_schema_version if buffer_schema_version is not None else ""
                ),
                "checkpoint_file": checkpoint_file or "",
                "sample_source": sampling.get("sample_source") or "",
                "sampled_selfplay_rows": sampling.get("sampled_selfplay_rows", ""),
                "sampled_unique_games": sampling.get("sampled_unique_games", ""),
                "staleness_fallbacks": sampling.get("staleness_fallbacks", ""),
            }
            columns = getattr(self, "_csv_columns", None) or self._default_headers()
            row = [values.get(col, "") for col in columns]
            try:
                from python.common.artifacts import ArtifactLockError, artifact_lock

                try:
                    lock = artifact_lock(self.csv_path, timeout=15.0, owner="telemetry-csv")
                except ArtifactLockError:
                    lock = None
                if lock is None:
                    logger.warning(
                        "Telemetry CSV %s is locked by another writer; skipping this row",
                        self.csv_path,
                    )
                else:
                    with lock:
                        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                            self.rows_written += 1
                            # Audit SUS-01: an append without flush could be torn
                            # by a crash mid-write. Flush then fsync so a completed
                            # row is durable and a torn row cannot be half-written.
                            f.flush()
                            try:
                                os.fsync(f.fileno())
                            except OSError:
                                logger.debug("telemetry CSV fsync failed", exc_info=True)
            except (OSError, csv.Error) as e:
                logger.warning("Failed to append CSV %s: %s", self.csv_path, e)

        self.last_flush_time = time.perf_counter()
        return metrics

    def flush_final(
        self,
        step: int = 0,
        buffer_size: Optional[int] = None,
        model_version: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Ensure at least one durable telemetry record is written before close (audit F-15)."""
        if self.rows_written == 0 and self.csv_path is not None:
            try:
                self.log_step(step=step, buffer_size=buffer_size, model_version=model_version, **kwargs)
            except Exception as e:
                logger.warning("Final telemetry flush failed: %s", e)
        self.close()

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
