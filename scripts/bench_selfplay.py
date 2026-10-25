"""Benchmark harness for self-play, training, and combined throughput.

Executes the Task 1 measurement protocol:
- Fixed configuration: 400 full / 80 fast sims, full_search_prob=0.35, concurrency=6,
  tablebase on (Syzygy 3-4-5), resignation off.
- 12-minute run per trial: 2 minutes warm-up (uncounted), 10 minutes counted.
- 3 trials per mode.
- 10-second sampling of all required system and performance metrics.
- Computes mean, standard deviation, and run-to-run variance across trials.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root and build/cpp are on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_build_dir = _repo_root / "build" / "cpp"
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))

# Windows MinGW DLL path
_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_winlibs_bin)
    except OSError:
        pass

import numpy as np
import psutil
import torch

from python.model.chess_net import ChessNet
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import GameSummary, SavedPosition, SelfPlayWorker
from python.train.train_rl import RLTrainer
from python.cli.common import setup_logging
from python.common.progress import (
    banner,
    emit_error,
    emit_report,
    error_block,
    event_line,
    format_kv_line,
    header_block,
    parameters_table,
    run_context,
    summary_block,
)

# Single logging setup: rotating file for everything, WARNING+ on stderr.
# (No logging.basicConfig: one setup path, no duplicate lines.)
setup_logging(log_dir="logs", log_filename="bench_selfplay.log")
logger = logging.getLogger("bench_selfplay")


def query_gpu_telemetry() -> Dict[str, Any]:
    """Query GPU utilization, clocks, power, temperature, and memory."""
    res = {
        "gpu_util_pct": 0.0,
        "gpu_clock_mhz": 0.0,
        "mem_clock_mhz": 0.0,
        "power_w": 0.0,
        "temp_c": 0.0,
        "vram_used_mb": 0.0,
    }
    if not torch.cuda.is_available():
        return res
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, encoding="utf-8", timeout=3
        ).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            parts = [float(x.strip()) for x in lines[0].split(",")]
            if len(parts) >= 6:
                res["gpu_util_pct"] = parts[0]
                res["gpu_clock_mhz"] = parts[1]
                res["mem_clock_mhz"] = parts[2]
                res["power_w"] = parts[3]
                res["temp_c"] = parts[4]
                res["vram_used_mb"] = parts[5]
    except Exception:
        pass
    return res


def query_throttle_reasons() -> str:
    """Parse active performance throttling reasons from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            timeout=3,
        )
        reasons = []
        in_reasons = False
        for line in out.splitlines():
            line_str = line.strip()
            if "Clocks Event Reasons" in line_str and "Counters" not in line_str:
                in_reasons = True
                continue
            if in_reasons:
                if "Counters" in line_str or ":" not in line_str:
                    break
                k, v = [x.strip() for x in line_str.split(":", 1)]
                if v == "Active" and k != "Idle":
                    reasons.append(k)
        return ", ".join(reasons) if reasons else "None"
    except Exception:
        return "Unknown"


def query_system_telemetry() -> Dict[str, Any]:
    """Query per-core CPU utilization and free physical RAM."""
    cpu_cores = psutil.cpu_percent(percpu=True)
    cores_str = ";".join(f"{c:.1f}" for c in cpu_cores)
    free_ram_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
    return {
        "per_core_cpu_util": cores_str,
        "avg_cpu_util_pct": float(np.mean(cpu_cores)) if cpu_cores else 0.0,
        "free_ram_mb": free_ram_mb,
    }


def get_system_memory_snapshot() -> Dict[str, Any]:
    """Capture top 5 memory-using processes and Windows commit charge."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            mem = p.info.get("memory_info")
            if mem is not None:
                procs.append((p.info["pid"], p.info["name"], mem.rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x[2], reverse=True)
    top5 = [(pid, name, round(rss / (1024.0**2), 1)) for pid, name, rss in procs[:5]]
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    proc = psutil.Process()
    return {
        "top5": top5,
        "proc_rss_mb": round(proc.memory_info().rss / (1024.0**2), 1),
        "free_ram_mb": round(vm.available / (1024.0**2), 1),
        "total_ram_mb": round(vm.total / (1024.0**2), 1),
        "commit_used_mb": round(sm.used / (1024.0**2), 1),
        "commit_total_mb": round(sm.total / (1024.0**2), 1),
    }


class BenchmarkRunner:
    """Orchestrates 12-minute measurement runs across selfplay, train, and combined."""

    def __init__(
        self,
        mode: str,
        warmup_sec: float = 120.0,
        counted_sec: float = 600.0,
        sample_interval: float = 10.0,
        trials: int = 3,
        concurrency: int = 24,
        full_sims: int = 400,
        fast_sims: int = 80,
        full_search_prob: float = 0.35,
        tablebase_path: Optional[str] = "data/syzygy",
        tablebase_max_pieces: int = 5,
        allow_resignation: bool = False,
        output_dir: str = "docs/perf/baseline",
        checkpoint_path: str = "models/checkpoints_scaled/best_model.pt",
        human_data_dir: str = (
            "data/dataset_training/train"
            if Path("data/dataset_training/train").is_dir()
            else "data/dataset_scaled/train"
        ),
        batch_size: int = 256,
        base_seed: int = 42,
        stop_on_rss_6gb: bool = False,
        max_wait_ms: float = 1.5,
        pipelined: bool = False,
        watchdog_free_ram_mb: float = 2000.0,
    ) -> None:
        self.mode = mode
        self.warmup_sec = warmup_sec
        self.counted_sec = counted_sec
        self.sample_interval = sample_interval
        self.trials = trials
        self.stop_on_rss_6gb = stop_on_rss_6gb
        self.max_wait_ms = max_wait_ms
        self.pipelined = pipelined
        self.watchdog_free_ram_mb = float(watchdog_free_ram_mb)
        # Task 3(a): Concurrency cap of 24 unless free RAM > 3.0 GB
        if concurrency > 24:
            free_ram_gb = psutil.virtual_memory().available / (1024.0**3)
            if free_ram_gb < 3.0:
                logger.warning(
                    "Concurrency cap of 24 enforced (requested %d, free RAM %.2f GB < 3.0 GB)",
                    concurrency,
                    free_ram_gb,
                )
                concurrency = 24
        self.concurrency = concurrency
        self.full_sims = full_sims
        self.fast_sims = fast_sims
        self.full_search_prob = full_search_prob
        self.tablebase_path = tablebase_path
        self.tablebase_max_pieces = tablebase_max_pieces
        self.allow_resignation = allow_resignation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        self.human_data_dir = human_data_dir
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_trial(self, trial_idx: int) -> Dict[str, Any]:
        """Execute a single 12-minute trial (2 min warm-up + 10 min counted)."""
        logger.info(
            "=== Starting %s Trial %d/%d (Warm-up: %.0fs, Counted: %.0fs, Concurrency: %d) ===",
            self.mode.upper(),
            trial_idx,
            self.trials,
            self.warmup_sec,
            self.counted_sec,
            self.concurrency,
        )
        emit_report(
            event_line(
                "TRIAL",
                f"{self.mode} {trial_idx}/{self.trials} "
                f"warmup={self.warmup_sec:.0f}s counted={self.counted_sec:.0f}s "
                f"conc={self.concurrency} seed={self.base_seed + trial_idx * 1000}",
            )
        )

        trial_seed = self.base_seed + trial_idx * 1000
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)

        # Temporary buffer directory
        buf_dir = self.output_dir / f"tmp_buf_{self.mode}_{trial_idx}"
        if buf_dir.exists():
            shutil.rmtree(buf_dir, ignore_errors=True)
        buf_dir.mkdir(parents=True, exist_ok=True)

        buffer_cap = 20_000 if self.mode == "selfplay" else 100_000
        buffer = ReplayBuffer(
            buffer_dir=buf_dir,
            capacity=buffer_cap,
            initial_human_mix=0.25,
            final_human_mix=0.05,
            taper_steps=50_000,
            human_data_dir=self.human_data_dir if Path(self.human_data_dir).exists() else None,
            gpu_device=self.device,
            create_new=True,
        )

        # Initialize Inference Server & Client
        server: Optional[InferenceServer] = None
        worker: Optional[SelfPlayWorker] = None
        trainer: Optional[RLTrainer] = None
        trainer_thread: Optional[threading.Thread] = None
        stop_event = threading.Event()

        positions_generated = [0]
        games_completed = [0]

        if self.mode in ("selfplay", "combined"):
            server = InferenceServer(
                checkpoint_path=self.checkpoint_path
                if Path(self.checkpoint_path).exists()
                else None,
                device=self.device,
                min_batch_size=min(48, self.concurrency * 2),
                max_batch_size=128,
                max_wait_ms=self.max_wait_ms,
                use_bfloat16=True,
            )
            client = DirectInferenceClient(server)

            def sink(positions: List[SavedPosition], summary: GameSummary):
                buffer.add_positions(positions)
                positions_generated[0] += len(positions)
                games_completed[0] += 1

            worker = SelfPlayWorker(
                worker_id=trial_idx,
                client=client,
                num_concurrent_games=self.concurrency,
                game_type="chess",
                full_search_sims=self.full_sims,
                fast_search_sims=self.fast_sims,
                full_search_prob=self.full_search_prob,
                temp_cutoff_range=(15, 30),
                resign_threshold=-0.95,
                consecutive_resign_moves=4,
                disable_resign_prob=1.0,
                allow_resignation=self.allow_resignation,
                min_resign_ply=30,
                tablebase_path=self.tablebase_path,
                tablebase_max_pieces=self.tablebase_max_pieces,
                position_sink=sink,
                max_wait_ms=self.max_wait_ms,
                pipelined=self.pipelined,
            )

        if self.mode in ("train", "combined"):
            trainer_model = ChessNet(num_blocks=12, num_filters=128).to(self.device)
            if server is not None:
                trainer_model.load_state_dict(server.model.state_dict())
            elif Path(self.checkpoint_path).exists():
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
                trainer_model.load_state_dict(sd)

            trainer = RLTrainer(
                model=trainer_model,
                replay_buffer=buffer,
                device=self.device,
                base_lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                warmup_steps=500,
                total_steps=100_000,
                batch_size=self.batch_size,
                push_freq=200,
                checkpoint_interval_sec=3600.0,
                checkpoint_dir=buf_dir / "checkpoints",
                direct_server=server,
                use_bfloat16=True,
            )

            def train_loop():
                while not stop_event.is_set():
                    try:
                        trainer.train_step()
                    except Exception as e:
                        logger.warning("Train step exception: %s", e)
                        time.sleep(0.01)

            trainer_thread = threading.Thread(target=train_loop, name="trainer_thread", daemon=True)

        mem_start = get_system_memory_snapshot()
        logger.info(
            "=== MEMORY AT START: RSS=%.1f MB, Free RAM=%.1f MB, Commit Used=%.1f MB / %.1f MB ===",
            mem_start["proc_rss_mb"],
            mem_start["free_ram_mb"],
            mem_start["commit_used_mb"],
            mem_start["commit_total_mb"],
        )
        logger.info("Top 5 Processes at start: %s", mem_start["top5"])
        threads_start = {t.id: (t.user_time, t.system_time) for t in psutil.Process().threads()}

        # Launch workers / trainers
        if worker is not None:
            worker.start()
        if trainer_thread is not None:
            trainer_thread.start()

        # CSV Logging Setup
        csv_path = self.output_dir / f"{self.mode}_trial_{trial_idx}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp_sec",
            "phase",
            "sims_per_sec",
            "positions_per_hour",
            "games_per_hour",
            "avg_batch_size",
            "batch_fill_rate",
            "inference_latency_ms",
            "queue_wait_time_ms",
            "gpu_util_pct",
            "gpu_clock_mhz",
            "mem_clock_mhz",
            "power_w",
            "temp_c",
            "throttle_reasons",
            "avg_cpu_util_pct",
            "per_core_cpu_util",
            "free_ram_mb",
            "vram_used_mb",
            "training_steps_per_sec",
            "training_samples_per_sec",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        trial_start = time.time()
        counted_start: Optional[float] = None
        phase = "warmup"

        prev_time = trial_start
        prev_sims = 0
        prev_positions = 0
        prev_games = 0
        prev_train_steps = 0

        counted_rows: List[Dict[str, Any]] = []

        try:
            while True:
                now = time.time()
                elapsed_total = now - trial_start

                if phase == "warmup" and elapsed_total >= self.warmup_sec:
                    phase = "counted"
                    counted_start = now
                    logger.info(
                        ">>> Warm-up complete. Starting COUNTED measurement phase (%.0fs) <<<",
                        self.counted_sec,
                    )
                    emit_report(
                        event_line("COUNTED", f"warm-up done; measuring {self.counted_sec:.0f}s")
                    )
                    # Reset window counters
                    if server is not None:
                        server.get_window_stats()
                    if worker is not None:
                        worker.aggregator.get_window_queue_wait_ms()
                        prev_sims = worker.live_simulations
                    prev_positions = positions_generated[0]
                    prev_games = games_completed[0]
                    if trainer is not None:
                        prev_train_steps = trainer.step_num
                    prev_time = now

                if (
                    phase == "counted"
                    and counted_start is not None
                    and (now - counted_start) >= self.counted_sec
                ):
                    break

                time.sleep(self.sample_interval)

                now_sample = time.time()
                dt = max(1e-3, now_sample - prev_time)
                prev_time = now_sample

                # 1. Self-play counters
                cur_sims = worker.live_simulations if worker is not None else 0
                sims_per_sec = (cur_sims - prev_sims) / dt
                prev_sims = cur_sims

                cur_positions = positions_generated[0]
                positions_per_hour = ((cur_positions - prev_positions) / dt) * 3600.0
                prev_positions = cur_positions

                cur_games = games_completed[0]
                games_per_hour = ((cur_games - prev_games) / dt) * 3600.0
                prev_games = cur_games

                # 2. Inference stats
                avg_batch_size = 0.0
                inference_latency_ms = 0.0
                batch_fill_rate = 0.0
                if server is not None:
                    avg_batch_size, inference_latency_ms, batch_fill_rate = (
                        server.get_window_stats()
                    )

                # 3. Queue wait time
                queue_wait_time_ms = 0.0
                if worker is not None:
                    queue_wait_time_ms = worker.aggregator.get_window_queue_wait_ms()

                # 4. Training stats
                train_steps_sec = 0.0
                train_samples_sec = 0.0
                if trainer is not None:
                    cur_train_steps = trainer.step_num
                    train_steps_sec = (cur_train_steps - prev_train_steps) / dt
                    train_samples_sec = train_steps_sec * self.batch_size
                    prev_train_steps = cur_train_steps

                # 5. Hardware queries
                gpu_data = query_gpu_telemetry()
                throttle_str = query_throttle_reasons()
                sys_data = query_system_telemetry()

                # MANDATORY SAFETY WATCHDOG: abort if RSS > 6 GB or Free RAM < 2 GB
                proc_rss_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
                free_ram_mb = sys_data["free_ram_mb"]
                if proc_rss_mb > 6000.0:
                    logger.critical(
                        "WATCHDOG TRIGGERED: Process RSS=%.1f MB exceeded 6.0 GB limit (6000 MB)! Halting benchmark immediately.",
                        proc_rss_mb,
                    )
                    emit_report(
                        event_line("WATCHDOG", f"RSS {proc_rss_mb:.0f}MB > 6000MB; halting")
                    )
                    break
                if free_ram_mb < self.watchdog_free_ram_mb:
                    logger.critical(
                        "WATCHDOG TRIGGERED: System Free RAM=%.1f MB dropped below 2.0 GB safety floor (2000 MB)! Halting benchmark immediately.",
                        free_ram_mb,
                    )
                    emit_report(
                        event_line("WATCHDOG", f"free RAM {free_ram_mb:.0f}MB < 2000MB; halting")
                    )
                    break

                elapsed_phase = (
                    (now_sample - counted_start) if counted_start is not None else elapsed_total
                )
                row = {
                    "timestamp_sec": round(elapsed_phase, 1),
                    "phase": phase,
                    "sims_per_sec": round(sims_per_sec, 1),
                    "positions_per_hour": round(positions_per_hour, 1),
                    "games_per_hour": round(games_per_hour, 1),
                    "avg_batch_size": round(avg_batch_size, 2),
                    "batch_fill_rate": round(batch_fill_rate, 4),
                    "inference_latency_ms": round(inference_latency_ms, 2),
                    "queue_wait_time_ms": round(queue_wait_time_ms, 2),
                    "gpu_util_pct": round(gpu_data["gpu_util_pct"], 1),
                    "gpu_clock_mhz": round(gpu_data["gpu_clock_mhz"], 0),
                    "mem_clock_mhz": round(gpu_data["mem_clock_mhz"], 0),
                    "power_w": round(gpu_data["power_w"], 1),
                    "temp_c": round(gpu_data["temp_c"], 1),
                    "throttle_reasons": throttle_str,
                    "avg_cpu_util_pct": round(sys_data["avg_cpu_util_pct"], 1),
                    "per_core_cpu_util": sys_data["per_core_cpu_util"],
                    "free_ram_mb": round(sys_data["free_ram_mb"], 1),
                    "vram_used_mb": round(gpu_data["vram_used_mb"], 1),
                    "training_steps_per_sec": round(train_steps_sec, 2),
                    "training_samples_per_sec": round(train_samples_sec, 1),
                }

                csv_writer.writerow(row)
                csv_file.flush()

                if phase == "counted":
                    counted_rows.append(row)
                    logger.info(
                        "[%s T%d] [%.0fs/%.0fs] Sims/s: %6.1f | Pos/hr: %7.1f | Games/hr: %5.1f | GPU: %4.1f%% (%3.0fW, %2.0fC) | Train steps/s: %4.1f",
                        self.mode.upper(),
                        trial_idx,
                        elapsed_phase,
                        self.counted_sec,
                        sims_per_sec,
                        positions_per_hour,
                        games_per_hour,
                        gpu_data["gpu_util_pct"],
                        gpu_data["power_w"],
                        gpu_data["temp_c"],
                        train_steps_sec,
                    )
                    emit_report(
                        format_kv_line(
                            [
                                ("trial", f"{trial_idx}/{self.trials}"),
                                ("t", f"{elapsed_phase:.0f}/{self.counted_sec:.0f}s"),
                                ("sims_sec", f"{sims_per_sec:.1f}"),
                                ("pos_hr", f"{positions_per_hour:.1f}"),
                                ("games_hr", f"{games_per_hour:.1f}"),
                                ("gpu", f"{gpu_data['gpu_util_pct']:.1f}%"),
                                ("batch_avg", f"{avg_batch_size:.2f}"),
                                ("lat_ms", f"{inference_latency_ms:.2f}"),
                                ("train_steps_sec", f"{train_steps_sec:.2f}"),
                            ]
                        )
                    )
                else:
                    logger.info(
                        "[%s T%d WARMUP] [%.0fs/%.0fs] Sims/s: %6.1f | GPU: %4.1f%% (%3.0fW) | Train steps/s: %4.1f",
                        self.mode.upper(),
                        trial_idx,
                        elapsed_total,
                        self.warmup_sec,
                        sims_per_sec,
                        gpu_data["gpu_util_pct"],
                        gpu_data["power_w"],
                        train_steps_sec,
                    )
                    emit_report(
                        format_kv_line(
                            [
                                ("trial", f"{trial_idx}/{self.trials}"),
                                ("phase", "warmup"),
                                ("t", f"{elapsed_total:.0f}/{self.warmup_sec:.0f}s"),
                                ("sims_sec", f"{sims_per_sec:.1f}"),
                                ("gpu", f"{gpu_data['gpu_util_pct']:.1f}%"),
                            ]
                        )
                    )

        finally:
            stop_event.set()
            if worker is not None:
                worker.stop()
            if trainer_thread is not None:
                trainer_thread.join(timeout=5)
            csv_file.close()
            shutil.rmtree(buf_dir, ignore_errors=True)

        mem_end = get_system_memory_snapshot()
        threads_end = {t.id: (t.user_time, t.system_time) for t in psutil.Process().threads()}
        logger.info(
            "=== MEMORY AT END: RSS=%.1f MB, Free RAM=%.1f MB, Commit Used=%.1f MB / %.1f MB ===",
            mem_end["proc_rss_mb"],
            mem_end["free_ram_mb"],
            mem_end["commit_used_mb"],
            mem_end["commit_total_mb"],
        )
        logger.info("Top 5 Processes at end: %s", mem_end["top5"])
        delta_rss = mem_end["proc_rss_mb"] - mem_start["proc_rss_mb"]
        delta_free_ram = mem_start["free_ram_mb"] - mem_end["free_ram_mb"]
        logger.info(
            "MEMORY GAP: Process RSS increase = %.1f MB, System Free RAM drop = %.1f MB (Gap = %.1f MB)",
            delta_rss,
            delta_free_ram,
            delta_free_ram - delta_rss,
        )

        thread_cpu_summary = []
        for tid, (u1, s1) in threads_end.items():
            u0, s0 = threads_start.get(tid, (0.0, 0.0))
            du = max(0.0, u1 - u0)
            ds = max(0.0, s1 - s0)
            if (du + ds) > 0.05:
                thread_cpu_summary.append(
                    {
                        "tid": tid,
                        "user_sec": round(du, 2),
                        "sys_sec": round(ds, 2),
                        "total_sec": round(du + ds, 2),
                    }
                )
        thread_cpu_summary.sort(key=lambda x: x["total_sec"], reverse=True)

        gil_summary = {}
        if worker is not None and hasattr(worker, "aggregator"):
            agg = worker.aggregator
            t_eval = agg.total_gil_eval_sec
            t_dispatch = agg.total_gil_dispatch_sec
            t_wait = agg.total_gil_wait_sec
            t_gil = t_eval + t_dispatch
            wall_sec = max(1e-6, time.time() - trial_start)
            gil_fraction = (t_gil / wall_sec) * 100.0
            total_flushes = agg.flush_timeout_count + agg.flush_capacity_count
            timeout_share = (agg.flush_timeout_count / max(1, total_flushes)) * 100.0
            gil_summary = {
                "gil_eval_sec": round(t_eval, 3),
                "gil_dispatch_sec": round(t_dispatch, 3),
                "gil_wait_sec": round(t_wait, 3),
                "total_gil_sec": round(t_gil, 3),
                "wall_sec": round(wall_sec, 2),
                "gil_held_pct": round(gil_fraction, 2),
                "flush_timeout_count": agg.flush_timeout_count,
                "flush_capacity_count": agg.flush_capacity_count,
                "timeout_share_pct": round(timeout_share, 2),
                "total_eval_calls": agg.total_eval_calls,
            }
            logger.info(
                "GIL ESTIMATE: Held %.2f%% of wall clock time (Eval: %.2fs, Dispatch: %.2fs, Wait: %.2fs)",
                gil_fraction,
                t_eval,
                t_dispatch,
                t_wait,
            )
            logger.info(
                "BATCH FLUSHES: %d timeouts (%.1f%%), %d capacity",
                agg.flush_timeout_count,
                timeout_share,
                agg.flush_capacity_count,
            )

        logger.info(
            "Trial %d completed. %d counted rows logged to %s",
            trial_idx,
            len(counted_rows),
            csv_path,
        )
        emit_report(
            event_line(
                "TRIAL_END",
                f"trial {trial_idx} done; {len(counted_rows)} counted rows in {csv_path}",
            )
        )
        summary = self._summarize_rows(counted_rows)
        summary["mem_start"] = mem_start
        summary["mem_end"] = mem_end
        summary["gil_summary"] = gil_summary
        summary["thread_cpu_summary"] = thread_cpu_summary
        return summary

    @staticmethod
    def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute mean of numeric columns over counted rows."""
        if not rows:
            return {}
        summary = {}
        for key in rows[0]:
            if key in ("timestamp_sec", "phase", "throttle_reasons", "per_core_cpu_util"):
                continue
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            summary[key] = float(np.mean(vals)) if vals else 0.0
        return summary

    def run_benchmark(self) -> Dict[str, Any]:
        """Run all trials and report run-to-run statistics and variance."""
        trial_summaries = []
        for t in range(1, self.trials + 1):
            s = self.run_trial(t)
            trial_summaries.append(s)

        # Aggregate across trials
        keys = list(trial_summaries[0].keys()) if trial_summaries else []
        aggregated: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = [ts[k] for ts in trial_summaries if k in ts and isinstance(ts[k], (int, float))]
            if not vals:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            variance_val = float(np.var(vals))
            cv_pct = float((std_val / mean_val) * 100.0) if abs(mean_val) > 1e-6 else 0.0
            aggregated[k] = {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "variance": round(variance_val, 4),
                "cv_pct": round(cv_pct, 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
            }

        result = {
            "mode": self.mode,
            "trials": self.trials,
            "warmup_sec": self.warmup_sec,
            "counted_sec": self.counted_sec,
            "concurrency": self.concurrency,
            "full_sims": self.full_sims,
            "fast_sims": self.fast_sims,
            "full_search_prob": self.full_search_prob,
            "trial_summaries": trial_summaries,
            "aggregated": aggregated,
            "timestamp": datetime.now().isoformat(),
        }

        summary_json_path = self.output_dir / f"{self.mode}_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        self._print_summary_table(result)
        return result

    def _print_summary_table(self, result: Dict[str, Any]) -> None:
        agg = result["aggregated"]
        print("\n" + "=" * 90)
        print(
            f"BENCHMARK SUMMARY: {self.mode.upper()} ({self.trials} Trials, {self.counted_sec}s counted each)"
        )
        print("=" * 90)
        print(
            f"{'Metric':<28} | {'Mean':>12} | {'Std Dev':>10} | {'CV (%)':>8} | {'Range [Min, Max]':>22}"
        )
        print("-" * 90)
        key_order = [
            "sims_per_sec",
            "positions_per_hour",
            "games_per_hour",
            "training_steps_per_sec",
            "training_samples_per_sec",
            "avg_batch_size",
            "batch_fill_rate",
            "inference_latency_ms",
            "queue_wait_time_ms",
            "gpu_util_pct",
            "power_w",
            "temp_c",
            "gpu_clock_mhz",
            "mem_clock_mhz",
            "avg_cpu_util_pct",
            "free_ram_mb",
            "vram_used_mb",
        ]
        for k in key_order:
            if k in agg:
                stats = agg[k]
                range_str = f"[{stats['min']:.1f}, {stats['max']:.1f}]"
                print(
                    f"{k:<28} | {stats['mean']:>12.2f} | {stats['std']:>10.2f} | {stats['cv_pct']:>7.2f}% | {range_str:>22}"
                )
        print("=" * 90 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 Benchmark Protocol")
    parser.add_argument(
        "--mode",
        choices=["selfplay", "train", "combined", "all"],
        default="combined",
        help="Benchmark workload mode",
    )
    parser.add_argument(
        "--warmup-sec", type=float, default=120.0, help="Warm-up seconds (uncounted)"
    )
    parser.add_argument(
        "--counted-sec", type=float, default=600.0, help="Counted seconds per trial"
    )
    parser.add_argument(
        "--sample-interval", type=float, default=10.0, help="Sampling interval in seconds"
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of repeat trials")
    parser.add_argument("--concurrency", type=int, default=24, help="Worker concurrency")
    parser.add_argument("--full-sims", type=int, default=400, help="Full search simulations")
    parser.add_argument("--fast-sims", type=int, default=80, help="Fast search simulations")
    parser.add_argument(
        "--full-search-prob", type=float, default=0.35, help="Full search probability"
    )
    parser.add_argument(
        "--tablebase-path", type=str, default="data/syzygy", help="Tablebase directory"
    )
    parser.add_argument("--tablebase-max-pieces", type=int, default=5, help="Tablebase max pieces")
    parser.add_argument(
        "--allow-resignation", action="store_true", default=False, help="Allow resignation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/perf/baseline", help="Directory for raw data"
    )
    default_human_dir = (
        "data/dataset_training/train"
        if Path("data/dataset_training/train").is_dir()
        else "data/dataset_scaled/train"
    )
    parser.add_argument("--human-data-dir", type=str, default=default_human_dir)
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        type=str,
        default="models/checkpoints_scaled/best_model.pt",
        help="Path to checkpoint weights",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--stop-on-rss-6gb",
        action="store_true",
        default=False,
        help="Halt trial if process RSS exceeds 6 GB",
    )
    parser.add_argument(
        "--max-wait-ms", type=float, default=1.5, help="Dispatcher batch aggregation timeout in ms"
    )
    parser.add_argument(
        "--pipelined",
        action="store_true",
        default=False,
        help="Enable 3-stage pipelined inference dispatcher",
    )
    return parser.parse_args()


def main():
    setup_logging(log_dir="logs", log_filename="bench_selfplay.log")
    args = parse_args()
    ctx = run_context("benchmark")
    t0 = time.time()
    log_path = "logs/bench_selfplay.log"
    emit_report(banner("benchmark", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            str(args.checkpoint_path),
            "auto",
            str(args.base_seed),
        )
    )
    emit_report(
        parameters_table(
            [
                ("mode", args.mode, "user" if args.mode != "combined" else "default"),
                ("warmup_sec", args.warmup_sec, "user" if args.warmup_sec != 120.0 else "default"),
                (
                    "counted_sec",
                    args.counted_sec,
                    "user" if args.counted_sec != 600.0 else "default",
                ),
                (
                    "sample_interval",
                    args.sample_interval,
                    "user" if args.sample_interval != 10.0 else "default",
                ),
                ("trials", args.trials, "user" if args.trials != 3 else "default"),
                ("concurrency", args.concurrency, "user" if args.concurrency != 24 else "default"),
                ("full_sims", args.full_sims, "user" if args.full_sims != 400 else "default"),
                ("fast_sims", args.fast_sims, "user" if args.fast_sims != 80 else "default"),
                (
                    "full_search_prob",
                    args.full_search_prob,
                    "user" if args.full_search_prob != 0.35 else "default",
                ),
                (
                    "tablebase_path",
                    args.tablebase_path,
                    "user" if args.tablebase_path != "data/syzygy" else "default",
                ),
                (
                    "tablebase_max_pieces",
                    args.tablebase_max_pieces,
                    "user" if args.tablebase_max_pieces != 5 else "default",
                ),
                (
                    "allow_resignation",
                    args.allow_resignation,
                    "user" if args.allow_resignation else "default",
                ),
                (
                    "output_dir",
                    args.output_dir,
                    "user" if args.output_dir != "docs/perf/baseline" else "default",
                ),
                (
                    "checkpoint_path",
                    args.checkpoint_path,
                    "user"
                    if args.checkpoint_path != "models/checkpoints_scaled/best_model.pt"
                    else "default",
                ),
                (
                    "human_data_dir",
                    args.human_data_dir,
                    "user" if args.human_data_dir != "data/dataset_scaled/train" else "default",
                ),
                ("batch_size", args.batch_size, "user" if args.batch_size != 256 else "default"),
                ("base_seed", args.base_seed, "user" if args.base_seed != 42 else "default"),
                (
                    "stop_on_rss_6gb",
                    args.stop_on_rss_6gb,
                    "user" if args.stop_on_rss_6gb else "default",
                ),
                ("max_wait_ms", args.max_wait_ms, "user" if args.max_wait_ms != 1.5 else "default"),
                ("pipelined", args.pipelined, "user" if args.pipelined else "default"),
            ]
        )
    )
    modes = ["selfplay", "train", "combined"] if args.mode == "all" else [args.mode]
    all_results = {}
    try:
        for m in modes:
            runner = BenchmarkRunner(
                mode=m,
                warmup_sec=args.warmup_sec,
                counted_sec=args.counted_sec,
                sample_interval=args.sample_interval,
                trials=args.trials,
                concurrency=args.concurrency,
                full_sims=args.full_sims,
                fast_sims=args.fast_sims,
                full_search_prob=args.full_search_prob,
                tablebase_path=args.tablebase_path,
                tablebase_max_pieces=args.tablebase_max_pieces,
                allow_resignation=args.allow_resignation,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint_path,
                human_data_dir=args.human_data_dir,
                batch_size=args.batch_size,
                base_seed=args.base_seed,
                stop_on_rss_6gb=args.stop_on_rss_6gb,
                max_wait_ms=args.max_wait_ms,
                pipelined=args.pipelined,
            )
            all_results[m] = runner.run_benchmark()

        # Save master baseline report
        report_path = Path(args.output_dir) / "BASELINE_REPORT.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Task 1 Baseline Measurement Protocol Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Trials per mode:** {args.trials}\n")
            f.write(
                f"**Duration per trial:** {args.warmup_sec:.0f}s warm-up + {args.counted_sec:.0f}s counted (12 minutes total)\n"
            )
            f.write(
                f"**Settings:** Concurrency={args.concurrency}, Sims={args.full_sims}/{args.fast_sims}, Full-search={args.full_search_prob}, Syzygy=on (5pc), Resignation=off\n\n"
            )
            for m, res in all_results.items():
                f.write(f"## Mode: {m.upper()}\n\n")
                f.write("| Metric | Mean | Std Dev | CV (%) | Min | Max |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                for k, stats in res["aggregated"].items():
                    f.write(
                        f"| `{k}` | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['cv_pct']:.2f}% | {stats['min']:.2f} | {stats['max']:.2f} |\n"
                    )
                f.write("\n")
        logger.info("Master baseline report written to %s", report_path)
        emit_report(event_line("REPORT", f"master baseline report in {report_path}"))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"benchmark failed: {e}",
                "OOM/watchdog trip, bad checkpoint, or missing dataset",
                "check free RAM, --checkpoint-path, --human-data-dir, then see LOG",
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
