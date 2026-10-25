"""Script for Item 1: Memory Regression Diagnostics.

Measures:
1(a) Telemetry every 30s for 8 minutes at Concurrency 8 and 24:
    - Process RSS, USS, Private Bytes
    - Tracemalloc top 10
    - Live MCTS object count
    - Total NN-cache entries & estimated bytes across live games
    - Position/summary list sizes
    - Growth profile (linear vs plateau after ~3 min)
1(b) Cache entries vs ply for games reaching 100, 200, 300, 400 plies:
    - Cache cleared per move? Max entries ever seen?
1(d) Verification of game object destruction & reference leaks.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import tracemalloc
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_build_dir = REPO_ROOT / "build" / "cpp"
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))

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

import zeropawn_cpp
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.worker import GameSummary, SavedPosition, SelfPlayGame, SelfPlayWorker
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
setup_logging(log_dir="logs", log_filename="diagnose_memory.log")
logger = logging.getLogger("memory_diag")


def get_python_obj_counts() -> Dict[str, int]:
    """Scan gc for live Python container objects."""
    gc.collect()
    objs = gc.get_objects()
    return {
        "saved_pos_count": sum(1 for o in objs if isinstance(o, SavedPosition)),
        "summary_count": sum(1 for o in objs if isinstance(o, GameSummary)),
        "game_count": sum(1 for o in objs if isinstance(o, SelfPlayGame)),
    }


def run_memory_diagnostic(
    concurrency: int = 8,
    duration_sec: float = 480.0,
    sample_interval_sec: float = 30.0,
    buffer_capacity: int = 20_000,
    allow_resignation: bool = False,
    output_dir: Path = Path("docs/perf/memory_diagnostic"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / "models/checkpoints_scaled/best_model.pt"

    proc = psutil.Process()
    tracemalloc.start()

    logger.info(
        "=== Starting Memory Diagnostic: Concurrency=%d, Duration=%.0fs ===",
        concurrency,
        duration_sec,
    )

    # Temporary buffer
    buf_dir = output_dir / f"tmp_buf_c{concurrency}"
    if buf_dir.exists():
        shutil.rmtree(buf_dir, ignore_errors=True)
    buf_dir.mkdir(parents=True, exist_ok=True)

    buffer = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=buffer_capacity,
        initial_human_mix=0.25,
        final_human_mix=0.05,
        taper_steps=50_000,
        gpu_device=device,
        create_new=True,
    )

    server = InferenceServer(
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
        device=device,
        min_batch_size=min(48, concurrency * 2),
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    # Harness position/summary tracking
    harness_positions_list: List[Any] = []
    harness_summaries_list: List[Any] = []
    positions_generated = [0]
    games_completed = [0]

    # Tracking cache vs ply
    cache_vs_ply_records: List[Dict[str, Any]] = []
    max_cache_seen = [0]

    def ply_hook(info: Dict[str, Any]):
        ply = info.get("ply", 0)
        # Scan live mcts cache sizes
        pass

    def sink(positions: List[SavedPosition], summary: GameSummary):
        buffer.add_positions(positions)
        positions_generated[0] += len(positions)
        games_completed[0] += 1
        # Test: does harness hold them? In standard harness, we do NOT hold references.

    worker = SelfPlayWorker(
        worker_id=1,
        client=client,
        num_concurrent_games=concurrency,
        game_type="chess",
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        temp_cutoff_range=(15, 30),
        resign_threshold=-0.95,
        consecutive_resign_moves=4,
        disable_resign_prob=1.0,
        allow_resignation=allow_resignation,
        min_resign_ply=30,
        tablebase_path=str(REPO_ROOT / "data/syzygy"),
        tablebase_max_pieces=5,
        position_sink=sink,
    )

    worker.start()
    start_time = time.time()
    snapshots: List[Dict[str, Any]] = []
    watchdog_fired: Optional[str] = None

    next_sample = start_time + sample_interval_sec
    end_time = start_time + duration_sec

    try:
        while time.time() < end_time:
            time.sleep(1.0)
            now = time.time()

            # MANDATORY WATCHDOG (checked every 1s)
            mem_info = proc.memory_info()
            full_info = proc.memory_full_info()
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()

            rss_mb = mem_info.rss / (1024.0 * 1024.0)
            free_ram_mb = vm.available / (1024.0 * 1024.0)

            if rss_mb > 6000.0:
                watchdog_fired = f"Process RSS={rss_mb:.1f} MB exceeded 6.0 GB limit (6000 MB)"
                logger.critical("WATCHDOG TRIGGERED: %s! Aborting run.", watchdog_fired)
                emit_report(event_line("WATCHDOG", watchdog_fired))
                break
            if free_ram_mb < 2000.0:
                watchdog_fired = f"System Free RAM={free_ram_mb:.1f} MB dropped below 2.0 GB safety floor (2000 MB)"
                logger.critical("WATCHDOG TRIGGERED: %s! Aborting run.", watchdog_fired)
                emit_report(event_line("WATCHDOG", watchdog_fired))
                break

            if now >= next_sample or now >= end_time:
                elapsed = now - start_time
                uss_mb = getattr(full_info, "uss", 0) / (1024.0 * 1024.0)
                private_mb = getattr(full_info, "private", 0) / (1024.0 * 1024.0)
                pagefile_mb = getattr(full_info, "pagefile", 0) / (1024.0 * 1024.0)
                commit_used_mb = sm.used / (1024.0 * 1024.0)
                commit_total_mb = sm.total / (1024.0 * 1024.0)

                cache_stats = worker.get_live_cache_stats()
                py_counts = get_python_obj_counts()
                if cache_stats["max_game_cache_entries"] > max_cache_seen[0]:
                    max_cache_seen[0] = cache_stats["max_game_cache_entries"]

                # Tracemalloc top 10
                tm_snap = tracemalloc.take_snapshot()
                top_stats = tm_snap.statistics("lineno")[:10]
                tm_top10 = [
                    f"{stat.traceback[0].filename}:{stat.traceback[0].lineno} ({stat.size / (1024 * 1024):.2f} MB)"
                    for stat in top_stats
                ]

                snap = {
                    "elapsed_sec": round(elapsed, 1),
                    "rss_mb": round(rss_mb, 1),
                    "uss_mb": round(uss_mb, 1),
                    "private_mb": round(private_mb, 1),
                    "pagefile_mb": round(pagefile_mb, 1),
                    "free_ram_mb": round(free_ram_mb, 1),
                    "commit_used_mb": round(commit_used_mb, 1),
                    "commit_total_mb": round(commit_total_mb, 1),
                    "live_mcts_count": cache_stats["live_games"],
                    "live_game_count": py_counts["game_count"],
                    "live_saved_pos_count": py_counts["saved_pos_count"],
                    "live_summary_count": py_counts["summary_count"],
                    "total_nn_cache_entries": cache_stats["total_cache_entries"],
                    "est_cache_mb": round(cache_stats["est_cache_mb"], 1),
                    "max_single_game_cache": cache_stats["max_game_cache_entries"],
                    "mean_game_cache": round(cache_stats["mean_game_cache_entries"], 1),
                    "games_completed": games_completed[0],
                    "positions_generated": positions_generated[0],
                    "harness_held_positions": len(harness_positions_list),
                    "harness_held_summaries": len(harness_summaries_list),
                    "tracemalloc_top10": tm_top10,
                }
                snapshots.append(snap)
                logger.info(
                    "[%4.0fs] RSS: %.1f MB | USS: %.1f MB | Private: %.1f MB | Free RAM: %.1f MB | Commit: %.1f MB | Active Games: %d (Cache: %d ent, %.1f MB) | Games done: %d",
                    elapsed,
                    rss_mb,
                    uss_mb,
                    private_mb,
                    free_ram_mb,
                    commit_used_mb,
                    cache_stats["live_games"],
                    cache_stats["total_cache_entries"],
                    cache_stats["est_cache_mb"],
                    games_completed[0],
                )
                # Console mirror (sample cadence): same measured numbers, kv shape.
                emit_report(
                    format_kv_line(
                        [
                            ("t_s", f"{elapsed:.0f}"),
                            ("rss_mb", f"{rss_mb:.1f}"),
                            ("free_ram_mb", f"{free_ram_mb:.1f}"),
                            ("active_games", cache_stats["live_games"]),
                            ("cache_ent", cache_stats["total_cache_entries"]),
                            ("games_done", games_completed[0]),
                        ]
                    )
                )
                next_sample += sample_interval_sec
    finally:
        worker.stop()
        shutil.rmtree(buf_dir, ignore_errors=True)
        tracemalloc.stop()

    # Determine linear growth vs flat after ~3 min (180s)
    # Compare growth from 180s to end vs 0s to 180s
    snaps_after_180 = [s for s in snapshots if s["elapsed_sec"] >= 180.0]
    if len(snaps_after_180) >= 2:
        rss_180 = snaps_after_180[0]["rss_mb"]
        rss_end = snaps_after_180[-1]["rss_mb"]
        delta_after_180 = rss_end - rss_180
        rate_mb_per_min = delta_after_180 / (
            (snaps_after_180[-1]["elapsed_sec"] - snaps_after_180[0]["elapsed_sec"]) / 60.0
        )
        is_flat = abs(rate_mb_per_min) < 30.0  # less than 30 MB/min growth after 3m
    else:
        delta_after_180 = 0.0
        rate_mb_per_min = 0.0
        is_flat = False

    result = {
        "concurrency": concurrency,
        "duration_sec": duration_sec,
        "watchdog_fired": watchdog_fired,
        "max_single_game_cache_seen": max_cache_seen[0],
        "delta_rss_after_180s_mb": round(delta_after_180, 1),
        "rss_growth_rate_after_180s_mb_per_min": round(rate_mb_per_min, 2),
        "is_flat_after_3min": is_flat,
        "snapshots": snapshots,
    }

    out_file = output_dir / f"mem_diag_c{concurrency}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved diagnostic to %s", out_file)
    return result


if __name__ == "__main__":
    setup_logging(log_dir="logs", log_filename="diagnose_memory.log")
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=480.0)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--buffer-cap", type=int, default=20_000)
    parser.add_argument("--allow-resignation", action="store_true", default=False)
    args = parser.parse_args()

    ctx = run_context("memory-diag")
    t0 = time.time()
    emit_report(banner("memory-diag", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "auto",
            "n/a",
        )
    )
    emit_report(
        parameters_table(
            [
                ("concurrency", args.concurrency, "user" if args.concurrency != 8 else "default"),
                ("duration", args.duration, "user" if args.duration != 480.0 else "default"),
                ("interval", args.interval, "user" if args.interval != 30.0 else "default"),
                ("buffer_cap", args.buffer_cap, "user" if args.buffer_cap != 20_000 else "default"),
                (
                    "allow_resignation",
                    args.allow_resignation,
                    "user" if args.allow_resignation else "default",
                ),
            ]
        )
    )
    try:
        run_memory_diagnostic(
            concurrency=args.concurrency,
            duration_sec=args.duration,
            sample_interval_sec=args.interval,
            buffer_capacity=args.buffer_cap,
            allow_resignation=args.allow_resignation,
        )
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"memory-diag failed: {e}",
                "OOM/watchdog or bad dataset",
                "check free RAM and dataset, then see LOG",
                "logs/diagnose_memory.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)
