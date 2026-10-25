"""Status, telemetry, checkpoints, and hardware diagnostics for ZeroPawn."""

from __future__ import annotations

import argparse
import csv
import datetime
import shutil
import subprocess

import psutil
import torch

from python.cli.common import (
    get_available_checkpoints,
    get_default_stockfish_path,
    safe_resolve,
)
from python.common.progress import (
    banner,
    emit_report,
    format_kv_line,
    header_block,
    results_table,
    run_context,
    summary_block,
)
import logging

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "status",
        aliases=["info", "stat"],
        help="View system health, checkpoints, training telemetry, and hardware info.",
        description="Display active model checkpoints, historical evaluation metrics, replay buffer stats, and hardware status.",
    )
    parser.set_defaults(func=run_status)


def run_status(args: argparse.Namespace) -> int:
    import time

    ctx = run_context("status")
    t0 = time.time()
    emit_report(banner("status", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none (inventory below)",
            "auto",
            "n/a",
        )
    )

    # 1. Hardware & Runtime Diagnostics (same values, ASCII layout)
    hw_rows = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_alloc_gb = torch.cuda.memory_allocated(0) / (1024**3)
        hw_rows.append(("gpu", f"{gpu_name} (M)"))
        hw_rows.append(
            (
                "vram_process",
                f"{vram_alloc_gb:.2f} GB / {vram_total_gb:.2f} GB "
                f"(M, this process only, not system-wide)",
            )
        )
        hw_rows.append(("cuda", f"{torch.version.cuda} (M)"))
    else:
        hw_rows.append(("gpu", "CUDA Not Available (CPU Mode) (M)"))

    hw_rows.append(
        (
            "cpu",
            f"{psutil.cpu_count(logical=False)} Cores "
            f"({psutil.cpu_count(logical=True)} Threads) (M)",
        )
    )
    ram = psutil.virtual_memory()
    hw_rows.append(
        (
            "ram_system",
            f"{ram.used / (1024**3):.1f} GB / {ram.total / (1024**3):.1f} GB "
            f"({ram.percent}% used) (M)",
        )
    )
    hw_rows.append(("pytorch", f"{torch.__version__} (M)"))

    # WinLibs GCC
    winlibs_gcc = shutil.which("g++")
    if winlibs_gcc:
        try:
            ver_out = subprocess.check_output([winlibs_gcc, "--version"], text=True).splitlines()[0]
            hw_rows.append(("cxx_compiler", f"{ver_out} (M, from PATH)"))
        except (OSError, subprocess.SubprocessError, IndexError) as e:
            logger.debug("g++ version probe failed: %s", e)
            hw_rows.append(("cxx_compiler", f"{winlibs_gcc} (M)"))
    else:
        hw_rows.append(("cxx_compiler", "Not in system PATH (M)"))

    # Stockfish
    sf_path = get_default_stockfish_path()
    if sf_path and sf_path.exists():
        hw_rows.append(("stockfish", f"{sf_path} (M)"))
    else:
        hw_rows.append(("stockfish", "tools/stockfish.exe not found (M)"))

    emit_report(results_table(hw_rows))

    # 2. Checkpoints (first 12, newest first)
    checkpoints = get_available_checkpoints()
    ckpt_rows = [("checkpoints_found", f"{len(checkpoints)} (M)")]
    for i, c in enumerate(checkpoints[:12]):
        date_str = datetime.datetime.fromtimestamp(c["mtime"]).strftime("%Y-%m-%d %H:%M")
        latest = " latest" if i == 0 else ""
        ckpt_rows.append(
            (
                f"ckpt_{i}{latest}",
                f"{c['name']} in {c['dir']} step={c['step']} "
                f"net={c['num_blocks']}b/{c['num_filters']}f size={c['size_mb']} MB "
                f"mtime={date_str} (M)",
            )
        )
    emit_report(results_table(ckpt_rows))

    # 3. Evaluation History
    try:
        eval_csv = safe_resolve("runs/phase4_rl/eval_history.csv")
    except (ValueError, OSError) as e:
        logger.debug("eval_history path rejected: %s", e)
        return 0
    if eval_csv.is_file():
        eval_rows = []
        try:
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    eval_rows.append(r)
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            logger.debug("eval_history read failed: %s", e)

        if eval_rows:
            hist = [("evals_found", f"{len(eval_rows)} (M)")]
            for j, r in enumerate(eval_rows):
                elapsed_hrs = f"{float(r.get('elapsed_hours', 0)):.2f}h"
                hist.append(
                    (
                        f"eval_{j}",
                        f"elapsed={elapsed_hrs} step={r.get('step', '-')} "
                        f"ckpt={r.get('checkpoint_file', '-')} "
                        f"puzzles={float(r.get('puzzle_solve_rate', 0)) * 100:.1f}% "
                        f"({r.get('puzzle_solved_count', 0)}/{r.get('puzzle_total_count', 0)}) "
                        f"score={float(r.get('match_score', 0)) * 100:.1f}% "
                        f"W/D/L={r.get('match_engine_wins', 0)}/{r.get('match_engine_draws', 0)}/"
                        f"{r.get('match_engine_losses', 0)} (M)",
                    )
                )
            emit_report(results_table(hist))

    # 4. Replay Buffer status
    try:
        buf_dir = safe_resolve("runs/phase4_rl/buffer")
    except (ValueError, OSError) as e:
        logger.debug("buffer dir rejected: %s", e)
        return 0
    if buf_dir.is_dir():
        npz_files = list(buf_dir.glob("*.npz"))
        total_buf_mb = sum(f.stat().st_size for f in npz_files) / (1024 * 1024)
        emit_report(
            format_kv_line(
                [
                    ("replay_buffer", f"{buf_dir.name}/"),
                    ("chunks", f"{len(npz_files)} (M)"),
                    ("stored_mb", f"{total_buf_mb:.1f} (M)"),
                ]
            )
        )
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    return 0
