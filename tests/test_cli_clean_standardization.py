"""Verification tests for ZeroPawn CLI standardization, CMD deduplication, and phase-free commands."""

import subprocess
import sys
from pathlib import Path
import pytest

from python.cli.main import build_parser
from python.cli.common import REPO_ROOT


def test_purpose_based_cli_commands():
    """Verify all standardized purpose-based CLI commands parse without phase naming."""
    parser = build_parser()

    # zeropawn status
    args = parser.parse_args(["status"])
    assert args.command == "status"

    # zeropawn train
    args = parser.parse_args(["train", "--max-hours", "10.0"])
    assert args.command in ["train", "tr", "selfplay"]
    assert args.max_hours == 10.0

    # zeropawn selfplay alias
    args = parser.parse_args(["selfplay", "--max-hours", "5.0"])
    assert args.command in ["train", "tr", "selfplay"]
    assert args.max_hours == 5.0

    # zeropawn train --supervised
    args = parser.parse_args(["train", "--supervised", "--epochs", "3"])
    assert args.supervised is True
    assert args.epochs == 3

    # zeropawn train --resume
    args = parser.parse_args(["train", "--resume", "models/checkpoints_rl/live_weights.pt"])
    assert args.resume == "models/checkpoints_rl/live_weights.pt"

    # zeropawn eval
    args = parser.parse_args(["eval", "baseline", "--games", "10"])
    assert args.command == "eval"
    assert args.eval_subcommand == "baseline"

    # zeropawn match
    args = parser.parse_args(["match", "run", "-n", "20"])
    assert args.command in ["match", "play", "stockfish", "vs"]
    assert args.games == 20

    # zeropawn bench
    args = parser.parse_args(["bench", "batch-sizes", "--duration", "5.0"])
    assert args.command == "bench"
    assert args.bench_subcommand == "batch-sizes"

    # zeropawn data list
    args = parser.parse_args(["data", "list"])
    assert args.command == "data"
    assert args.data_subcommand == "list"

    # zeropawn data validate
    args = parser.parse_args(["data", "validate", "--dir", "data/dataset_training"])
    assert args.command == "data"
    assert args.data_subcommand == "validate"

    # zeropawn models list
    args = parser.parse_args(["models", "list"])
    assert args.command in ["models", "model"]
    assert args.models_subcommand == "list"

    # zeropawn diagnose
    args = parser.parse_args(["diagnose", "health"])
    assert args.command == "diagnose"
    assert args.diag_subcommand == "health"


def test_obsolete_launchers_removed():
    """Verify obsolete duplicate and phase-centric batch files are eliminated."""
    removed_launchers = [
        REPO_ROOT / "run_phase4_rl.cmd",
        REPO_ROOT / "run_batch_benchmark.cmd",
        REPO_ROOT / "chess_cli.cmd",
    ]
    for p in removed_launchers:
        assert not p.exists(), f"Obsolete launcher still exists: {p}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch files require cmd.exe")
def test_canonical_launcher_and_wrapper_behavior():
    """Verify primary launcher zeropawn.cmd and compatibility wrapper chess.cmd."""
    # 1. Primary launcher zeropawn.cmd forwards arguments and preserves exit code
    zp_proc = subprocess.run(
        ["cmd.exe", "/c", "zeropawn.cmd", "--version"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert zp_proc.returncode == 0
    assert "ZeroPawn" in zp_proc.stdout

    # 2. Compatibility wrapper chess.cmd forwards arguments and prints deprecation to stderr
    ch_proc = subprocess.run(
        ["cmd.exe", "/c", "chess.cmd", "--version"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert ch_proc.returncode == 0
    assert "ZeroPawn" in ch_proc.stdout
    assert "[DEPRECATION]" in ch_proc.stderr

    # 3. Compatibility wrapper chess.cmd in UCI mode suppresses deprecation on stderr
    uci_proc = subprocess.run(
        ["cmd.exe", "/c", "chess.cmd", "uci"],
        input="quit\n",
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert "[DEPRECATION]" not in uci_proc.stderr
