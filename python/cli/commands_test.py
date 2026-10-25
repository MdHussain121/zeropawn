"""Test runner subcommand for ZeroPawn."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from python.cli.common import (
    REPO_ROOT,
    safe_resolve,
)
from python.common.progress import (
    banner,
    emit_error,
    emit_report,
    error_block,
    header_block,
    parameters_table,
    run_context,
    summary_block,
)
import logging

logger = logging.getLogger(__name__)

DENIED_PYTEST_ARGS = {"-p", "--rootdir", "-c", "--config-file", "--rootdir="}
ALLOWED_PYTEST_PREFIXES = ("-k", "-x", "-q", "-v", "-s", "--lf", "--ff", "-m", "--maxfail", "-l")

SUITE_MAPPING = {
    "all": ["tests/"],
    "fast": [
        "tests/test_import.py",
        "tests/test_board.py",
        "tests/test_terminal.py",
        "tests/test_encoder.py",
        "tests/test_perft.py",
        "tests/test_puzzles.py",
        "tests/test_mcts.py",
        "tests/test_metrics.py",
    ],
    "puzzles": ["tests/test_puzzles.py"],
    "model": ["tests/test_chess_net.py"],
    "mcts": ["tests/test_mcts.py"],
    "match": ["tests/test_match_runner.py"],
    "board": ["tests/test_board.py", "tests/test_perft.py", "tests/test_terminal.py"],
    "worker": ["tests/test_worker.py"],
    "replay": ["tests/test_replay_buffer.py"],
    "supervised": ["tests/test_sanity_supervised.py", "tests/test_train_supervised.py"],
    "rl": ["tests/test_rl_trainer.py", "tests/test_correctness.py"],
    "server": ["tests/test_inference_server.py"],
    "differential": ["tests/test_differential.py"],
    "target": ["tests/test_stockfish_target.py"],
    "connect4": ["tests/test_connect4.py"],
    "soak": ["scripts/soak_test_runner.py"],
    "guardrails": ["scripts/verify_guardrails.py"],
    "parity": ["scripts/check_terminal_parity.py"],
    "cli": ["tests/test_cli.py", "tests/test_cli_overhaul.py"],
    "uci": ["tests/test_uci.py"],
    "cpp": ["build_zeropawn/cpp/zeropawn_tests.exe"],
}


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "test",
        aliases=["tests", "t"],
        help="Run pytest test suites with customizable filters and flags.",
        description="Run test suites across the engine, MCTS, neural net, replay buffer, and selfplay.",
    )
    parser.add_argument(
        "suite_pos",
        nargs="?",
        default=None,
        help=f"Target suite name: {', '.join(SUITE_MAPPING.keys())} (default: all)",
    )
    parser.add_argument(
        "--suite",
        "-s",
        type=str,
        default="all",
        choices=list(SUITE_MAPPING.keys()),
        help="Target test suite to run (default: all)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Specific test file to run (e.g. tests/test_puzzles.py)",
    )
    parser.add_argument(
        "-k",
        "--filter",
        type=str,
        default=None,
        help="Pytest keyword expression to filter test functions (e.g. -k 'mate or wdl')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Run pytest in verbose mode showing each test item (default: True)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Run pytest in quiet mode without per-operation verbosity",
    )
    parser.add_argument(
        "--show-output",
        "--nocapture",
        action="store_true",
        default=True,
        help="Show live print/log output from test operations (-s) (default: True)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Suppress live print output from test operations",
    )
    parser.add_argument(
        "-x",
        "--failfast",
        action="store_true",
        help="Stop on first test failure (-x)",
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Collect coverage report for python/ package",
    )
    parser.add_argument(
        "--lf",
        "--last-failed",
        action="store_true",
        help="Rerun only tests that failed in the previous run",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Collect and list available tests without executing them",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Total soak duration in hours (default: 1.0)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of random games for differential test",
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help="Specific operation to test parity for",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional raw arguments forwarded directly to pytest",
    )
    parser.set_defaults(func=run_tests)


def run_tests(args: argparse.Namespace) -> int:
    suite_name = args.suite_pos if args.suite_pos else args.suite

    if suite_name == "cpp":
        from python.cli.commands_build import run_build_test

        return run_build_test(args)

    if suite_name == "soak":
        from python.cli.runner import execute_script_with_monitoring

        h = getattr(args, "hours", 1.0) or 1.0
        return execute_script_with_monitoring(
            script_rel_path="scripts/soak_test_runner.py",
            args=["--max-hours", str(h)],
            op_name="test soak",
            description="72-Hour Soak Test Runner",
            parameters={"hours": h},
        )

    if suite_name == "guardrails":
        from python.cli.runner import execute_script_with_monitoring

        return execute_script_with_monitoring(
            script_rel_path="scripts/verify_guardrails.py",
            args=[],
            op_name="test guardrails",
            description="Correctness Guardrails Verification",
        )

    if suite_name == "parity":
        from python.cli.commands_diagnose import run_diagnose_parity

        return run_diagnose_parity(args)

    if suite_name == "differential" and getattr(args, "games", None):
        from python.cli.runner import execute_script_with_monitoring

        return execute_script_with_monitoring(
            script_rel_path="scripts/run_differential.py",
            args=[str(args.games)],
            op_name="test differential",
            description="Differential Test vs Python-Chess",
            parameters={"games": args.games},
        )

    cmd = [sys.executable, "-m", "pytest"]

    if args.list:
        cmd.append("--collect-only")

    # Show verbose item progress unless explicitly silenced
    if not getattr(args, "quiet", False) and getattr(args, "verbose", True):
        cmd.append("-v")

    # Show live operational stdout (-s) unless explicitly captured
    if not getattr(args, "capture", False) and getattr(args, "show_output", True):
        cmd.append("-s")

    if args.failfast:
        cmd.append("-x")

    if args.lf:
        cmd.append("--lf")

    if args.cov:
        cmd.extend(["--cov=python", "--cov-report=term-missing"])

    if args.filter:
        cmd.extend(["-k", args.filter])

    if args.file:
        try:
            test_path = safe_resolve(args.file)
        except (ValueError, OSError) as e:
            emit_error(
                error_block(
                    f"Invalid --file path: {e}",
                    "path escapes repo or is malformed",
                    "pass a tests/test_*.py path, then see LOG",
                )
            )
            return 1
        # Must live inside tests/ and look like test_*.py
        tests_base = (REPO_ROOT / "tests").resolve()
        if test_path.suffix != ".py" or not test_path.name.startswith("test_"):
            emit_error(
                error_block(
                    "--file must be a tests/test_*.py file.",
                    "wrong file pattern",
                    "pass tests/test_*.py, then see LOG",
                )
            )
            return 1
        try:
            test_path.relative_to(tests_base)
        except ValueError:
            emit_error(
                error_block(
                    "--file must be inside tests/ directory.",
                    "path outside tests/",
                    "pass a tests/ file, then see LOG",
                )
            )
            return 1
        cmd.append(str(test_path))
    else:
        suite_targets = SUITE_MAPPING.get(suite_name, ["tests/"])
        for target in suite_targets:
            cmd.append(str(REPO_ROOT / target))

    if args.pytest_args:
        for a in args.pytest_args:
            base_flag = a.split("=")[0]
            if (
                base_flag in DENIED_PYTEST_ARGS
                or a in ("-p", "--rootdir", "-c", "--override-ini")
                or base_flag.startswith(("--import-mode", "--capture"))
                or "\x00" in a
                or (not a.startswith("-") and ".." in a)
            ):
                emit_error(
                    error_block(
                        f"Denied pytest arg: {a} (allows only safe flags).",
                        "unsafe pytest passthrough",
                        "use an allowed flag, then see LOG",
                    )
                )
                return 1
        cmd.extend(args.pytest_args)

    ctx = run_context("test")
    t0 = time.time()
    emit_report(banner("test", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu",
            "n/a",
        )
    )
    emit_report(
        parameters_table(
            [
                ("suite", suite_name, "user" if suite_name != "all" else "default"),
                (
                    "file",
                    str(args.file) if args.file else "none",
                    "user" if args.file else "default",
                ),
                (
                    "filter",
                    str(args.filter) if args.filter else "none",
                    "user" if args.filter else "default",
                ),
                ("pytest_cmd", " ".join(cmd[3:]), "derived"),
            ]
        )
    )

    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        # Pytest output passes straight through (its own format, unmodified).
        ret = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        status = "SUCCESS" if ret.returncode == 0 else "FAILED"
        emit_report(summary_block(status, time.time() - t0, 0, ret.returncode))
        return ret.returncode
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        logger.debug("pytest execution failed: %s", e)
        emit_error(
            error_block(
                f"Failed to execute tests: {e}",
                "pytest could not start",
                "check python/pytest install, then see LOG",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        return 1
