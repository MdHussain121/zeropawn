"""Profiling subcommands for ZeroPawn: throughput, CUDA events, pacing, and memory."""

from __future__ import annotations

import argparse

from python.cli.runner import execute_script_with_monitoring


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    prof_parser = subparsers.add_parser(
        "profile",
        help="Performance profiling: throughput, CUDA events, pacing, and memory diagnostics.",
        description="Low-level instrumentation and performance profiling tools.",
    )
    sub = prof_parser.add_subparsers(dest="profile_subcommand", help="Profiling actions")

    # 1. Throughput
    p_tp = sub.add_parser(
        "throughput", help="Throughput profiling script for self-play infrastructure."
    )
    p_tp.add_argument("--duration", type=float, default=2.0, help="Benchmark duration in seconds")
    p_tp.add_argument("--games", type=int, default=8, help="Concurrent game workers")
    p_tp.set_defaults(func=run_profile_throughput)

    # 2. CUDA
    p_cuda = sub.add_parser("cuda", help="CUDA event timing with clocks and power logging.")
    p_cuda.set_defaults(func=run_profile_cuda)

    # 3. Pacing
    p_pace = sub.add_parser("pacing", help="Measure combined mode under real pacing.")
    p_pace.set_defaults(func=run_profile_pacing)

    # 4. Memory
    p_mem = sub.add_parser("memory", help="Memory regression and reproduction diagnostics.")
    p_mem.set_defaults(func=run_profile_memory)

    # 5. Trace
    p_trace = sub.add_parser("trace", help="cProfile wall-clock breakdown of self-play pipeline.")
    p_trace.set_defaults(func=run_profile_trace)


def run_profile_throughput(args: argparse.Namespace) -> int:
    cmd_args = []
    if getattr(args, "duration", None) is not None:
        cmd_args.append(str(args.duration))
        if getattr(args, "games", None) is not None:
            cmd_args.append(str(args.games))
    return execute_script_with_monitoring(
        script_rel_path="scripts/profile_throughput.py",
        args=cmd_args,
        op_name="profile throughput",
        description="Self-play throughput profiling",
        parameters=vars(args),
    )


def run_profile_cuda(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/profile_cuda_events.py",
        args=[],
        op_name="profile cuda",
        description="CUDA event timing and power telemetry",
    )


def run_profile_pacing(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/profile_real_pacing.py",
        args=[],
        op_name="profile pacing",
        description="Real pacing combined mode measurement",
    )


def run_profile_memory(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/diagnose_memory.py",
        args=[],
        op_name="profile memory",
        description="Memory regression diagnostics",
    )


def run_profile_trace(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/profile_trace.py",
        args=[],
        op_name="profile trace",
        description="cProfile wall-clock execution trace",
    )
