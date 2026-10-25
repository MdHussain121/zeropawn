"""Benchmark subcommands for ZeroPawn: self-play throughput, batch sizes, and dispatcher sweeps."""

from __future__ import annotations

import argparse

from python.cli.models import get_default_model_path
from python.cli.runner import execute_script_with_monitoring


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    bench_parser = subparsers.add_parser(
        "bench",
        aliases=["benchmark"],
        help="Benchmark commands: throughput, batch sizes, pipelined dispatcher, and concurrency sweeps.",
        description="Comprehensive benchmarking suite for self-play inference, MCTS, and throughput.",
    )
    sub = bench_parser.add_subparsers(dest="bench_subcommand", help="Benchmark actions")

    # 1. Selfplay
    p_sp = sub.add_parser("selfplay", help="Run self-play throughput and simulation benchmark.")
    p_sp.add_argument(
        "--mode", type=str, default="selfplay", choices=["selfplay", "training", "combined"]
    )
    p_sp.add_argument("--warmup-sec", type=float, default=2.0, help="Warm-up duration in seconds")
    p_sp.add_argument(
        "--counted-sec", type=float, default=10.0, help="Counted measurement duration in seconds"
    )
    p_sp.add_argument("--trials", type=int, default=1, help="Number of benchmark trials")
    p_sp.add_argument("--concurrency", type=int, default=24, help="Concurrent worker count")
    p_sp.add_argument("--full-sims", type=int, default=50, help="Full MCTS simulations per move")
    p_sp.add_argument("--fast-sims", type=int, default=10, help="Fast MCTS simulations per move")
    p_sp.add_argument(
        "--checkpoint-path", "-c", type=str, default=None, help="Model checkpoint path"
    )
    p_sp.add_argument("--output-dir", type=str, default="tmp/bench_out", help="Output directory")
    p_sp.set_defaults(func=run_selfplay_bench)

    # 2. Batch sizes
    p_bs = sub.add_parser("batch-sizes", help="Automated batch size & timeout sweep.")
    p_bs.add_argument(
        "--duration", type=float, default=2.0, help="Duration per batch size test in seconds"
    )
    p_bs.add_argument("--timeout", type=float, default=3.0, help="Max wait timeout in ms")
    p_bs.add_argument(
        "--concurrent-games", type=int, default=8, help="Concurrent games for load generation"
    )
    p_bs.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    p_bs.set_defaults(func=run_batch_sizes)

    # 3. Pipelined
    p_pip = sub.add_parser("pipelined", help="Benchmark pipelined vs non-pipelined dispatcher.")
    p_pip.add_argument("--warmup-sec", type=float, default=2.0, help="Warmup duration in seconds")
    p_pip.add_argument("--counted-sec", type=float, default=5.0, help="Counted duration in seconds")
    p_pip.add_argument("--trials", type=int, default=1, help="Number of trials per mode")
    p_pip.set_defaults(func=run_pipelined)

    # 4. Dispatcher
    p_disp = sub.add_parser("dispatcher", help="Dispatcher ceiling benchmark (IPC ceiling).")
    p_disp.add_argument(
        "--duration", type=float, default=2.0, help="Duration per K burst trial in seconds"
    )
    p_disp.add_argument("--games", type=int, default=8, help="Number of stub game threads")
    p_disp.set_defaults(func=run_dispatcher)

    # 5. AB
    p_ab = sub.add_parser("ab", help="A/B benchmark discipline (interleaved pairs + bracketing).")
    p_ab.add_argument(
        "--out-dir", type=str, default="runs/ab", help="Output directory (default: runs/ab)"
    )
    p_ab.add_argument("--chunk", type=str, default="0:1", help="Slice of benchmark plan (e.g. 0:1)")
    p_ab.set_defaults(func=run_ab)

    # 6. Sweep Concurrency (legacy)
    p_sc = sub.add_parser("sweep-concurrency", help="Worker concurrency scaling sweep.")
    p_sc.add_argument("--warmup", type=float, default=2.0, help="Warmup seconds")
    p_sc.add_argument("--counted", type=float, default=5.0, help="Counted seconds")
    p_sc.add_argument(
        "--concurrencies", type=int, nargs="+", default=[4, 8], help="Concurrencies to sweep"
    )
    p_sc.set_defaults(func=run_sweep_concurrency)

    # 7. Sweep Timeout (legacy)
    p_st = sub.add_parser("sweep-timeout", help="Dispatcher timeout sweep at concurrency 24.")
    p_st.set_defaults(func=run_sweep_timeout)

    # 8. Extended Batch Scaling Sweep (Stage D / D.1)
    p_bsw = sub.add_parser(
        "batch-sweep",
        aliases=["sweep-batch", "batch-scaling"],
        help="Extended batch scaling sweep (B=1 to 256) across pure inference and full MCTS.",
    )
    p_bsw.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,24,32,64,128,256",
        help="Comma-separated batch sizes to sweep (default: 1,4,8,16,24,32,64,128,256)",
    )
    p_bsw.add_argument(
        "--sims",
        type=int,
        default=400,
        help="MCTS simulations per tested batch size (default: 400)",
    )
    p_bsw.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    p_bsw.add_argument(
        "--output",
        "-o",
        type=str,
        default="docs/perf/BATCH_SWEEP_EXTENDED.json",
        help="Path to save benchmark JSON results",
    )
    p_bsw.set_defaults(func=run_batch_sweep)

    # 9. Extended Concurrency Scaling Sweep (Stage D / D.1)
    p_csw = sub.add_parser(
        "concurrency-sweep",
        aliases=["concurrency-scaling"],
        help="Multi-worker concurrency and pipelining delay sweep (C=1 to 32, delta=0.5 to 3.0ms).",
    )
    p_csw.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,24,32",
        help="Comma-separated worker counts to sweep (default: 1,2,4,8,16,24,32)",
    )
    p_csw.add_argument(
        "--delays",
        type=str,
        default="0.5,1.0,2.0,2.5,3.0",
        help="Comma-separated batch timeouts in ms to sweep (default: 0.5,1.0,2.0,2.5,3.0)",
    )
    p_csw.add_argument(
        "--output",
        "-o",
        type=str,
        default="docs/perf/CONCURRENCY_SWEEP.json",
        help="Path to save benchmark JSON results",
    )
    p_csw.set_defaults(func=run_concurrency_sweep)

    # 10. Verification Suite (Stage D / D.1)
    p_ver = sub.add_parser(
        "verify",
        aliases=["verify-pipeline"],
        help="Full end-to-end pipeline verification (inference, MCTS, puzzles, match).",
    )
    p_ver.add_argument(
        "--output",
        "-o",
        type=str,
        default="docs/perf/STAGE_D_VERIFIED_REPORT.json",
        help="Path to save verification report JSON",
    )
    p_ver.set_defaults(func=run_verify)

    # Allow top-level `bench --batch-sizes ...` or default action
    bench_parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Run batch scaling sweep with comma-separated batch sizes",
    )
    bench_parser.add_argument(
        "--sims", type=int, default=400, help="Simulations per tested batch size (default: 400)"
    )
    bench_parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    bench_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="docs/perf/BATCH_SWEEP_EXTENDED.json",
        help="Path to save benchmark JSON results",
    )
    bench_parser.set_defaults(func=run_bench_default)


def run_selfplay_bench(args: argparse.Namespace) -> int:
    ckpt = args.checkpoint_path or str(get_default_model_path())
    mode = "train" if args.mode == "training" else args.mode
    cmd_args = [
        "--mode",
        mode,
        "--warmup-sec",
        str(args.warmup_sec),
        "--counted-sec",
        str(args.counted_sec),
        "--trials",
        str(args.trials),
        "--concurrency",
        str(args.concurrency),
        "--full-sims",
        str(args.full_sims),
        "--fast-sims",
        str(args.fast_sims),
        "--checkpoint-path",
        ckpt,
        "--output-dir",
        args.output_dir,
    ]
    return execute_script_with_monitoring(
        script_rel_path="scripts/bench_selfplay.py",
        args=cmd_args,
        op_name="bench selfplay",
        description="Self-play throughput and simulation benchmark",
        parameters=vars(args),
        model_path=ckpt,
    )


def run_batch_sizes(args: argparse.Namespace) -> int:
    cmd_args = []
    if getattr(args, "duration", None):
        cmd_args.extend(["--duration", str(args.duration)])
    if getattr(args, "timeout", None):
        cmd_args.extend(["--timeout", str(args.timeout)])
    if getattr(args, "concurrent_games", None):
        cmd_args.extend(["--concurrent-games", str(args.concurrent_games)])
    if getattr(args, "checkpoint", None):
        cmd_args.extend(["--checkpoint", str(args.checkpoint)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/benchmark_batch_sizes.py",
        args=cmd_args,
        op_name="bench batch-sizes",
        description="Automated batch size and timeout benchmark",
        parameters=vars(args),
    )


def run_pipelined(args: argparse.Namespace) -> int:
    cmd_args = []
    if getattr(args, "warmup_sec", None):
        cmd_args.extend(["--warmup-sec", str(args.warmup_sec)])
    if getattr(args, "counted_sec", None):
        cmd_args.extend(["--counted-sec", str(args.counted_sec)])
    if getattr(args, "trials", None):
        cmd_args.extend(["--trials", str(args.trials)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/bench_pipelined.py",
        args=cmd_args,
        op_name="bench pipelined",
        description="Pipelined vs non-pipelined dispatcher benchmark",
        parameters=vars(args),
    )


def run_dispatcher(args: argparse.Namespace) -> int:
    cmd_args = []
    if getattr(args, "duration", None):
        cmd_args.extend(["--duration", str(args.duration)])
    if getattr(args, "games", None):
        cmd_args.extend(["--games", str(args.games)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/bench_dispatcher_ceiling.py",
        args=cmd_args,
        op_name="bench dispatcher",
        description="Dispatcher IPC ceiling benchmark",
        parameters=vars(args),
    )


def run_ab(args: argparse.Namespace) -> int:
    cmd_args = []
    if getattr(args, "out_dir", None):
        cmd_args.extend(["--out-dir", str(args.out_dir)])
    if getattr(args, "chunk", None):
        cmd_args.extend(["--chunk", str(args.chunk)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/ab_bench.py",
        args=cmd_args,
        op_name="bench ab",
        description="A/B interleaved benchmark comparison",
        parameters=vars(args),
    )


def run_sweep_concurrency(args: argparse.Namespace) -> int:
    cmd_args = []
    if getattr(args, "warmup", None):
        cmd_args.extend(["--warmup", str(args.warmup)])
    if getattr(args, "counted", None):
        cmd_args.extend(["--counted", str(args.counted)])
    if getattr(args, "concurrencies", None):
        cmd_args.extend(["--concurrencies"] + [str(c) for c in args.concurrencies])
    return execute_script_with_monitoring(
        script_rel_path="scripts/sweep_concurrency.py",
        args=cmd_args,
        op_name="bench sweep-concurrency",
        description="Concurrency sweep over worker counts",
        parameters=vars(args),
    )


def run_sweep_timeout(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/sweep_dispatcher_timeout.py",
        args=[],
        op_name="bench sweep-timeout",
        description="Dispatcher timeout sweep",
    )


def run_batch_sweep(args: argparse.Namespace) -> int:
    """Run extended batch scaling sweep (B=1 to 256) across pure inference and MCTS."""
    cmd_args = []
    if getattr(args, "batch_sizes", None):
        cmd_args.extend(["--batch-sizes", str(args.batch_sizes)])
    if getattr(args, "sims", None):
        cmd_args.extend(["--sims", str(args.sims)])
    if getattr(args, "checkpoint", None):
        cmd_args.extend(["--checkpoint", str(args.checkpoint)])
    if getattr(args, "output", None):
        cmd_args.extend(["--output", str(args.output)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/benchmark_batch_sweep.py",
        args=cmd_args,
        op_name="bench batch-sweep",
        description="Extended batch scaling sweep (B=1 to 256)",
        parameters=vars(args),
    )


def run_concurrency_sweep(args: argparse.Namespace) -> int:
    """Run multi-worker concurrency and pipelining delay sweep."""
    cmd_args = []
    if getattr(args, "concurrencies", None):
        val = (
            args.concurrencies
            if isinstance(args.concurrencies, str)
            else ",".join(str(x) for x in args.concurrencies)
        )
        cmd_args.extend(["--concurrencies", val])
    if getattr(args, "delays", None):
        cmd_args.extend(["--delays", str(args.delays)])
    if getattr(args, "checkpoint", None):
        cmd_args.extend(["--checkpoint", str(args.checkpoint)])
    if getattr(args, "output", None):
        cmd_args.extend(["--output", str(args.output)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/benchmark_concurrency_sweep.py",
        args=cmd_args,
        op_name="bench concurrency-sweep",
        description="Multi-worker concurrency and delay sweep",
        parameters=vars(args),
    )


def run_verify(args: argparse.Namespace) -> int:
    """Run end-to-end pipeline verification (inference, MCTS, puzzles, match)."""
    cmd_args = []
    if getattr(args, "output", None):
        cmd_args.extend(["--output", str(args.output)])
    return execute_script_with_monitoring(
        script_rel_path="scripts/benchmark_verification.py",
        args=cmd_args,
        op_name="bench verify",
        description="Full pipeline verification report",
        parameters=vars(args),
    )


def run_bench_default(args: argparse.Namespace) -> int:
    """Default handler for `chess bench`."""
    if getattr(args, "batch_sizes", None):
        return run_batch_sweep(args)
    # Default to self-play benchmark with standard parameters if no subcommand
    setattr(args, "mode", "selfplay")
    setattr(args, "warmup_sec", 2.0)
    setattr(args, "counted_sec", 10.0)
    setattr(args, "trials", 1)
    setattr(args, "concurrency", 24)
    setattr(args, "full_sims", 50)
    setattr(args, "fast_sims", 10)
    setattr(args, "checkpoint_path", None)
    setattr(args, "output_dir", "tmp/bench_out")
    return run_selfplay_bench(args)
