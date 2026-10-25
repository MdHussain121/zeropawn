"""Hardware autotuning, profiling, and device discovery commands for ZeroPawn."""

from __future__ import annotations

import argparse
import json
import logging

from rich import box
from rich.panel import Panel
from rich.table import Table

from python.cli.common import console, print_error, print_info, print_success
from python.hardware.detector import detect_hardware
from python.hardware.profile import (
    calibrate_and_create_profile,
    get_profile_path,
    load_profile,
)

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    hw_parser = subparsers.add_parser(
        "hardware",
        aliases=["hw"],
        help="Hardware profiling: autotuning, persistent profile inspection, and calibration.",
        description="Inspect hardware capabilities, manage persistent hardware profiles, and calibrate autotuning.",
    )
    sub = hw_parser.add_subparsers(dest="hardware_subcommand", help="Hardware actions")

    # 1. Profile command
    p_prof = sub.add_parser(
        "profile", aliases=["prof"], help="Inspect or recalibrate the persistent hardware profile."
    )
    p_prof.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the active hardware profile (default action)",
    )
    p_prof.add_argument(
        "--recalibrate",
        "-r",
        action="store_true",
        default=False,
        help="Run live calibration and overwrite cached profile",
    )
    p_prof.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete cached profile to force recalibration on next run",
    )
    p_prof.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="Calibration time budget in seconds (default: 5.0)",
    )
    p_prof.add_argument(
        "--workload",
        type=str,
        default="all",
        choices=["all", "uci", "puzzles", "match", "selfplay"],
        help="Target workload to inspect",
    )
    p_prof.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output format (default: table)",
    )
    p_prof.set_defaults(func=run_hardware_profile)

    # 2. Info command
    p_info = sub.add_parser(
        "info",
        aliases=["status", "detect"],
        help="Detect and display low-level system hardware capabilities.",
    )
    p_info.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output format (default: table)",
    )
    p_info.set_defaults(func=run_hardware_info)

    # 3. Calibrate command
    p_cal = sub.add_parser("calibrate", help="Execute standalone hardware calibration benchmark.")
    p_cal.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="Calibration time budget in seconds (default: 5.0)",
    )
    p_cal.set_defaults(func=run_hardware_calibrate)

    # Set default action on `chess hardware` without subcommand
    hw_parser.set_defaults(func=run_hardware_profile_default)


def run_hardware_profile_default(args: argparse.Namespace) -> int:
    """Default handler for `chess hardware` when no subcommand is specified."""
    setattr(args, "show", True)
    setattr(args, "recalibrate", False)
    setattr(args, "reset", False)
    setattr(args, "workload", "all")
    setattr(args, "format", "table")
    return run_hardware_profile(args)


def run_hardware_info(args: argparse.Namespace) -> int:
    """Inspect and format hardware capabilities."""
    try:
        hw = detect_hardware()
    except Exception as e:
        print_error(f"Hardware detection failed: {e}")
        return 1

    if getattr(args, "format", "table") == "json":
        data = {
            "os": f"{hw.os_name} {hw.os_release} ({hw.os_arch})",
            "cpu_physical_cores": hw.cpu_physical_cores,
            "cpu_logical_threads": hw.cpu_logical_threads,
            "system_total_ram_gb": round(hw.system_total_ram_gb, 2),
            "gpu_name": hw.gpu_name,
            "gpu_compute_capability": hw.gpu_compute_capability,
            "gpu_total_vram_mb": round(hw.gpu_total_vram_mb, 1),
            "cuda_available": hw.cuda_available,
            "pytorch_version": hw.pytorch_version,
            "cuda_runtime_version": hw.cuda_runtime_version,
            "native_cpp_version": hw.native_cpp_version,
            "native_cpp_available": hw.native_cpp_available,
            "bf16_supported": hw.bf16_supported,
            "fp16_supported": hw.fp16_supported,
        }
        console.print(json.dumps(data, indent=2))
        return 0

    table = Table(
        title="[bold bright_cyan]Detected System Hardware & Capabilities[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Hardware Component", style="bright_white")
    table.add_column("Specification / Value", style="bold yellow")
    table.add_column("Status / Note", style="dim")

    table.add_row(
        "Operating System", f"{hw.os_name} {hw.os_release} ({hw.os_arch})", "Host Platform"
    )
    table.add_row(
        "CPU Architecture",
        f"{hw.cpu_physical_cores} physical / {hw.cpu_logical_threads} logical threads",
        "Multithreaded",
    )
    table.add_row("System RAM", f"{hw.system_total_ram_gb:.1f} GB", "System Memory")
    gpu_status = (
        "[bold green]CUDA Accelerated[/bold green]"
        if hw.cuda_available
        else "[yellow]CPU Only[/yellow]"
    )
    table.add_row("Graphics Processor", hw.gpu_name, gpu_status)
    if hw.cuda_available:
        table.add_row("GPU VRAM", f"{hw.gpu_total_vram_mb:.0f} MB", "Dedicated GDDR")
        table.add_row("Compute Capability", str(hw.gpu_compute_capability), "NVIDIA Architecture")
        table.add_row("CUDA Runtime", str(hw.cuda_runtime_version), "Driver / Toolkit")
        precisions = []
        if hw.bf16_supported:
            precisions.append("BF16")
        if hw.fp16_supported:
            precisions.append("FP16")
        prec_str = ", ".join(precisions) if precisions else "FP32 only"
        table.add_row("Tensor Precision", prec_str, "Hardware Mixed Precision")

    table.add_row("PyTorch Version", hw.pytorch_version, "Deep Learning Engine")
    cpp_status = (
        "[bold green]Ready (0.1.0-phase1)[/bold green]"
        if hw.native_cpp_available
        else "[red]Not Built[/red]"
    )
    table.add_row("Native C++ Extension", "zeropawn_cpp", cpp_status)

    console.print(table)
    return 0


def run_hardware_profile(args: argparse.Namespace) -> int:
    """Inspect, recalibrate, or reset persistent hardware profile."""
    hw = detect_hardware()
    profile_path = get_profile_path(hw)

    # 1. Reset profile if requested
    if getattr(args, "reset", False):
        if profile_path.is_file():
            profile_path.unlink()
            print_success(f"Removed cached hardware profile: {profile_path}")
        else:
            print_info(f"No cached profile found at: {profile_path}")
        return 0

    # 2. Recalibrate if requested or profile does not exist
    if getattr(args, "recalibrate", False) or not profile_path.is_file():
        budget = getattr(args, "budget", 5.0)
        console.print(
            f"[bold cyan]Calibrating hardware autotuner (budget: {budget:.1f}s)...[/bold cyan]"
        )
        try:
            profile = calibrate_and_create_profile(hw, budget_sec=budget, force=True)
            print_success(f"Calibration complete! Saved profile to: {profile_path}")
        except Exception as e:
            print_error(f"Hardware calibration failed: {e}")
            return 1
    else:
        profile = load_profile(hw)
        if profile is None:
            console.print("[yellow]Cached profile missing or invalid. Calibrating...[/yellow]")
            profile = calibrate_and_create_profile(hw, budget_sec=3.0, force=False)

    # Output formatting
    fmt = getattr(args, "format", "table")
    workload_filter = getattr(args, "workload", "all")

    if fmt == "json":
        data = profile.to_dict()
        if workload_filter != "all":
            data["workloads"] = {workload_filter: data["workloads"].get(workload_filter, {})}
        console.print(json.dumps(data, indent=2))
        return 0

    # Table output
    summary_panel = Panel(
        f"[bold cyan]Profile Fingerprint:[/] [bold yellow]{profile.fingerprint}[/bold yellow]  |  "
        f"[bold cyan]Created:[/] {profile.created_at}  |  "
        f"[bold cyan]Schema Version:[/] {profile.schema_version}\n"
        f"[dim]Cache Path: {profile_path}[/dim]",
        title="[bold green]ZeroPawn Hardware Autotuning Profile[/bold green]",
        box=box.ROUNDED,
    )
    console.print(summary_panel)

    table = Table(
        title="[bold bright_cyan]Autotuned Workload Targets[/bold bright_cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Workload", style="bold yellow")
    table.add_column("Search Mode", style="bright_white")
    table.add_column("Batch Size", justify="right", style="bold green")
    table.add_column("Max Delay", justify="right", style="cyan")
    table.add_column("Concurrency", justify="right", style="magenta")
    table.add_column("Device / Precision", style="bright_white")
    table.add_column("Optimization Rationale", style="dim")

    for wl_name, wl_cfg in profile.workloads.items():
        if workload_filter != "all" and wl_name != workload_filter:
            continue
        table.add_row(
            wl_name.upper(),
            wl_cfg.get("search_mode", "batched"),
            str(wl_cfg.get("batch_size", 16)),
            f"{wl_cfg.get('max_wait_ms', 1.0):.1f} ms",
            str(wl_cfg.get("concurrency", 1)),
            f"{wl_cfg.get('device', 'cuda')} / {wl_cfg.get('precision', 'bfloat16')}",
            wl_cfg.get("rationale", ""),
        )

    console.print(table)
    return 0


def run_hardware_calibrate(args: argparse.Namespace) -> int:
    """Shortcut command to run calibration directly."""
    setattr(args, "recalibrate", True)
    setattr(args, "show", True)
    setattr(args, "reset", False)
    setattr(args, "workload", "all")
    setattr(args, "format", "table")
    return run_hardware_profile(args)
