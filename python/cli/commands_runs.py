"""Run history, telemetry inspection, interrupted run detection, and recovery CLI commands for ZeroPawn."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from rich import box
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER, console


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    runs_parser = subparsers.add_parser(
        "runs",
        aliases=["run-history", "experiments"],
        help="Run management: list training runs, inspect telemetry, detect interrupted jobs, and recover checkpoints.",
        description="Monitor, inspect, and recover past and ongoing ZeroPawn training and benchmark runs.",
    )
    sub = runs_parser.add_subparsers(dest="runs_subcommand", help="Run actions")

    # 1. List
    p_list = sub.add_parser(
        "list", help="List all discovered experiment and training runs in runs/."
    )
    p_list.set_defaults(func=run_runs_list)

    # 2. Inspect
    p_insp = sub.add_parser(
        "inspect", help="Inspect telemetry, steps, loss curves, and parameters for a run."
    )
    p_insp.add_argument("run_id", help="Directory name or path of the run (e.g. phase4_rl)")
    p_insp.set_defaults(func=run_runs_inspect)

    # 3. Interrupted
    p_int = sub.add_parser("interrupted", help="Detect interrupted or abnormally terminated runs.")
    p_int.set_defaults(func=run_runs_interrupted)

    # 4. Recover
    p_rec = sub.add_parser(
        "recover", help="Recover latest checkpoint and print command to resume an interrupted run."
    )
    p_rec.add_argument("run_id", help="Run directory to recover")
    p_rec.set_defaults(func=run_runs_recover)


def run_runs_list(args: argparse.Namespace) -> int:
    runs_root = REPO_ROOT / "runs"
    if not runs_root.is_dir():
        console.print("[bold yellow]No runs/ directory found.[/bold yellow]")
        return 0

    table = Table(
        title="[bold bright_cyan]Discovered ZeroPawn Runs (runs/)[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="bold yellow")
    table.add_column("Run Directory", style="bright_white")
    table.add_column("Telemetry Records", justify="right", style="cyan")
    table.add_column("Events Size", justify="right", style="dim")
    table.add_column("Last Modified", style="dim")
    table.add_column("Status", justify="center")

    idx = 1
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue

        telem_file = d / "phase4_telemetry.csv"
        records_count = 0
        if telem_file.is_file():
            try:
                with open(telem_file, "r", encoding="utf-8") as f:
                    records_count = sum(1 for _ in f) - 1
            except Exception:
                pass

        total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        size_mb = total_bytes / (1024 * 1024)

        mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        # Status heuristic
        status = "[green]COMPLETED[/green]"
        if (d / "phase4_final_checkpoint.pt").is_file():
            status = "[bold green]COMPLETED[/bold green]"
        elif records_count > 0:
            status = "[yellow]INTERRUPTED / ACTIVE[/yellow]"

        table.add_row(
            str(idx),
            d.name,
            f"{records_count:,}" if records_count > 0 else "N/A",
            f"{size_mb:.1f} MB",
            mtime,
            status,
        )
        idx += 1

    console.print(table)
    return 0


def run_runs_inspect(args: argparse.Namespace) -> int:
    d = Path(args.run_id) if Path(args.run_id).is_absolute() else (REPO_ROOT / "runs" / args.run_id)
    if not d.is_dir():
        d = REPO_ROOT / args.run_id
    if not d.is_dir():
        console.print(f"[bold red]Run directory not found: {args.run_id}[/bold red]")
        return 1

    table = Table(
        title=f"[bold bright_cyan]Run Inspection: {d.name}[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value / Details", style="bright_white")

    table.add_row("Run Directory", str(d.resolve()))

    telem_file = d / "phase4_telemetry.csv"
    if telem_file.is_file():
        try:
            with open(telem_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                table.add_row("Telemetry Rows", f"{len(rows):,} recorded steps")
                if rows:
                    last = rows[-1]
                    table.add_row("Last Step", str(last.get("step", "N/A")))
                    table.add_row("Last Total Loss", str(last.get("total_loss", "N/A")))
                    table.add_row("Last Sim Rate", f"{last.get('sims_per_sec', 'N/A')} sims/s")
        except Exception as e:
            table.add_row("Telemetry Read Error", str(e))
    else:
        table.add_row("Telemetry CSV", "Not found in run directory")

    eval_file = d / "eval_history.csv"
    if eval_file.is_file():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f) - 1
                table.add_row("Evaluation Cycles", f"{count} 6-hour benchmarks")
        except Exception:
            pass

    # Check for replay buffer
    bin_files = list(d.glob("*.bin"))
    if bin_files:
        bin_mb = sum(f.stat().st_size for f in bin_files) / (1024 * 1024)
        table.add_row("Replay Buffer Disk Usage", f"{len(bin_files)} files ({bin_mb:.1f} MB)")

    console.print(table)
    return 0


def run_runs_interrupted(args: argparse.Namespace) -> int:
    runs_root = REPO_ROOT / "runs"
    if not runs_root.is_dir():
        console.print("[dim]No runs/ directory found.[/dim]")
        return 0

    interrupted = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        # If has telemetry or buffer but no final checkpoint marker
        telem = d / "phase4_telemetry.csv"
        final_ckpt = REPO_ROOT / "models" / "checkpoints_rl" / "phase4_final_checkpoint.pt"
        if telem.is_file() and not final_ckpt.is_file():
            interrupted.append(d)

    if not interrupted:
        console.print(
            "[bold green]No interrupted runs detected. All jobs completed cleanly or workspace is clean.[/bold green]"
        )
        return 0

    table = Table(title="[bold yellow]Detected Interrupted Runs[/bold yellow]", box=box.ROUNDED)
    table.add_column("Run Directory", style="bright_white")
    table.add_column("Recovery Guidance", style="cyan")

    for d in interrupted:
        table.add_row(
            d.name,
            f"Run 'chess runs recover {d.name}' to resume execution from latest checkpoint",
        )

    console.print(table)
    return 0


def run_runs_recover(args: argparse.Namespace) -> int:
    d = Path(args.run_id) if Path(args.run_id).is_absolute() else (REPO_ROOT / "runs" / args.run_id)
    if not d.is_dir():
        d = REPO_ROOT / args.run_id
    if not d.is_dir():
        console.print(f"[bold red]Run directory not found: {args.run_id}[/bold red]")
        return 1

    # Check for latest checkpoint in models/checkpoints_rl
    ckpt_dir = REPO_ROOT / "models" / "checkpoints_rl"
    latest_ckpt = ckpt_dir / "latest_checkpoint.pt"

    if latest_ckpt.is_file():
        console.print(f"[bold green]Recovered active checkpoint:[/] [cyan]{latest_ckpt}[/cyan]")
        console.print(
            "\n[bold yellow]To resume this training run cleanly, execute:[/bold yellow]\n"
            f"  [bold cyan]chess train --resume {latest_ckpt}[/bold cyan]\n"
        )
        return 0
    else:
        # Check all checkpoints in ckpt_dir
        ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if ckpts:
            cand = ckpts[0]
            console.print(f"[bold green]Recovered newest checkpoint:[/] [cyan]{cand}[/cyan]")
            console.print(
                "\n[bold yellow]To resume from this checkpoint, execute:[/bold yellow]\n"
                f"  [bold cyan]chess train --resume {cand}[/bold cyan]\n"
            )
            return 0
        else:
            console.print(
                "[bold red]No resumable checkpoint found in models/checkpoints_rl.[/bold red]"
            )
            return 1
