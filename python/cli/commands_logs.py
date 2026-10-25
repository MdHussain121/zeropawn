"""Log management CLI subcommands for ZeroPawn: tail, export, and clear."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich import box
from rich.panel import Panel

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER, console

DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "zeropawn.log"


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    logs_parser = subparsers.add_parser(
        "logs",
        aliases=["log"],
        help="Log management: view log tail, export logs, and clear logs.",
        description="Inspect, tail, and export application execution logs.",
    )
    sub = logs_parser.add_subparsers(dest="logs_subcommand", help="Log actions")

    # 1. Tail
    p_tail = sub.add_parser("tail", help="Display the most recent lines of the application log.")
    p_tail.add_argument(
        "--lines", "-n", type=int, default=50, help="Number of trailing lines to view (default: 50)"
    )
    p_tail.add_argument("--file", "-f", type=str, default=None, help="Custom log file path")
    p_tail.set_defaults(func=run_logs_tail)

    # 2. Export
    p_exp = sub.add_parser("export", help="Export sanitized log file to a destination path.")
    p_exp.add_argument("output", help="Destination log file path")
    p_exp.add_argument("--file", "-f", type=str, default=None, help="Source log file path")
    p_exp.set_defaults(func=run_logs_export)

    # 3. Clear
    p_clr = sub.add_parser("clear", help="Clear or truncate application log file.")
    p_clr.add_argument("--yes", "-y", action="store_true", help="Confirm clearing log file")
    p_clr.add_argument("--file", "-f", type=str, default=None, help="Log file path to clear")
    p_clr.set_defaults(func=run_logs_clear)


def run_logs_tail(args: argparse.Namespace) -> int:
    log_p = Path(args.file) if getattr(args, "file", None) else DEFAULT_LOG_FILE
    if not log_p.is_file():
        console.print(f"[bold yellow]Log file '{log_p}' does not exist yet.[/bold yellow]")
        return 0

    lines_to_read = getattr(args, "lines", 50)
    try:
        with open(log_p, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines_to_read:]

        content = "".join(tail_lines)
        console.print(
            Panel(
                content.strip() or "[dim]Empty log[/dim]",
                title=f"[bold bright_cyan]Log Tail: {log_p.name} (Last {len(tail_lines)} lines)[/bold bright_cyan]",
                box=box.ROUNDED,
                border_style=COLOR_BORDER,
            )
        )
        return 0
    except OSError as e:
        console.print(f"[bold red]Failed to read log file: {e}[/bold red]")
        return 1


def run_logs_export(args: argparse.Namespace) -> int:
    src_p = Path(args.file) if getattr(args, "file", None) else DEFAULT_LOG_FILE
    out_p = Path(args.output) if Path(args.output).is_absolute() else (REPO_ROOT / args.output)

    if not src_p.is_file():
        console.print(f"[bold red]Source log file not found: {src_p}[/bold red]")
        return 1

    out_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src_p, out_p)
        console.print(f"[bold green]Log exported successfully to {out_p}[/bold green]")
        return 0
    except OSError as e:
        console.print(f"[bold red]Export failed: {e}[/bold red]")
        return 1


def run_logs_clear(args: argparse.Namespace) -> int:
    log_p = Path(args.file) if getattr(args, "file", None) else DEFAULT_LOG_FILE
    if not log_p.is_file():
        console.print(f"[dim]Log file '{log_p}' does not exist.[/dim]")
        return 0

    if not getattr(args, "yes", False):
        console.print(f"[bold yellow]Pass --yes to confirm clearing '{log_p}'.[/bold yellow]")
        return 0

    try:
        with open(log_p, "w", encoding="utf-8") as f:
            f.truncate(0)
        console.print(f"[bold green]Log file '{log_p}' cleared.[/bold green]")
        return 0
    except OSError as e:
        console.print(f"[bold red]Failed to clear log file: {e}[/bold red]")
        return 1
