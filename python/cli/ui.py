"""Rich-powered presentation, live monitoring, hardware telemetry, and UI contract for ZeroPawn."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import psutil
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Shared console
console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

# Theme color constants
COLOR_PRIMARY = "bright_cyan"
COLOR_SECONDARY = "cyan"
COLOR_ACCENT = "magenta"
COLOR_SUCCESS = "bold green"
COLOR_WARNING = "bold yellow"
COLOR_ERROR = "bold red"
COLOR_MUTED = "dim white"
COLOR_HIGHLIGHT = "bright_white"
COLOR_BORDER = "bright_blue"


def is_interactive() -> bool:
    """Return True if stdout and stdin are interactive terminals."""
    try:
        return sys.stdout.isatty() and sys.stdin.isatty() and not os.environ.get("NO_COLOR")
    except Exception:
        return False


def get_hardware_metrics() -> Dict[str, Any]:
    """Sample real-time hardware telemetry (CPU, RAM, GPU, VRAM).

    Returns measured values or honest 'N/A' strings when unavailable.
    Never fabricates metrics.
    """
    metrics: Dict[str, Any] = {
        "cpu_percent": "N/A",
        "ram_used_gb": "N/A",
        "ram_total_gb": "N/A",
        "ram_percent": "N/A",
        "gpu_name": "N/A (CPU-only)",
        "gpu_util": "N/A",
        "vram_used_mb": "N/A",
        "vram_total_mb": "N/A",
        "vram_percent": "N/A",
    }

    try:
        cpu = psutil.cpu_percent(interval=None)
        metrics["cpu_percent"] = f"{cpu:.1f}%"
    except Exception:
        pass

    try:
        mem = psutil.virtual_memory()
        metrics["ram_used_gb"] = f"{mem.used / (1024**3):.1f} GB"
        metrics["ram_total_gb"] = f"{mem.total / (1024**3):.1f} GB"
        metrics["ram_percent"] = f"{mem.percent:.1f}%"
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            dev_idx = 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            metrics["gpu_name"] = dev_name
            total_vram = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**2)
            alloc_vram = torch.cuda.memory_allocated(dev_idx) / (1024**2)
            metrics["vram_used_mb"] = f"{alloc_vram:.0f} MB"
            metrics["vram_total_mb"] = f"{total_vram:.0f} MB"
            metrics["vram_percent"] = (
                f"{(alloc_vram / total_vram * 100):.1f}%" if total_vram > 0 else "0.0%"
            )
            # If py3nvml or similar is not present, mark gpu compute util as active if tensors in memory
            metrics["gpu_util"] = "Active" if alloc_vram > 10 else "Idle"
        else:
            metrics["gpu_name"] = "CPU (No CUDA device)"
    except Exception:
        pass

    return metrics


def render_hardware_panel() -> Panel:
    """Render a compact hardware telemetry status panel."""
    hw = get_hardware_metrics()
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="left")

    cpu_text = f"[bold cyan]CPU:[/bold cyan] {hw['cpu_percent']}"
    ram_text = f"[bold cyan]RAM:[/bold cyan] {hw['ram_used_gb']} / {hw['ram_total_gb']} ({hw['ram_percent']})"
    gpu_text = f"[bold cyan]GPU:[/bold cyan] {hw['gpu_name']} | [bold cyan]VRAM:[/bold cyan] {hw['vram_used_mb']} / {hw['vram_total_mb']}"

    grid.add_row(cpu_text, ram_text, gpu_text)

    try:
        from python.hardware.profile import load_profile

        prof = load_profile()
        if prof and getattr(prof, "workloads", None):
            uci_b = prof.workloads.get("uci", {}).get("batch_size", 24)
            puz_b = prof.workloads.get("puzzles", {}).get("batch_size", 32)
            match_b = prof.workloads.get("match", {}).get("batch_size", 48)
            self_b = prof.workloads.get("selfplay", {}).get("batch_size", 64)
            c1 = f"[bold cyan]Autotuned Targets:[/] UCI [green]B={uci_b}[/green] | Puzzles [green]B={puz_b}[/green]"
            c2 = f"Match [green]B={match_b}[/green] | SelfPlay [green]B={self_b}[/green]"
            c3 = f"[dim]Profile: {prof.fingerprint}[/dim]"
            grid.add_row(c1, c2, c3)
    except Exception:
        pass

    return Panel(
        grid,
        title="[bold bright_blue]Hardware Telemetry[/bold bright_blue]",
        border_style=COLOR_BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def print_app_banner(subtitle: str = "Deep Reinforcement Learning Chess Engine") -> None:
    """Render the application header banner."""
    content = Text()
    content.append("ZEROPAWN CONTROL CENTER\n", style="bold bright_cyan")
    content.append(subtitle, style="dim white")
    console.print(
        Panel(
            content,
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            title="[bold magenta] ZeroPawn [/bold magenta]",
            title_align="center",
            padding=(0, 2),
        )
    )


def render_operation_header(
    op_name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[Any] = None,
) -> None:
    """Pre-run output contract: Display selected operation and effective configuration."""
    title_text = Text()
    title_text.append(f"OPERATION: {op_name.upper()}\n", style="bold bright_cyan")
    title_text.append(f"{description}\n\n", style="italic white")

    meta_parts = []
    if model_path:
        meta_parts.append(f"[bold cyan]Model:[/bold cyan] {model_path}")
    if device:
        meta_parts.append(f"[bold cyan]Device:[/bold cyan] {device}")
    if seed is not None:
        meta_parts.append(f"[bold cyan]Seed:[/bold cyan] {seed}")
    meta_parts.append(
        f"[bold cyan]Time:[/bold cyan] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    title_text.append(" | ".join(meta_parts))

    console.print(
        Panel(
            title_text,
            title="[bold green]Preparing Execution[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    if parameters:
        table = Table(
            title="Effective Configuration Parameters",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            show_header=True,
            padding=(0, 2),
        )
        table.add_column("Parameter", style="bright_white")
        table.add_column("Value", style="yellow")
        table.add_column("Type", style="dim")

        for k, v in sorted(parameters.items()):
            table.add_row(str(k), str(v), type(v).__name__)
        console.print(table)


def render_operation_summary(
    op_name: str,
    status: str,  # SUCCESS, FAILED, CANCELLED
    duration_sec: float,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Sequence[Union[str, Path, Tuple[str, int]]]] = None,
    warnings_count: int = 0,
    exit_code: int = 0,
    next_step_hint: Optional[str] = None,
) -> None:
    """Post-run output contract: Display final results, duration, metrics, and generated artifacts."""
    status_color = (
        "bold green"
        if status.upper() == "SUCCESS"
        else ("bold yellow" if status.upper() == "CANCELLED" else "bold red")
    )

    header = Text()
    header.append(f"EXECUTION SUMMARY: {op_name.upper()}\n", style="bold bright_cyan")
    header.append("Status: ", style="bold white")
    header.append(f"{status.upper()}  ", style=status_color)
    header.append(
        f"|  Duration: {duration_sec:.2f}s  |  Warnings: {warnings_count}  |  Exit Code: {exit_code}",
        style="white",
    )

    console.print(
        Panel(
            header,
            border_style="green"
            if status.upper() == "SUCCESS"
            else ("yellow" if status.upper() == "CANCELLED" else "red"),
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    if metrics and isinstance(metrics, dict):
        table = Table(
            title="Measured Results & Metrics",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 2),
        )
        table.add_column("Metric", style="bright_white")
        table.add_column("Measured Value", style="bold yellow")
        for k, v in metrics.items():
            table.add_row(str(k), str(v))
        console.print(table)

    if artifacts:
        art_table = Table(
            title="Generated Checkpoints & Output Artifacts",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 2),
        )
        art_table.add_column("Artifact Path", style="bright_white")
        art_table.add_column("Size", style="dim")

        for art in artifacts:
            if isinstance(art, tuple):
                p_str, size_bytes = art[0], art[1]
                size_str = f"{size_bytes / (1024**2):.2f} MB" if size_bytes >= 0 else "N/A"
                art_table.add_row(str(p_str), size_str)
            else:
                p = Path(art)
                if p.is_file():
                    sz = p.stat().st_size
                    sz_str = (
                        f"{sz / (1024**2):.2f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.1f} KB"
                    )
                    art_table.add_row(str(p), sz_str)
                else:
                    art_table.add_row(str(p), "Directory / Created")
        console.print(art_table)

    if next_step_hint:
        console.print(
            Panel(
                f"[bold cyan]Recommended Next Action:[/bold cyan] {next_step_hint}",
                box=box.ROUNDED,
                border_style="dim",
            )
        )


class LiveMonitoringDashboard:
    """Real-time monitoring dashboard supporting live metrics, progress, hardware, and logs."""

    def __init__(
        self,
        title: str,
        total_steps: Optional[int] = None,
        step_unit: str = "steps",
        refresh_per_second: float = 4.0,
    ) -> None:
        self.title = title
        self.total_steps = total_steps
        self.step_unit = step_unit
        self.refresh_per_second = refresh_per_second
        self.current_step = 0
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}
        self.recent_logs: List[str] = []
        self._live: Optional[Live] = None
        self._is_active = False

        self.progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30, complete_style="bold green", finished_style="bold bright_cyan"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self.task_id = self.progress.add_task(
            f"[cyan]{title}",
            total=total_steps if total_steps and total_steps > 0 else None,
        )

    def start(self) -> None:
        if not is_interactive():
            console.print(f"[bold cyan]Starting task:[/] {self.title}")
            return
        self._is_active = True
        self._live = Live(
            self._render_view(),
            console=console,
            refresh_per_second=self.refresh_per_second,
            transient=False,
        )
        self._live.start()

    def update(
        self,
        step: Optional[int] = None,
        advance: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        log_message: Optional[str] = None,
    ) -> None:
        if step is not None:
            self.current_step = step
            self.progress.update(self.task_id, completed=step)
        elif advance > 0:
            self.current_step += advance
            self.progress.advance(self.task_id, advance)

        if metrics:
            self.metrics.update(metrics)

        if log_message:
            ts = datetime.now().strftime("%H:%M:%S")
            self.recent_logs.append(f"[{ts}] {log_message}")
            if len(self.recent_logs) > 6:
                self.recent_logs.pop(0)

        if self._live and self._is_active:
            self._live.update(self._render_view())
        elif not is_interactive() and log_message:
            console.print(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")

    def stop(self) -> None:
        if self._live and self._is_active:
            self._is_active = False
            self._live.stop()
            self._live = None

    def __enter__(self) -> "LiveMonitoringDashboard":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def _render_view(self) -> Group:
        elapsed = time.time() - self.start_time
        hw = get_hardware_metrics()

        # Metrics table
        m_table = Table.grid(expand=True, padding=(0, 2))
        m_table.add_column(style="bold bright_white", justify="left")
        m_table.add_column(style="bold yellow", justify="left")
        m_table.add_column(style="bold bright_white", justify="left")
        m_table.add_column(style="bold yellow", justify="left")

        metric_items = list(self.metrics.items())
        for i in range(0, len(metric_items), 2):
            k1, v1 = metric_items[i]
            if i + 1 < len(metric_items):
                k2, v2 = metric_items[i + 1]
                m_table.add_row(f"{k1}:", str(v1), f"{k2}:", str(v2))
            else:
                m_table.add_row(f"{k1}:", str(v1), "", "")

        # System info line
        sys_info = (
            f"[dim]CPU: [bold cyan]{hw['cpu_percent']}[/bold cyan] | "
            f"RAM: [bold cyan]{hw['ram_used_gb']}/{hw['ram_total_gb']}[/bold cyan] | "
            f"GPU: [bold cyan]{hw['gpu_name']}[/bold cyan] (VRAM: [bold cyan]{hw['vram_used_mb']}[/bold cyan]) | "
            f"Elapsed: [bold cyan]{elapsed:.1f}s[/bold cyan][/dim]"
        )

        elements: List[Any] = [
            Panel(
                Group(
                    self.progress,
                    m_table if metric_items else Text("Running...", style="dim"),
                    Text.from_markup(f"\n{sys_info}"),
                ),
                title=f"[bold bright_cyan]{self.title}[/bold bright_cyan]",
                box=box.ROUNDED,
                border_style=COLOR_BORDER,
                padding=(0, 1),
            )
        ]

        if self.recent_logs:
            log_group = Group(
                *[Text.from_markup(f"[dim]{line}[/dim]") for line in self.recent_logs]
            )
            elements.append(
                Panel(
                    log_group,
                    title="[bold dim]Recent Output & Events[/bold dim]",
                    box=box.ROUNDED,
                    border_style="dim",
                    padding=(0, 1),
                )
            )

        return Group(*elements)
