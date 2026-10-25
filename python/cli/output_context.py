"""Unified output architecture for ZeroPawn.

Provides three explicit output modes:
1. RICH: Full Rich terminal UI when invoked from the main interactive CLI.
2. PLAIN: Clean, cp437-safe plain-text output when subordinate scripts or components
   are run standalone.
3. PROTOCOL: Strict machine-readable protocol streams (UCI, JSON, subprocess comms),
   guaranteeing zero ANSI escapes or decor on stdout.
"""

from __future__ import annotations

import contextvars
import enum
import sys
import threading
from typing import Any, Dict, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class OutputMode(str, enum.Enum):
    """Execution output presentation modes."""

    RICH = "rich"
    PLAIN = "plain"
    PROTOCOL = "protocol"


# Context variable for the active thread / async context
_current_output_context: contextvars.ContextVar[Optional["OutputContext"]] = contextvars.ContextVar(
    "_current_output_context", default=None
)


class OutputContext:
    """Manages output formatting and dispatch according to the active mode."""

    def __init__(
        self,
        mode: Optional[OutputMode | str] = None,
        console: Optional[Console] = None,
        err_console: Optional[Console] = None,
    ) -> None:
        if mode is None:
            # Detect from environment or default to PLAIN for standalone scripts
            from python.common.env import get_env

            env_mode = (get_env("OUTPUT_MODE", "") or "").lower().strip()
            if env_mode == "rich":
                self.mode = OutputMode.RICH
            elif env_mode == "protocol":
                self.mode = OutputMode.PROTOCOL
            else:
                self.mode = OutputMode.PLAIN
        elif isinstance(mode, str):
            self.mode = OutputMode(mode.lower().strip())
        else:
            self.mode = mode

        self.console = console or Console(highlight=False)
        self.err_console = err_console or Console(stderr=True, highlight=False)
        self._lock = threading.Lock()

    @classmethod
    def get_current(cls) -> "OutputContext":
        """Get the current active output context, or initialize a default one."""
        ctx = _current_output_context.get()
        if ctx is None:
            ctx = cls()
            _current_output_context.set(ctx)
        return ctx

    @classmethod
    def set_current(cls, ctx: "OutputContext") -> None:
        """Set the active output context."""
        _current_output_context.set(ctx)

    def is_rich(self) -> bool:
        if self.mode == OutputMode.PROTOCOL or self.mode == OutputMode.PLAIN:
            return False
        from python.common.env import get_env

        if get_env("OUTPUT_MODE") == "plain":
            return False
        if get_env("FORCE_RICH") == "1" or get_env("OUTPUT_MODE") == "rich":
            return True
        return self.mode == OutputMode.RICH and sys.stdout.isatty()

    def is_protocol(self) -> bool:
        return self.mode == OutputMode.PROTOCOL

    def is_plain(self) -> bool:
        return not self.is_rich() and not self.is_protocol()

    def emit_line(self, line: str, is_stderr: bool = False) -> None:
        """Emit a single line respecting output mode."""
        with self._lock:
            if self.mode == OutputMode.PROTOCOL:
                target = sys.stderr if is_stderr else sys.stdout
                target.write(line + "\n")
                target.flush()
            elif self.is_rich():
                con = self.err_console if is_stderr else self.console
                con.print(line)
            else:
                target = sys.stderr if is_stderr else sys.stdout
                target.write(line + "\n")
                target.flush()

    def _try_render_rich_report(self, text: str) -> bool:
        """Attempt to parse structured report text into Rich elements. Returns True if rendered."""
        stripped = text.strip()
        lines = [l.strip() for l in stripped.splitlines() if l.strip()]
        if not lines:
            return False

        # 1. Operation Banner (=== OPERATION: ... ===)
        if len(lines) >= 3 and lines[0].startswith("===") and lines[-1].startswith("==="):
            op_title = lines[1].replace("OPERATION:", "").strip()
            subtitle = lines[2].strip() if len(lines) > 3 and not lines[2].startswith("===") else ""
            content = Text()
            content.append(f"OPERATION: {op_title.upper()}\n", style="bold bright_cyan")
            if subtitle:
                content.append(subtitle, style="dim white")
            self.console.print(
                Panel(
                    content,
                    title="[bold magenta] ZeroPawn [/bold magenta]",
                    border_style="bright_blue",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
            )
            return True

        # 2. Run Identity Header (START: ..., GIT: ...)
        if any(l.startswith("START:") for l in lines) and any(l.startswith("GIT:") for l in lines):
            grid = Table.grid(padding=(0, 2))
            grid.add_column(style="bold cyan", justify="right")
            grid.add_column(style="white")
            for l in lines:
                if l.startswith("---"):
                    continue
                if ":" in l:
                    k, v = l.split(":", 1)
                    grid.add_row(k.strip() + ":", v.strip())
            self.console.print(
                Panel(
                    grid,
                    title="[bold green]Run Identity[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            return True

        # 3. Parameters Table (PARAMETERS (name = value [source]): ...)
        if lines[0].startswith("PARAMETERS"):
            table = Table(
                title="Effective Configuration Parameters",
                box=box.ROUNDED,
                border_style="bright_blue",
                header_style="bold cyan",
                padding=(0, 2),
            )
            table.add_column("Parameter", style="bright_white")
            table.add_column("Value", style="bold yellow")
            table.add_column("Source", style="dim")
            for l in lines[1:]:
                if not l or l.startswith("---"):
                    continue
                if "=" in l:
                    k_part, rest = l.split("=", 1)
                    name = k_part.strip()
                    val = rest.strip()
                    source = "default"
                    if val.endswith("]") and "[" in val:
                        val_part, src_part = val.rsplit("[", 1)
                        val = val_part.strip()
                        source = src_part.rstrip("]").strip()
                    table.add_row(name, val, source)
            self.console.print(table)
            return True

        # 4. Results Table (RESULTS: ...)
        if lines[0] == "RESULTS:":
            table = Table(
                title="Measured Results & Metrics",
                box=box.ROUNDED,
                border_style="bright_blue",
                header_style="bold cyan",
                padding=(0, 2),
            )
            table.add_column("Metric", style="bright_white")
            table.add_column("Measured Value", style="bold yellow")
            for l in lines[1:]:
                if not l or l.startswith("---"):
                    continue
                if ":" in l:
                    k, v = l.split(":", 1)
                    table.add_row(k.strip(), v.strip())
            self.console.print(table)
            return True

        # 5. Artifacts List (ARTIFACTS (path, size_bytes, sha256): ...)
        if lines[0].startswith("ARTIFACTS"):
            table = Table(
                title="Generated Checkpoints & Output Artifacts",
                box=box.ROUNDED,
                border_style="bright_blue",
                header_style="bold cyan",
                padding=(0, 2),
            )
            table.add_column("Artifact Path", style="bright_white")
            table.add_column("Size", style="dim")
            table.add_column("SHA-256", style="dim cyan")
            for l in lines[1:]:
                parts = l.split()
                if len(parts) >= 3:
                    p, sz, sha = parts[0], parts[1], parts[2]
                    try:
                        sz_int = int(sz)
                        sz_str = f"{sz_int / (1024**2):.2f} MB" if sz_int >= 0 else "N/A"
                    except ValueError:
                        sz_str = sz
                    table.add_row(p, sz_str, sha)
                elif len(parts) == 2:
                    table.add_row(parts[0], parts[1], "-")
                elif l and not l.startswith("---"):
                    table.add_row(l, "-", "-")
            self.console.print(table)
            return True

        # 6. Summary Block (STATUS: ... DURATION: ...)
        for l in lines:
            if l.startswith("STATUS:"):
                parts = l.split()
                st, dur, warn, ex = "UNKNOWN", "0s", "0", "0"
                for i, p in enumerate(parts):
                    if p == "STATUS:" and i + 1 < len(parts):
                        st = parts[i + 1]
                    elif p == "DURATION:" and i + 1 < len(parts):
                        dur = parts[i + 1]
                    elif p == "WARNINGS:" and i + 1 < len(parts):
                        warn = parts[i + 1]
                    elif p == "EXIT:" and i + 1 < len(parts):
                        ex = parts[i + 1]
                st_color = (
                    "bold green"
                    if st == "SUCCESS"
                    else ("bold yellow" if st == "INTERRUPTED" else "bold red")
                )
                border = (
                    "green" if st == "SUCCESS" else ("yellow" if st == "INTERRUPTED" else "red")
                )
                header = Text()
                header.append("EXECUTION STATUS: ", style="bold white")
                header.append(f"{st}  ", style=st_color)
                header.append(
                    f"|  Duration: {dur}  |  Warnings: {warn}  |  Exit Code: {ex}", style="white"
                )
                self.console.print(
                    Panel(
                        header,
                        title="[bold green]Execution Summary[/bold green]",
                        border_style=border,
                        box=box.ROUNDED,
                        padding=(0, 2),
                    )
                )
                return True

        # 7. Key=Value Progress Line ([HH:MM:SS] k1=v1 k2=v2 ...)
        if stripped.startswith("[") and "]" in stripped and "=" in stripped:
            ts_part, rest = stripped.split("]", 1)
            t = Text()
            t.append(ts_part + "] ", style="bold cyan")
            for part in rest.strip().split(" "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    t.append(k, style="dim white")
                    t.append("=", style="dim")
                    t.append(v, style="bold yellow")
                else:
                    t.append(part, style="white")
                t.append(" ")
            self.console.print(t)
            return True

        return False

    def render_report(self, text: str) -> None:
        """Render a report block emitted by python.common.progress."""
        with self._lock:
            if self.is_rich():
                rendered = self._try_render_rich_report(text)
                if not rendered:
                    self.console.print(text)
            elif self.mode == OutputMode.PROTOCOL:
                sys.stderr.write(text + "\n")
                sys.stderr.flush()
            else:
                sys.stdout.write(text + "\n")
                sys.stdout.flush()

    def render_error(self, text: str) -> None:
        """Render an error block emitted by python.common.progress."""
        with self._lock:
            if self.is_rich():
                self.err_console.print(
                    Panel(
                        Text(text, style="bold red"),
                        title="[bold red]Error[/bold red]",
                        border_style="red",
                        box=box.ROUNDED,
                    )
                )
            else:
                sys.stderr.write(text + "\n")
                sys.stderr.flush()

    def render_header(
        self,
        op_name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """Render standard operation header."""
        if self.mode == OutputMode.RICH:
            from python.cli.ui import render_operation_header

            render_operation_header(op_name, description, parameters, model_path)
        elif self.mode == OutputMode.PLAIN:
            from python.common.progress import banner, header_block, parameters_table, run_context

            ctx = run_context(op_name)
            sys.stdout.write(banner(op_name) + "\n")
            sys.stdout.write(
                header_block(
                    ctx["start"],
                    ctx["command"],
                    ctx["workdir"],
                    ctx["commit"],
                    "none",
                    model_path or "none",
                    "auto",
                    "none",
                )
                + "\n"
            )
            if parameters:
                rows = [(k, v, "user") for k, v in parameters.items()]
                sys.stdout.write(parameters_table(rows) + "\n")
            sys.stdout.flush()

    def render_summary(
        self,
        op_name: str,
        status: str,
        duration_sec: float,
        exit_code: int,
    ) -> None:
        """Render standard operation summary."""
        if self.mode == OutputMode.RICH:
            from python.cli.ui import render_operation_summary

            render_operation_summary(
                op_name=op_name,
                status=status,
                duration_sec=duration_sec,
                exit_code=exit_code,
            )
        elif self.mode == OutputMode.PLAIN:
            from python.common.progress import summary_block

            sys.stdout.write(
                summary_block(status, duration_sec, 0 if exit_code == 0 else 1, exit_code) + "\n"
            )
            sys.stdout.flush()


def set_output_mode(mode: OutputMode | str) -> None:
    """Convenience helper to set active output mode and propagate to environment."""
    ctx = OutputContext.get_current()
    if isinstance(mode, str):
        mode = OutputMode(mode.lower().strip())
    ctx.mode = mode
    from python.common.env import set_env

    set_env("OUTPUT_MODE", mode.value)


def get_output_mode() -> OutputMode:
    """Return current output mode."""
    return OutputContext.get_current().mode
