"""Centralized Rich terminal theme, styling, and presentation utilities for ZeroPawn."""

from __future__ import annotations

from typing import Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Central console instance for the application
console = Console()

# Theme palette
THEME = {
    "primary": "bright_cyan",
    "secondary": "cyan",
    "accent": "magenta",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "muted": "dim white",
    "highlight": "bright_white",
    "border": "bright_blue",
    "panel_bg": "black",
    "table_header": "bold cyan",
    "info": "bold blue",
}


def styled_text(text: str, style_key: str = "primary") -> Text:
    return Text(text, style=THEME.get(style_key, style_key))


def banner_panel(title: str, subtitle: Optional[str] = None, border: str = "ROUNDED") -> Panel:
    content = Text()
    content.append(title, style=f"bold {THEME['primary']}")
    if subtitle:
        content.append("\n")
        content.append(subtitle, style=THEME["muted"])
    return Panel(
        content,
        box=getattr(box, border, box.ROUNDED),
        border_style=THEME["border"],
        padding=(1, 2),
        title=styled_text("  ZeroPawn  ", "accent"),
        title_align="center",
    )


def status_panel(status: str, message: str, status_style: str = "success") -> Panel:
    status_style_str = THEME.get(status_style, status_style)
    # Build content as a single Text with assembled segments
    status_text = Text(status.upper(), style=f"bold reverse {status_style_str}")
    msg_text = Text(f"  {message}", style=THEME["highlight"])
    content_text = Text()
    content_text.append_text(status_text)
    content_text.append_text(msg_text)
    return Panel(
        content_text,
        box=box.ROUNDED,
        border_style=status_style_str,
        padding=(1, 2),
    )


def info_box(title: str, content_lines: Sequence[str]) -> Panel:
    inner = "\n".join(f"  {line}" for line in content_lines)
    return Panel(
        Text.assemble(
            (f"\n{title}\n", f"bold underline {THEME['primary']}"),
            (f"\n{inner}\n", THEME["highlight"]),
        ),
        box=box.ROUNDED,
        border_style=THEME["border"],
        padding=(1, 2),
    )


def result_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> Table:
    table = Table(
        title=f"[bold {THEME['primary']}]{title}[/]",
        box=box.ROUNDED,
        header_style=THEME["table_header"],
        border_style=THEME["border"],
        padding=(0, 2),
        show_lines=True,
    )
    for col in columns:
        table.add_column(col, overflow="fold")
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    return table


def progress_display(description: str) -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn(f"[bold cyan]{description}[/]"),
        BarColumn(bar_width=40, complete_style=THEME["success"], finished_style=THEME["primary"]),
        TaskProgressColumn(),
        console=console,
    )
