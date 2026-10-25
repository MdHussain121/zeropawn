"""Preset management subcommands for ZeroPawn."""

from __future__ import annotations

import argparse

from python.cli.presets import (
    delete_preset,
    duplicate_preset,
    export_preset,
    get_preset_by_id,
    import_preset,
    render_presets_table,
)
from python.cli.ui import COLOR_BORDER, console
from rich import box
from rich.table import Table


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    presets_parser = subparsers.add_parser(
        "presets",
        aliases=["preset"],
        help="Preset management: list, inspect, run, duplicate, delete, import, and export.",
        description="Manage built-in and user-defined experiment and workload presets.",
    )
    sub = presets_parser.add_subparsers(dest="presets_subcommand", help="Preset actions")

    # 1. List
    p_list = sub.add_parser("list", help="List all available presets.")
    p_list.set_defaults(func=run_presets_list)

    # 2. Show
    p_show = sub.add_parser("show", help="Show details and parameters of a preset.")
    p_show.add_argument("preset_id", help="Preset ID to inspect")
    p_show.set_defaults(func=run_presets_show)

    # 3. Duplicate
    p_dup = sub.add_parser("duplicate", help="Duplicate a preset into a new user preset.")
    p_dup.add_argument("source_id", help="Source preset ID")
    p_dup.add_argument("new_id", help="New preset ID")
    p_dup.add_argument("new_name", help="New preset display name")
    p_dup.set_defaults(func=run_presets_duplicate)

    # 4. Delete
    p_del = sub.add_parser("delete", help="Delete a user preset.")
    p_del.add_argument("preset_id", help="Preset ID to delete")
    p_del.set_defaults(func=run_presets_delete)

    # 5. Export
    p_exp = sub.add_parser("export", help="Export a preset to a JSON file.")
    p_exp.add_argument("preset_id", help="Preset ID to export")
    p_exp.add_argument("output_file", help="Destination JSON file path")
    p_exp.set_defaults(func=run_presets_export)

    # 6. Import
    p_imp = sub.add_parser("import", help="Import a preset from a JSON file.")
    p_imp.add_argument("input_file", help="Source JSON file path")
    p_imp.set_defaults(func=run_presets_import)

    # 7. Run
    p_run = sub.add_parser("run", help="Run an operation using a preset.")
    p_run.add_argument("preset_id", help="Preset ID to execute")
    p_run.add_argument(
        "extra_args", nargs=argparse.REMAINDER, help="Optional CLI overrides passed to preset"
    )
    p_run.set_defaults(func=run_preset_exec)

    # 8. Diff
    p_diff = sub.add_parser("diff", help="Compare two presets side-by-side.")
    p_diff.add_argument("preset_a", help="First preset ID")
    p_diff.add_argument("preset_b", help="Second preset ID")
    p_diff.set_defaults(func=run_presets_diff)


def run_presets_list(args: argparse.Namespace) -> int:
    render_presets_table()
    return 0


def run_presets_show(args: argparse.Namespace) -> int:
    p = get_preset_by_id(args.preset_id)
    if not p:
        console.print(f"[bold red]Preset '{args.preset_id}' not found.[/bold red]")
        return 1

    table = Table(
        title=f"Preset Details: {p.get('name')}", box=box.ROUNDED, border_style=COLOR_BORDER
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="bold yellow")

    table.add_row("ID", str(p.get("id")))
    table.add_row("Type", "Built-in" if p.get("is_builtin") else "User-defined")
    table.add_row("Operation", str(p.get("operation")))
    table.add_row("Subcommand", str(p.get("subcommand", "None")))
    table.add_row("Description", str(p.get("description", "")))
    table.add_row("CLI Arguments", " ".join(p.get("args", [])))
    table.add_row("Model", str(p.get("model", "None")))
    if "tradeoff" in p:
        table.add_row("Documented Tradeoff", str(p["tradeoff"]))

    console.print(table)
    return 0


def run_presets_duplicate(args: argparse.Namespace) -> int:
    ok = duplicate_preset(args.source_id, args.new_id, args.new_name)
    if ok:
        console.print(
            f"[bold green]Preset '{args.source_id}' duplicated as '{args.new_id}' successfully.[/bold green]"
        )
        return 0
    return 1


def run_presets_delete(args: argparse.Namespace) -> int:
    ok = delete_preset(args.preset_id)
    if ok:
        console.print(f"[bold green]User preset '{args.preset_id}' deleted.[/bold green]")
        return 0
    return 1


def run_presets_export(args: argparse.Namespace) -> int:
    ok = export_preset(args.preset_id, args.output_file)
    if ok:
        console.print(f"[bold green]Preset exported to {args.output_file}[/bold green]")
        return 0
    return 1


def run_presets_import(args: argparse.Namespace) -> int:
    ok = import_preset(args.input_file)
    if ok:
        console.print(f"[bold green]Preset imported from {args.input_file}[/bold green]")
        return 0
    return 1


def run_preset_exec(args: argparse.Namespace) -> int:
    from python.cli.main import build_parser

    p = get_preset_by_id(args.preset_id)
    if not p:
        console.print(f"[bold red]Preset '{args.preset_id}' not found.[/bold red]")
        return 1

    cmd_tokens = [p["operation"]]
    if p.get("subcommand"):
        cmd_tokens.append(p["subcommand"])
    cmd_tokens.extend(p.get("args", []))
    if getattr(args, "extra_args", None):
        cmd_tokens.extend(args.extra_args)

    parser = build_parser()
    sub_args = parser.parse_args(cmd_tokens)
    if hasattr(sub_args, "func"):
        return sub_args.func(sub_args)
    else:
        console.print(
            f"[bold red]Cannot dispatch command for preset '{args.preset_id}'.[/bold red]"
        )
        return 1


def run_presets_diff(args: argparse.Namespace) -> int:
    p_a = get_preset_by_id(args.preset_a)
    p_b = get_preset_by_id(args.preset_b)

    if not p_a:
        console.print(f"[bold red]Preset '{args.preset_a}' not found.[/bold red]")
        return 1
    if not p_b:
        console.print(f"[bold red]Preset '{args.preset_b}' not found.[/bold red]")
        return 1

    table = Table(
        title=f"[bold bright_cyan]Preset Comparison: {p_a.get('name')} vs {p_b.get('name')}[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
    )
    table.add_column("Property", style="bright_white")
    table.add_column(p_a.get("name", args.preset_a), style="bold yellow")
    table.add_column(p_b.get("name", args.preset_b), style="bold cyan")

    table.add_row("ID", str(p_a.get("id")), str(p_b.get("id")))
    table.add_row(
        "Type",
        "Built-in" if p_a.get("is_builtin") else "User",
        "Built-in" if p_b.get("is_builtin") else "User",
    )
    table.add_row("Operation", str(p_a.get("operation")), str(p_b.get("operation")))
    table.add_row(
        "Subcommand", str(p_a.get("subcommand", "None")), str(p_b.get("subcommand", "None"))
    )
    table.add_row("CLI Arguments", " ".join(p_a.get("args", [])), " ".join(p_b.get("args", [])))
    table.add_row("Model", str(p_a.get("model", "None")), str(p_b.get("model", "None")))

    # Parameters diff
    params_a = p_a.get("parameters", {})
    params_b = p_b.get("parameters", {})
    all_param_keys = sorted(set(params_a.keys()) | set(params_b.keys()))
    for k in all_param_keys:
        va = str(params_a.get(k, "<not set>"))
        vb = str(params_b.get(k, "<not set>"))
        table.add_row(f"Param: {k}", va, vb)

    if "tradeoff" in p_a or "tradeoff" in p_b:
        table.add_row("Tradeoff", str(p_a.get("tradeoff", "")), str(p_b.get("tradeoff", "")))

    console.print(table)
    return 0
