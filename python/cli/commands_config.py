"""Configuration management subcommands for ZeroPawn: list, show, get, set, reset, diff, validate, import, export, and resolved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich import box
from rich.table import Table

from python.cli.config import (
    ALIAS_MAP,
    CONFIG_REGISTRY,
    DEFAULT_CONFIG_FILE,
    get_config_option,
    load_config_file,
    reset_config_option,
    resolve_configuration,
    save_config_file,
    set_config_option,
    validate_config_dict,
)
from python.cli.presets import get_preset_by_id
from python.cli.ui import COLOR_BORDER, console


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "config",
        aliases=["cfg", "configure"],
        help="Configuration management: list, inspect, set, reset, diff, validate, import, export, and resolve options.",
        description="Comprehensive configuration control: inspect all options, customize settings, and verify 4-tier precedence.",
    )
    sub = parser.add_subparsers(dest="config_subcommand", help="Configuration actions")

    # 1. List
    p_list = sub.add_parser(
        "list", help="List all recognized configuration fields across all subsystems."
    )
    p_list.add_argument(
        "--domain",
        "-d",
        type=str,
        default=None,
        help="Filter by domain prefix (e.g. training, selfplay, mcts, system)",
    )
    p_list.set_defaults(func=run_config_list)

    # 2. Show
    p_show = sub.add_parser(
        "show", help="Display all currently active settings in configs/config.json."
    )
    p_show.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to config file (default: configs/config.json)",
    )
    p_show.set_defaults(func=run_config_show)

    # 3. Get
    p_get = sub.add_parser(
        "get", help="Inspect a specific configuration option, its default, type, and status."
    )
    p_get.add_argument(
        "key", help="Canonical field key or alias (e.g. 'training.max_hours' or 'max_hours')"
    )
    p_get.add_argument("--file", "-f", type=str, default=None, help="Path to config file")
    p_get.set_defaults(func=run_config_get)

    # 4. Set
    p_set = sub.add_parser("set", help="Set a configuration value with validation and persistence.")
    p_set.add_argument("key", help="Canonical field key or alias")
    p_set.add_argument("value", help="New value to assign")
    p_set.add_argument("--file", "-f", type=str, default=None, help="Target config file")
    p_set.set_defaults(func=run_config_set)

    # 5. Reset
    p_reset = sub.add_parser(
        "reset", help="Reset a specific option or all options to built-in defaults."
    )
    p_reset.add_argument(
        "key", nargs="?", default="all", help="Key to reset, or 'all' to reset entire file"
    )
    p_reset.add_argument("--file", "-f", type=str, default=None, help="Target config file")
    p_reset.set_defaults(func=run_config_reset)

    # 6. Validate
    p_val = sub.add_parser(
        "validate", help="Validate a configuration file or current settings against schema."
    )
    p_val.add_argument(
        "file", nargs="?", default=None, help="Path to configuration file to validate"
    )
    p_val.set_defaults(func=run_config_validate)

    # 7. Diff
    p_diff = sub.add_parser(
        "diff", help="Compare two configuration files or compare config file against defaults."
    )
    p_diff.add_argument("file_a", nargs="?", default=None, help="First configuration file")
    p_diff.add_argument(
        "file_b",
        nargs="?",
        default=None,
        help="Second configuration file (optional; defaults to built-in)",
    )
    p_diff.set_defaults(func=run_config_diff)

    # 8. Export
    p_exp = sub.add_parser(
        "export", help="Export effective configuration to a standalone JSON file."
    )
    p_exp.add_argument("output", help="Destination JSON file path")
    p_exp.add_argument(
        "--preset", "-p", type=str, default=None, help="Incorporate preset into export"
    )
    p_exp.set_defaults(func=run_config_export)

    # 9. Import
    p_imp = sub.add_parser("import", help="Import configuration settings from a JSON file.")
    p_imp.add_argument("input", help="Source JSON file path")
    p_imp.add_argument(
        "--target", type=str, default=None, help="Target config file (default: configs/config.json)"
    )
    p_imp.set_defaults(func=run_config_import)

    # 10. Resolved
    p_res = sub.add_parser(
        "resolved", help="Display resolved configuration and source provenance table."
    )
    p_res.add_argument(
        "--preset", "-p", type=str, default=None, help="Selected preset to incorporate"
    )
    p_res.add_argument("--command", "-c", type=str, default="train", help="Target command context")
    p_res.add_argument(
        "--all", "-a", action="store_true", help="Show all settings across all domains"
    )
    p_res.set_defaults(func=run_config_resolved)


def run_config_list(args: argparse.Namespace) -> int:
    domain_filter = getattr(args, "domain", None)
    table = Table(
        title="[bold bright_cyan]Recognized ZeroPawn Configuration Fields[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Canonical Key", style="bright_white", no_wrap=True)
    table.add_column("Alias", style="yellow", no_wrap=True)
    table.add_column("Type", style="dim", justify="center")
    table.add_column("Default", style="cyan")
    table.add_column("Allowed / Bounds", style="magenta")
    table.add_column("Description", style="dim")

    count = 0
    for key, cf in sorted(CONFIG_REGISTRY.items()):
        if domain_filter and not key.startswith(domain_filter.lower()):
            continue
        range_str = (
            f"[{cf.min_val}..{cf.max_val}]"
            if cf.min_val is not None
            else (f"{cf.choices}" if cf.choices else cf.type_name)
        )
        table.add_row(
            cf.key,
            cf.alias,
            cf.type_name,
            str(cf.default),
            range_str,
            cf.description,
        )
        count += 1

    console.print(table)
    console.print(f"[dim]Total configuration fields displayed: {count}[/dim]")
    return 0


def run_config_show(args: argparse.Namespace) -> int:
    cfg_file = getattr(args, "file", None) or DEFAULT_CONFIG_FILE
    settings = load_config_file(cfg_file)

    if not settings:
        console.print(
            f"[bold yellow]No custom configuration overrides in {cfg_file}. All settings using built-in defaults.[/bold yellow]"
        )
        return 0

    table = Table(
        title=f"[bold bright_cyan]Active Configuration Settings ({cfg_file})[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
    )
    table.add_column("Key", style="bright_white")
    table.add_column("Value", style="bold yellow")
    table.add_column("Default", style="dim")
    table.add_column("Status", style="green")

    for k, v in sorted(settings.items()):
        cf = CONFIG_REGISTRY.get(k) or ALIAS_MAP.get(k)
        def_str = str(cf.default) if cf else "N/A"
        table.add_row(k, str(v), def_str, "[green]Custom Override[/green]")

    console.print(table)
    return 0


def run_config_get(args: argparse.Namespace) -> int:
    cfg_file = getattr(args, "file", None)
    val, cf = get_config_option(args.key, cfg_file)

    if cf is None:
        console.print(f"[bold red]Unknown configuration option '{args.key}'.[/bold red]")
        return 1

    table = Table(
        title=f"Configuration Option: [bold yellow]{cf.key}[/bold yellow]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="bright_white")

    table.add_row("Canonical Key", cf.key)
    table.add_row("CLI Alias", cf.alias)
    table.add_row("CLI Flags", ", ".join(cf.cli_flags))
    table.add_row("Type", cf.type_name)
    table.add_row("Current Value", str(val))
    table.add_row("Default Value", str(cf.default))
    table.add_row(
        "Allowed Values",
        f"[{cf.min_val}..{cf.max_val}]"
        if cf.min_val is not None
        else (str(cf.choices) if cf.choices else "any"),
    )
    table.add_row("Runtime Destination", cf.runtime_dest)
    table.add_row("Description", cf.description)
    table.add_row("Applicable Workflows", ", ".join(cf.applicable_workflows))

    console.print(table)
    return 0


def run_config_set(args: argparse.Namespace) -> int:
    cfg_file = getattr(args, "file", None)
    ok, msg = set_config_option(args.key, args.value, cfg_file)
    if ok:
        console.print(f"[bold green]{msg}[/bold green]")
        return 0
    else:
        console.print(f"[bold red]{msg}[/bold red]")
        return 1


def run_config_reset(args: argparse.Namespace) -> int:
    cfg_file = getattr(args, "file", None)
    ok, msg = reset_config_option(args.key, cfg_file)
    if ok:
        console.print(f"[bold green]{msg}[/bold green]")
        return 0
    else:
        console.print(f"[bold red]{msg}[/bold red]")
        return 1


def run_config_validate(args: argparse.Namespace) -> int:
    cfg_file = getattr(args, "file", None) or DEFAULT_CONFIG_FILE
    settings = load_config_file(cfg_file)
    errors = validate_config_dict(settings)
    if not errors:
        console.print(
            f"[bold green]Configuration '{cfg_file}' is VALID. All {len(settings)} overrides adhere to schema constraints.[/bold green]"
        )
        return 0
    else:
        console.print(
            f"[bold red]Configuration '{cfg_file}' has {len(errors)} validation error(s):[/bold red]"
        )
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        return 1


def run_config_diff(args: argparse.Namespace) -> int:
    file_a = args.file_a or str(DEFAULT_CONFIG_FILE)
    file_b = args.file_b

    dict_a = load_config_file(file_a)

    if file_b:
        dict_b = load_config_file(file_b)
        label_a = Path(file_a).name
        label_b = Path(file_b).name
    else:
        dict_b = {k: cf.default for k, cf in CONFIG_REGISTRY.items()}
        label_a = Path(file_a).name
        label_b = "Built-in Defaults"

    all_keys = sorted(set(dict_a.keys()) | set(dict_b.keys()))
    diff_rows = []
    for k in all_keys:
        va = dict_a.get(k, "<not set>")
        vb = dict_b.get(k, "<not set>")
        if va != vb:
            diff_rows.append((k, str(va), str(vb)))

    if not diff_rows:
        console.print(f"[bold green]No differences between {label_a} and {label_b}.[/bold green]")
        return 0

    table = Table(
        title=f"[bold bright_cyan]Configuration Diff: {label_a} vs {label_b}[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
    )
    table.add_column("Setting", style="bright_white")
    table.add_column(label_a, style="bold yellow")
    table.add_column(label_b, style="bold cyan")

    for k, va, vb in diff_rows:
        table.add_row(k, va, vb)

    console.print(table)
    return 0


def run_config_export(args: argparse.Namespace) -> int:
    preset_dict = get_preset_by_id(args.preset) if args.preset else None
    resolved = resolve_configuration(preset_dict=preset_dict)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.0",
        "exported_at": json.loads(json.dumps(Path.cwd().name)),  # dummy trigger
        "preset": args.preset,
        "configuration": resolved.as_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    console.print(
        f"[bold green]Resolved configuration successfully exported to {out_path}[/bold green]"
    )
    return 0


def run_config_import(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.is_file():
        console.print(f"[bold red]Import file not found: {in_path}[/bold red]")
        return 1

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            settings = data.get("settings", data.get("configuration", data))
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Failed to parse JSON: {e}[/bold red]")
        return 1

    errors = validate_config_dict(settings)
    if errors:
        console.print(f"[bold red]Cannot import: {len(errors)} validation errors:[/bold red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        return 1

    ok = save_config_file(settings, args.target)
    if ok:
        console.print(
            f"[bold green]Successfully imported {len(settings)} configuration options to {args.target or DEFAULT_CONFIG_FILE}[/bold green]"
        )
        return 0
    return 1


def run_config_resolved(args: argparse.Namespace) -> int:
    preset_dict = get_preset_by_id(args.preset) if getattr(args, "preset", None) else None
    resolved = resolve_configuration(
        command_name=getattr(args, "command", "train"),
        preset_dict=preset_dict,
    )
    table = resolved.summary_table(show_all=getattr(args, "all", False))
    console.print(table)
    return 0
