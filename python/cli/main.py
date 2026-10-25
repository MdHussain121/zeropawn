"""Main dispatcher, parser registration, and Rich interactive REPL menu for ZeroPawn."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table

from python.cli import (
    commands_add_puzzles,
    commands_bench,
    commands_book,
    commands_build,
    commands_cleanup,
    commands_config,
    commands_data,
    commands_diagnose,
    commands_eval,
    commands_finetune,
    commands_hardware,
    commands_logs,
    commands_match,
    commands_models,
    commands_presets,
    commands_profile,
    commands_puzzles,
    commands_runs,
    commands_search,
    commands_status,
    commands_tablebase,
    commands_test,
    commands_train,
    commands_uci,
)
from python.cli.common import (
    print_error,
    print_warning,
    setup_environment,
    wait_for_enter,
)
from python.cli.output_context import OutputMode, set_output_mode
from python.cli.models import get_default_model_path, select_model_interactive
from python.cli.operations import (
    CATEGORIES,
    OperationDef,
    get_operations_by_category,
)
from python.cli.presets import get_preset_by_id, render_presets_table
from python.cli.ui import COLOR_BORDER, console, print_app_banner, render_hardware_panel
from python.cli.wizard import enter_custom_arguments_mode, run_configuration_wizard


def build_parser() -> argparse.ArgumentParser:
    """Construct unified ArgumentParser with all operations, subcommands, and aliases."""
    parser = argparse.ArgumentParser(
        prog="zeropawn",
        description="Unified CLI for ZeroPawn: Deep Reinforcement Learning Chess Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zeropawn test --suite fast                   # Run fast unit test suites
  zeropawn test -k "puzzles or mcts"          # Run specific test cases
  zeropawn train --supervised                 # Supervised learning warm-start
  zeropawn train --max-hours 12               # Continue RL self-play training
  zeropawn train --verify                     # Quick 30s smoke verification
  zeropawn selfplay                           # Start RL self-play training loop
  zeropawn finetune                           # Fine-tune checkpoint on dataset
  zeropawn match -n 20                        # Play 20-game match vs Stockfish
  zeropawn match live --engine-color white    # Watch AI play Stockfish move-by-move
  zeropawn match human                        # Play against our AI engine
  zeropawn eval baseline --elo 1500           # Calibrated Stockfish Elo baseline
  zeropawn puzzles eval                       # Benchmark policy & MCTS on puzzles
  zeropawn puzzles solve --human-guess        # Step through tactics interactively
  zeropawn add-puzzles fetch -n 1000          # Download 1000 puzzles from Lichess
  zeropawn data list                          # Scan and list all discovered datasets
  zeropawn data validate                      # Validate binary memmaps and schema
  zeropawn data filter --input games.pgn      # Filter PGN games by Elo
  zeropawn bench selfplay                     # Throughput & simulation benchmark
  zeropawn diagnose lag                       # Measure model version lag
  zeropawn tablebase download                 # Download Syzygy 3-4-5 tablebases
  zeropawn models list                        # Inspect discovered checkpoints
  zeropawn presets list                       # List all built-in and user presets
  zeropawn status                             # View GPU, checkpoints, and telemetry
  zeropawn uci                                # Run as UCI protocol engine for GUIs
  zeropawn interactive                        # Launch guided interactive menu
""",
    )

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="ZeroPawn 1.0.0 (Deep ZeroPawn Chess Engine)",
        help="Show version information and exit.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Disable interactive wizards and prompts.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file overriding defaults.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Active preset name to resolve before CLI overrides.",
    )
    parser.add_argument(
        "--print-resolved",
        action="store_true",
        default=False,
        help="Print resolved configuration and provenance before executing command.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Core existing commands
    commands_test.register_parser(subparsers)
    commands_train.register_parser(subparsers)
    commands_match.register_parser(subparsers)
    commands_puzzles.register_parser(subparsers)
    commands_add_puzzles.register_parser(subparsers)
    commands_status.register_parser(subparsers)
    commands_uci.register_parser(subparsers)
    commands_cleanup.register_parser(subparsers)

    # Registered parity commands
    commands_finetune.register_parser(subparsers)
    commands_data.register_parser(subparsers)
    commands_eval.register_parser(subparsers)
    commands_bench.register_parser(subparsers)
    commands_diagnose.register_parser(subparsers)
    commands_tablebase.register_parser(subparsers)
    commands_book.register_parser(subparsers)
    commands_profile.register_parser(subparsers)
    commands_models.register_parser(subparsers)
    commands_presets.register_parser(subparsers)

    # Universal Configuration & System Operations
    commands_config.register_parser(subparsers)
    commands_runs.register_parser(subparsers)
    commands_logs.register_parser(subparsers)
    commands_build.register_parser(subparsers)
    commands_hardware.register_parser(subparsers)
    commands_search.register_parser(subparsers)

    # Interactive subcommand
    interactive_parser = subparsers.add_parser(
        "interactive",
        aliases=["menu", "gui"],
        help="Launch the interactive terminal menu and command discovery tools.",
    )
    interactive_parser.add_argument(
        "--browser",
        "-b",
        action="store_true",
        default=False,
        help="Launch directly into the Complete Command Browser (114+ commands).",
    )
    interactive_parser.add_argument(
        "--combinations",
        "-e",
        metavar="COMMAND",
        type=str,
        default=None,
        help="Explore valid option combinations for a command (e.g. 'train' or 'match run').",
    )
    interactive_parser.add_argument(
        "--options",
        "-o",
        metavar="COMMAND",
        type=str,
        default=None,
        help="Inspect complete argument inventory for a command.",
    )
    interactive_parser.add_argument(
        "--builder",
        metavar="COMMAND",
        type=str,
        default=None,
        help="Open interactive argument builder for a command.",
    )
    interactive_parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number when viewing combinations.",
    )
    interactive_parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Combinations per page.",
    )
    interactive_parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Disable interactive wizards and prompts.",
    )
    interactive_parser.add_argument(
        "--filter",
        "-f",
        dest="filter_query",
        type=str,
        default=None,
        help="Filter combinations by flag or keyword.",
    )
    interactive_parser.set_defaults(func=handle_interactive_command)

    return parser


def handle_interactive_command(args: argparse.Namespace) -> int:
    """Dispatcher for interactive subcommand with direct tool routing."""
    from python.cli.discovery import get_command_catalog
    from python.cli.combinations import get_combination_engine
    from python.cli.interactive_discovery import (
        run_command_browser,
        run_option_explorer,
        run_argument_builder,
        run_combination_explorer,
    )

    catalog = get_command_catalog()

    if getattr(args, "browser", False):
        run_command_browser(non_interactive=getattr(args, "non_interactive", False))
        return 0

    if getattr(args, "options", None):
        node = catalog.get_command(args.options)
        if not node:
            console.print(f"[bold red]Command '{args.options}' not found in catalog.[/bold red]")
            return 1
        if getattr(args, "non_interactive", False):
            table = Table(title=f"Option Inventory for {node.full_command}", box=box.ROUNDED)
            table.add_column("Argument", style="bold yellow")
            table.add_column("Type", style="cyan")
            table.add_column("Default", style="dim")
            table.add_column("Choices / Constraints", style="dim")
            table.add_column("Description")
            for a in node.arguments:
                table.add_row(a.display_name, a.type, str(a.default), str(a.choices or "-"), a.help)
            console.print(table)
            return 0
        run_option_explorer(node)
        return 0

    if getattr(args, "combinations", None):
        node = catalog.get_command(args.combinations)
        if not node:
            console.print(
                f"[bold red]Command '{args.combinations}' not found in catalog.[/bold red]"
            )
            return 1
        if getattr(args, "non_interactive", False):
            engine = get_combination_engine()
            combos, total, pages = engine.get_paginated_combinations(
                node,
                page=getattr(args, "page", 1),
                page_size=getattr(args, "page_size", 20),
                filter_query=getattr(args, "filter_query", None),
                include_invalid=True,
            )
            table = Table(
                title=f"Combinations for {node.full_command} (Page {args.page}/{pages}, Total {total})",
                box=box.ROUNDED,
            )
            table.add_column("#", justify="right", style="bold yellow")
            table.add_column("Mode", style="cyan")
            table.add_column("Configuration", style="bright_white")
            table.add_column("Valid", style="bold")
            table.add_column("Notes", style="dim")
            for c in combos:
                valid_str = "[green]YES[/green]" if c.is_valid else "[red]NO[/red]"
                table.add_row(
                    str(c.index),
                    c.mode or "default",
                    c.summary_description(),
                    valid_str,
                    c.rejection_reason or "OK",
                )
            console.print(table)
            return 0
        run_combination_explorer(node)
        return 0

    if getattr(args, "builder", None):
        node = catalog.get_command(args.builder)
        if not node:
            console.print(f"[bold red]Command '{args.builder}' not found in catalog.[/bold red]")
            return 1
        run_argument_builder(node)
        return 0

    return run_interactive_menu()


def get_downloaded_puzzles_info(
    puzzles_dir: str | Path = "data/puzzles",
) -> tuple[int, list[tuple[Path, int]]]:
    """Scan puzzles directory, count total downloaded puzzles and return per-file counts."""
    pdir = Path(puzzles_dir)
    if not pdir.is_dir():
        return 0, []

    files_info: list[tuple[Path, int]] = []
    total = 0
    for f in sorted(pdir.glob("*.csv")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reader = csv.reader(fp)
                next(reader, None)
                count = sum(1 for row in reader if row and len(row) >= 4)
                if count > 0:
                    files_info.append((f, count))
                    total += count
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
    return total, files_info


def execute_operation_from_interactive(
    op: OperationDef,
    params: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> int:
    """Execute any operation using its registered CLI parser and arguments."""
    parser = build_parser()

    # Build token list for argparse
    tokens: List[str] = []

    # Handle direct command prefix
    cmd_parts = op.direct_command.split()
    if cmd_parts and cmd_parts[0] in ("chess", "zeropawn"):
        cmd_parts = cmd_parts[1:]

    # Primary subcommand
    if cmd_parts:
        tokens.append(cmd_parts[0])

    # Secondary subcommand (e.g. 'run', 'eval', 'download', etc.) and fixed flags (e.g. '--verify')
    arg_flags = {a.flag for a in op.arguments}
    for part in cmd_parts[1:]:
        if part.startswith(("<", "[")):
            continue
        if part.startswith("-"):
            if part not in arg_flags and part not in tokens:
                tokens.append(part)
        else:
            tokens.append(part)

    # Map parameters to command flags / positional arguments
    for arg_def in op.arguments:
        val = params.get(arg_def.name, arg_def.default)
        if getattr(arg_def, "is_positional", False):
            if val is not None and str(val).strip() != "":
                tokens.append(str(val))
            elif arg_def.default is not None and str(arg_def.default).strip() != "":
                tokens.append(str(arg_def.default))
        else:
            if val != arg_def.default or arg_def.required:
                if arg_def.type == "bool":
                    if val:
                        tokens.append(arg_def.flag)
                elif val is not None:
                    if arg_def.flag == "--inputs" and isinstance(val, str) and " " in val:
                        tokens.append(arg_def.flag)
                        tokens.extend(val.split())
                    else:
                        tokens.extend([arg_def.flag, str(val)])

    # Target subparser inspection for intelligent checkpoint appending
    if model_path and op.requires_model != "none":
        curr: argparse.ArgumentParser = parser
        for tok in tokens:
            subaction = next(
                (a for a in curr._actions if isinstance(a, argparse._SubParsersAction)), None
            )
            if subaction and tok in subaction.choices:
                curr = subaction.choices[tok]
            else:
                break

        option_strings: set[str] = set()
        positional_dests: list[str] = []
        for action in curr._actions:
            if action.option_strings:
                option_strings.update(action.option_strings)
            elif action.dest != "help":
                positional_dests.append(action.dest)

        if "--checkpoint" in option_strings and "--checkpoint" not in tokens:
            tokens.extend(["--checkpoint", str(model_path)])
        elif any(dest in ("checkpoint", "model_path", "model") for dest in positional_dests):
            if not any(str(model_path) in t or ".pt" in t for t in tokens):
                tokens.append(str(model_path))
        elif "--resume" in option_strings and "--resume" not in tokens:
            tokens.extend(["--resume", str(model_path)])
        # Otherwise: do not inject --checkpoint

    console.print(f"[dim]Executing: zeropawn {' '.join(tokens)}[/dim]")

    try:
        sub_args = parser.parse_args(tokens)
        if hasattr(sub_args, "func"):
            return sub_args.func(sub_args)
        else:
            console.print("[bold red]Command has no execution function.[/bold red]")
            return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Execution error: {e}[/bold red]")
        return 1
    finally:
        # Crucial: Immediately reclaim GPU VRAM cache and host RAM after every operation
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def launch_preset_quick(preset_id: str) -> int:
    """Path A: Quick Run of a preset."""
    p = get_preset_by_id(preset_id)
    if not p:
        console.print(f"[bold red]Preset '{preset_id}' not found.[/bold red]")
        return 1

    parser = build_parser()
    tokens = [p["operation"]]
    subcmd = p.get("subcommand")
    args = list(p.get("args", []))
    if subcmd:
        tokens.append(subcmd)
        if args and args[0] == subcmd:
            args = args[1:]
    tokens.extend(args)

    console.print(
        Panel(
            f"[bold cyan]Quick Running Preset:[/] {p['name']}\n[dim]Command: zeropawn {' '.join(tokens)}[/dim]",
            box=box.ROUNDED,
            border_style="green",
        )
    )

    try:
        sub_args = parser.parse_args(tokens)
        if hasattr(sub_args, "func"):
            return sub_args.func(sub_args)
        else:
            return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation interrupted.[/bold yellow]")
        return 130
    finally:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def run_operation_selection_flow(op: OperationDef, active_model: Optional[Path]) -> None:
    """Flow after an operation is selected: choose Quick Run, Wizard, Custom Args, or Preset."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold bright_cyan]{op.name}[/bold bright_cyan]\n"
                f"[white]{op.description}[/white]\n\n"
                f"[dim]Direct Command:[/] [cyan]{op.direct_command}[/cyan]\n"
                f"[dim]Category:[/] {op.category_name} | [dim]Model Requirement:[/] {op.requires_model.capitalize()}",
                title="[bold green]Operation Selection[/bold green]",
                box=box.ROUNDED,
                border_style="green",
            )
        )

        model_display = active_model.name if active_model else "None (Auto-detect)"
        console.print(f"[bold cyan]Active Model Checkpoint:[/] [yellow]{model_display}[/yellow]\n")

        console.print(
            "[bold cyan]Select Execution Mode:[/bold cyan]\n"
            "  [bold yellow][1][/bold yellow] Quick Run with Defaults\n"
            "  [bold yellow][2][/bold yellow] Configuration Wizard (Configure all arguments - Path B)\n"
            "  [bold yellow][3][/bold yellow] Advanced Custom Arguments String (Path C)\n"
            "  [bold yellow][O][/bold yellow] Complete Option Explorer (All arguments, types, constraints)\n"
            "  [bold yellow][E][/bold yellow] Exhaustive Combination Explorer (Enumerate valid combinations)\n"
            "  [bold yellow][V][/bold yellow] Generated Command Preview & Dry-Run Validation\n"
            "  [bold yellow][M][/bold yellow] Change Selected Model Checkpoint\n"
            "  [bold red][B][/bold red] Back to Category Menu\n"
        )

        try:
            choice = (
                console.input("[bold yellow]Select mode [1-3, O, E, V, M, B] > [/bold yellow]")
                .strip()
                .upper()
            )
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("B", "BACK", "Q"):
            return

        elif choice == "M" and op.requires_model != "none":
            chosen = select_model_interactive(default_path=active_model)
            if chosen:
                active_model = chosen

        elif choice == "O":
            from python.cli.discovery import get_command_catalog
            from python.cli.interactive_discovery import run_option_explorer

            node = get_command_catalog().get_command(op.direct_command)
            if node:
                run_option_explorer(node)
            else:
                console.print(
                    f"[bold red]Catalog node not found for {op.direct_command}[/bold red]"
                )
                wait_for_enter("Press [Enter] to continue...")

        elif choice == "E":
            from python.cli.discovery import get_command_catalog
            from python.cli.interactive_discovery import run_combination_explorer

            node = get_command_catalog().get_command(op.direct_command)
            if node:
                run_combination_explorer(node)
            else:
                console.print(
                    f"[bold red]Catalog node not found for {op.direct_command}[/bold red]"
                )
                wait_for_enter("Press [Enter] to continue...")

        elif choice == "V":
            from python.cli.combinations import SafeParserValidator
            import shlex

            cmd_tokens = [
                tok
                for tok in op.direct_command.split()
                if tok not in ("zeropawn", "chess") and not tok.startswith("<")
            ]
            default_params = {a.name: a.default for a in op.arguments}
            for a in op.arguments:
                val = default_params.get(a.name)
                if a.required or getattr(a, "is_positional", False):
                    if (
                        getattr(a, "is_positional", False)
                        and val is not None
                        and str(val).strip() != ""
                    ):
                        cmd_tokens.append(str(val))
                    elif not getattr(a, "is_positional", False):
                        if a.type == "bool" and val:
                            cmd_tokens.append(a.flag)
                        elif val is not None:
                            cmd_tokens.extend([a.flag, str(val)])
            if active_model and op.requires_model != "none":
                cmd_tokens.extend(["--checkpoint", str(active_model)])

            preview_str = f"zeropawn {' '.join(shlex.quote(t) for t in cmd_tokens)}"
            ok, err = SafeParserValidator.validate_tokens(cmd_tokens)
            status_text = (
                "[bold green]VALID (Parser accepts command)[/bold green]"
                if ok
                else f"[bold red]INVALID: {err}[/bold red]"
            )
            console.print(
                Panel(
                    f"[bold cyan]Generated Command Preview:[/bold cyan]\n[white]{preview_str}[/white]\n\n"
                    f"[dim]Dry-Run Parser Validation:[/] {status_text}",
                    title="[bold green]Command Preview & Dry-Run[/bold green]",
                    box=box.ROUNDED,
                    border_style="green" if ok else "red",
                )
            )
            wait_for_enter("Press [Enter] to return to menu...")

        elif choice == "1":
            # Quick run
            default_params = {a.name: a.default for a in op.arguments}
            execute_operation_from_interactive(op, default_params, active_model)
            wait_for_enter("\nPress [Enter] to continue...")
            return

        elif choice == "2":
            # Wizard
            res = run_configuration_wizard(op, initial_model=active_model)
            if res is not None:
                params, chosen_model = res
                execute_operation_from_interactive(op, params, chosen_model)
                wait_for_enter("\nPress [Enter] to continue...")
            return

        elif choice == "3":
            # Custom arguments
            default_params = {a.name: a.default for a in op.arguments}
            updated = enter_custom_arguments_mode(op, default_params)
            if updated is not None:
                execute_operation_from_interactive(op, updated, active_model)
                wait_for_enter("\nPress [Enter] to continue...")
            return


def run_category_menu(category_id: int, active_model: Optional[Path]) -> None:
    """Display all operations within a selected category."""
    cat_name = CATEGORIES.get(category_id, f"Category {category_id}")
    ops = get_operations_by_category(category_id)

    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold bright_cyan]{cat_name}[/bold bright_cyan]\n"
                f"[dim]Select an operation number to configure and run.[/dim]",
                title=f"[bold green]Category {category_id} / 10[/bold green]",
                box=box.ROUNDED,
                border_style=COLOR_BORDER,
            )
        )

        table = Table(box=box.ROUNDED, border_style=COLOR_BORDER, header_style="bold cyan")
        table.add_column("#", justify="right", style="bold yellow")
        table.add_column("Operation", style="bright_white")
        table.add_column("Direct CLI Command", style="cyan")
        table.add_column("Description", style="dim")

        op_map: Dict[int, OperationDef] = {}
        for idx, op in enumerate(ops, 1):
            table.add_row(
                str(idx),
                op.name,
                op.direct_command,
                op.description[:55],
            )
            op_map[idx] = op

        console.print(table)
        console.print("[dim]Enter [B] to return to Main Menu.[/dim]")

        try:
            choice = (
                console.input(f"[bold yellow]Select operation [1-{len(ops)}, B] > [/bold yellow]")
                .strip()
                .upper()
            )
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("B", "BACK", "Q"):
            return

        if choice.isdigit():
            idx_int = int(choice)
            if idx_int in op_map:
                run_operation_selection_flow(op_map[idx_int], active_model)
            else:
                console.print(f"[bold red]Selection {idx_int} out of range.[/bold red]")
                wait_for_enter("Press [Enter] to continue...")


def run_interactive_menu() -> int:
    """Rich-powered Interactive Control Center for ZeroPawn."""
    active_model: Optional[Path] = get_default_model_path()

    while True:
        try:
            console.clear()
            print_app_banner()
            console.print(render_hardware_panel())

            model_name = active_model.name if active_model else "None (Auto-detected)"
            console.print(f"[dim]Active Model: [bold cyan]{model_name}[/bold cyan][/dim]\n")

            menu_table = Table(
                title="[bold bright_cyan]Main Operation Categories[/bold bright_cyan]",
                box=box.ROUNDED,
                border_style=COLOR_BORDER,
                header_style="bold cyan",
                padding=(0, 2),
            )
            menu_table.add_column("#", justify="right", style="bold yellow")
            menu_table.add_column("Category", style="bright_white")
            menu_table.add_column("Scope / Capabilities", style="dim")

            cat_descriptions = {
                1: "Phase 4 Self-play, Supervised Warm-start, Fine-tuning, Connect 4",
                2: "Lichess PGN download, Filtering, Memmaps, Tactics, Opening Books",
                3: "List checkpoints, inspect layers/parameters, compare model checkpoints",
                4: "Stockfish matches, Live moves viewer, Human vs AI, Elo baselines",
                5: "Throughput benchmarks, batch sizes, pipelined dispatcher, sweeps",
                6: "Tactical puzzle solver, Model lag, NN cache curve, Syzygy tablebases",
                7: "Pytest suites, Guardrails, Differential testing, Soak tests, Parity",
                8: "Run built-in quick presets (Path A), manage persistent user presets",
                9: "System health, Resignation diagnostics, Value signs, Profiling",
                10: "Cleanup temporary buffers, UCI protocol engine mode, Janitor",
            }

            for c_id in range(1, 11):
                menu_table.add_row(
                    str(c_id),
                    CATEGORIES[c_id],
                    cat_descriptions.get(c_id, ""),
                )

            console.print(menu_table)

            nav_text = (
                "[bold cyan]Exploration Gates:[/bold cyan]  "
                "[bold yellow][C][/bold yellow] Complete Command Browser (114+ Commands)  |  "
                "[bold yellow][B][/bold yellow] Argument Builder  |  "
                "[bold yellow][E][/bold yellow] Combination Explorer\n"
                "[bold cyan]Quick Actions:[/bold cyan]      "
                "[bold yellow][P][/bold yellow] Quick Presets (Path A)  |  "
                "[bold yellow][M][/bold yellow] Model Manager  |  "
                "[bold yellow][S][/bold yellow] System Status  |  "
                "[bold red][Q][/bold red] Exit"
            )
            console.print(Panel(nav_text, box=box.ROUNDED, border_style="dim", padding=(0, 1)))

            choice = (
                console.input(
                    "[bold yellow]Select category [1-10] or Action [C, B, E, P, M, S, Q] > [/bold yellow]"
                )
                .strip()
                .upper()
            )

            if choice in ("Q", "QUIT", "EXIT"):
                console.print("[bold cyan]Goodbye![/bold cyan]")
                return 0

            elif choice == "C":
                # Complete Command Browser (114+ commands)
                from python.cli.interactive_discovery import run_command_browser

                run_command_browser()

            elif choice == "B":
                # Interactive Argument Builder
                from python.cli.discovery import get_command_catalog
                from python.cli.interactive_discovery import run_argument_builder

                catalog = get_command_catalog()
                cmd_name = console.input(
                    "[bold cyan]Enter command name to configure (e.g. 'train' or 'data filter') > [/bold cyan]"
                ).strip()
                if cmd_name:
                    node = catalog.get_command(cmd_name)
                    if node:
                        run_argument_builder(node)
                    else:
                        console.print(f"[bold red]Command '{cmd_name}' not found.[/bold red]")
                        wait_for_enter("Press [Enter] to continue...")

            elif choice == "E":
                # Combination Explorer
                from python.cli.discovery import get_command_catalog
                from python.cli.interactive_discovery import run_combination_explorer

                catalog = get_command_catalog()
                cmd_name = console.input(
                    "[bold cyan]Enter command name to explore combinations (e.g. 'train' or 'match run') > [/bold cyan]"
                ).strip()
                if cmd_name:
                    node = catalog.get_command(cmd_name)
                    if node:
                        run_combination_explorer(node)
                    else:
                        console.print(f"[bold red]Command '{cmd_name}' not found.[/bold red]")
                        wait_for_enter("Press [Enter] to continue...")

            elif choice == "P":
                # Quick Presets Launcher (Path A)
                console.clear()
                items = render_presets_table()
                p_choice = console.input(
                    "[bold yellow]Enter Preset # to launch immediately (or Enter to cancel) > [/bold yellow]"
                ).strip()
                if p_choice.isdigit():
                    p_idx = int(p_choice)
                    matching = [p_info for idx, pid, p_info in items if idx == p_idx]
                    if matching:
                        launch_preset_quick(matching[0]["id"])
                        wait_for_enter("\nPress [Enter] to return to menu...")

            elif choice == "M":
                # Model manager
                chosen = select_model_interactive(default_path=active_model)
                if chosen:
                    active_model = chosen

            elif choice == "S":
                # System status
                parser = build_parser()
                sub_args = parser.parse_args(["status"])
                sub_args.func(sub_args)
                wait_for_enter("\nPress [Enter] to return to menu...")

            elif choice.isdigit():
                cat_int = int(choice)
                if 1 <= cat_int <= 10:
                    run_category_menu(cat_int, active_model)
                else:
                    console.print(f"[bold red]Selection {cat_int} out of range [1-10].[/bold red]")
                    wait_for_enter("Press [Enter] to continue...")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            return 0
        except Exception as e:
            console.print(f"\n[bold red]Menu error: {e}[/bold red]")
            try:
                wait_for_enter("Press [Enter] to continue...")
            except (KeyboardInterrupt, EOFError):
                return 0

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    setup_environment()
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    # If launched with no arguments, launch interactive control center in Rich mode
    if not argv:
        set_output_mode(OutputMode.RICH)
        return run_interactive_menu()

    # If running UCI mode, enforce strict protocol isolation
    if argv and argv[0] == "uci":
        set_output_mode(OutputMode.PROTOCOL)
    else:
        set_output_mode(OutputMode.RICH)

    try:
        args = parser.parse_args(argv)

        # Always resolve configuration to honor strict 5-tier precedence:
        # Defaults < Config file < Presets < Environment variables < Explicit CLI overrides
        from python.cli.config import resolve_configuration

        preset_dict = None
        preset_id = getattr(args, "preset", None)
        if preset_id:
            from python.cli.presets import get_preset_by_id

            preset_dict = get_preset_by_id(preset_id)

        resolved = resolve_configuration(
            command_name=getattr(args, "command", "") or (argv[0] if argv else ""),
            cli_args=args,
            preset_dict=preset_dict,
            config_path=getattr(args, "config", None),
            raw_argv=argv,
        )
        setattr(args, "_resolved_config", resolved)

        # Propagate resolved non-default values back to args attributes if not explicitly passed
        for canon_key, rec in resolved.records.items():
            if rec.source in ("config_file", "preset", "env"):
                attr_names = [rec.field_spec.alias, rec.field_spec.key]
                for f in rec.field_spec.cli_flags:
                    attr_names.append(f.lstrip("-").replace("-", "_"))
                for attr in attr_names:
                    if hasattr(args, attr):
                        setattr(args, attr, rec.effective_value)

        if getattr(args, "print_resolved", False):
            table = resolved.summary_table()
            console.print(table)

        if hasattr(args, "func"):
            return args.func(args)
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print_warning("\nOperation interrupted by user.")
        return 130
    except Exception as e:
        print_error(f"Command execution error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
