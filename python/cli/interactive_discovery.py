"""Interactive Command Browser, Option Explorer, Argument Builder, and Combination Explorer for ZeroPawn."""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional, Set

from rich import box
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from python.cli.combinations import (
    ArgumentCombination,
    SafeParserValidator,
    evaluate_combination_constraints,
    get_combination_engine,
)
from python.cli.common import wait_for_enter
from python.cli.discovery import (
    ArgumentMetadata,
    CommandNode,
    get_command_catalog,
)
from python.cli.operations import CATEGORIES
from python.cli.ui import COLOR_BORDER, console


def run_command_browser(
    initial_category: Optional[int] = None, non_interactive: bool = False
) -> None:
    """Browse the complete hierarchy of all 114+ ZeroPawn CLI commands."""
    catalog = get_command_catalog()
    current_category: Optional[int] = initial_category
    search_query: Optional[str] = None

    while True:
        console.clear()
        title_extra = (
            f" (Category {current_category}: {CATEGORIES.get(current_category, '')})"
            if current_category
            else " (All Commands)"
        )
        if search_query:
            title_extra += f" [Filter: '{search_query}']"

        console.print(
            Panel(
                f"[bold bright_cyan]ZeroPawn Complete Command Browser[/bold bright_cyan]{title_extra}\n"
                f"[dim]Navigate all executable commands, inspect argument inventories, build arguments, and explore combinations.[/dim]",
                title="[bold green]Command Browser (Completeness Gate)[/bold green]",
                box=box.ROUNDED,
                border_style=COLOR_BORDER,
            )
        )

        # Get matching commands
        if search_query:
            nodes = catalog.search_commands(search_query)
        elif current_category:
            nodes = catalog.filter_by_category(current_category)
        else:
            nodes = catalog.all_leaf_commands()

        # Sort alphabetically by full_command
        nodes = sorted(nodes, key=lambda n: n.full_command)

        table = Table(
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("#", justify="right", style="bold yellow")
        table.add_column("Command", style="bright_white")
        table.add_column("Category", style="cyan")
        table.add_column("Class", style="dim")
        table.add_column("Args", justify="right", style="yellow")
        table.add_column("Aliases", style="dim")
        table.add_column("Description", style="dim")

        node_map: Dict[int, CommandNode] = {}
        for idx, node in enumerate(nodes, 1):
            class_badge = node.classification
            if class_badge == "deprecated":
                class_style = "[red]deprecated[/red]"
            elif class_badge in ("advanced", "experimental"):
                class_style = f"[yellow]{class_badge}[/yellow]"
            else:
                class_style = f"[dim]{class_badge}[/dim]"

            alias_str = ", ".join(node.aliases) if node.aliases else "-"
            table.add_row(
                str(idx),
                node.full_command,
                f"Cat {node.category_id}",
                class_style,
                str(len(node.arguments)),
                alias_str,
                node.description[:45] if node.description else "-",
            )
            node_map[idx] = node

        console.print(table)

        if non_interactive:
            return

        console.print(
            f"[dim]Showing {len(nodes)} command(s).[/dim]\n\n"
            "[bold cyan]Actions:[/bold cyan]\n"
            "  [bold yellow][1-N][/bold yellow] Select command to explore (Options, Builder, Combinations)\n"
            "  [bold yellow]\\[/][/bold yellow]   Search by keyword (e.g. '/tablebase' or '/puzzle')\n"
            "  [bold yellow][C][/bold yellow]   Filter by Category (1-10) or [0] for All Categories\n"
            "  [bold yellow][R][/bold yellow]   Reset search / category filters\n"
            "  [bold red][B][/bold red]   Back to Main Menu\n"
        )

        try:
            choice = console.input(
                "[bold yellow]Enter command # or action > [/bold yellow]"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return

        if choice.upper() in ("B", "BACK", "Q", "QUIT"):
            return

        elif choice.startswith("/") or choice.upper() == "S":
            q = (
                choice[1:].strip()
                if choice.startswith("/")
                else console.input("[bold cyan]Enter search query: [/bold cyan]").strip()
            )
            search_query = q if q else None

        elif choice.upper() == "C":
            cat_choice = console.input(
                "[bold cyan]Enter Category # [1-10] or 0 for All: [/bold cyan]"
            ).strip()
            if cat_choice.isdigit():
                c_num = int(cat_choice)
                current_category = c_num if 1 <= c_num <= 10 else None

        elif choice.upper() == "R":
            search_query = None
            current_category = None

        elif choice.isdigit():
            c_idx = int(choice)
            if c_idx in node_map:
                run_command_action_menu(node_map[c_idx])
            else:
                console.print(f"[bold red]Command index {c_idx} out of range.[/bold red]")
                wait_for_enter("Press [Enter] to continue...")


def run_command_action_menu(node: CommandNode) -> None:
    """Action menu for a selected command: Option Explorer, Argument Builder, Combination Explorer, Quick Run."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold bright_cyan]{node.full_command}[/bold bright_cyan]\n"
                f"[white]{node.description or 'No description provided.'}[/white]\n\n"
                f"[dim]Category:[/] {node.category_name} (Cat {node.category_id}) | "
                f"[dim]Classification:[/] {node.classification.capitalize()} | "
                f"[dim]Model Requirement:[/] {node.requires_model.capitalize()}",
                title=f"[bold green]Command: {node.name}[/bold green]",
                box=box.ROUNDED,
                border_style="green",
            )
        )

        console.print(
            "[bold cyan]Select Exploration Capability:[/bold cyan]\n"
            "  [bold yellow][1][/bold yellow] Complete Option Explorer (Inspect all arguments, types, choices, constraints)\n"
            "  [bold yellow][2][/bold yellow] Interactive Argument Builder (Configure arguments, real-time validation, preview)\n"
            "  [bold yellow][3][/bold yellow] Exhaustive Combination Explorer (Enumerate valid finite option combinations)\n"
            "  [bold yellow][4][/bold yellow] Quick Run with Defaults (Execute directly)\n"
            "  [bold red][B][/bold red] Back to Command Browser\n"
        )

        try:
            choice = (
                console.input("[bold yellow]Select action [1-4, B] > [/bold yellow]")
                .strip()
                .upper()
            )
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("B", "BACK", "Q"):
            return
        elif choice == "1":
            run_option_explorer(node)
        elif choice == "2":
            run_argument_builder(node)
        elif choice == "3":
            run_combination_explorer(node)
        elif choice == "4":
            # Quick run
            default_params = {a.name: a.default for a in node.arguments if not a.is_inherited}
            _execute_command_node(node, default_params)
            wait_for_enter("\nPress [Enter] to continue...")


def run_option_explorer(node: CommandNode) -> None:
    """Complete option explorer for a command."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold bright_cyan]{node.full_command}[/bold bright_cyan] — Option Inventory\n"
                f"[dim]Total Arguments: {len(node.arguments)} ({len(node.positional_arguments)} positional, "
                f"{len(node.optional_arguments)} options, {len(node.inherited_arguments)} inherited global)[/dim]",
                title="[bold green]Complete Option Explorer[/bold green]",
                box=box.ROUNDED,
                border_style=COLOR_BORDER,
            )
        )

        table = Table(
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Type / Kind", style="dim")
        table.add_column("Option / Flag", style="bold yellow")
        table.add_column("Value Type", style="cyan")
        table.add_column("Default", style="dim")
        table.add_column("Choices / Constraints", style="dim")
        table.add_column("Status", style="bright_white")
        table.add_column("Description", style="dim")

        # 1. Positional Arguments
        for arg in node.positional_arguments:
            status = "[bold red]Required[/bold red]" if arg.required else "[dim]Optional[/dim]"
            choices_str = (
                f"Choices: [{', '.join(arg.choices)}]" if arg.choices else "Arbitrary Value"
            )
            table.add_row(
                "Positional",
                f"<{arg.name}>",
                arg.type,
                str(arg.default) if arg.default is not None else "-",
                choices_str,
                status,
                arg.help,
            )

        # 2. Command Flags & Options
        for arg in node.optional_arguments:
            status = "[bold red]Required[/bold red]" if arg.required else "[dim]Optional[/dim]"
            flag_disp = ", ".join(arg.flags) if arg.flags else f"--{arg.name}"
            if arg.choices:
                constraint_str = f"Choices: [{', '.join(arg.choices)}]"
            elif arg.is_flag_only:
                constraint_str = "Boolean Flag (no value)"
            else:
                constraint_str = "Value Option"
            table.add_row(
                "Flag" if arg.is_flag_only else "Option",
                flag_disp,
                arg.type,
                str(arg.default) if arg.default is not None else "-",
                constraint_str,
                status,
                arg.help,
            )

        # 3. Inherited Global Options
        for arg in node.inherited_arguments:
            flag_disp = ", ".join(arg.flags)
            table.add_row(
                "Global",
                flag_disp,
                arg.type,
                str(arg.default) if arg.default is not None else "-",
                "Inherited from Root",
                "[dim]Global[/dim]",
                arg.help,
            )

        console.print(table)

        # Print semantic constraints if any
        if node.constraints:
            console.print(
                "\n[bold cyan]Registered Semantic Constraints & Dependencies:[/bold cyan]"
            )
            for rule in node.constraints:
                console.print(
                    f"  • [bold yellow]{rule.rule_type}:[/bold yellow] {rule.description}"
                )

        console.print("\n[dim]Enter [B] to return to Command Menu.[/dim]")
        try:
            choice = console.input("[bold yellow]Action [B] > [/bold yellow]").strip().upper()
        except (KeyboardInterrupt, EOFError):
            return
        if choice in ("B", "BACK", "Q", ""):
            return


def run_argument_builder(
    node: CommandNode,
    initial_params: Optional[Dict[str, Any]] = None,
    initial_explicit: Optional[Set[str]] = None,
) -> None:
    """Interactive Argument Builder for constructing and validating command invocations."""
    params: Dict[str, Any] = {}
    explicit: Set[str] = set()

    for arg in node.arguments:
        if not arg.is_inherited:
            params[arg.name] = arg.default

    if initial_params:
        params.update(initial_params)
    if initial_explicit:
        explicit.update(initial_explicit)

    # Initialize required positionals
    for pos in node.positional_arguments:
        if pos.required and (pos.name not in params or params[pos.name] is None):
            params[pos.name] = "required_value"
            explicit.add(pos.name)

    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold bright_cyan]{node.full_command}[/bold bright_cyan] — Interactive Argument Builder\n"
                f"[dim]Configure arguments, review real-time constraint validation, preview command line, and execute safely.[/dim]",
                title="[bold green]Argument Builder (Path B+)[/bold green]",
                box=box.ROUNDED,
                border_style="green",
            )
        )

        table = Table(
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("#", justify="right", style="bold yellow")
        table.add_column("Argument", style="bright_white")
        table.add_column("Flag / Position", style="cyan")
        table.add_column("Current Setting", style="bold yellow")
        table.add_column("Default", style="dim")
        table.add_column("Choices / Type", style="dim")
        table.add_column("Status", style="dim")

        row_map: Dict[int, ArgumentMetadata] = {}
        idx = 1

        command_args = [a for a in node.arguments if not a.is_inherited]
        for arg in command_args:
            val = params.get(arg.name, arg.default)
            is_active = arg.name in explicit
            flag_disp = (
                f"<{arg.name}>"
                if arg.is_positional
                else (", ".join(arg.flags) if arg.flags else f"--{arg.name}")
            )

            if is_active:
                setting_disp = f"[bold green]{val}[/bold green]"
                status_disp = "[green]Explicit[/green]"
            else:
                setting_disp = f"[dim]{val}[/dim]"
                status_disp = "[dim]Default[/dim]"

            if arg.choices:
                type_disp = f"[{', '.join(arg.choices)}]"
            else:
                type_disp = arg.type

            table.add_row(
                str(idx),
                arg.name,
                flag_disp,
                setting_disp,
                str(arg.default) if arg.default is not None else "-",
                type_disp,
                status_disp,
            )
            row_map[idx] = arg
            idx += 1

        console.print(table)

        # Real-time constraint evaluation
        is_valid, reason = evaluate_combination_constraints(node, params, explicit)
        if not is_valid:
            console.print(
                Panel(
                    f"[bold red]Constraint Warning:[/bold red] {reason}",
                    box=box.ROUNDED,
                    border_style="red",
                )
            )
        else:
            console.print("[dim green]✓ All registered constraints satisfied.[/dim green]")

        # Command Preview Box
        rendered_cmd = _render_command_line(node, params, explicit)
        console.print(
            Panel(
                f"[bold cyan]Generated Command Line Preview:[/bold cyan]\n"
                f"[bold white]{rendered_cmd}[/bold white]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )

        console.print(
            "[bold cyan]Actions:[/bold cyan]\n"
            "  [bold yellow][1-N][/bold yellow] Edit or toggle argument\n"
            "  [bold yellow][R][/bold yellow]   Reset all arguments to defaults\n"
            "  [bold yellow][D][/bold yellow]   Dry-Run Validate (Tests parsing without executing heavy work)\n"
            "  [bold green][X][/bold green]   Execute Command Now\n"
            "  [bold red][B][/bold red]   Back to Command Menu\n"
        )

        try:
            choice = console.input("[bold yellow]Enter selection > [/bold yellow]").strip().upper()
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("B", "BACK", "Q"):
            return

        elif choice == "R":
            explicit.clear()
            for arg in node.arguments:
                if not arg.is_inherited:
                    params[arg.name] = arg.default
            console.print("[bold green]Reset all fields to defaults.[/bold green]")

        elif choice == "D":
            # Dry run validation
            tokens = _build_tokens(node, params, explicit)
            ok, err = SafeParserValidator.validate_tokens(tokens)
            if ok:
                console.print(
                    Panel(
                        f"[bold green]Dry-Run PASSED![/bold green] Argument parser successfully validated this configuration:\n"
                        f"[cyan]zeropawn {' '.join(tokens)}[/cyan]",
                        box=box.ROUNDED,
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[bold red]Dry-Run FAILED:[/bold red] {err}",
                        box=box.ROUNDED,
                        border_style="red",
                    )
                )
            wait_for_enter("Press [Enter] to continue...")

        elif choice in ("X", "RUN", "EXECUTE"):
            if not is_valid:
                console.print(f"[bold red]Cannot execute: {reason}[/bold red]")
                wait_for_enter("Press [Enter] to continue...")
                continue
            _execute_command_node(node, params, explicit)
            wait_for_enter("\nPress [Enter] to continue...")

        elif choice.isdigit():
            val_idx = int(choice)
            if val_idx in row_map:
                target_arg = row_map[val_idx]
                curr_val = params.get(target_arg.name, target_arg.default)

                if target_arg.is_flag_only:
                    # Toggle boolean flag
                    new_val = not curr_val
                    params[target_arg.name] = new_val
                    if new_val != target_arg.default:
                        explicit.add(target_arg.name)
                    else:
                        explicit.discard(target_arg.name)

                elif target_arg.choices:
                    console.print(f"Valid choices: {', '.join(target_arg.choices)}")
                    new_c = Prompt.ask(
                        f"Enter choice for [cyan]{target_arg.name}[/cyan]",
                        choices=target_arg.choices,
                        default=str(curr_val) if curr_val is not None else target_arg.choices[0],
                    )
                    params[target_arg.name] = new_c
                    explicit.add(target_arg.name)

                elif target_arg.type == "int":
                    new_int = IntPrompt.ask(
                        f"Enter new value for [cyan]{target_arg.name}[/cyan]",
                        default=int(curr_val) if curr_val is not None else 0,
                    )
                    params[target_arg.name] = new_int
                    explicit.add(target_arg.name)

                elif target_arg.type == "float":
                    new_float = FloatPrompt.ask(
                        f"Enter new value for [cyan]{target_arg.name}[/cyan]",
                        default=float(curr_val) if curr_val is not None else 0.0,
                    )
                    params[target_arg.name] = new_float
                    explicit.add(target_arg.name)

                else:
                    new_s = Prompt.ask(
                        f"Enter new value for [cyan]{target_arg.name}[/cyan]",
                        default=str(curr_val) if curr_val is not None else "",
                    )
                    params[target_arg.name] = new_s
                    explicit.add(target_arg.name)


def run_combination_explorer(node: CommandNode) -> None:
    """Exhaustive, paginated combination explorer for a command."""
    engine = get_combination_engine()
    page = 1
    page_size = 15
    filter_query: Optional[str] = None
    include_invalid = True

    while True:
        combos, total_items, total_pages = engine.get_paginated_combinations(
            node,
            page=page,
            page_size=page_size,
            filter_query=filter_query,
            include_invalid=include_invalid,
        )

        console.clear()
        filter_str = f" [Filter: '{filter_query}']" if filter_query else ""
        console.print(
            Panel(
                f"[bold bright_cyan]{node.full_command}[/bold bright_cyan] — Combination Explorer\n"
                f"[dim]Total Combinations in Space: {total_items} | Showing Page {page} of {total_pages}{filter_str}[/dim]",
                title="[bold green]Complete Combination Explorer[/bold green]",
                box=box.ROUNDED,
                border_style="green",
            )
        )

        table = Table(
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("#", justify="right", style="bold yellow")
        table.add_column("Mode", style="cyan")
        table.add_column("Configuration / Active Flags", style="bright_white")
        table.add_column("Status", style="bright_white")
        table.add_column("Constraint Rule / Notes", style="dim")

        combo_map: Dict[int, ArgumentCombination] = {}
        for c in combos:
            if c.is_valid:
                status_badge = "[bold green]VALID[/bold green]"
                notes = "[dim green]Satisfies constraints[/dim green]"
            else:
                status_badge = "[bold red]INVALID[/bold red]"
                notes = f"[red]{c.rejection_reason}[/red]"

            table.add_row(
                str(c.index),
                c.mode or "default",
                c.summary_description(),
                status_badge,
                notes,
            )
            combo_map[c.index] = c

        console.print(table)

        console.print(
            f"[dim]Displaying combinations {combos[0].index if combos else 0} - {combos[-1].index if combos else 0} of {total_items}.[/dim]\n\n"
            "[bold cyan]Navigation & Actions:[/bold cyan]\n"
            "  [bold yellow][N][/bold yellow] Next page  |  [bold yellow][P][/bold yellow] Previous page  |  [bold yellow][G][/bold yellow] Go to page\n"
            "  [bold yellow][F][/bold yellow] Filter by flag/keyword  |  [bold yellow][R][/bold yellow] Reset filter\n"
            "  [bold yellow][1-N][/bold yellow] Select combination index to inspect / load into Builder\n"
            "  [bold red][B][/bold red]   Back to Command Menu\n"
        )

        try:
            choice = (
                console.input("[bold yellow]Enter command or index > [/bold yellow]")
                .strip()
                .upper()
            )
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("B", "BACK", "Q"):
            return
        elif choice == "N" and page < total_pages:
            page += 1
        elif choice == "P" and page > 1:
            page -= 1
        elif choice == "G":
            target = console.input(
                f"[bold cyan]Enter page number [1-{total_pages}]: [/bold cyan]"
            ).strip()
            if target.isdigit():
                page = max(1, min(int(target), total_pages))
        elif choice == "F":
            q = console.input(
                "[bold cyan]Enter flag or keyword to filter combinations: [/bold cyan]"
            ).strip()
            filter_query = q if q else None
            page = 1
        elif choice == "R":
            filter_query = None
            page = 1
        elif choice.isdigit():
            c_idx = int(choice)
            if c_idx in combo_map:
                selected_combo = combo_map[c_idx]
                _inspect_or_load_combination(node, selected_combo)


def _inspect_or_load_combination(node: CommandNode, combo: ArgumentCombination) -> None:
    """Inspect a combination in detail and offer to load into Builder or Dry-Run."""
    console.clear()
    console.print(
        Panel(
            f"[bold bright_cyan]Combination #{combo.index}[/bold bright_cyan]\n"
            f"[dim]Command:[/] {node.full_command}\n"
            f"[dim]Mode:[/] {combo.mode}\n"
            f"[dim]Validity:[/] {'[bold green]VALID[/bold green]' if combo.is_valid else '[bold red]INVALID[/bold red]'}\n"
            + (
                f"[red]Rejection Reason: {combo.rejection_reason}[/red]\n"
                if not combo.is_valid
                else ""
            )
            + f"\n[bold cyan]Command Line Invocation:[/bold cyan]\n[white]{combo.to_command_line()}[/white]",
            title="[bold green]Combination Detail[/bold green]",
            box=box.ROUNDED,
            border_style="green",
        )
    )

    console.print(
        "[bold cyan]Actions:[/bold cyan]\n"
        "  [bold yellow][L][/bold yellow] Load this combination into Argument Builder\n"
        "  [bold yellow][D][/bold yellow] Dry-Run Validate against Argparse Parser\n"
        "  [bold red][B][/bold red] Back to Combination List\n"
    )

    try:
        choice = console.input("[bold yellow]Action [L, D, B] > [/bold yellow]").strip().upper()
    except (KeyboardInterrupt, EOFError):
        return

    if choice == "L":
        run_argument_builder(
            node,
            initial_params=combo.parameters,
            initial_explicit=combo.explicit_flags,
        )
    elif choice == "D":
        ok, err = SafeParserValidator.validate_tokens(combo.to_argv())
        if ok:
            console.print(
                f"[bold green]Parser validation PASSED for combination #{combo.index}![/bold green]"
            )
        else:
            console.print(f"[bold red]Parser validation FAILED:[/bold red] {err}")
        wait_for_enter("Press [Enter] to continue...")


def _build_tokens(node: CommandNode, params: Dict[str, Any], explicit: Set[str]) -> List[str]:
    """Build tokens list matching node arguments."""
    tokens = list(node.path)
    for arg in node.arguments:
        if arg.is_inherited:
            continue
        if arg.is_positional:
            val = params.get(arg.name)
            if val is not None and str(val).strip() != "":
                tokens.append(str(val))
        elif arg.name in explicit:
            val = params.get(arg.name)
            flag_str = arg.flags[0] if arg.flags else f"--{arg.name.replace('_', '-')}"
            if arg.is_flag_only:
                if val:
                    tokens.append(flag_str)
            else:
                if val is not None:
                    tokens.extend([flag_str, str(val)])
    return tokens


def _render_command_line(node: CommandNode, params: Dict[str, Any], explicit: Set[str]) -> str:
    """Render safely quoted command line string."""
    tokens = ["zeropawn"] + _build_tokens(node, params, explicit)
    return " ".join(shlex.quote(tok) for tok in tokens)


def _execute_command_node(
    node: CommandNode,
    params: Dict[str, Any],
    explicit: Optional[Set[str]] = None,
) -> int:
    """Execute a CommandNode using the real build_parser dispatcher."""
    from python.cli.main import build_parser

    explicit_set = explicit or set()
    tokens = _build_tokens(node, params, explicit_set)

    console.print(f"[dim]Executing: zeropawn {' '.join(tokens)}[/dim]")
    parser = build_parser()

    try:
        args = parser.parse_args(tokens)
        if hasattr(args, "func"):
            return args.func(args)
        else:
            console.print("[bold red]Command has no execution function.[/bold red]")
            return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation interrupted by user.[/bold yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Execution error: {e}[/bold red]")
        return 1
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
