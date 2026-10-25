"""Interactive Configuration Wizard, Advanced Custom Argument Parser, and Quick Run for ZeroPawn."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich import box
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from python.cli.models import get_default_model_path, select_model_interactive
from python.cli.operations import OpArgument, OperationDef
from python.cli.presets import save_preset
from python.cli.ui import COLOR_BORDER, console
from python.cli.common import wait_for_enter


_WIZARD_SESSION_STATE: Dict[str, Dict[str, Any]] = {}
_WIZARD_SELECTED_MODELS: Dict[str, Optional[Path]] = {}


def run_configuration_wizard(
    op: OperationDef,
    initial_params: Optional[Dict[str, Any]] = None,
    initial_model: Optional[Path] = None,
) -> Optional[Tuple[Dict[str, Any], Optional[Path]]]:
    """Path B: Interactive Configuration Wizard.

    Presents all supported, meaningful options for the operation.
    Supports editing individual fields, resetting, model selection,
    reviewing before execution, and saving as preset.
    Retains session state across Back [B] navigation.
    """
    effective_params: Dict[str, Any] = {}
    if initial_params:
        effective_params = dict(initial_params)
    elif op.id in _WIZARD_SESSION_STATE:
        effective_params = dict(_WIZARD_SESSION_STATE[op.id])
    else:
        for arg in op.arguments:
            effective_params[arg.name] = arg.default

    selected_model: Optional[Path] = initial_model
    if selected_model is None and op.id in _WIZARD_SELECTED_MODELS:
        selected_model = _WIZARD_SELECTED_MODELS[op.id]
    if op.requires_model in ("required", "optional") and selected_model is None:
        selected_model = get_default_model_path()

    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold bright_cyan]{op.name}[/bold bright_cyan]\n[italic white]{op.description}[/italic white]",
                title="[bold green]Configuration Wizard (Path B)[/bold green]",
                box=box.ROUNDED,
                border_style="green",
            )
        )

        table = Table(
            title="Configurable Operation Arguments",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("#", justify="right", style="bold yellow")
        table.add_column("Argument", style="bright_white")
        table.add_column("Flag", style="cyan")
        table.add_column("Current Value", style="bold yellow")
        table.add_column("Default", style="dim")
        table.add_column("Type / Range", style="dim")
        table.add_column("Description", style="dim")

        row_map: Dict[int, OpArgument] = {}
        idx = 1
        for arg in op.arguments:
            curr_val = effective_params.get(arg.name, arg.default)
            flag_disp = f"<{arg.name}>" if getattr(arg, "is_positional", False) else arg.flag
            if arg.min_val is not None and arg.max_val is not None:
                range_str = f"[{arg.min_val}..{arg.max_val}]"
            elif arg.min_val is not None:
                range_str = f">={arg.min_val}"
            elif arg.max_val is not None:
                range_str = f"<={arg.max_val}"
            elif arg.choices:
                range_str = f"[{', '.join(str(c) for c in arg.choices)}]"
            else:
                range_str = arg.type
            table.add_row(
                str(idx),
                arg.name,
                flag_disp,
                str(curr_val),
                str(arg.default),
                range_str,
                arg.help,
            )
            row_map[idx] = arg
            idx += 1

        if op.requires_model != "none":
            model_name = (
                selected_model.name if selected_model else "[yellow]None (Select model)[/yellow]"
            )
            model_req = (
                "[bold red]Required[/bold red]"
                if op.requires_model == "required"
                else "[dim]Optional[/dim]"
            )
            table.add_row(
                "M",
                "Model Checkpoint",
                "--checkpoint",
                model_name,
                "best_model.pt",
                model_req,
                "Neural network weights checkpoint",
            )

        console.print(table)

        console.print(
            "\n[bold cyan]Actions:[/bold cyan]\n"
            "  [bold yellow][1-N][/bold yellow] Edit specific argument\n"
            f"  [bold yellow][M][/bold yellow]   Select/Change Model Checkpoint ({'Active' if op.requires_model != 'none' else 'Disabled'})\n"
            "  [bold yellow][E][/bold yellow]   Combination Explorer (Enumerate all valid combinations)\n"
            "  [bold yellow][V][/bold yellow]   Preview Generated Command & Dry-Run\n"
            "  [bold yellow][A][/bold yellow]   Advanced Custom Arguments Mode (Path C)\n"
            "  [bold yellow][S][/bold yellow]   Save current configuration as Preset\n"
            "  [bold yellow][R][/bold yellow]   Reset all arguments to defaults\n"
            "  [bold green][X][/bold green]   Execute Operation Now\n"
            "  [bold cyan][B][/bold cyan]   Back (retains session state)\n"
            "  [bold yellow][C][/bold yellow]   Cancel (abandons changes)\n"
            "  [bold red][Q][/bold red]   Quit / Exit\n"
        )

        try:
            choice = console.input("[bold yellow]Enter selection > [/bold yellow]").strip().upper()
        except (KeyboardInterrupt, EOFError):
            return None

        if choice in ("B", "BACK"):
            _WIZARD_SESSION_STATE[op.id] = dict(effective_params)
            _WIZARD_SELECTED_MODELS[op.id] = selected_model
            return None

        elif choice in ("C", "CANCEL"):
            _WIZARD_SESSION_STATE.pop(op.id, None)
            _WIZARD_SELECTED_MODELS.pop(op.id, None)
            console.print("[dim]Changes abandoned.[/dim]")
            return None

        elif choice in ("Q", "QUIT", "EXIT"):
            return None

        elif choice in ("X", "EXECUTE", "RUN", ""):
            # Review and execute
            if op.requires_model == "required" and selected_model is None:
                console.print(
                    "[bold red]This operation requires a model checkpoint. Please select one [M].[/bold red]"
                )
                wait_for_enter("Press [Enter] to continue...")
                continue
            _WIZARD_SESSION_STATE[op.id] = dict(effective_params)
            _WIZARD_SELECTED_MODELS[op.id] = selected_model
            return effective_params, selected_model

        elif choice == "E":
            # Combination Explorer for this operation
            from python.cli.discovery import get_command_catalog
            from python.cli.interactive_discovery import run_combination_explorer

            catalog = get_command_catalog()
            node = catalog.get_command(op.direct_command)
            if node:
                run_combination_explorer(node)
            else:
                console.print(
                    f"[bold red]Could not find parser node for {op.direct_command}[/bold red]"
                )
                wait_for_enter("Press [Enter] to continue...")

        elif choice == "V":
            # Preview and dry-run
            from python.cli.combinations import SafeParserValidator

            cmd_tokens = [
                tok
                for tok in op.direct_command.split()
                if tok not in ("zeropawn", "chess") and not tok.startswith("<")
            ]
            for a in op.arguments:
                val = effective_params.get(a.name, a.default)
                if val != a.default or a.required:
                    if getattr(a, "is_positional", False):
                        if val is not None and str(val).strip() != "":
                            cmd_tokens.append(str(val))
                    elif a.type == "bool":
                        if val:
                            cmd_tokens.append(a.flag)
                    elif val is not None:
                        cmd_tokens.extend([a.flag, str(val)])
            if selected_model and op.requires_model != "none":
                cmd_tokens.extend(["--checkpoint", str(selected_model)])

            preview_str = f"zeropawn {' '.join(shlex.quote(t) for t in cmd_tokens)}"
            ok, err = SafeParserValidator.validate_tokens(cmd_tokens)
            status_text = (
                "[bold green]VALID (Parser accepts command)[/bold green]"
                if ok
                else f"[bold red]INVALID: {err}[/bold red]"
            )
            console.print(
                Panel(
                    f"[bold cyan]Command Preview:[/bold cyan]\n[white]{preview_str}[/white]\n\n"
                    f"[dim]Parser Validation Status:[/] {status_text}",
                    title="[bold green]Command Preview & Dry-Run[/bold green]",
                    box=box.ROUNDED,
                    border_style="green" if ok else "red",
                )
            )
            wait_for_enter("Press [Enter] to return to Wizard...")

        elif (
            choice == "M" or (choice.isdigit() and int(choice) == idx)
        ) and op.requires_model != "none":
            chosen = select_model_interactive(default_path=selected_model)
            if chosen is not None:
                selected_model = chosen

        elif choice in ("A", "ADVANCED"):
            # Advanced custom arguments
            updated = enter_custom_arguments_mode(op, effective_params)
            if updated is not None:
                effective_params = updated

        elif choice == "S":
            # Save as preset
            p_name = console.input("[bold cyan]Enter preset name: [/bold cyan]").strip()
            if p_name:
                p_id = p_name.lower().replace(" ", "-")
                arg_list = []
                for k, v in effective_params.items():
                    for a in op.arguments:
                        if a.name == k and v != a.default:
                            if a.type == "bool" and v:
                                arg_list.append(a.flag)
                            else:
                                arg_list.extend([a.flag, str(v)])
                if selected_model:
                    arg_list.extend(["--checkpoint", str(selected_model)])
                ok = save_preset(
                    preset_id=p_id,
                    name=p_name,
                    description=f"User preset for {op.name}",
                    operation=op.id,
                    args=arg_list,
                    parameters=effective_params,
                    model=str(selected_model) if selected_model else None,
                )
                if ok:
                    console.print(f"[bold green]Preset '{p_name}' saved successfully![/bold green]")
                wait_for_enter("Press [Enter] to continue...")

        elif choice == "R":
            for a in op.arguments:
                effective_params[a.name] = a.default
            console.print("[bold green]Reset all fields to defaults.[/bold green]")

        elif choice.isdigit():
            val_idx = int(choice)
            if val_idx in row_map:
                target_arg = row_map[val_idx]
                curr_val = effective_params[target_arg.name]

                if target_arg.type == "bool":
                    # Toggle boolean
                    new_bool = not curr_val
                    effective_params[target_arg.name] = new_bool
                elif target_arg.type == "int":
                    new_int = IntPrompt.ask(
                        f"Enter new value for [cyan]{target_arg.name}[/cyan]",
                        default=int(curr_val) if curr_val is not None else 0,
                    )
                    if target_arg.min_val is not None and new_int < target_arg.min_val:
                        console.print(f"[yellow]Clamped to minimum {target_arg.min_val}[/yellow]")
                        new_int = int(target_arg.min_val)
                    if target_arg.max_val is not None and new_int > target_arg.max_val:
                        console.print(f"[yellow]Clamped to maximum {target_arg.max_val}[/yellow]")
                        new_int = int(target_arg.max_val)
                    effective_params[target_arg.name] = new_int
                elif target_arg.type == "float":
                    new_float = FloatPrompt.ask(
                        f"Enter new value for [cyan]{target_arg.name}[/cyan]",
                        default=float(curr_val) if curr_val is not None else 0.0,
                    )
                    if target_arg.min_val is not None and new_float < target_arg.min_val:
                        new_float = float(target_arg.min_val)
                    if target_arg.max_val is not None and new_float > target_arg.max_val:
                        new_float = float(target_arg.max_val)
                    effective_params[target_arg.name] = new_float
                elif target_arg.type == "choice" and target_arg.choices:
                    console.print(f"Valid choices: {', '.join(target_arg.choices)}")
                    new_c = Prompt.ask(
                        f"Enter choice for [cyan]{target_arg.name}[/cyan]",
                        choices=target_arg.choices,
                        default=str(curr_val),
                    )
                    effective_params[target_arg.name] = new_c
                elif target_arg.type in ("str", "path"):
                    new_s = Prompt.ask(
                        f"Enter new value for [cyan]{target_arg.name}[/cyan]",
                        default=str(curr_val) if curr_val is not None else "",
                    )
                    effective_params[target_arg.name] = new_s
            else:
                max_opt = len(row_map)
                m_hint = (
                    f" or {idx} / M for Model Checkpoint" if op.requires_model != "none" else ""
                )
                console.print(
                    f"[bold red]Invalid selection '{choice}'. Enter 1-{max_opt}{m_hint}, or an action like X to execute.[/bold red]"
                )


def enter_custom_arguments_mode(
    op: OperationDef,
    current_params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Path C: Advanced Custom Arguments Mode.

    Safely parses custom command-line string with shlex.split,
    preserves boundaries and paths with spaces, validates flags against op schema.
    """
    console.print(
        Panel(
            "[bold bright_cyan]Advanced Custom Arguments Mode (Path C)[/bold bright_cyan]\n"
            "Enter command-line arguments string (e.g. [cyan]--max-hours 12 --concurrent-games 24[/cyan])\n"
            "[dim]Press [Enter] with empty line to cancel.[/dim]",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
        )
    )

    try:
        raw_input = console.input("[bold yellow]Arguments > [/bold yellow]").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not raw_input:
        return None

    try:
        tokens = shlex.split(raw_input)
    except ValueError as e:
        console.print(f"[bold red]Failed to parse arguments (unbalanced quotes):[/bold red] {e}")
        wait_for_enter("Press [Enter] to continue...")
        return None

    updated_params = dict(current_params)
    flag_to_arg: Dict[str, OpArgument] = {a.flag: a for a in op.arguments}

    i = 0
    unknown_flags = []
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("-"):
            if token in flag_to_arg:
                arg_def = flag_to_arg[token]
                if arg_def.type == "bool":
                    updated_params[arg_def.name] = True
                    i += 1
                else:
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                        val_str = tokens[i + 1]
                        i += 2
                        try:
                            if arg_def.type == "int":
                                updated_params[arg_def.name] = int(val_str)
                            elif arg_def.type == "float":
                                updated_params[arg_def.name] = float(val_str)
                            else:
                                updated_params[arg_def.name] = val_str
                        except ValueError:
                            console.print(
                                f"[bold red]Invalid {arg_def.type} value '{val_str}' for flag {token}[/bold red]"
                            )
                    else:
                        console.print(f"[bold red]Flag {token} requires a value.[/bold red]")
                        i += 1
            else:
                unknown_flags.append(token)
                i += 1
        else:
            i += 1

    if unknown_flags:
        console.print(
            f"[bold yellow]Warning: Ignored unrecognized flags: {', '.join(unknown_flags)}[/bold yellow]"
        )

    console.print("[bold green]Arguments parsed and applied successfully.[/bold green]")
    wait_for_enter("Press [Enter] to return to Wizard...")
    return updated_params
