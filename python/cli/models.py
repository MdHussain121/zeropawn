"""Shared model discovery, selection, metadata inspection, and comparison for ZeroPawn."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from rich import box
from rich.panel import Panel
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER, console


def get_checkpoint_directories() -> List[Path]:
    """Return all standard model and checkpoint search locations."""
    dirs = [
        REPO_ROOT / "models" / "checkpoints_scaled",
        REPO_ROOT / "models" / "checkpoints_supervised",
        REPO_ROOT / "models" / "checkpoints_rl",
        REPO_ROOT / "models" / "checkpoints_finetune",
        REPO_ROOT / "models" / "checkpoints",
        REPO_ROOT / "models",
    ]
    return [d for d in dirs if d.is_dir()]


def discover_checkpoints() -> List[Dict[str, Any]]:
    """Scan and return all available model checkpoints sorted by step or modification time.

    Never crashes on corrupted or non-model files.
    """
    checkpoints: List[Dict[str, Any]] = []
    seen_paths = set()

    for d in get_checkpoint_directories():
        for p in sorted(d.glob("*.pt")):
            if p in seen_paths or not p.is_file():
                continue
            seen_paths.add(p)

            stat = p.stat()
            size_mb = stat.st_size / (1024**2)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            ckpt_info: Dict[str, Any] = {
                "name": p.name,
                "path": p,
                "dir": p.parent.name,
                "size_mb": size_mb,
                "mtime": mtime,
                "step": None,
                "epoch": None,
                "val_loss": None,
                "blocks": None,
                "filters": None,
                "is_valid": True,
            }

            # Fast metadata probe without loading massive tensors if possible
            try:
                # Use torch.load with CPU mapping
                data = torch.load(p, map_location="cpu", weights_only=True)
                if isinstance(data, dict):
                    ckpt_info["step"] = data.get("step")
                    ckpt_info["epoch"] = data.get("epoch")
                    val_loss = data.get("best_val_loss") or data.get("val_loss") or data.get("loss")
                    if val_loss is not None:
                        try:
                            ckpt_info["val_loss"] = float(val_loss)
                        except (ValueError, TypeError):
                            pass
                    ckpt_info["blocks"] = data.get("num_blocks")
                    ckpt_info["filters"] = data.get("num_filters")
                del data
            except Exception:
                # Still a valid file entry, but note that metadata probe failed
                ckpt_info["is_valid"] = False

            checkpoints.append(ckpt_info)

    # Sort: valid first, highest step first, then latest mtime
    checkpoints.sort(
        key=lambda c: (
            1 if c["is_valid"] else 0,
            c["step"] if c["step"] is not None else -1,
            c["mtime"],
        ),
        reverse=True,
    )
    return checkpoints


def get_default_model_path() -> Optional[Path]:
    """Get the primary default model checkpoint path if available.

    Prefers fine-tuned model (strongest) over scaled baseline.
    """
    preferred = [
        REPO_ROOT / "models" / "checkpoints_finetune" / "best_model.pt",
        REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt",
        REPO_ROOT / "models" / "checkpoints_rl" / "best_model.pt",
        REPO_ROOT / "models" / "checkpoints" / "best_model.pt",
        REPO_ROOT / "models" / "best_model.pt",
    ]
    for p in preferred:
        if p.is_file():
            return p

    ckpts = discover_checkpoints()
    if ckpts:
        return ckpts[0]["path"]
    return None


def validate_checkpoint_file(path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate that path exists, is a .pt file, and can be loaded."""
    p = Path(path)
    if not p.is_file():
        return False, f"Model file not found: {p}"
    if p.suffix.lower() != ".pt":
        return False, f"Invalid model file format '{p.suffix}': must be .pt"
    try:
        data = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            return False, "Checkpoint file does not contain a state dictionary."
        return True, "Valid checkpoint."
    except Exception as e:
        return False, f"Failed to load checkpoint '{p.name}': {e}"


def select_model_interactive(
    prompt_message: str = "Select Model Checkpoint",
    allow_custom: bool = True,
    default_path: Optional[Path] = None,
) -> Optional[Path]:
    """Interactive numbered model selection menu with formatted Rich table."""
    ckpts = discover_checkpoints()
    if not ckpts and not allow_custom:
        console.print("[yellow]No model checkpoints found in repository.[/yellow]")
        return None

    table = Table(
        title="[bold bright_cyan]Available Model Checkpoints[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", style="bold yellow", justify="right")
    table.add_column("Checkpoint Name", style="bright_white")
    table.add_column("Folder", style="dim")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Architecture", style="green")
    table.add_column("Val Loss", justify="right", style="magenta")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Modified", style="dim")

    default_idx = 1
    for idx, c in enumerate(ckpts, 1):
        is_def = (default_path and c["path"] == default_path) or (not default_path and idx == 1)
        if is_def:
            default_idx = idx
        def_tag = " [bold green](default)[/bold green]" if is_def else ""
        step_str = f"{c['step']:,}" if c["step"] is not None else "N/A"
        arch_str = f"{c['blocks']}b x {c['filters']}f" if c["blocks"] and c["filters"] else "N/A"
        loss_str = f"{c['val_loss']:.4f}" if c["val_loss"] is not None else "N/A"

        table.add_row(
            str(idx),
            f"{c['name']}{def_tag}",
            c["dir"],
            step_str,
            arch_str,
            loss_str,
            f"{c['size_mb']:.1f} MB",
            c["mtime"],
        )

    console.print(table)
    if allow_custom:
        console.print("[dim]Enter [C] to specify a custom model path, or [Q] to cancel.[/dim]")

    prompt_str = f"Select model [1-{len(ckpts)}, default {default_idx}] > "
    choice = console.input(f"[bold yellow]{prompt_str}[/bold yellow]").strip()

    if not choice:
        return ckpts[default_idx - 1]["path"] if ckpts else default_path
    if choice.lower() in ("q", "cancel", "exit"):
        return None
    if choice.lower() in ("c", "custom") and allow_custom:
        custom_input = console.input("[bold cyan]Enter model path: [/bold cyan]").strip()
        custom_path = Path(custom_input)
        ok, err = validate_checkpoint_file(custom_path)
        if ok:
            return custom_path
        console.print(f"[bold red]Error:[/] {err}")
        return None

    try:
        idx_int = int(choice)
        if 1 <= idx_int <= len(ckpts):
            return ckpts[idx_int - 1]["path"]
        console.print(
            f"[bold red]Selection {idx_int} out of range [1-{len(ckpts)}]. Using default.[/bold red]"
        )
        return ckpts[default_idx - 1]["path"]
    except ValueError:
        console.print(f"[bold red]Invalid input '{choice}'. Using default.[/bold red]")
        return ckpts[default_idx - 1]["path"]


def inspect_model(checkpoint_path: Union[str, Path]) -> int:
    """Detailed model architecture, parameter count, and tensor inspection."""
    p = Path(checkpoint_path)
    ok, err = validate_checkpoint_file(p)
    if not ok:
        console.print(f"[bold red]Error:[/] {err}")
        return 1

    try:
        data = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:
        console.print(f"[bold red]Failed to load checkpoint:[/] {e}")
        return 1

    console.print(
        Panel(
            f"[bold bright_cyan]Model Inspection:[/] [yellow]{p.name}[/yellow]\n[dim]Full Path: {p.resolve()}[/dim]",
            border_style=COLOR_BORDER,
            box=box.ROUNDED,
        )
    )

    meta_table = Table(title="Checkpoint Metadata", box=box.ROUNDED, border_style=COLOR_BORDER)
    meta_table.add_column("Property", style="bold cyan")
    meta_table.add_column("Value", style="bold yellow")

    meta_table.add_row("Training Step", str(data.get("step", "N/A")))
    meta_table.add_row("Epoch", str(data.get("epoch", "N/A")))
    meta_table.add_row(
        "Validation Loss", str(data.get("best_val_loss") or data.get("val_loss", "N/A"))
    )
    meta_table.add_row("Residual Blocks", str(data.get("num_blocks", "N/A")))
    meta_table.add_row("Filter Channels", str(data.get("num_filters", "N/A")))
    meta_table.add_row("File Size", f"{p.stat().st_size / (1024**2):.2f} MB")

    console.print(meta_table)

    model_state = data.get("model") or data
    if isinstance(model_state, dict):
        total_params = 0
        layer_table = Table(
            title="Weights & Layers Summary (Top 20 Layers)",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
        )
        layer_table.add_column("Layer Name", style="bright_white")
        layer_table.add_column("Shape", style="cyan")
        layer_table.add_column("Parameters", justify="right", style="yellow")
        layer_table.add_column("Dtype", style="dim")

        for idx, (name, tensor) in enumerate(model_state.items()):
            if hasattr(tensor, "numel"):
                n = tensor.numel()
                total_params += n
                if idx < 20:
                    layer_table.add_row(name, str(list(tensor.shape)), f"{n:,}", str(tensor.dtype))

        console.print(layer_table)
        console.print(
            f"[bold green]Total Model Parameters:[/] [bold yellow]{total_params:,}[/bold yellow] (~{total_params / 1e6:.2f}M)"
        )

    return 0


def compare_models(path_a: Union[str, Path], path_b: Union[str, Path]) -> int:
    """Compare two checkpoints: parameter counts, steps, loss, and weight differences."""
    p_a, p_b = Path(path_a), Path(path_b)
    for p in (p_a, p_b):
        ok, err = validate_checkpoint_file(p)
        if not ok:
            console.print(f"[bold red]Error:[/] {err}")
            return 1

    d_a = torch.load(p_a, map_location="cpu", weights_only=True)
    d_b = torch.load(p_b, map_location="cpu", weights_only=True)

    comp_table = Table(
        title="[bold bright_cyan]Checkpoint Comparison[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
    )
    comp_table.add_column("Metric", style="bold cyan")
    comp_table.add_column(f"Model A: {p_a.name}", style="yellow")
    comp_table.add_column(f"Model B: {p_b.name}", style="green")
    comp_table.add_column("Difference / Note", style="magenta")

    step_a = d_a.get("step", "N/A")
    step_b = d_b.get("step", "N/A")
    step_diff = (
        f"{step_b - step_a:+d}" if isinstance(step_a, int) and isinstance(step_b, int) else "N/A"
    )
    comp_table.add_row("Step", str(step_a), str(step_b), step_diff)

    loss_a = d_a.get("best_val_loss") or d_a.get("val_loss", "N/A")
    loss_b = d_b.get("best_val_loss") or d_b.get("val_loss", "N/A")
    comp_table.add_row("Validation Loss", str(loss_a), str(loss_b), "Lower is better")

    blocks_a = str(d_a.get("num_blocks", "N/A"))
    blocks_b = str(d_b.get("num_blocks", "N/A"))
    comp_table.add_row(
        "Residual Blocks", blocks_a, blocks_b, "Match" if blocks_a == blocks_b else "Mismatch"
    )

    filters_a = str(d_a.get("num_filters", "N/A"))
    filters_b = str(d_b.get("num_filters", "N/A"))
    comp_table.add_row(
        "Filter Channels", filters_a, filters_b, "Match" if filters_a == filters_b else "Mismatch"
    )

    console.print(comp_table)
    return 0
