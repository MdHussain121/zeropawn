"""Built-in and persistent user preset management for ZeroPawn.

Precedence rules:
  Built-in defaults -> Saved preset values -> Explicit CLI argument overrides.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich import box
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER, console

PRESETS_FILE = REPO_ROOT / "configs" / "presets.json"

# Built-in presets designed for real workloads with documented trade-offs
BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick-smoke": {
        "id": "quick-smoke",
        "name": "Quick RL Smoke Test (30s)",
        "description": "Fast 30-second verification of self-play, MCTS search, and gradient updates.",
        "operation": "train",
        "subcommand": None,
        "args": ["--verify"],
        "parameters": {"verify": True},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Exercises whole pipeline; does not produce a stronger model.",
    },
    "fast-tests": {
        "id": "fast-tests",
        "name": "Fast Unit Test Suite",
        "description": "Runs core board, terminal logic, encoder, perft, MCTS, and metrics unit tests.",
        "operation": "test",
        "subcommand": None,
        "args": ["--suite", "fast"],
        "parameters": {"suite": "fast"},
        "model": None,
        "is_builtin": True,
        "tradeoff": "Completes in seconds; omits full integration/soak tests.",
    },
    "all-tests": {
        "id": "all-tests",
        "name": "Full Unit & Integration Test Suite",
        "description": "Runs all unit and integration test files under tests/.",
        "operation": "test",
        "subcommand": None,
        "args": ["--suite", "all"],
        "parameters": {"suite": "all"},
        "model": None,
        "is_builtin": True,
        "tradeoff": "Thorough validation; takes 1-2 minutes.",
    },
    "puzzles-smoke": {
        "id": "puzzles-smoke",
        "name": "Quick Tactical Puzzle Evaluation",
        "description": "Evaluates latest model on 20 tactical puzzles with policy and MCTS.",
        "operation": "puzzles",
        "subcommand": "eval",
        "args": ["eval", "-n", "20", "--mcts-puzzles", "5", "--mcts-sims", "30"],
        "parameters": {"limit": 20, "mcts_puzzles": 5, "mcts_sims": 30},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Fast estimate; larger sample needed for statistically strict solve rates.",
    },
    "puzzles-full": {
        "id": "puzzles-full",
        "name": "Comprehensive Puzzle Benchmark (1,000 Puzzles)",
        "description": "Full tactical benchmark measuring policy top-1 and 50-sim MCTS solve rate.",
        "operation": "puzzles",
        "subcommand": "eval",
        "args": ["eval", "-n", "1000", "--mcts-puzzles", "50", "--mcts-sims", "50"],
        "parameters": {"limit": 1000, "mcts_puzzles": 50, "mcts_sims": 50},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Highly reliable measurement; takes ~2-5 minutes depending on GPU.",
    },
    "match-short": {
        "id": "match-short",
        "name": "Quick Stockfish Match (10 Games)",
        "description": "Plays 10 games vs Stockfish with 30 MCTS simulations per move.",
        "operation": "match",
        "subcommand": "run",
        "args": ["run", "-n", "10", "--mcts-sims", "30"],
        "parameters": {"games": 10, "mcts_sims": 30},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Quick strength check; wide 95% Elo confidence interval.",
    },
    "match-standard": {
        "id": "match-standard",
        "name": "Standard Stockfish Benchmark Match (50 Games)",
        "description": "Plays 50 games vs Stockfish with 50 MCTS simulations per move for Elo tracking.",
        "operation": "match",
        "subcommand": "run",
        "args": ["run", "-n", "50", "--mcts-sims", "50"],
        "parameters": {"games": 50, "mcts_sims": 50},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Tight Elo confidence interval (+/- 30 Elo); takes ~10-15 minutes.",
    },
    "bench-throughput": {
        "id": "bench-throughput",
        "name": "Self-Play Throughput & Simulation Benchmark",
        "description": "Measures real simulations/sec and batching latency across workers.",
        "operation": "bench",
        "subcommand": "selfplay",
        "args": ["selfplay", "--trials", "1", "--counted-sec", "5"],
        "parameters": {"trials": 1, "counted_sec": 5},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Short 5s counted run; isolates hardware inference throughput.",
    },
    "batched-match": {
        "id": "batched-match",
        "name": "Hardware-Autotuned Batched Match (20 Games)",
        "description": "Plays 20 games vs Stockfish using C++ batched MCTS and autotuned batch size.",
        "operation": "match",
        "subcommand": "run",
        "args": [
            "run",
            "-n",
            "20",
            "--mcts-sims",
            "50",
            "--search-mode",
            "batched",
            "--batch-size",
            "auto",
        ],
        "parameters": {
            "games": 20,
            "mcts_sims": 50,
            "search_mode": "batched",
            "batch_size": "auto",
        },
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Up to 3-5x faster game completion via batched GPU evaluation.",
    },
    "batched-puzzles": {
        "id": "batched-puzzles",
        "name": "Hardware-Autotuned Batched Tactical Puzzles",
        "description": "Evaluates 50 puzzles using C++ batched MCTS with autotuned batch size.",
        "operation": "puzzles",
        "subcommand": "eval",
        "args": [
            "eval",
            "-n",
            "50",
            "--mcts-puzzles",
            "50",
            "--mcts-sims",
            "50",
            "--search-mode",
            "batched",
            "--mcts-batch-size",
            "auto",
        ],
        "parameters": {
            "limit": 50,
            "mcts_puzzles": 50,
            "mcts_sims": 50,
            "search_mode": "batched",
            "mcts_batch_size": "auto",
        },
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Significantly faster tactical validation on GPU without accuracy degradation.",
    },
    "batch-sweep": {
        "id": "batch-sweep",
        "name": "Extended Batch Scaling Sweep (B=1 to 256)",
        "description": "Measures inference latency and full MCTS search scaling across powers of 2.",
        "operation": "bench",
        "subcommand": "batch-sweep",
        "args": ["batch-sweep", "--batch-sizes", "1,4,8,16,24,32,64,128,256", "--sims", "400"],
        "parameters": {"batch_sizes": "1,4,8,16,24,32,64,128,256", "sims": 400},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Comprehensive benchmark across all batch candidates (~15-30s on CUDA).",
    },
    "concurrency-sweep": {
        "id": "concurrency-sweep",
        "name": "Worker Concurrency & Pipelining Delay Sweep",
        "description": "Sweeps concurrent workers (1-32) and batching delays (0.5-3.0ms) to identify throughput peaks.",
        "operation": "bench",
        "subcommand": "concurrency-sweep",
        "args": [
            "concurrency-sweep",
            "--concurrencies",
            "1,2,4,8,16,24,32",
            "--delays",
            "0.5,1.0,2.0,2.5,3.0",
        ],
        "parameters": {"concurrencies": "1,2,4,8,16,24,32", "delays": "0.5,1.0,2.0,2.5,3.0"},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Maps multi-threaded GPU queuing characteristics and saturation plateau.",
    },
    "supervised-smoke": {
        "id": "supervised-smoke",
        "name": "Supervised Learning Smoke Test (10 Steps)",
        "description": "10-step supervised learning verification verifying loss calculation and gradients.",
        "operation": "supervised",
        "subcommand": None,
        "args": ["--verify"],
        "parameters": {"verify": True},
        "model": None,
        "is_builtin": True,
        "tradeoff": "Checks training loop and dataset access in under 15 seconds.",
    },
    "supervised-standard": {
        "id": "supervised-standard",
        "name": "Standard Supervised Warm-Start (5 Epochs)",
        "description": "Trains 12b x 128f network on human games dataset with effective batch size 512.",
        "operation": "supervised",
        "subcommand": None,
        "args": ["--epochs", "5", "--batch-size", "512", "--lr", "0.0002"],
        "parameters": {"epochs": 5, "batch_size": 512, "lr": 0.0002},
        "model": None,
        "is_builtin": True,
        "tradeoff": "Produces calibrated warm-start model; requires GPU and 10-30 minutes.",
    },
    "guardrails-check": {
        "id": "guardrails-check",
        "name": "Correctness Guardrails & Numerical Verification",
        "description": "Validates value signs, move probabilities, invariance, and numerical bounds.",
        "operation": "test",
        "subcommand": "guardrails",
        "args": ["guardrails"],
        "parameters": {},
        "model": "default",
        "is_builtin": True,
        "tradeoff": "Deterministic sanity check for RL training stability.",
    },
    "tablebase-check": {
        "id": "tablebase-check",
        "name": "Syzygy Tablebase Adjudication Feasibility",
        "description": "Probes 2 games to check Syzygy tablebase hits and adjudication feasibility.",
        "operation": "tablebase",
        "subcommand": "feasibility",
        "args": ["feasibility", "--max-games", "2"],
        "parameters": {"max_games": 2},
        "model": None,
        "is_builtin": True,
        "tradeoff": "Requires data/syzygy tablebase files; fast check.",
    },
}


def load_user_presets() -> Dict[str, Dict[str, Any]]:
    """Load user presets from configs/presets.json. Returns empty dict on error/missing."""
    if not PRESETS_FILE.is_file():
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "presets" in data and isinstance(data["presets"], dict):
                return data["presets"]
            elif isinstance(data, dict):
                # Flat format fallback
                return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def save_user_presets(presets: Dict[str, Dict[str, Any]]) -> bool:
    """Save user presets atomically to configs/presets.json."""
    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "updated_at": datetime.now().isoformat(),
        "presets": presets,
    }
    # Atomic write pattern
    tmp_path = PRESETS_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if PRESETS_FILE.exists():
            PRESETS_FILE.unlink()
        tmp_path.rename(PRESETS_FILE)
        return True
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        console.print(f"[bold red]Failed to save presets:[/] {e}")
        return False


def get_all_presets() -> Dict[str, Dict[str, Any]]:
    """Return combined dictionary of built-in and user presets."""
    combined = dict(BUILTIN_PRESETS)
    user_p = load_user_presets()
    for k, v in user_p.items():
        v["is_builtin"] = False
        combined[k] = v
    return combined


def get_preset_by_id(preset_id: str) -> Optional[Dict[str, Any]]:
    """Look up a preset by its ID (built-in or user-defined)."""
    return get_all_presets().get(preset_id)


def save_preset(
    preset_id: str,
    name: str,
    description: str,
    operation: str,
    args: List[str],
    parameters: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    subcommand: Optional[str] = None,
) -> bool:
    """Create or update a user-defined preset."""
    if preset_id in BUILTIN_PRESETS:
        console.print(
            f"[bold red]Cannot overwrite built-in preset '{preset_id}'. Use duplicate instead.[/bold red]"
        )
        return False

    user_presets = load_user_presets()
    user_presets[preset_id] = {
        "id": preset_id,
        "name": name,
        "description": description,
        "operation": operation,
        "subcommand": subcommand,
        "args": args,
        "parameters": parameters or {},
        "model": model,
        "is_builtin": False,
        "created_at": user_presets.get(preset_id, {}).get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
    }
    return save_user_presets(user_presets)


def delete_preset(preset_id: str) -> bool:
    """Delete a user-defined preset. Built-in presets cannot be deleted."""
    if preset_id in BUILTIN_PRESETS:
        console.print(f"[bold red]Cannot delete built-in preset '{preset_id}'.[/bold red]")
        return False

    user_presets = load_user_presets()
    if preset_id not in user_presets:
        console.print(f"[bold yellow]Preset '{preset_id}' not found.[/bold yellow]")
        return False

    del user_presets[preset_id]
    return save_user_presets(user_presets)


def duplicate_preset(source_id: str, new_id: str, new_name: str) -> bool:
    """Duplicate any preset (built-in or user) into a new user preset."""
    src = get_preset_by_id(source_id)
    if not src:
        console.print(f"[bold red]Source preset '{source_id}' does not exist.[/bold red]")
        return False

    return save_preset(
        preset_id=new_id,
        name=new_name,
        description=f"Duplicate of {src['name']}",
        operation=src["operation"],
        args=list(src.get("args", [])),
        parameters=dict(src.get("parameters", {})),
        model=src.get("model"),
        subcommand=src.get("subcommand"),
    )


def export_preset(preset_id: str, export_path: Union[str, Path]) -> bool:
    """Export a single preset to an external JSON file."""
    src = get_preset_by_id(preset_id)
    if not src:
        console.print(f"[bold red]Preset '{preset_id}' not found.[/bold red]")
        return False

    out_p = Path(export_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(src, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[bold red]Export failed:[/] {e}")
        return False


def import_preset(import_path: Union[str, Path]) -> bool:
    """Import a preset from an external JSON file with validation."""
    in_p = Path(import_path)
    if not in_p.is_file():
        console.print(f"[bold red]Import file not found: {in_p}[/bold red]")
        return False

    try:
        with open(in_p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Failed to parse JSON preset:[/] {e}")
        return False

    if not isinstance(data, dict) or "id" not in data or "operation" not in data:
        console.print("[bold red]Invalid preset schema: missing 'id' or 'operation'.[/bold red]")
        return False

    pid = str(data["id"])
    if pid in BUILTIN_PRESETS:
        pid = f"imported-{pid}"

    return save_preset(
        preset_id=pid,
        name=data.get("name", pid),
        description=data.get("description", "Imported preset"),
        operation=data["operation"],
        args=data.get("args", []),
        parameters=data.get("parameters", {}),
        model=data.get("model"),
        subcommand=data.get("subcommand"),
    )


def render_presets_table(filter_op: Optional[str] = None) -> List[Tuple[int, str, Dict[str, Any]]]:
    """Display all presets in a styled Rich table and return indexed lookup list."""
    all_p = get_all_presets()
    items = []

    table = Table(
        title="[bold bright_cyan]Workload & Experiment Presets[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="bold yellow")
    table.add_column("Type", style="dim")
    table.add_column("Preset Name", style="bright_white")
    table.add_column("Operation", style="cyan")
    table.add_column("Command Preview", style="green")
    table.add_column("Description", style="dim")

    idx = 1
    for pid, p in sorted(all_p.items(), key=lambda x: (not x[1].get("is_builtin", False), x[0])):
        if filter_op and p.get("operation") != filter_op:
            continue
        type_str = "[green]Built-in[/green]" if p.get("is_builtin", False) else "[cyan]User[/cyan]"
        cmd_prev = f"{p['operation']} " + " ".join(p.get("args", []))
        if len(cmd_prev) > 40:
            cmd_prev = cmd_prev[:37] + "..."

        table.add_row(
            str(idx),
            type_str,
            p["name"],
            p["operation"],
            cmd_prev,
            p.get("description", "")[:50],
        )
        items.append((idx, pid, p))
        idx += 1

    console.print(table)
    return items
