"""Standalone MCTS Search command for ZeroPawn supporting serial and batched modes."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from rich import box
from rich.panel import Panel

from python.cli.common import (
    console,
    get_default_checkpoint_path,
    load_engine_model,
    print_error,
    print_warning,
    resolve_device,
    setup_environment,
    validate_fen_or_raise,
)
from python.hardware.profile import resolve_batch_size

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser(
        "search",
        aliases=["mcts", "think"],
        help="Execute standalone Monte Carlo Tree Search on a position (serial or batched).",
        description="Run C++ MCTS search on any FEN position with serial or hardware-tuned batched evaluation.",
    )
    search_parser.add_argument(
        "--fen",
        type=str,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="Position FEN string to analyze (default: startpos)",
    )
    search_parser.add_argument(
        "--sims",
        "-n",
        type=int,
        default=400,
        help="Number of MCTS simulations to perform (default: 400)",
    )
    search_parser.add_argument(
        "--mode",
        "--search-mode",
        dest="search_mode",
        choices=["serial", "batched"],
        default="batched",
        help="MCTS search path implementation: serial or batched (default: batched)",
    )
    search_parser.add_argument(
        "--batch-size",
        "-b",
        type=str,
        default="auto",
        help="Inference batch size for batched search ('auto' for hardware-tuned, or integer 1-256; default: auto)",
    )
    search_parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="PUCT exploration constant (default: 2.0)",
    )
    search_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device: 'cuda' or 'cpu' (default: auto)",
    )
    search_parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help="Path to neural network model checkpoint (default: auto-detect latest)",
    )
    search_parser.add_argument(
        "--top-moves",
        type=int,
        default=5,
        help="Number of top candidate moves to display (default: 5)",
    )
    search_parser.add_argument(
        "--format",
        type=str,
        default="pretty",
        choices=["pretty", "json"],
        help="Output format: pretty (Rich table) or json (default: pretty)",
    )
    search_parser.set_defaults(func=run_search)


def run_search(args: argparse.Namespace) -> int:
    """Execute standalone MCTS search on the given FEN."""
    setup_environment()
    import zeropawn_cpp

    try:
        validate_fen_or_raise(args.fen)
    except Exception as e:
        print_error(f"Invalid FEN position: {e}")
        return 1

    device = resolve_device(args.device)
    ckpt_path = args.checkpoint or get_default_checkpoint_path()
    model = None
    if ckpt_path and Path(ckpt_path).is_file():
        try:
            model, _ = load_engine_model(ckpt_path, device=device)
            model.eval()
        except Exception as e:
            print_warning(f"Could not load checkpoint ({e}); using uniform evaluator.")
            model = None

    # Autotune batch size if needed
    effective_batch_size = 1
    if args.search_mode == "batched":
        effective_batch_size = resolve_batch_size(args.batch_size, workload="uci", device=device)

    # Initialize MCTS
    mcts_config = zeropawn_cpp.MCTSConfig()
    mcts_config.c_puct = args.c_puct
    mcts_config.fpu_reduction = 0.2
    mcts_config.virtual_loss = 1
    mcts_config.add_noise = False
    mcts = zeropawn_cpp.MCTS(mcts_config)
    game = zeropawn_cpp.ChessGame(args.fen)

    # Define callbacks
    def _serial_eval(g: Any) -> Tuple[List[float], float]:
        if model is None:
            num = g.num_actions()
            return ([1.0 / num] * num, 0.0)
        planes = zeropawn_cpp.encode_state(g)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device, dtype=torch.bfloat16):
                    p_logits, v_wdl = model(t)
            else:
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            return (p_probs, float(v_probs[0] - v_probs[2]))

    def _batched_eval(batch_planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(batch_planes.shape[0])
        if n == 0 or model is None:
            return (np.zeros((n, 4672), dtype=np.float32), np.zeros(n, dtype=np.float32))
        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32))
        t = t.view(n, 19, 8, 8).to(device, non_blocking=True)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device, dtype=torch.bfloat16):
                    p_logits, v_wdl = model(t)
            else:
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            scalars = (v_probs[:, 0] - v_probs[:, 2]).astype(np.float32)
            return (p_probs.astype(np.float32), scalars)

    # Synchronize and search
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        if args.search_mode == "batched":
            res = mcts.search_batched(game, args.sims, effective_batch_size, _batched_eval)
        else:
            res = mcts.search(game, args.sims, _serial_eval)
    except KeyboardInterrupt:
        print_warning("\nSearch interrupted by user.")
        return 130
    except Exception as e:
        print_error(f"Search execution failed: {e}")
        return 1

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - t0
    throughput = args.sims / max(1e-5, elapsed_sec)

    best_action = res.best_action
    best_move = game.action_to_string(best_action)
    root_val = res.root_value
    score_cp = int(root_val * 400.0)

    # Format output
    if getattr(args, "format", "pretty") == "json":
        data = {
            "fen": args.fen,
            "best_move": best_move,
            "root_value": round(root_val, 4),
            "score_cp": score_cp,
            "simulations": args.sims,
            "search_mode": args.search_mode,
            "batch_size": effective_batch_size,
            "elapsed_ms": round(elapsed_sec * 1000.0, 2),
            "throughput_sims_s": round(throughput, 1),
            "device": device,
        }
        console.print(json.dumps(data, indent=2))
        return 0

    panel_text = (
        f"[bold cyan]Best Move:[/] [bold green]{best_move}[/bold green]  |  "
        f"[bold cyan]Score:[/] {score_cp:+d} cp (V={root_val:+.4f})  |  "
        f"[bold cyan]Mode:[/] {args.search_mode} (B={effective_batch_size})\n"
        f"[bold cyan]Throughput:[/] [bold yellow]{throughput:,.1f} sims/s[/bold yellow] "
        f"({args.sims} sims in {elapsed_sec * 1000.0:.1f} ms on {device.upper()})"
    )
    console.print(
        Panel(
            panel_text,
            title="[bold bright_cyan]MCTS Search Result[/bold bright_cyan]",
            box=box.ROUNDED,
        )
    )
    return 0
