"""Stockfish match, live watch, and interactive play subcommand for ZeroPawn."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich import box

from python.cli.common import (
    console,
    print_banner,
    print_error,
    print_success,
    print_warning,
    print_info,
    get_default_checkpoint_path,
    get_default_stockfish_path,
    load_engine_model,
    render_ascii_board,
    resolve_device,
    resolve_stockfish,
    safe_resolve,
    validate_fen_or_raise,
)
from python.cli.eval_callbacks import make_serial_eval_fn
from python.model.checkpoint_registry import DefaultCheckpointError
from python.common.progress import (
    Heartbeat,
    banner,
    emit_error,
    emit_report,
    error_block,
    event_line,
    header_block,
    parameters_table,
    results_table,
    run_context,
    short_hash,
    summary_block,
)
import logging

logger = logging.getLogger(__name__)


def _elo_diff_ci(wins: int, draws: int, losses: int) -> str:
    """Elo difference + 95% Agresti-Caffo CI from W/D/L.

    Replaces legacy clamped Wald binomial SE with canonical Agresti-Caffo logit CI.
    """
    from python.eval.stats import elo_confidence_interval

    n = wins + draws + losses
    if n <= 0:
        return "n/a (no games)"
    res = elo_confidence_interval(wins, draws, losses)
    elo = res["elo_diff"]
    lo, hi = res["lo"], res["hi"]
    if elo is not None and abs(elo) < 1e-9:
        elo = 0.0
    if elo is None:
        return f"n/a (score 0% or 100%: unbounded) [{lo:+.1f}, {hi:+.1f}]"
    half = (hi - lo) / 2.0
    return f"{elo:+.1f} +/- {half:.1f} (95% CI Agresti-Caffo: [{lo:+.1f}, {hi:+.1f}], N={n})"


def _append_match_ledger(entry: dict, ledger_path: str = "runs/matches/match_ledger.jsonl") -> None:
    """Append a match summary record to the persistent match ledger."""
    try:
        p = Path(ledger_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to write to match ledger %s: %s", ledger_path, e)


def _validate_match_numerics(args: argparse.Namespace) -> Optional[str]:
    """Return error string if numeric args out of range, else None."""
    if getattr(args, "games", None) is not None and args.games < 2:
        return "--games must be >= 2."
    if getattr(args, "stockfish_nodes", None) is not None and args.stockfish_nodes < 10:
        return "--stockfish-nodes must be >= 10."
    if getattr(args, "mcts_sims", None) is not None and args.mcts_sims < 1:
        return "--mcts-sims must be >= 1."
    if getattr(args, "workers", None) is not None and not (1 <= args.workers <= 32):
        return "--workers must be in 1..32."
    if getattr(args, "c_puct", None) is not None and args.c_puct < 0:
        return "--c-puct must be >= 0."
    if getattr(args, "delay", None) is not None and args.delay < 0:
        return "--delay must be >= 0."
    if getattr(args, "batch_size", None) is not None:
        bs_str = str(args.batch_size).strip().lower()
        if bs_str != "auto":
            try:
                bs = int(bs_str)
                if not (1 <= bs <= 256):
                    return "--batch-size must be in 1..256 or 'auto'."
            except ValueError:
                return "--batch-size must be an integer (1..256) or 'auto'."
    return None


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "match",
        aliases=["play", "stockfish", "vs"],
        help="Make our AI play against Stockfish (batch match, live watch, or play as human).",
        description="Benchmarking against Stockfish: Run 100-game match pairs, watch a live game move-by-move, or play against our AI.",
    )

    match_subparsers = parser.add_subparsers(dest="match_subcommand", help="Mode of match/play")

    # Mode 1: Batch Match (default)
    batch_parser = match_subparsers.add_parser(
        "run",
        help="Run batch match against Stockfish across opening pairs.",
    )
    _add_batch_args(batch_parser)

    # Mode 2: Live Watch
    live_parser = match_subparsers.add_parser(
        "live",
        aliases=["watch"],
        help="Watch a single live game between our AI and Stockfish in the terminal.",
    )
    _add_live_args(live_parser)

    # Mode 3: Human vs AI
    human_parser = match_subparsers.add_parser(
        "human",
        help="Play an interactive game against our AI directly in your terminal.",
    )
    _add_human_args(human_parser)

    # Also add arguments to top-level 'match' so running `match -n 10` works directly
    _add_batch_args(parser)
    parser.set_defaults(func=route_match)


def _add_search_mode_arg(parser: argparse.ArgumentParser) -> None:
    """Explicit search-mode setting (default serial; batched is opt-in)."""
    parser.add_argument(
        "--search-mode",
        choices=["serial", "batched"],
        default="serial",
        help="MCTS search implementation (default: serial; 'batched' is opt-in "
        "and requires a build with the batched C++ search path)",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="MCTS batch size for batched search mode ('auto' for hardware-tuned, or integer 1-256; default: auto)",
    )


def _add_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=20,
        help="Total number of games to play (plays pairs on openings, default: 20)",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to AI model checkpoint (default: auto-detect latest). "
            "'' is the explicit no-model signal and is rejected here "
            "(this command requires network weights)."
        ),
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: auto-detect tools/stockfish.exe)",
    )
    parser.add_argument(
        "--stockfish-nodes",
        type=int,
        default=1000,
        help="Node limit for Stockfish search (default: 1000)",
    )
    parser.add_argument(
        "--elo",
        "--stockfish-elo",
        dest="elo",
        type=int,
        default=None,
        help="Stockfish UCI_Elo limit (default: None, full strength or node-limited)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="Simulations per move for our MCTS search (default: 50)",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="PUCT exploration constant (default: 2.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent match worker threads (default: 4)",
    )
    parser.add_argument(
        "--reuse-tree",
        action="store_true",
        default=False,
        help="Retain and advance MCTS search tree between moves during play-time search (default: False)",
    )
    parser.add_argument(
        "--openings",
        type=str,
        default="data/openings/eval_openings_50.epd",
        help="Path to opening EPD book (default: data/openings/eval_openings_50.epd)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--book",
        type=str,
        default="data/openings/curated_book.json",
        help="Path to curated opening book JSON (default: data/openings/curated_book.json)",
    )
    parser.add_argument(
        "--no-book",
        action="store_true",
        help="Disable opening book usage during match play",
    )
    parser.add_argument(
        "--book-policy",
        choices=["best", "weighted_blend", "frequency", "score", "uniform"],
        default="best",
        help="Opening book selection policy for matches (default: best)",
    )
    parser.add_argument(
        "--pgn",
        type=str,
        default=None,
        help="Path to output PGN file (default: auto-saved in runs/matches/match_<stamp>.pgn)",
    )
    parser.add_argument(
        "--no-pgn",
        action="store_true",
        help="Disable automatic PGN recording for the match",
    )
    parser.add_argument(
        "--output",
        "--json",
        dest="output_json",
        type=str,
        default=None,
        help="Path to output JSON match summary (default: auto-saved in runs/matches/match_<stamp>.json)",
    )
    parser.add_argument(
        "--cap-policy",
        choices=["exclude", "draw", "loss", "adjudicate"],
        default="exclude",
        help="Termination policy for games reaching maximum plies (default: exclude)",
    )
    parser.add_argument(
        "--enable-value-adjudication",
        action="store_true",
        default=False,
        help="Enable value-head resignation/win adjudication (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for opening shuffle and evaluation (default: None)",
    )
    _add_search_mode_arg(parser)


def _add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine-color",
        choices=["white", "black"],
        default="white",
        help="Color for our AI engine (default: white)",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="Starting position FEN (default: standard start position)",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to AI model checkpoint (default: auto-detect latest). "
            "'' is the explicit no-model signal and is rejected here "
            "(this command requires network weights)."
        ),
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: auto-detect tools/stockfish.exe)",
    )
    parser.add_argument(
        "--stockfish-nodes",
        type=int,
        default=1000,
        help="Node limit for Stockfish search (default: 1000)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=150,
        help="Simulations per move for our MCTS search (default: 150)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Delay in seconds between moves for visualization (default: 0.8s)",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Wait for user to press Enter after each move",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--reuse-tree",
        action="store_true",
        default=False,
        help="Retain and advance MCTS search tree between moves during play-time search (default: False)",
    )
    parser.add_argument(
        "--search-mode",
        choices=["serial", "batched"],
        default="serial",
        help="MCTS search implementation (default: serial; note: match live only supports serial search)",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="1",
        help="MCTS batch size for match live (default: 1; note: match live only supports serial search with batch_size=1)",
    )


def _add_human_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--human-color",
        choices=["white", "black"],
        default="white",
        help="Color you will play as (default: white)",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to AI model checkpoint (default: auto-detect latest). "
            "'' is the explicit no-model signal and is rejected here "
            "(this command requires network weights)."
        ),
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=200,
        help="Simulations per move for our AI (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: auto cuda/cpu)",
    )
    _add_search_mode_arg(parser)


def route_match(args: argparse.Namespace) -> int:
    subcmd = getattr(args, "match_subcommand", None)
    if subcmd in ["live", "watch"]:
        return run_live_game(args)
    elif subcmd == "human":
        return run_human_game(args)
    else:
        return run_batch_match(args)


def run_batch_match(args: argparse.Namespace) -> int:
    from python.eval.match_runner import MatchRunner

    ctx = run_context("match")
    t0 = time.time()
    log_path = "logs/zeropawn.log"

    def _fail(what: str, likely: str, try_: str) -> int:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1

    emit_report(banner("match", ctx["start"] + " (local)"))

    err = _validate_match_numerics(args)
    if err:
        return _fail(err, "argument out of range", "fix the flagged argument, then see LOG")

    try:
        # S5-07: "" is the explicit no-model signal (engine contract c20cb95)
        # and must not collapse into the designated default. Batch matches
        # require weights, so an explicit "" is rejected instead.
        if args.checkpoint is None:
            ckpt_path = get_default_checkpoint_path()
        elif args.checkpoint == "":
            return _fail(
                "--checkpoint '' (explicit no-model) is not supported by 'match': "
                "match play requires network weights.",
                "explicit empty --checkpoint",
                "omit --checkpoint for the designated default, or pass a .pt path, then see LOG",
            )
        else:
            ckpt_path = args.checkpoint
    except DefaultCheckpointError as e:
        return _fail(
            "No usable default model checkpoint.",
            str(e),
            "restore/designate the registry default, or pass --checkpoint <file.pt>, then see LOG",
        )
    try:
        ckpt_path = safe_resolve(ckpt_path)
    except (ValueError, OSError) as e:
        return _fail(
            f"Invalid --checkpoint path: {e}",
            "path escapes repo or is malformed",
            "pass a repo-relative .pt path, then see LOG",
        )

    try:
        sf_path = (
            resolve_stockfish(args.stockfish_path)
            if args.stockfish_path
            else get_default_stockfish_path()
        )
    except (ValueError, FileNotFoundError) as e:
        return _fail(
            str(e), "stockfish binary not found", "install tools/stockfish.exe, then see LOG"
        )
    if not sf_path or not Path(sf_path).is_file():
        return _fail(
            f"Stockfish binary not found at '{sf_path}'. Please check tools/stockfish.exe",
            "missing engine binary",
            "install tools/stockfish.exe, then see LOG",
        )

    try:
        device = resolve_device(args.device)
    except ValueError as e:
        return _fail(
            str(e), "unknown --device value", "pass --device cuda or --device cpu, then see LOG"
        )

    try:
        model, info = load_engine_model(ckpt_path, device=device)
    except FileNotFoundError as e:
        logger.debug("load_engine_model failed: %s", e)
        return _fail(
            f"Model checkpoint not found: {ckpt_path}\n"
            "To obtain pre-trained weights:\n"
            "  1. Run 'zeropawn models download' or specify '--checkpoint <path>'\n"
            "  2. Or train a baseline checkpoint via 'zeropawn supervised --verify'",
            "missing checkpoint file",
            "download or specify --checkpoint <path>, then see LOG",
        )
    except (RuntimeError, OSError, ValueError) as e:
        logger.debug("load_engine_model failed: %s", e)
        return _fail(
            f"Failed to load engine model: {e}",
            "corrupt/incompatible checkpoint or no VRAM",
            "check the file and free VRAM, then see LOG",
        )

    try:
        openings_path = safe_resolve(args.openings)
    except (ValueError, OSError) as e:
        return _fail(
            f"Invalid --openings path: {e}",
            "path escapes repo or is malformed",
            "pass a repo-relative openings book, then see LOG",
        )
    if not openings_path.is_file():
        return _fail(
            f"Openings book not found at: {openings_path}",
            "wrong path",
            "check --openings, then see LOG",
        )

    # Number of openings = games // 2 (since each opening played twice)
    num_openings = max(1, args.games // 2)
    total_games = num_openings * 2

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    runs_dir = Path("runs/matches")
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = short_hash(str(ckpt_path))

    if getattr(args, "no_pgn", False):
        pgn_path = None
    elif getattr(args, "pgn", None):
        pgn_path = str(safe_resolve(args.pgn))
    else:
        pgn_path = str(runs_dir / f"match_{timestamp_str}_{ckpt_tag}.pgn")

    if getattr(args, "output_json", None):
        json_path = str(safe_resolve(args.output_json))
    else:
        json_path = str(runs_dir / f"match_{timestamp_str}_{ckpt_tag}.json")

    ckpt_id = f"{Path(ckpt_path).name} step={info['step']} sha={short_hash(str(ckpt_path))} (M)"
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            ckpt_id,
            f"{device} (M)",
            "n/a (fixed opening order)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("games", args.games, "user" if args.games != 20 else "default"),
                ("openings_total", f"{total_games} ({num_openings} x 2 colors) (D)", "default"),
                (
                    "checkpoint",
                    str(args.checkpoint) if args.checkpoint else "auto",
                    "user" if args.checkpoint else "default",
                ),
                (
                    "stockfish_path",
                    str(args.stockfish_path) if args.stockfish_path else "auto",
                    "user" if args.stockfish_path else "default",
                ),
                (
                    "stockfish_nodes",
                    args.stockfish_nodes,
                    "user" if args.stockfish_nodes != 1000 else "default",
                ),
                ("mcts_sims", args.mcts_sims, "user" if args.mcts_sims != 50 else "default"),
                ("c_puct", args.c_puct, "user" if args.c_puct != 2.0 else "default"),
                ("workers", args.workers, "user" if args.workers != 4 else "default"),
                (
                    "openings",
                    str(args.openings),
                    "user"
                    if str(args.openings) != "data/openings/eval_openings_50.epd"
                    else "default",
                ),
                (
                    "device",
                    str(args.device) if args.device else "auto",
                    "user" if args.device else "default",
                ),
                (
                    "search_mode",
                    getattr(args, "search_mode", "serial"),
                    "user" if getattr(args, "search_mode", "serial") != "serial" else "default",
                ),
                (
                    "batch_size",
                    getattr(args, "batch_size", "auto"),
                    "user" if getattr(args, "batch_size", "auto") != "auto" else "default",
                ),
                (
                    "pgn",
                    pgn_path or "disabled",
                    "user"
                    if getattr(args, "pgn", None) or getattr(args, "no_pgn", False)
                    else "default (auto)",
                ),
                (
                    "output_json",
                    json_path,
                    "user" if getattr(args, "output_json", None) else "default (auto)",
                ),
                (
                    "cap_policy",
                    getattr(args, "cap_policy", "exclude"),
                    "user" if getattr(args, "cap_policy", "exclude") != "exclude" else "default",
                ),
            ]
        )
    )

    runner = MatchRunner(
        model=model,
        device=device,
        stockfish_path=sf_path,
        stockfish_nodes=args.stockfish_nodes,
        stockfish_elo=getattr(args, "elo", None),
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        openings_path=openings_path,
        num_workers=args.workers,
        search_mode=getattr(args, "search_mode", "serial"),
        batch_size=getattr(args, "batch_size", "auto"),
        use_book=not getattr(args, "no_book", False),
        book_path=getattr(args, "book", "data/openings/curated_book.json"),
        book_policy=getattr(args, "book_policy", "best"),
        reuse_tree=getattr(args, "reuse_tree", False),
        cap_policy=getattr(args, "cap_policy", "exclude"),
        enable_value_adjudication=getattr(args, "enable_value_adjudication", False),
        seed=getattr(args, "seed", None),
    )

    emit_report(
        event_line(
            "MATCH",
            f"playing {total_games} games "
            f"({num_openings} openings x 2 colors, {args.workers} workers)",
        )
    )
    # Heartbeat while the runner works: per-game progress lives in the log
    # file (runner-owned), so the console gets a liveness line on silence.
    hb = Heartbeat(30.0)
    hb_stop = threading.Event()

    def _beat() -> None:
        while not hb_stop.wait(1.0):
            line = hb.maybe_emit(f"match running ({total_games} games, per-game progress in LOG)")
            if line is not None:
                emit_report(line)

    hb_thread = threading.Thread(target=_beat, name="match-heartbeat", daemon=True)
    hb_thread.start()
    interrupted = False
    results = None
    try:
        t_match = time.time()
        results = runner.run_match(
            num_openings=num_openings,
            pgn_path=pgn_path,
        )
        elapsed = time.time() - t_match
        if results.get("interrupted", False):
            interrupted = True
    except KeyboardInterrupt:
        elapsed = time.time() - t_match
        interrupted = True
        logger.info("Match interrupted by user.")
    finally:
        hb_stop.set()

    if results is None:
        results = {
            "total_games": 0,
            "engine_wins": 0,
            "engine_draws": 0,
            "engine_losses": 0,
            "genuine_draws": 0,
            "capped_games": 0,
            "effective_games": 0,
            "engine_score": 0.0,
            "draw_rate": 0.0,
            "elo_diff": 0.0,
            "elo_ci_95": [-math.inf, math.inf],
            "interrupted": True,
        }

    match_status = "INTERRUPTED" if interrupted else "COMPLETED"

    # Save JSON summary
    summary_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "match_status": match_status,
        "command": ctx["command"],
        "git_commit": ctx["commit"],
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": short_hash(str(ckpt_path)),
        "device": str(device),
        "stockfish_path": str(sf_path),
        "stockfish_nodes": args.stockfish_nodes,
        "stockfish_elo": getattr(args, "elo", None),
        "mcts_sims": args.mcts_sims,
        "c_puct": args.c_puct,
        "workers": args.workers,
        "search_mode": getattr(args, "search_mode", "serial"),
        "openings_path": str(openings_path),
        "use_book": not getattr(args, "no_book", False),
        "book_path": getattr(args, "book", "data/openings/curated_book.json"),
        "book_policy": getattr(args, "book_policy", "best"),
        "cap_policy": getattr(args, "cap_policy", "exclude"),
        "enable_value_adjudication": getattr(args, "enable_value_adjudication", False),
        "pgn_path": pgn_path,
        "json_path": json_path,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_record, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write JSON summary %s: %s", json_path, e)

    # Append to match ledger
    ledger_entry = {
        "timestamp": summary_record["timestamp"],
        "match_status": match_status,
        "checkpoint": Path(ckpt_path).name,
        "checkpoint_sha256": short_hash(str(ckpt_path)),
        "git_commit": ctx["commit"],
        "stockfish_nodes": args.stockfish_nodes,
        "stockfish_elo": getattr(args, "elo", None),
        "total_games_scheduled": total_games,
        "total_games_completed": results.get("total_games", 0),
        "effective_games": results.get("effective_games", results.get("total_games", 0)),
        "engine_wins": results.get("engine_wins", 0),
        "engine_draws": results.get("engine_draws", 0),
        "engine_losses": results.get("engine_losses", 0),
        "genuine_draws": results.get("genuine_draws", results.get("engine_draws", 0)),
        "capped_games": results.get("capped_games", 0),
        "engine_score": results.get("engine_score", 0.0),
        "elo_diff": results.get("elo_diff", 0.0),
        "elo_ci_95": results.get("elo_ci_95", [0.0, 0.0]),
        "pairing_analysis": results.get("pairing_analysis", {}),
        "pgn_path": pgn_path,
        "json_path": json_path,
    }
    _append_match_ledger(ledger_entry)

    # Display results (same numbers, ASCII key=value layout + Elo CI)
    w = results.get("engine_wins", 0)
    d = results.get("engine_draws", 0)
    l = results.get("engine_losses", 0)
    genuine_d = results.get("genuine_draws", d)
    capped = results.get("capped_games", 0)
    eff_n = results.get("effective_games", w + d + l)
    score = results.get("engine_score", 0.0)

    pairing = results.get("pairing_analysis", {})
    paired_str = "n/a"
    if pairing and pairing.get("pairs_count", 0) > 0:
        p_elo = pairing.get("elo_diff", 0.0)
        p_lo, p_hi = pairing.get("ci_95", (0.0, 0.0))
        p_half = (p_hi - p_lo) / 2.0
        paired_str = f"{p_elo:+.1f} +/- {p_half:.1f} (paired 95%: [{p_lo:+.1f}, {p_hi:+.1f}])"

    res_items = [
        ("scheduled_games", f"{total_games}"),
        ("completed_games", f"{results['total_games']} (M)"),
        ("effective_games", f"{eff_n} (M, decisive + genuine draws)"),
        ("engine_wins", f"{w} (M)"),
        ("genuine_draws", f"{genuine_d} (M)"),
        ("capped_games", f"{capped} (M, cap_policy={getattr(args, 'cap_policy', 'exclude')})"),
        ("engine_losses", f"{l} (M)"),
        ("effective_score", f"{score * 100:.1f}% (M, (W+0.5*D_genuine)/N_eff)"),
        ("search_mode", f"{results.get('search_mode', 'serial')} (M)"),
        ("draw_rate", f"{results.get('draw_rate', 0.0) * 100:.1f}% (M)"),
        ("elo_diff_95ci", f"{_elo_diff_ci(w, genuine_d, l)}"),
        ("paired_elo_diff_95ci", f"{paired_str}"),
        ("pgn_file", pgn_path or "disabled"),
        ("json_file", json_path),
        (
            "crashes_illegal_forfeits",
            "n/a (not tracked by runner; non-decisive engine faults count as losses)",
        ),
        (
            "total_time",
            f"{elapsed:.1f}s ({elapsed / max(1, results['total_games']):.1f}s/game) (M)",
        ),
    ]
    emit_report(results_table(res_items))

    if interrupted:
        emit_report(
            event_line(
                "INTERRUPTED",
                f"Match stopped early: completed {results['total_games']} of {total_games} scheduled games. Results saved to {json_path}.",
            )
        )
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 1))
        return 130

    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    return 0


def run_live_game(args: argparse.Namespace) -> int:
    import zeropawn_cpp
    from python.eval.match_runner import StockfishPlayer

    print_banner("LIVE GAME: AI VS STOCKFISH", "Watch real-time move-by-move match on terminal")

    err = _validate_match_numerics(args)
    if err:
        print_error(err)
        return 1

    try:
        # S5-07: "" is the explicit no-model signal (engine contract c20cb95)
        # and must not collapse into the designated default. This command
        # requires weights, so an explicit "" is rejected instead.
        if args.checkpoint is None:
            ckpt_path = get_default_checkpoint_path()
        elif args.checkpoint == "":
            print_error(
                "--checkpoint '' (explicit no-model) is not supported by this "
                "command: it requires network weights."
            )
            return 1
        else:
            ckpt_path = args.checkpoint
    except DefaultCheckpointError as e:
        print_error(str(e))
        return 1
    try:
        ckpt_path = safe_resolve(ckpt_path)
    except (ValueError, OSError) as e:
        print_error(f"Invalid --checkpoint path: {e}")
        return 1

    try:
        sf_path = (
            resolve_stockfish(args.stockfish_path)
            if args.stockfish_path
            else get_default_stockfish_path()
        )
    except (ValueError, FileNotFoundError) as e:
        print_error(str(e))
        return 1
    if not sf_path or not Path(sf_path).is_file():
        print_error(f"Stockfish binary not found at '{sf_path}'.")
        return 1

    try:
        device = resolve_device(args.device)
    except ValueError as e:
        print_error(str(e))
        return 1
    try:
        fen = validate_fen_or_raise(args.fen)
    except ValueError as e:
        print_error(str(e))
        return 1
    try:
        model, info = load_engine_model(ckpt_path, device=device)
    except FileNotFoundError as e:
        print_error(
            f"{e}\n"
            "To obtain pre-trained weights:\n"
            "  1. Run 'zeropawn models download' or specify '--checkpoint <path>'\n"
            "  2. Or train a baseline checkpoint via 'zeropawn supervised --verify'"
        )
        return 1
    except (RuntimeError, OSError, ValueError) as e:
        print_error(f"Failed to load checkpoint '{ckpt_path}': {e}")
        return 1

    from python.eval.match_runner import require_serial_search

    search_mode = getattr(args, "search_mode", "serial")
    if search_mode != "serial":
        print_error(
            f"match live only supports serial search (search_mode='serial'); received search_mode={search_mode!r}."
        )
        return 1

    raw_bs = getattr(args, "batch_size", "1")
    bs_str = str(raw_bs).strip().lower()
    if bs_str == "auto":
        print_error(
            "match live only supports serial search with batch_size=1; received batch_size='auto'."
        )
        return 1
    try:
        bs_val = int(bs_str)
        if bs_val > 1:
            print_error(
                f"match live only supports serial search with batch_size=1; received batch_size={bs_val}."
            )
            return 1
        if bs_val < 1:
            print_error(f"--batch-size must be >= 1; received batch_size={bs_val}.")
            return 1
    except ValueError:
        print_error(f"--batch-size must be an integer (1..256) or 'auto'; received {raw_bs!r}.")
        return 1

    try:
        require_serial_search(search_mode, "match live")
    except RuntimeError as e:
        print_error(str(e))
        return 1

    stockfish = StockfishPlayer(stockfish_path=sf_path, nodes=args.stockfish_nodes)

    mcts_cfg = zeropawn_cpp.MCTSConfig()
    mcts_cfg.c_puct = 2.0
    mcts_cfg.add_noise = False
    mcts_cfg.virtual_loss = 1
    mcts_cfg.reuse_tree = getattr(args, "reuse_tree", False)
    mcts_cfg.batch_size = 1
    mcts = zeropawn_cpp.MCTS(mcts_cfg)

    # Shared factory: encodes the passed game directly (history preserved,
    # plane 18 repetition live).  The former inline callback FEN-round-tripped
    # the game and zeroed the history planes (audit M-1).
    eval_fn = make_serial_eval_fn(model, device)

    game = zeropawn_cpp.ChessGame(fen)
    engine_is_white = args.engine_color.lower() == "white"

    console.print(
        Panel(
            f"AI Engine: [bold green]{Path(ckpt_path).name}[/bold green] (Color: {args.engine_color.upper()})\n"
            f"Stockfish: [bold cyan]Stockfish (nodes {args.stockfish_nodes})[/bold cyan]\n"
            f"Search mode: [bold yellow]{search_mode}[/bold yellow]\n"
            f"MCTS Simulations per move: [bold yellow]{args.mcts_sims}[/bold yellow]",
            box=box.ROUNDED,
        )
    )

    console.print(render_ascii_board(game.get_fen()))
    move_count = 1

    try:
        while not game.is_game_over() and move_count <= 200:
            # NOTE: side_to_move() returns bool (True = White to move).
            side = game.side_to_move()
            is_engine_turn = bool(side) == engine_is_white
            side_str = "White" if side else "Black"

            if is_engine_turn:
                console.print(
                    f"\n[bold green]✦ AI ({side_str}) is thinking ({args.mcts_sims} sims)...[/bold green]"
                )
                res = mcts.search(game, args.mcts_sims, eval_fn)
                move_uci = game.action_to_string(res.best_action)
                root_val = res.root_value
                console.print(
                    f"[green]AI Move:[/green] [bold yellow]{move_uci}[/bold yellow]  |  [dim]Eval: {root_val:+.3f}[/dim]"
                )
            else:
                console.print(
                    f"\n[bold cyan]✦ Stockfish ({side_str}) is calculating...[/bold cyan]"
                )
                move_uci = stockfish.get_move(game.get_fen())
                console.print(f"[cyan]Stockfish Move:[/cyan] [bold yellow]{move_uci}[/bold yellow]")

            if not move_uci or not game.is_legal(move_uci):
                print_error(f"Illegal or empty move received: '{move_uci}'")
                break

            game.make_move_uci(move_uci)
            console.print(render_ascii_board(game.get_fen(), highlight_move=move_uci))
            console.print(f"[dim]FEN: {game.get_fen()}[/dim]")

            if args.step:
                input("Press [Enter] for next move...")
            else:
                time.sleep(args.delay)

            move_count += 1

        # Game outcome
        console.print("\n" + "=" * 50)
        if game.is_checkmate():
            # Loser is the side to move; winner is the other side.
            winner = "Black" if game.side_to_move() else "White"
            print_success(f"CHECKMATE! {winner} wins!")
        elif game.is_stalemate():
            print_info("STALEMATE! Game drawn.")
        elif game.is_50_move_rule():
            print_info("DRAW by 50-move rule.")
        elif game.is_repetition():
            print_info("DRAW by threefold repetition.")
        else:
            print_info("Game terminated.")

    except KeyboardInterrupt:
        print_warning("\nLive game stopped by user.")
    finally:
        stockfish.close()

    return 0


def run_human_game(args: argparse.Namespace) -> int:
    import zeropawn_cpp

    print_banner("HUMAN VS AI CHESS", "Challenge our Neural Network Engine")

    try:
        # S5-07: "" is the explicit no-model signal (engine contract c20cb95)
        # and must not collapse into the designated default. This command
        # requires weights, so an explicit "" is rejected instead.
        if args.checkpoint is None:
            ckpt_path = get_default_checkpoint_path()
        elif args.checkpoint == "":
            print_error(
                "--checkpoint '' (explicit no-model) is not supported by this "
                "command: it requires network weights."
            )
            return 1
        else:
            ckpt_path = args.checkpoint
    except DefaultCheckpointError as e:
        print_error(str(e))
        return 1
    try:
        ckpt_path = safe_resolve(ckpt_path)
    except (ValueError, OSError) as e:
        print_error(f"Invalid --checkpoint path: {e}")
        return 1

    err = _validate_match_numerics(args)
    if err:
        print_error(err)
        return 1
    try:
        device = resolve_device(args.device)
    except ValueError as e:
        print_error(str(e))
        return 1
    try:
        model, info = load_engine_model(ckpt_path, device=device)
    except FileNotFoundError as e:
        print_error(
            f"{e}\n"
            "To obtain pre-trained weights:\n"
            "  1. Run 'zeropawn models download' or specify '--checkpoint <path>'\n"
            "  2. Or train a baseline checkpoint via 'zeropawn supervised --verify'"
        )
        return 1
    except (RuntimeError, OSError, ValueError) as e:
        print_error(f"Failed to load checkpoint '{ckpt_path}': {e}")
        return 1

    from python.eval.match_runner import require_serial_search

    search_mode = getattr(args, "search_mode", "serial")
    try:
        require_serial_search(search_mode, "match human")
    except RuntimeError as e:
        print_error(str(e))
        return 1

    mcts_cfg = zeropawn_cpp.MCTSConfig()
    mcts_cfg.c_puct = 2.0
    mcts_cfg.add_noise = False
    mcts = zeropawn_cpp.MCTS(mcts_cfg)

    # Shared factory: encodes the passed game directly (history preserved).
    # The former inline callback FEN-round-tripped the game (audit M-1).
    eval_fn = make_serial_eval_fn(model, device)

    game = zeropawn_cpp.ChessGame()
    human_is_white = args.human_color.lower() == "white"

    console.print(
        Panel(
            f"You are playing as [bold white]{args.human_color.upper()}[/bold white].\n"
            f"Search mode: [bold yellow]{search_mode}[/bold yellow]\n"
            f"Enter moves in UCI notation (e.g. 'e2e4', 'g1f3', 'e7e8q').\n"
            f"Type 'quit' or 'exit' to end the game.",
            box=box.ROUNDED,
        )
    )

    console.print(render_ascii_board(game.get_fen()))

    while not game.is_game_over():
        side = game.side_to_move()
        is_human_turn = bool(side) == human_is_white

        if is_human_turn:
            legal_moves = game.legal_moves_uci()
            while True:
                user_input = console.input("[bold cyan]Your move > [/bold cyan]").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print_info("Game terminated by user.")
                    return 0
                if user_input in legal_moves:
                    game.make_move_uci(user_input)
                    break
                print_error(
                    f"Illegal move: '{user_input}'. Legal moves include: {', '.join(legal_moves[:8])}..."
                )
        else:
            console.print(f"[dim]AI thinking ({args.mcts_sims} sims)...[/dim]")
            mcts.clear()
            res = mcts.search(game, args.mcts_sims, eval_fn)
            move_uci = game.action_to_string(res.best_action)
            root_val = res.root_value
            console.print(
                f"[bold green]AI plays:[/bold green] [bold yellow]{move_uci}[/bold yellow] (Eval: {root_val:+.2f})"
            )
            game.make_move_uci(move_uci)

        console.print(render_ascii_board(game.get_fen()))

    if game.is_checkmate():
        # Loser is the side to move (True = White mated); human wins iff the
        # loser is the AI side.
        winner = "You win!" if (bool(game.side_to_move()) != human_is_white) else "AI wins!"
        print_success(f"CHECKMATE! {winner}")
    elif game.is_stalemate():
        print_info("DRAW by Stalemate!")
    else:
        print_info("DRAW!")

    return 0
