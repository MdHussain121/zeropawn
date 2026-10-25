"""Stockfish match, live watch, and interactive play subcommand for ZeroPawn."""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Optional

import torch
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
    """Elo difference + 95% CI from W/D/L (logistic, binomial SE).

    elo = -400*log10(1/s - 1), s = (W + 0.5*D) / N;
    SE = (400/ln10) * sqrt(1 / (N*s*(1-s))); CI = elo +/- 1.96*SE.
    Returns 'n/a (reason)' at the s=0/1 edges instead of inventing numbers.
    """
    n = wins + draws + losses
    if n <= 0:
        return "n/a (no games)"
    s = (wins + 0.5 * draws) / n
    if s <= 0.0 or s >= 1.0:
        return "n/a (score 0% or 100%: unbounded)"
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = (400.0 / math.log(10.0)) * math.sqrt(1.0 / (n * s * (1.0 - s)))
    return f"{elo:+.1f} +/- {1.96 * se:.1f} (95% CI, N={n})"


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
        help="Path to AI model checkpoint (default: auto-detect latest)",
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
        help="Path to AI model checkpoint (default: auto-detect latest)",
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
        help="Path to AI model checkpoint (default: auto-detect latest)",
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

    ckpt_path = args.checkpoint or get_default_checkpoint_path()
    if not ckpt_path:
        return _fail(
            "No model checkpoint found! Please train or specify --checkpoint.",
            "no checkpoint at default locations",
            "pass --checkpoint <file.pt>, then see LOG",
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
    try:
        t_match = time.time()
        results = runner.run_match(num_openings=num_openings)
        elapsed = time.time() - t_match
    finally:
        hb_stop.set()

    # Display results (same numbers, ASCII key=value layout + Elo CI)
    w, d, l = results["engine_wins"], results["engine_draws"], results["engine_losses"]
    emit_report(
        results_table(
            [
                ("games", f"{results['total_games']} (M)"),
                ("engine_wins", f"{w} (M)"),
                ("engine_draws", f"{d} (M)"),
                ("engine_losses", f"{l} (M)"),
                ("score", f"{results['engine_score'] * 100:.1f}% (M, (W+0.5D)/N)"),
                ("search_mode", f"{results.get('search_mode', 'serial')} (M)"),
                ("draw_rate", f"{results['draw_rate'] * 100:.1f}% (M)"),
                ("elo_diff_95ci", f"{_elo_diff_ci(w, d, l)} (D, logistic+binomial)"),
                (
                    "crashes_illegal_forfeits",
                    "n/a (not tracked by runner; non-decisive engine faults count as losses)",
                ),
                ("total_time", f"{elapsed:.1f}s ({elapsed / max(1, total_games):.1f}s/game) (M)"),
            ]
        )
    )
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

    ckpt_path = args.checkpoint or get_default_checkpoint_path()
    if not ckpt_path:
        print_error("No model checkpoint found! Please specify --checkpoint.")
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

    def eval_fn(game: zeropawn_cpp.Game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

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

    ckpt_path = args.checkpoint or get_default_checkpoint_path()
    if not ckpt_path:
        print_error("No model checkpoint found! Please specify --checkpoint.")
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

    def eval_fn(game: zeropawn_cpp.Game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            t = t.to(memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

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
