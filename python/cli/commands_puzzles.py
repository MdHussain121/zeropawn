"""Puzzle evaluation and interactive solving subcommand for ZeroPawn."""

from __future__ import annotations

import argparse
import sys
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
    get_default_checkpoint_path,
    load_engine_model,
    render_ascii_board,
    resolve_device,
    safe_resolve,
    validate_fen_or_raise,
)
from python.common.progress import (
    banner,
    emit_error,
    emit_report,
    error_block,
    event_line,
    format_kv_line,
    header_block,
    parameters_table,
    results_table,
    run_context,
    short_hash,
    summary_block,
)
from python.eval.baselines import get_baseline_for_model, set_baseline_for_model
from python.eval.puzzles import PuzzleEvaluator, load_puzzles
import logging

logger = logging.getLogger(__name__)


def _validate_puzzle_numerics(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "mcts_sims", None) is not None and args.mcts_sims < 1:
        return "--mcts-sims must be >= 1."
    if getattr(args, "mcts_puzzles", None) is not None and args.mcts_puzzles < 0:
        return "--mcts-puzzles must be >= 0."
    if getattr(args, "batch_size", None) is not None and args.batch_size < 1:
        return "--batch-size must be >= 1."
    if getattr(args, "mcts_batch_size", None) is not None:
        mbs_str = str(args.mcts_batch_size).strip().lower()
        if mbs_str != "auto":
            try:
                mbs = int(mbs_str)
                if not (1 <= mbs <= 256):
                    return "--mcts-batch-size must be in 1..256 or 'auto'."
            except ValueError:
                return "--mcts-batch-size must be an integer (1..256) or 'auto'."
    if getattr(args, "limit", None) is not None and args.limit is not None and args.limit < 1:
        return "--limit must be >= 1."
    for attr in ("min_rating", "max_rating"):
        v = getattr(args, attr, None)
        if v is not None and not (200 <= v <= 3500):
            return f"--{attr.replace('_', '-')} must be in 200..3500."
    mn, mx = getattr(args, "min_rating", None), getattr(args, "max_rating", None)
    if mn is not None and mx is not None and mn > mx:
        return "--min-rating must be <= --max-rating."
    if getattr(args, "start_index", None) is not None and args.start_index < 0:
        return "--start-index must be >= 0."
    return None


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "puzzles",
        aliases=["puzzle", "p"],
        help="Evaluate model accuracy on tactical puzzles or solve interactively.",
        description="Benchmark policy and MCTS search on tactical puzzles, or step through puzzles interactively.",
    )

    puzzle_subparsers = parser.add_subparsers(dest="puzzle_subcommand", help="Puzzle mode")

    # Mode 1: Benchmark / Eval
    bench_parser = puzzle_subparsers.add_parser(
        "eval",
        aliases=["benchmark", "bench"],
        help="Benchmark model policy and MCTS solve rates across puzzle dataset.",
    )
    _add_bench_args(bench_parser)

    # Mode 2: Interactive Solve
    solve_parser = puzzle_subparsers.add_parser(
        "solve",
        aliases=["interactive"],
        help="Interactive puzzle solver: view boards, guess moves, or watch AI solve.",
    )
    _add_solve_args(solve_parser)

    # Mode 3: Active Error-Driven Learning
    learn_parser = puzzle_subparsers.add_parser(
        "learn",
        aliases=["learn-mistakes", "train-mistakes"],
        help="Active learning from tactical mistakes: mines blunders and updates weights live.",
    )
    _add_learn_args(learn_parser)

    # Also register default arguments on main puzzle parser
    _add_bench_args(parser)
    parser.set_defaults(func=route_puzzles)


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--puzzles-file",
        "-p",
        type=str,
        default="data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        help="Path to puzzles CSV file (default: data/puzzles/lichess_puzzles_1000_1200_1600.csv)",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect latest)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of puzzles to evaluate (default: all)",
    )
    parser.add_argument(
        "--mcts-puzzles",
        type=int,
        default=50,
        help="Number of puzzles to test with MCTS search (default: 50, 0 to skip)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="MCTS simulations per puzzle (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for policy evaluation (default: 128)",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="Filter puzzles by minimum rating",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=None,
        help="Filter puzzles by maximum rating",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Baseline solve rate (%%) to compare against (default: auto-detect for model or 35.6%%)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update the baseline for this model in puzzle_baselines.json with this result",
    )
    parser.add_argument(
        "--search-mode",
        choices=["serial", "batched"],
        default="serial",
        help="MCTS search implementation (default: serial; 'batched' is opt-in "
        "and requires a build with the batched C++ search path)",
    )
    parser.add_argument(
        "--mcts-batch-size",
        type=str,
        default="auto",
        help="MCTS batch size for batched search mode ('auto' for hardware-tuned, or integer 1-256; default: auto)",
    )


def _add_solve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--puzzles-file",
        "-p",
        type=str,
        default="data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        help="Path to puzzles CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting puzzle index (default: 0)",
    )
    parser.add_argument(
        "--human-guess",
        action="store_true",
        help="Prompt you to guess the move before revealing AI answer",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="Simulations per puzzle for AI (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: auto cuda/cpu)",
    )


def _add_learn_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--puzzles-file",
        "-p",
        type=str,
        default="data/puzzles/tactics_train_300k.csv",
        help="Path to puzzles CSV file (auto-downloads from Lichess if missing)",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect latest)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=250,
        help="Number of puzzles to stream & practice on (default: 250)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Number of mistakes to accumulate before gradient step (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Fine-tuning learning rate (default: 2e-5 for gentle regularization)",
    )
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=0.50,
        help="Fraction of batch reserved for solved/anchor regularizer positions (default: 0.50)",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1200,
        help="Minimum puzzle rating (default: 1200)",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=1800,
        help="Maximum puzzle rating (default: 1800)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models/checkpoints_rl/mistake_learned_latest.pt",
        help="Path to save improved checkpoint",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 1,000-puzzle benchmark before and after to verify improvement",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Run autonomous belt-by-belt progression from start rating up to 3000 Elo",
    )
    parser.add_argument(
        "--start-rating",
        type=int,
        default=1000,
        help="Curriculum starting rating (default: 1000)",
    )
    parser.add_argument(
        "--end-rating",
        type=int,
        default=3000,
        help="Curriculum ending rating (default: 3000)",
    )
    parser.add_argument(
        "--tier-puzzles",
        type=int,
        default=250,
        help="Puzzles per curriculum tier (default: 250)",
    )
    parser.add_argument(
        "--max-tier-puzzles",
        type=int,
        default=600,
        help="Max puzzles per tier before advancing (default: 600)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--sweet-spot",
        action="store_true",
        help="Focus on core weakness band 1100-2100 Elo (audit P0, §11)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Multi-pass repetition passes per mistake batch (default: 3, audit §3)",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.25,
        help="Replay buffer fraction (default: 0.25, audit §1.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (audit §10, D10)",
    )


def route_puzzles(args: argparse.Namespace) -> int:
    subcmd = getattr(args, "puzzle_subcommand", None)
    if subcmd in ["solve", "interactive"]:
        return run_solve_puzzles(args)
    elif subcmd in ["learn", "learn-mistakes", "train-mistakes"]:
        return run_learn_puzzles(args)
    else:
        return run_eval_puzzles(args)


def run_learn_puzzles(args: argparse.Namespace) -> int:
    from python.train.learn_from_mistakes import (
        MistakeLearner,
        ensure_puzzle_dataset,
        load_puzzle_items,
        run_curriculum,
        DEFAULT_BENCHMARK_PATH,
        CurriculumTier,
    )
    from rich.table import Table

    device = resolve_device(args.device)
    ckpt_path = args.checkpoint or get_default_checkpoint_path()
    if not ckpt_path:
        print_error("No model checkpoint found! Please specify --checkpoint.")
        return 1

    print_banner(
        "ZeroPawn Tactical Mistake Learner",
        "Autonomous Curriculum (1000 -> 3000 Elo)"
        if getattr(args, "curriculum", False)
        else "Active Learning from Offline Puzzles",
    )

    console.print(
        f"[cyan]Loading model from:[/cyan] [bold]{Path(ckpt_path).name}[/bold] on [bold]{device}[/bold]..."
    )
    model, info = load_engine_model(ckpt_path, device=device)
    print_success(
        f"Model loaded: {info.get('num_blocks', 12)} blocks, {info.get('num_filters', 128)} filters."
    )

    learner = MistakeLearner(
        model=model,
        lr=args.lr,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Autonomous Curriculum Mode ──────────────────────────────────────────
    if getattr(args, "curriculum", False):
        start_rating = getattr(args, "start_rating", 1000)
        end_rating = getattr(args, "end_rating", 3000)
        tier_puzzles = getattr(args, "tier_puzzles", 250)
        max_tier_puzzles = getattr(args, "max_tier_puzzles", 600)

        console.print(
            f"[bold magenta]Starting Autonomous Curriculum Progression:[/bold magenta] "
            f"Ratings [cyan]{start_rating}[/cyan] to [cyan]{end_rating}[/cyan] | "
            f"[yellow]{tier_puzzles}[/yellow] puzzles/tier (adaptive max: {max_tier_puzzles})\n"
        )

        def on_tier_start(t: CurriculumTier):
            console.print(
                Panel(
                    f"[bold cyan]Tier {t.tier_number}: {t.name}[/bold cyan] | "
                    f"Rating Band: [bold yellow]{t.min_rating} - {t.max_rating}[/bold yellow] | "
                    f"Mastery Target: [bold green]{t.mastery_threshold * 100:.0f}%[/bold green]",
                    box=box.ROUNDED,
                    border_style="cyan",
                )
            )

        def on_batch_train(batch_len: int, metrics: dict[str, float], resolved: int):
            console.print(
                f"    [magenta]⚡ Gradient Step:[/magenta] {batch_len} mistakes | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Retest: [green]{resolved}/{batch_len} mastered ({resolved / batch_len * 100:.1f}%)[/green]"
            )

        def on_tier_complete(t: CurriculumTier, res: dict, bench: dict | None):
            status = (
                "[bold green]GRADUATED ✔[/bold green]"
                if res["initial_solve_rate"] >= t.mastery_threshold
                else "[yellow]TIER COMPLETED[/yellow]"
            )
            console.print(
                f"  {status} -> Solved: [bold cyan]{res['initial_solved']}/{res['total_puzzles']} ({res['initial_solve_rate'] * 100:.1f}%)[/bold cyan] | "
                f"Mistakes Trained: {res['total_mistakes_mined']} | "
                f"Mastered on Retest: [green]{res['mistakes_resolved_on_retest']}[/green]"
            )
            if bench:
                console.print(
                    f"  [dim]Benchmark score after Tier {t.tier_number}: {bench['solve_rate'] * 100:.2f}%[/dim]"
                )
            console.print("")

        bench_cb = None
        if getattr(args, "benchmark", False) and DEFAULT_BENCHMARK_PATH.exists():

            def _bench(m):
                evaluator = PuzzleEvaluator(
                    model=m, device=device, puzzle_csv_path=DEFAULT_BENCHMARK_PATH
                )
                return evaluator.evaluate_policy(batch_size=128)

            bench_cb = _bench

        t0 = time.time()
        summaries = run_curriculum(
            learner=learner,
            start_rating=start_rating,
            end_rating=end_rating,
            base_tier_puzzles=tier_puzzles,
            max_tier_puzzles=max_tier_puzzles,
            anchor_ratio=getattr(args, "anchor_ratio", 0.50),
            benchmark_callback=bench_cb,
            on_tier_start=on_tier_start,
            on_tier_complete=on_tier_complete,
            on_batch_train=on_batch_train,
        )
        total_time = time.time() - t0

        final_ckpt = Path("models/checkpoints_rl/curriculum_final_latest.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "source": "autonomous_curriculum",
                "start_rating": start_rating,
                "end_rating": end_rating,
                "timestamp": time.time(),
            },
            final_ckpt,
        )
        print_success(f"Autonomous Curriculum Complete! Final model saved to {final_ckpt}")

        grand_table = Table(title="Autonomous Curriculum Progression Report", box=box.ROUNDED)
        grand_table.add_column("Tier", justify="center", style="bold")
        grand_table.add_column("Rating Band", style="yellow")
        grand_table.add_column("Focus", style="cyan")
        grand_table.add_column("Puzzles", justify="right")
        grand_table.add_column("Solve Rate", justify="right")
        grand_table.add_column("Mistakes Mined", justify="right")
        grand_table.add_column("Mastered Retest", justify="right", style="green")
        grand_table.add_column("Status", justify="center")

        for s in summaries:
            t = s["tier"]
            r = s["results"]
            grad_text = (
                "[bold green]PASSED[/bold green]" if s["graduated"] else "[yellow]ADVANCED[/yellow]"
            )
            grand_table.add_row(
                str(t.tier_number),
                f"{t.min_rating}-{t.max_rating}",
                t.name,
                str(r["total_puzzles"]),
                f"{r['initial_solve_rate'] * 100:.1f}%",
                str(r["total_mistakes_mined"]),
                str(r["mistakes_resolved_on_retest"]),
                grad_text,
            )
        console.print(grand_table)
        console.print(f"[dim]Total Curriculum Runtime: {total_time:.1f}s[/dim]")
        return 0

    # ── Single-Set Mode ─────────────────────────────────────────────────────
    puzzle_path = safe_resolve(args.puzzles_file)
    console.print(f"[cyan]Checking puzzle dataset:[/cyan] [bold]{puzzle_path}[/bold] ...")

    def on_dl_progress(cur: int, tot: int):
        console.print(f"  [dim]Downloaded {cur}/{tot} puzzles from Lichess...[/dim]")

    puzzle_path = ensure_puzzle_dataset(
        csv_path=puzzle_path,
        target_count=max(args.count, 1000),
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        progress_callback=on_dl_progress,
    )

    puzzles = load_puzzle_items(puzzle_path, limit=args.count)
    if not puzzles:
        print_error("No valid puzzles loaded.")
        return 1

    init_bench = None
    if getattr(args, "benchmark", False) and DEFAULT_BENCHMARK_PATH.exists():
        console.print(
            "\n[bold yellow]Running baseline benchmark on 1,000 held-out Lichess puzzles...[/bold yellow]"
        )
        evaluator = PuzzleEvaluator(
            model=model, device=device, puzzle_csv_path=DEFAULT_BENCHMARK_PATH
        )
        init_bench = evaluator.evaluate_policy(batch_size=128)
        print_success(
            f"Baseline Solve Rate: {init_bench['solve_rate'] * 100:.2f}% ({init_bench['solved']}/{init_bench['total']})"
        )

    console.print("\n[bold]Starting live error-driven learning session...[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("Puzzle #", justify="right", style="cyan")
    table.add_column("Rating", justify="right", style="yellow")
    table.add_column("Status", justify="center")
    table.add_column("Model Choice", justify="left")
    table.add_column("Target Move", justify="left")
    table.add_column("Detail", justify="left")

    def on_puzzle_result(num: int, p, is_solved: bool, chosen: str, prob: float):
        if not is_solved:
            table.add_row(
                str(num),
                str(p.rating),
                "[bold red]MISS[/bold red]",
                f"{chosen} ({prob:.2f})",
                f"[bold green]{p.solution_move}[/bold green]",
                "[red]Mistake captured into buffer[/red]",
            )
        elif num % 25 == 0:
            table.add_row(
                str(num),
                str(p.rating),
                "[bold green]PASS[/bold green]",
                f"{chosen} ({prob:.2f})",
                f"{p.solution_move}",
                "[dim]Tactical pattern already known[/dim]",
            )

    updates_count = [0]

    def on_batch_train(batch_len: int, metrics: dict[str, float], resolved: int):
        updates_count[0] += 1
        console.print(
            f"  [bold magenta]⚡ Gradient Step #{updates_count[0]}:[/bold magenta] "
            f"Trained on {batch_len} mistakes | Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Instant Retest: [bold green]{resolved}/{batch_len} mistakes mastered ({resolved / batch_len * 100:.1f}%)[/bold green]"
        )

    t0 = time.time()
    results = learner.learn_from_puzzle_stream(
        puzzles=puzzles,
        anchor_ratio=getattr(args, "anchor_ratio", 0.50),
        on_puzzle_result=on_puzzle_result,
        on_batch_train=on_batch_train,
    )
    elapsed = time.time() - t0

    console.print("\n")
    console.print(table)

    post_bench = None
    if getattr(args, "benchmark", False) and DEFAULT_BENCHMARK_PATH.exists():
        console.print(
            "\n[bold yellow]Re-running benchmark on 1,000 held-out Lichess puzzles...[/bold yellow]"
        )
        evaluator = PuzzleEvaluator(
            model=model, device=device, puzzle_csv_path=DEFAULT_BENCHMARK_PATH
        )
        post_bench = evaluator.evaluate_policy(batch_size=128)

    out_file = safe_resolve(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model": model.state_dict(),
        "num_blocks": info.get("num_blocks", 12),
        "num_filters": info.get("num_filters", 128),
        "source": "mistake_learner",
        "puzzles_practiced": args.count,
        "mistakes_mined": results["total_mistakes_mined"],
        "timestamp": time.time(),
    }
    torch.save(checkpoint_payload, out_file)
    print_success(f"Updated weights saved to {out_file}")

    summary_table = Table(title="Tactical Learning Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_row("Puzzles Evaluated", str(results["total_puzzles"]))
    summary_table.add_row(
        "Initial Solved",
        f"{results['initial_solved']} ({results['initial_solve_rate'] * 100:.1f}%)",
    )
    summary_table.add_row("Mistakes Mined", str(results["total_mistakes_mined"]))
    summary_table.add_row(
        "Mistakes Mastered On Retest",
        f"[green]{results['mistakes_resolved_on_retest']} ({results['retest_resolution_rate'] * 100:.1f}%)[/green]",
    )
    summary_table.add_row("Average Training Loss", f"{results['avg_loss']:.4f}")
    summary_table.add_row(
        "Total Time", f"{elapsed:.1f}s ({results['total_puzzles'] / elapsed:.1f} puzzles/sec)"
    )

    if init_bench and post_bench:
        delta = (post_bench["solve_rate"] - init_bench["solve_rate"]) * 100
        delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        summary_table.add_row(
            "Held-Out 1,000 Benchmark Before", f"{init_bench['solve_rate'] * 100:.2f}%"
        )
        summary_table.add_row(
            "Held-Out 1,000 Benchmark After",
            f"[bold green]{post_bench['solve_rate'] * 100:.2f}%[/bold green] ({delta_str})",
        )

    console.print(summary_table)
    return 0


def run_eval_puzzles(args: argparse.Namespace) -> int:
    ctx = run_context("puzzles-eval")
    t0 = time.time()
    log_path = "logs/zeropawn.log"

    def _fail(what: str, likely: str, try_: str) -> int:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1

    emit_report(banner("puzzles-eval", ctx["start"] + " (local)"))

    err = _validate_puzzle_numerics(args)
    if err:
        return _fail(err, "argument out of range", "fix the flagged argument, then see LOG")

    ckpt_path = args.checkpoint or get_default_checkpoint_path()
    if not ckpt_path:
        return _fail(
            "No model checkpoint found! Please specify --checkpoint.",
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
        device = resolve_device(args.device)
    except ValueError as e:
        return _fail(
            str(e), "unknown --device value", "pass --device cuda or --device cpu, then see LOG"
        )
    try:
        model, info = load_engine_model(ckpt_path, device=device)
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
        logger.debug("load_engine_model failed: %s", e)
        return _fail(
            f"Failed to load engine model: {e}",
            "corrupt/incompatible checkpoint or no VRAM",
            "check the file and free VRAM, then see LOG",
        )

    puzzle_path = None
    if getattr(args, "puzzles_file", None):
        try:
            p_cand = safe_resolve(args.puzzles_file)
            if p_cand.exists():
                puzzle_path = p_cand
        except (ValueError, OSError):
            pass
    if puzzle_path is None:
        for cand_name in [
            "data/puzzles/lichess_puzzles_5000_1000_1600.csv",
            "data/puzzles/lichess_puzzles_5000_1800_2800.csv",
            "data/puzzles/tactics_train_clean.csv",
            "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
            "data/puzzles",
        ]:
            try:
                p_cand = safe_resolve(cand_name)
                if p_cand.exists():
                    puzzle_path = p_cand
                    break
            except (ValueError, OSError):
                pass
    if puzzle_path is None or not puzzle_path.exists():
        return _fail(
            f"Puzzle file/dir not found: {args.puzzles_file}",
            "wrong path",
            "check --puzzles-file, then see LOG",
        )

    from python.eval.puzzles import PuzzleEvaluator

    all_puzzles = load_puzzles(puzzle_path)
    filtered = all_puzzles
    if args.min_rating is not None:
        filtered = [p for p in filtered if p.rating >= args.min_rating]
    if args.max_rating is not None:
        filtered = [p for p in filtered if p.rating <= args.max_rating]
    if args.limit is not None:
        filtered = filtered[: args.limit]

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
            "n/a (fixed puzzle order)",
        )
    )
    emit_report(
        parameters_table(
            [
                (
                    "puzzles_file",
                    str(args.puzzles_file),
                    "user"
                    if args.puzzles_file != "data/puzzles/lichess_puzzles_1000_1200_1600.csv"
                    else "default",
                ),
                (
                    "checkpoint",
                    str(args.checkpoint) if args.checkpoint else "auto",
                    "user" if args.checkpoint else "default",
                ),
                ("limit", args.limit, "user" if args.limit is not None else "default"),
                (
                    "mcts_puzzles",
                    args.mcts_puzzles,
                    "user" if args.mcts_puzzles != 50 else "default",
                ),
                ("mcts_sims", args.mcts_sims, "user" if args.mcts_sims != 50 else "default"),
                ("batch_size", args.batch_size, "user" if args.batch_size != 128 else "default"),
                (
                    "min_rating",
                    args.min_rating,
                    "user" if args.min_rating is not None else "default",
                ),
                (
                    "max_rating",
                    args.max_rating,
                    "user" if args.max_rating is not None else "default",
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
                    "mcts_batch_size",
                    getattr(args, "mcts_batch_size", "auto"),
                    "user" if getattr(args, "mcts_batch_size", "auto") != "auto" else "default",
                ),
                (
                    "baseline",
                    args.baseline,
                    "user" if getattr(args, "baseline", None) is not None else "default",
                ),
                (
                    "update_baseline",
                    getattr(args, "update_baseline", False),
                    "user" if getattr(args, "update_baseline", False) else "default",
                ),
            ]
        )
    )
    emit_report(
        format_kv_line([("puzzles", f"{len(filtered)} (M)"), ("model", f"{Path(ckpt_path).name}")])
    )

    evaluator = PuzzleEvaluator(
        model=model,
        device=device,
        puzzles=filtered,
        search_mode=getattr(args, "search_mode", "serial"),
        batch_size=getattr(args, "mcts_batch_size", "auto"),
    )

    # 1. Policy Top-1 Evaluation
    emit_report(event_line("PHASE", f"policy top-1 over {len(filtered)} puzzles"))
    t_phase = time.perf_counter()
    policy_res = evaluator.evaluate_policy(batch_size=args.batch_size)
    t_policy = time.perf_counter() - t_phase

    solved_policy = policy_res["solved"]
    total_policy = policy_res["total"]
    policy_rate = policy_res["solve_rate"] * 100.0

    # 2. MCTS Search Evaluation
    mcts_rate = 0.0
    mcts_solved = 0
    mcts_total = 0
    t_mcts = 0.0

    if args.mcts_puzzles > 0:
        num_mcts = min(args.mcts_puzzles, len(filtered))
        emit_report(event_line("PHASE", f"MCTS {args.mcts_sims} sims over {num_mcts} puzzles"))
        t_phase = time.perf_counter()
        mcts_res = evaluator.evaluate_mcts(num_simulations=args.mcts_sims, max_puzzles=num_mcts)
        t_mcts = time.perf_counter() - t_phase
        mcts_rate = mcts_res["solve_rate"] * 100.0
        mcts_solved = mcts_res["solved"]
        mcts_total = mcts_res["total"]

    model_baseline = (
        float(args.baseline)
        if getattr(args, "baseline", None) is not None
        else get_baseline_for_model(ckpt_path)
    )

    # Results (same numbers, ASCII key=value layout)
    if model_baseline is not None:
        diff_policy = policy_rate - model_baseline
        sign_policy = "+" if diff_policy >= 0 else ""
        vs_policy_str = (
            f"; vs_baseline {sign_policy}{diff_policy:.2f}% (D, baseline {model_baseline:.1f}%)"
        )
    else:
        vs_policy_str = "; vs_baseline N/A (no baseline recorded)"

    rows = [
        ("search_mode", f"{getattr(args, 'search_mode', 'serial')} (M)"),
        (
            "policy_top1",
            f"{solved_policy}/{total_policy} = {policy_rate:.2f}% (M) in {t_policy:.2f}s (M){vs_policy_str}",
        ),
    ]
    if args.mcts_puzzles > 0:
        if model_baseline is not None:
            diff_mcts = mcts_rate - model_baseline
            sign_mcts = "+" if diff_mcts >= 0 else ""
            vs_mcts_str = (
                f"; vs_baseline {sign_mcts}{diff_mcts:.2f}% (D, baseline {model_baseline:.1f}%)"
            )
        else:
            vs_mcts_str = "; vs_baseline N/A (no baseline recorded)"
        rows.append(
            (
                f"mcts_{args.mcts_sims}sims",
                f"{mcts_solved}/{mcts_total} = {mcts_rate:.2f}% (M) in {t_mcts:.2f}s (M){vs_mcts_str}",
            ),
        )
    emit_report(results_table(rows))

    if getattr(args, "update_baseline", False):
        set_baseline_for_model(
            ckpt_path,
            policy_rate,
            mcts_rate=mcts_rate if args.mcts_puzzles > 0 else None,
            total_puzzles=total_policy,
        )
        emit_report(
            event_line(
                "BASELINE",
                f"updated baseline for '{Path(ckpt_path).name}' "
                f"to {policy_rate:.2f}% (saved in puzzle_baselines.json)",
            )
        )
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    return 0


def run_solve_puzzles(args: argparse.Namespace) -> int:
    import zeropawn_cpp

    print_banner("INTERACTIVE PUZZLE SOLVER", "Explore Chess Tactics and Test Engine Reasoning")

    err = _validate_puzzle_numerics(args)
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
        device = resolve_device(args.device)
    except ValueError as e:
        print_error(str(e))
        return 1
    try:
        model, info = load_engine_model(ckpt_path, device=device)
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
        logger.debug("load_engine_model failed: %s", e)
        print_error(f"Failed to load engine model: {e}")
        return 1

    puzzle_path = None
    if getattr(args, "puzzles_file", None):
        try:
            p_cand = safe_resolve(args.puzzles_file)
            if p_cand.exists():
                puzzle_path = p_cand
        except (ValueError, OSError):
            pass
    if puzzle_path is None:
        for cand_name in [
            "data/puzzles/lichess_puzzles_5000_1000_1600.csv",
            "data/puzzles/lichess_puzzles_5000_1800_2800.csv",
            "data/puzzles/tactics_train_clean.csv",
            "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
            "data/puzzles",
        ]:
            try:
                p_cand = safe_resolve(cand_name)
                if p_cand.exists():
                    puzzle_path = p_cand
                    break
            except (ValueError, OSError):
                pass
    if puzzle_path is None or not puzzle_path.exists():
        print_error(f"Puzzle file/dir not found: {args.puzzles_file}")
        return 1

    try:
        import zeropawn_cpp
    except ImportError:
        print_error(
            "zeropawn_cpp extension module is not installed or compiled. "
            "Compile it first using 'python cli.py build compile' or 'cmake --build build'."
        )
        return 1

    puzzles = load_puzzles(puzzle_path)
    total_puzzles = len(puzzles)
    idx = max(0, min(args.start_index, total_puzzles - 1))

    is_tty = bool(sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty())
    if args.human_guess and not is_tty:
        print_error("Human guess mode requires an interactive terminal (stdin is not a TTY).")
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

    score_correct = 0
    attempted = 0

    while idx < total_puzzles:
        p = puzzles[idx]
        try:
            validate_fen_or_raise(p.fen)
            g = zeropawn_cpp.ChessGame(p.fen)
        except (ValueError, RuntimeError) as e:
            logger.debug("skipping puzzle with bad FEN: %s", e)
            idx += 1
            continue
        if not g.is_legal(p.opponent_move):
            idx += 1
            continue

        g.make_move_uci(p.opponent_move)
        side_name = "White" if g.side_to_move() else "Black"

        console.print(
            Panel(
                f"Puzzle #{idx + 1} / {total_puzzles}  |  ID: [bold cyan]{p.puzzle_id}[/bold cyan]  |  Rating: [bold yellow]{p.rating}[/bold yellow]\n"
                f"Opponent just played: [bold red]{p.opponent_move}[/bold red]\n"
                f"Your turn: [bold white]{side_name} to move[/bold white] (Find the winning continuation!)",
                box=box.ROUNDED,
            )
        )

        console.print(render_ascii_board(g.get_fen(), highlight_move=p.opponent_move))

        # Human guess mode
        if args.human_guess:
            user_move = console.input(
                "[bold cyan]Enter your move (UCI or 'pass' to see AI) > [/bold cyan]"
            ).strip()
            if user_move.lower() in ["quit", "q", "exit"]:
                break
            if user_move.lower() != "pass":
                if user_move == p.solution_move:
                    print_success(f"Brilliant! {user_move} is the correct solution!")
                    score_correct += 1
                else:
                    print_error(f"{user_move} is incorrect! The solution was {p.solution_move}.")
                attempted += 1

        # Run AI solution
        console.print(f"[dim]Calculating AI response ({args.mcts_sims} MCTS simulations)...[/dim]")
        mcts.clear()
        res = mcts.search(g, args.mcts_sims, eval_fn)
        ai_move = g.action_to_string(res.best_action)
        root_val = res.root_value

        ai_is_correct = ai_move == p.solution_move
        ai_badge = (
            "[bold green]SOLVED[/bold green]" if ai_is_correct else "[bold red]FAILED[/bold red]"
        )

        console.print(
            f"AI Chose: [bold yellow]{ai_move}[/bold yellow] ({ai_badge})  |  "
            f"Eval: [dim]{root_val:+.3f}[/dim]  |  Expected: [bold green]{p.solution_move}[/bold green]"
        )

        # Prompt for next (exit cleanly if unattended / non-interactive)
        if not is_tty:
            break
        next_action = console.input(
            "\n[bold cyan]Press [Enter] for next puzzle, or 'q' to quit > [/bold cyan]"
        ).strip()
        if next_action.lower() in ["q", "quit", "exit"]:
            break
        idx += 1

    if attempted > 0:
        print_banner(
            "SESSION SUMMARY",
            f"Human Score: {score_correct} / {attempted} ({score_correct / attempted * 100:.1f}%)",
        )

    return 0
