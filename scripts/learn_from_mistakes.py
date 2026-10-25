"""CLI script for error-driven learning on offline tactical puzzles.

Key Features:
1. Multi-Pass Spaced Repetition (--epochs 3-5): reiterates on mistake batches until mastered.
2. Cumulative Replay Buffer (--replay-ratio 0.25): mixes past mistakes to prevent forgetting.
3. Sweet Spot Curriculum (--sweet-spot): focuses on high-leverage 1100-2100 Elo tactics.
4. Dual Benchmark: evaluates BOTH Raw Policy accuracy AND MCTS Search solve rate.

Usage:
    # Run sweet spot curriculum (1100 -> 2100 Elo) with 3 epochs & MCTS benchmark
    python scripts/learn_from_mistakes.py --curriculum --sweet-spot --epochs 3 --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python  # Sets up DLL paths and zeropawn_cpp
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from python.cli.common import (
    get_default_checkpoint_path,
    load_engine_model,
    print_banner,
    print_error,
    print_success,
    resolve_device,
)
from python.eval.puzzles import PuzzleEvaluator, load_puzzles
from python.train.learn_from_mistakes import (
    MistakeLearner,
    ensure_puzzle_dataset,
    load_puzzle_items,
    run_curriculum,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_CURRICULUM_TIERS,
    SWEET_SPOT_TIERS,
    CurriculumTier,
)

console = Console()


def run_benchmark(
    model: torch.nn.Module,
    device: str,
    mcts_puzzles: int = 50,
    mcts_sims: int = 50,
) -> dict[str, Any]:
    """Run evaluation on the clean held-out 1,000-puzzle benchmark with Policy + MCTS."""
    evaluator = PuzzleEvaluator(
        model=model,
        device=device,
        puzzle_csv_path=DEFAULT_BENCHMARK_PATH,
    )
    # 1. Raw Policy on full 1,000 benchmark
    policy_res = evaluator.evaluate_policy(batch_size=128)

    # 2. MCTS Search on subset
    mcts_res = None
    if mcts_puzzles > 0:
        mcts_res = evaluator.evaluate_mcts(
            num_simulations=mcts_sims,
            max_puzzles=mcts_puzzles,
        )

    return {
        "policy_rate": policy_res["solve_rate"],
        "policy_solved": policy_res["solved"],
        "policy_total": policy_res["total"],
        "mcts_rate": mcts_res["solve_rate"] if mcts_res else None,
        "mcts_solved": mcts_res["solved"] if mcts_res else None,
        "mcts_total": mcts_res["total"] if mcts_res else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn live from tactical puzzle mistakes (Active Hard Example Mining)."
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to initial model checkpoint (default: latest checkpoint).",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Autonomous curriculum: progress belt-by-belt through difficulty tiers.",
    )
    parser.add_argument(
        "--sweet-spot",
        action="store_true",
        help="Focus on the engine's core weakness band (1100 -> 2100 Elo) instead of full 3000.",
    )
    parser.add_argument(
        "--start-rating",
        type=int,
        default=1000,
        help="Curriculum starting rating (default: 1000).",
    )
    parser.add_argument(
        "--end-rating",
        type=int,
        default=3000,
        help="Curriculum target rating (default: 3000).",
    )
    parser.add_argument(
        "--tier-puzzles",
        type=int,
        default=250,
        help="Puzzles per curriculum tier (default: 250).",
    )
    parser.add_argument(
        "--max-tier-puzzles",
        type=int,
        default=600,
        help="Max puzzles to practice in a tier if struggling before advancing (default: 600).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Multi-pass spaced repetition passes per mistake batch (default: 3).",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.25,
        help="Fraction of cumulative replay memory to mix into each batch (default: 0.25).",
    )
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=0.50,
        help="Fraction of batch reserved for solved/anchor regularizer positions (default: 0.50).",
    )
    parser.add_argument(
        "--puzzles",
        "-p",
        type=str,
        default="data/puzzles/tactics_train_300k.csv",
        help="Path to puzzle CSV for single-set mode (auto-downloads if missing).",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=250,
        help="Number of puzzles in single-set mode (default: 250).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Number of mistakes to accumulate before each gradient step (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Fine-tuning learning rate (default: 2e-5 for gentle regularization).",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1200,
        help="Minimum puzzle rating for single-set mode (default: 1200).",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=1800,
        help="Maximum puzzle rating for single-set mode (default: 1800).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference & training device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models/checkpoints_finetune/mistake_learned_latest.pt",
        help="Destination path for updated model checkpoint.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 1,000-puzzle benchmark (Policy + MCTS) to verify improvement.",
    )
    parser.add_argument(
        "--mcts-puzzles",
        type=int,
        default=50,
        help="Number of benchmark puzzles to test with MCTS search (default: 50, 0 to skip).",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="MCTS simulations per puzzle in benchmark (default: 50).",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    # 1. Resolve starting checkpoint
    ckpt_path = Path(args.weights) if args.weights else get_default_checkpoint_path()
    if not ckpt_path or not ckpt_path.is_file():
        console.print(f"[bold red]Error:[/bold red] Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    title_desc = (
        "Autonomous Sweet-Spot Curriculum (1100 -> 2100 Elo)"
        if args.sweet_spot
        else (
            "Autonomous Full Curriculum (1000 -> 3000 Elo)"
            if args.curriculum
            else "Active Learning on Tactical Puzzles"
        )
    )
    print_banner("ZeroPawn Tactical Mistake Learner", title_desc)

    # 2. Load Model
    console.print(
        f"[cyan]Loading model from:[/cyan] [bold]{ckpt_path.name}[/bold] on [bold]{device}[/bold]..."
    )
    model, info = load_engine_model(ckpt_path, device=device)
    console.print(
        f"[green]✔[/green] Model loaded: {info.get('num_blocks', 12)} blocks, {info.get('num_filters', 128)} filters."
    )

    learner = MistakeLearner(
        model=model,
        lr=args.lr,
        device=device,
        batch_size=args.batch_size,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # MODE A: Autonomous Curriculum Progression
    # ──────────────────────────────────────────────────────────────────────────
    if args.curriculum or args.sweet_spot:
        selected_tiers = SWEET_SPOT_TIERS if args.sweet_spot else DEFAULT_CURRICULUM_TIERS
        start_r = (
            args.start_rating if args.start_rating != 1000 else (1100 if args.sweet_spot else 1000)
        )
        end_r = args.end_rating if args.end_rating != 3000 else (2100 if args.sweet_spot else 3000)

        console.print(
            f"[bold magenta]Curriculum Settings:[/bold magenta] "
            f"Ratings [cyan]{start_r}[/cyan] to [cyan]{end_r}[/cyan] | "
            f"[yellow]{args.tier_puzzles}[/yellow] puzzles/tier | "
            f"[green]{args.epochs} multi-pass epochs[/green] | "
            f"[blue]{args.replay_ratio * 100:.0f}% replay memory[/blue] | "
            f"[magenta]{args.anchor_ratio * 100:.0f}% anchor regularization[/magenta] (LR={args.lr})\n"
        )

        initial_bench = None
        if args.benchmark and DEFAULT_BENCHMARK_PATH.exists():
            console.print(
                "[bold yellow]Running initial baseline on held-out Lichess benchmark (Policy + MCTS)...[/bold yellow]"
            )
            initial_bench = run_benchmark(
                model=model,
                device=device,
                mcts_puzzles=args.mcts_puzzles,
                mcts_sims=args.mcts_sims,
            )
            mcts_str = (
                f" | MCTS: [bold green]{initial_bench['mcts_rate'] * 100:.1f}%[/bold green]"
                if initial_bench["mcts_rate"] is not None
                else ""
            )
            console.print(
                f"[green]✔[/green] Baseline -> Policy: [bold cyan]{initial_bench['policy_rate'] * 100:.2f}%[/bold cyan]{mcts_str}\n"
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
                f"    [magenta]⚡ Gradient Step ({args.epochs} passes):[/magenta] {batch_len} mistakes (anchored {args.anchor_ratio * 100:.0f}%) | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Retest: [green]{resolved}/{batch_len} mastered ({resolved / batch_len * 100:.1f}%)[/green]"
            )

        def on_tier_complete(t: CurriculumTier, res: dict, bench: dict | None):
            status = (
                "[bold green]GRADUATED ✔[/bold green]"
                if res["initial_solve_rate"] >= t.mastery_threshold
                else "[yellow]TIER COMPLETED[/yellow]"
            )
            bench_str = ""
            if bench:
                bench_str = f" | Benchmark Policy: {bench['policy_rate'] * 100:.2f}%"
                if bench.get("mcts_rate") is not None:
                    bench_str += f" | MCTS: {bench['mcts_rate'] * 100:.1f}%"

            console.print(
                f"  {status} -> Solved: [bold cyan]{res['initial_solved']}/{res['total_puzzles']} ({res['initial_solve_rate'] * 100:.1f}%)[/bold cyan] | "
                f"Mistakes Trained: {res['total_mistakes_mined']} | "
                f"Mastered on Retest: [green]{res['mistakes_resolved_on_retest']}[/green]{bench_str}\n"
            )

        bench_cb = None
        if args.benchmark:

            def bench_cb(m):
                return run_benchmark(
                    model=m,
                    device=device,
                    mcts_puzzles=args.mcts_puzzles,
                    mcts_sims=args.mcts_sims,
                )

        t0 = time.time()
        summaries = run_curriculum(
            learner=learner,
            tiers=selected_tiers,
            start_rating=start_r,
            end_rating=end_r,
            base_tier_puzzles=args.tier_puzzles,
            max_tier_puzzles=args.max_tier_puzzles,
            epochs=args.epochs,
            replay_ratio=args.replay_ratio,
            anchor_ratio=args.anchor_ratio,
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
                "start_rating": start_r,
                "end_rating": end_r,
                "timestamp": time.time(),
            },
            final_ckpt,
        )
        console.print(
            f"[bold green]✔ Autonomous Curriculum Complete![/bold green] Final model saved to [bold]{final_ckpt}[/bold]\n"
        )

        # Grand Curriculum Summary Table
        grand_table = Table(title="Autonomous Curriculum Progression Report", box=box.ROUNDED)
        grand_table.add_column("Tier", justify="center", style="bold")
        grand_table.add_column("Rating Band", style="yellow")
        grand_table.add_column("Focus", style="cyan")
        grand_table.add_column("Puzzles", justify="right")
        grand_table.add_column("Solve Rate", justify="right")
        grand_table.add_column("Mistakes Mined", justify="right")
        grand_table.add_column("Mastered Retest", justify="right", style="green")
        grand_table.add_column("Benchmark Policy", justify="right")
        grand_table.add_column("Benchmark MCTS", justify="right")
        grand_table.add_column("Status", justify="center")

        for s in summaries:
            t = s["tier"]
            r = s["results"]
            b = s.get("benchmark") or {}
            p_bench = f"{b['policy_rate'] * 100:.2f}%" if "policy_rate" in b else "-"
            m_bench = f"{b['mcts_rate'] * 100:.1f}%" if b.get("mcts_rate") is not None else "-"
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
                p_bench,
                m_bench,
                grad_text,
            )

        console.print(grand_table)
        console.print(f"[dim]Total Curriculum Runtime: {total_time:.1f}s[/dim]")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # MODE B: Single-Set Mistake Learning
    # ──────────────────────────────────────────────────────────────────────────
    puzzle_path = Path(args.puzzles)
    console.print(f"[cyan]Checking puzzle dataset:[/cyan] [bold]{puzzle_path}[/bold] ...")

    def on_download_progress(current: int, total: int):
        console.print(f"  [dim]Downloaded {current}/{total} puzzles from Lichess...[/dim]")

    puzzle_path = ensure_puzzle_dataset(
        csv_path=puzzle_path,
        target_count=max(args.count, 1000),
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        progress_callback=on_download_progress,
    )

    console.print(
        f"[green]✔[/green] Loading [bold]{args.count}[/bold] puzzles from [bold]{puzzle_path.name}[/bold]..."
    )
    puzzles = load_puzzle_items(puzzle_path, limit=args.count)
    if not puzzles:
        console.print("[bold red]Error:[/bold red] No valid puzzles loaded.")
        sys.exit(1)

    init_bench = None
    if args.benchmark and DEFAULT_BENCHMARK_PATH.exists():
        console.print("\n[bold yellow]Running baseline benchmark (Policy + MCTS)...[/bold yellow]")
        init_bench = run_benchmark(
            model, device, mcts_puzzles=args.mcts_puzzles, mcts_sims=args.mcts_sims
        )
        mcts_str = (
            f" | MCTS: [bold green]{init_bench['mcts_rate'] * 100:.1f}%[/bold green]"
            if init_bench["mcts_rate"] is not None
            else ""
        )
        console.print(
            f"[green]✔[/green] Baseline -> Policy: [bold cyan]{init_bench['policy_rate'] * 100:.2f}%[/bold cyan]{mcts_str}"
        )

    console.print(
        f"\n[bold]Starting live error-driven learning ({args.epochs} passes, {args.replay_ratio * 100:.0f}% replay)...[/bold]"
    )
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
                f"[red]Mistake captured into buffer[/red]",
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
            f"  [bold magenta]⚡ Gradient Step #{updates_count[0]} ({args.epochs} passes):[/bold magenta] "
            f"Trained on {batch_len} mistakes + replay | Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Instant Retest: [bold green]{resolved}/{batch_len} mistakes mastered ({resolved / batch_len * 100:.1f}%)[/bold green]"
        )

    t0 = time.time()
    results = learner.learn_from_puzzle_stream(
        puzzles=puzzles,
        epochs=args.epochs,
        replay_ratio=args.replay_ratio,
        anchor_ratio=args.anchor_ratio,
        on_puzzle_result=on_puzzle_result,
        on_batch_train=on_batch_train,
    )
    elapsed = time.time() - t0

    console.print("\n")
    console.print(table)

    post_bench = None
    if args.benchmark and DEFAULT_BENCHMARK_PATH.exists():
        console.print(
            "\n[bold yellow]Re-running benchmark on held-out Lichess puzzles to verify general transfer...[/bold yellow]"
        )
        post_bench = run_benchmark(
            model, device, mcts_puzzles=args.mcts_puzzles, mcts_sims=args.mcts_sims
        )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model": model.state_dict(),
        "optimizer": learner.optimizer.state_dict() if hasattr(learner, "optimizer") else None,
        "num_blocks": info.get("num_blocks", 12),
        "num_filters": info.get("num_filters", 128),
        "source": "mistake_learner",
        "puzzles_practiced": args.count,
        "mistakes_mined": results["total_mistakes_mined"],
        "timestamp": time.time(),
    }
    torch.save(checkpoint_payload, out_file)
    console.print(f"[green]✔[/green] Updated weights saved to [bold]{out_file}[/bold]")

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
        p_delta = (post_bench["policy_rate"] - init_bench["policy_rate"]) * 100
        p_delta_str = f"+{p_delta:.2f}%" if p_delta >= 0 else f"{p_delta:.2f}%"
        summary_table.add_row(
            "Benchmark Policy Rate (Before -> After)",
            f"{init_bench['policy_rate'] * 100:.2f}% -> [bold green]{post_bench['policy_rate'] * 100:.2f}%[/bold green] ({p_delta_str})",
        )

        if init_bench["mcts_rate"] is not None and post_bench["mcts_rate"] is not None:
            m_delta = (post_bench["mcts_rate"] - init_bench["mcts_rate"]) * 100
            m_delta_str = f"+{m_delta:.1f}%" if m_delta >= 0 else f"{m_delta:.1f}%"
            summary_table.add_row(
                "Benchmark MCTS Solve Rate (Before -> After)",
                f"{init_bench['mcts_rate'] * 100:.1f}% -> [bold green]{post_bench['mcts_rate'] * 100:.1f}%[/bold green] ({m_delta_str})",
            )

    console.print(summary_table)


if __name__ == "__main__":
    main()
