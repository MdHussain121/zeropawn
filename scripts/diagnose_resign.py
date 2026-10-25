"""Diagnostic script for Phase 4 resignation and value head calibration.

Measures un-resigned self-play games under real worker and inference server
infrastructure, recording per-ply root evaluations, search types, and per-game
termination reasons.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_build_cpp = _repo_root / "build" / "cpp"
if _build_cpp.is_dir() and str(_build_cpp) not in sys.path:
    sys.path.insert(0, str(_build_cpp))

_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    try:
        os.add_dll_directory(_winlibs_bin)
    except (AttributeError, OSError):
        pass

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import zeropawn_cpp
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.cli.common import setup_logging
from python.selfplay.worker import (
    GameSummary,
    SavedPosition,
    SelfPlayGame,
    WorkerBatchAggregator,
)

logger = logging.getLogger("diagnose_resign")


def verify_hook_identity(checkpoint_path: str, device: str = "cuda") -> bool:
    """Prove that games with hook ON and hook OFF produce identical moves and outcomes."""
    print("=" * 70)
    print("VERIFYING PASSIVE HOOK IDENTITY (Seeded 3-Game Comparison)")
    print("=" * 70)

    server = InferenceServer(
        checkpoint_path=checkpoint_path,
        device=device,
        max_batch_size=8,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    results_off: List[Dict[str, Any]] = []
    results_on: List[Dict[str, Any]] = []

    # Run with Hook OFF
    aggregator = WorkerBatchAggregator(client=client, max_batch_size=1)
    t1 = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    t1.start()

    for g_idx in range(3):
        seed = 1000 + g_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t_g0 = time.time()
        print(f"  [Hook OFF] Playing game {g_idx + 1}/3 (seed {seed})...", flush=True)
        game = SelfPlayGame(
            game_id=g_idx,
            aggregator=aggregator,
            game_type="chess",
            full_search_sims=400,
            fast_search_sims=80,
            full_search_prob=0.35,
            temp_cutoff_range=(15, 30),
            resign_threshold=-0.95,
            consecutive_resign_moves=4,
            disable_resign_prob=1.0,
            ply_hook=None,
        )
        game.mcts_config.seed = seed
        game.mcts.set_seed(seed)
        _, summary = game.play_game()
        print(
            f"  [Hook OFF] Game {g_idx + 1}/3 finished in {time.time() - t_g0:.1f}s: {summary.length_plies} plies, {summary.result}",
            flush=True,
        )
        results_off.append(
            {
                "moves": summary.moves,
                "result": summary.result,
                "termination": summary.termination_reason,
                "plies": summary.length_plies,
            }
        )
    aggregator.stop()

    # Run with Hook ON
    hook_records: List[Dict[str, Any]] = []

    def test_hook(rec: Dict[str, Any]):
        hook_records.append(rec)

    aggregator2 = WorkerBatchAggregator(client=client, max_batch_size=1)
    t2 = threading.Thread(target=aggregator2.dispatcher_loop, daemon=True)
    t2.start()

    for g_idx in range(3):
        seed = 1000 + g_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t_g0 = time.time()
        print(f"  [Hook ON] Playing game {g_idx + 1}/3 (seed {seed})...", flush=True)
        game = SelfPlayGame(
            game_id=g_idx,
            aggregator=aggregator2,
            game_type="chess",
            full_search_sims=400,
            fast_search_sims=80,
            full_search_prob=0.35,
            temp_cutoff_range=(15, 30),
            resign_threshold=-0.95,
            consecutive_resign_moves=4,
            disable_resign_prob=1.0,
            ply_hook=test_hook,
        )
        game.mcts_config.seed = seed
        game.mcts.set_seed(seed)
        _, summary = game.play_game()
        print(
            f"  [Hook ON] Game {g_idx + 1}/3 finished in {time.time() - t_g0:.1f}s: {summary.length_plies} plies, {summary.result}",
            flush=True,
        )
        results_on.append(
            {
                "moves": summary.moves,
                "result": summary.result,
                "termination": summary.termination_reason,
                "plies": summary.length_plies,
            }
        )
    aggregator2.stop()

    # Compare
    is_identical = True
    for idx in range(3):
        off_res = results_off[idx]
        on_res = results_on[idx]
        match = (off_res["moves"] == on_res["moves"]) and (off_res["result"] == on_res["result"])
        print(
            f"Game {idx + 1}: Plies={off_res['plies']} Result={off_res['result']} Term={off_res['termination']} -> Match={match}"
        )
        if not match:
            is_identical = False
            print(f"  DIFF OFF: {off_res['moves'][:10]}")
            print(f"  DIFF ON:  {on_res['moves'][:10]}")

    print(f"Hook Total Records Captured: {len(hook_records)}")
    print(f"Bitwise Move Identity Check: {'PASS' if is_identical else 'FAIL'}")
    print("=" * 70)
    return is_identical


def run_diagnostic_games(
    checkpoint_path: str,
    num_games: int,
    output_path: str,
    concurrency: int = 6,
    sims_full: int = 400,
    sims_fast: int = 80,
    full_search_prob: float = 0.35,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run diagnostic games with resignation DISABLED and per-ply recording hook.

    Uses a pre-assigned task queue of (game_id, seed) tuples to run until ALL
    pre-assigned games complete without early termination or short-game bias.
    Monitors free physical memory and stops if it drops below 2.0 GB.
    """
    print("=" * 70)
    print(f"RUNNING DIAGNOSTIC GAMES: N={num_games}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Concurrency:      {concurrency}")
    print(f"Simulations:      {sims_full} full / {sims_fast} fast (p={full_search_prob})")
    print(f"Resignation:      DISABLED (disable_resign_prob=1.0)")
    print(f"Output:           {output_path}")
    print("=" * 70)

    # Initial memory check
    mem = psutil.virtual_memory()
    free_ram_gb = mem.available / (1024**3)
    print(f"Initial Free RAM: {free_ram_gb:.2f} GB (Threshold: 2.00 GB)")
    if free_ram_gb < 2.0:
        raise RuntimeError(
            f"Cannot start: Free physical RAM is only {free_ram_gb:.2f} GB (< 2.0 GB threshold)."
        )

    server = InferenceServer(
        checkpoint_path=checkpoint_path,
        device=device,
        max_batch_size=concurrency * 2,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)
    aggregator = WorkerBatchAggregator(
        client=client, max_batch_size=concurrency * 2, max_wait_ms=2.0
    )

    dispatcher_thread = threading.Thread(target=aggregator.dispatcher_loop, daemon=True)
    dispatcher_thread.start()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    game_lock = threading.Lock()
    games_plies: Dict[int, List[Dict[str, Any]]] = {}
    completed_games: List[Dict[str, Any]] = []
    stop_event = threading.Event()
    stop_reason: Optional[str] = None

    def passive_ply_hook(record: Dict[str, Any]):
        gid = record["game_id"]
        # Check RAM safety
        cur_free_gb = psutil.virtual_memory().available / (1024**3)
        if cur_free_gb < 2.0:
            nonlocal stop_reason
            stop_reason = f"CRITICAL STOP TRIGGER: Free physical RAM dropped below 2.0 GB ({cur_free_gb:.2f} GB) on game {gid} ply {record.get('ply')}."
            stop_event.set()
            raise MemoryError(stop_reason)

        with game_lock:
            if gid not in games_plies:
                games_plies[gid] = []
            games_plies[gid].append(record)

    t0 = time.time()

    def game_sink(positions: List[SavedPosition], summary: GameSummary):
        gid = summary.game_id
        with game_lock:
            plies_rec = games_plies.pop(gid, [])
            record = {
                "game_id": gid,
                "length_plies": summary.length_plies,
                "result": summary.result,
                "termination_reason": summary.termination_reason,
                "model_version": summary.model_version,
                "moves": summary.moves,
                "plies": plies_rec,
            }
            completed_games.append(record)
            done_count = len(completed_games)

            # Append directly to JSONL
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            cur_free_ram = psutil.virtual_memory().available / (1024**3)
            elapsed = time.time() - t0
            gps = done_count / max(1e-4, elapsed)
            rem_sec = (num_games - done_count) / max(1e-4, gps)
            print(
                f"[{done_count:>3}/{num_games}] "
                f"Term={summary.termination_reason:<18} "
                f"Result={summary.result:<7} "
                f"Plies={summary.length_plies:<3} | "
                f"Elapsed={elapsed:.1f}s ({gps * 3600:.1f} g/h) | "
                f"FreeRAM={cur_free_ram:.2f}GB | "
                f"ETA={rem_sec / 60:.1f}m",
                flush=True,
            )

    # Pre-assign exact tasks: 100 tuples of (game_id, seed)
    task_queue: queue.Queue[Tuple[int, int]] = queue.Queue()
    for gid in range(num_games):
        seed = 5000 + gid
        task_queue.put((gid, seed))

    def worker_thread_fn():
        while not stop_event.is_set():
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break

            gid, seed = task
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                game = SelfPlayGame(
                    game_id=gid,
                    aggregator=aggregator,
                    game_type="chess",
                    full_search_sims=sims_full,
                    fast_search_sims=sims_fast,
                    full_search_prob=full_search_prob,
                    temp_cutoff_range=(15, 30),
                    resign_threshold=-0.95,
                    consecutive_resign_moves=4,
                    disable_resign_prob=1.0,  # Resignation completely disabled
                    ply_hook=passive_ply_hook,
                )
                game.mcts_config.seed = seed
                game.mcts.set_seed(seed)
                positions, summary = game.play_game()
                game_sink(positions, summary)
            except Exception as e:
                if not stop_event.is_set():
                    logger.exception("Error in game %d: %s", gid, e)
                    nonlocal stop_reason
                    stop_reason = f"Exception in game {gid}: {e}"
                    stop_event.set()
            finally:
                task_queue.task_done()

    threads = []
    for _ in range(concurrency):
        t = threading.Thread(target=worker_thread_fn, daemon=True)
        t.start()
        threads.append(t)

    # Monitor until completion or stop trigger
    while not stop_event.is_set():
        with game_lock:
            if len(completed_games) >= num_games:
                break
        time.sleep(0.5)

    if stop_event.is_set():
        print(f"\n[ABORTED] {stop_reason}", flush=True)
        # Drain remaining tasks
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
                task_queue.task_done()
            except Exception:
                break

    task_queue.join()
    aggregator.stop()
    for t in threads:
        t.join(timeout=2.0)

    total_time = time.time() - t0
    print("=" * 70)
    print(
        f"DIAGNOSTIC RUN FINISHED: {len(completed_games)}/{num_games} games in {total_time:.1f}s ({total_time / 60:.2f}m)"
    )
    print(f"Saved records to {out_file.resolve()}")
    print("=" * 70)

    if stop_reason:
        raise RuntimeError(stop_reason)

    return {
        "num_games": len(completed_games),
        "total_time_sec": total_time,
        "output_path": str(out_file),
    }


def main():
    # Single logging setup: rotating file for everything, WARNING+ on stderr.
    setup_logging(log_dir="logs", log_filename="diagnose_resign.log")
    parser = argparse.ArgumentParser(description="Resignation Diagnostic Runner")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints_scaled/best_model.pt")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--sims-full", type=int, default=400)
    parser.add_argument("--sims-fast", type=int, default=80)
    parser.add_argument("--full-search-prob", type=float, default=0.35)
    parser.add_argument("--output", type=str, default="docs/resignfix/diagnosis_games.jsonl")
    parser.add_argument(
        "--verify-hook", action="store_true", help="Run 3-game hook on/off identity test"
    )
    parser.add_argument("--benchmark", type=int, default=0, help="Benchmark N games and report ETA")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.verify_hook:
        ok = verify_hook_identity(args.checkpoint, device=args.device)
        sys.exit(0 if ok else 1)

    if args.benchmark > 0:
        print(
            f"Benchmarking {args.benchmark} un-resigned games at concurrency {args.concurrency}..."
        )
        res = run_diagnostic_games(
            checkpoint_path=args.checkpoint,
            num_games=args.benchmark,
            output_path="docs/resignfix/benchmark_temp.jsonl",
            concurrency=args.concurrency,
            sims_full=args.sims_full,
            sims_fast=args.sims_fast,
            full_search_prob=args.full_search_prob,
            device=args.device,
        )
        t_elapsed = res["total_time_sec"]
        est_100 = (t_elapsed / args.benchmark) * 100.0
        print(
            f"\nBENCHMARK RESULT: {args.benchmark} games took {t_elapsed:.1f}s ({t_elapsed / args.benchmark:.1f}s/game, {args.benchmark / t_elapsed * 3600:.1f} games/h)"
        )
        print(f"ESTIMATED TIME FOR 100 GAMES: {est_100:.1f}s (~{est_100 / 60:.1f} minutes)")
        p_tmp = Path("docs/resignfix/benchmark_temp.jsonl")
        if p_tmp.exists():
            p_tmp.unlink()
        sys.exit(0)

    run_diagnostic_games(
        checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        output_path=args.output,
        concurrency=args.concurrency,
        sims_full=args.sims_full,
        sims_fast=args.sims_fast,
        full_search_prob=args.full_search_prob,
        device=args.device,
    )


if __name__ == "__main__":
    main()
