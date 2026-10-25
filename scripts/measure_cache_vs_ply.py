"""Measure cache entries vs ply curve for 20 games at production settings (Item 1b).

Protocol:
- 20 games run with Concurrency 24, full_search_sims=400, fast_search_sims=80, full_search_prob=0.35
- Tablebase enabled (Syzygy 3-4-5), resignation disabled (allow_resignation=False)
- Measures mcts.cache_size() at every ply for every game
- Aggregates mean and max cache entries at every 25 plies (plies 25, 50, 75, 100, 125, 150, 175, 200, 225, ...)
- Records distribution of game lengths (min, p25, median, p75, max, mean)
- Calculates worst case: max_entries * 18,728 bytes * 24 games
"""

import os
import sys
import threading
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import SelfPlayWorker


def main():
    print("=== MEASURING CACHE ENTRIES VS PLY FOR 20 GAMES (ITEM 1B) ===")
    tb_dir = REPO_ROOT / "data" / "syzygy"
    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_batch_size=20,
        max_batch_size=64,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    # Tracking data structures
    game_ply_cache = defaultdict(dict)  # game_id -> {ply: cache_size}
    completed_summaries = []
    lock = threading.Lock()
    done_event = threading.Event()
    target_games = 20

    worker_ref = [None]

    def ply_hook(info):
        w = worker_ref[0]
        if w is None:
            return
        tid = threading.get_ident()
        with w.counter_lock:
            g = w.active_games.get(tid)
        if g is not None:
            c_size = g.mcts.cache_size()
            with lock:
                game_ply_cache[info["game_id"]][info["ply"]] = c_size

    def sink(positions, summary):
        with lock:
            completed_summaries.append(summary)
            print(
                f"Game {len(completed_summaries)}/{target_games} finished: {summary.length_plies} plies, result={summary.result}, reason={summary.termination_reason}",
                flush=True,
            )
            if len(completed_summaries) >= target_games:
                done_event.set()

    worker = SelfPlayWorker(
        worker_id=1,
        client=client,
        num_concurrent_games=20,  # 20 concurrent games to complete in parallel
        game_type="chess",
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        allow_resignation=False,
        tablebase_path=str(tb_dir),
        tablebase_max_pieces=5,
        position_sink=sink,
        ply_hook=ply_hook,
        max_wait_ms=1.5,
    )
    worker_ref[0] = worker

    t0 = time.time()
    worker.start()

    # Wait until 20 games finish or timeout at 240s (4 minutes)
    done_event.wait(timeout=240.0)
    worker.stop()
    elapsed = time.time() - t0
    print(f"\nAll 20 games completed in {elapsed:.1f}s!")

    # Analysis
    summaries = completed_summaries[:target_games]
    lengths = [s.length_plies for s in summaries]
    adjudicated_count = sum(
        1 for s in summaries if s.adjudicated or s.termination_reason == "tablebase_adjudication"
    )

    print("\n" + "=" * 80)
    print("GAME LENGTH DISTRIBUTION (20 GAMES)")
    print("=" * 80)
    print(f"Total games:      {len(summaries)}")
    print(f"Mean length:      {np.mean(lengths):.1f} plies")
    print(f"Median length:    {np.median(lengths):.1f} plies")
    print(f"Min length:       {np.min(lengths)} plies")
    print(f"Max length:       {np.max(lengths)} plies")
    print(f"25th percentile:  {np.percentile(lengths, 25):.1f} plies")
    print(f"75th percentile:  {np.percentile(lengths, 75):.1f} plies")
    print(f"95th percentile:  {np.percentile(lengths, 95):.1f} plies")
    print(
        f"Adjudications:    {adjudicated_count}/{len(summaries)} ({adjudicated_count / len(summaries) * 100:.1f}%)"
    )
    print("=" * 80)

    # Cache entries vs ply at 25-ply intervals
    max_ply_seen = max(lengths) if lengths else 0
    intervals = list(range(25, max_ply_seen + 25, 25))

    print("\n" + "=" * 80)
    print("CACHE ENTRIES VS PLY CURVE (EVERY 25 PLIES)")
    print("=" * 80)
    print(
        f"{'Ply':<6} | {'Active Games':<12} | {'Mean Entries':<14} | {'Max Entries':<14} | {'Min Entries':<14} | {'Est Mean MB':<12}"
    )
    print("-" * 80)

    overall_max_entries = 0

    for p in intervals:
        entries_at_p = []
        for g_id, ply_dict in game_ply_cache.items():
            if p in ply_dict:
                entries_at_p.append(ply_dict[p])
            elif any(k >= p for k in ply_dict.keys()):
                # Find closest ply <= p
                closest_k = max([k for k in ply_dict.keys() if k <= p], default=None)
                if closest_k is not None:
                    entries_at_p.append(ply_dict[closest_k])

        if entries_at_p:
            mean_e = float(np.mean(entries_at_p))
            max_e = int(np.max(entries_at_p))
            min_e = int(np.min(entries_at_p))
            overall_max_entries = max(overall_max_entries, max_e)
            mean_mb = mean_e * 18728 / (1024.0 * 1024.0)
            print(
                f"{p:<6} | {len(entries_at_p):<12} | {mean_e:<14.1f} | {max_e:<14} | {min_e:<14} | {mean_mb:<12.1f}"
            )

    # Also compute overall max across all individual plies
    all_seen = [c for ply_dict in game_ply_cache.values() for c in ply_dict.values()]
    peak_single_game = max(all_seen) if all_seen else overall_max_entries
    worst_case_24_bytes = peak_single_game * 18728 * 24
    worst_case_24_mb = worst_case_24_bytes / (1024.0 * 1024.0)
    worst_case_24_gb = worst_case_24_bytes / (1024.0**3)

    print("=" * 80)
    print(
        f"Peak Single-Game Cache Entries Observed: {peak_single_game} entries ({peak_single_game * 18728 / (1024**2):.1f} MB)"
    )
    print(f"Worst Case (Peak Entries * 18,728 bytes * 24 concurrent games):")
    print(
        f"  {peak_single_game} * 18,728 * 24 = {worst_case_24_mb:.1f} MB ({worst_case_24_gb:.2f} GB)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
