"""Task 3(a): Concurrency sweep over {6, 8, 12, 16, 24} workers.
Protocol: 2 minutes warm-up (uncounted), 5 minutes counted per setting.
Safety: Aborts and halts sweep if free RAM drops below 2.0 GB.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_build_dir = _repo_root / "build" / "cpp"
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))
_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_winlibs_bin)
    except OSError:
        pass

from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import SelfPlayWorker, GameSummary, SavedPosition


def get_system_memory() -> Dict[str, float]:
    mem = psutil.virtual_memory()
    free_ram_gb = mem.available / (1024.0**3)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
    return {
        "free_ram_gb": free_ram_gb,
        "vram_mb": vram_mb,
    }


def run_single_concurrency_bench(
    concurrency: int,
    warmup_sec: float = 120.0,
    counted_sec: float = 300.0,
) -> Dict[str, Any]:
    # Check free RAM safety floor before starting
    initial_mem = get_system_memory()
    initial_free_ram = initial_mem["free_ram_gb"]
    if initial_free_ram < 2.0:
        print(
            f"[CONCURRENCY {concurrency}] ABORTED: Free RAM ({initial_free_ram:.2f} GB) < 2.0 GB safety floor."
        )
        return {"concurrency": concurrency, "aborted": True, "reason": "free_ram < 2.0 GB"}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_batch_size=min(48, concurrency * 2),
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

    completed_summaries: List[GameSummary] = []
    counted_completed_summaries: List[GameSummary] = []
    total_positions_generated: List[int] = [0]
    counted_positions_generated: List[int] = [0]
    counted_model_lags: List[int] = []

    is_in_counted_phase = [False]

    def sink(positions: List[SavedPosition], summary: GameSummary):
        completed_summaries.append(summary)
        total_positions_generated[0] += len(positions)
        if is_in_counted_phase[0]:
            counted_completed_summaries.append(summary)
            counted_positions_generated[0] += len(positions)
            current_ver = server.model_version
            for p in positions:
                counted_model_lags.append(max(0, current_ver - p.model_version))

    worker = SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=concurrency,
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        allow_resignation=False,
        tablebase_path="data/syzygy",
        tablebase_max_pieces=5,
        position_sink=sink,
    )

    # Instrument aggregator to track capacity vs timeout flushes
    agg = worker.aggregator
    flush_counts = {"capacity": 0, "timeout": 0}
    orig_flush = agg._flush_locked

    def flush_wrapper(source: str):
        if is_in_counted_phase[0]:
            with agg.lock:
                sz = len(agg.pending)
            if sz > 0:
                flush_counts[source] += 1
        orig_flush()

    orig_eval = agg.evaluate_game_state

    def eval_wrapper(game):
        if hasattr(game, "num_planes") and game.num_planes() == 3:
            state = game.encode_state()
        else:
            import zeropawn_cpp

            state = zeropawn_cpp.encode_state(game)

        res_box = [None, None]
        import threading

        evt = threading.Event()

        with agg._req_lock:
            with agg.lock:
                agg.pending.append((state, res_box, evt))
                need_flush = len(agg.pending) >= agg.max_batch_size
                if not need_flush:
                    agg.cond.notify()
            if need_flush:
                flush_wrapper("capacity")

        if not evt.wait(timeout=60):
            raise TimeoutError("Timed out waiting for batched inference")
        return res_box[0], res_box[1]

    def dispatcher_wrapper():
        while agg.running:
            with agg.lock:
                if not agg.pending:
                    agg.cond.wait(timeout=agg.max_wait_sec)
                needs_flush = bool(agg.pending)
            if needs_flush:
                try:
                    flush_wrapper("timeout")
                except Exception:
                    pass

    agg.evaluate_game_state = eval_wrapper
    agg.dispatcher_loop = dispatcher_wrapper

    print(f"\n{'=' * 75}")
    print(
        f"Starting Concurrency {concurrency} (Warm-up: {warmup_sec:.0f}s, Counted: {counted_sec:.0f}s)"
    )
    print(f"Initial Free RAM: {initial_free_ram:.2f} GB | Min Batch Size: {server.min_batch_size}")
    print(f"{'=' * 75}")

    min_free_ram = initial_free_ram
    worker.start()
    t_start = time.time()
    last_sims = 0
    t_last = t_start

    sims_at_warmup_end = 0
    counted_start_time = 0.0
    breached_ram = False

    try:
        # Phase 1: Warm-up
        while (time.time() - t_start) < warmup_sec:
            time.sleep(10.0)
            now = time.time()
            elapsed = now - t_start
            dt = now - t_last
            t_last = now

            cur_sims = worker.live_simulations
            sims_delta = cur_sims - last_sims
            last_sims = cur_sims
            current_rate = sims_delta / max(0.001, dt)

            mem = get_system_memory()
            if mem["free_ram_gb"] < min_free_ram:
                min_free_ram = mem["free_ram_gb"]

            if mem["free_ram_gb"] < 2.0:
                print(
                    f"[CONCURRENCY {concurrency}] EMERGENCY STOP: Free RAM dropped to {mem['free_ram_gb']:.2f} GB (< 2.0 GB) during warm-up!"
                )
                breached_ram = True
                break

            avg_bs, avg_lat, fill_rt = server.get_window_stats()
            print(
                f"[WARMUP {elapsed:4.0f}s / {warmup_sec:.0f}s] Sims/s: {current_rate:7.1f} | Avg Batch: {avg_bs:4.1f} | Latency: {avg_lat:5.2f}ms | Free RAM: {mem['free_ram_gb']:4.2f}GB | Peak VRAM: {mem['vram_mb']:5.0f}MB"
            )

        if not breached_ram:
            # Transition to Counted Phase
            print(
                f"\n>>> Warm-up complete. Starting {counted_sec:.0f}s COUNTED phase for Concurrency {concurrency} <<<\n"
            )
            is_in_counted_phase[0] = True
            sims_at_warmup_end = worker.live_simulations
            counted_start_time = time.time()
            server.get_window_stats()  # Reset server window stats
            last_sims = sims_at_warmup_end
            t_last = counted_start_time

            # Phase 2: Counted Phase
            while (time.time() - counted_start_time) < counted_sec:
                time.sleep(10.0)
                now = time.time()
                elapsed = now - counted_start_time
                dt = now - t_last
                t_last = now

                cur_sims = worker.live_simulations
                sims_delta = cur_sims - last_sims
                last_sims = cur_sims
                current_rate = sims_delta / max(0.001, dt)

                mem = get_system_memory()
                if mem["free_ram_gb"] < min_free_ram:
                    min_free_ram = mem["free_ram_gb"]

                if mem["free_ram_gb"] < 2.0:
                    print(
                        f"[CONCURRENCY {concurrency}] EMERGENCY STOP: Free RAM dropped to {mem['free_ram_gb']:.2f} GB (< 2.0 GB) during counted phase!"
                    )
                    breached_ram = True
                    break

                avg_bs, avg_lat, fill_rt = server.get_window_stats()
                print(
                    f"[COUNTED {elapsed:4.0f}s / {counted_sec:.0f}s] Sims/s: {current_rate:7.1f} | Avg Batch: {avg_bs:4.1f} | Latency: {avg_lat:5.2f}ms | Free RAM: {mem['free_ram_gb']:4.2f}GB | Peak VRAM: {mem['vram_mb']:5.0f}MB"
                )

    finally:
        worker.stop()

    if breached_ram:
        return {
            "concurrency": concurrency,
            "aborted": True,
            "reason": f"free_ram dropped to {min_free_ram:.2f} GB (< 2.0 GB)",
            "min_free_ram_gb": min_free_ram,
            "max_ram_drop_gb": initial_free_ram - min_free_ram,
        }

    counted_elapsed = time.time() - counted_start_time
    total_counted_sims = worker.live_simulations - sims_at_warmup_end
    sims_per_sec = total_counted_sims / counted_elapsed
    pos_count = counted_positions_generated[0]
    pos_per_hour = pos_count * (3600.0 / counted_elapsed)
    games_count = len(counted_completed_summaries)
    games_per_hour = games_count * (3600.0 / counted_elapsed)

    # Projected training steps per hour at --max-reuse-ratio 8
    # pos_per_hour * 8 samples / 256 batch_size = pos_per_hour / 32
    proj_train_steps_per_hour = (pos_per_hour * 8.0) / 256.0

    durations = [s.duration_sec for s in counted_completed_summaries]
    avg_game_time = float(np.mean(durations)) if durations else 0.0

    avg_lag = float(np.mean(counted_model_lags)) if counted_model_lags else 0.0

    avg_batch_sz, avg_lat_ms, fill_rate = server.get_window_stats()

    total_flushes = sum(flush_counts.values())
    timeout_flush_pct = (flush_counts["timeout"] / max(1, total_flushes)) * 100.0

    max_ram_drop = max(0.0, initial_free_ram - min_free_ram)
    final_mem = get_system_memory()

    result = {
        "concurrency": concurrency,
        "aborted": False,
        "counted_sec": counted_elapsed,
        "sims_per_sec": sims_per_sec,
        "positions_per_hour": pos_per_hour,
        "games_per_hour": games_per_hour,
        "completed_games": games_count,
        "proj_train_steps_hr": proj_train_steps_per_hour,
        "avg_game_duration_sec": avg_game_time,
        "avg_batch_size": avg_batch_sz,
        "avg_latency_ms": avg_lat_ms,
        "fill_rate": fill_rate,
        "timeout_flush_pct": timeout_flush_pct,
        "total_flushes": total_flushes,
        "initial_free_ram_gb": initial_free_ram,
        "min_free_ram_gb": min_free_ram,
        "max_ram_drop_gb": max_ram_drop,
        "peak_vram_mb": final_mem["vram_mb"],
        "avg_model_version_lag": avg_lag,
    }

    print(f"\n--- Concurrency {concurrency} Counted Phase Results ---")
    print(f"Simulations/sec:               {sims_per_sec:.1f}")
    print(f"Positions/hour:                {pos_per_hour:.1f}")
    print(f"Games/hour:                    {games_per_hour:.1f} (Counted Games: {games_count})")
    print(
        f"Projected Train Steps/hour:    {proj_train_steps_per_hour:.1f} (under --max-reuse-ratio 8)"
    )
    print(f"Avg Game Wall-Clock Time:      {avg_game_time:.1f}s")
    print(f"Avg Batch Size:                {avg_batch_sz:.2f} (Fill Rate: {fill_rate * 100:.1f}%)")
    print(f"Avg Inference Latency:         {avg_lat_ms:.2f} ms")
    print(
        f"1.5ms Timeout Flush %:         {timeout_flush_pct:.1f}% ({flush_counts['timeout']}/{total_flushes})"
    )
    print(
        f"Max Free RAM Drop:             {max_ram_drop:.2f} GB (Min Free RAM: {min_free_ram:.2f} GB)"
    )
    print(f"Peak VRAM:                     {final_mem['vram_mb']:.1f} MB")
    print(f"Avg Model-Version Lag:         {avg_lag:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Sweep selfplay concurrency levels")
    parser.add_argument(
        "--warmup", type=float, default=120.0, help="Warm-up seconds (default: 120s = 2m)"
    )
    parser.add_argument(
        "--counted", type=float, default=300.0, help="Counted seconds (default: 300s = 5m)"
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[6, 8, 12, 16, 24],
        help="Concurrencies to sweep",
    )
    args = parser.parse_args()

    results = []
    for c in args.concurrencies:
        res = run_single_concurrency_bench(
            concurrency=c,
            warmup_sec=args.warmup,
            counted_sec=args.counted,
        )
        results.append(res)
        # Save to CSV immediately
        csv_path = Path("docs/perf/sweep_concurrency.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        file_exists = csv_path.is_file()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "concurrency",
                    "aborted",
                    "counted_sec",
                    "sims_per_sec",
                    "positions_per_hour",
                    "games_per_hour",
                    "completed_games",
                    "proj_train_steps_hr",
                    "avg_game_duration_sec",
                    "avg_batch_size",
                    "avg_latency_ms",
                    "fill_rate",
                    "timeout_flush_pct",
                    "total_flushes",
                    "initial_free_ram_gb",
                    "min_free_ram_gb",
                    "max_ram_drop_gb",
                    "peak_vram_mb",
                    "avg_model_version_lag",
                ],
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(res)

        if res.get("aborted") or res.get("min_free_ram_gb", 10.0) < 2.0:
            print(f"HALTING SWEEP: Concurrency {c} breached RAM safety threshold (< 2.0 GB).")
            break

    print("\n" + "=" * 115)
    print("TASK 3(a) CONCURRENCY SWEEP SUMMARY TABLE")
    print("=" * 115)
    header = (
        f"{'Conc':>5} | {'Sims/s':>9} | {'Pos/hr':>10} | {'Games/hr':>9} | {'ProjTrain/h':>11} | "
        f"{'BatchSz':>8} | {'Timeout%':>9} | {'GameTime(s)':>11} | {'RAM Drop(GB)':>12} | {'MinRAM(GB)':>10} | {'ModelLag':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("aborted"):
            print(f"{r['concurrency']:>5} | ABORTED: {r.get('reason')}")
            continue
        print(
            f"{r['concurrency']:>5} | {r['sims_per_sec']:>9.1f} | {r['positions_per_hour']:>10.1f} | "
            f"{r['games_per_hour']:>9.1f} | {r['proj_train_steps_hr']:>11.1f} | {r['avg_batch_size']:>8.2f} | "
            f"{r['timeout_flush_pct']:>8.1f}% | {r['avg_game_duration_sec']:>11.1f} | {r['max_ram_drop_gb']:>12.2f} | "
            f"{r['min_free_ram_gb']:>10.2f} | {r['avg_model_version_lag']:>9.2f}"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
