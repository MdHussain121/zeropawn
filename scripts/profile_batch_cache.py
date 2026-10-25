"""Measure batch size distribution and flush trigger reasons (capacity vs timeout)."""

import os
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

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
from python.selfplay.worker import SelfPlayWorker


def run_flush_profile(concurrency: int = 6, duration_sec: float = 20.0):
    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt",
        device="cuda",
        min_batch_size=min(48, concurrency * 2),
        max_batch_size=128,
        max_wait_ms=1.5,
        use_bfloat16=True,
    )
    client = DirectInferenceClient(server)

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
    )
    agg = worker.aggregator

    # Tracking
    flush_source = {"capacity": 0, "timeout": 0}
    batch_sizes = []

    # Intercept flush calls
    orig_flush = agg._flush_locked

    def flush_wrapper(source):
        with agg.lock:
            sz = len(agg.pending)
        if sz > 0:
            batch_sizes.append(sz)
            flush_source[source] += 1
        orig_flush()

    # Intercept evaluate_game_state
    orig_eval = agg.evaluate_game_state

    def eval_wrapper(game):
        t0 = time.perf_counter()
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
            raise TimeoutError("Timed out")
        return res_box[0], res_box[1]

    # Intercept dispatcher_loop
    orig_dispatcher = agg.dispatcher_loop

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

    worker.start()
    time.sleep(duration_sec)
    worker.stop()

    total_flushes = sum(flush_source.values())
    print("\n--- BATCH-SIZE DISTRIBUTION & FLUSH TRIGGER REASONS ---")
    print(f"Total Batches Recorded: {total_flushes}")
    print(
        f"Flushed by Capacity (>= {agg.max_batch_size}): {flush_source['capacity']} ({flush_source['capacity'] / max(1, total_flushes) * 100:.1f}%)"
    )
    print(
        f"Flushed by 1.5ms Timeout: {flush_source['timeout']} ({flush_source['timeout'] / max(1, total_flushes) * 100:.1f}%)"
    )
    if batch_sizes:
        print(f"Batch Size Distribution (Counter): {dict(Counter(batch_sizes))}")
        print(f"Mean Batch Size: {np.mean(batch_sizes):.2f}")
        print(f"Median Batch Size: {np.median(batch_sizes):.1f}")
        print(f"Min / Max Batch Size: {np.min(batch_sizes)} / {np.max(batch_sizes)}")


if __name__ == "__main__":
    run_flush_profile()
