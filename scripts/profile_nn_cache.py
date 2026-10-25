"""Measure NN Cache entries, hit rate, and growth during selfplay games."""

import os
import sys
import time
from pathlib import Path

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

import zeropawn_cpp
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import SelfPlayGame, WorkerBatchAggregator


def measure_nn_cache():
    server = InferenceServer(
        checkpoint_path="models/checkpoints_scaled/best_model.pt", device="cuda"
    )
    client = DirectInferenceClient(server)
    agg = WorkerBatchAggregator(client=client, max_batch_size=1, max_wait_ms=1.0)

    game = SelfPlayGame(
        game_id=1,
        aggregator=agg,
        game_type="chess",
        full_search_sims=400,
        fast_search_sims=80,
        full_search_prob=0.35,
        max_plies=40,
        tablebase_path="data/syzygy",
        tablebase_max_pieces=5,
    )

    eval_calls = [0]
    orig_eval = agg.evaluate_game_state

    def counted_eval(g):
        eval_calls[0] += 1
        return orig_eval(g)

    agg.evaluate_game_state = counted_eval

    # Play 30 plies
    positions, summary = game.play_game()

    cache_size = game.mcts.cache_size()
    total_evals = eval_calls[0]
    total_sims = game.total_simulations

    # Theoretical maximum evaluations without cache = total_sims
    # Cache hits = total_sims - total_evals
    hits = max(0, total_sims - total_evals)
    hit_rate = hits / max(1, total_sims) * 100.0

    bytes_per_entry = (
        (4672 * 4) + 4 + 32
    )  # 18,688 bytes policy + 4 bytes value + map node overhead ~ 18.72 KB
    total_cache_bytes = cache_size * bytes_per_entry

    print("\n--- NN CACHE MEASUREMENTS ---")
    print(f"Total Simulations in Game: {total_sims}")
    print(f"Actual NN Evaluations: {total_evals}")
    print(f"Cache Hits: {hits} ({hit_rate:.1f}%)")
    print(f"Final NN Cache Entry Count: {cache_size}")
    print(f"Bytes Per Entry: {bytes_per_entry} bytes (~{bytes_per_entry / 1024.0:.2f} KB)")
    print(f"Total Cache Memory for Game: {total_cache_bytes / (1024.0 * 1024.0):.2f} MB")


if __name__ == "__main__":
    measure_nn_cache()
