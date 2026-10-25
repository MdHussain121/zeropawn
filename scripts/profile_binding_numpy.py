import os
import sys
import time
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

import zeropawn_cpp


def bench_binding():
    game = zeropawn_cpp.ChessGame()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    # 1. Test with Python list
    p_list = [1.0 / 4672.0] * 4672

    def eval_list(g):
        return (p_list, 0.0)

    # 2. Test with NumPy float32 array
    p_arr_f32 = np.full(4672, 1.0 / 4672.0, dtype=np.float32)

    def eval_f32(g):
        return (p_arr_f32, 0.0)

    # 3. Test with NumPy float64 array
    p_arr_f64 = np.full(4672, 1.0 / 4672.0, dtype=np.float64)

    def eval_f64(g):
        return (p_arr_f64, 0.0)

    N_SIMS = 100

    # Warm-up
    mcts.clear()
    res1 = mcts.search(game, 10, eval_list)
    mcts.clear()
    res2 = mcts.search(game, 10, eval_f32)
    mcts.clear()
    res3 = mcts.search(game, 10, eval_f64)

    # Verification: do they give identical visit counts?
    v1 = res1.action_visits
    v2 = res2.action_visits
    v3 = res3.action_visits
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(v1, v3)
    print(
        "Correctness check: Python list, NumPy float32, and NumPy float64 return IDENTICAL visit counts."
    )

    # Time list
    TRIALS = 20
    times_list = []
    for _ in range(TRIALS):
        mcts.clear()
        t0 = time.perf_counter()
        mcts.search(game, N_SIMS, eval_list)
        times_list.append(time.perf_counter() - t0)

    # Time numpy float32
    times_f32 = []
    for _ in range(TRIALS):
        mcts.clear()
        t0 = time.perf_counter()
        mcts.search(game, N_SIMS, eval_f32)
        times_f32.append(time.perf_counter() - t0)

    # Time numpy float64
    times_f64 = []
    for _ in range(TRIALS):
        mcts.clear()
        t0 = time.perf_counter()
        mcts.search(game, N_SIMS, eval_f64)
        times_f64.append(time.perf_counter() - t0)

    # Also benchmark the pure eval conversion time alone (without MCTS)
    # by measuring pybind11 casting via calling search with 1 sim many times
    print(f"\n--- 100-Simulation MCTS Search Timing ({TRIALS} trials, stub evaluator) ---")
    print(
        f"Python list (4672 floats):       {np.mean(times_list) * 1000:.2f} ms ± {np.std(times_list) * 1000:.2f} ms ({np.mean(times_list) / N_SIMS * 1e6:.1f} µs/sim)"
    )
    print(
        f"NumPy float32 (4672 floats):     {np.mean(times_f32) * 1000:.2f} ms ± {np.std(times_f32) * 1000:.2f} ms ({np.mean(times_f32) / N_SIMS * 1e6:.1f} µs/sim)"
    )
    print(
        f"NumPy float64 (4672 floats):     {np.mean(times_f64) * 1000:.2f} ms ± {np.std(times_f64) * 1000:.2f} ms ({np.mean(times_f64) / N_SIMS * 1e6:.1f} µs/sim)"
    )


if __name__ == "__main__":
    bench_binding()
