"""Profile full combined selfplay + training under cProfile to get exact wall-clock breakdown."""

import cProfile
import pstats
import io
import time
from pathlib import Path
import sys
import os

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

from scripts.bench_selfplay import BenchmarkRunner


def main():
    runner = BenchmarkRunner(
        mode="combined",
        warmup_sec=5.0,
        counted_sec=25.0,
        sample_interval=5.0,
        trials=1,
        concurrency=6,
        full_sims=400,
        fast_sims=80,
        output_dir="docs/perf/profile_temp",
    )

    pr = cProfile.Profile()
    pr.enable()
    runner.run_trial(1)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)

    out_text = s.getvalue()
    with open("docs/perf/cprofile_combined.txt", "w", encoding="utf-8") as f:
        f.write(out_text)
    print("\n--- TOP 25 CUMULATIVE TIME IN COMBINED RUN ---")
    lines = out_text.splitlines()[:35]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
