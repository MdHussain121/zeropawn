"""Stockfish value target generation and blending.

Calculates Stockfish evaluations at a fixed node limit, converts centipawns
to a win/draw/loss distribution using a smooth logistic model, and blends 50/50
with the recorded game outcome.

Logistic Formula:
  P_win_raw  = 1.0 / (1.0 + exp(-cp / 200.0))
  P_loss_raw = 1.0 / (1.0 + exp(+cp / 200.0))
  P_draw_raw = 0.5 * exp(- (cp / 150.0)^2)
  S = P_win_raw + P_draw_raw + P_loss_raw
  W = P_win_raw / S,  D = P_draw_raw / S,  L = P_loss_raw / S

Blended Target:
  y_blended = 0.5 * [W, D, L]_sf + 0.5 * [W, D, L]_game
"""

from __future__ import annotations

import argparse
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

DEFAULT_STOCKFISH_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tools"
    / "stockfish"
    / "stockfish-windows-x86-64-avx2.exe"
)


def centipawns_to_wdl(cp: float) -> tuple[float, float, float]:
    """Convert centipawn evaluation (from side to move) to WDL probabilities."""
    # Clamp extreme evaluations
    cp_clamped = max(-2000.0, min(2000.0, float(cp)))

    p_win_raw = 1.0 / (1.0 + math.exp(-cp_clamped / 200.0))
    p_loss_raw = 1.0 / (1.0 + math.exp(+cp_clamped / 200.0))
    p_draw_raw = 0.5 * math.exp(-((cp_clamped / 150.0) ** 2))

    total = p_win_raw + p_draw_raw + p_loss_raw
    w = p_win_raw / total
    d = p_draw_raw / total
    l = p_loss_raw / total
    return (w, d, l)


def outcome_to_one_hot(outcome: int) -> np.ndarray:
    """Convert integer outcome (0=Win, 1=Draw, 2=Loss) to 3-class probability array."""
    arr = np.zeros(3, dtype=np.float32)
    if 0 <= outcome <= 2:
        arr[outcome] = 1.0
    return arr


def blend_targets(sf_wdl: tuple[float, float, float] | np.ndarray, outcome: int) -> np.ndarray:
    """Blend 50/50 Stockfish WDL target with game outcome."""
    sf_arr = np.array(sf_wdl, dtype=np.float32)
    game_arr = outcome_to_one_hot(outcome)
    return 0.5 * sf_arr + 0.5 * game_arr


class StockfishEvaluator:
    """Persistent Stockfish UCI process evaluator with non-blocking readline timeouts."""

    def __init__(self, stockfish_path: Path | str, threads: int = 4, hash_mb: int = 64) -> None:
        self.path = Path(stockfish_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Stockfish binary not found at {self.path}")

        self._line_queue: queue.Queue[Optional[str]] = queue.Queue()
        self.proc = subprocess.Popen(
            [str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self._reader_thread.start()

        self._send_cmd("uci")
        self._wait_for("uciok")
        self._send_cmd(f"setoption name Threads value {threads}")
        self._send_cmd(f"setoption name Hash value {hash_mb}")
        self._send_cmd("isready")
        self._wait_for("readyok")

    def _reader_worker(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        try:
            for line in iter(self.proc.stdout.readline, ""):
                self._line_queue.put(line.strip())
        except (OSError, ValueError):
            pass
        self._line_queue.put(None)

    def _readline(self, timeout: float = 10.0) -> Optional[str]:
        if not self.proc:
            return None
        try:
            return self._line_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush_queue(self, max_drain: int = 200) -> None:
        """Drain leftover lines from the reader queue to prevent stale output poisoning."""
        if not self.proc:
            return
        drained = 0
        while drained < max_drain:
            try:
                self._line_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break

    def _send_cmd(self, cmd: str) -> None:
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except (OSError, BrokenPipeError):
                pass

    def _wait_for(self, target: str, timeout: float = 10.0) -> list[str]:
        lines = []
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            line = self._readline(timeout=max(0.1, timeout - (time.monotonic() - start)))
            if line is None:
                break
            lines.append(line)
            if target in line:
                break
        return lines

    def evaluate_fen(self, fen: str, nodes: int = 10000, timeout: float = 30.0) -> float:
        """Evaluate a FEN position at fixed node count; returns centipawns."""
        self._send_cmd(f"position fen {fen}")
        self._send_cmd(f"go nodes {nodes}")

        last_cp = 0.0
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            line_str = self._readline(timeout=max(0.1, timeout - (time.monotonic() - start)))
            if line_str is None:
                break
            if "score cp" in line_str:
                parts = line_str.split()
                try:
                    idx = parts.index("cp")
                    last_cp = float(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            elif "score mate" in line_str:
                parts = line_str.split()
                try:
                    idx = parts.index("mate")
                    mate_dist = int(parts[idx + 1])
                    last_cp = 10000.0 if mate_dist > 0 else -10000.0
                except (ValueError, IndexError):
                    pass
            if line_str.startswith("bestmove"):
                break

        return last_cp

    def close(self) -> None:
        try:
            self._send_cmd("quit")
            self.proc.terminate()
            self.proc.wait(timeout=2.0)
        except (subprocess.TimeoutExpired, OSError, BrokenPipeError):
            try:
                self.proc.kill()
            except (subprocess.SubprocessError, OSError):
                pass
        except Exception:
            pass

    def __enter__(self) -> StockfishEvaluator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def generate_value_targets_for_dataset(
    data_dir: Path | str,
    stockfish_path: Path | str = DEFAULT_STOCKFISH_PATH,
    subset_size: int = 50000,
    node_limit: int = 10000,
    threads: int = 8,
) -> None:
    """Generate Stockfish WDL targets for a subset of the dataset and save value_targets.bin."""
    from python.data.dataset import BinaryChessDataset
    from python.data.fen_utils import decode_planes_to_fen

    dir_path = Path(data_dir)
    ds = BinaryChessDataset(dir_path)
    num_samples = len(ds)
    target_subset = min(subset_size, num_samples)

    print("=" * 60)
    print("STOCKFISH VALUE TARGET GENERATION")
    print(f"Dataset directory: {dir_path} ({num_samples} total samples)")
    print(f"Subset size:       {target_subset} positions")
    print(f"Node limit:        {node_limit} nodes/position")
    print(f"Threads:           {threads}")
    print("=" * 60)

    val_targets_path = dir_path / "value_targets.bin"
    # Initialize all targets with pure game outcome
    val_targets = np.memmap(val_targets_path, dtype=np.float32, mode="w+", shape=(num_samples, 3))
    for i in range(num_samples):
        outcome = int(ds.outcomes[i])
        val_targets[i] = outcome_to_one_hot(outcome)

    # Pick deterministic random subset of indices
    rng = np.random.RandomState(42)
    subset_indices = rng.choice(num_samples, size=target_subset, replace=False)

    with StockfishEvaluator(stockfish_path, threads=threads) as sf:
        for count, idx in enumerate(subset_indices):
            idx = int(idx)
            ply = int(ds.plies[idx])
            outcome = int(ds.outcomes[idx])

            # Positions in first 10 plies get no value loss, but can still be evaluated
            is_white = ply % 2 == 0
            planes = np.array(ds.planes[idx], dtype=np.float32)
            fen = decode_planes_to_fen(planes, is_white_to_move=is_white, ply=ply)

            cp = sf.evaluate_fen(fen, nodes=node_limit)
            sf_wdl = centipawns_to_wdl(cp)
            blended = blend_targets(sf_wdl, outcome)

            val_targets[idx] = blended

            if (count + 1) % 500 == 0 or (count + 1) == target_subset:
                print(f"Evaluated {count + 1} / {target_subset} positions...")

    val_targets.flush()
    print(f"\nSaved Stockfish blended value targets to {val_targets_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stockfish value targets for a dataset.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing dataset.")
    parser.add_argument(
        "--stockfish",
        type=str,
        default=str(DEFAULT_STOCKFISH_PATH),
        help="Path to Stockfish binary.",
    )
    parser.add_argument(
        "--subset", type=int, default=50000, help="Number of positions to evaluate."
    )
    parser.add_argument("--nodes", type=int, default=10000, help="Nodes per position.")
    parser.add_argument("--threads", type=int, default=8, help="Stockfish threads.")

    args = parser.parse_args()
    generate_value_targets_for_dataset(
        data_dir=args.data_dir,
        stockfish_path=args.stockfish,
        subset_size=args.subset,
        node_limit=args.nodes,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
