"""ZeroPawn Stage D.1 Concurrency, Batching Delay, and Workload Sweep.

Evaluates:
1. Multi-game self-play throughput across Concurrency C=[1, 2, 4, 6, 8, 12, 16, 24, 32]
   under batching delays [0.5ms, 1.0ms, 2.0ms, 3.0ms].
2. Puzzle benchmark throughput and accuracy across batch sizes B=[8, 16, 32, 64].
3. Interactive UCI move latency across batch sizes.
4. CPU-only fallback behavior.

Outputs: docs/perf/CONCURRENCY_SWEEP.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # ensures DLL loading
import zeropawn_cpp
from python.hardware.detector import detect_hardware
from python.model.chess_net import ChessNet
from python.eval.puzzles import PuzzleEvaluator, load_puzzles
from python.selfplay.client import DirectInferenceClient
from python.selfplay.inference_server import InferenceServer
from python.selfplay.worker import SelfPlayWorker


def get_system_metrics() -> Dict[str, float]:
    vm = psutil.virtual_memory()
    res = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((vm.total - vm.available) / 1e9, 3),
        "ram_free_gb": round(vm.available / 1e9, 3),
        "ram_percent": vm.percent,
    }
    if torch.cuda.is_available():
        res["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        res["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
    return res


def benchmark_selfplay_concurrency(
    checkpoint_path: str,
    device_str: str = "cuda",
    concurrency_list: List[int] = [1, 2, 4, 6, 8, 12, 16, 24, 32],
    delays_ms: List[float] = [1.0, 2.0, 3.0],
    sims_per_move: int = 50,
    moves_per_game: int = 10,
) -> Dict[str, Any]:
    print("\n--- Part 1: Self-Play Concurrency & Batching Delay Sweep ---")
    results = {}

    for delay in delays_ms:
        results[f"delay_{delay}ms"] = {}
        print(f"\n>> Testing Batching Delay: {delay:.1f} ms <<")

        for c in concurrency_list:
            server = InferenceServer(
                checkpoint_path=checkpoint_path,
                device=device_str,
                min_batch_size=1,
                max_batch_size=max(64, c * 4),
                max_wait_ms=delay,
                use_bfloat16=True,
                channels_last=True,
            )
            client = DirectInferenceClient(server)
            worker = SelfPlayWorker(
                worker_id=0,
                client=client,
                num_concurrent_games=c,
                max_wait_ms=delay,
                full_search_sims=sims_per_move,
                fast_search_sims=sims_per_move,
                max_plies=moves_per_game,
                pipelined=True,
                search_mode="serial",
            )

            worker.start()
            time.sleep(2.0)  # warmup

            evals_start = server.total_evaluations
            batches_start = server.total_batches
            t0 = time.perf_counter()

            time.sleep(6.0)  # measured window

            t1 = time.perf_counter()
            evals_end = server.total_evaluations
            batches_end = server.total_batches
            elapsed = t1 - t0

            worker.stop()

            delta_evals = evals_end - evals_start
            delta_batches = batches_end - batches_start
            evals_per_sec = float(delta_evals / max(1e-4, elapsed))
            avg_batch = float(delta_evals / max(1, delta_batches))
            avg_queue_wait_ms = float(worker.aggregator.get_window_queue_wait_ms())

            print(
                f"C={c:2d} | Evals/s: {evals_per_sec:7.1f} | Avg Batch: {avg_batch:4.1f} | "
                f"Queue Wait: {avg_queue_wait_ms:4.2f} ms | Window: {elapsed:4.1f}s"
            )

            results[f"delay_{delay}ms"][f"C_{c}"] = {
                "concurrency": c,
                "elapsed_sec": round(elapsed, 2),
                "evals_per_sec": round(evals_per_sec, 1),
                "avg_batch_size": round(avg_batch, 2),
                "avg_queue_wait_ms": round(avg_queue_wait_ms, 2),
            }

    return results


def benchmark_puzzle_batch_sizes(
    model: ChessNet,
    device: torch.device,
    batch_sizes: List[int] = [8, 16, 32, 64],
    num_puzzles: int = 20,
    sims_per_puzzle: int = 50,
) -> Dict[str, Any]:
    print(f"\n--- Part 2: Puzzle Evaluation Across Batch Sizes ({num_puzzles} puzzles) ---")
    puzzles = load_puzzles("data/puzzles/lichess_puzzles_1000_1200_1600.csv")[:num_puzzles]
    results = {}

    # Serial baseline
    eval_ser = PuzzleEvaluator(
        model=model, device=str(device), puzzles=puzzles, search_mode="serial"
    )
    t0 = time.perf_counter()
    res_ser = eval_ser.evaluate_mcts(num_simulations=sims_per_puzzle)
    t_ser = time.perf_counter() - t0
    print(
        f"Serial Baseline: {t_ser:5.2f}s | Solved {res_ser['solved']}/{num_puzzles} ({res_ser['solve_rate'] * 100:.1f}%)"
    )

    results["serial"] = {
        "time_sec": round(t_ser, 2),
        "solved": res_ser["solved"],
        "solve_rate": round(res_ser["solve_rate"], 3),
        "puzzles_per_sec": round(num_puzzles / max(1e-4, t_ser), 2),
    }

    for b in batch_sizes:
        eval_bat = PuzzleEvaluator(
            model=model, device=str(device), puzzles=puzzles, search_mode="batched", batch_size=b
        )
        t0 = time.perf_counter()
        res_bat = eval_bat.evaluate_mcts(num_simulations=sims_per_puzzle)
        t_bat = time.perf_counter() - t0
        speedup = t_ser / max(1e-4, t_bat)
        puz_sec = num_puzzles / max(1e-4, t_bat)
        print(
            f"Batched B={b:2d}: {t_bat:5.2f}s | Solved {res_bat['solved']}/{num_puzzles} ({res_bat['solve_rate'] * 100:.1f}%) | Speedup: {speedup:5.2f}x | {puz_sec:4.1f} puzzles/s"
        )

        results[f"B_{b}"] = {
            "batch_size": b,
            "time_sec": round(t_bat, 2),
            "speedup_vs_serial": round(speedup, 2),
            "solved": res_bat["solved"],
            "solve_rate": round(res_bat["solve_rate"], 3),
            "puzzles_per_sec": round(puz_sec, 2),
        }

    return results


def benchmark_cpu_fallback(
    model: ChessNet,
    num_simulations: int = 50,
) -> Dict[str, Any]:
    print("\n--- Part 3: CPU-Only Fallback Benchmark ---")
    cpu_model = ChessNet(num_blocks=12, num_filters=128).cpu().eval()

    def cpu_eval(game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            p_logits, v_wdl = cpu_model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

    def cpu_batched_eval(batch_planes: np.ndarray):
        n = int(batch_planes.shape[0])
        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32)).view(n, 19, 8, 8)
        with torch.no_grad():
            p_logits, v_wdl = cpu_model(t)
            policy = torch.softmax(p_logits.float(), dim=-1).numpy()
            wdl = torch.softmax(v_wdl.float(), dim=-1).numpy()
            values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)
        return policy.astype(np.float32), values

    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    mcts = zeropawn_cpp.MCTS(cfg)

    game = zeropawn_cpp.ChessGame()

    # Serial CPU
    mcts.clear()
    t0 = time.perf_counter()
    res_ser = mcts.search(game, num_simulations, cpu_eval)
    t_ser = time.perf_counter() - t0
    ser_sims_sec = num_simulations / max(1e-4, t_ser)
    print(
        f"CPU Serial MCTS:  {t_ser * 1000:6.1f} ms | {ser_sims_sec:5.1f} sims/s | Action: {game.action_to_string(res_ser.best_action)}"
    )

    # Batched CPU (B=4)
    mcts.clear()
    t0 = time.perf_counter()
    res_bat = mcts.search_batched(game, num_simulations, 4, cpu_batched_eval)
    t_bat = time.perf_counter() - t0
    bat_sims_sec = num_simulations / max(1e-4, t_bat)
    print(
        f"CPU Batched B=4:  {t_bat * 1000:6.1f} ms | {bat_sims_sec:5.1f} sims/s | Action: {game.action_to_string(res_bat.best_action)}"
    )

    return {
        "num_simulations": num_simulations,
        "cpu_serial": {
            "time_ms": round(t_ser * 1000, 2),
            "sims_per_sec": round(ser_sims_sec, 1),
            "best_action": game.action_to_string(res_ser.best_action),
        },
        "cpu_batched_B4": {
            "time_ms": round(t_bat * 1000, 2),
            "sims_per_sec": round(bat_sims_sec, 1),
            "best_action": game.action_to_string(res_bat.best_action),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="ZeroPawn Concurrency & Workload Sweep")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints_scaled/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,24,32",
        help="Comma-separated worker concurrencies",
    )
    parser.add_argument(
        "--delays", type=str, default="1.0,2.0,3.0", help="Comma-separated batching delays in ms"
    )
    parser.add_argument("--output", type=str, default="docs/perf/CONCURRENCY_SWEEP.json")
    args = parser.parse_args()

    hw = detect_hardware()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== ZeroPawn Concurrency & Workload Sweep ===")
    print(f"Device: {device} ({hw.gpu_name if hw.cuda_available else 'CPU'})")

    model = ChessNet(num_blocks=12, num_filters=128)
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
            model.load_state_dict(sd)
            print(f"Loaded checkpoint from: {args.checkpoint}")
        except Exception as e:
            print(f"Warning: Failed loading checkpoint ({e})")
    model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    report: Dict[str, Any] = {
        "hardware": hw.to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_metrics_start": get_system_metrics(),
    }

    # Parse concurrencies and delays
    c_list = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    d_list = [float(d.strip()) for d in args.delays.split(",") if d.strip()]

    # 1. Self-Play Concurrency & Delay Sweep
    report["selfplay_concurrency"] = benchmark_selfplay_concurrency(
        checkpoint_path=args.checkpoint,
        device_str=str(device),
        concurrency_list=c_list if c_list else [1, 2, 4, 8, 16, 24, 32],
        delays_ms=d_list if d_list else [1.0, 2.0, 3.0],
        sims_per_move=40,
        moves_per_game=8,
    )

    # 2. Puzzle Evaluation Across Batch Sizes
    report["puzzle_eval"] = benchmark_puzzle_batch_sizes(
        model=model,
        device=device,
        batch_sizes=[8, 16, 32, 64],
        num_puzzles=15,
        sims_per_puzzle=40,
    )

    # 3. CPU-Only Fallback
    report["cpu_fallback"] = benchmark_cpu_fallback(model=model, num_simulations=40)

    report["system_metrics_end"] = get_system_metrics()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Concurrency & Workload Sweep Complete. Report saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
