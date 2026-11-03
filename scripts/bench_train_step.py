"""M1-style RL trainer step benchmark: interleaved arms, bf16 autocast on vs off.

Fixed trainer (N-02 restored) at production shape: ChessNet(12,128), batch 256,
35% full-search rows, SGD momentum 0.9 lr 5e-4. Alternating 30-warmup + 120-timed
step windows per arm; reports steps/s and peak VRAM per arm.
Usage: python tmp_bench/bench_train_step.py
"""

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import RLTrainer

N = 256
FULL_FRAC = 0.35
WARMUP = 30
TIMED = 120
ROUNDS = 3


def make_batch():
    rng = np.random.default_rng(123)
    return {
        "planes": torch.from_numpy(rng.standard_normal((N, 19, 8, 8), dtype=np.float32)),
        "visits": torch.from_numpy(rng.random((N, 4672), dtype=np.float32)),
        "outcomes": torch.from_numpy(rng.integers(0, 3, N).astype(np.int64)),
        "is_full_search": torch.from_numpy((rng.random(N) < FULL_FRAC).astype(np.bool_)),
    }


def run_arm(trainer, batch, tag):
    for _ in range(WARMUP):
        trainer.train_step(batch_dict=batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    losses = []
    for _ in range(TIMED):
        m = trainer.train_step(batch_dict=batch)
        losses.append(m["total_loss"])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    sps = TIMED / dt
    ms = dt / TIMED * 1000
    print(
        f"{tag}: {sps:.1f} steps/s ({ms:.1f} ms/step) peak VRAM {peak:.0f} MB "
        f"loss[last5]={statistics.mean(losses[-5:]):.4f}"
    )
    return sps, peak


def main():
    buf = ReplayBuffer(buffer_dir="tmp_bench/buf_m1", capacity=4096, create_new=True)
    batch = make_batch()
    print(
        f"N={N} full_frac={FULL_FRAC} warmup={WARMUP} timed={TIMED} rounds={ROUNDS} (interleaved)"
    )

    def fresh_trainer(use_bf16):
        torch.manual_seed(0)
        model = ChessNet(num_blocks=12, num_filters=128)
        return RLTrainer(
            model=model,
            replay_buffer=buf,
            device="cuda",
            batch_size=N,
            use_bfloat16=use_bf16,
            checkpoint_dir="tmp_bench/ck_m1",
            sync_weights_path="tmp_bench/sync_m1.pt",
        )

    # Warm-up both arms to stable clocks.
    t_bf16 = fresh_trainer(True)
    t_fp32 = fresh_trainer(False)
    run_arm(t_bf16, batch, "warmup bf16 ")
    run_arm(t_fp32, batch, "warmup fp32 ")

    bf16_sps, fp32_sps, bf16_peak, fp32_peak = [], [], [], []
    for r in range(ROUNDS):
        s, p = run_arm(t_bf16, batch, f"[r{r + 1}] bf16    ")
        bf16_sps.append(s)
        bf16_peak.append(p)
        s, p = run_arm(t_fp32, batch, f"[r{r + 1}] fp32    ")
        fp32_sps.append(s)
        fp32_peak.append(p)

    b_med, f_med = statistics.median(bf16_sps), statistics.median(fp32_sps)
    print(
        f"bf16: median {b_med:.1f} steps/s (runs {[f'{x:.1f}' for x in bf16_sps]}), peak VRAM median {statistics.median(bf16_peak):.0f} MB"
    )
    print(
        f"fp32: median {f_med:.1f} steps/s (runs {[f'{x:.1f}' for x in fp32_sps]}), peak VRAM median {statistics.median(fp32_peak):.0f} MB"
    )
    print(
        f"autocast restoration speedup: {b_med / f_med:.2f}x | peak VRAM change: {(statistics.median(bf16_peak) - statistics.median(fp32_peak)) / statistics.median(fp32_peak) * 100:+.0f}%"
    )


if __name__ == "__main__":
    main()
