"""Evaluate value head calibration and reliability on held-out human validation data."""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
import numpy as np
import torch
import torch.nn.functional as F

from python.model.chess_net import ChessNet


def _find_val_dir(val_dir: str | None = None) -> Path:
    if val_dir and (Path(val_dir) / "metadata.json").is_file():
        return Path(val_dir)
    for cand in [
        "data/dataset_training/val",
        "data/dataset_scaled/val",
        "data/dataset_elite2500/val",
        "data/dataset/val",
    ]:
        p = Path(cand)
        if (p / "metadata.json").is_file():
            return p
    return Path(val_dir or "data/dataset_training/val")


def evaluate_val_set(
    ckpt_path: str, val_dir: str | None = None, device: str = "cuda", max_samples: int | None = None
):
    print(f"\n{'=' * 75}")
    print(f"EVALUATING HELD-OUT VAL CALIBRATION ON: {ckpt_path}")
    print(f"{'=' * 75}")

    val_path = _find_val_dir(val_dir)
    with open(val_path / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_avail = meta["num_samples"]
    num_samples = min(total_avail, max_samples) if max_samples else total_avail
    print(f"Loaded validation set: {num_samples} samples from {val_path}")

    # Memory map planes and outcomes
    planes_mm = np.memmap(
        val_path / "planes.bin", dtype=np.float32, mode="r", shape=(num_samples, 19, 8, 8)
    )
    outcomes_mm = np.memmap(
        val_path / "outcomes.bin", dtype=np.uint8, mode="r", shape=(num_samples,)
    )

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    num_blocks = ckpt.get("num_blocks", 12)
    num_filters = ckpt.get("num_filters", 128)

    model = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    bins = [
        (-1.01, -0.95),
        (-0.95, -0.90),
        (-0.90, -0.80),
        (-0.80, -0.60),
        (-0.60, -0.40),
        (-0.40, -0.20),
        (-0.20, 0.00),
        (0.00, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.80),
        (0.80, 0.90),
        (0.90, 0.95),
        (0.95, 1.01),
    ]

    bin_counts = [0] * len(bins)
    bin_scores = [0.0] * len(bins)
    bin_values = [0.0] * len(bins)

    batch_size = 512
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.inference_mode():
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(num_samples, (b + 1) * batch_size)

            planes_batch = torch.from_numpy(np.array(planes_mm[start_idx:end_idx])).to(device)
            outcomes_batch = outcomes_mm[start_idx:end_idx]

            _, w_logits = model(planes_batch)
            w_probs = F.softmax(w_logits.float(), dim=-1).cpu().numpy()

            # Scalar value from side-to-move view: P(win) - P(loss)
            values = w_probs[:, 0] - w_probs[:, 2]

            # Actual scores: 0=win (1.0), 1=draw (0.5), 2=loss (0.0)
            actual_scores = np.where(
                outcomes_batch == 0, 1.0, np.where(outcomes_batch == 1, 0.5, 0.0)
            )

            for v, s in zip(values, actual_scores):
                for b_idx, (b_low, b_high) in enumerate(bins):
                    if b_low <= v < b_high or (b_idx == len(bins) - 1 and b_low <= v <= b_high):
                        bin_counts[b_idx] += 1
                        bin_values[b_idx] += float(v)
                        bin_scores[b_idx] += float(s)
                        break

    import math

    def wilson_interval(successes: float, trials: int, z: float = 1.95996):
        if trials <= 0:
            return (0.0, 0.0)
        p = successes / trials
        denom = 1.0 + (z**2) / trials
        centre = (p + (z**2) / (2.0 * trials)) / denom
        spread = (
            z * math.sqrt(max(0.0, (p * (1.0 - p)) / trials + (z**2) / (4.0 * (trials**2))))
        ) / denom
        return (max(0.0, centre - spread), min(1.0, centre + spread))

    header = f"{'Value Bin':<18} {'Count':<8} {'Pct':<8} {'Mean Val':<10} {'Mean Score':<12} {'Wilson 95% CI':<20} {'Expected':<10} {'Cal Gap':<10}"
    print(header)
    print("-" * len(header))

    total = sum(bin_counts)
    for b_idx, (b_low, b_high) in enumerate(bins):
        cnt = bin_counts[b_idx]
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        mean_v = (bin_values[b_idx] / cnt) if cnt > 0 else 0.0
        mean_s = (bin_scores[b_idx] / cnt) if cnt > 0 else 0.0
        ci_l, ci_u = wilson_interval(bin_scores[b_idx], cnt)
        ci_str = f"[{ci_l:.3f}, {ci_u:.3f}]" if cnt > 0 else "N/A"
        exp_s = (mean_v + 1.0) / 2.0
        gap = mean_s - exp_s
        bin_label = f"[{max(-1.0, b_low):.2f}, {min(1.0, b_high):.2f})"
        print(
            f"{bin_label:<18} {cnt:<8} {pct:<7.2f}% {mean_v:<+10.4f} {mean_s:<12.4f} {ci_str:<20} {exp_s:<10.4f} {gap:<+10.4f}"
        )

    print("-" * len(header))
    print(f"Total Evaluated: {total}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate value head reliability and calibration")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--val-dir", type=str, default=None, help="Validation dataset directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.checkpoint:
        evaluate_val_set(
            args.checkpoint, val_dir=args.val_dir, device=args.device, max_samples=args.max_samples
        )
    else:
        dev = args.device
        default_ckpt = "models/checkpoints_scaled/best_model.pt"
        if not Path(default_ckpt).exists():
            default_ckpt = "models/best_model.pt"
        if Path(default_ckpt).exists():
            evaluate_val_set(
                default_ckpt, val_dir=args.val_dir, device=dev, max_samples=args.max_samples
            )
        if Path("models/checkpoints_rl/phase4_final_checkpoint.pt").exists():
            evaluate_val_set(
                "models/checkpoints_rl/phase4_final_checkpoint.pt",
                val_dir=args.val_dir,
                device=dev,
                max_samples=args.max_samples,
            )
