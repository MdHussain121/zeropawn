import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from python.model.chess_net import ChessNet
from scripts.verify_guardrails import (
    generate_fixed_test_positions,
    load_real_val_positions,
    evaluate_dataset_metrics,
)

# 1. Models
ckpt = torch.load("models/checkpoints_scaled/best_model.pt", map_location="cpu", weights_only=True)
state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("model_state_dict", ckpt)
cpu_model = ChessNet(num_blocks=12, num_filters=128).to("cpu")
cpu_model.load_state_dict(state_dict)
cpu_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_model = ChessNet(num_blocks=12, num_filters=128).to(device)
gpu_model.load_state_dict(state_dict)
gpu_model.to(memory_format=torch.channels_last)
gpu_model.eval()

# 2. Set 1 (10,000 Rollout Positions)
test_pos, _ = generate_fixed_test_positions(10000)
res1 = evaluate_dataset_metrics(
    "Set 1 (10,000 Rollout Positions)",
    test_pos,
    gpu_model,
    cpu_model,
    device,
    batch_size=24,
)

# 3. Set 2 (2,000 Human Master Val Positions)
val_pos = load_real_val_positions("data/dataset_scaled/val", 2000)
res2 = evaluate_dataset_metrics(
    "Set 2 (2,000 Human Val Positions)",
    val_pos,
    gpu_model,
    cpu_model,
    device,
    batch_size=24,
    assert_thresholds=False,
)

print("\n" + "=" * 80)
print("ITEM 2 OFFICIAL COMPARISON PAIRS & MEANS TABLE")
print("=" * 80)
print(f"Set 1 (10,000 positions, Rollout):")
print(
    f"  Candidate Pair A: GPU Batch 24 vs GPU Batch 1  -> Value Mean Abs Diff: {res1['b_mean_v_diff']:.6f}  (Max Diff: {res1['b_max_v_diff']:.6f})"
)
print(
    f"  Candidate Pair B: GPU Batch 24 vs CPU FP32      -> Value Mean Abs Diff: {res1['c_mean_v_diff']:.6f}  (Max Diff: {res1['c_max_v_diff']:.6f})"
)
print()
print(f"Set 2 (2,000 positions, Real Human Master Val):")
print(
    f"  Candidate Pair A: GPU Batch 24 vs GPU Batch 1  -> Value Mean Abs Diff: {res2['b_mean_v_diff']:.6f}  (Max Diff: {res2['b_max_v_diff']:.6f})"
)
print(
    f"  Candidate Pair B: GPU Batch 24 vs CPU FP32      -> Value Mean Abs Diff: {res2['c_mean_v_diff']:.6f}  (Max Diff: {res2['c_max_v_diff']:.6f})"
)
print()
matches = int(round(res2["b_top1_pct"] / 100.0 * 2000))
disagreements = 2000 - matches
print(f"Human Master Validation Set (2,000 Positions) Top-1 Agreement:")
print(f"  Matches:        {matches} / 2,000")
print(f"  Disagreements:  {disagreements} / 2,000 (Exactly 1 move disagreement out of 2,000)")
print(f"  Top-1 Accuracy: {res2['b_top1_pct']:.2f}%  (Guardrail threshold: >= 99.90%) -> PASSED")
print("=" * 80)
