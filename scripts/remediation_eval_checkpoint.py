"""Per-domain checkpoint evaluation for the 2026-10-31 remediation (Phase H).

Computes, on one dataset split, with identical preprocessing for every
candidate checkpoint:
  - policy top-1 / top-3 (raw logits, eval convention)
  - policy cross-entropy
  - WDL cross-entropy over rows the trainer's mask admits (ply >= 10, sum > 0)
  - per-class WDL cross-entropy + per-class prediction rates
    (P(predict c | true c)) — the SLQ-01 corruption signature detector
  - mean predicted WDL distribution, Brier score over admitted rows

Usage:
    python scripts/remediation_eval_checkpoint.py --checkpoint <pt> --split <dir> [--rows 4096]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.data.dataset import BinaryChessDataset
from python.model.chess_net import ChessNet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    num_blocks = ck.get("num_blocks", 12) if isinstance(ck, dict) else 12
    num_filters = ck.get("num_filters", 128) if isinstance(ck, dict) else 128
    model = ChessNet(in_channels=19, num_filters=num_filters, num_blocks=num_blocks, num_actions=4672)
    model.load_state_dict(state)
    model = model.to(device).eval()

    ds = BinaryChessDataset(args.split)
    n = len(ds)
    rng = np.random.default_rng(20261031)
    idxs = rng.choice(n, size=min(args.rows, n), replace=False)

    tot_top1 = tot_top3 = tot_n = 0
    sum_pol_ce = 0.0
    sum_wdl_ce = 0.0
    n_wdl = 0
    # per-class accumulators over admitted rows
    class_ce = [0.0, 0.0, 0.0]
    class_n = [0, 0, 0]
    confusion = np.zeros((3, 3), dtype=np.int64)  # [true][predicted]
    pred_sum = [0.0, 0.0, 0.0]
    brier = 0.0

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(idxs), args.batch_size):
            batch_idx = idxs[start : start + args.batch_size]
            planes = torch.stack([ds[int(i)]["planes"] for i in batch_idx]).to(device)
            moves = torch.tensor([int(ds[int(i)]["move"]) for i in batch_idx], device=device)
            vtargets = torch.stack([ds[int(i)]["value_target"] for i in batch_idx]).to(device)
            plies = torch.tensor([int(ds[int(i)]["ply"]) for i in batch_idx], device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, wdl = model(planes)
            logits = logits.float()
            wdl = wdl.float()

            tot_n += len(batch_idx)
            sum_pol_ce += F.cross_entropy(logits, moves, reduction="sum").item()
            pred = logits.topk(3, dim=1).indices
            tot_top1 += (pred[:, 0] == moves).sum().item()
            tot_top3 += (pred == moves.unsqueeze(1)).any(dim=1).sum().item()

            log_wdl = F.log_softmax(wdl, dim=-1)
            ce = -(vtargets * log_wdl).sum(dim=-1)
            mask = (plies >= 10) & (vtargets.sum(dim=-1) > 0)
            sum_wdl_ce += (ce * mask).sum().item()
            n_wdl += int(mask.sum().item())

            probs = F.softmax(wdl, dim=-1)
            admitted = mask.nonzero(as_tuple=True)[0]
            for row in admitted.tolist():
                c = int(vtargets[row].argmax().item())
                p = int(probs[row].argmax().item())
                class_ce[c] += ce[row].item()
                class_n[c] += 1
                confusion[c, p] += 1
                pred_sum[0] += probs[row, 0].item()
                pred_sum[1] += probs[row, 1].item()
                pred_sum[2] += probs[row, 2].item()
                brier += ((probs[row] - vtargets[row]) ** 2).sum().item()

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_prefix": (
            __import__("hashlib").sha256(Path(args.checkpoint).read_bytes()).hexdigest()[:16]
            if Path(args.checkpoint).stat().st_size < 1 << 30
            else "skipped (>1GB)"
        ),
        "split": str(args.split),
        "rows_sampled": int(tot_n),
        "policy_top1_pct": round(100.0 * tot_top1 / max(1, tot_n), 3),
        "policy_top3_pct": round(100.0 * tot_top3 / max(1, tot_n), 3),
        "policy_ce": round(sum_pol_ce / max(1, tot_n), 4),
        "wdl_ce_admitted": round(sum_wdl_ce / max(1, n_wdl), 4),
        "wdl_admitted_rows": n_wdl,
        "per_class_ce": {
            ["win", "draw", "loss"][c]: (
                round(class_ce[c] / class_n[c], 4) if class_n[c] else None
            )
            for c in range(3)
        },
        "per_class_rows": {["win", "draw", "loss"][c]: class_n[c] for c in range(3)},
        "p_predict_given_true": {
            ["win", "draw", "loss"][c]: (
                round(float(confusion[c, c]) / class_n[c], 4) if class_n[c] else None
            )
            for c in range(3)
        },
        "confusion_true_pred": {
            ["win", "draw", "loss"][c]: {
                ["win", "draw", "loss"][k]: int(confusion[c, k]) for k in range(3)
            }
            for c in range(3)
        },
        "mean_pred_wdl_admitted": [round(x / max(1, n_wdl), 4) for x in pred_sum],
        "brier_admitted": round(brier / max(1, n_wdl), 4),
        "device": str(device),
        "duration_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
