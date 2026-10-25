"""Evaluation script for held-out validation games.

Evaluates trained checkpoint on held-out validation dataset and outputs:
- Top-1 move accuracy (raw and legally masked)
- Top-3 move accuracy (raw and legally masked)
- Policy Shannon entropy
- WDL cross-entropy loss
- 100% legal moves verification
- Gate check: Top-1 >= 35% and Top-3 >= 60%
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # ensures zeropawn_cpp
import zeropawn_cpp
from python.cli.common import setup_logging
from python.common.progress import (
    banner,
    emit_error,
    emit_report,
    error_block,
    format_kv_line,
    header_block,
    parameters_table,
    results_table,
    run_context,
    short_hash,
    summary_block,
)
from python.data.dataset import BinaryChessDataset
from python.model.chess_net import ChessNet
from python.data.fen_utils import decode_planes_to_fen
from python.train.train_supervised import compute_policy_entropy


def evaluate_heldout(
    checkpoint_path: str | Path,
    val_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, float | bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit_report(
        format_kv_line(
            [
                ("checkpoint", str(checkpoint_path)),
                ("val_dir", str(val_dir)),
                ("device", str(device)),
            ]
        )
    )

    val_ds = BinaryChessDataset(val_dir)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Auto-detect architecture from checkpoint metadata or state_dict keys
    num_blocks = checkpoint.get("num_blocks") if isinstance(checkpoint, dict) else None
    if num_blocks is None:
        block_indices = [
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("blocks.") and k.endswith(".bn2.weight")
        ]
        num_blocks = max(block_indices) + 1 if block_indices else 12

    num_filters = checkpoint.get("num_filters", 128) if isinstance(checkpoint, dict) else 128
    print(f"Model Architecture: {num_blocks} ResBlocks, {num_filters} filters")

    model = ChessNet(
        in_channels=19, num_filters=num_filters, num_blocks=num_blocks, num_actions=4672
    )
    model.load_state_dict(state_dict)
    model = model.to(device=device, memory_format=torch.channels_last)
    model.eval()

    total_samples = 0
    raw_correct_top1 = 0
    raw_correct_top3 = 0
    masked_correct_top1 = 0
    masked_correct_top3 = 0
    legal_top1_count = 0

    sum_policy_loss = 0.0
    sum_wdl_loss = 0.0
    sum_wdl_valid_count = 0
    sum_entropy = 0.0

    # Value calibration accumulators
    brier_sum = 0.0
    pred_w_sum = 0.0
    pred_d_sum = 0.0
    pred_l_sum = 0.0
    target_w_sum = 0.0
    target_d_sum = 0.0
    target_l_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            planes = batch["planes"].to(
                device, memory_format=torch.channels_last, non_blocking=True
            )
            moves = batch["move"].to(device, non_blocking=True)
            value_targets = batch["value_target"].to(device, non_blocking=True)
            plies = batch["ply"].to(device, non_blocking=True)
            batch_sz = moves.size(0)

            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_ctx:
                policy_logits, wdl_logits = model(planes)

                pol_loss = F.cross_entropy(policy_logits, moves, reduction="sum")
                log_wdl = F.log_softmax(wdl_logits, dim=-1)
                ce_wdl = -(value_targets * log_wdl).sum(dim=-1)
                val_mask = ((plies >= 10) & (value_targets.sum(dim=-1) > 0)).float()
                wdl_loss_sum = (ce_wdl * val_mask).sum().item()
                valid_count = int(val_mask.sum().item())

            sum_policy_loss += float(pol_loss.item())
            sum_wdl_loss += wdl_loss_sum
            sum_wdl_valid_count += valid_count
            total_samples += batch_sz
            sum_entropy += compute_policy_entropy(policy_logits) * batch_sz

            # Value calibration stats on valid samples
            if valid_count > 0:
                wdl_probs = F.softmax(wdl_logits, dim=-1)
                mask_bool = val_mask.bool()
                valid_probs = wdl_probs[mask_bool]
                valid_targets = value_targets[mask_bool]

                brier_batch = ((valid_probs - valid_targets) ** 2).sum().item()
                brier_sum += brier_batch
                pred_w_sum += float(valid_probs[:, 0].sum().item())
                pred_d_sum += float(valid_probs[:, 1].sum().item())
                pred_l_sum += float(valid_probs[:, 2].sum().item())
                target_w_sum += float(valid_targets[:, 0].sum().item())
                target_d_sum += float(valid_targets[:, 1].sum().item())
                target_l_sum += float(valid_targets[:, 2].sum().item())

            # Raw top-k accuracy
            _, raw_pred = policy_logits.topk(3, dim=1, largest=True, sorted=True)
            raw_pred_t = raw_pred.t()
            raw_correct = raw_pred_t.eq(moves.view(1, -1).expand_as(raw_pred_t))
            raw_correct_top1 += int(raw_correct[:1].reshape(-1).float().sum().item())
            raw_correct_top3 += int(raw_correct[:3].reshape(-1).float().sum().item())

            # Legally-masked top-k accuracy (Contract: Illegal moves are masked before softmax)
            planes_cpu = batch["planes"].numpy()
            plies_cpu = batch["ply"].numpy()
            moves_cpu = batch["move"].cpu().numpy()

            for i in range(batch_sz):
                ply = int(plies_cpu[i])
                is_white = ply % 2 == 0
                fen = decode_planes_to_fen(planes_cpu[i], is_white_to_move=is_white, ply=ply)
                g = zeropawn_cpp.ChessGame(fen)
                legal_acts = g.legal_actions()

                logits_i = policy_logits[i].clone()
                mask = torch.full((4672,), -1e9, device=device)
                mask[legal_acts] = 0.0
                masked_logits = logits_i + mask

                _, pred_m = masked_logits.topk(3, largest=True, sorted=True)
                top1_act = int(pred_m[0].item())
                top3_acts = [int(a.item()) for a in pred_m]
                target_act = int(moves_cpu[i])

                if top1_act in legal_acts:
                    legal_top1_count += 1
                if target_act == top1_act:
                    masked_correct_top1 += 1
                if target_act in top3_acts:
                    masked_correct_top3 += 1

    raw_top1 = (raw_correct_top1 / total_samples) * 100.0 if total_samples > 0 else 0.0
    raw_top3 = (raw_correct_top3 / total_samples) * 100.0 if total_samples > 0 else 0.0
    masked_top1 = (masked_correct_top1 / total_samples) * 100.0 if total_samples > 0 else 0.0
    masked_top3 = (masked_correct_top3 / total_samples) * 100.0 if total_samples > 0 else 0.0
    pct_legal = (legal_top1_count / total_samples) * 100.0 if total_samples > 0 else 0.0
    policy_loss = sum_policy_loss / max(1, total_samples)
    wdl_loss = sum_wdl_loss / max(1, sum_wdl_valid_count)
    entropy = sum_entropy / max(1, total_samples)

    # Calibration statistics
    n_v = max(1, sum_wdl_valid_count)
    brier_score = brier_sum / n_v
    mean_pred_v = (pred_w_sum - pred_l_sum) / n_v
    mean_target_v = (target_w_sum - target_l_sum) / n_v
    mean_pred_w = pred_w_sum / n_v
    mean_pred_d = pred_d_sum / n_v
    mean_pred_l = pred_l_sum / n_v
    mean_target_w = target_w_sum / n_v
    mean_target_d = target_d_sum / n_v
    mean_target_l = target_l_sum / n_v

    gate_pass = (masked_top1 >= 35.0) and (masked_top3 >= 60.0)
    emit_report(
        results_table(
            [
                ("total_samples", f"{total_samples:,} (M)"),
                ("masked_top1", f"{masked_top1:.2f}% (M, gate >= 35.0%)"),
                ("masked_top3", f"{masked_top3:.2f}% (M, gate >= 60.0%)"),
                ("raw_top1", f"{raw_top1:.2f}% (M)"),
                ("raw_top3", f"{raw_top3:.2f}% (M)"),
                ("pct_legal_top1", f"{pct_legal:.2f}% (M, required 100.0%)"),
                ("policy_loss", f"{policy_loss:.4f} (M)"),
                ("wdl_loss", f"{wdl_loss:.4f} (M)"),
                ("entropy_nats", f"{entropy:.4f} (M)"),
                ("brier_score", f"{brier_score:.4f} (M)"),
                ("mean_pred_value", f"{mean_pred_v:+.4f} (target {mean_target_v:+.4f}) (M)"),
                ("pred_wdl", f"W={mean_pred_w:.3f} D={mean_pred_d:.3f} L={mean_pred_l:.3f} (M)"),
                (
                    "target_wdl",
                    f"W={mean_target_w:.3f} D={mean_target_d:.3f} L={mean_target_l:.3f} (M)",
                ),
                ("gate", "PASS (D)" if gate_pass else "DIAGNOSTIC BELOW GATE (D)"),
            ]
        )
    )

    return {
        "total_samples": total_samples,
        "masked_top1": masked_top1,
        "masked_top3": masked_top3,
        "raw_top1": raw_top1,
        "raw_top3": raw_top3,
        "pct_legal": pct_legal,
        "policy_loss": policy_loss,
        "wdl_loss": wdl_loss,
        "entropy": entropy,
        "brier_score": brier_score,
        "mean_pred_value": mean_pred_v,
        "mean_target_value": mean_target_v,
        "pred_wdl": (mean_pred_w, mean_pred_d, mean_pred_l),
        "target_wdl": (mean_target_w, mean_target_d, mean_target_l),
        "gate_pass": gate_pass,
    }


def main() -> None:
    setup_logging(log_dir="logs", log_filename="eval_heldout.log")
    ctx = run_context("heldout-eval")
    t0 = time.time()
    log_path = "logs/eval_heldout.log"

    parser = argparse.ArgumentParser(description="Evaluate checkpoint on held-out validation set.")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path (default: auto-detected)."
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Validation dataset directory (default: auto-detected).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")

    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        resolved_ckpt = Path(args.checkpoint)
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found at '{resolved_ckpt}'")
    else:
        from python.cli.models import get_default_model_path

        def_ckpt = get_default_model_path()
        if def_ckpt and def_ckpt.is_file():
            resolved_ckpt = def_ckpt
        else:
            fallback = _repo_root / "models" / "checkpoints_scaled" / "best_model.pt"
            if fallback.is_file():
                resolved_ckpt = fallback
            else:
                raise FileNotFoundError(
                    "No checkpoint found. Looked for models/checkpoints_scaled/best_model.pt. "
                    "Specify a checkpoint using --checkpoint."
                )

    # Resolve val_dir
    if args.val_dir:
        resolved_val = Path(args.val_dir)
        if not resolved_val.is_dir():
            raise FileNotFoundError(f"Validation directory not found at '{resolved_val}'")
    else:
        candidates = [
            _repo_root / "data" / "dataset_training" / "val",
            _repo_root / "data" / "dataset_scaled" / "val",
            _repo_root / "data" / "dataset" / "val",
        ]
        resolved_val = None
        for c in candidates:
            if c.is_dir():
                resolved_val = c
                break
        if resolved_val is None:
            raise FileNotFoundError(
                f"No validation directory found. Checked: {[str(c) for c in candidates]}. "
                "Specify validation directory using --val-dir."
            )

    emit_report(banner("heldout-eval", ctx["start"] + " (local)"))
    ckpt_id = f"{resolved_ckpt} sha={short_hash(resolved_ckpt)} (M)"
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            ckpt_id,
            "auto",
            "n/a (dataset order)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("checkpoint", str(resolved_ckpt), "user" if args.checkpoint else "auto-detected"),
                ("val_dir", str(resolved_val), "user" if args.val_dir else "auto-detected"),
                ("batch_size", args.batch_size, "user" if args.batch_size != 256 else "default"),
            ]
        )
    )

    try:
        evaluate_heldout(
            checkpoint_path=resolved_ckpt,
            val_dir=resolved_val,
            batch_size=args.batch_size,
        )
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 130))
        raise
    except Exception as e:
        emit_error(
            error_block(
                f"heldout-eval failed: {e}",
                "missing checkpoint, empty val dir, or OOM/VRAM",
                "check --checkpoint/--val-dir and free VRAM, then see LOG",
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
