"""Supervised Opening Policy Training and Fine-Tuning Module.

Teaches the existing dual-head ResNet (12 blocks, 128 filters) to select
recognized, sound opening moves across mainline and sound sideline variations.

Key Invariants:
- Preserves the model architecture strictly (12 ResBlocks, 128 filters).
- Preserves the side-to-move boolean truthiness convention.
- Supports both single-move cross-entropy and multi-move distribution targets.
- Preserves existing checkpoints: NEVER overwrites models/checkpoints_scaled/best_model.pt.
- Supports mixed data training (interleaving opening batches with general dataset batches).
- Fail-closed resume provenance (audit SA3-03): the default resume checkpoint
  is the registry-designated verified baseline (never a hardcoded superseded
  artifact), the resume checkpoint's registry standing is validated before
  any load, and every run persists a ``run_manifest.json`` that a later
  resume must match (see python/train/opening_run_manifest.py).
"""

from __future__ import annotations

import logging
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
from python.cli.common import (
    REPO_ROOT,
    load_engine_model,
    resolve_device,
    setup_environment,
)
from python.data.dataset import BinaryChessDataset, SourceAwareCombinedDataset
from python.model.chess_net import ChessNet
from python.train.train_supervised import (
    WarmupCosineScheduler,
    _atomic_torch_save,
    compute_accuracy_topk,
    compute_policy_entropy,
)

logger = logging.getLogger(__name__)


def train_opening_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_autocast: bool = True,
    policy_weight: float = 1.0,
    wdl_weight: float = 0.1,
) -> Tuple[float, float, float, float, float, float]:
    """Execute a single supervised training step on an opening batch."""
    planes = batch["planes"].to(device, memory_format=torch.channels_last, non_blocking=True)
    moves = batch["move"].to(device, non_blocking=True)
    value_targets = batch["value_target"].to(device, non_blocking=True)
    plies = batch["ply"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    autocast_context = (
        torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16
        )
        if use_autocast and device.type == "cuda"
        else nullcontext()
    )

    with autocast_context:
        policy_logits, wdl_logits = model(planes)

        # Policy loss: soft-target CE when policy_target available, else hard-label CE with label smoothing
        if "policy_target" in batch and batch["policy_target"] is not None:
            policy_target = batch["policy_target"].to(device, non_blocking=True)
            log_policy = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(policy_target * log_policy).sum(dim=-1).mean()
        else:
            policy_loss = F.cross_entropy(policy_logits, moves, label_smoothing=0.05)

        # WDL cross-entropy: apply WDL loss mask (ply >= 10 and value target non-zero) (DI-21)
        log_wdl = F.log_softmax(wdl_logits, dim=-1)
        val_mask = ((plies >= 10) & (value_targets.sum(dim=-1) > 0)).float()
        valid_count = val_mask.sum()
        if valid_count > 0:
            per_sample_wdl = -(value_targets * log_wdl).sum(dim=-1)
            ce_wdl = (per_sample_wdl * val_mask).sum() / valid_count
        else:
            ce_wdl = torch.tensor(0.0, device=device, dtype=wdl_logits.dtype)

        total_loss = policy_weight * policy_loss + wdl_weight * ce_wdl

    if not torch.isfinite(total_loss):
        logger.warning("Non-finite opening loss encountered; skipping optimizer step")
        optimizer.zero_grad(set_to_none=True)
        top1, top3 = compute_accuracy_topk(policy_logits, moves, topk=(1, 3))
        entropy = compute_policy_entropy(policy_logits)
        return (
            float(total_loss.item()),
            float(policy_loss.item()),
            float(ce_wdl.item()),
            top1,
            top3,
            entropy,
        )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    top1, top3 = compute_accuracy_topk(policy_logits, moves, topk=(1, 3))
    entropy = compute_policy_entropy(policy_logits)

    return (
        float(total_loss.item()),
        float(policy_loss.item()),
        float(ce_wdl.item()),
        top1,
        top3,
        entropy,
    )


def evaluate_opening_policy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate policy accuracy on held-out opening validation set."""
    model.eval()
    total_samples = 0
    sum_policy_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            planes = batch["planes"].to(
                device, memory_format=torch.channels_last, non_blocking=True
            )
            moves = batch["move"].to(device, non_blocking=True)
            batch_sz = moves.size(0)

            autocast_context = (
                torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16
                )
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                policy_logits, _ = model(planes)
                if "policy_target" in batch and batch["policy_target"] is not None:
                    policy_target = batch["policy_target"].to(device, non_blocking=True)
                    log_policy = F.log_softmax(policy_logits, dim=-1)
                    p_loss = -(policy_target * log_policy).sum(dim=-1).sum()
                else:
                    p_loss = F.cross_entropy(policy_logits, moves, reduction="sum")

            sum_policy_loss += float(p_loss.item())
            top1, top3 = compute_accuracy_topk(policy_logits, moves, topk=(1, 3))
            # Accumulate float correct counts directly (fix truncation bias) rather
            # than reconstructing from float percentages then truncating.
            # compute_accuracy_topk returns percentages; we reverse-compute counts.
            correct_top1_float = top1 * batch_sz / 100.0
            correct_top3_float = top3 * batch_sz / 100.0
            correct_top1 += correct_top1_float
            correct_top3 += correct_top3_float
            total_samples += batch_sz

    if total_samples == 0:
        return {"val_policy_loss": 0.0, "val_top1": 0.0, "val_top3": 0.0}

    return {
        "val_policy_loss": sum_policy_loss / total_samples,
        "val_top1": (int(round(correct_top1)) / total_samples) * 100.0,
        "val_top3": (int(round(correct_top3)) / total_samples) * 100.0,
        "val_samples": total_samples,
    }


def train_opening_policy(
    opening_data_dir: Union[str, Path],
    general_data_dir: Optional[Union[str, Path]] = None,
    extra_data_dirs: Optional[List[Union[str, Path]]] = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    output_checkpoint_dir: Union[str, Path] = "models/checkpoints_opening",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 0.0001,
    mixing_ratio_opening: float = 0.7,
    max_steps: Optional[int] = None,
    num_blocks: int = 12,
    num_filters: int = 128,
    seed: int = 42,
    device_name: Optional[str] = None,
    allow_uncertified_data: bool = False,
    repo_root: Optional[Union[str, Path]] = None,
    allow_superseded_resume: bool = False,
    allow_resume_config_drift: bool = False,
    allow_missing_run_manifest: bool = False,
) -> Dict[str, Any]:
    """Execute opening policy fine-tuning.

    Resume provenance (audit SA3-03): ``resume_checkpoint=None`` resolves the
    registry-designated, verified baseline fail-closed (no hardcoded fallback
    artifact). An explicit path is the operator's choice, but its registry
    standing is validated before any load, and resuming from an artifact a
    previous opening-policy run produced requires that run's manifest to
    match this invocation's dataset identity, architecture, and
    configuration. The ``allow_*`` flags keep each refusal explicitly
    reachable; every override is logged.
    """
    setup_environment()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from python.train.opening_run_manifest import (
        build_run_manifest,
        finalize_run_manifest,
        receipt_files_digest,
        read_run_manifest,
        require_manifest_for_resume_artifact,
        resolve_resume_checkpoint,
        validate_manifest_for_resume,
        validate_resume_provenance,
        write_run_manifest,
    )

    root = Path(repo_root) if repo_root is not None else Path(REPO_ROOT)

    device = torch.device(resolve_device(device_name))
    output_dir = Path(output_checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SA3-03 step 1: resolve the resume checkpoint to an exact identity and
    # validate its registry standing BEFORE any load path runs. This closes
    # the old hole where the raw torch.load fallback bypassed provenance.
    provenance = resolve_resume_checkpoint(resume_checkpoint, repo_root=root)
    # Both gates always run: an unregistered-path warning (or a superseded
    # override) must not short-circuit the missing-manifest gate, or a
    # manifest-less trainer-output artifact would resume without its own
    # explicit --allow-missing-run-manifest override.
    provenance_warnings = [
        validate_resume_provenance(provenance, allow_superseded=allow_superseded_resume),
        require_manifest_for_resume_artifact(
            provenance.path, allow_missing=allow_missing_run_manifest
        ),
    ]
    resume_warning = "; ".join(w for w in provenance_warnings if w) or None
    if resume_warning:
        logger.warning("%s", resume_warning)

    # Safety check: Never overwrite original baseline
    res_path = provenance.path
    target_best = (output_dir / "best_opening_model.pt").resolve()
    if target_best == res_path:
        raise ValueError("Output checkpoint cannot be identical to resume_checkpoint")

    # Load base model
    try:
        model, model_info = load_engine_model(provenance.path, device=str(device))
    except Exception as e:
        logger.info("Direct checkpoint load fallback: %s", e)
        model = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
        ckpt = torch.load(provenance.path, map_location=device, weights_only=True)
        st = ckpt.get("model", ckpt)
        model.load_state_dict(st)
    model.train()

    # DI-21: Require validated dataset before consumption
    from python.data.dataset_validation import require_validated_dataset

    # SA3-03 step 2: bind every consumed split's identity (resolved directory
    # + tamper-evident validation-receipt digest) into the run manifest, so a
    # later resume can prove it continues the same data.
    dataset_identity: Dict[str, Dict[str, Any]] = {}

    def _gated(split_dir: Path, label: str) -> None:
        gate = require_validated_dataset(
            split_dir,
            allow_uncertified=allow_uncertified_data,
            consumer=f"opening-policy-trainer[{label}]",
        )
        try:
            resolved_dir = str(Path(split_dir).resolve())
        except OSError:
            resolved_dir = str(split_dir)
        dataset_identity[label] = {
            "data_dir": resolved_dir,
            "certified": bool(gate.certified),
            "receipt_files_digest": receipt_files_digest(gate.receipt),
        }

    # Load opening datasets (and optional extra datasets such as tactical puzzles or Syzygy endgames)
    train_datasets = []
    val_datasets = []

    op_train_dir = Path(opening_data_dir) / "train"
    op_val_dir = Path(opening_data_dir) / "val"
    if op_train_dir.is_dir():
        _gated(op_train_dir, "train")
        train_datasets.append(BinaryChessDataset(op_train_dir))
    if op_val_dir.is_dir():
        _gated(op_val_dir, "val")
        val_datasets.append(BinaryChessDataset(op_val_dir))

    if extra_data_dirs:
        for ed_idx, ed in enumerate(extra_data_dirs):
            if ed:
                ed_path = Path(ed)
                if (ed_path / "train").is_dir():
                    _gated(ed_path / "train", f"extra_train[{ed_idx}]")
                    train_datasets.append(BinaryChessDataset(ed_path / "train"))
                if (ed_path / "val").is_dir():
                    _gated(ed_path / "val", f"extra_val[{ed_idx}]")
                    val_datasets.append(BinaryChessDataset(ed_path / "val"))

    if not train_datasets:
        raise ValueError(f"No valid train datasets found in {opening_data_dir}")

    # Combine opening and extra datasets using SourceAwareCombinedDataset with weights
    if len(train_datasets) > 1:
        master_train_ds = train_datasets[0]
        extra_train_list = train_datasets[1:]
        if len(extra_train_list) == 1:
            extra_train_ds = extra_train_list[0]
        else:
            extra_train_ds = torch.utils.data.ConcatDataset(extra_train_list)
        total_master_train = len(master_train_ds)
        total_extra_train = len(extra_train_ds)
        tactics_ratio_train = (
            total_extra_train / (total_master_train + total_extra_train)
            if (total_master_train + total_extra_train) > 0
            else 0.3
        )
        tactics_ratio_train = max(0.0, min(1.0, float(tactics_ratio_train)))
        op_train_ds = SourceAwareCombinedDataset(
            master_train_ds, extra_train_ds, tactics_ratio=tactics_ratio_train
        )
    else:
        op_train_ds = train_datasets[0]

    if not val_datasets:
        raise ValueError(
            f"No validation dataset found in {opening_data_dir}/val; opening policy fine-tuning requires a separate val set."
        )
    if len(val_datasets) > 1:
        master_val_ds = val_datasets[0]
        extra_val_list = val_datasets[1:]
        if len(extra_val_list) == 1:
            extra_val_ds = extra_val_list[0]
        else:
            extra_val_ds = torch.utils.data.ConcatDataset(extra_val_list)
        total_master_val = len(master_val_ds)
        total_extra_val = len(extra_val_ds)
        tactics_ratio_val = (
            total_extra_val / (total_master_val + total_extra_val)
            if (total_master_val + total_extra_val) > 0
            else 0.3
        )
        tactics_ratio_val = max(0.0, min(1.0, float(tactics_ratio_val)))
        op_val_ds = SourceAwareCombinedDataset(
            master_val_ds, extra_val_ds, tactics_ratio=tactics_ratio_val
        )
    else:
        op_val_ds = val_datasets[0]

    if isinstance(op_train_ds, SourceAwareCombinedDataset):
        total_train_samples = len(op_train_ds.master_ds) + len(op_train_ds.tactics_ds)
        op_train_loader = DataLoader(
            op_train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=op_train_ds.get_sampler(num_samples=total_train_samples),
            drop_last=False,
        )
    else:
        op_train_loader = DataLoader(
            op_train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )

    if isinstance(op_val_ds, SourceAwareCombinedDataset):
        total_val_samples = len(op_val_ds.master_ds) + len(op_val_ds.tactics_ds)
        op_val_loader = DataLoader(
            op_val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=op_val_ds.get_sampler(num_samples=total_val_samples),
        )
    else:
        op_val_loader = DataLoader(op_val_ds, batch_size=batch_size, shuffle=False)

    # General dataset if mixed training enabled
    gen_train_loader = None
    if general_data_dir and (Path(general_data_dir) / "train").is_dir():
        gen_train_path = Path(general_data_dir) / "train"
        _gated(gen_train_path, "general_train")
        gen_train_ds = BinaryChessDataset(gen_train_path)
        gen_train_loader = DataLoader(
            gen_train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
        gen_iter = iter(gen_train_loader)
        logger.info("Mixed training active: %d general samples loaded", len(gen_train_ds))

    # SA3-03 step 3: resuming from a previous opening-policy run's artifact
    # must match that run's manifest (dataset identity, architecture,
    # configuration) before any output of this run is written.
    config_record: Dict[str, Any] = {
        "opening_data_dir": str(Path(opening_data_dir).resolve()),
        "general_data_dir": (str(Path(general_data_dir).resolve()) if general_data_dir else None),
        "extra_data_dirs": (
            [str(Path(d).resolve()) for d in extra_data_dirs if d] if extra_data_dirs else []
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "mixing_ratio_opening": mixing_ratio_opening,
        "max_steps": max_steps,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
        "seed": seed,
        "device_name": device_name,
        "allow_uncertified_data": allow_uncertified_data,
    }
    previous_manifest = read_run_manifest(provenance.path.parent)
    if previous_manifest is not None:
        manifest_warning = validate_manifest_for_resume(
            previous_manifest,
            manifest_dir=provenance.path.parent,
            dataset_identity=dataset_identity,
            config=config_record,
            resume_provenance=provenance,
            allow_config_drift=allow_resume_config_drift,
            allow_superseded=allow_superseded_resume,
        )
        if manifest_warning:
            logger.warning("%s", manifest_warning)

    # SA3-03 step 4: persist the run manifest (status "running"). Written only
    # after every validation gate passed, so an invalid resume never rotates
    # away the previous run's provenance record.
    from python.data.dataset_contract import detect_git_commit

    manifest_path = write_run_manifest(
        output_dir,
        build_run_manifest(
            config=config_record,
            dataset_identity=dataset_identity,
            resume_provenance=provenance,
            output_dir=output_dir,
            git_commit=detect_git_commit(),
        ),
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = (len(op_train_loader) * epochs) if max_steps is None else max_steps
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=min(50, total_steps // 10), total_steps=total_steps, base_lr=lr
    )

    logger.info(
        "Starting opening policy training: %d samples, %d total steps on %s",
        len(op_train_ds),
        total_steps,
        device,
    )

    step = 0
    best_val_loss = float("inf")
    metrics_history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        for batch in op_train_loader:
            if max_steps is not None and step >= max_steps:
                break

            # If mixing with general data, occasionally train on general batch
            if gen_train_loader and random.random() > mixing_ratio_opening:
                try:
                    mix_batch = next(gen_iter)
                except StopIteration:
                    gen_iter = iter(gen_train_loader)
                    mix_batch = next(gen_iter)
                batch_to_use = mix_batch
            else:
                batch_to_use = batch

            tot_l, pol_l, wdl_l, top1, top3, entropy = train_opening_step(
                model=model,
                batch=batch_to_use,
                optimizer=optimizer,
                device=device,
                use_autocast=(device.type == "cuda"),
            )
            step += 1
            scheduler.step()

            if step % 25 == 0 or step == total_steps:
                logger.info(
                    "Step %d/%d: loss=%.4f (pol=%.4f, wdl=%.4f), top1=%.1f%%, top3=%.1f%%",
                    step,
                    total_steps,
                    tot_l,
                    pol_l,
                    wdl_l,
                    top1,
                    top3,
                )

        # Validation at epoch end
        val_res = evaluate_opening_policy(model, op_val_loader, device=device)
        logger.info(
            "Epoch %d Val: pol_loss=%.4f, top1=%.1f%%, top3=%.1f%%",
            epoch,
            val_res["val_policy_loss"],
            val_res["val_top1"],
            val_res["val_top3"],
        )

        ckpt_state = {
            "checkpoint_type": "opening_policy",
            "weights_only_container": True,
            "model": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_res["val_policy_loss"],
            "val_top1": val_res["val_top1"],
            "num_blocks": num_blocks,
            "num_filters": num_filters,
        }
        _atomic_torch_save(ckpt_state, output_dir / f"checkpoint_epoch_{epoch}.pt")

        if val_res["val_policy_loss"] < best_val_loss:
            best_val_loss = val_res["val_policy_loss"]
            _atomic_torch_save(ckpt_state, target_best)
            logger.info(
                "New best opening model saved to %s (val_loss=%.4f)", target_best, best_val_loss
            )

        metrics_history.append({"epoch": epoch, "step": step, **val_res})

        if max_steps is not None and step >= max_steps:
            break

    # Save final model
    final_path = output_dir / "latest_opening_model.pt"
    _atomic_torch_save(
        {
            "checkpoint_type": "opening_policy",
            "weights_only_container": True,
            "model": model.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "num_blocks": num_blocks,
            "num_filters": num_filters,
        },
        final_path,
    )

    # SA3-03 step 5: finalize the manifest with the SHA-256 of every
    # generated checkpoint, so a later resume can prove the artifact it loads
    # is the one this run produced.
    finalize_run_manifest(output_dir)

    return {
        "final_step": step,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(target_best if target_best.exists() else final_path),
        "history": metrics_history,
        "run_manifest": str(manifest_path),
    }
