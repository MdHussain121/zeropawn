"""Training runner subcommand for ZeroPawn Phase 4 RL selfplay & optimization."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

from python.cli.common import (
    REPO_ROOT,
    get_default_checkpoint_path,
    resolve_device,
    resolve_stockfish,
    safe_resolve,
)
from python.common.progress import (
    artifacts_list,
    banner,
    emit_error,
    emit_report,
    error_block,
    file_record,
    format_kv_line,
    header_block,
    parameters_table,
    run_context,
    summary_block,
)
import logging

logger = logging.getLogger(__name__)


class ExplicitTrackingAction(argparse.Action):
    """Custom argparse Action that records which arguments were explicitly supplied."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        explicit = getattr(namespace, "_explicit_args", None)
        if explicit is None:
            explicit = set()
            setattr(namespace, "_explicit_args", explicit)
        explicit.add(self.dest)


class ExplicitTrackingStoreTrueAction(argparse._StoreTrueAction):
    """Custom argparse Action for boolean flags that records explicit supply."""

    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        explicit = getattr(namespace, "_explicit_args", None)
        if explicit is None:
            explicit = set()
            setattr(namespace, "_explicit_args", explicit)
        explicit.add(self.dest)


def _add_supervised_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auto",
        "--auto-settings",
        action=ExplicitTrackingStoreTrueAction,
        default=False,
        help="Use recommended auto settings (batch 256, LR 0.001, GPU-resident, warmup 500)",
    )
    parser.add_argument(
        "--train-dir",
        "--data-dir",
        action=ExplicitTrackingAction,
        type=str,
        default=None,
        help="Path to training dataset chunks (default: auto-detect data/dataset_scaled/train)",
    )
    parser.add_argument(
        "--val-dir",
        action=ExplicitTrackingAction,
        type=str,
        default=None,
        help="Path to validation dataset chunks (default: auto-detect data/dataset_scaled/val)",
    )
    parser.add_argument(
        "--epochs",
        action=ExplicitTrackingAction,
        type=int,
        default=5,
        help="Number of supervised training epochs (default: 5)",
    )
    parser.add_argument(
        "--steps",
        "--max-steps",
        dest="max_steps",
        action=ExplicitTrackingAction,
        type=int,
        default=None,
        help="Maximum training steps across all epochs (default: None, full epochs)",
    )
    parser.add_argument(
        "--batch-size",
        action=ExplicitTrackingAction,
        type=int,
        default=512,
        help="Training effective batch size (default: 512)",
    )
    parser.add_argument(
        "--micro-batch-size",
        action=ExplicitTrackingAction,
        type=int,
        default=None,
        help="Physical micro-batch size for DataLoader forward/backward passes",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        action=ExplicitTrackingAction,
        type=int,
        default=None,
        help="Number of gradient accumulation steps before optimizer update",
    )
    parser.add_argument(
        "--lr",
        action=ExplicitTrackingAction,
        type=float,
        default=0.0002,
        help="Base learning rate for AdamW (default: 0.0002)",
    )
    parser.add_argument(
        "--warmup-steps",
        action=ExplicitTrackingAction,
        type=int,
        default=500,
        help="Linear warmup steps (default: 500)",
    )

    parser.add_argument(
        "--no-gpu-resident",
        action=ExplicitTrackingStoreTrueAction,
        default=False,
        help="Disable GPU VRAM caching of dataset and use standard CPU DataLoader",
    )
    parser.add_argument(
        "--workers",
        action=ExplicitTrackingAction,
        type=int,
        default=2,
        help="DataLoader worker processes when not GPU-resident (default: 2)",
    )
    parser.add_argument(
        "--num-blocks",
        action=ExplicitTrackingAction,
        type=int,
        default=12,
        help="Number of ResNet residual blocks (default: 12)",
    )
    parser.add_argument(
        "--num-filters",
        action=ExplicitTrackingAction,
        type=int,
        default=128,
        help="Number of convolutional filters (default: 128)",
    )
    parser.add_argument(
        "--tactics-ratio",
        "--source-ratio",
        dest="tactics_ratio",
        action=ExplicitTrackingAction,
        type=float,
        default=None,
        help="Target ratio for tactical puzzle positions in training (0.0 to 1.0, default: natural composition)",
    )
    parser.add_argument(
        "--tactics-dir",
        action=ExplicitTrackingAction,
        type=str,
        default=None,
        help="Path to dedicated tactical dataset chunks (default: None)",
    )
    try:
        parser.add_argument(
            "--checkpoint",
            "--resume",
            "-c",
            action=ExplicitTrackingAction,
            type=str,
            default=None,
            help="Path to initial weights or checkpoint to warm-start or resume from",
        )
    except argparse.ArgumentError:
        pass
    parser.add_argument(
        "--seed",
        action=ExplicitTrackingAction,
        type=int,
        default=None,
        help="RNG seed for reproducible runs (default: None = unseeded)",
    )


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        aliases=["tr", "selfplay"],
        help="Continue or start RL self-play or supervised training loop.",
        description="Launch continuous reinforcement learning self-play or supervised learning warm-start.",
    )
    parser.add_argument(
        "--mode",
        choices=["rl", "supervised"],
        default="rl",
        help="Training mode: 'rl' (Reinforcement learning self-play, default) or 'supervised' (Supervised warm-start)",
    )
    parser.add_argument(
        "--supervised",
        "-s",
        action="store_true",
        help="Shortcut to run supervised learning warm-start mode",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        "-c",
        type=str,
        default=None,
        help="Path to initial weights or checkpoint to start from (default: models/checkpoints_scaled/best_model.pt)",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        default=False,
        help="Initialize weights randomly for pure reinforcement learning without warm-start",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing RL checkpoint to resume training state, step, and optimizer from",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=72.0,
        help="Maximum training duration in hours (default: 72.0)",
    )
    parser.add_argument(
        "--eval-interval-sec",
        type=float,
        default=21600.0,
        help="Evaluation interval in seconds (default: 21600s = 6 hours)",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=48,
        help="Number of concurrent self-play games (default: 48)",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=500_000,
        help="Replay buffer capacity in positions (default: 500,000)",
    )
    parser.add_argument(
        "--cleanup-buffer-on-exit",
        action="store_true",
        default=False,
        help="Delete the preallocated replay buffer (*.bin) after final checkpoint is saved",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep newest N periodic checkpoints, prune older (default: 3)",
    )
    parser.add_argument(
        "--keep-tfevents",
        type=int,
        default=1,
        help="Keep newest N TensorBoard event files per log dir (default: 1)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints_rl",
        help="Directory to save periodic and final checkpoints (default: models/checkpoints_rl)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/phase4_rl",
        help="Directory for tensorboard runs and telemetry CSV (default: runs/phase4_rl)",
    )
    parser.add_argument(
        "--verify",
        "--test-verify",
        action="store_true",
        help="Run quick verification mode (30s for RL, 10 steps for Supervised)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference and training on (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--train-device",
        type=str,
        default=None,
        help="Device to run RL neural network training on (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--infer-device",
        type=str,
        default=None,
        help="Device to run self-play inference on (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Run bounded hardware benchmark & latency calibration (<=30s) before training",
    )
    parser.add_argument(
        "--auto-hardware",
        action="store_true",
        default=False,
        help="Auto-configure batch size, concurrency, and devices using hardware heuristics",
    )
    # ── Task 5: Recommended launch parameters ───────────────────────────────
    parser.add_argument(
        "--tablebase-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to Syzygy tablebase directory.  Prevents 300+ ply draw drifts "
            "that inflate the MCTS cache.  Example: data/syzygy  (default: disabled)"
        ),
    )
    parser.add_argument(
        "--tablebase-max-pieces",
        type=int,
        default=5,
        metavar="N",
        help="Maximum total pieces for tablebase adjudication (default: 5, requires --tablebase-path)",
    )
    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=1.0,
        metavar="MS",
        help=(
            "Dispatcher timeout for the inference server in milliseconds.  "
            "1.0 ms gives optimal throughput for 48-game concurrency (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--watchdog-rss-gb",
        type=float,
        default=7.5,
        metavar="GB",
        help=(
            "Process RSS abort ceiling in GB.  Training is halted if the process "
            "exceeds this physical memory limit (default: 7.5)"
        ),
    )
    parser.add_argument(
        "--watchdog-free-ram-gb",
        type=float,
        default=2.0,
        metavar="GB",
        help=(
            "System free-RAM floor in GB.  Training is halted if system free RAM "
            "drops below this threshold (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--full-sims",
        type=int,
        default=800,
        help="Full MCTS search simulations for training games (default: 800)",
    )
    parser.add_argument(
        "--fast-sims",
        type=int,
        default=150,
        help="Fast MCTS search simulations for training games (default: 150)",
    )
    parser.add_argument(
        "--gpu-cache-capacity",
        type=int,
        default=160_000,
        help="GPU VRAM replay cache capacity (default: 160,000)",
    )
    parser.add_argument(
        "--human-cache-size",
        type=int,
        default=80_000,
        help="Human dataset active sample cached in VRAM (default: 80,000)",
    )
    parser.add_argument(
        "--min-buffer-fill",
        type=int,
        default=0,
        help="Minimum self-play positions in replay buffer before training begins (default: 0)",
    )
    parser.add_argument(
        "--max-reuse-ratio",
        type=int,
        default=16,
        help="Maximum sample reuse ratio for pacing training steps (default: 16)",
    )
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        default=False,
        help="Skip baseline evaluation cycle on startup and proceed directly to self-play",
    )
    parser.add_argument(
        "--reset-buffer",
        action="store_true",
        default=False,
        help="Start with a fresh empty replay buffer and warm-start checkpoint",
    )
    parser.add_argument(
        "--stockfish-nodes",
        type=int,
        default=1000,
        help="Stockfish evaluation nodes per move during periodic benchmarks (default: 1000)",
    )
    parser.add_argument(
        "--policy-target-temp",
        type=float,
        default=0.5,
        help="Temperature for sharpening policy targets on full-search rows (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.001,
        help="Policy entropy bonus coefficient in total loss (default: 0.001)",
    )
    # ────────────────────────────────────────────────────────────────────────
    _add_supervised_arguments(parser)
    parser.set_defaults(func=run_training)

    train_subparsers = parser.add_subparsers(dest="train_subcommand")
    c4_parser = train_subparsers.add_parser(
        "connect4",
        help="Train Connect 4 baseline neural network for architecture validation.",
    )
    c4_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (default: 5)",
    )
    c4_parser.add_argument(
        "--games",
        "--num-games",
        dest="num_games",
        type=int,
        default=10,
        help="Self-play games (default: 10)",
    )
    c4_parser.add_argument(
        "--sims",
        type=int,
        default=25,
        help="MCTS simulations per move (default: 25)",
    )
    c4_parser.add_argument(
        "--save-path",
        type=str,
        default="models/connect4_model.pt",
        help="Output model path (default: models/connect4_model.pt)",
    )
    c4_parser.set_defaults(func=run_train_connect4)

    # Dedicated 'supervised' subcommand
    sup_parser = subparsers.add_parser(
        "supervised",
        aliases=["train-supervised", "sl"],
        help="Train supervised warm-start model from human PGN dataset.",
        description="Train policy and value heads on binary dataset chunks before starting RL.",
    )
    _add_supervised_arguments(sup_parser)
    sup_parser.add_argument(
        "--checkpoint-dir",
        action=ExplicitTrackingAction,
        type=str,
        default="models/checkpoints_scaled",
        help="Directory to save checkpoints (default: models/checkpoints_scaled)",
    )
    sup_parser.add_argument(
        "--log-dir",
        action=ExplicitTrackingAction,
        type=str,
        default="runs/supervised",
        help="Directory for TensorBoard telemetry (default: runs/supervised)",
    )
    sup_parser.add_argument(
        "--verify",
        "--test-verify",
        action=ExplicitTrackingStoreTrueAction,
        default=False,
        help="Run quick 10-step verification mode",
    )
    sup_parser.add_argument(
        "--device",
        action=ExplicitTrackingAction,
        type=str,
        default=None,
        help="Device to train on (default: auto cuda/cpu)",
    )
    sup_parser.set_defaults(func=run_supervised_training)


def _auto_detect_data_dir(split_name: str) -> Optional[Path]:
    candidates = [
        REPO_ROOT / "data" / "dataset_training",
        REPO_ROOT / "data" / "dataset_scaled",
        REPO_ROOT / "data" / "dataset",
        REPO_ROOT / "data" / "dataset_elite2500",
        REPO_ROOT / "data" / "dataset_tactics",
    ]
    data_dir = REPO_ROOT / "data"
    if data_dir.is_dir():
        for d in sorted(data_dir.glob("dataset_*")):
            if d not in candidates:
                candidates.append(d)

    for base in candidates:
        cand = base / split_name
        if cand.is_dir() and (cand / "metadata.json").is_file():
            return cand
    return None


def run_supervised_training(args: argparse.Namespace) -> int:
    import json
    from python.train.train_supervised import train_supervised

    ctx = run_context("warm-start")
    t0 = time.time()
    log_path = "logs/zeropawn.log"

    def _fail(what: str, likely: str, try_: str) -> int:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        return 1

    emit_report(banner("warm-start", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "auto",
            str(getattr(args, "seed", None)),
        )
    )

    # 1. Resolve train_dir
    train_dir_input = getattr(args, "train_dir", None) or getattr(args, "data_dir", None)
    if train_dir_input:
        try:
            train_dir = safe_resolve(train_dir_input)
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid --train-dir path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative dataset dir, then see LOG",
            )
        if not train_dir.is_dir():
            return _fail(
                f"Train directory does not exist: {train_dir}",
                "wrong path or dataset not built yet",
                "run data-convert first or fix --train-dir, then see LOG",
            )
    else:
        train_dir = _auto_detect_data_dir("train")
        if not train_dir:
            return _fail(
                "No training dataset found in data/dataset_training/train or data/dataset_scaled/train.",
                "dataset not built yet",
                "run data-convert or pass --train-dir, then see LOG",
            )

    # 2. Resolve val_dir
    val_dir_input = getattr(args, "val_dir", None)
    if val_dir_input:
        try:
            val_dir = safe_resolve(val_dir_input)
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid --val-dir path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative dataset dir, then see LOG",
            )
        if not val_dir.is_dir():
            return _fail(
                f"Validation directory does not exist: {val_dir}",
                "wrong path or dataset not built yet",
                "run data-convert first or fix --val-dir, then see LOG",
            )
    else:
        val_dir = _auto_detect_data_dir("val")
        if not val_dir:
            return _fail(
                "No validation dataset found in data/dataset_training/val or data/dataset_scaled/val.",
                "dataset not built yet",
                "run data-convert or pass --val-dir, then see LOG",
            )

    # 3. Numeric validation and resolution
    is_auto = getattr(args, "auto", False)

    def _is_explicitly_supplied(dest: str, default_val: Any) -> bool:
        """Detect whether an option was explicitly provided by user rather than defaulted.

        Checks (in priority order):
        1. Direct provenance recorded on namespace by ExplicitTrackingAction.
        2. Programmatic or CLI raw argv list attached to namespace as _raw_argv.
        3. ResolvedConfiguration provenance records from 5-tier config system.
        4. Fallback for raw Namespaces constructed in tests: compare against known default.
        """
        explicit = getattr(args, "_explicit_args", None)
        if explicit is not None:
            return dest in explicit

        raw_argv = getattr(args, "_raw_argv", None)
        if raw_argv is not None:
            dest_flag = "--" + dest.replace("_", "-")
            for token in raw_argv:
                if token == dest_flag or token.startswith(dest_flag + "="):
                    return True
                if dest == "batch_size" and (token == "-b" or token.startswith("-b=")):
                    return True
                if dest == "lr" and (token == "--lr" or token.startswith("--lr=")):
                    return True
                if dest == "warmup_steps" and (
                    token == "--warmup-steps" or token.startswith("--warmup-steps=")
                ):
                    return True
                if dest == "no_gpu_resident" and token == "--no-gpu-resident":
                    return True
            return False

        resolved = getattr(args, "_resolved_config", None)
        if resolved and hasattr(resolved, "records") and dest in resolved.records:
            return resolved.records[dest].source == "cli"

        if hasattr(args, dest):
            val = getattr(args, dest)
            if default_val is not None:
                return val != default_val
            return val is not None
        return False

    user_batch = _is_explicitly_supplied("batch_size", 512)
    user_micro = _is_explicitly_supplied("micro_batch_size", None)
    user_accum = _is_explicitly_supplied("gradient_accumulation_steps", None)
    user_lr = _is_explicitly_supplied("lr", 0.0002)
    user_warmup = _is_explicitly_supplied("warmup_steps", 500)
    user_gpu = _is_explicitly_supplied("no_gpu_resident", False)

    try:
        import torch

        cuda_safe = torch.cuda.is_available() and getattr(args, "device", None) != "cpu"
    except ImportError:
        cuda_safe = False

    if is_auto:
        batch_size = getattr(args, "batch_size", 512) if user_batch else 256
        micro_batch_size = (
            getattr(args, "micro_batch_size", None) if user_micro else min(64, batch_size)
        )
        gradient_accumulation_steps = (
            getattr(args, "gradient_accumulation_steps", None)
            if user_accum
            else max(1, batch_size // (micro_batch_size or 64))
        )
        lr = getattr(args, "lr", 0.0002) if user_lr else 0.001
        warmup_steps = getattr(args, "warmup_steps", 500) if user_warmup else 500
        gpu_resident = False if getattr(args, "no_gpu_resident", False) else cuda_safe
    else:
        batch_size = getattr(args, "batch_size", 512)
        micro_batch_size = getattr(args, "micro_batch_size", None)
        gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", None)
        lr = getattr(args, "lr", 0.0002)
        warmup_steps = getattr(args, "warmup_steps", 500)
        gpu_resident = False if getattr(args, "no_gpu_resident", False) else cuda_safe

    batch_origin = "user" if user_batch else ("auto" if is_auto else "default")
    micro_origin = "user" if user_micro else ("auto" if is_auto else "default")
    accum_origin = "user" if user_accum else ("auto" if is_auto else "default")
    lr_origin = "user" if user_lr else ("auto" if is_auto else "default")
    gpu_origin = "user" if user_gpu else ("auto" if is_auto else "default")

    try:
        batch_size_int = int(batch_size)
    except (ValueError, TypeError):
        return _fail(
            f"--batch-size must be an integer > 0, got '{batch_size}'",
            "invalid batch size type",
            "pass --batch-size 512 or larger, then see LOG",
        )
    if batch_size_int <= 0:
        return _fail(
            "--batch-size must be > 0.",
            "non-positive batch size",
            "pass --batch-size 8 or larger, then see LOG",
        )
    batch_size = batch_size_int
    if micro_batch_size is not None and micro_batch_size <= 0:
        return _fail(
            "--micro-batch-size must be > 0.",
            "non-positive micro-batch size",
            "pass a positive micro-batch size, then see LOG",
        )
    if gradient_accumulation_steps is not None and gradient_accumulation_steps <= 0:
        return _fail(
            "--gradient-accumulation-steps must be > 0.",
            "non-positive accumulation steps",
            "pass a positive accumulation step count, then see LOG",
        )
    if lr <= 0:
        return _fail(
            "--lr must be > 0.", "non-positive learning rate", "pass a positive --lr, then see LOG"
        )
    epochs = getattr(args, "epochs", 5)
    if epochs <= 0:
        return _fail(
            "--epochs must be > 0.",
            "non-positive epoch count",
            "pass --epochs 1 or larger, then see LOG",
        )
    num_blocks = getattr(args, "num_blocks", 12)
    num_filters = getattr(args, "num_filters", 128)

    # 4. Resolve checkpoint_dir and log_dir
    iso_tmp_dir: Optional[Path] = None
    if getattr(args, "verify", False):
        iso_tmp_dir = REPO_ROOT / "tmp" / f"verify_supervised_{int(time.time() * 1000)}"
        ckpt_dir_path = iso_tmp_dir / "checkpoints"
        log_dir_path = iso_tmp_dir / "runs"
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        raw_ckpt_dir = None
        raw_log_dir = None
    else:
        raw_ckpt_dir = getattr(args, "checkpoint_dir", None)
        if not raw_ckpt_dir or raw_ckpt_dir == "models/checkpoints_rl":
            ckpt_dir_path = REPO_ROOT / "models" / "checkpoints_scaled"
        else:
            try:
                ckpt_dir_path = safe_resolve(raw_ckpt_dir)
            except (ValueError, OSError) as e:
                return _fail(
                    f"Invalid --checkpoint-dir: {e}",
                    "path escapes repo or is malformed",
                    "pass a repo-relative dir, then see LOG",
                )
            if ckpt_dir_path.is_file() or str(ckpt_dir_path).endswith(".pt"):
                if not getattr(args, "checkpoint", None) and not getattr(args, "resume", None):
                    setattr(args, "checkpoint", str(ckpt_dir_path))
                ckpt_dir_path = ckpt_dir_path.parent

        raw_log_dir = getattr(args, "log_dir", None)
        if not raw_log_dir or raw_log_dir == "runs/phase4_rl":
            log_dir_path = REPO_ROOT / "runs" / "supervised"
        else:
            try:
                log_dir_path = safe_resolve(raw_log_dir)
            except (ValueError, OSError) as e:
                return _fail(
                    f"Invalid --log-dir: {e}",
                    "path escapes repo or is malformed",
                    "pass a repo-relative dir, then see LOG",
                )

    # 5. Device
    try:
        device_resolved = resolve_device(getattr(args, "device", None))
    except ValueError as e:
        return _fail(
            str(e), "unknown --device value", "pass --device cuda or --device cpu, then see LOG"
        )

    # 6. Verify mode vs standard run
    is_verify = getattr(args, "verify", False)
    max_steps = getattr(args, "max_steps", None)
    if is_verify:
        max_steps = 10
        warmup_steps = 2
        epochs = 1
        gpu_resident = False
        if not user_micro and not user_accum:
            batch_size = 512
            micro_batch_size = 64
            gradient_accumulation_steps = 8

    # 7. Check for resume checkpoint
    resume_path = getattr(args, "resume", None) or getattr(args, "checkpoint", None)
    if resume_path:
        try:
            resume_resolved = safe_resolve(resume_path)
            if not resume_resolved.is_file():
                return _fail(
                    f"Resume checkpoint does not exist: {resume_resolved}",
                    "wrong path or nothing saved yet",
                    "fix the path or train without --resume, then see LOG",
                )
            resume_path = str(resume_resolved)
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid resume checkpoint path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative .pt path, then see LOG",
            )

    # 8. Read sample counts from metadata if available
    train_samples_str = "Unknown"
    val_samples_str = "Unknown"
    try:
        with open(train_dir / "metadata.json", "r", encoding="utf-8") as f:
            t_meta = json.load(f)
            train_samples_str = f"{t_meta.get('num_samples', 0):,} samples"
    except Exception:
        pass
    try:
        with open(val_dir / "metadata.json", "r", encoding="utf-8") as f:
            v_meta = json.load(f)
            val_samples_str = f"{v_meta.get('num_samples', 0):,} samples"
    except Exception:
        pass

    tactics_dir_input = getattr(args, "tactics_dir", None)
    tactics_dir: Optional[Path] = None
    if tactics_dir_input:
        try:
            tactics_dir = safe_resolve(tactics_dir_input)
            if not tactics_dir.is_dir():
                return _fail(
                    f"Tactics directory does not exist: {tactics_dir}",
                    "wrong path or dataset not built yet",
                    "fix --tactics-dir, then see LOG",
                )
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid --tactics-dir path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative dir, then see LOG",
            )

    if is_verify:
        gpu_resident = False
        workers = 0
    else:
        workers = getattr(args, "workers", 0 if sys.platform == "win32" else 2)

    # 9. Effective parameters (ASCII table replaces the Rich summary panel)
    seed = getattr(args, "seed", None)
    tactics_ratio = getattr(args, "tactics_ratio", None)
    if tactics_ratio is None:
        tactics_ratio = getattr(args, "source_ratio", None)

    emit_report(
        parameters_table(
            [
                (
                    "profile",
                    "Auto Settings (Recommended)" if is_auto else "Custom",
                    "user" if is_auto else "default",
                ),
                (
                    "train_dir",
                    f"{train_dir} ({train_samples_str})",
                    "user" if train_dir_input else "default",
                ),
                (
                    "val_dir",
                    f"{val_dir} ({val_samples_str})",
                    "user" if val_dir_input else "default",
                ),
                (
                    "architecture",
                    f"{num_blocks} ResBlocks, {num_filters} filters (channels-last) (M, ckpt-verified)",
                    "default",
                ),
                ("batch_size", batch_size, batch_origin),
                ("micro_batch_size", micro_batch_size, micro_origin),
                ("gradient_accumulation_steps", gradient_accumulation_steps, accum_origin),
                ("lr", f"{lr} (warmup {warmup_steps} steps)", lr_origin),
                (
                    "duration",
                    "10 steps (Verification Mode)" if is_verify else f"{epochs} epochs",
                    "default",
                ),
                (
                    "max_steps",
                    max_steps if (max_steps and not is_verify) else "none",
                    "user" if max_steps else "default",
                ),
                (
                    "gpu_resident",
                    "Disabled" if not gpu_resident else "Enabled (bfloat16 in VRAM)",
                    gpu_origin,
                ),
                ("checkpoint_dir", str(ckpt_dir_path), "user" if raw_ckpt_dir else "default"),
                ("tensorboard_log_dir", str(log_dir_path), "user" if raw_log_dir else "default"),
                (
                    "device",
                    f"{device_resolved} (M)",
                    "user" if getattr(args, "device", None) else "default",
                ),
                (
                    "resume",
                    str(resume_path) if resume_path else "none",
                    "user" if resume_path else "default",
                ),
                (
                    "tactics_ratio",
                    f"{tactics_ratio:.2f}"
                    if tactics_ratio is not None
                    else "natural (dataset composition)",
                    "user" if tactics_ratio is not None else "default",
                ),
                (
                    "tactics_dir",
                    str(tactics_dir) if tactics_dir else "none",
                    "user" if tactics_dir_input else "default",
                ),
                ("seed", seed, "user" if seed is not None else "default"),
            ]
        )
    )
    emit_report(format_kv_line([("note", "Ctrl+C pauses training and saves checkpoint")]))

    try:
        metrics = train_supervised(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            base_lr=lr,
            warmup_steps=warmup_steps,
            epochs=epochs,
            max_steps=max_steps,
            checkpoint_dir=str(ckpt_dir_path),
            log_dir=str(log_dir_path),
            resume_path=resume_path,
            num_workers=workers,
            gpu_resident=gpu_resident,
            num_blocks=num_blocks,
            num_filters=num_filters,
            tactics_ratio=tactics_ratio,
            tactics_dir=str(tactics_dir) if tactics_dir else None,
            seed=seed,
        )
        emit_report(
            format_kv_line(
                [
                    ("phase", "final"),
                    ("val_loss", f"{metrics.get('total_loss', float('nan')):.4f}"),
                    ("top1", f"{metrics.get('top1', float('nan')):.2f}"),
                ]
            )
        )
        records = []
        if ckpt_dir_path.is_dir():
            for path in sorted(ckpt_dir_path.glob("*.pt")):
                records.append(file_record(str(path)))
        records.append((f"tensorboard: {log_dir_path}", -1, None))
        emit_report(artifacts_list(records))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
        return 0

    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 0))
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        logger.debug("supervised training failed: %s", e)
        return _fail(
            f"Supervised training failed: {e}",
            "bad dataset, OOM/VRAM, or unreadable resume checkpoint",
            "check dataset dirs and free VRAM, then see LOG",
        )
    finally:
        if iso_tmp_dir and iso_tmp_dir.is_dir():
            try:
                import shutil

                shutil.rmtree(iso_tmp_dir, ignore_errors=True)
            except Exception:
                pass
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_train_connect4(args: argparse.Namespace) -> int:
    """Train Connect 4 baseline neural network."""
    t0 = time.time()
    ctx = run_context("train-connect4")
    epochs = getattr(args, "epochs", 5)
    num_games = getattr(args, "num_games", 10)
    sims = getattr(args, "sims", 25)
    save_path = getattr(args, "save_path", "models/connect4_model.pt")

    emit_report(banner("train-connect4", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "auto",
            "none",
        )
    )
    emit_report(
        parameters_table(
            [
                ("epochs", epochs, "user" if epochs != 5 else "default"),
                ("games", num_games, "user" if num_games != 10 else "default"),
                ("sims", sims, "user" if sims != 25 else "default"),
                ("save_path", save_path, "default"),
            ]
        )
    )

    try:
        from python.train.train_connect4 import train_connect4

        saved = train_connect4(
            num_games=num_games,
            sims=sims,
            epochs=epochs,
            save_path=save_path,
        )
        records = [file_record(saved)]
        emit_report(artifacts_list(records))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
        return 0
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 0))
        return 0
    except Exception as e:
        logger.exception("Connect4 training failed: %s", e)
        emit_error(
            error_block(
                f"Connect4 training failed: {e}",
                "runtime failure",
                "see trace above",
                "logs/zeropawn.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 0, 1))
        return 1


def run_training(args: argparse.Namespace) -> int:
    if getattr(args, "train_subcommand", None) == "connect4":
        return run_train_connect4(args)
    if getattr(args, "supervised", False) or getattr(args, "mode", "rl") == "supervised":
        return run_supervised_training(args)

    from scripts.run_phase4_rl import Phase4Orchestrator, SafetyTriggerException

    t0 = time.time()
    log_path = "logs/zeropawn.log"

    def _fail(what: str, likely: str, try_: str, code: int = 1) -> int:
        emit_error(error_block(what, likely, try_, log_path))
        emit_report(summary_block("FAILED", time.time() - t0, 0, code))
        return code

    # Validate initialization options for conflicts
    checkpoint_arg = getattr(args, "checkpoint", None)
    random_init_arg = getattr(args, "random_init", False)
    resume_arg = getattr(args, "resume", None)

    init_flags_count = sum(
        1 for f in [bool(checkpoint_arg), bool(random_init_arg), bool(resume_arg)] if f
    )
    if init_flags_count > 1:
        return _fail(
            "Conflicting initialization options: specify at most one of --checkpoint, --random-init, or --resume.",
            "multiple initialization flags supplied",
            "choose either --checkpoint <path>, --random-init, or --resume <path>, then see LOG",
        )

    base_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    init_desc = ""

    if random_init_arg:
        base_checkpoint = None
        resume_checkpoint = None
        init_desc = "Random Initialization (Pure RL from scratch)"
    elif resume_arg:
        try:
            resume_resolved = safe_resolve(resume_arg)
            if not resume_resolved.is_file():
                return _fail(
                    f"Resume checkpoint does not exist: {resume_resolved}",
                    "wrong path or nothing saved yet",
                    "pass an existing checkpoint to --resume, then see LOG",
                )
            resume_checkpoint = str(resume_resolved)
            base_checkpoint = None
            init_desc = f"Resume from checkpoint: {resume_resolved.name}"
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid --resume path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative .pt path, then see LOG",
            )
    elif checkpoint_arg:
        try:
            checkpoint_resolved = safe_resolve(checkpoint_arg)
            if not checkpoint_resolved.is_file():
                return _fail(
                    f"Checkpoint file does not exist: {checkpoint_resolved}",
                    "no pre-trained model at that path",
                    "train a warm-start via 'chess train --supervised' first, then see LOG",
                )
            base_checkpoint = str(checkpoint_resolved)
            resume_checkpoint = None
            init_desc = f"Warm-start checkpoint: {checkpoint_resolved.name}"
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid --checkpoint path: {e}",
                "path escapes repo or is malformed",
                "pass a repo-relative .pt path, then see LOG",
            )
    else:
        # Default protected supervised warm-start baseline
        default_ckpt = get_default_checkpoint_path() or safe_resolve(
            "models/checkpoints_scaled/best_model.pt"
        )
        try:
            checkpoint_resolved = safe_resolve(default_ckpt)
            if not checkpoint_resolved.is_file():
                return _fail(
                    f"Default warm-start checkpoint does not exist: {checkpoint_resolved}",
                    "protected baseline missing",
                    "verify models/checkpoints_scaled/best_model.pt exists, then see LOG",
                )
            base_checkpoint = str(checkpoint_resolved)
            resume_checkpoint = None
            init_desc = f"Default Supervised Warm-Start: {checkpoint_resolved.name}"
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid default checkpoint path: {e}",
                "path escapes repo or is malformed",
                "verify models/checkpoints_scaled/best_model.pt, then see LOG",
            )

    # Numeric validation
    if args.max_hours is not None and args.max_hours <= 0:
        return _fail(
            "--max-hours must be > 0.",
            "non-positive duration",
            "pass a positive --max-hours, then see LOG",
        )
    if args.eval_interval_sec is not None and args.eval_interval_sec < 60:
        return _fail(
            "--eval-interval-sec must be >= 60.",
            "evaluation interval too short",
            "pass --eval-interval-sec 60 or larger, then see LOG",
        )
    if args.concurrent_games is not None and not (1 <= args.concurrent_games <= 128):
        return _fail(
            "--concurrent-games must be in 1..128.",
            "concurrency out of range",
            "pass --concurrent-games 1..128, then see LOG",
        )
    if args.buffer_capacity is not None and not (10_000 <= args.buffer_capacity <= 2_000_000):
        return _fail(
            "--buffer-capacity must be in 10k..2M.",
            "buffer capacity out of range",
            "pass --buffer-capacity 10000..2000000, then see LOG",
        )

    # Device resolution & Hardware Awareness
    from python.hardware.detector import detect_hardware
    from python.hardware.recommender import recommend_configuration
    from python.hardware.calibration import run_bounded_calibration
    from python.cli.output_context import OutputContext

    calib = None
    if getattr(args, "calibrate", False):
        emit_report(format_kv_line([("action", "running hardware calibration (<=30s budget)...")]))
        calib = run_bounded_calibration(max_duration_sec=20.0)
        emit_report(
            format_kv_line(
                [
                    ("calib_cpu_infer_ms", f"{calib.cpu_infer_latencies_ms.get(1, 0.0):.2f}"),
                    ("calib_gpu_infer_ms", f"{calib.gpu_infer_latencies_ms.get(64, 0.0):.2f}"),
                ]
            )
        )

    hw = detect_hardware()
    rec = recommend_configuration(
        hw,
        calib=calib,
        workload="rl",
        force_device=getattr(args, "device", None),
    )

    # Render hardware recommendations
    ctx_out = OutputContext.get_current()
    if ctx_out.is_rich():
        from rich.table import Table
        from rich import box

        table = Table(title="Hardware & Device Configuration", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Configured / Recommended", style="green")
        table.add_column("Rationale", style="dim")
        for row in rec.summary_table():
            table.add_row(*row)
        ctx_out.console.print(table)
    else:
        rec_rows = [(p, v, "hardware") for p, v, _ in rec.summary_table()]
        emit_report(parameters_table(rec_rows))

    try:
        resolved_common = resolve_device(args.device) if args.device else None
        train_device = getattr(args, "train_device", None) or resolved_common or rec.train_device
        infer_device = getattr(args, "infer_device", None) or resolved_common or rec.infer_device
        train_device = resolve_device(train_device)
        infer_device = resolve_device(infer_device)
    except ValueError as e:
        return _fail(
            str(e),
            "unknown device value",
            "pass --device cuda, --device cpu, or leave default, then see LOG",
        )

    iso_tmp_dir: Optional[Path] = None
    is_verify = getattr(args, "verify", False)
    if is_verify:
        iso_tmp_dir = safe_resolve(f"tmp/verify_rl_{int(time.time() * 1000)}")
        iso_tmp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = iso_tmp_dir / "checkpoints"
        log_dir = iso_tmp_dir / "runs"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            checkpoint_dir = safe_resolve(args.checkpoint_dir)
            log_dir = safe_resolve(args.log_dir)
        except (ValueError, OSError) as e:
            return _fail(
                f"Invalid directory path: {e}",
                "path escapes repo or is malformed",
                "pass repo-relative dirs, then see LOG",
            )

    # Resolve Task 5 parameters
    tablebase_path = getattr(args, "tablebase_path", None)
    tablebase_max_pieces = getattr(args, "tablebase_max_pieces", 5)
    max_wait_ms = getattr(args, "max_wait_ms", 1.0)
    watchdog_rss_gb = getattr(args, "watchdog_rss_gb", 7.5)
    watchdog_free_ram_gb = getattr(args, "watchdog_free_ram_gb", 2.0)

    # Print summary table
    mode_note = "30 seconds (Verify Mode)" if is_verify else f"{args.max_hours:.1f} hours"
    emit_report(
        parameters_table(
            [
                ("launcher", "cli-train", "default"),
                (
                    "mode",
                    "verify (isolated)" if is_verify else "rl",
                    "user" if is_verify else "default",
                ),
                (
                    "initialization",
                    init_desc,
                    "user"
                    if (random_init_arg or resume_arg or checkpoint_arg)
                    else "default (protected baseline)",
                ),
                (
                    "train_device",
                    str(train_device),
                    "user" if getattr(args, "train_device", None) else "auto",
                ),
                (
                    "infer_device",
                    str(infer_device),
                    "user" if getattr(args, "infer_device", None) else "auto",
                ),
                ("duration", mode_note, "default"),
                (
                    "concurrency",
                    f"{args.concurrent_games} games" if not is_verify else "8 games (verify)",
                    "default",
                ),
                (
                    "buffer_capacity",
                    f"{args.buffer_capacity:,}" if not is_verify else "10,000 (verify)",
                    "default",
                ),
                (
                    "tablebase",
                    f"{tablebase_path} (max {tablebase_max_pieces} pieces)"
                    if tablebase_path
                    else "disabled",
                    "user" if tablebase_path else "default",
                ),
                ("checkpoint_dir", str(checkpoint_dir), "isolated_tmp" if is_verify else "default"),
                ("log_dir", str(log_dir), "isolated_tmp" if is_verify else "default"),
            ]
        )
    )

    try:
        sf_path = str(resolve_stockfish(None) or safe_resolve("tools/stockfish.exe"))
    except (ValueError, FileNotFoundError) as e:
        return _fail(
            str(e), "stockfish binary not found", "install tools/stockfish.exe, then see LOG"
        )
    puzzle_candidates = [
        "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        "data/puzzles/lichess_puzzles_5000_1800_2800.csv",
        "data/puzzles/lichess_puzzles_5000_1000_1600.csv",
        "data/puzzles/lichess_puzzles_50000_1200_3000.csv",
        "data/puzzles/tactics_train_300k.csv",
    ]
    puzzle_path = None
    for candidate in puzzle_candidates:
        p = safe_resolve(candidate)
        if p.exists():
            puzzle_path = str(p)
            break
    if puzzle_path is None:
        return _fail(
            "No puzzle CSV found in data/puzzles",
            "puzzle dataset missing",
            "run the CLI 'Add More Puzzles' option to download puzzles",
        )
    openings_path = str(safe_resolve("data/openings/eval_openings_50.epd"))
    canonical_human = safe_resolve("data/dataset_training/train")
    scaled_human = safe_resolve("data/dataset_scaled/train")
    human_dir = str(canonical_human if canonical_human.is_dir() else scaled_human)

    try:
        if is_verify:
            orchestrator = Phase4Orchestrator(
                max_hours=30.0 / 3600.0,
                eval_interval_sec=15.0,
                num_concurrent_games=8,
                buffer_capacity=10_000,
                gpu_cache_capacity=min(getattr(args, "gpu_cache_capacity", 160_000), 10_000),
                human_cache_size=min(getattr(args, "human_cache_size", 80_000), 5_000),
                checkpoint_dir=str(checkpoint_dir),
                log_dir=str(log_dir),
                base_checkpoint=base_checkpoint,
                resume_checkpoint=resume_checkpoint,
                human_data_dir=human_dir,
                puzzle_csv_path=puzzle_path,
                openings_path=openings_path,
                stockfish_path=sf_path,
                stockfish_nodes=getattr(args, "stockfish_nodes", 1000),
                is_verify_mode=True,
                skip_baseline_eval=getattr(args, "skip_baseline_eval", False),
                tablebase_path=tablebase_path,
                tablebase_max_pieces=tablebase_max_pieces,
                min_buffer_fill=min(getattr(args, "min_buffer_fill", 0), 100),
                max_reuse_ratio=getattr(args, "max_reuse_ratio", 16),
                full_search_sims=getattr(args, "full_sims", 800),
                fast_search_sims=getattr(args, "fast_sims", 150),
                reset_buffer=getattr(args, "reset_buffer", False),
                max_wait_ms=max_wait_ms,
                watchdog_rss_gb=watchdog_rss_gb,
                watchdog_free_ram_gb=watchdog_free_ram_gb,
                cleanup_buffer_on_exit=getattr(args, "cleanup_buffer_on_exit", False),
                keep_checkpoints=getattr(args, "keep_checkpoints", 3),
                keep_tfevents=getattr(args, "keep_tfevents", 1),
                device=args.device,
                train_device=train_device,
                infer_device=infer_device,
                seed=getattr(args, "seed", None),
                policy_target_temp=getattr(args, "policy_target_temp", 0.5),
                entropy_bonus=getattr(args, "entropy_bonus", 0.001),
            )
        else:
            orchestrator = Phase4Orchestrator(
                max_hours=args.max_hours,
                eval_interval_sec=args.eval_interval_sec,
                num_concurrent_games=args.concurrent_games,
                buffer_capacity=args.buffer_capacity,
                gpu_cache_capacity=getattr(args, "gpu_cache_capacity", 160_000),
                human_cache_size=getattr(args, "human_cache_size", 80_000),
                checkpoint_dir=str(checkpoint_dir),
                log_dir=str(log_dir),
                base_checkpoint=base_checkpoint,
                resume_checkpoint=resume_checkpoint,
                human_data_dir=human_dir,
                puzzle_csv_path=puzzle_path,
                openings_path=openings_path,
                stockfish_path=sf_path,
                stockfish_nodes=getattr(args, "stockfish_nodes", 1000),
                is_verify_mode=False,
                skip_baseline_eval=getattr(args, "skip_baseline_eval", False),
                tablebase_path=tablebase_path,
                tablebase_max_pieces=tablebase_max_pieces,
                min_buffer_fill=getattr(args, "min_buffer_fill", 0),
                max_reuse_ratio=getattr(args, "max_reuse_ratio", 16),
                full_search_sims=getattr(args, "full_sims", 800),
                fast_search_sims=getattr(args, "fast_sims", 150),
                reset_buffer=getattr(args, "reset_buffer", False),
                max_wait_ms=max_wait_ms,
                watchdog_rss_gb=watchdog_rss_gb,
                watchdog_free_ram_gb=watchdog_free_ram_gb,
                cleanup_buffer_on_exit=getattr(args, "cleanup_buffer_on_exit", False),
                keep_checkpoints=getattr(args, "keep_checkpoints", 3),
                keep_tfevents=getattr(args, "keep_tfevents", 1),
                device=args.device,
                train_device=train_device,
                infer_device=infer_device,
                seed=getattr(args, "seed", None),
                policy_target_temp=getattr(args, "policy_target_temp", 0.5),
                entropy_bonus=getattr(args, "entropy_bonus", 0.001),
            )

        code = orchestrator.run()
        # Legacy mapping preserved: safety-trigger stop reports exit 2 via CLI.
        if code == 1 and getattr(orchestrator, "_exit_reason", "") == "safety":
            emit_report(summary_block("FAILED", time.time() - t0, 0, 2))
            return 2
        return code

    except SafetyTriggerException as ste:
        return _fail(
            f"Training stopped due to Safety Trigger: {ste}",
            "guardrail tripped during startup",
            "inspect the log, fix the cause, resume",
            code=2,
        )
    except KeyboardInterrupt:
        emit_report(summary_block("INTERRUPTED", time.time() - t0, 0, 0))
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        logger.debug("training failed: %s", e)
        return _fail(
            f"Training failed with error: {e}",
            "startup failure (paths, device, checkpoint)",
            "check arguments and files, then see LOG",
        )
    finally:
        if iso_tmp_dir and iso_tmp_dir.is_dir():
            try:
                import gc
                import shutil

                try:
                    del orchestrator
                except NameError:
                    pass
                gc.collect()
                for _ in range(4):
                    time.sleep(0.25)
                    shutil.rmtree(iso_tmp_dir, ignore_errors=True)
                    if not iso_tmp_dir.exists():
                        break
            except Exception:
                pass
