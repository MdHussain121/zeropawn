"""Universal Configuration Management, Schema Registry, and 4-Tier Resolution Engine for ZeroPawn.

Precedence Rules:
  1. Explicit CLI arguments (passed via command-line flags)
  2. Selected preset (from configs/presets.json or built-in presets)
  3. Configuration file values (configs/config.json or custom --config file)
  4. Built-in defaults

Every configuration field records its type, allowed values, description,
applicable workflows, runtime destination, and tracks provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rich import box
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER
from python.common.env import get_env

DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "config.json"


@dataclass
class ConfigField:
    """Specification of a single recognized configuration field."""

    key: str  # Dotted canonical key (e.g. 'training.max_hours')
    alias: str  # Flattened alias (e.g. 'max_hours')
    type_name: str  # 'int', 'float', 'bool', 'str', 'choice', 'path'
    default: Any  # Implementation default value
    description: str  # Purpose and runtime effect
    applicable_workflows: List[str]  # Subcommands using this setting (e.g. ['train', 'bench'])
    cli_flags: List[str]  # Command-line flags mapping to this field (e.g. ['--max-hours'])
    runtime_dest: str  # Runtime component that consumes it
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    choices: Optional[List[str]] = None

    def validate(self, val: Any) -> Tuple[bool, Optional[str]]:
        """Validate value against type and constraints."""
        if val is None:
            return True, None

        if self.type_name == "int":
            try:
                iv = int(val)
                if self.min_val is not None and iv < self.min_val:
                    return False, f"Value {iv} is below minimum {self.min_val}"
                if self.max_val is not None and iv > self.max_val:
                    return False, f"Value {iv} exceeds maximum {self.max_val}"
                return True, None
            except (ValueError, TypeError):
                return False, f"Cannot convert '{val}' to integer"

        elif self.type_name == "float":
            try:
                fv = float(val)
                if self.min_val is not None and fv < self.min_val:
                    return False, f"Value {fv} is below minimum {self.min_val}"
                if self.max_val is not None and fv > self.max_val:
                    return False, f"Value {fv} exceeds maximum {self.max_val}"
                return True, None
            except (ValueError, TypeError):
                return False, f"Cannot convert '{val}' to float"

        elif self.type_name == "bool":
            if isinstance(val, bool):
                return True, None
            if str(val).lower() in ("true", "1", "yes", "y", "enable", "enabled"):
                return True, None
            if str(val).lower() in ("false", "0", "no", "n", "disable", "disabled"):
                return True, None
            return False, f"Cannot parse '{val}' as boolean"

        elif self.type_name == "choice":
            if self.choices and str(val) not in self.choices:
                return False, f"Value '{val}' must be one of: {', '.join(self.choices)}"
            return True, None

        elif self.type_name in ("str", "path"):
            return True, None

        return True, None

    def coerce(self, val: Any) -> Any:
        """Coerce string/raw value to proper Python type."""
        if val is None:
            return None
        if self.type_name == "int":
            try:
                return int(val)
            except (ValueError, TypeError):
                return val
        elif self.type_name == "float":
            try:
                return float(val)
            except (ValueError, TypeError):
                return val
        elif self.type_name == "bool":
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes", "y", "enable", "enabled")
        elif self.type_name in ("str", "choice", "path"):
            return str(val)
        return val


# ── Canonical Configuration Registry ───────────────────────────────────────
CONFIG_REGISTRY: Dict[str, ConfigField] = {
    # ── System & Hardware ──
    "system.device": ConfigField(
        key="system.device",
        alias="device",
        type_name="choice",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        description="Execution compute device for neural network inference and training",
        applicable_workflows=["train", "supervised", "match", "bench", "eval", "puzzles", "uci"],
        cli_flags=["--device"],
        runtime_dest="resolve_device() -> torch.device",
    ),
    "system.train_device": ConfigField(
        key="system.train_device",
        alias="train_device",
        type_name="choice",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        description="Dedicated compute device for RL neural network training (Modes A-F)",
        applicable_workflows=["train"],
        cli_flags=["--train-device"],
        runtime_dest="RLTrainer.device",
    ),
    "system.infer_device": ConfigField(
        key="system.infer_device",
        alias="infer_device",
        type_name="choice",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        description="Dedicated compute device for self-play inference server (Modes A-F)",
        applicable_workflows=["train", "bench"],
        cli_flags=["--infer-device"],
        runtime_dest="InferenceServer.device",
    ),
    "system.precision": ConfigField(
        key="system.precision",
        alias="precision",
        type_name="choice",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        description="Inference and training numeric precision (bfloat16 requires Ampere+ CUDA)",
        applicable_workflows=["train", "supervised", "bench", "eval"],
        cli_flags=["--precision"],
        runtime_dest="InferenceServer.use_bfloat16 / RLTrainer.use_bfloat16",
    ),
    "system.channels_last": ConfigField(
        key="system.channels_last",
        alias="channels_last",
        type_name="bool",
        default=True,
        description="Enable PyTorch channels-last memory format for Tensor Core optimization",
        applicable_workflows=["train", "bench", "eval"],
        cli_flags=["--channels-last", "--no-channels-last"],
        runtime_dest="InferenceServer.channels_last",
    ),
    "system.seed": ConfigField(
        key="system.seed",
        alias="seed",
        type_name="int",
        default=None,
        description="Global RNG seed for reproducible experiments and search",
        applicable_workflows=["train", "supervised", "match", "test"],
        cli_flags=["--seed"],
        runtime_dest="torch.manual_seed() / np.random.seed() / zeropawn_cpp.MCTSConfig.seed",
    ),
    # ── Model Architecture ──
    "model.num_blocks": ConfigField(
        key="model.num_blocks",
        alias="num_blocks",
        type_name="int",
        default=12,
        min_val=2,
        max_val=40,
        description="Number of residual blocks in the ChessNet backbone",
        applicable_workflows=["train", "supervised", "models"],
        cli_flags=["--num-blocks"],
        runtime_dest="ChessNet(num_blocks=...)",
    ),
    "model.num_filters": ConfigField(
        key="model.num_filters",
        alias="num_filters",
        type_name="int",
        default=128,
        min_val=32,
        max_val=512,
        description="Number of convolutional filters per residual block",
        applicable_workflows=["train", "supervised", "models"],
        cli_flags=["--num-filters"],
        runtime_dest="ChessNet(num_filters=...)",
    ),
    "model.checkpoint": ConfigField(
        key="model.checkpoint",
        alias="checkpoint",
        type_name="path",
        default="models/checkpoints_scaled/best_model.pt",
        description="Path to default neural network checkpoint weights",
        applicable_workflows=["train", "match", "eval", "puzzles", "bench", "models", "uci"],
        cli_flags=["--checkpoint", "-c", "--weights"],
        runtime_dest="load_engine_model()",
    ),
    # ── RL Training & Orchestration ──
    "training.max_hours": ConfigField(
        key="training.max_hours",
        alias="max_hours",
        type_name="float",
        default=72.0,
        min_val=0.0001,
        max_val=10000.0,
        description="Maximum training duration in wall-clock hours",
        applicable_workflows=["train"],
        cli_flags=["--max-hours"],
        runtime_dest="Phase4Orchestrator.max_hours",
    ),
    "training.batch_size": ConfigField(
        key="training.batch_size",
        alias="batch_size",
        type_name="int",
        default=512,
        min_val=16,
        max_val=4096,
        description="Neural network optimization effective batch size for gradient updates",
        applicable_workflows=["train", "supervised", "finetune"],
        cli_flags=["--batch-size"],
        runtime_dest="RLTrainer.batch_size / SupervisedTrainer.batch_size",
    ),
    "training.micro_batch_size": ConfigField(
        key="training.micro_batch_size",
        alias="micro_batch_size",
        type_name="int",
        default=None,
        min_val=1,
        max_val=2048,
        description="Physical micro-batch size for DataLoader forward/backward passes before accumulation",
        applicable_workflows=["train", "supervised"],
        cli_flags=["--micro-batch-size"],
        runtime_dest="train_supervised(micro_batch_size=...)",
    ),
    "training.gradient_accumulation_steps": ConfigField(
        key="training.gradient_accumulation_steps",
        alias="gradient_accumulation_steps",
        type_name="int",
        default=None,
        min_val=1,
        max_val=1024,
        description="Number of micro-batches accumulated before optimizer step",
        applicable_workflows=["train", "supervised"],
        cli_flags=["--gradient-accumulation-steps"],
        runtime_dest="train_supervised(gradient_accumulation_steps=...)",
    ),
    "training.lr": ConfigField(
        key="training.lr",
        alias="lr",
        type_name="float",
        default=0.0002,
        min_val=1e-7,
        max_val=1.0,
        description="Peak learning rate for AdamW/SGD optimizer",
        applicable_workflows=["train", "supervised", "finetune"],
        cli_flags=["--lr"],
        runtime_dest="RLTrainer.base_lr / SupervisedTrainer.base_lr",
    ),
    "training.min_lr": ConfigField(
        key="training.min_lr",
        alias="min_lr",
        type_name="float",
        default=0.00001,
        min_val=1e-8,
        max_val=0.1,
        description="Minimum learning rate floor for cosine decay scheduler",
        applicable_workflows=["train", "supervised"],
        cli_flags=["--min-lr"],
        runtime_dest="WarmupCosineDecayScheduler.min_lr",
    ),
    "training.warmup_steps": ConfigField(
        key="training.warmup_steps",
        alias="warmup_steps",
        type_name="int",
        default=500,
        min_val=0,
        max_val=100000,
        description="Linear learning rate warmup step count",
        applicable_workflows=["train", "supervised"],
        cli_flags=["--warmup-steps"],
        runtime_dest="WarmupCosineDecayScheduler.warmup_steps",
    ),
    "training.epochs": ConfigField(
        key="training.epochs",
        alias="epochs",
        type_name="int",
        default=5,
        min_val=1,
        max_val=1000,
        description="Number of supervised training epochs over human games",
        applicable_workflows=["supervised", "finetune"],
        cli_flags=["--epochs"],
        runtime_dest="train_supervised(epochs=...)",
    ),
    "training.eval_interval_sec": ConfigField(
        key="training.eval_interval_sec",
        alias="eval_interval_sec",
        type_name="float",
        default=21600.0,
        min_val=10.0,
        max_val=864000.0,
        description="Interval between Stockfish baseline and puzzle evaluations in seconds (default: 6h)",
        applicable_workflows=["train"],
        cli_flags=["--eval-interval-sec"],
        runtime_dest="Phase4Orchestrator.eval_interval_sec",
    ),
    "training.push_freq": ConfigField(
        key="training.push_freq",
        alias="push_freq",
        type_name="int",
        default=200,
        min_val=1,
        max_val=10000,
        description="Number of training steps between model weight pushes to self-play inference",
        applicable_workflows=["train"],
        cli_flags=["--push-freq"],
        runtime_dest="RLTrainer.push_freq",
    ),
    "training.keep_checkpoints": ConfigField(
        key="training.keep_checkpoints",
        alias="keep_checkpoints",
        type_name="int",
        default=3,
        min_val=1,
        max_val=100,
        description="Number of recent periodic checkpoints to retain",
        applicable_workflows=["train", "cleanup"],
        cli_flags=["--keep-checkpoints"],
        runtime_dest="Phase4Orchestrator.keep_checkpoints",
    ),
    "training.keep_tfevents": ConfigField(
        key="training.keep_tfevents",
        alias="keep_tfevents",
        type_name="int",
        default=1,
        min_val=1,
        max_val=100,
        description="Number of recent TensorBoard event files to retain per log directory",
        applicable_workflows=["train", "cleanup"],
        cli_flags=["--keep-tfevents"],
        runtime_dest="Phase4Orchestrator.keep_tfevents",
    ),
    # ── Self-Play & Workers ──
    "selfplay.concurrent_games": ConfigField(
        key="selfplay.concurrent_games",
        alias="concurrent_games",
        type_name="int",
        default=48,
        min_val=1,
        max_val=128,
        description="Number of concurrent self-play game simulations",
        applicable_workflows=["train", "bench"],
        cli_flags=["--concurrent-games"],
        runtime_dest="Phase4Orchestrator.num_concurrent_games",
    ),
    "selfplay.full_search_sims": ConfigField(
        key="selfplay.full_search_sims",
        alias="full_sims",
        type_name="int",
        default=800,
        min_val=10,
        max_val=10000,
        description="MCTS simulation count for full playout cap moves (prob=0.25)",
        applicable_workflows=["train", "bench"],
        cli_flags=["--full-sims"],
        runtime_dest="Phase4Orchestrator.full_search_sims / SelfPlayGame.full_search_sims",
    ),
    "selfplay.fast_search_sims": ConfigField(
        key="selfplay.fast_search_sims",
        alias="fast_sims",
        type_name="int",
        default=150,
        min_val=1,
        max_val=2000,
        description="MCTS simulation count for fast playout cap moves (prob=0.75)",
        applicable_workflows=["train", "bench"],
        cli_flags=["--fast-sims"],
        runtime_dest="Phase4Orchestrator.fast_search_sims / SelfPlayGame.fast_search_sims",
    ),
    "selfplay.buffer_capacity": ConfigField(
        key="selfplay.buffer_capacity",
        alias="buffer_capacity",
        type_name="int",
        default=500000,
        min_val=1000,
        max_val=5000000,
        description="Maximum positions capacity in memory-mapped replay buffer",
        applicable_workflows=["train"],
        cli_flags=["--buffer-capacity"],
        runtime_dest="Phase4Orchestrator.buffer_capacity / ReplayBuffer",
    ),
    "selfplay.gpu_cache_capacity": ConfigField(
        key="selfplay.gpu_cache_capacity",
        alias="gpu_cache_capacity",
        type_name="int",
        default=160000,
        min_val=1000,
        max_val=1000000,
        description="Capacity of GPU VRAM active replay cache",
        applicable_workflows=["train"],
        cli_flags=["--gpu-cache-capacity"],
        runtime_dest="RLTrainer.gpu_cache_capacity",
    ),
    "selfplay.human_cache_size": ConfigField(
        key="selfplay.human_cache_size",
        alias="human_cache_size",
        type_name="int",
        default=80000,
        min_val=0,
        max_val=500000,
        description="Active sample of human master dataset cached in VRAM for anti-forgetting",
        applicable_workflows=["train"],
        cli_flags=["--human-cache-size"],
        runtime_dest="Phase4Orchestrator.human_cache_size",
    ),
    "selfplay.min_buffer_fill": ConfigField(
        key="selfplay.min_buffer_fill",
        alias="min_buffer_fill",
        type_name="int",
        default=0,
        min_val=0,
        max_val=100000,
        description="Minimum self-play positions in replay buffer before RL training begins",
        applicable_workflows=["train"],
        cli_flags=["--min-buffer-fill"],
        runtime_dest="Phase4Orchestrator.min_buffer_fill",
    ),
    "selfplay.max_reuse_ratio": ConfigField(
        key="selfplay.max_reuse_ratio",
        alias="max_reuse_ratio",
        type_name="int",
        default=16,
        min_val=1,
        max_val=100,
        description="Maximum sample reuse ratio for pacing training steps relative to self-play generation",
        applicable_workflows=["train"],
        cli_flags=["--max-reuse-ratio"],
        runtime_dest="Phase4Orchestrator.max_reuse_ratio",
    ),
    "selfplay.max_wait_ms": ConfigField(
        key="selfplay.max_wait_ms",
        alias="max_wait_ms",
        type_name="float",
        default=1.0,
        min_val=0.1,
        max_val=100.0,
        description="Dispatcher aggregation timeout in milliseconds before flushing leaf batches",
        applicable_workflows=["train", "bench"],
        cli_flags=["--max-wait-ms"],
        runtime_dest="InferenceServer.max_wait_sec / WorkerBatchAggregator.max_wait_sec",
    ),
    "selfplay.pipelined": ConfigField(
        key="selfplay.pipelined",
        alias="pipelined",
        type_name="bool",
        default=False,
        description="Enable multi-stage pipelined GPU and delivery worker threads",
        applicable_workflows=["train", "bench"],
        cli_flags=["--pipelined"],
        runtime_dest="WorkerBatchAggregator.pipelined",
    ),
    # ── MCTS Search Tree ──
    "mcts.c_puct": ConfigField(
        key="mcts.c_puct",
        alias="c_puct",
        type_name="float",
        default=2.0,
        min_val=0.0,
        max_val=10.0,
        description="PUCT formula exploration constant balancing Q value vs policy prior",
        applicable_workflows=["train", "match", "eval", "puzzles", "uci"],
        cli_flags=["--c-puct"],
        runtime_dest="zeropawn_cpp.MCTSConfig.c_puct",
    ),
    "mcts.dirichlet_alpha": ConfigField(
        key="mcts.dirichlet_alpha",
        alias="dirichlet_alpha",
        type_name="float",
        default=0.3,
        min_val=0.01,
        max_val=2.0,
        description="Dirichlet noise concentration parameter for root move exploration",
        applicable_workflows=["train"],
        cli_flags=["--dirichlet-alpha"],
        runtime_dest="zeropawn_cpp.MCTSConfig.dirichlet_alpha",
    ),
    "mcts.dirichlet_weight": ConfigField(
        key="mcts.dirichlet_weight",
        alias="dirichlet_weight",
        type_name="float",
        default=0.25,
        min_val=0.0,
        max_val=1.0,
        description="Dirichlet noise mixture fraction applied to root policy",
        applicable_workflows=["train"],
        cli_flags=["--dirichlet-weight"],
        runtime_dest="zeropawn_cpp.MCTSConfig.dirichlet_weight",
    ),
    "mcts.fpu_reduction": ConfigField(
        key="mcts.fpu_reduction",
        alias="fpu_reduction",
        type_name="float",
        default=0.2,
        min_val=0.0,
        max_val=2.0,
        description="First Play Urgency (FPU) reduction below parent Q value for unvisited moves",
        applicable_workflows=["train", "match", "uci"],
        cli_flags=["--fpu-reduction"],
        runtime_dest="zeropawn_cpp.MCTSConfig.fpu_reduction",
    ),
    "mcts.virtual_loss": ConfigField(
        key="mcts.virtual_loss",
        alias="virtual_loss",
        type_name="int",
        default=1,
        min_val=0,
        max_val=10,
        description="Virtual loss count applied during parallel search descent",
        applicable_workflows=["train", "match", "uci"],
        cli_flags=["--virtual-loss"],
        runtime_dest="zeropawn_cpp.MCTSConfig.virtual_loss",
    ),
    "mcts.search_mode": ConfigField(
        key="mcts.search_mode",
        alias="search_mode",
        type_name="choice",
        default="serial",
        choices=["serial", "batched"],
        description="MCTS search implementation mode (serial verified baseline or batched)",
        applicable_workflows=["match", "uci", "puzzles", "search", "bench"],
        cli_flags=["--search-mode", "--mode"],
        runtime_dest="MatchRunner.search_mode / UCIEngine.search_mode",
    ),
    "mcts.batch_size": ConfigField(
        key="mcts.batch_size",
        alias="mcts_batch_size",
        type_name="str",
        default="auto",
        description="Inference batch size for batched MCTS search ('auto' for hardware-tuned, or integer 1-256)",
        applicable_workflows=["match", "uci", "puzzles", "search", "bench"],
        cli_flags=["--batch-size", "--mcts-batch-size", "-b"],
        runtime_dest="resolve_batch_size()",
    ),
    "mcts.reuse_tree": ConfigField(
        key="mcts.reuse_tree",
        alias="reuse_tree",
        type_name="bool",
        default=False,
        description="Retain and advance MCTS search tree between moves during play-time search",
        applicable_workflows=["match", "uci"],
        cli_flags=["--reuse-tree"],
        runtime_dest="zeropawn_cpp.MCTSConfig.reuse_tree",
    ),
    # ── Evaluation & Matches ──
    "eval.games": ConfigField(
        key="eval.games",
        alias="games",
        type_name="int",
        default=20,
        min_val=2,
        max_val=5000,
        description="Total games played in match evaluations (paired across openings)",
        applicable_workflows=["match", "eval"],
        cli_flags=["--games", "-n"],
        runtime_dest="MatchRunner.num_games",
    ),
    "eval.mcts_sims": ConfigField(
        key="eval.mcts_sims",
        alias="mcts_sims",
        type_name="int",
        default=50,
        min_val=1,
        max_val=10000,
        description="MCTS simulation budget per move during match evaluation",
        applicable_workflows=["match", "eval", "puzzles"],
        cli_flags=["--mcts-sims", "--sims"],
        runtime_dest="MatchRunner.mcts_sims",
    ),
    "eval.stockfish_nodes": ConfigField(
        key="eval.stockfish_nodes",
        alias="stockfish_nodes",
        type_name="int",
        default=1000,
        min_val=10,
        max_val=10000000,
        description="Search node limit for Stockfish reference opponent",
        applicable_workflows=["match", "eval"],
        cli_flags=["--stockfish-nodes"],
        runtime_dest="MatchRunner.stockfish_nodes",
    ),
    "eval.stockfish_path": ConfigField(
        key="eval.stockfish_path",
        alias="stockfish_path",
        type_name="path",
        default="tools/stockfish.exe",
        description="Path to Stockfish reference executable",
        applicable_workflows=["match", "eval", "book", "tablebase"],
        cli_flags=["--stockfish-path"],
        runtime_dest="resolve_stockfish()",
    ),
    "eval.workers": ConfigField(
        key="eval.workers",
        alias="workers",
        type_name="int",
        default=4,
        min_val=1,
        max_val=32,
        description="Number of concurrent worker threads during evaluation matches",
        applicable_workflows=["match", "eval"],
        cli_flags=["--workers"],
        runtime_dest="MatchRunner.workers",
    ),
    "eval.openings_path": ConfigField(
        key="eval.openings_path",
        alias="openings",
        type_name="path",
        default="data/openings/eval_openings_50.epd",
        description="Path to opening EPD book for balanced match pairs",
        applicable_workflows=["match", "eval"],
        cli_flags=["--openings"],
        runtime_dest="MatchRunner.openings_path",
    ),
    "eval.use_book": ConfigField(
        key="eval.use_book",
        alias="use_book",
        type_name="bool",
        default=True,
        description="Enable curated opening book lookup during match evaluation and play",
        applicable_workflows=["match", "uci"],
        cli_flags=["--book", "--no-book"],
        runtime_dest="MatchRunner.use_book / UCIEngine.use_book",
    ),
    "eval.book_path": ConfigField(
        key="eval.book_path",
        alias="book_path",
        type_name="path",
        default="data/openings/curated_book.json",
        description="Path to curated opening book JSON catalog",
        applicable_workflows=["match", "book", "uci"],
        cli_flags=["--book"],
        runtime_dest="OpeningBook.book_path",
    ),
    "eval.book_policy": ConfigField(
        key="eval.book_policy",
        alias="book_policy",
        type_name="choice",
        default="best",
        choices=["best", "weighted_blend", "frequency", "score", "uniform"],
        description="Candidate move selection strategy from opening book",
        applicable_workflows=["match", "book", "uci"],
        cli_flags=["--book-policy"],
        runtime_dest="OpeningBook.policy",
    ),
    # ── Syzygy Endgame Tablebases ──
    "tablebase.path": ConfigField(
        key="tablebase.path",
        alias="tablebase_path",
        type_name="path",
        default=None,
        description="Path to Syzygy WDL tablebase files for instant endgame adjudication",
        applicable_workflows=["train", "tablebase"],
        cli_flags=["--tablebase-path"],
        runtime_dest="TablebaseAdjudicator(path=...)",
    ),
    "tablebase.max_pieces": ConfigField(
        key="tablebase.max_pieces",
        alias="tablebase_max_pieces",
        type_name="int",
        default=5,
        min_val=3,
        max_val=7,
        description="Maximum piece count (including kings) eligible for tablebase probe",
        applicable_workflows=["train", "tablebase"],
        cli_flags=["--tablebase-max-pieces"],
        runtime_dest="TablebaseAdjudicator.max_pieces",
    ),
    # ── Memory Safety & Watchdogs ──
    "watchdog.rss_gb": ConfigField(
        key="watchdog.rss_gb",
        alias="watchdog_rss_gb",
        type_name="float",
        default=7.5,
        min_val=1.0,
        max_val=128.0,
        description="Process RSS physical memory abort ceiling in GB (halts training if breached)",
        applicable_workflows=["train"],
        cli_flags=["--watchdog-rss-gb"],
        runtime_dest="Phase4Orchestrator.watchdog_rss_gb",
    ),
    "watchdog.free_ram_gb": ConfigField(
        key="watchdog.free_ram_gb",
        alias="watchdog_free_ram_gb",
        type_name="float",
        default=2.0,
        min_val=0.2,
        max_val=64.0,
        description="Host system free RAM floor in GB (halts training if breached)",
        applicable_workflows=["train"],
        cli_flags=["--watchdog-free-ram-gb"],
        runtime_dest="Phase4Orchestrator.watchdog_free_ram_gb",
    ),
    # ── Dataset Directories ──
    "dataset.train_dir": ConfigField(
        key="dataset.train_dir",
        alias="train_dir",
        type_name="path",
        default="data/dataset_training/train",
        description="Directory containing training binary memory-mapped chunk files",
        applicable_workflows=["train", "supervised", "data"],
        cli_flags=["--train-dir", "--data-dir"],
        runtime_dest="BinaryChessDataset(train_dir)",
    ),
    "dataset.val_dir": ConfigField(
        key="dataset.val_dir",
        alias="val_dir",
        type_name="path",
        default="data/dataset_training/val",
        description="Directory containing validation binary memory-mapped chunk files",
        applicable_workflows=["train", "supervised", "eval", "data"],
        cli_flags=["--val-dir"],
        runtime_dest="BinaryChessDataset(val_dir)",
    ),
    # ── Paths & Output ──
    "paths.checkpoint_dir": ConfigField(
        key="paths.checkpoint_dir",
        alias="checkpoint_dir",
        type_name="path",
        default="models/checkpoints_rl",
        description="Directory to save periodic and final model checkpoints",
        applicable_workflows=["train", "supervised", "cleanup"],
        cli_flags=["--checkpoint-dir"],
        runtime_dest="Phase4Orchestrator.checkpoint_dir",
    ),
    "paths.log_dir": ConfigField(
        key="paths.log_dir",
        alias="log_dir",
        type_name="path",
        default="runs/phase4_rl",
        description="Directory for TensorBoard event streams and telemetry CSV logs",
        applicable_workflows=["train", "supervised", "cleanup"],
        cli_flags=["--log-dir"],
        runtime_dest="Phase4Orchestrator.log_dir",
    ),
}

# Lookup map from alias to ConfigField (detect collisions at import time)
ALIAS_MAP: Dict[str, ConfigField] = {}
_duplicate_aliases: set[str] = set()
for cf in CONFIG_REGISTRY.values():
    if cf.alias in ALIAS_MAP:
        existing = ALIAS_MAP[cf.alias].key
        if existing != cf.key:
            _duplicate_aliases.add(cf.alias)
            # Skip the duplicate alias to prevent collision; keep first entry.
            continue
    ALIAS_MAP[cf.alias] = cf
    # Also register disambiguated alias for fields that had collisions
    if cf.alias in _duplicate_aliases:
        ALIAS_MAP[cf.alias + "_disambiguated"] = cf


# ── Provenance Record ──
@dataclass
class ProvenanceRecord:
    key: str
    effective_value: Any
    source: str  # 'cli', 'preset', 'config_file', 'default'
    default_value: Any
    is_override: bool
    field_spec: ConfigField


class ResolvedConfiguration:
    """Encapsulates the final resolved configuration with full provenance tracking."""

    def __init__(self, records: Dict[str, ProvenanceRecord], command_name: str = ""):
        self.records = records
        self.command_name = command_name

    def get(self, key_or_alias: str, default: Any = None) -> Any:
        """Retrieve effective value for canonical key or alias."""
        if key_or_alias in self.records:
            return self.records[key_or_alias].effective_value
        # Check alias
        if key_or_alias in ALIAS_MAP:
            canon = ALIAS_MAP[key_or_alias].key
            if canon in self.records:
                return self.records[canon].effective_value
        return default

    def get_record(self, key_or_alias: str) -> Optional[ProvenanceRecord]:
        if key_or_alias in self.records:
            return self.records[key_or_alias]
        if key_or_alias in ALIAS_MAP:
            return self.records.get(ALIAS_MAP[key_or_alias].key)
        return None

    def as_dict(self) -> Dict[str, Any]:
        """Flattened dict of canonical keys and their effective values."""
        return {k: r.effective_value for k, r in self.records.items()}

    def summary_table(self, title: Optional[str] = None, show_all: bool = False) -> Table:
        """Render a Rich summary table with color-coded provenance."""
        t_title = title or (
            f"Resolved Configuration ({self.command_name})"
            if self.command_name
            else "Resolved Configuration"
        )
        table = Table(
            title=f"[bold bright_cyan]{t_title}[/bold bright_cyan]",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Setting", style="bright_white", no_wrap=True)
        table.add_column("Flag / Key", style="dim", no_wrap=True)
        table.add_column("Effective Value", style="bold yellow")
        table.add_column("Source", justify="center")
        table.add_column("Default", style="dim")
        table.add_column("Description", style="dim")

        # Source badge formatting
        source_styles = {
            "cli": "[bold green]CLI Override[/bold green]",
            "env": "[bold magenta]Environment[/bold magenta]",
            "preset": "[bold cyan]Preset[/bold cyan]",
            "config_file": "[bold yellow]Config File[/bold yellow]",
            "default": "[dim]Built-in Default[/dim]",
        }

        for key, rec in sorted(self.records.items()):
            # Filter by applicable workflow if not showing all
            if not show_all and self.command_name:
                cmd_base = self.command_name.split()[0]
                if (
                    cmd_base not in rec.field_spec.applicable_workflows
                    and "all" not in rec.field_spec.applicable_workflows
                ):
                    continue

            eff_str = str(rec.effective_value)
            def_str = str(rec.default_value)
            flag_str = (
                rec.field_spec.cli_flags[0] if rec.field_spec.cli_flags else rec.field_spec.alias
            )
            src_str = source_styles.get(rec.source, rec.source)

            table.add_row(
                rec.field_spec.alias,
                flag_str,
                eff_str,
                src_str,
                def_str,
                rec.field_spec.description[:45]
                + ("..." if len(rec.field_spec.description) > 45 else ""),
            )

        return table


# ── Configuration File I/O ──


def load_config_file(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load JSON configuration file or return empty dict if not found."""
    p = Path(config_path) if config_path else DEFAULT_CONFIG_FILE
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (
                data.get("settings", {})
                if isinstance(data, dict) and "settings" in data
                else (data if isinstance(data, dict) else {})
            )
    except (OSError, json.JSONDecodeError):
        return {}


def save_config_file(
    settings: Dict[str, Any], config_path: Optional[Union[str, Path]] = None
) -> bool:
    """Save configuration dictionary atomically to JSON file with backup."""
    p = Path(config_path) if config_path else DEFAULT_CONFIG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.is_file():
        bak = p.with_suffix(".json.bak")
        try:
            shutil.copy2(p, bak)
        except OSError:
            pass

    payload = {
        "version": "1.0",
        "updated_at": datetime.now().isoformat(),
        "settings": settings,
    }

    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
        return True
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False


def get_config_option(
    key_or_alias: str, config_path: Optional[Union[str, Path]] = None
) -> Tuple[Optional[Any], Optional[ConfigField]]:
    """Retrieve value and field spec for a configuration key."""
    cf = CONFIG_REGISTRY.get(key_or_alias) or ALIAS_MAP.get(key_or_alias)
    if cf is None:
        return None, None
    file_settings = load_config_file(config_path)
    val = file_settings.get(cf.key, file_settings.get(cf.alias, cf.default))
    return val, cf


def set_config_option(
    key_or_alias: str,
    raw_value: Any,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Validate and set a configuration option in configs/config.json."""
    cf = CONFIG_REGISTRY.get(key_or_alias) or ALIAS_MAP.get(key_or_alias)
    if cf is None:
        # Search suggestions
        matches = [k for k in CONFIG_REGISTRY if key_or_alias.lower() in k.lower()]
        hint = f" Did you mean: {', '.join(matches[:3])}?" if matches else ""
        return False, f"Unknown configuration option '{key_or_alias}'.{hint}"

    is_valid, err = cf.validate(raw_value)
    if not is_valid:
        return False, f"Validation failed for '{cf.key}': {err}"

    coerced = cf.coerce(raw_value)
    settings = load_config_file(config_path)
    settings[cf.key] = coerced

    ok = save_config_file(settings, config_path)
    if ok:
        return True, f"Successfully set '{cf.key}' = {coerced}"
    return False, f"Failed to write configuration to {config_path or DEFAULT_CONFIG_FILE}"


def reset_config_option(
    key_or_alias: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Reset a specific option or all options to defaults in configs/config.json."""
    if key_or_alias is None or key_or_alias.lower() == "all":
        ok = save_config_file({}, config_path)
        if ok:
            return True, "Reset all configuration settings to built-in defaults."
        return False, "Failed to write configuration file."

    cf = CONFIG_REGISTRY.get(key_or_alias) or ALIAS_MAP.get(key_or_alias)
    if cf is None:
        return False, f"Unknown configuration option '{key_or_alias}'."

    settings = load_config_file(config_path)
    removed = settings.pop(cf.key, None) is not None or settings.pop(cf.alias, None) is not None
    if removed:
        ok = save_config_file(settings, config_path)
        if ok:
            return True, f"Reset '{cf.key}' to built-in default ({cf.default})."
        return False, "Failed to write configuration file."
    return True, f"'{cf.key}' is already at its default ({cf.default})."


def validate_config_dict(data: Dict[str, Any]) -> List[str]:
    """Validate a dictionary of configuration fields against the canonical registry."""
    errors: List[str] = []
    for k, v in data.items():
        cf = CONFIG_REGISTRY.get(k) or ALIAS_MAP.get(k)
        if cf is None:
            errors.append(f"Unrecognized field: '{k}'")
            continue
        is_valid, err = cf.validate(v)
        if not is_valid:
            errors.append(f"Invalid value for '{k}': {err}")
    return errors


# ── The 5-Tier Configuration Resolution Engine ──


def resolve_configuration(
    command_name: str = "",
    cli_args: Optional[Union[argparse.Namespace, Dict[str, Any]]] = None,
    preset_dict: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    raw_argv: Optional[List[str]] = None,
) -> ResolvedConfiguration:
    """Derive effective configuration honoring strict 5-tier precedence:

    Tier 1 (Highest): Explicit CLI arguments (passed on the command line)
    Tier 2: Environment variables (ZEROPAWN_* with fallback to legacy env vars)
    Tier 3: Explicitly selected preset parameters (--preset)
    Tier 4: Configuration file values (configs/config.json)
    Tier 5 (Lowest): Built-in defaults
    """
    records: Dict[str, ProvenanceRecord] = {}

    # Tier 4: Config file
    file_settings = load_config_file(config_path)

    # Tier 3: Preset parameters
    if preset_dict:
        if "parameters" in preset_dict and isinstance(preset_dict["parameters"], dict):
            preset_params = preset_dict["parameters"]
        else:
            preset_params = preset_dict
    else:
        preset_params = {}

    # Tier 1: Extract CLI values and detect which options were explicitly supplied
    cli_dict: Dict[str, Any] = {}
    is_cli_namespace = isinstance(cli_args, argparse.Namespace)
    if is_cli_namespace:
        cli_dict = vars(cli_args)
    elif isinstance(cli_args, dict):
        cli_dict = dict(cli_args)

    explicit_flags: Set[str] = set()
    if raw_argv is not None:
        for token in raw_argv:
            if token.startswith("-"):
                flag_name = token.split("=")[0]
                explicit_flags.add(flag_name)
    elif is_cli_namespace:
        raw_from_ns = getattr(cli_args, "_raw_argv", None)
        if raw_from_ns is not None:
            for token in raw_from_ns:
                if token.startswith("-"):
                    explicit_flags.add(token.split("=")[0])

    for canon_key, cf in CONFIG_REGISTRY.items():
        effective_val = cf.default
        source = "default"
        is_override = False

        # Check Tier 4: Config file
        if cf.key in file_settings:
            effective_val = cf.coerce(file_settings[cf.key])
            source = "config_file"
            is_override = True
        elif cf.alias in file_settings:
            effective_val = cf.coerce(file_settings[cf.alias])
            source = "config_file"
            is_override = True

        # Check Tier 3: Preset parameter
        if cf.alias in preset_params:
            effective_val = cf.coerce(preset_params[cf.alias])
            source = "preset"
            is_override = True
        elif cf.key in preset_params:
            effective_val = cf.coerce(preset_params[cf.key])
            source = "preset"
            is_override = True

        # Check Tier 2: Environment variables (ZEROPAWN_* with fallback to legacy env vars)
        env_names = [
            cf.key.replace(".", "_").upper(),
            cf.alias.upper(),
        ]
        for env_name in env_names:
            env_val = get_env(env_name)
            if env_val is not None:
                effective_val = cf.coerce(env_val)
                source = "env"
                is_override = True
                break

        # Check Tier 1: Explicit CLI arguments
        if explicit_flags:
            matched = False
            for flag in cf.cli_flags:
                if flag in explicit_flags:
                    matched = True
                    break
            if not matched:
                dash_alias = f"--{cf.alias.replace('_', '-')}"
                dash_key = f"--{cf.key.replace('.', '-')}"
                if dash_alias in explicit_flags or dash_key in explicit_flags:
                    matched = True

            if (
                matched
                and command_name
                and cf.applicable_workflows
                and command_name not in cf.applicable_workflows
            ):
                matched = False

            if matched:
                for flag_key in [cf.alias, cf.key] + [
                    f.lstrip("-").replace("-", "_") for f in cf.cli_flags
                ]:
                    if flag_key in cli_dict and cli_dict[flag_key] is not None:
                        effective_val = cf.coerce(cli_dict[flag_key])
                        source = "cli"
                        is_override = True
                        break
        elif not is_cli_namespace and cli_dict:
            # Explicit dictionary passed directly (e.g. in tests)
            matched = True
            if (
                command_name
                and cf.applicable_workflows
                and command_name not in cf.applicable_workflows
            ):
                matched = False
            if matched:
                for flag_key in [cf.alias, cf.key] + [
                    f.lstrip("-").replace("-", "_") for f in cf.cli_flags
                ]:
                    if flag_key in cli_dict and cli_dict[flag_key] is not None:
                        effective_val = cf.coerce(cli_dict[flag_key])
                        source = "cli"
                        is_override = True
                        break

        records[canon_key] = ProvenanceRecord(
            key=canon_key,
            effective_value=effective_val,
            source=source,
            default_value=cf.default,
            is_override=is_override,
            field_spec=cf,
        )

    return ResolvedConfiguration(records, command_name=command_name)
