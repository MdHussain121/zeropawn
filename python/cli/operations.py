"""Central Operation Registry and schema definitions for all ZeroPawn operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class OpArgument:
    name: str
    flag: str
    type: str  # "int", "float", "str", "bool", "choice", "path"
    default: Any
    help: str
    required: bool = False
    choices: Optional[List[str]] = None
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    is_positional: bool = False


@dataclass
class OperationDef:
    id: str
    category_id: int
    category_name: str
    name: str
    description: str
    direct_command: str
    requires_model: str  # "none", "optional", "required"
    arguments: List[OpArgument] = field(default_factory=list)


CATEGORIES: Dict[int, str] = {
    1: "Training & Reinforcement Learning",
    2: "Self-Play & Data Generation",
    3: "Models & Checkpoints",
    4: "Evaluation, Matches & Elo",
    5: "Inference & Benchmarking",
    6: "MCTS & Search Diagnostics",
    7: "Tests & Validation",
    8: "Presets & Experiment Configurations",
    9: "Logs, Telemetry & Diagnostics",
    10: "Maintenance & Utilities",
}

# The complete, unified 42-operation registry
OPERATIONS_REGISTRY: List[OperationDef] = [
    # ---------------- Category 1: Training & RL ----------------
    OperationDef(
        id="train_rl",
        category_id=1,
        category_name="Training & Reinforcement Learning",
        name="Reinforcement Learning Self-Play Continuous Training",
        description="Continuous self-play MCTS simulation and policy/value network optimization.",
        direct_command="zeropawn train --max-hours <N>",
        requires_model="optional",
        arguments=[
            OpArgument(
                "max_hours",
                "--max-hours",
                "float",
                72.0,
                "Maximum training runtime in hours",
                min_val=0.001,
                max_val=10000.0,
            ),
            OpArgument(
                "concurrent_games",
                "--concurrent-games",
                "int",
                48,
                "Number of concurrent self-play workers",
                min_val=1,
                max_val=128,
            ),
            OpArgument(
                "sims",
                "--full-sims",
                "int",
                800,
                "Full MCTS search simulations per move",
                min_val=10,
                max_val=10000,
            ),
            OpArgument(
                "fast_sims",
                "--fast-sims",
                "int",
                50,
                "Fast MCTS simulations per move",
                min_val=1,
                max_val=1000,
            ),
            OpArgument(
                "batch_size",
                "--batch-size",
                "int",
                256,
                "Training batch size for neural network updates",
                min_val=16,
                max_val=2048,
            ),
            OpArgument(
                "lr",
                "--lr",
                "float",
                0.0002,
                "Learning rate for RL trainer",
                min_val=1e-6,
                max_val=1.0,
            ),
        ],
    ),
    OperationDef(
        id="train_verify",
        category_id=1,
        category_name="Training & Reinforcement Learning",
        name="Quick RL Smoke Verification (30s)",
        description="Fast 30-second verification of self-play, MCTS search, and gradient updates.",
        direct_command="zeropawn train --verify",
        requires_model="none",
        arguments=[
            OpArgument(
                "verify",
                "--verify",
                "bool",
                True,
                "Execute 30-second RL smoke verification",
                required=True,
            ),
        ],
    ),
    OperationDef(
        id="train_supervised",
        category_id=1,
        category_name="Training & Reinforcement Learning",
        name="Supervised Learning Warm-Start",
        description="Trains policy and value network on human master games dataset.",
        direct_command="zeropawn supervised --epochs <N>",
        requires_model="optional",
        arguments=[
            OpArgument(
                "epochs", "--epochs", "int", 5, "Number of training epochs", min_val=1, max_val=100
            ),
            OpArgument(
                "batch_size",
                "--batch-size",
                "int",
                512,
                "Effective batch size",
                min_val=16,
                max_val=2048,
            ),
            OpArgument(
                "micro_batch_size",
                "--micro-batch-size",
                "int",
                None,
                "Physical micro-batch size",
                min_val=1,
                max_val=2048,
            ),
            OpArgument(
                "gradient_accumulation_steps",
                "--gradient-accumulation-steps",
                "int",
                None,
                "Gradient accumulation steps",
                min_val=1,
                max_val=1024,
            ),
            OpArgument("lr", "--lr", "float", 0.0002, "Learning rate", min_val=1e-6, max_val=1.0),
            OpArgument(
                "train_dir",
                "--train-dir",
                "path",
                "data/dataset_training/train",
                "Training data directory",
            ),
            OpArgument(
                "val_dir",
                "--val-dir",
                "path",
                "data/dataset_training/val",
                "Validation data directory",
            ),
        ],
    ),
    OperationDef(
        id="train_supervised_smoke",
        category_id=1,
        category_name="Training & Reinforcement Learning",
        name="Supervised Learning 10-Step Smoke Test",
        description="Executes a 10-step supervised training check to verify dataset reads and loss calculations.",
        direct_command="zeropawn supervised --verify",
        requires_model="none",
        arguments=[
            OpArgument(
                "verify",
                "--verify",
                "bool",
                True,
                "Execute 10-step supervised smoke test",
                required=True,
            ),
        ],
    ),
    OperationDef(
        id="train_finetune",
        category_id=1,
        category_name="Training & Reinforcement Learning",
        name="Fine-Tune Checkpoint on Dataset",
        description="Fine-tunes an existing model checkpoint on a selected dataset or merged mix.",
        direct_command="zeropawn finetune",
        requires_model="required",
        arguments=[
            OpArgument(
                "dataset",
                "--dataset",
                "str",
                "dataset_tactics",
                "Dataset name or number: 'dataset_tactics' (puzzles), 'dataset_training' (games), or '1,2'",
            ),
            OpArgument(
                "steps",
                "--steps",
                "int",
                1000,
                "Number of fine-tuning steps",
                min_val=10,
                max_val=100000,
            ),
            OpArgument(
                "batch_size", "--batch-size", "int", 128, "Batch size", min_val=16, max_val=1024
            ),
            OpArgument(
                "lr",
                "--lr",
                "float",
                0.0001,
                "Fine-tuning learning rate",
                min_val=1e-6,
                max_val=0.1,
            ),
        ],
    ),
    OperationDef(
        id="train_connect4",
        category_id=1,
        category_name="Training & Reinforcement Learning",
        name="Train Connect 4 Baseline",
        description="Trains the Connect 4 baseline neural network for architecture validation.",
        direct_command="zeropawn train connect4",
        requires_model="none",
        arguments=[
            OpArgument("epochs", "--epochs", "int", 5, "Training epochs", min_val=1, max_val=100),
        ],
    ),
    # ---------------- Category 2: Self-Play & Data Generation ----------------
    OperationDef(
        id="data_download",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Download Lichess Monthly PGN",
        description="Downloads, validates license, or queries metadata for Lichess standard rated PGN zst.",
        direct_command="zeropawn data download",
        requires_model="none",
        arguments=[
            OpArgument("month", "--month", "str", "2026-09", "Dataset month in YYYY-MM format"),
            OpArgument(
                "info", "--info", "bool", False, "Only query size and URL without downloading"
            ),
            OpArgument(
                "check_license",
                "--check-license",
                "bool",
                False,
                "Check Lichess open database license",
            ),
        ],
    ),
    OperationDef(
        id="data_filter",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Filter PGN Games by Elo & Time Control",
        description="Filters raw PGN by minimum Elo, normal termination, and valid time controls.",
        direct_command="zeropawn data filter",
        requires_model="none",
        arguments=[
            OpArgument(
                "input",
                "--input",
                "path",
                "data/elite_2500.pgn"
                if Path("data/elite_2500.pgn").exists()
                else "data/filtered_2200.pgn",
                "Source PGN file path",
            ),
            OpArgument(
                "output", "--output", "path", "data/filtered.pgn", "Output filtered PGN file path"
            ),
            OpArgument(
                "max_kept",
                "--max-kept",
                "int",
                10000,
                "Maximum games to keep",
                min_val=1,
                max_val=1000000,
            ),
            OpArgument(
                "min_elo",
                "--min-elo",
                "int",
                2200,
                "Minimum player rating for both sides",
                min_val=1000,
                max_val=3500,
            ),
            OpArgument(
                "min_tc",
                "--min-tc",
                "int",
                180,
                "Minimum estimated game length in seconds (base + 40*increment; 180=3m blitz, 600=10m rapid)",
                min_val=0,
                max_val=7200,
            ),
            OpArgument(
                "min_plies",
                "--min-plies",
                "int",
                20,
                "Minimum plies (half-moves) in game",
                min_val=2,
                max_val=200,
            ),
        ],
    ),
    OperationDef(
        id="data_build",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Build Binary Dataset (Planes & Moves)",
        description="Converts filtered PGN games into preallocated binary memmap tensors for training.",
        direct_command="zeropawn data build",
        requires_model="none",
        arguments=[
            OpArgument(
                "pgn",
                "--pgn",
                "path",
                next(
                    (
                        p
                        for p in [
                            "data/filtered.pgn",
                            "data/filtered_2500.pgn",
                            "tmp/filtered.pgn",
                            "data/elite_2500.pgn",
                        ]
                        if Path(p).exists()
                    ),
                    "data/filtered.pgn",
                ),
                "Input PGN file path",
            ),
            OpArgument(
                "output_dir",
                "--output-dir",
                "path",
                "data/dataset_training",
                "Destination dataset directory",
            ),
            OpArgument(
                "max_games",
                "--max-games",
                "int",
                5000,
                "Maximum games to convert",
                min_val=1,
                max_val=1000000,
            ),
        ],
    ),
    OperationDef(
        id="data_build_tactics",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Build Tactics Dataset",
        description="Extracts and constructs tactical training positions from puzzle sets.",
        direct_command="zeropawn data build-tactics",
        requires_model="none",
        arguments=[
            OpArgument(
                "csv",
                "--csv",
                "path",
                next(
                    (str(p).replace("\\", "/") for p in Path("data/puzzles").glob("*.csv")),
                    "data/puzzles/lichess_puzzles_10000_1600_2800.csv",
                ),
                "Input puzzles CSV file path",
            ),
            OpArgument(
                "output_dir",
                "--output-dir",
                "path",
                "data/dataset_tactics",
                "Destination tactics dataset directory",
            ),
            OpArgument(
                "max_solver_plies",
                "--max-solver-plies",
                "int",
                4,
                "Maximum solver plies per puzzle",
                min_val=1,
                max_val=20,
            ),
        ],
    ),
    OperationDef(
        id="data_fetch_elite",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Fetch Elite Games (2500+ Elo)",
        description="Downloads master and grandmaster game PGNs from the Lichess Elite database.",
        direct_command="zeropawn data fetch-elite",
        requires_model="none",
        arguments=[
            OpArgument(
                "max_games",
                "--max-games",
                "int",
                1000,
                "Maximum games to fetch",
                min_val=10,
                max_val=100000,
            ),
            OpArgument(
                "min_tc",
                "--min-tc",
                "int",
                600,
                "Minimum time control in seconds (600 = 10+0 rapid)",
                min_val=0,
                max_val=7200,
            ),
            OpArgument(
                "min_rating",
                "--min-rating",
                "int",
                2500,
                "Minimum player rating for both sides",
                min_val=2000,
                max_val=3500,
            ),
        ],
    ),
    OperationDef(
        id="data_fetch_tactics",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Fetch Tactics PGN",
        description="Fetches training-grade tactical positions from the online puzzle database.",
        direct_command="zeropawn data fetch-tactics",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="data_build_book",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Build Self-Play Opening Book",
        description="Generates a balanced opening book from high-level games for diverse self-play.",
        direct_command="zeropawn data build-book",
        requires_model="none",
        arguments=[
            OpArgument(
                "input",
                "--input",
                "path",
                next(
                    (
                        p
                        for p in [
                            "data/filtered.pgn",
                            "data/filtered_2500.pgn",
                            "data/elite_2500.pgn",
                        ]
                        if Path(p).exists()
                    ),
                    "data/filtered.pgn",
                ),
                "Input PGN file",
            ),
            OpArgument(
                "output", "--output", "path", "data/openings/opening_book.bin", "Output binary book"
            ),
            OpArgument(
                "max_ply",
                "--max-ply",
                "int",
                16,
                "Maximum ply depth for opening book",
                min_val=4,
                max_val=30,
            ),
        ],
    ),
    OperationDef(
        id="data_generate_openings",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Generate Balanced Openings Suite (EPD)",
        description="Creates balanced, unbiased 50-opening benchmark suite in standard EPD format.",
        direct_command="zeropawn data generate-openings",
        requires_model="none",
        arguments=[
            OpArgument(
                "output",
                "--output",
                "path",
                "data/openings/eval_openings_50.epd",
                "Output EPD file path",
            ),
        ],
    ),
    OperationDef(
        id="add_puzzles_fetch",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Stream Puzzles from Lichess",
        description="Downloads tactical puzzle CSVs filtered by rating range directly from Lichess.",
        direct_command="zeropawn add-puzzles fetch",
        requires_model="none",
        arguments=[
            OpArgument(
                "count", "-n", "int", 500, "Number of puzzles to stream", min_val=10, max_val=100000
            ),
            OpArgument(
                "min_rating",
                "--min-rating",
                "int",
                1200,
                "Minimum puzzle rating",
                min_val=400,
                max_val=3200,
            ),
            OpArgument(
                "max_rating",
                "--max-rating",
                "int",
                1600,
                "Maximum puzzle rating",
                min_val=400,
                max_val=3200,
            ),
        ],
    ),
    OperationDef(
        id="add_puzzles_manual",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Add Custom Puzzle Manually",
        description="Interactively or programmatically adds a custom FEN and verified solution.",
        direct_command="zeropawn add-puzzles manual",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="add_puzzles_list",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="List Downloaded Puzzle Sets",
        description="Scans data/puzzles/ and reports all puzzle CSVs and their puzzle counts.",
        direct_command="zeropawn add-puzzles list",
        requires_model="none",
        arguments=[],
    ),
    # ---------------- Category 3: Models & Checkpoints ----------------
    OperationDef(
        id="models_list",
        category_id=3,
        category_name="Models & Checkpoints",
        name="List Available Checkpoints",
        description="Discovers and formats a table of all checkpoints, steps, architectures, and losses.",
        direct_command="zeropawn models list",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="models_inspect",
        category_id=3,
        category_name="Models & Checkpoints",
        name="Inspect Model Architecture & Parameters",
        description="Inspects layer shapes, parameter counts, and training metadata for a checkpoint.",
        direct_command="zeropawn models inspect",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="models_compare",
        category_id=3,
        category_name="Models & Checkpoints",
        name="Compare Two Checkpoints",
        description="Performs side-by-side comparison of two models: steps, losses, and parameter counts.",
        direct_command="zeropawn models compare <P1> <P2>",
        requires_model="required",
        arguments=[
            OpArgument(
                "model_a",
                "model_a",
                "path",
                "models/checkpoints_scaled/best_model.pt",
                "Path to first checkpoint",
                required=True,
                is_positional=True,
            ),
            OpArgument(
                "model_b",
                "model_b",
                "path",
                "models/checkpoints_scaled/best_model.pt",
                "Path to second checkpoint",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    # ---------------- Category 4: Evaluation, Matches & Elo ----------------
    OperationDef(
        id="match_run",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Stockfish Benchmark Match (Elo Estimation)",
        description="Plays batch match against Stockfish engine with logistic 95% CI Elo calculation.",
        direct_command="zeropawn match run -n <N>",
        requires_model="required",
        arguments=[
            OpArgument(
                "games", "-n", "int", 20, "Number of games to play", min_val=2, max_val=10000
            ),
            OpArgument(
                "mcts_sims",
                "--mcts-sims",
                "int",
                50,
                "MCTS simulations per engine move",
                min_val=1,
                max_val=10000,
            ),
            OpArgument(
                "workers",
                "--workers",
                "int",
                1,
                "Concurrent match worker threads",
                min_val=1,
                max_val=32,
            ),
            OpArgument(
                "elo",
                "--stockfish-elo",
                "int",
                1500,
                "Stockfish UCI_Elo limit",
                min_val=500,
                max_val=3500,
            ),
            OpArgument(
                "search_mode",
                "--search-mode",
                "choice",
                "serial",
                "MCTS search mode (serial or batched)",
                choices=["serial", "batched"],
            ),
            OpArgument(
                "batch_size",
                "--batch-size",
                "str",
                "auto",
                "Inference batch size ('auto' or integer 1-256)",
            ),
        ],
    ),
    OperationDef(
        id="match_live",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Watch Live Game vs Stockfish",
        description="Renders move-by-move ASCII board, engine evaluations, and thinking process.",
        direct_command="zeropawn match live",
        requires_model="required",
        arguments=[
            OpArgument(
                "engine_color",
                "--engine-color",
                "choice",
                "white",
                "Color our AI engine plays",
                choices=["white", "black"],
            ),
            OpArgument(
                "mcts_sims",
                "--mcts-sims",
                "int",
                50,
                "MCTS search simulations per move",
                min_val=1,
                max_val=2000,
            ),
        ],
    ),
    OperationDef(
        id="match_human",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Play Human vs Engine in Terminal",
        description="Interactive chess game where the user plays moves against the AI engine.",
        direct_command="zeropawn match human",
        requires_model="required",
        arguments=[
            OpArgument(
                "human_color",
                "--human-color",
                "choice",
                "white",
                "Your piece color",
                choices=["white", "black"],
            ),
            OpArgument(
                "mcts_sims",
                "--mcts-sims",
                "int",
                50,
                "Engine MCTS search simulations",
                min_val=1,
                max_val=2000,
            ),
        ],
    ),
    OperationDef(
        id="eval_baseline",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Calibrated Stockfish Elo Baseline",
        description="Calibrates engine playing strength against Stockfish UCI_Elo levels with PGN logging.",
        direct_command="zeropawn eval baseline",
        requires_model="required",
        arguments=[
            OpArgument("games", "--games", "int", 20, "Games per level", min_val=2, max_val=200),
            OpArgument(
                "elo", "--elo", "int", 1500, "Stockfish UCI_Elo level", min_val=500, max_val=3200
            ),
        ],
    ),
    OperationDef(
        id="eval_all_baseline",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Overnight All-Elo Baseline Chain",
        description="Runs sequential 100-game matches across 6 Stockfish Elo levels for complete curve.",
        direct_command="zeropawn eval all-baseline",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="eval_heldout",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Evaluate Held-Out Human Validation Games",
        description="Computes policy top-1, top-3, cross-entropy, and value head loss on held-out test data.",
        direct_command="zeropawn eval heldout",
        requires_model="required",
        arguments=[
            OpArgument(
                "val_dir",
                "--val-dir",
                "path",
                "data/dataset_scaled/val",
                "Validation dataset directory",
            ),
            OpArgument(
                "batch_size",
                "--batch-size",
                "int",
                256,
                "Evaluation batch size",
                min_val=16,
                max_val=1024,
            ),
        ],
    ),
    OperationDef(
        id="eval_reliability",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Value Head Calibration & Reliability",
        description="Measures value head calibration curves and Brier score against game outcomes.",
        direct_command="zeropawn eval reliability",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="eval_connect4",
        category_id=4,
        category_name="Evaluation, Matches & Elo",
        name="Evaluate Connect 4 Agent",
        description="Evaluates Connect 4 agent over 100 games against a random player.",
        direct_command="zeropawn eval connect4",
        requires_model="none",
        arguments=[
            OpArgument("games", "--games", "int", 100, "Number of games", min_val=10, max_val=1000),
        ],
    ),
    # ---------------- Category 5: Inference & Benchmarking ----------------
    OperationDef(
        id="bench_selfplay",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Self-Play Throughput & MCTS Benchmark",
        description="Measures real self-play simulations per second, worker throughput, and batching latency.",
        direct_command="zeropawn bench selfplay",
        requires_model="required",
        arguments=[
            OpArgument(
                "concurrency",
                "--concurrency",
                "int",
                24,
                "Concurrent self-play workers",
                min_val=1,
                max_val=64,
            ),
            OpArgument(
                "counted_sec",
                "--counted-sec",
                "float",
                10.0,
                "Counted benchmark duration in seconds",
                min_val=2.0,
                max_val=300.0,
            ),
            OpArgument(
                "trials", "--trials", "int", 1, "Number of benchmark trials", min_val=1, max_val=5
            ),
            OpArgument(
                "full_sims",
                "--full-sims",
                "int",
                50,
                "Full MCTS simulations per move",
                min_val=1,
                max_val=800,
            ),
        ],
    ),
    OperationDef(
        id="bench_batch_sizes",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Batch Size & Latency Sweep",
        description="Measures inference server throughput and latency across varying batch sizes.",
        direct_command="zeropawn bench batch-sizes",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="bench_pipelined",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Pipelined vs Non-Pipelined Dispatcher",
        description="Benchmarks dispatcher pipelining efficiency at concurrency 24.",
        direct_command="zeropawn bench pipelined",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="bench_dispatcher",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Dispatcher Ceiling Benchmark",
        description="Isolates IPC communication ceiling without neural network forward passes.",
        direct_command="zeropawn bench dispatcher",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="bench_ab",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="A/B Benchmark Discipline",
        description="Runs interleaved pairs with bracketing references to eliminate thermal drift.",
        direct_command="zeropawn bench ab",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="bench_concurrency",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Worker Concurrency Sweep",
        description="Sweeps worker counts {6, 8, 12, 16, 24} to locate throughput scaling peaks.",
        direct_command="zeropawn bench sweep-concurrency",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="bench_timeout",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Dispatcher Timeout Sweep",
        description="Measures batch size distribution across varying microsecond dispatch timeouts.",
        direct_command="zeropawn bench sweep-timeout",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="bench_batch_sweep",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Extended Batch Scaling Sweep (B=1 to 256)",
        description="Measures pure CUDA inference latency and full MCTS search scaling across configurable batch sizes.",
        direct_command="zeropawn bench batch-sweep",
        requires_model="required",
        arguments=[
            OpArgument(
                "batch_sizes",
                "--batch-sizes",
                "str",
                "1,4,8,16,24,32,64,128,256",
                "Comma-separated batch sizes to sweep",
            ),
            OpArgument(
                "sims",
                "--sims",
                "int",
                400,
                "MCTS simulations per tested batch size",
                min_val=10,
                max_val=2000,
            ),
        ],
    ),
    OperationDef(
        id="bench_concurrency_sweep",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="Worker Concurrency & Pipelining Delay Sweep",
        description="Sweeps concurrent workers (1-32) and batching delays (0.5-3.0ms) to identify throughput peaks.",
        direct_command="zeropawn bench concurrency-sweep",
        requires_model="required",
        arguments=[
            OpArgument(
                "concurrencies",
                "--concurrencies",
                "str",
                "1,2,4,8,16,24,32",
                "Comma-separated worker counts to sweep",
            ),
            OpArgument(
                "delays",
                "--delays",
                "str",
                "0.5,1.0,2.0,2.5,3.0",
                "Comma-separated batch timeouts in ms to sweep",
            ),
        ],
    ),
    OperationDef(
        id="bench_verify",
        category_id=5,
        category_name="Inference & Benchmarking",
        name="End-to-End Pipeline Verification",
        description="Runs full verification across neural inference, MCTS search, tactical puzzles, and match play.",
        direct_command="zeropawn bench verify",
        requires_model="required",
        arguments=[],
    ),
    # ---------------- Category 6: MCTS & Search Diagnostics ----------------
    OperationDef(
        id="puzzles_eval",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Tactical Puzzle Evaluation",
        description="Benchmarks policy top-1 and MCTS solve rate on tactical puzzle collections.",
        direct_command="zeropawn puzzles eval",
        requires_model="required",
        arguments=[
            OpArgument(
                "limit",
                "-n",
                "int",
                50,
                "Number of puzzles to test (0 for all)",
                min_val=0,
                max_val=100000,
            ),
            OpArgument(
                "mcts_puzzles",
                "--mcts-puzzles",
                "int",
                20,
                "Number of puzzles evaluated with MCTS",
                min_val=0,
                max_val=10000,
            ),
            OpArgument(
                "mcts_sims",
                "--mcts-sims",
                "int",
                50,
                "MCTS simulations per puzzle",
                min_val=1,
                max_val=1000,
            ),
            OpArgument("puzzle_file", "-p", "path", "data/puzzles", "Puzzle CSV file or directory"),
            OpArgument(
                "search_mode",
                "--search-mode",
                "choice",
                "serial",
                "MCTS search mode (serial or batched)",
                choices=["serial", "batched"],
            ),
            OpArgument(
                "mcts_batch_size",
                "--mcts-batch-size",
                "str",
                "auto",
                "Inference batch size ('auto' or integer 1-256)",
            ),
        ],
    ),
    OperationDef(
        id="puzzles_solve",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Interactive Puzzle Solver",
        description="Interactive tactical training where user guesses moves against AI engine.",
        direct_command="zeropawn puzzles solve",
        requires_model="required",
        arguments=[
            OpArgument(
                "human_guess",
                "--human-guess",
                "bool",
                True,
                "Guess moves interactively before seeing AI solution",
            ),
        ],
    ),
    OperationDef(
        id="puzzles_learn",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Learn from Tactical Mistakes (Live Active Learning)",
        description="Mines tactical blunders from puzzles and trains neural network live with instant re-test verification.",
        direct_command="zeropawn puzzles learn",
        requires_model="required",
        arguments=[
            OpArgument(
                "curriculum",
                "--curriculum",
                "bool",
                False,
                "Autonomous curriculum mode (1000 -> 3000 Elo progression)",
            ),
            OpArgument(
                "start_rating",
                "--start-rating",
                "int",
                1000,
                "Curriculum starting Elo rating",
                min_val=400,
                max_val=3500,
            ),
            OpArgument(
                "end_rating",
                "--end-rating",
                "int",
                3000,
                "Curriculum target Elo rating",
                min_val=400,
                max_val=3500,
            ),
            OpArgument(
                "tier_puzzles",
                "--tier-puzzles",
                "int",
                250,
                "Puzzles per curriculum tier",
                min_val=10,
                max_val=10000,
            ),
            OpArgument(
                "count",
                "-n",
                "int",
                250,
                "Number of puzzles in single-set mode",
                min_val=10,
                max_val=100000,
            ),
            OpArgument(
                "batch_size",
                "-b",
                "int",
                16,
                "Number of mistakes before each gradient update",
                min_val=4,
                max_val=256,
            ),
            OpArgument(
                "lr", "--lr", "float", 2e-5, "Fine-tuning learning rate", min_val=1e-7, max_val=0.01
            ),
            OpArgument(
                "anchor_ratio",
                "--anchor-ratio",
                "float",
                0.50,
                "Fraction of batch reserved for anchor regularization",
                min_val=0.0,
                max_val=0.9,
            ),
            OpArgument(
                "puzzles_file",
                "-p",
                "path",
                "data/puzzles/tactics_train_300k.csv",
                "Puzzle CSV file (auto-downloads if missing)",
            ),
            OpArgument(
                "benchmark",
                "--benchmark",
                "bool",
                False,
                "Run 1,000-puzzle benchmark before and after",
            ),
        ],
    ),
    OperationDef(
        id="mcts_search",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="MCTS Position Search & Evaluation",
        description="Executes serial or batched MCTS search on any FEN position and reports best move and throughput.",
        direct_command="zeropawn search",
        requires_model="optional",
        arguments=[
            OpArgument(
                "fen",
                "--fen",
                "str",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "FEN string of position to evaluate",
            ),
            OpArgument(
                "sims", "--sims", "int", 400, "Number of MCTS simulations", min_val=1, max_val=50000
            ),
            OpArgument(
                "mode",
                "--mode",
                "choice",
                "batched",
                "MCTS search mode (serial or batched)",
                choices=["batched", "serial"],
            ),
            OpArgument(
                "batch_size",
                "--batch-size",
                "str",
                "auto",
                "Search batch size ('auto' or integer 1-256)",
            ),
            OpArgument(
                "device", "--device", "choice", "cuda", "Inference device", choices=["cuda", "cpu"]
            ),
        ],
    ),
    OperationDef(
        id="diagnose_lag",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Model Version Lag Diagnostics",
        description="Measures version lag between training updates and worker inference caching.",
        direct_command="zeropawn diagnose lag",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="diagnose_cache",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="NN Cache vs Ply Curve",
        description="Measures neural network transposition cache entries and hit rate vs ply depth.",
        direct_command="zeropawn diagnose cache",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="tablebase_validate",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Offline Tablebase Validation",
        description="Validates tablebase win/draw/loss adjudication against Stockfish ground truth.",
        direct_command="zeropawn tablebase validate",
        requires_model="none",
        arguments=[
            OpArgument(
                "max_games", "--max-games", "int", 2, "Games to validate", min_val=1, max_val=100
            ),
        ],
    ),
    OperationDef(
        id="tablebase_feasibility",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Tablebase Adjudication Feasibility",
        description="Analyzes selfplay game plies to measure tablebase early adjudication hit rates.",
        direct_command="zeropawn tablebase feasibility",
        requires_model="none",
        arguments=[
            OpArgument(
                "max_games", "--max-games", "int", 2, "Games to inspect", min_val=1, max_val=100
            ),
        ],
    ),
    OperationDef(
        id="tablebase_download",
        category_id=6,
        category_name="MCTS & Search Diagnostics",
        name="Download & Verify Syzygy Tablebases",
        description="Downloads and verifies 145 Syzygy 3-4-5 piece WDL (.rtbw) tablebase files.",
        direct_command="zeropawn tablebase download",
        requires_model="none",
        arguments=[],
    ),
    # ---------------- Category 7: Tests & Validation ----------------
    OperationDef(
        id="test_suites",
        category_id=7,
        category_name="Tests & Validation",
        name="Run Pytest Test Suites",
        description="Executes targeted or full pytest test suites across the engine and training components.",
        direct_command="zeropawn test [--suite <S>]",
        requires_model="none",
        arguments=[
            OpArgument(
                "suite",
                "--suite",
                "choice",
                "fast",
                "Target test suite",
                choices=[
                    "fast",
                    "all",
                    "puzzles",
                    "match",
                    "mcts",
                    "replay",
                    "supervised",
                    "rl",
                    "cli",
                ],
            ),
            OpArgument(
                "filter", "-k", "str", None, "Pytest keyword filter (e.g. 'mcts or puzzles')"
            ),
        ],
    ),
    OperationDef(
        id="test_soak",
        category_id=7,
        category_name="Tests & Validation",
        name="72-Hour Soak Test Runner",
        description="Monitors self-play process RSS and VRAM memory stability over extended runtimes.",
        direct_command="zeropawn test soak",
        requires_model="required",
        arguments=[
            OpArgument(
                "hours",
                "--hours",
                "float",
                1.0,
                "Test duration in hours",
                min_val=0.01,
                max_val=72.0,
            ),
        ],
    ),
    OperationDef(
        id="test_differential",
        category_id=7,
        category_name="Tests & Validation",
        name="Differential Test vs Python-Chess",
        description="Differential validation against python-chess across random games and underpromotions.",
        direct_command="zeropawn test differential",
        requires_model="none",
        arguments=[
            OpArgument(
                "games", "--games", "int", 100, "Random games to compare", min_val=10, max_val=10000
            ),
        ],
    ),
    OperationDef(
        id="test_guardrails",
        category_id=7,
        category_name="Tests & Validation",
        name="Correctness Guardrails Verification",
        description="Validates value head signs, move probability bounds, and invariance checks.",
        direct_command="zeropawn test guardrails",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="test_parity",
        category_id=7,
        category_name="Tests & Validation",
        name="Terminal Output Parity Gate",
        description="Verifies normalized output parity between direct Python commands and chess.cmd.",
        direct_command="zeropawn test parity",
        requires_model="none",
        arguments=[
            OpArgument("op", "--op", "str", "test-fast", "Specific operation to check"),
        ],
    ),
    OperationDef(
        id="test_connect4",
        category_id=7,
        category_name="Tests & Validation",
        name="Connect 4 Pipeline Smoke Test",
        description="Validates the self-play & RL infrastructure on the Connect 4 test environment.",
        direct_command="zeropawn test connect4",
        requires_model="none",
        arguments=[],
    ),
    # ---------------- Category 8: Presets ----------------
    OperationDef(
        id="presets_manage",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Manage Workload Presets",
        description="Create, list, load, edit, duplicate, rename, delete, import, or export presets.",
        direct_command="zeropawn presets list",
        requires_model="none",
        arguments=[],
    ),
    # ---------------- Category 9: Logs & Diagnostics ----------------
    OperationDef(
        id="status_dashboard",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="System Health & Telemetry Status",
        description="Displays GPU, RAM, CPU, disk usage, active runs, replay buffers, and checkpoints.",
        direct_command="zeropawn status",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="diagnose_resign",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Resignation Diagnosis & Value Head",
        description="Analyzes selfplay resignation false-positive rates and value head calibration.",
        direct_command="zeropawn diagnose resign",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="diagnose_valuesigns",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Value Sign Perspective Check",
        description="Tests value head perspective and sign consistency across canonical winning/losing positions.",
        direct_command="zeropawn diagnose value-signs",
        requires_model="required",
        arguments=[],
    ),
    OperationDef(
        id="profile_throughput",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Detailed Performance Profiling",
        description="Profiles throughput, CUDA events, real pacing, and wall-clock breakdown.",
        direct_command="zeropawn profile throughput",
        requires_model="none",
        arguments=[
            OpArgument(
                "duration",
                "--duration",
                "float",
                2.0,
                "Benchmark duration in seconds",
                min_val=0.1,
                max_val=3600.0,
            ),
            OpArgument(
                "games", "--games", "int", 8, "Concurrent game workers", min_val=1, max_val=128
            ),
        ],
    ),
    OperationDef(
        id="hardware_profile",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Hardware Profile & Autotuning Targets",
        description="Inspects active hardware fingerprint, autotuned batch size targets, or triggers recalibration.",
        direct_command="zeropawn hardware profile",
        requires_model="none",
        arguments=[
            OpArgument("show", "--show", "bool", True, "Display the active hardware profile"),
            OpArgument(
                "recalibrate",
                "--recalibrate",
                "bool",
                False,
                "Run live calibration benchmark and overwrite profile",
            ),
            OpArgument("reset", "--reset", "bool", False, "Reset profile to force recalibration"),
            OpArgument(
                "workload",
                "--workload",
                "choice",
                "all",
                "Target workload to inspect",
                choices=["all", "uci", "puzzles", "match", "selfplay"],
            ),
        ],
    ),
    OperationDef(
        id="hardware_info",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Low-Level System Hardware Detection",
        description="Detects CPU cores/threads, RAM, GPU, VRAM, BF16/FP16, and C++ extensions.",
        direct_command="zeropawn hardware info",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="hardware_calibrate",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Calibrate Hardware Autotuner",
        description="Runs micro-benchmarks across batch sizes to calibrate optimal targets for current machine.",
        direct_command="zeropawn hardware calibrate",
        requires_model="none",
        arguments=[
            OpArgument(
                "budget",
                "--budget",
                "float",
                5.0,
                "Calibration time budget in seconds",
                min_val=1.0,
                max_val=60.0,
            ),
        ],
    ),
    # ---------------- Category 10: Maintenance & Utilities ----------------
    OperationDef(
        id="cleanup_junk",
        category_id=10,
        category_name="Maintenance & Utilities",
        name="Clean Up Disk Junk & Cache Buffers",
        description="Safely removes replay buffers, temporary event logs, old checkpoints, and compiler caches.",
        direct_command="zeropawn cleanup",
        requires_model="none",
        arguments=[
            OpArgument("force", "--force", "bool", False, "Proceed without dry-run confirmation"),
            OpArgument("all", "--all", "bool", False, "Clean buffers, old checkpoints, and caches"),
        ],
    ),
    OperationDef(
        id="uci_engine",
        category_id=10,
        category_name="Maintenance & Utilities",
        name="Run UCI Protocol Engine for GUIs",
        description="Runs ZeroPawn in standard Universal Chess Interface protocol mode for Arena/Cutechess/Lichess.",
        direct_command="zeropawn uci",
        requires_model="required",
        arguments=[
            OpArgument(
                "sims",
                "--sims",
                "int",
                50,
                "Default MCTS simulations per move",
                min_val=1,
                max_val=10000,
            ),
        ],
    ),
    # ---------------- Extended Operations ----------------
    # Category 2: Data Extended
    OperationDef(
        id="data_validate",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Validate Dataset Integrity",
        description="Verify checksums, schemas, and chunk readability for binary datasets.",
        direct_command="zeropawn data validate --dir <PATH>",
        requires_model="none",
        arguments=[
            OpArgument(
                "dir", "--dir", "path", "data/dataset_training", "Dataset directory to validate"
            ),
        ],
    ),
    OperationDef(
        id="data_leakage",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Check Train/Val Data Leakage",
        description="Check position hashes between train and val splits to ensure strict isolation.",
        direct_command="zeropawn data check-leakage --train-dir <PATH> --val-dir <PATH>",
        requires_model="none",
        arguments=[
            OpArgument(
                "train_dir",
                "--train-dir",
                "path",
                "data/dataset_training/train",
                "Train dataset directory",
            ),
            OpArgument(
                "val_dir", "--val-dir", "path", "data/dataset_training/val", "Val dataset directory"
            ),
        ],
    ),
    OperationDef(
        id="data_merge",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Merge Datasets",
        description="Merge two or more binary datasets into a single unified dataset.",
        direct_command="zeropawn data merge --inputs <D1> <D2> --output <OUT>",
        requires_model="none",
        arguments=[
            OpArgument(
                "inputs",
                "--inputs",
                "str",
                "data/dataset_training/train data/dataset_training/val",
                "Input dataset directories",
                required=True,
            ),
            OpArgument(
                "output",
                "--output",
                "path",
                "data/dataset_merged",
                "Merged output directory",
                required=True,
            ),
        ],
    ),
    OperationDef(
        id="data_split",
        category_id=2,
        category_name="Self-Play & Data Generation",
        name="Split Dataset",
        description="Split a dataset into train and validation splits.",
        direct_command="zeropawn data split --input <DIR> --output <OUT>",
        requires_model="none",
        arguments=[
            OpArgument(
                "input",
                "--input",
                "path",
                "data/dataset_training",
                "Source dataset directory",
                required=True,
            ),
            OpArgument(
                "output",
                "--output",
                "path",
                "data/dataset_split",
                "Output dataset directory",
                required=True,
            ),
            OpArgument(
                "val_ratio",
                "--val-ratio",
                "float",
                0.1,
                "Validation split ratio",
                min_val=0.01,
                max_val=0.5,
            ),
        ],
    ),
    # Category 3: Models Extended
    OperationDef(
        id="models_validate",
        category_id=3,
        category_name="Models & Checkpoints",
        name="Validate Model Numerical Stability",
        description="Verify forward pass, NaN/Inf checks, and output probabilities on a model.",
        direct_command="zeropawn models validate [PATH]",
        requires_model="optional",
        arguments=[],
    ),
    OperationDef(
        id="models_export",
        category_id=3,
        category_name="Models & Checkpoints",
        name="Export Model Weights",
        description="Export PyTorch checkpoint to clean inference weights or TorchScript.",
        direct_command="zeropawn models export --output <OUT> [checkpoint]",
        requires_model="required",
        arguments=[
            OpArgument(
                "output",
                "--output",
                "path",
                "models/exported_model.pt",
                "Destination export path (.pt)",
                required=True,
            ),
            OpArgument(
                "weights_only",
                "--weights-only",
                "bool",
                True,
                "Strip optimizer, scheduler, and RNG state",
            ),
        ],
    ),
    OperationDef(
        id="models_compat",
        category_id=3,
        category_name="Models & Checkpoints",
        name="Check Checkpoint Compatibility",
        description="Check whether a checkpoint matches the current architecture specification.",
        direct_command="zeropawn models check-compat [PATH]",
        requires_model="optional",
        arguments=[],
    ),
    # Category 8: Config Extended
    OperationDef(
        id="config_show",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Show Configuration",
        description="Show all configured options, defaults, and current overrides.",
        direct_command="zeropawn config show",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="config_get",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Get Configuration Value",
        description="Get the effective value and provenance of a specific configuration option.",
        direct_command="zeropawn config get <KEY>",
        requires_model="none",
        arguments=[
            OpArgument(
                "key",
                "key",
                "str",
                "training.max_hours",
                "Configuration key or alias",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    OperationDef(
        id="config_set",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Set Configuration Value",
        description="Validate and persistently set a configuration option.",
        direct_command="zeropawn config set <KEY> <VALUE>",
        requires_model="none",
        arguments=[
            OpArgument(
                "key",
                "key",
                "str",
                "training.max_hours",
                "Configuration key or alias",
                required=True,
                is_positional=True,
            ),
            OpArgument(
                "val", "val", "str", "72.0", "Value to set", required=True, is_positional=True
            ),
        ],
    ),
    OperationDef(
        id="config_validate",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Validate Configuration",
        description="Validate all configuration settings against constraints.",
        direct_command="zeropawn config validate",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="config_diff",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Diff Configuration vs Defaults",
        description="Display differences between effective configuration and built-in defaults.",
        direct_command="zeropawn config diff",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="config_export",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Export Configuration",
        description="Export effective configuration to a JSON file.",
        direct_command="zeropawn config export <FILE>",
        requires_model="none",
        arguments=[
            OpArgument(
                "output",
                "output",
                "path",
                "configs/exported_config.json",
                "Output file path",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    OperationDef(
        id="presets_diff",
        category_id=8,
        category_name="Presets & Experiment Configurations",
        name="Diff Two Presets",
        description="Compare settings between two presets and show exact differences.",
        direct_command="zeropawn presets diff <P1> <P2>",
        requires_model="none",
        arguments=[
            OpArgument(
                "preset1",
                "preset1",
                "str",
                "quick-smoke",
                "First preset name",
                required=True,
                is_positional=True,
            ),
            OpArgument(
                "preset2",
                "preset2",
                "str",
                "fast-tests",
                "Second preset name",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    # Category 9: Runs & Logs Extended
    OperationDef(
        id="runs_list",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="List Training & Eval Runs",
        description="List all historical training and evaluation runs with telemetry.",
        direct_command="zeropawn runs list",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="runs_inspect",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Inspect Run Details",
        description="Display detailed metrics, checkpoints, and telemetry for a specific run.",
        direct_command="zeropawn runs inspect [RUN_ID]",
        requires_model="none",
        arguments=[
            OpArgument(
                "run_id",
                "run_id",
                "str",
                "phase4_rl",
                "Run directory or ID",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    OperationDef(
        id="runs_recover",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Recover Interrupted Run",
        description="Detect interrupted training runs and display resume commands.",
        direct_command="zeropawn runs recover [RUN_ID]",
        requires_model="none",
        arguments=[
            OpArgument(
                "run_id",
                "run_id",
                "str",
                "phase4_rl",
                "Run directory to recover",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    OperationDef(
        id="logs_tail",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Live Tail Logs",
        description="Tail active log file in real-time.",
        direct_command="zeropawn logs tail",
        requires_model="none",
        arguments=[
            OpArgument(
                "lines",
                "--lines",
                "int",
                50,
                "Number of initial lines to show",
                min_val=5,
                max_val=1000,
            ),
        ],
    ),
    OperationDef(
        id="logs_export",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Export Logs",
        description="Package log files into a consolidated archive or file.",
        direct_command="zeropawn logs export <OUT>",
        requires_model="none",
        arguments=[
            OpArgument(
                "output",
                "output",
                "path",
                "logs/exported_logs.zip",
                "Export output path",
                required=True,
                is_positional=True,
            ),
        ],
    ),
    OperationDef(
        id="diagnose_env",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Inspect Environment & Hardware",
        description="Detailed inspection of Python, PyTorch, CUDA, toolchains, and hardware.",
        direct_command="zeropawn diagnose env",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="diagnose_health",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="System Health Check",
        description="Comprehensive health check across dependencies, C++ bindings, models, and disk space.",
        direct_command="zeropawn diagnose health",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="diagnose_bundle",
        category_id=9,
        category_name="Logs, Telemetry & Diagnostics",
        name="Collect Diagnostic Bundle",
        description="Collect system info, health checks, config, and logs into a diagnostic zip archive.",
        direct_command="zeropawn diagnose bundle",
        requires_model="none",
        arguments=[],
    ),
    # Category 10: Build & C++ Native
    OperationDef(
        id="build_test",
        category_id=10,
        category_name="Maintenance & Utilities",
        name="Run Native C++ Extension Tests",
        description="Execute C++ unit tests binary (zeropawn_tests.exe) for bitboard and movegen contracts.",
        direct_command="zeropawn build test",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="build_contracts",
        category_id=10,
        category_name="Maintenance & Utilities",
        name="Verify C++/Python Binding Contracts",
        description="Check pybind11 module exported types and ensure strict ABI compatibility.",
        direct_command="zeropawn build contracts",
        requires_model="none",
        arguments=[],
    ),
    OperationDef(
        id="build_extension",
        category_id=10,
        category_name="Maintenance & Utilities",
        name="Compile Native C++ Extension",
        description="Build or rebuild the C++ movegen and search accelerator extension.",
        direct_command="zeropawn build ext",
        requires_model="none",
        arguments=[],
    ),
]

# Quick lookup catalog by op_id
OPERATIONS_CATALOG: Dict[str, OperationDef] = {op.id: op for op in OPERATIONS_REGISTRY}


_IS_ENRICHED: bool = False


def enrich_operations_registry_from_parser() -> None:
    """Enrich OPERATIONS_REGISTRY with all discoverable parser subcommands and arguments."""
    global _IS_ENRICHED
    if _IS_ENRICHED:
        return
    try:
        from python.cli.discovery import get_command_catalog
        from python.cli.combinations import _get_sample_value_for_argument

        catalog = get_command_catalog()

        # Index existing ops by command path
        op_by_cmd: Dict[str, List[OperationDef]] = {}
        for op in OPERATIONS_REGISTRY:
            cmd_parts = [
                p
                for p in op.direct_command.split()
                if p not in ("zeropawn", "chess") and not p.startswith(("-", "<", "["))
            ]
            path_str = " ".join(cmd_parts)
            op_by_cmd.setdefault(path_str, []).append(op)

        # For each command in catalog, ensure it exists in registry and has full arguments
        for node in catalog.all_leaf_commands():
            node_path_str = " ".join(node.path)
            matching_ops = op_by_cmd.get(node_path_str, [])

            if matching_ops:
                # Enrich existing op(s) with missing arguments from parser
                for op in matching_ops:
                    existing_flags = {a.flag for a in op.arguments}
                    existing_names = {a.name for a in op.arguments}
                    num_existing_pos = len(
                        [a for a in op.arguments if getattr(a, "is_positional", False)]
                    )
                    for arg_meta in node.arguments:
                        if arg_meta.is_inherited:
                            continue
                        if arg_meta.is_positional and num_existing_pos >= len(
                            node.positional_arguments
                        ):
                            continue
                        flag_str = arg_meta.flags[0] if arg_meta.flags else arg_meta.name
                        if flag_str not in existing_flags and arg_meta.name not in existing_names:
                            default_val = arg_meta.default
                            if arg_meta.required and default_val is None:
                                default_val = _get_sample_value_for_argument(arg_meta)
                            op.arguments.append(
                                OpArgument(
                                    name=arg_meta.name,
                                    flag=flag_str,
                                    type=arg_meta.type,
                                    default=default_val,
                                    help=arg_meta.help,
                                    required=arg_meta.required,
                                    choices=arg_meta.choices,
                                    min_val=arg_meta.min_val,
                                    max_val=arg_meta.max_val,
                                    is_positional=arg_meta.is_positional,
                                )
                            )
                            if arg_meta.is_positional:
                                num_existing_pos += 1
            else:
                # Newly discovered command not in legacy registry: add full OperationDef
                op_args = []
                for arg_meta in node.arguments:
                    if arg_meta.is_inherited:
                        continue
                    flag_str = arg_meta.flags[0] if arg_meta.flags else arg_meta.name
                    default_val = arg_meta.default
                    if arg_meta.required and default_val is None:
                        default_val = _get_sample_value_for_argument(arg_meta)
                    op_args.append(
                        OpArgument(
                            name=arg_meta.name,
                            flag=flag_str,
                            type=arg_meta.type,
                            default=default_val,
                            help=arg_meta.help,
                            required=arg_meta.required,
                            choices=arg_meta.choices,
                            min_val=arg_meta.min_val,
                            max_val=arg_meta.max_val,
                            is_positional=arg_meta.is_positional,
                        )
                    )

                new_id = (
                    node.full_command.replace("zeropawn ", "").replace(" ", "_").replace("-", "_")
                )
                if new_id in OPERATIONS_CATALOG:
                    new_id = f"op_{new_id}"

                new_op = OperationDef(
                    id=new_id,
                    category_id=node.category_id,
                    category_name=node.category_name,
                    name=node.name.replace("-", " ").capitalize()
                    if node.name
                    else node.full_command,
                    description=node.description or f"Execute {node.full_command}",
                    direct_command=node.full_command,
                    requires_model=node.requires_model,
                    arguments=op_args,
                )
                OPERATIONS_REGISTRY.append(new_op)
                OPERATIONS_CATALOG[new_op.id] = new_op

        _IS_ENRICHED = True
    except Exception:
        # Fallback to static registry if parser cannot be built during early bootstrap
        pass


def get_operation_by_id(op_id: str) -> Optional[OperationDef]:
    """Look up an operation by its unique ID."""
    enrich_operations_registry_from_parser()
    return OPERATIONS_CATALOG.get(op_id)


def get_operations_by_category(category_id: int) -> List[OperationDef]:
    """Return all operations belonging to a specific category."""
    enrich_operations_registry_from_parser()
    return [op for op in OPERATIONS_REGISTRY if op.category_id == category_id]
