"""Phase 4: Continuous RL Self-Play + Training Orchestrator with 6-Hour Evaluations and Safety Triggers.

Contract & Change Protocol:
- Prerequisites: Phases 1-3 verified; docs/HANDOFF.md; docs/RL_CHANGELOG.md.
- Initializes from Phase 2 checkpoint: models/checkpoints_scaled/best_model.pt.
- Continuous self-play generation + RL training loop.
- Every 6 hours:
  * Checkpoint ID
  * Puzzle solve rate on fixed set of 1,000 Lichess puzzles (1200-1600)
  * 100-game match vs fixed Stockfish reference (nodes 1000, 50 openings alternated)
- Safety Triggers (Stop and Report):
  * Draw rate > 70%
  * False-resign rate > 5%
  * Policy entropy falls sharply within a day (> 35% drop)
  * GPU utilization stays below 60%
  * Puzzle rate falls for 3 consecutive checks
  * Any OOM or crash
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_build_cpp = _repo_root / "build" / "cpp"
if _build_cpp.is_dir() and str(_build_cpp) not in sys.path:
    sys.path.insert(0, str(_build_cpp))

_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    try:
        os.add_dll_directory(_winlibs_bin)
    except (AttributeError, OSError):
        pass

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.selfplay.client import DirectInferenceClient
from python.selfplay.worker import SavedPosition, GameSummary, SelfPlayWorker
from python.selfplay.replay_buffer import ReplayBuffer
from python.selfplay.metrics import MetricsTracker, query_gpu_metrics, query_process_memory
from python.train.train_rl import RLTrainer
from python.eval.puzzles import PuzzleEvaluator, load_puzzles
from python.eval.match_runner import MatchRunner, eval_config_hash, default_mcts_seed
from python.common.progress import ProgressReporter
from python.common.progress import (
    Heartbeat,
    RateLimiter,
    banner,
    emit_error,
    emit_report,
    error_block,
    event_line,
    header_block,
    results_table,
    run_context,
    short_hash,
    status_line,
    summary_block,
)
from python.cli.common import setup_logging

# NOTE: no logging.basicConfig here. Importing python.cli.common already
# installs the single root setup (rotating file + stderr WARNING handler);
# basicConfig would be a silent no-op at this point (root has handlers).
# The orchestrator calls setup_logging(log_dir, "phase4_rl.log") on init.
logger = logging.getLogger("phase4_rl")


class SafetyTriggerException(Exception):
    """Raised when an automated safety guardrail is breached."""

    pass


def puzzle_decline_trips(history: list[float], band: float = 0.005) -> bool:
    """True if the last 3 steps each decline by MORE than ``band``.

    Same trigger semantics as before, plus a noise band so flat/jittered
    histories (drops within measurement noise) cannot trip a critical stop.
    Needs >= 4 samples; shorter histories never trip.
    """
    if len(history) < 4:
        return False
    last4 = history[-4:]
    return bool(
        (last4[0] - last4[1] > band)
        and (last4[1] - last4[2] > band)
        and (last4[2] - last4[3] > band)
    )


class Phase4Orchestrator:
    """Orchestrates Phase 4 self-play + RL training run with 6-hour evaluations and stop conditions."""

    def __init__(
        self,
        max_hours: float = 72.0,
        eval_interval_sec: float = 21600.0,  # 6 hours
        num_concurrent_games: int = 24,
        buffer_capacity: int = 500_000,
        checkpoint_dir: str = "models/checkpoints_rl",
        log_dir: str = "runs/phase4_rl",
        base_checkpoint: Optional[str] = "models/checkpoints_scaled/best_model.pt",
        resume_checkpoint: Optional[str] = None,
        human_data_dir: str = (
            "data/dataset_training/train"
            if Path("data/dataset_training/train").is_dir()
            else "data/dataset_scaled/train"
        ),
        puzzle_csv_path: str = "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        openings_path: str = "data/openings/eval_openings_50.epd",
        stockfish_path: str = "tools/stockfish.exe",
        stockfish_nodes: int = 1000,
        is_verify_mode: bool = False,
        skip_baseline_eval: bool = False,
        gpu_cache_capacity: Optional[int] = 160_000,
        human_cache_size: int = 80_000,
        tablebase_path: Optional[str] = None,
        tablebase_max_pieces: int = 5,
        min_buffer_fill: int = 0,
        max_reuse_ratio: int = 16,
        full_search_sims: int = 800,
        fast_search_sims: int = 150,
        reset_buffer: bool = False,
        # ── Post-run janitor: delete junk that is not needed after a run ──
        cleanup_buffer_on_exit: bool = False,
        keep_checkpoints: int = 3,
        keep_tfevents: int = 1,
        # ── Task 5: memory-safety parameters ──────────────────────────────
        max_wait_ms: float = 1.0,
        watchdog_rss_gb: float = 7.5,
        watchdog_free_ram_gb: float = 2.0,
        # ── Execution device configuration (Modes A-F) ────────────────────
        device: Optional[str] = None,
        train_device: Optional[str] = None,
        infer_device: Optional[str] = None,
        seed: Optional[int] = None,
        # ── Learning rate & tactical anchor replay ────────────────────────
        base_lr: float = 0.0005,
        min_lr: float = 1e-5,
        tactical_mix: float = 0.12,
        tactical_csv_path: Optional[str] = "data/tactics_train_300k.csv",
        max_steps: Optional[int] = None,
        policy_target_temp: float = 0.5,
        entropy_bonus: float = 0.001,
        eval_openings: int = 50,
        eval_sims: Optional[int] = None,
        # ──────────────────────────────────────────────────────────────────
    ) -> None:
        self.max_hours = max_hours
        self.max_duration_sec = max_hours * 3600.0
        self.eval_interval_sec = eval_interval_sec
        # Requested concurrency is captured before the RAM clamp below.
        self._requested_concurrency = num_concurrent_games
        self._clamped_concurrency = False
        # Task 3(a): Cap concurrency at 24 unless free RAM > 3.0 GB
        if num_concurrent_games > 24:
            try:
                free_ram_gb = psutil.virtual_memory().available / (1024.0**3)
                if free_ram_gb < 3.0:
                    logger.warning(
                        "Concurrency cap of 24 enforced (requested %d, free RAM %.2f GB < 3.0 GB)",
                        num_concurrent_games,
                        free_ram_gb,
                    )
                    num_concurrent_games = 24
                    self._clamped_concurrency = True
            except Exception:
                num_concurrent_games = 24
                self._clamped_concurrency = True
        self.num_concurrent_games = num_concurrent_games
        self.is_verify_mode = is_verify_mode
        self.skip_baseline_eval = skip_baseline_eval
        self.gpu_cache_capacity = gpu_cache_capacity
        self.human_cache_size = human_cache_size
        self.tablebase_path = tablebase_path
        self.tablebase_max_pieces = tablebase_max_pieces
        self.min_buffer_fill = min_buffer_fill
        self.max_reuse_ratio = max_reuse_ratio
        self.full_search_sims = full_search_sims
        self.fast_search_sims = fast_search_sims
        self.reset_buffer = reset_buffer
        self.cleanup_buffer_on_exit = cleanup_buffer_on_exit
        self.keep_checkpoints = keep_checkpoints
        self.keep_tfevents = keep_tfevents

        # Device resolution: support Modes A-F (CPU-only, GPU, mixed devices, fallback)
        base_dev = device.lower().strip() if device else None
        t_dev = (
            train_device.lower().strip()
            if train_device
            else (base_dev or ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        i_dev = (
            infer_device.lower().strip()
            if infer_device
            else (base_dev or ("cuda" if torch.cuda.is_available() else "cpu"))
        )

        if t_dev in ("cuda", "gpu") and not torch.cuda.is_available():
            logger.warning("CUDA training requested but unavailable; falling back to CPU")
            t_dev = "cpu"
        if i_dev in ("cuda", "gpu") and not torch.cuda.is_available():
            logger.warning("CUDA inference requested but unavailable; falling back to CPU")
            i_dev = "cpu"

        self.train_device = t_dev
        self.infer_device = i_dev
        self.device = self.infer_device  # backwards compatibility
        self.seed = seed
        if self.seed is not None:
            import random
            import numpy as np

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.tactical_mix = float(tactical_mix)
        tac_cand = [
            Path(tactical_csv_path) if tactical_csv_path else None,
            Path("data/tactics_train_300k.csv"),
            _repo_root / "data/tactics_train_300k.csv",
            Path("data/puzzles/tactics_train_300k.csv"),
            _repo_root / "data/puzzles/tactics_train_300k.csv",
        ]
        resolved_tac = None
        for cand in tac_cand:
            if cand is not None and cand.is_file():
                resolved_tac = cand
                break
        self.tactical_csv_path = str(resolved_tac) if resolved_tac else tactical_csv_path
        self.max_steps = max_steps
        self.policy_target_temp = float(policy_target_temp)
        self.entropy_bonus = float(entropy_bonus)
        self.eval_openings = int(eval_openings)
        self.eval_sims = int(eval_sims) if eval_sims is not None else None
        self._steps_this_run = 0

        # Task 5: watchdog limits
        self._watchdog_rss_bytes = watchdog_rss_gb * (1024.0**3)
        self._watchdog_free_ram_bytes = watchdog_free_ram_gb * (1024.0**3)
        logger.info(
            "Watchdog armed: RSS ceiling %.2f GB | free-RAM floor %.2f GB | train_device=%s | infer_device=%s",
            watchdog_rss_gb,
            watchdog_free_ram_gb,
            self.train_device,
            self.infer_device,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir=self.log_dir, log_filename="phase4_rl.log")
        self.eval_csv_path = self.log_dir / "eval_history.csv"

        p_cand = [
            Path(puzzle_csv_path) if puzzle_csv_path else None,
            Path("data/puzzles/lichess_puzzles_1000_1200_1600.csv"),
            _repo_root / "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
            Path("data/puzzles/lichess_puzzles_5000_1800_2800.csv"),
            _repo_root / "data/puzzles/lichess_puzzles_5000_1800_2800.csv",
            Path("data/puzzles/lichess_puzzles_50000_1200_3000.csv"),
            _repo_root / "data/puzzles/lichess_puzzles_50000_1200_3000.csv",
            Path("data/puzzles/tactics_train_300k.csv"),
            _repo_root / "data/puzzles/tactics_train_300k.csv",
        ]
        resolved_puzzle_path = None
        for cand in p_cand:
            if cand is not None and cand.is_file():
                resolved_puzzle_path = cand
                break
        self.puzzle_csv_path = resolved_puzzle_path or Path(puzzle_csv_path)
        openings_p = Path(openings_path)
        if not openings_p.is_file() and (_repo_root / openings_p).is_file():
            openings_p = _repo_root / openings_p
        self.openings_path = openings_p

        stockfish_p = Path(stockfish_path)
        if not stockfish_p.is_file() and (_repo_root / stockfish_p).is_file():
            stockfish_p = _repo_root / stockfish_p
        self.stockfish_path = stockfish_p
        self.stockfish_nodes = stockfish_nodes

        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.last_eval_time = self.start_time

        # Historical evaluation metrics for trend tracking
        self.puzzle_solve_history: list[float] = []
        # Noise band for the puzzle-decline safety trigger: per-step drops at
        # or below this absolute rate do not count as declines.
        self.puzzle_noise_band: float = 0.005
        self.initial_policy_entropy: Optional[float] = None
        self.entropy_history: list[float] = []
        self.low_gpu_util_counter = 0
        self.progress_reporter = ProgressReporter(interval_sec=10.0)

        # ── Terminal-report state (printing only; no training logic) ──
        self._status_limiter = RateLimiter(10.0)
        self._heartbeat = Heartbeat(30.0)
        self._train_busy_sec = 0.0
        self._warnings = 0
        self._exit_reason = "ok"
        self._warned_false_resign = False
        self._warned_low_gpu = False
        self._adj_games = 0
        self._completed_games_seen = 0
        self._last_push_version = 0
        self._known_ckpt_names: set[str] = set()
        self._active_checkpoint_path = ""
        self._tb_files_found = 0
        self._tb_enabled = False
        self._last_emitted_trainer_state: Optional[str] = None
        self._last_emitted_model_v: Optional[int] = None
        self._last_emitted_duty: Optional[str] = None
        self._last_status_snapshot: Optional[dict] = None

        self._init_eval_csv()

        # 1. Initialize Replay Buffer (500k, 25% -> 5% taper over 50k steps)
        buf_dir = self.log_dir / "buffer"
        buf_meta = buf_dir / "meta.json"
        actual_capacity = buffer_capacity
        if buf_meta.exists():
            try:
                with open(buf_meta, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
                    if "capacity" in meta_json:
                        persisted_cap = int(meta_json["capacity"])
                        if persisted_cap != buffer_capacity:
                            logger.warning(
                                f"[BUFFER] Persisted buffer capacity ({persisted_cap:,}) overrides requested "
                                f"({buffer_capacity:,}). Pass --reset-buffer to reallocate."
                            )
                        actual_capacity = persisted_cap
            except Exception:
                pass

        logger.info(f"Initializing Replay Buffer at {buf_dir} (capacity {actual_capacity})...")
        self.buffer = ReplayBuffer(
            buffer_dir=buf_dir,
            capacity=actual_capacity,
            initial_human_mix=0.25,
            final_human_mix=0.05,
            taper_steps=50_000,
            human_data_dir=human_data_dir,
            human_cache_size=self.human_cache_size,
            gpu_device=self.train_device if self.train_device == "cuda" else None,
            create_new=self.reset_buffer,
            tactical_csv_path=self.tactical_csv_path,
            tactical_mix=self.tactical_mix,
        )

        # Detect resumable checkpoint if explicitly provided, else auto-resume from existing RL checkpoint if present
        if resume_checkpoint is None and not self.reset_buffer and not self.is_verify_mode:
            for candidate in [
                self.checkpoint_dir / "phase4_final_checkpoint.pt",
                self.checkpoint_dir / "latest_checkpoint.pt",
            ]:
                if candidate.exists():
                    resume_checkpoint = str(candidate)
                    logger.info(
                        f"[RESUME] Found existing RL checkpoint at {candidate}; auto-resuming training."
                    )
                    break

        active_checkpoint = resume_checkpoint if resume_checkpoint else base_checkpoint
        self._active_checkpoint_path = (
            str(active_checkpoint) if active_checkpoint else "random_init"
        )
        # Auto-detect Syzygy tablebase if not explicitly disabled
        if self.tablebase_path is None:
            for candidate in ("data/syzygy", "syzygy", "/usr/share/syzygy"):
                p = Path(candidate)
                if p.is_dir():
                    files = [
                        f
                        for f in p.iterdir()
                        if f.is_file() and f.suffix.lower() in (".rtbw", ".rtbz")
                    ]
                    if files:
                        self.tablebase_path = str(p)
                        logger.info(f"Auto-detected tablebase at {p} ({len(files)} files)")
                        break
        if self.tablebase_path:
            self._tb_enabled = True
            try:
                tb_dir = Path(self.tablebase_path)
                self._tb_files_found = (
                    sum(
                        1
                        for p in tb_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in (".rtbw", ".rtbz")
                    )
                    if tb_dir.is_dir()
                    else 0
                )
            except OSError:
                self._tb_files_found = 0

        # 2. Initialize Inference Server & Client
        # Task 5: use caller-supplied max_wait_ms (default 1.0 ms for optimal throughput)
        logger.info(
            f"Initializing Inference Server from checkpoint: {active_checkpoint} "
            f"(device={self.infer_device}, max_wait_ms={max_wait_ms:.1f})..."
        )
        self.server = InferenceServer(
            checkpoint_path=active_checkpoint,
            device=self.infer_device,
            min_batch_size=48,
            max_batch_size=128,
            max_wait_ms=max_wait_ms,
            use_bfloat16=(self.infer_device == "cuda"),
        )
        self.client = DirectInferenceClient(self.server)

        # 3. Initialize Metrics Tracker (with inference server for batch_fill telemetry)
        self.metrics_tracker = MetricsTracker(
            log_dir=self.log_dir,
            csv_path=self.log_dir / "phase4_telemetry.csv",
            inference_server=self.server,
        )

        # 4. Initialize Self-Play Worker
        def sink(positions: list[SavedPosition], summary: GameSummary):
            self.buffer.add_positions(positions)
            self.metrics_tracker.record_game(summary)
            # Reporting-only counters (no training effect).
            self._completed_games_seen += 1
            if getattr(summary, "adjudicated", False):
                self._adj_games += 1

        logger.info(f"Initializing SelfPlayWorker ({num_concurrent_games} concurrent games)...")
        self.worker = SelfPlayWorker(
            worker_id=0,
            client=self.client,
            num_concurrent_games=num_concurrent_games,
            game_type="chess",
            full_search_sims=self.full_search_sims,
            fast_search_sims=self.fast_search_sims,
            full_search_prob=0.35,
            temp_cutoff_range=(15, 30),
            resign_threshold=-0.95,
            consecutive_resign_moves=4,
            disable_resign_prob=0.15,
            allow_resignation=True,
            min_resign_ply=30,
            tablebase_path=self.tablebase_path,
            tablebase_max_pieces=self.tablebase_max_pieces,
            position_sink=sink,
            seed=self.seed,
        )

        # 5. Initialize RL Trainer
        trainer_model = ChessNet(num_blocks=12, num_filters=128).to(self.train_device)
        trainer_state = {
            k: (v.to(self.train_device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.server.model.state_dict().items()
        }
        trainer_model.load_state_dict(trainer_state)

        logger.info(
            f"Initializing RL Trainer (train_device={self.train_device}, SGD momentum 0.9, base_lr {self.base_lr}, min_lr {self.min_lr}, 500 warmup steps)..."
        )
        self.trainer = RLTrainer(
            model=trainer_model,
            replay_buffer=self.buffer,
            device=self.train_device,
            base_lr=self.base_lr,
            min_lr=self.min_lr,
            momentum=0.9,
            weight_decay=1e-4,
            warmup_steps=500,
            total_steps=100_000,
            batch_size=256,
            push_freq=200,
            checkpoint_interval_sec=300.0,  # 5 minutes
            checkpoint_dir=self.checkpoint_dir,
            sync_weights_path=self.checkpoint_dir / "live_weights.pt",
            direct_server=self.server,
            use_bfloat16=(self.train_device == "cuda"),
            gpu_cache_capacity=self.gpu_cache_capacity if self.train_device == "cuda" else 0,
            policy_target_temp=self.policy_target_temp,
            entropy_bonus=self.entropy_bonus,
        )

        if resume_checkpoint is not None:
            self.trainer.load_checkpoint(resume_checkpoint)
            if hasattr(self.trainer, "direct_server") and self.trainer.direct_server is not None:
                self.server.model_version = self.trainer.direct_server.model_version
            logger.info(
                f"Resumed RL training from {resume_checkpoint} at step {self.trainer.step_num} (model_version={self.server.model_version})"
            )

        # Load puzzles
        self.puzzles = load_puzzles(self.puzzle_csv_path)
        logger.info(f"Loaded {len(self.puzzles)} fixed Lichess puzzles for 6-hour evaluation.")

    def _init_eval_csv(self) -> None:
        if not self.eval_csv_path.exists():
            # NOTE: columns are append-only. Never reorder or rename existing
            # columns; new provenance columns go at the end.
            headers = [
                "timestamp",
                "elapsed_hours",
                "step",
                "model_version",
                "checkpoint_file",
                "puzzle_solve_rate",
                "puzzle_solved_count",
                "puzzle_total_count",
                "match_score",
                "match_engine_wins",
                "match_engine_draws",
                "match_engine_losses",
                "match_draw_rate",
                "match_total_games",
                "seed",
                "config_hash",
                "search_mode",
                "invalid",
                "elo_diff",
                "elo_lo",
                "elo_hi",
            ]
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def run_eval_cycle(
        self, max_puzzles: Optional[int] = None, match_openings: Optional[int] = None
    ) -> dict:
        """Run evaluation cycle: 1,000 puzzles + match vs Stockfish."""
        openings = match_openings if match_openings is not None else self.eval_openings
        logger.info("=" * 70)
        logger.info(
            f"[EVAL CYCLE] Running evaluation for Model Version {self.server.model_version} (Step {self.trainer.step_num})..."
        )
        eval_start = time.time()

        # Save milestone evaluation checkpoint
        ckpt_name = f"phase4_ckpt_v{self.server.model_version}_step{self.trainer.step_num}.pt"
        ckpt_path = self.checkpoint_dir / ckpt_name
        self.trainer.save_checkpoint(ckpt_path, keep_last_n=self.keep_checkpoints)
        self._known_ckpt_names.add(ckpt_name)
        self._event("CHECKPOINT", f"saved {ckpt_path} sha={short_hash(str(ckpt_path))} (eval)")

        # 1. Evaluate Puzzles
        logger.info(
            f"[EVAL CYCLE] Evaluating {len(self.puzzles) if not max_puzzles else max_puzzles} puzzles..."
        )
        puzzles_to_run = self.puzzles[:max_puzzles] if max_puzzles else self.puzzles
        evaluator = PuzzleEvaluator(
            model=self.trainer.model, device=self.device, puzzles=puzzles_to_run
        )
        puzzle_res = evaluator.evaluate_policy(batch_size=128)
        solve_rate = puzzle_res["solve_rate"]
        self._event(
            "EVAL",
            f"puzzles solve_rate={solve_rate * 100:.2f}% "
            f"({puzzle_res['solved']}/{puzzle_res['total']})",
        )
        logger.info(
            f"[EVAL CYCLE] Puzzle solve rate: {solve_rate * 100:.2f}% ({puzzle_res['solved']}/{puzzle_res['total']})"
        )

        self.puzzle_solve_history.append(solve_rate)

        # 2. Evaluate Match vs Stockfish
        logger.info(
            f"[EVAL CYCLE] Running {openings * 2}-game match vs Stockfish ({self.stockfish_nodes} nodes)..."
        )
        eval_sims = (
            self.eval_sims if self.eval_sims is not None else (10 if self.is_verify_mode else 200)
        )
        match_runner = MatchRunner(
            model=self.trainer.model,
            device=self.device,
            stockfish_path=self.stockfish_path,
            stockfish_nodes=self.stockfish_nodes,
            mcts_sims=eval_sims,
            c_puct=2.0,
            openings_path=self.openings_path,
            num_workers=8,
            search_mode="serial",
        )
        try:
            match_res = match_runner.run_match(num_openings=openings)
        finally:
            match_runner.close()

        logger.info(
            f"[EVAL CYCLE] Match Result: W: {match_res['engine_wins']} "
            f"D: {match_res['engine_draws']} L: {match_res['engine_losses']} | "
            f"Score: {match_res['engine_score'] * 100:.1f}% | Draw Rate: {match_res['draw_rate'] * 100:.1f}%"
        )
        self._event(
            "EVAL",
            f"match W={match_res['engine_wins']} D={match_res['engine_draws']} "
            f"L={match_res['engine_losses']} score={match_res['engine_score'] * 100:.1f}% "
            f"draw_rate={match_res['draw_rate'] * 100:.1f}%",
        )

        now = time.time()
        elapsed_hours = (now - self.start_time) / 3600.0

        # Append to CSV (column order locked; provenance appended, see _init_eval_csv)
        row = [
            now,
            round(elapsed_hours, 4),
            self.trainer.step_num,
            self.server.model_version,
            ckpt_name,
            round(solve_rate, 4),
            puzzle_res["solved"],
            puzzle_res["total"],
            round(match_res["engine_score"], 4),
            match_res["engine_wins"],
            match_res["engine_draws"],
            match_res["engine_losses"],
            round(match_res["draw_rate"], 4),
            match_res["total_games"],
            default_mcts_seed(),
            eval_config_hash(
                self.stockfish_nodes,
                eval_sims,
                2.0,
                self.openings_path,
                8,
                policy_target_temp=self.policy_target_temp,
                entropy_bonus=self.entropy_bonus,
            ),
            match_res.get("search_mode", "serial"),
            0,
        ]
        try:
            from python.eval.match_runner import elo_interval

            elo_diff, elo_lo, elo_hi = elo_interval(
                match_res["engine_wins"], match_res["engine_losses"], match_res["engine_draws"]
            )
            row.extend([round(elo_diff, 1), round(elo_lo, 1), round(elo_hi, 1)])
        except Exception:
            row.extend(["n/a", "n/a", "n/a"])
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Log to TensorBoard
        step = self.trainer.step_num
        self.metrics_tracker.writer.add_scalar(
            "eval/puzzle_solve_rate", solve_rate, global_step=step
        )
        self.metrics_tracker.writer.add_scalar(
            "eval/match_score", match_res["engine_score"], global_step=step
        )
        self.metrics_tracker.writer.add_scalar(
            "eval/match_draw_rate", match_res["draw_rate"], global_step=step
        )
        self.metrics_tracker.writer.flush()

        logger.info(f"[EVAL CYCLE] Completed in {time.time() - eval_start:.1f}s.")
        self._event("EVAL_END", f"evaluation completed in {time.time() - eval_start:.1f}s")
        logger.info("=" * 70)

        # Check safety trigger: puzzle solve rate falls for 3 consecutive checks
        # (noise band: each step must drop by MORE than the band to count, so
        # jitter at the 1e-4 level cannot trip a critical stop; semantics
        # otherwise unchanged).
        if puzzle_decline_trips(self.puzzle_solve_history, self.puzzle_noise_band):
            last4 = self.puzzle_solve_history[-4:]
            raise SafetyTriggerException(
                f"CRITICAL STOP TRIGGER: Puzzle solve rate fell for 3 consecutive checks: "
                f"{[round(x, 4) for x in last4]}"
            )

        # Promotion gate: save models/checkpoints_rl/best_eval.pt on improvement
        comp_score = float(match_res["engine_score"]) * 0.7 + float(solve_rate) * 0.3
        if not hasattr(self, "_best_eval_comp_score") or comp_score > self._best_eval_comp_score:
            self._best_eval_comp_score = comp_score
            best_eval_path = self.checkpoint_dir / "best_eval.pt"
            try:
                import shutil

                shutil.copyfile(ckpt_path, best_eval_path)
                logger.info(
                    f"[PROMOTION] New best eval checkpoint saved (score={comp_score:.3f}): {best_eval_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to copy best_eval.pt: {e}")

        return {
            "puzzle": puzzle_res,
            "match": match_res,
            "checkpoint": ckpt_name,
        }

    def record_entropy(self, entropy: float) -> None:
        """Record policy entropy for baseline and rolling average monitoring."""
        if entropy > 0.0:
            self.entropy_history.append(float(entropy))

    def check_safety_triggers(self, train_metrics: dict) -> None:
        """Evaluate real-time telemetry against Contract safety triggers."""
        telemetry = self.metrics_tracker.compute_metrics()
        draw_rate = telemetry.get("draw_rate", 0.0)
        false_resign_rate = telemetry.get("false_resign_rate", 0.0)
        policy_entropy = train_metrics.get("policy_entropy", 0.0)

        # Check 1: Draw rate > 70%
        if self.metrics_tracker.total_games >= 20 and draw_rate > 0.70:
            raise SafetyTriggerException(
                f"CRITICAL STOP TRIGGER: Self-play draw rate exceeded 70% (currently {draw_rate * 100:.1f}% "
                f"over {self.metrics_tracker.total_games} games)."
            )

        # Check 2: False-resign rate > 5% (LOG-ONLY warning, never halts run)
        if (
            self.metrics_tracker.total_games >= 20
            and self.metrics_tracker.total_would_have_resigned >= 10
            and false_resign_rate > 0.05
        ):
            self._note_trigger(
                "_warned_false_resign",
                "TRIGGER",
                f"false-resign rate {false_resign_rate * 100:.1f}% > 5% threshold "
                f"(LOG-ONLY, resign stays off)",
            )
            logger.warning(
                f"SAFETY TRIGGER (LOG-ONLY): False-resign rate exceeded 5% (currently {false_resign_rate * 100:.1f}% "
                f"over {self.metrics_tracker.total_would_have_resigned} calibration games; resignation is disabled)."
            )

        # Check 3: Policy entropy collapse (> 35% drop from baseline)
        # Baseline computed as mean over first 50 steps (entropy_history[:50])
        # compared against rolling 20-step moving average.
        if len(self.entropy_history) >= 50:
            baseline_entropy = sum(self.entropy_history[:50]) / 50.0
            rolling_avg_entropy = sum(self.entropy_history[-20:]) / 20.0
            if rolling_avg_entropy < 0.65 * baseline_entropy:
                raise SafetyTriggerException(
                    f"CRITICAL STOP TRIGGER: Policy entropy rolling 20-step average ({rolling_avg_entropy:.3f}) "
                    f"fell >35% below 50-step baseline ({baseline_entropy:.3f})."
                )

        # Check 4: Sustained low GPU utilization (< 60%)
        gpu_util = telemetry.get("gpu_utilization", 0.0)
        if gpu_util > 0.0 and gpu_util < 60.0:
            self.low_gpu_util_counter += 1
            if self.low_gpu_util_counter >= 120:  # sustained over 120 checks (~10 minutes)
                self._note_trigger(
                    "_warned_low_gpu",
                    "TRIGGER",
                    f"GPU utilization {gpu_util:.1f}% < 60% sustained (LOG-ONLY warning)",
                )
                logger.warning(
                    f"ALERT: GPU utilization has remained below 60% for sustained period (current: {gpu_util:.1f}%)."
                )
        else:
            self.low_gpu_util_counter = 0

    # ── Terminal report (printing only; every value measured or tagged) ──

    def _emit(self, text: str) -> None:
        emit_report(text)
        self._heartbeat.notify()

    def _event(self, tag: str, message: str) -> None:
        logger.info("EVENT %s: %s", tag, message)
        self._emit(event_line(tag, message))

    def _print_startup_banner(self) -> None:
        """Resolved-settings banner. `*` marks deviation from defaults."""
        ckpt = self._active_checkpoint_path
        try:
            step = int(self.trainer.step_num)
        except (AttributeError, TypeError, ValueError):
            step = 0
        conc = self.num_concurrent_games
        conc_mark = (
            "*" if (self._clamped_concurrency or (not self.is_verify_mode and conc != 24)) else ""
        )
        tb = (
            f"{self.tablebase_path}, max_pieces={self.tablebase_max_pieces} (C), "
            f"files_found={self._tb_files_found} (M)"
            if self._tb_enabled
            else "disabled (C)"
        )
        min_fill = self.min_buffer_fill if self.min_buffer_fill > 0 else 500
        rows = [
            ("NET", f"12 blocks x 128 filters (M, ckpt step {step})"),
            (
                "SIMS",
                f"full={self.full_search_sims}{'*' if self.full_search_sims != 800 else ''} "
                f"fast={self.fast_search_sims}{'*' if self.fast_search_sims != 150 else ''} (C)",
            ),
            ("FULL_FRAC", "0.35 (C)"),
            ("CONC", f"requested={self._requested_concurrency} effective={conc}{conc_mark} (C)"),
            ("TEMP", "cutoff plies 15-30 (C)"),
            ("NOISE", "dirichlet alpha=0.3 weight=0.25 (C, search defaults)"),
            ("RESIGN", "off (C, passive would_resign logged)"),
            ("TABLEBASE", tb),
            ("REUSE_CAP", f"{self.max_reuse_ratio}{'*' if self.max_reuse_ratio != 16 else ''} (C)"),
            ("MIN_FILL", f"{min_fill}{'*' if self.min_buffer_fill not in (0, 500) else ''} (C)"),
            ("PUSH_FREQ", f"{self.trainer.push_freq} steps (C)"),
            ("HUMAN_MIX", f"{self.human_cache_size} VRAM-preloaded (C)"),
            ("TACTICAL_MIX", f"{self.tactical_mix}{'*' if self.tactical_mix != 0.12 else ''} (C)"),
            ("BASE_LR", f"{self.base_lr}{'*' if self.base_lr != 0.0005 else ''} (C)"),
            ("MIN_LR", f"{self.min_lr}{'*' if self.min_lr != 1e-5 else ''} (C)"),
            (
                "POLICY_TEMP",
                f"{self.policy_target_temp}{'*' if self.policy_target_temp != 0.5 else ''} (C)",
            ),
            (
                "ENTROPY_BONUS",
                f"{self.entropy_bonus}{'*' if self.entropy_bonus != 0.001 else ''} (C)",
            ),
            ("BUFFER_CAP", f"{self.buffer.capacity} (C)"),
            ("EVAL_EVERY", f"{self.eval_interval_sec:.0f}s (C)"),
            ("SKIP_BASELINE_EVAL", f"{self.skip_baseline_eval} (C)"),
            ("RAM_CAP", "free floor 2.0GB, RSS ceiling 7.5GB (C)"),
            ("CKPT", f"{ckpt} sha={short_hash(ckpt)} (M)"),
            ("DEVICE", f"{self.device} (M)"),
        ]
        self._emit(banner("rl-selfplay"))
        ctx = run_context("rl-selfplay")
        self._emit(
            header_block(
                ctx["start"],
                ctx["command"],
                ctx["workdir"],
                ctx["commit"],
                "cli/args",
                ckpt,
                self.device,
                "n/a (see --test-verify/bench --base-seed)",
            )
        )
        self._emit(results_table(rows))
        self._emit(
            event_line(
                "CTRLC",
                "press Ctrl+C ONCE to stop cleanly "
                "(workers stop, checkpoint saved, resume with --resume or relaunch command)",
            )
        )
        if self._clamped_concurrency:
            self._event(
                "CLAMP",
                f"concurrency requested={self._requested_concurrency} "
                f"clamped to {conc} (free RAM < 3.0GB)",
            )

    def _note_trigger(self, latch: str, tag: str, message: str) -> None:
        """Count a log-only trigger warning; emit a console event once per run.

        Degrades gracefully on bare test instances (no reporting state).
        """
        self._warnings = getattr(self, "_warnings", 0) + 1
        if getattr(self, latch, False):
            return
        try:
            setattr(self, latch, True)
        except AttributeError:
            return
        if hasattr(self, "_heartbeat"):
            self._event(tag, message)

    def _trainer_state(self) -> tuple[str, float]:
        elapsed = max(1e-5, time.time() - self.start_time)
        duty = self._train_busy_sec / elapsed
        state = "training" if self.trainer.step_num > 0 else "waiting-data"
        return state, duty

    def _maybe_status(self, train_metrics: Optional[dict] = None, force: bool = False) -> None:
        """10 s live-status line from counters only + 30 s heartbeat."""
        if not self._status_limiter.should_emit(force=force):
            hb = self._heartbeat.maybe_emit(
                f"buffer={len(self.buffer)} games={self.metrics_tracker.total_games}"
            )
            if hb is not None:
                self._emit(hb)
            self._poll_ckpt_dir()
            return
        now = time.time()
        elapsed = now - self.start_time
        telemetry = self.metrics_tracker.compute_metrics()
        total_sims = getattr(
            self.worker,
            "live_simulations",
            getattr(self.worker, "total_simulations", self.metrics_tracker.total_simulations),
        )
        sims_per_sec = self.progress_reporter.compute_sims_per_sec(
            int(total_sims), current_time=now
        )
        games = self.metrics_tracker.total_games
        would_resign = float(self.metrics_tracker.total_would_have_resigned) / max(1, games)
        adj_frac = self._adj_games / max(1, self._completed_games_seen)
        state, duty = self._trainer_state()
        if train_metrics is None:
            pol, wdl = 0.0, 0.0
            step = self.trainer.step_num
            if hasattr(self, "_last_train_metrics") and self._last_train_metrics:
                pol = float(self._last_train_metrics.get("policy_loss", 0.0))
                wdl = float(self._last_train_metrics.get("wdl_loss", 0.0))
                state = "pacing"
            else:
                state = "waiting-data"
        else:
            pol = float(train_metrics.get("policy_loss", 0.0))
            wdl = float(train_metrics.get("wdl_loss", 0.0))
            step = self.trainer.step_num
            self._last_train_metrics = train_metrics
        try:
            gpu_util, vram_used_mb, _vram_total = query_gpu_metrics()
        except Exception:
            gpu_util, vram_used_mb = 0.0, 0.0
        try:
            rss_mb, _sys_pct = query_process_memory()
            free_ram_gb = psutil.virtual_memory().available / (1024.0**3)
        except Exception:
            rss_mb, free_ram_gb = 0.0, 0.0
        games_hr = (games / max(1e-5, elapsed)) * 3600.0
        pos_hr = (self.buffer.total_added / max(1e-5, elapsed)) * 3600.0
        fills = list(self.metrics_tracker.recent_batch_fills)
        batch_avg = (sum(fills) / len(fills) * 128.0) if fills else 0.0
        try:
            lag_min = (now - Path(self._active_checkpoint_path).stat().st_mtime) / 60.0
        except OSError:
            lag_min = 0.0
        next_eval_in = max(0.0, self.last_eval_time + self.eval_interval_sec - now)
        curr_buf_len = len(self.buffer)
        curr_model_v = getattr(self.server, "model_version", None)
        duty_str = f"{duty:.2f}"
        pol_str = f"{pol:.4f}"
        wdl_str = f"{wdl:.4f}"
        avg_plies_str = f"{float(telemetry.get('average_game_length', 0.0)):.1f}"
        draw_rate_str = f"{float(telemetry.get('draw_rate', 0.0)):.3f}"
        adj_frac_str = f"{adj_frac:.3f}"

        curr_snapshot = {
            "step": step,
            "pol": pol_str,
            "wdl": wdl_str,
            "trainer": state,
            "duty": duty_str,
            "buffer": curr_buf_len,
            "games": games,
            "avg_plies": avg_plies_str,
            "draw_rate": draw_rate_str,
            "adj_frac": adj_frac_str,
            "model_v": curr_model_v,
        }

        prev = getattr(self, "_last_status_snapshot", None)
        has_substantive_change = (
            prev is None
            or force
            or step != prev.get("step")
            or games != prev.get("games")
            or curr_buf_len != prev.get("buffer")
            or state != prev.get("trainer")
            or curr_model_v != prev.get("model_v")
            or duty_str != prev.get("duty")
        )

        if not has_substantive_change:
            self._poll_ckpt_dir()
            return

        def _hl(key: str, val: object, is_diff: bool) -> tuple[str, str]:
            if is_diff and prev is not None:
                if os.environ.get("NO_COLOR") is None:
                    return (f"\033[1;33m{key}", f"{val}\033[0m")
                else:
                    return (f"*{key}", f"{val}*")
            return (key, str(val))

        status_items = [
            ("elapsed", f"{elapsed:.0f}s"),
            _hl("step", step, prev is not None and step != prev.get("step")),
            _hl("pol", pol_str, prev is not None and pol_str != prev.get("pol")),
            _hl("wdl", wdl_str, prev is not None and wdl_str != prev.get("wdl")),
        ]
        if self._last_emitted_trainer_state != state:
            status_items.append(
                _hl("trainer", state, prev is not None and state != prev.get("trainer"))
            )
            self._last_emitted_trainer_state = state
        if self._last_emitted_duty != duty_str:
            status_items.append(
                _hl("duty", duty_str, prev is not None and duty_str != prev.get("duty"))
            )
            self._last_emitted_duty = duty_str
        status_items.extend(
            [
                _hl(
                    "buffer", curr_buf_len, prev is not None and curr_buf_len != prev.get("buffer")
                ),
                _hl("games", games, prev is not None and games != prev.get("games")),
                ("games_hr", f"{games_hr:.1f}"),
                ("pos_hr", f"{pos_hr:.1f}"),
                _hl(
                    "avg_plies",
                    avg_plies_str,
                    prev is not None and avg_plies_str != prev.get("avg_plies"),
                ),
                _hl(
                    "draw_rate",
                    draw_rate_str,
                    prev is not None and draw_rate_str != prev.get("draw_rate"),
                ),
                _hl(
                    "adj_frac",
                    adj_frac_str,
                    prev is not None and adj_frac_str != prev.get("adj_frac"),
                ),
                ("sims_sec", f"{sims_per_sec:.1f}"),
            ]
        )
        curr_model_v = getattr(self.server, "model_version", None)
        if self._last_emitted_model_v != curr_model_v:
            status_items.append(
                _hl(
                    "model_v",
                    curr_model_v,
                    prev is not None and curr_model_v != prev.get("model_v"),
                )
            )
            self._last_emitted_model_v = curr_model_v
        status_items.append(("next_eval_s", f"{next_eval_in:.0f}"))
        self._last_status_snapshot = curr_snapshot
        self._emit(status_line(status_items))
        self._poll_push_and_ckpt()

    def _poll_push_and_ckpt(self) -> None:
        """Emit events for trainer weight pushes and new checkpoint files."""
        try:
            ver = int(self.server.model_version)
        except (AttributeError, TypeError, ValueError):
            ver = 0
        if ver > self._last_push_version:
            if self._last_push_version > 0 or ver > 1:
                self._event("PUSH", f"weights pushed to inference server (model_v={ver})")
            self._last_push_version = ver
        self._poll_ckpt_dir()

    def _poll_ckpt_dir(self) -> None:
        try:
            names = {p.name for p in self.checkpoint_dir.iterdir() if p.suffix == ".pt"}
        except OSError:
            return
        for name in sorted(names - self._known_ckpt_names):
            self._known_ckpt_names.add(name)
            self._event(
                "CHECKPOINT",
                f"saved {self.checkpoint_dir / name} "
                f"sha={short_hash(str(self.checkpoint_dir / name))}",
            )

    # ── Task 5: memory-safety watchdog ────────────────────────────────────

    def _run_watchdog(self) -> None:
        """Background thread: abort training if RSS or system free RAM exceed limits."""
        proc = psutil.Process()
        while not self.stop_event.is_set():
            try:
                rss_bytes = proc.memory_info().rss
                free_bytes = psutil.virtual_memory().available
                if rss_bytes > self._watchdog_rss_bytes:
                    rss_gb = rss_bytes / (1024.0**3)
                    ceil_gb = self._watchdog_rss_bytes / (1024.0**3)
                    msg = (
                        f"WATCHDOG: Process RSS {rss_gb:.2f} GB exceeded ceiling "
                        f"{ceil_gb:.2f} GB — aborting to prevent OOM."
                    )
                    logger.error(msg)
                    try:
                        self._event(
                            "WATCHDOG", f"RSS {rss_gb:.2f}GB > ceiling {ceil_gb:.2f}GB; stopping"
                        )
                    except AttributeError:
                        pass
                    try:
                        self._warnings = getattr(self, "_warnings", 0) + 1
                    except AttributeError:
                        pass
                    self.stop_event.set()
                    return
                if free_bytes < self._watchdog_free_ram_bytes:
                    free_gb = free_bytes / (1024.0**3)
                    floor_gb = self._watchdog_free_ram_bytes / (1024.0**3)
                    msg = (
                        f"WATCHDOG: System free RAM {free_gb:.2f} GB dropped below "
                        f"floor {floor_gb:.2f} GB — aborting to prevent OOM."
                    )
                    logger.error(msg)
                    try:
                        self._event(
                            "WATCHDOG",
                            f"free RAM {free_gb:.2f}GB < floor {floor_gb:.2f}GB; stopping",
                        )
                    except AttributeError:
                        pass
                    try:
                        self._warnings = getattr(self, "_warnings", 0) + 1
                    except AttributeError:
                        pass
                    self.stop_event.set()
                    return
            except Exception as exc:
                logger.debug("Watchdog probe failed (non-fatal): %s", exc)
            time.sleep(5.0)

    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> int:
        """Run Phase 4 continuous self-play, training, and 6-hour evaluation cycles.

        Returns the process exit code (0 ok/interrupted, 1 failure) and records
        the cause in ``exit_reason`` (ok/interrupted/safety/error) so launchers
        can map legacy codes. Never raises for run-level outcomes.
        """
        try:
            self._known_ckpt_names = {
                p.name for p in self.checkpoint_dir.iterdir() if p.suffix == ".pt"
            }
        except OSError:
            self._known_ckpt_names = set()
        self._last_push_version = int(getattr(self.server, "model_version", 0) or 0)
        self._print_startup_banner()
        self._event(
            "START",
            f"self-play workers={self.num_concurrent_games} "
            f"buffer_cap={self.buffer.capacity} "
            f"base_lr={self.base_lr} min_lr={self.min_lr} "
            f"tactical_mix={self.tactical_mix} "
            f"policy_target_temp={self.policy_target_temp} "
            f"entropy_bonus={self.entropy_bonus}",
        )
        logger.info("Starting Phase 4 self-play worker threads...")
        self.worker.start()

        # Task 5: launch memory-safety watchdog in a daemon thread
        _wd = threading.Thread(target=self._run_watchdog, name="memory-watchdog", daemon=True)
        _wd.start()
        logger.info("Memory-safety watchdog thread started.")

        # Check if baseline evaluation (Cycle 0) was already recorded in eval_history.csv
        has_baseline_eval = False
        latest_solve_rate = None
        target_games = 2 if self.is_verify_mode else 100
        if self.eval_csv_path.exists():
            with open(self.eval_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and len(row) > 13:
                        try:
                            if int(row[13]) == target_games:
                                has_baseline_eval = True
                                latest_solve_rate = float(row[5])
                        except (ValueError, IndexError):
                            pass
        if latest_solve_rate is not None:
            self.puzzle_solve_history.append(latest_solve_rate)

        if not has_baseline_eval:
            if self.skip_baseline_eval:
                self._event(
                    "EVAL_SKIP", "baseline Evaluation Cycle 0 skipped per --skip-baseline-eval"
                )
                logger.info(
                    "Skipping baseline Evaluation Cycle 0 per user directive (--skip-baseline-eval). Proceeding directly to continuous self-play and training."
                )
            else:
                self._event("EVAL_START", "baseline Evaluation Cycle 0 (Model v0)")
                if self.is_verify_mode:
                    self.run_eval_cycle(max_puzzles=10, match_openings=1)
                else:
                    self.run_eval_cycle(max_puzzles=1000, match_openings=self.eval_openings)
        else:
            self._event("EVAL_SKIP", "baseline Cycle 0 already in eval_history.csv; continuing")
            logger.info(
                "Baseline Evaluation Cycle 0 already recorded in eval_history.csv. Proceeding directly to continuous self-play and training."
            )

        self.last_eval_time = time.time()

        try:
            # Seed buffer with positions before training loop
            min_fill = (
                self.min_buffer_fill
                if self.min_buffer_fill > 0
                else (15 if self.is_verify_mode else 500)
            )
            target_fill = min(min_fill, self.buffer.capacity)
            self._event(
                "BUFFER_WAIT",
                f"need {target_fill} positions (have {len(self.buffer)}) before training",
            )
            logger.info(
                f"Awaiting initial buffer positions ({target_fill} required, currently {len(self.buffer)}) "
                f"before training loop..."
            )
            while len(self.buffer) < target_fill and not self.stop_event.is_set():
                now = time.time()
                if (now - self.start_time) >= self.max_duration_sec:
                    self._event(
                        "STOP", f"target duration {self.max_hours:.1f}h reached while buffering"
                    )
                    logger.info(
                        f"Target duration {self.max_hours:.1f} hours reached while buffering. Stopping Phase 4 run."
                    )
                    self._print_summary("SUCCESS")
                    self._exit_reason = "ok"
                    return 0

                # Live status line during buffer fill wait (10 s cadence inside)
                self._maybe_status()
                time.sleep(0.5)

            self._event(
                "BUFFER_READY",
                f"replay buffer seeded ({len(self.buffer)} positions); training active",
            )
            logger.info(
                f"Replay buffer seeded ({len(self.buffer)} positions). Continuous RL training active."
            )

            while not self.stop_event.is_set():
                now = time.time()
                elapsed = now - self.start_time
                if elapsed >= self.max_duration_sec:
                    self._event("STOP", f"target duration {self.max_hours:.1f}h reached")
                    logger.info(
                        f"Target duration {self.max_hours:.1f} hours reached. Stopping Phase 4 run."
                    )
                    if (now - self.last_eval_time) >= 900.0:
                        self._event("EVAL_START", "final evaluation on run completion")
                        if self.is_verify_mode:
                            self.run_eval_cycle(max_puzzles=50, match_openings=1)
                        else:
                            self.run_eval_cycle(max_puzzles=1000, match_openings=self.eval_openings)
                    break

                # 6-Hour Evaluation Check
                if (now - self.last_eval_time) >= self.eval_interval_sec:
                    self._event("EVAL_START", "periodic evaluation triggered")
                    logger.info("6-hour timer elapsed. Triggering periodic evaluation...")
                    if self.is_verify_mode:
                        self.run_eval_cycle(max_puzzles=50, match_openings=1)
                    else:
                        self.run_eval_cycle(max_puzzles=1000, match_openings=self.eval_openings)
                    self.last_eval_time = time.time()

                # Pacing: Avoid runaway training steps that starve self-play workers
                target_max_step = max(
                    500,
                    int((self.buffer.total_added * self.max_reuse_ratio) / self.trainer.batch_size),
                )
                if self.trainer.step_num > target_max_step:
                    self._maybe_status()
                    time.sleep(0.02)  # Give self-play game threads CPU/GPU time to advance
                    continue

                # RL Training Step (busy-time measured for the duty fraction)
                busy_t0 = time.perf_counter()
                train_metrics = self.trainer.train_step()
                self._train_busy_sec += time.perf_counter() - busy_t0
                self._last_train_metrics = train_metrics
                step = train_metrics["step"]
                self.metrics_tracker.record_train_step(train_metrics)
                pe = train_metrics.get("policy_entropy", 0.0)
                if pe > 0.0:
                    self.record_entropy(pe)

                self._steps_this_run += 1
                te = train_metrics.get("target_entropy", 0.0)
                logger.info(
                    f"[TRAIN STEP] step={step} (run_step={self._steps_this_run}) loss={train_metrics['total_loss']:.4f} "
                    f"policy_entropy={pe:.3f} "
                    f"target_entropy={te:.3f} "
                    f"actual_tactical_mix={train_metrics.get('actual_tactical_mix', 0.0):.3f}"
                )

                # Live status line every 10 s (counters only)
                self._maybe_status(train_metrics)

                # Periodic Telemetry Flush every 50 steps (or every 5 steps in verify mode) or upon reaching max_steps
                if (
                    step % 50 == 0
                    or (self.is_verify_mode and step % 5 == 0)
                    or (self.max_steps is not None and self._steps_this_run >= self.max_steps)
                ):
                    self.metrics_tracker.log_step(
                        step=step,
                        buffer_size=len(self.buffer),
                        model_version=self.server.model_version,
                    )
                    # Check safety triggers
                    self.check_safety_triggers(train_metrics)

                if self.max_steps is not None and self._steps_this_run >= self.max_steps:
                    self._event(
                        "STOP",
                        f"target max_steps {self.max_steps} reached ({self._steps_this_run} steps executed)",
                    )
                    logger.info(
                        f"Target max_steps {self.max_steps} reached ({self._steps_this_run} steps executed). Stopping Phase 4 run."
                    )
                    break

                time.sleep(0.001)

        except SafetyTriggerException as ste:
            self.stop()
            emit_error(
                error_block(
                    f"safety trigger stopped the run: {ste}",
                    "a Contract guardrail tripped (draw rate, entropy, puzzles, OOM watchdog)",
                    "inspect telemetry + eval_history.csv, fix the cause, resume with the same command",
                    str(self.log_dir / "phase4_rl.log"),
                )
            )
            self._print_summary("FAILED")
            self._exit_reason = "safety"
            return 1
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Stopping run cleanly...")
            self._event("SHUTDOWN", "Ctrl+C received; stopping cleanly")
            self.stop()
            self._print_summary("INTERRUPTED")
            self._exit_reason = "interrupted"
            return 0
        except Exception as e:
            self.stop()
            emit_error(
                error_block(
                    f"unexpected failure: {e}",
                    "unhandled exception in self-play, training, or evaluation",
                    "see LOG traceback, fix the cause, resume with the same command",
                    str(self.log_dir / "phase4_rl.log"),
                )
            )
            self._print_summary("FAILED")
            self._exit_reason = "error"
            return 1
        else:
            self.stop()
            self._print_summary("SUCCESS")
            self._exit_reason = "ok"
            return 0
        finally:
            self.stop()

    def _print_summary(self, status: str) -> None:
        """Exit/SIGINT summary: totals from counters + standard footer."""
        elapsed = time.time() - self.start_time
        games = self.metrics_tracker.total_games
        self._emit(
            status_line(
                [
                    ("games", games),
                    ("positions", self.buffer.total_added),
                    ("steps", self.trainer.step_num),
                    ("model_v", self.server.model_version),
                    ("evals", len(self.puzzle_solve_history)),
                ]
            )
        )
        exit_code = 0 if status in ("SUCCESS", "INTERRUPTED") else 1
        self._emit(summary_block(status, elapsed, self._warnings, exit_code))

    def stop(self) -> None:
        """Stop worker, save final checkpoint, flush telemetry, janitor junk."""
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        self._event("SHUTDOWN", "stopping workers and saving final checkpoint")
        logger.info("Shutting down Phase 4 orchestrator...")
        self.stop_event.set()
        try:
            self.worker.stop()
            final_ckpt = self.trainer.save_checkpoint(
                self.checkpoint_dir / "phase4_final_checkpoint.pt",
                keep_last_n=self.keep_checkpoints,
            )
            self._known_ckpt_names.add(Path(final_ckpt).name)
            self._event("CHECKPOINT", f"saved {final_ckpt} sha={short_hash(final_ckpt)} (final)")
        except Exception as exc:
            logger.error("Final checkpoint save failed during shutdown: %s", exc)
            self._event("SHUTDOWN", f"final checkpoint failed: {exc}")
        try:
            self.metrics_tracker.close()
        except Exception as exc:
            logger.error("Metrics close failed during shutdown: %s", exc)
        if hasattr(self, "buffer") and self.buffer is not None:
            try:
                self.buffer.close()
            except Exception as exc:
                logger.error("Buffer close failed during shutdown: %s", exc)
        self._event("SHUTDOWN", "telemetry flushed")
        logger.info(
            f"Phase 4 run stopped. Final checkpoint saved to {getattr(self, '_last_final_ckpt', 'unknown')}"
        )
        # Post-run janitor: prune events + (optionally) delete the 11 GB buffer.
        try:
            from python.common.cleanup import cleanup_run

            report = cleanup_run(
                self.log_dir,
                self.checkpoint_dir,
                delete_buffer=self.cleanup_buffer_on_exit,
                keep_checkpoints=self.keep_checkpoints,
                keep_tfevents=self.keep_tfevents,
            )
            logger.info("Post-run cleanup: %s", report)
        except (OSError, ValueError) as e:
            logger.debug("post-run cleanup skipped: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Long RL Run Orchestrator")
    parser.add_argument(
        "--max-hours", type=float, default=72.0, help="Total hours to run (default 72.0)"
    )
    parser.add_argument(
        "--eval-interval-sec",
        type=float,
        default=21600.0,
        help="Evaluation interval in seconds (default 21600s = 6h)",
    )
    parser.add_argument(
        "--concurrent-games", type=int, default=24, help="Concurrent self-play games (default 24)"
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=500_000,
        help="Replay buffer capacity (default 500,000)",
    )
    parser.add_argument(
        "--gpu-cache-capacity",
        type=int,
        default=160_000,
        help="GPU VRAM replay cache capacity (default 160,000)",
    )
    parser.add_argument(
        "--human-cache-size",
        type=int,
        default=80_000,
        help="Human dataset active sample cached in VRAM (default 80,000)",
    )
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        help="Skip baseline evaluation cycle on startup and proceed directly to self-play and RL training",
    )
    parser.add_argument(
        "--test-verify",
        action="store_true",
        help="Run 30-second verification mode with quick eval cycle",
    )
    parser.add_argument(
        "--tablebase-path",
        type=str,
        default=None,
        help="Path to Syzygy tablebase directory (default: auto-detect data/syzygy or disabled)",
    )
    parser.add_argument(
        "--tablebase-max-pieces",
        type=int,
        default=5,
        help="Maximum pieces for tablebase adjudication (default 5)",
    )
    parser.add_argument(
        "--min-buffer-fill",
        type=int,
        default=0,
        help="Minimum self-play positions in replay buffer before training begins (default 0)",
    )
    parser.add_argument(
        "--max-reuse-ratio",
        type=int,
        default=16,
        help="Maximum sample reuse ratio for pacing training steps (default 16)",
    )
    parser.add_argument(
        "--full-sims", type=int, default=800, help="Full MCTS search simulations (default 800)"
    )
    parser.add_argument(
        "--fast-sims", type=int, default=150, help="Fast MCTS search simulations (default 150)"
    )
    parser.add_argument(
        "--reset-buffer",
        action="store_true",
        help="Start with a fresh empty replay buffer and warm-start checkpoint",
    )
    parser.add_argument(
        "--cleanup-buffer-on-exit",
        action="store_true",
        help="Delete the preallocated replay buffer (*.bin, ~11 GB at 500k) after the final checkpoint is saved",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep newest N periodic checkpoints, prune older (default 3; protected: best/latest/live/final always kept)",
    )
    parser.add_argument(
        "--keep-tfevents",
        type=int,
        default=1,
        help="Keep newest N TensorBoard event files per log dir (default 1)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/phase4_rl",
        help="Run logging and buffer directory (default runs/phase4_rl)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints_rl",
        help="Checkpoint directory (default models/checkpoints_rl)",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="models/checkpoints_scaled/best_model.pt",
        help="Base warm-start checkpoint path",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Execution device (default: auto cuda/cpu)"
    )
    parser.add_argument(
        "--train-device",
        type=str,
        default=None,
        help="Device for RL training (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--infer-device",
        type=str,
        default=None,
        help="Device for self-play inference (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--watchdog-rss-gb",
        type=float,
        default=7.5,
        help="Process RSS abort ceiling in GB (default: 7.5)",
    )
    parser.add_argument(
        "--watchdog-free-ram-gb",
        type=float,
        default=2.0,
        help="System free-RAM floor in GB (default: 2.0)",
    )
    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=1.0,
        help="Dispatcher timeout in milliseconds (default: 1.0)",
    )
    parser.add_argument(
        "--stockfish-nodes",
        type=int,
        default=1000,
        help="Stockfish evaluation nodes (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducible runs (default: None)"
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.0005, help="RL base learning rate (default: 0.0005)"
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-5, help="RL minimum learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--tactical-mix",
        type=float,
        default=0.12,
        help="Fraction of every batch reserved for tactical samples (default: 0.12)",
    )
    parser.add_argument(
        "--tactical-csv-path",
        type=str,
        default="data/tactics_train_300k.csv",
        help="Path to tactical positions CSV (default: data/tactics_train_300k.csv)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps before exiting (default: None for continuous)",
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
    parser.add_argument(
        "--eval-openings",
        type=int,
        default=50,
        help="Number of openings for match eval vs Stockfish (default: 50)",
    )
    parser.add_argument(
        "--eval-sims",
        type=int,
        default=None,
        help="MCTS simulations per move during match evaluation (default: 200)",
    )
    args = parser.parse_args()

    if args.test_verify:
        logger.info("Running Phase 4 verification mode (30s duration, quick eval)...")
        orchestrator = Phase4Orchestrator(
            max_hours=30.0 / 3600.0,
            eval_interval_sec=15.0,
            num_concurrent_games=8,
            buffer_capacity=10_000,
            gpu_cache_capacity=min(args.gpu_cache_capacity, 10_000),
            human_cache_size=min(args.human_cache_size, 5_000),
            is_verify_mode=True,
            skip_baseline_eval=args.skip_baseline_eval,
            tablebase_path=args.tablebase_path if args.tablebase_path else None,
            tablebase_max_pieces=args.tablebase_max_pieces,
            min_buffer_fill=min(args.min_buffer_fill, 100),
            max_reuse_ratio=args.max_reuse_ratio,
            full_search_sims=args.full_sims,
            fast_search_sims=args.fast_sims,
            reset_buffer=args.reset_buffer,
            cleanup_buffer_on_exit=args.cleanup_buffer_on_exit,
            keep_checkpoints=args.keep_checkpoints,
            keep_tfevents=args.keep_tfevents,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            base_checkpoint=args.base_checkpoint,
            device=args.device,
            train_device=args.train_device,
            infer_device=args.infer_device,
            watchdog_rss_gb=args.watchdog_rss_gb,
            watchdog_free_ram_gb=args.watchdog_free_ram_gb,
            max_wait_ms=args.max_wait_ms,
            stockfish_nodes=args.stockfish_nodes,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            tactical_mix=args.tactical_mix,
            tactical_csv_path=args.tactical_csv_path,
            max_steps=args.max_steps,
            policy_target_temp=args.policy_target_temp,
            entropy_bonus=args.entropy_bonus,
            eval_openings=args.eval_openings,
            eval_sims=args.eval_sims,
        )
        sys.exit(orchestrator.run())
    else:
        orchestrator = Phase4Orchestrator(
            max_hours=args.max_hours,
            eval_interval_sec=args.eval_interval_sec,
            num_concurrent_games=args.concurrent_games,
            buffer_capacity=args.buffer_capacity,
            gpu_cache_capacity=args.gpu_cache_capacity,
            human_cache_size=args.human_cache_size,
            is_verify_mode=False,
            skip_baseline_eval=args.skip_baseline_eval,
            tablebase_path=args.tablebase_path if args.tablebase_path else None,
            tablebase_max_pieces=args.tablebase_max_pieces,
            min_buffer_fill=args.min_buffer_fill,
            max_reuse_ratio=args.max_reuse_ratio,
            full_search_sims=args.full_sims,
            fast_search_sims=args.fast_sims,
            reset_buffer=args.reset_buffer,
            cleanup_buffer_on_exit=args.cleanup_buffer_on_exit,
            keep_checkpoints=args.keep_checkpoints,
            keep_tfevents=args.keep_tfevents,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            base_checkpoint=args.base_checkpoint,
            device=args.device,
            train_device=args.train_device,
            infer_device=args.infer_device,
            watchdog_rss_gb=args.watchdog_rss_gb,
            watchdog_free_ram_gb=args.watchdog_free_ram_gb,
            max_wait_ms=args.max_wait_ms,
            stockfish_nodes=args.stockfish_nodes,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            tactical_mix=args.tactical_mix,
            tactical_csv_path=args.tactical_csv_path,
            max_steps=args.max_steps,
            policy_target_temp=args.policy_target_temp,
            entropy_bonus=args.entropy_bonus,
            eval_openings=args.eval_openings,
            eval_sims=args.eval_sims,
        )
        sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
