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
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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
from python.eval.match_runner import MatchRunner, eval_config_hash
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


def eval_elo_columns(engine_wins: int, engine_draws: int, engine_losses: int) -> list:
    """Elo CSV columns ``[elo_diff, elo_lo, elo_hi]`` from W/D/L.

    P0-5 (Stage A §6): the signature order is (wins, draws, losses) and
    :func:`elo_interval` returns a dict. The historical call site passed
    ``(wins, losses, draws)`` — inverting the sign of every recorded Elo —
    AND unpacked the dict as a 3-tuple, which raised and made the surrounding
    ``except`` write "n/a" into every row. A W5/D10/L85 result must yield the
    correct ≈ −382 Elo, not the sign-flipped ≈ −17, and never "n/a".
    """
    from python.eval.match_runner import elo_interval

    interval = elo_interval(engine_wins, engine_draws, engine_losses)
    diff = interval["elo_diff"]
    if diff is None:
        # Zero completed games, or an exactly 0/1 score: unbounded interval.
        return ["n/a", "n/a", "n/a"]
    lo = round(interval["lo"], 1) if interval["lo"] is not None else "n/a"
    hi = round(interval["hi"], 1) if interval["hi"] is not None else "n/a"
    return [round(diff, 1), lo, hi]


class SafetyTriggerException(Exception):
    """Raised when an automated safety guardrail is breached."""

    pass


def pacing_target_step(total_added: int, batch_size: int, max_reuse_ratio: int) -> int:
    """Maximum RL training steps the buffer's production supports (pacing gate).

    Each full-search position may be reused at most ``max_reuse_ratio`` times,
    giving ``total_added * max_reuse_ratio / batch_size`` supported steps, with
    a floor of 500 so a fresh run can take its first steps before much data
    exists. Compared against run-local RL steps only (never an inherited
    supervised counter).
    """
    return max(500, int((total_added * max_reuse_ratio) / batch_size))


def puzzle_decline_trips(history: list[float], band: float = 0.015) -> bool:
    """True if the last 3 steps each decline by MORE than ``band``.

    Same trigger semantics as before, plus a noise band so flat/jittered
    histories (drops within measurement noise) cannot trip a critical stop.
    Needs >= 4 samples; shorter histories never trip.

    Band calibration: the certified 1,000-puzzle set has binomial sampling
    noise sigma = sqrt(p(1-p)/n) ~= 0.015 at p ~= 0.64, so per-step drops must
    exceed one standard deviation (~1.5 percentage points) to count as real
    decline rather than sampling jitter (audit finding F-14).
    """
    if len(history) < 4:
        return False
    last4 = history[-4:]
    return bool(
        (last4[0] - last4[1] > band)
        and (last4[1] - last4[2] > band)
        and (last4[2] - last4[3] > band)
    )


def load_quarantine_index(checkpoint_dir: Path) -> Dict[str, Any]:
    """Collect quarantine evidence for this checkpoint directory (SR-03).

    Sources: ``QUARANTINE.json`` in the checkpoint directory and in its
    ``quarantined/`` subdirectory (the layout written by the 2026-10-27
    strength remediation), plus ``docs/checkpoint_registry.json`` entries
    whose status is ``quarantined``/``missing``/``disqualified``.

    Returns ``{"names_by_dir": {dir: set(names)}, "paths": set(resolved),
    "sha256": set(hex or 16-prefix), "sizes": set(bytes)}``. Filename matches
    are scoped to the directory whose own ``QUARANTINE.json`` names them —
    matching bare basenames globally would falsely flag a healthy
    ``latest_checkpoint.pt`` merely because some other directory's manifest
    (or the registry) knows a quarantined file with the same basename.
    """
    index: Dict[str, Any] = {
        "names_by_dir": {},
        "paths": set(),
        "sha256": set(),
        "sizes": set(),
    }

    def _ingest_manifest(manifest_path: Path) -> None:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        names_here = index["names_by_dir"].setdefault(
            os.path.normcase(str(manifest_path.parent.resolve())), set()
        )
        entries = data.get("files", data if isinstance(data, list) else [])
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            for key in ("file", "filename"):
                if entry.get(key):
                    names_here.add(Path(str(entry[key])).name)
            for key in ("moved_to", "path"):
                if entry.get(key):
                    try:
                        resolved = Path(str(entry[key]))
                        if not resolved.is_absolute():
                            resolved = manifest_path.parent / resolved
                        index["paths"].add(os.path.normcase(str(resolved.resolve())))
                    except OSError:
                        pass
            if entry.get("sha256"):
                digest = str(entry["sha256"])
                index["sha256"].add(digest)
                index["sha256"].add(digest[:16])
            if entry.get("size_bytes") is not None:
                try:
                    index["sizes"].add(int(entry["size_bytes"]))
                except (TypeError, ValueError):
                    pass

    for candidate in (
        checkpoint_dir / "QUARANTINE.json",
        checkpoint_dir / "quarantined" / "QUARANTINE.json",
    ):
        if candidate.is_file():
            _ingest_manifest(candidate)
    try:
        from python.model.checkpoint_registry import load_registry

        repo_root = Path(__file__).resolve().parent.parent
        registry = load_registry(repo_root)
        for rec in registry.get("checkpoints", []):
            if rec.get("status") in ("quarantined", "missing", "disqualified"):
                if rec.get("sha256"):
                    index["sha256"].add(str(rec["sha256"]))
                    index["sha256"].add(str(rec["sha256"])[:16])
                if rec.get("path"):
                    try:
                        index["paths"].add(
                            os.path.normcase(str((repo_root / rec["path"]).resolve()))
                        )
                    except OSError:
                        pass
                # NOTE: registry entries contribute resolved PATHS only, never
                # bare basenames (a registry-known quarantined
                # ".../quarantined/latest_checkpoint.pt" must not blacklist a
                # healthy "latest_checkpoint.pt" in a live checkpoint dir).
    except (FileNotFoundError, ValueError, OSError):
        logger.debug("[RESUME] checkpoint registry unavailable for quarantine index", exc_info=True)
    return index


def quarantine_verdict(path: Path, index: Dict[str, Any]) -> Optional[str]:
    """Return a reason string when ``path`` is provably quarantined."""
    try:
        resolved_path = path.resolve()
        resolved = os.path.normcase(str(resolved_path))
    except OSError:
        resolved_path = path
        resolved = os.path.normcase(str(path))
    if "quarantined" in Path(resolved).parts:
        return "inside a quarantined/ directory"
    if resolved in index["paths"]:
        return "listed in QUARANTINE.json / checkpoint registry"
    names_here = index.get("names_by_dir", {}).get(
        os.path.normcase(str(resolved_path.parent)), set()
    )
    if path.name in names_here:
        return "filename listed in this directory's QUARANTINE.json"
    # Hash/size comparison only when the size matches a quarantined entry
    # (keeps the scan cheap: two 100 MB hashes at most per candidate).
    try:
        if index["sha256"] and path.stat().st_size in index["sizes"]:
            from python.model.checkpoint_registry import compute_sha256

            digest = compute_sha256(path)
            if digest in index["sha256"] or digest[:16] in index["sha256"]:
                return "sha256 matches a quarantined checkpoint"
    except OSError:
        pass
    return None



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
        base_checkpoint: Optional[str] = "default",
        resume_checkpoint: Optional[str] = None,
        pipelined: bool = False,
        human_data_dir: str = (
            "data/dataset_training/train"
            if Path("data/dataset_training/train").is_dir()
            else "data/dataset_scaled/train"
        ),
        puzzle_csv_path: str = "data/puzzles/eval_certified_1200_1800.csv",
        openings_path: str = "data/openings/eval_openings_50.epd",
        stockfish_path: str = "tools/stockfish.exe",
        stockfish_nodes: int = 1000,
        # Opponent calibration (EV-01): full-strength Stockfish at 1000 nodes is
        # a floor opponent for the promotion match (measured 0/30 — the match
        # term of the promotion composite saturates at 0 and promotion rides on
        # a policy metric). The default is the calibrated UCI_Elo 1500 cell of
        # the baseline protocol; pass --stockfish-full-strength to restore the
        # legacy behavior deliberately.
        stockfish_elo: Optional[int] = 1500,
        allow_unverified_warm_start: bool = False,
        allow_buffer_mismatch: bool = False,
        # Audit RL-07: adopt a resumed checkpoint's foreign scheduler geometry
        # (warmup/total) only when the operator acknowledges it explicitly.
        allow_scheduler_geometry_mismatch: bool = False,
        is_verify_mode: bool = False,
        skip_baseline_eval: bool = False,
        data_role: Optional[str] = None,
        allow_uncertified_data: bool = False,
        allow_uncertified_eval: bool = False,
        require_certified_eval: bool = True,
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
        # Audit S4-08: max_wait_ms is currently INERT end-to-end — it fed only
        # the InferenceServer SHM-serving-loop knob (no production consumer);
        # the worker's dispatcher uses its own default. The CLI flag is kept
        # for interface stability and documented as a no-op until a deliberate
        # decision threads it into the worker's aggregator (behavior change).
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
        promotion_min_games: int = 100,
        promotion_min_score: float = 0.55,
        # ── Checkpoint provenance (2026-10-30 integration) ────────────────
        promotion_sprt_max_games: int = 0,
        allow_unverified_promotion: bool = False,
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
        self.pipelined = bool(pipelined)
        self._consecutive_train_failures = 0
        self._max_consecutive_train_failures = 5
        self.data_role = data_role
        self.allow_uncertified_data = allow_uncertified_data
        self.allow_uncertified_eval = allow_uncertified_eval
        self.require_certified_eval = require_certified_eval
        # Explicit operator override permitting a checkpoint whose buffer
        # snapshot contradicts the live buffer (audit SR-04; loudly logged).
        self.allow_buffer_mismatch = bool(allow_buffer_mismatch)
        self.allow_scheduler_geometry_mismatch = bool(allow_scheduler_geometry_mismatch)
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
        # Strength gate for best_eval.pt (R-07): promotion now needs a real sample and
        # a real score advantage instead of any composite-score improvement.
        self.promotion_min_games = int(promotion_min_games)
        self.promotion_min_score = float(promotion_min_score)
        # 0 disables the promotion SPRT (threshold gate only, documented);
        # >0 requires an ACCEPT_H1 SPRT vs best_eval/warm-start to promote.
        self.promotion_sprt_max_games = int(promotion_sprt_max_games)
        self.allow_unverified_promotion = bool(allow_unverified_promotion)
        self._baseline_puzzle_rate: Optional[float] = None
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
        # Persisted promotion state so a resumed run does not overwrite a better
        # best_eval.pt just because in-memory state was lost.
        self._best_eval_state_path = self.checkpoint_dir / "best_eval.json"
        self._best_eval_comp_score: Optional[float] = None
        # Only trust the persisted score when the checkpoint it describes still exists.
        if (
            self._best_eval_state_path.is_file()
            and (self.checkpoint_dir / "best_eval.pt").is_file()
        ):
            try:
                previous = json.loads(self._best_eval_state_path.read_text(encoding="utf-8"))
                self._best_eval_comp_score = float(previous["comp_score"])
                logger.info(
                    "[PROMOTION] Loaded previous best_eval composite score %.4f (step %s)",
                    self._best_eval_comp_score,
                    previous.get("step"),
                )
            except Exception as exc:
                logger.warning(
                    "[PROMOTION] Could not read %s (%s); starting promotion tracking fresh",
                    self._best_eval_state_path,
                    exc,
                )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir=self.log_dir, log_filename="phase4_rl.log")
        self.eval_csv_path = self.log_dir / "eval_history.csv"
        # Durable eval artifacts (Stage A §7): per-game match rows and the
        # puzzle-phase result are persisted incrementally with fsync so an
        # interrupted evaluation keeps everything completed so far.
        self.match_games_csv_path = self.log_dir / "match_games.csv"
        self.puzzle_eval_jsonl_path = self.log_dir / "eval_puzzle_history.jsonl"
        self._run_commit = self._git_commit()
        # ── Run identity and per-game provenance (audit SR-08) ──────────────
        # A run id ties this run's telemetry, per-game log, checkpoints (via
        # run_metadata) and shutdown summary together. The per-game JSONL
        # records every sunk game with its termination reason, result and
        # buffer-insertion statistics; the run manifest records the resolved
        # configuration and the checkpoint/buffer association at startup.
        import uuid as _uuid

        self.run_id = _uuid.uuid4().hex[:16]
        self.games_jsonl_path = self.log_dir / "games.jsonl"
        self.run_manifest_path = self.log_dir / "run_manifest.json"
        self.run_summary_path = self.log_dir / "run_summary.json"
        self._game_log_lock = threading.Lock()
        # Reporting-only counters (telemetry, never training inputs).
        self._aborted_games_seen = 0
        self._invalid_games_seen = 0
        self._records_rejected = 0

        p_cand = [
            Path(puzzle_csv_path) if puzzle_csv_path else None,
            # Contamination-free eval set (see scripts/check_puzzle_overlap.py).
            Path("data/puzzles/eval_certified_1200_1800.csv"),
            _repo_root / "data/puzzles/eval_certified_1200_1800.csv",
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
        self.puzzle_certified = self._report_puzzle_certification()
        if (
            not self.puzzle_certified
            and self.require_certified_eval
            and not self.allow_uncertified_eval
            and not self.is_verify_mode
        ):
            raise RuntimeError(
                f"Refusing to run Phase 4 against an UNCERTIFIED evaluation puzzle set "
                f"({self.puzzle_csv_path}): its certification manifest is missing, stale, or "
                f"incomplete, so reported puzzle solve rates would not be trustworthy. "
                f"Re-certify with scripts/check_puzzle_overlap.py, or pass "
                f"--allow-uncertified-eval to run with uncertified evaluation explicitly."
            )
        openings_p = Path(openings_path)
        if not openings_p.is_file() and (_repo_root / openings_p).is_file():
            openings_p = _repo_root / openings_p
        self.openings_path = openings_p

        stockfish_p = Path(stockfish_path)
        if not stockfish_p.is_file() and (_repo_root / stockfish_p).is_file():
            stockfish_p = _repo_root / stockfish_p
        self.stockfish_path = stockfish_p
        self.stockfish_nodes = stockfish_nodes
        self.stockfish_elo = stockfish_elo
        self.allow_unverified_warm_start = allow_unverified_warm_start

        self.human_data_dir = str(human_data_dir) if human_data_dir else None
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.last_eval_time = self.start_time

        # Historical evaluation metrics for trend tracking
        self.puzzle_solve_history: list[float] = []
        # Noise band for the puzzle-decline safety trigger: per-step drops at
        # or below this absolute rate do not count as declines. Calibrated to
        # one standard deviation of the certified 1,000-puzzle set
        # (~1.5 percentage points at p~0.64); see puzzle_decline_trips.
        self.puzzle_noise_band: float = 0.015
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
        # Identity of the most recently saved eval checkpoint (provenance).
        self._last_saved_ckpt_name: str = ""
        self._last_saved_ckpt_path: str = ""
        # Audit RL-27: the shutdown banner used to read this attribute with a
        # getattr fallback of 'unknown' while nothing ever assigned it; it is
        # initialized here and set by stop() when the final save succeeds.
        self._last_final_ckpt: str = ""
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

        def _build_replay_buffer(with_human_data: bool) -> ReplayBuffer:
            # Audit RL-25: the human/tactical cache composition is derived from
            # the run seed when one is given (same --seed → same cache).
            return ReplayBuffer(
                buffer_dir=buf_dir,
                capacity=actual_capacity,
                initial_human_mix=0.25,
                final_human_mix=0.05,
                taper_steps=50_000,
                human_data_dir=human_data_dir if with_human_data else None,
                human_cache_size=self.human_cache_size,
                gpu_device=self.train_device if self.train_device == "cuda" else None,
                create_new=self.reset_buffer,
                tactical_csv_path=self.tactical_csv_path,
                tactical_mix=self.tactical_mix,
                data_role=self.data_role,
                allow_uncertified_data=self.allow_uncertified_data,
                cache_rng_seed=(self.seed + 9176) if self.seed is not None else None,
            )

        # Audit RL-03: the human-data consumption gate lives inside buffer
        # construction, which precedes auto-resume selection — an uncertified
        # or invalid OPTIONAL human dataset therefore blocked even
        # crash-recovery resume of a valid RL checkpoint. The gate stays hard
        # for fresh and warm starts; only when a genuinely resumable RL
        # checkpoint exists may the run continue WITHOUT the human mix (human
        # data is optional by architecture). The dataset itself is never
        # bypassed: it is simply not consumed this run, loudly.
        from python.data.dataset_contract import DatasetContractError

        self.human_data_disabled_for_recovery = False
        try:
            self.buffer = _build_replay_buffer(with_human_data=True)
        except DatasetContractError as gate_exc:
            if self.reset_buffer or self.is_verify_mode:
                raise
            if resume_checkpoint is not None:
                # Explicit resume: a resumable checkpoint is already named.
                recovered = resume_checkpoint
            else:
                # Crash-recovery scan without a live buffer (the selection
                # scan tolerates that; the authoritative buffer reconciliation
                # still runs fail-closed inside load_checkpoint).
                self.buffer = None
                recovered = self._select_resume_checkpoint()
            if recovered is None:
                # No resumable checkpoint: this is a fresh/warm start and the
                # dataset requirement stands. Re-raise the original error.
                raise
            self.buffer = _build_replay_buffer(with_human_data=False)
            self.human_data_disabled_for_recovery = True
            resume_checkpoint = recovered
            logger.warning(
                "[RECOVERY] Human-data gate rejected the dataset (%s). A resumable RL "
                "checkpoint exists (%s), so this run continues WITHOUT the optional human "
                "mix (audit RL-03). The dataset was not bypassed and is not consumed; "
                "certify or fix it before the next fresh start.",
                gate_exc,
                Path(recovered).name,
            )

        # Detect resumable checkpoint if explicitly provided, else auto-resume from
        # the best SGD-compatible RL checkpoint (never by filename ordering — the
        # poisoned final/latest artifacts of failed runs must not win).
        if resume_checkpoint is None and not self.reset_buffer and not self.is_verify_mode:
            resume_checkpoint = self._select_resume_checkpoint()

        is_resuming = resume_checkpoint is not None
        if not is_resuming:
            if base_checkpoint == "default":
                # Default warm-start: canonical resolution via registry (audit F-13)
                from python.cli.common import get_default_checkpoint_path
                from python.model.checkpoint_registry import DefaultCheckpointError

                try:
                    resolved_default = get_default_checkpoint_path()
                    base_checkpoint = str(resolved_default)
                    logger.info(
                        "[WARM-START] Canonical default checkpoint resolved: %s",
                        base_checkpoint,
                    )
                except DefaultCheckpointError as e:
                    raise RuntimeError(
                        "Refusing to start Phase 4 RL: cannot resolve a trustworthy "
                        f"default warm-start checkpoint.\n{e}"
                    ) from e
            elif base_checkpoint is not None and (
                base_checkpoint == ""
                or (isinstance(base_checkpoint, str) and base_checkpoint.strip().lower() == "none")
            ):
                base_checkpoint = None

        active_checkpoint = resume_checkpoint if is_resuming else base_checkpoint
        self._resumed_from = str(resume_checkpoint) if is_resuming else None
        self._active_checkpoint_path = (
            str(active_checkpoint) if active_checkpoint else "random_init"
        )
        self.allow_unverified_warm_start = bool(allow_unverified_warm_start)
        # P0-4 (Stage A §5): fail-closed warm-start provenance. A missing,
        # unreadable, architecture-mismatched, unregistered, quarantined, or
        # undesignated checkpoint stops the run before any model exists — the
        # inference server must never silently fall back to random init.
        # Resume of an existing RL checkpoint is validated by
        # RLTrainer.load_checkpoint instead (registry designation applies to
        # warm starts, not crash recovery).
        self._warm_start_record: Optional[dict] = None
        if active_checkpoint and resume_checkpoint is None:
            from python.model.checkpoint_registry import validate_warm_start

            try:
                self._warm_start_record = validate_warm_start(
                    active_checkpoint,
                    repo_root=str(Path(__file__).resolve().parent.parent),
                    allow_unverified=self.allow_unverified_warm_start,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Refusing to start Phase 4 RL from '{active_checkpoint}': {e}"
                ) from e
            self._active_checkpoint_sha256 = self._warm_start_record["sha256"]
            logger.info(
                "[WARM-START] verified %s (sha256=%s, status=%s, role=%s)",
                self._warm_start_record["path"],
                self._active_checkpoint_sha256[:16],
                self._warm_start_record["status"],
                self._warm_start_record.get("role", "n/a"),
            )
        elif active_checkpoint:
            # Crash-recovery resume: verify the checkpoint is readable before
            # building anything on top of it.
            if not Path(active_checkpoint).is_file():
                raise RuntimeError(
                    f"Refusing to resume: RL checkpoint '{active_checkpoint}' does not exist."
                )
            self._active_checkpoint_sha256 = None
        else:
            logger.warning(
                "[WARM-START] base_checkpoint explicitly empty: starting from RANDOM "
                "initialization. Unapproved for any strength-relevant evaluation."
            )
            self._active_checkpoint_sha256 = None
        # Auto-detect Syzygy tablebase if not explicitly disabled
        if self.tablebase_path is None:
            for candidate in (str(_repo_root / "data" / "syzygy"), "data/syzygy", "syzygy", "/usr/share/syzygy"):
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
        # Audit S4-08: the server's min_batch_size / max_wait_ms knobs are
        # SHM-serving-loop-only (zero production consumers — the direct client
        # path batches in the worker's aggregator), so the launcher no longer
        # pretends to tune them. max_batch_size is still passed explicitly:
        # its one live production effect is the batch_fill_rate telemetry
        # normalization (predict_batch-driven), and dropping it would silently
        # re-scale that recorded metric from /128 to the /512 ctor default.
        logger.info(
            f"Initializing Inference Server from checkpoint: {active_checkpoint} "
            f"(device={self.infer_device})..."
        )
        self.server = InferenceServer(
            checkpoint_path=active_checkpoint,
            device=self.infer_device,
            max_batch_size=128,
            use_bfloat16=(self.infer_device == "cuda"),
        )
        self.client = DirectInferenceClient(self.server)

        # 3. Initialize Metrics Tracker (with inference server for batch_fill telemetry)
        self.metrics_tracker = MetricsTracker(
            log_dir=self.log_dir,
            csv_path=self.log_dir / "phase4_telemetry.csv",
            inference_server=self.server,
        )
        # Stamp the run id into telemetry rows (audit SR-08).
        self.metrics_tracker.run_id = self.run_id

        # 4. Initialize Self-Play Worker
        def sink(positions: list[SavedPosition], summary: GameSummary):
            # Sink admission guard (audit SR-01, defense in depth): a game cut
            # short by shutdown has no result and no valid experience. It is
            # refused here even if it somehow reaches the sink, so a partial
            # game can never be inserted as draw-labeled data or counted as a
            # draw in telemetry.
            if not SelfPlayWorker.accepts_for_sink(summary):
                self._aborted_games_seen += 1
                logger.warning(
                    "[SINK] refused aborted game %s (%s): no positions inserted",
                    getattr(summary, "game_id", "?"),
                    getattr(summary, "termination_reason", "?"),
                )
                return
            self.buffer.add_positions(positions)
            self.metrics_tracker.record_game(summary)
            self._append_game_record(summary, len(positions))
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
            # Deliberately 0.15 (B2-era calibration, the value used for the
            # Oct-26 buffer): the production launcher pins 15% of games to run
            # with resignation disabled, while SelfPlayGame's own default is
            # 0.10 (used by benchmarks/profiles). Both are intentional; audit
            # S2-05 records this as documentation drift, not a bug.
            disable_resign_prob=0.15,  # also stamped on self for banner/trigger text
            allow_resignation=True,
            min_resign_ply=30,
            tablebase_path=self.tablebase_path,
            tablebase_max_pieces=self.tablebase_max_pieces,
            position_sink=sink,
            seed=self.seed,
            pipelined=self.pipelined,
        )
        self.disable_resign_prob = 0.15

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
        # Production launcher: this script IS the authorized writer for the
        # run's artifact tree. The ticket records the destination so the
        # protected-artifact guard can distinguish a deliberate production run
        # from a test-scale process that merely inherited the same CWD
        # (finding CKP-04).
        from python.common.artifacts import ProductionWriteTicket

        self.artifact_ticket = ProductionWriteTicket.issue_for_launcher(
            "scripts/run_phase4_rl.py",
            reason=f"production RL campaign checkpoint_dir={self.checkpoint_dir}",
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
            artifact_ticket=self.artifact_ticket,
        )

        # Audit RL-03: record the recovery decision durably in the run ledger
        # so a run that trained without the human mix is identifiable from its
        # checkpoints alone.
        if getattr(self, "human_data_disabled_for_recovery", False):
            cfg = dict(self.trainer.run_metadata.get("config") or {})
            cfg["human_data"] = "disabled_for_recovery"
            self.trainer.run_metadata["config"] = cfg

        if is_resuming:
            self.trainer.load_checkpoint(
                resume_checkpoint,
                allow_buffer_mismatch=self.allow_buffer_mismatch,
                reset_buffer=self.reset_buffer,
                allow_scheduler_geometry_mismatch=self.allow_scheduler_geometry_mismatch,
            )
            if hasattr(self.trainer, "direct_server") and self.trainer.direct_server is not None:
                self.server.model_version = self.trainer.direct_server.model_version
            logger.info(
                f"Resumed RL training from {resume_checkpoint} at step {self.trainer.step_num} (model_version={self.server.model_version})"
            )
            # The resumed trainer builds its own ledger (with the predecessor
            # recorded); attach this run's identity and the resume provenance
            # to that ledger instead of overwriting it, so the predecessor
            # lineage and the buffer resume decision survive (CKP-01/CKP-05;
            # SR-08 provenance fields preserved).
            ledger = dict(getattr(self.trainer, "run_metadata", {}) or {})
            ledger["run_id"] = self.run_id
            ledger["resumed_from"] = str(resume_checkpoint)
            ledger["resumed_at_step"] = int(getattr(self.trainer, "step_num", 0))
            ledger["seed"] = self.seed
            ledger["active_checkpoint"] = str(resume_checkpoint)
            self.trainer.run_metadata = ledger
        else:
            # Warm start (or cold start): the weights already came in via the
            # inference server and RL counters are fresh at 0. The warm-start
            # checkpoint's own step counter is inherited supervised progress —
            # provenance metadata, never RL pacing progress (PR-03).
            # Audit SR-02 / SLQ-07 (fixed independently on both branches): this
            # read `self._warm_start_info`, an attribute never assigned anywhere
            # — every construction path raised AttributeError before training.
            # The validated record is `self._warm_start_record`.
            ws = self._warm_start_record or {}
            from python.model.checkpoint_schema import build_run_metadata

            self.trainer.run_metadata = build_run_metadata(
                run_type="rl-warm-start" if base_checkpoint else "rl-fresh",
                seed=self.seed,
                run_id=self.run_id,
                config=dict(self.trainer.run_metadata.get("config") or {}),
                parent_checkpoint=(
                    dict(ws) if base_checkpoint and ws else None
                ),
                extra={
                    "warm_start": str(base_checkpoint) if base_checkpoint else "random_init",
                    "warm_start_sha256": ws.get("sha256", "unverified"),
                    "warm_start_step": ws.get("step", 0),
                },
            )
            logger.info(
                "[RUN-STATUS] new ledger run_id=%s type=%s state=%s",
                self.trainer.run_metadata["run_id"],
                self.trainer.run_metadata["run_type"],
                self.trainer.run_metadata["run_state"],
            )
            if ws.get("step"):
                self.trainer.inherited_step_num = int(ws["step"])
                self.trainer.warm_start_path = str(base_checkpoint)
                self.trainer.warm_start_step = int(ws["step"])
            logger.info(
                "[WARM-START] RL counters start at 0 (inherited supervised step=%s as provenance)",
                ws.get("step", 0),
            )

        # Load puzzles
        self.puzzles = load_puzzles(self.puzzle_csv_path)
        logger.info(f"Loaded {len(self.puzzles)} fixed Lichess puzzles for 6-hour evaluation.")

    @property
    def _warm_start_info(self) -> Optional[dict]:
        """Backward-compatibility alias for self._warm_start_record (audit F-12)."""
        return self._warm_start_record

    def _select_resume_checkpoint(self) -> Optional[str]:
        """Pick the best RL-resumable checkpoint in ``checkpoint_dir``, or None.

        Eligibility contract (audit SR-03; every condition is code-enforced):

        1. readable, model weights present, architecture 12x128 (existing);
        2. NOT quarantined: files listed in a ``QUARANTINE.json`` beside them
           (or in the checkpoint dir), files inside a ``quarantined/``
           directory, and files whose sha256/size match a quarantined registry
           entry are rejected outright — a quarantined checkpoint can never
           become eligible through an automatic selection fallback;
        3. NOT disqualified: registry entries with status
           ``quarantined``/``missing``/``disqualified``;
        4. type evidence: ``checkpoint_type == "rl"`` is eligible; an explicit
           non-RL ``checkpoint_type`` is never eligible; a pre-stamp legacy
           file (no type field) is eligible only when its per-parameter
           optimizer state positively identifies an SGD trainer (empty/absent
           optimizer state with no type stamp is exactly the hole this finding
           closed — such a file is skipped with a loud warning); a weights-only
           container (export/curriculum payload) is never eligible (CKP-08/09);
        5. optimizer state SGD-compatible or empty (existing);
        6. buffer association: when the candidate carries a full ``buffer_state``
           its ``buffer_id`` must match the configured buffer directory unless
           the operator set ``allow_buffer_mismatch`` (logged).

        Selection among eligible candidates is by (step, mtime) — never by
        filename — and is deterministic.
        """
        from python.train.train_rl import inspect_rl_checkpoint, is_resumable_candidate

        rejected: List[str] = []
        candidates: List[Dict[str, Any]] = []
        try:
            files = sorted(self.checkpoint_dir.glob("*.pt"))
        except OSError:
            return None

        quarantine = load_quarantine_index(self.checkpoint_dir)
        live_buffer_id = None
        buffer = getattr(self, "buffer", None)
        if buffer is not None and hasattr(buffer, "buffer_id"):
            live_buffer_id = buffer.buffer_id
        allow_mismatch = bool(getattr(self, "allow_buffer_mismatch", False))

        for f in files:
            if not f.is_file():
                continue
            verdict = quarantine_verdict(f, quarantine)
            if verdict is not None:
                rejected.append(f"{f.name}: quarantined ({verdict})")
                continue
            info = inspect_rl_checkpoint(f)
            if info is None:
                rejected.append(f"{f.name}: unreadable or invalid")
                continue
            # Weights-only containers (exports, model-only curriculum files)
            # are not resume points even though they load: a candidate must
            # carry SGD state or declare itself an RL checkpoint (audit
            # CKP-08/CKP-09). This shared gate also refuses foreign optimizer
            # states and unstamped files without SGD evidence (audit SR-03).
            acceptable, reject_reason = is_resumable_candidate(info)
            if not acceptable:
                rejected.append(f"{f.name}: {reject_reason}")
                continue
            ckpt_type = info.get("checkpoint_type")
            if ckpt_type is not None and ckpt_type != "rl":
                rejected.append(
                    f"{f.name}: checkpoint_type={ckpt_type!r} is not an RL training checkpoint"
                )
                continue
            buffer_state = info.get("buffer_state")
            if (
                live_buffer_id is not None
                and isinstance(buffer_state, dict)
                and buffer_state.get("buffer_id") not in (None, live_buffer_id)
            ):
                line = (
                    f"{f.name}: buffer_state.buffer_id={buffer_state.get('buffer_id')} does not "
                    f"match the configured buffer {live_buffer_id}"
                )
                if allow_mismatch:
                    logger.warning("[RESUME] including mismatched candidate (override): %s", line)
                else:
                    rejected.append(line + " (pass --allow-buffer-mismatch to override)")
                    continue
            candidates.append(info)
        for line in rejected:
            logger.warning("[RESUME] excluded checkpoint: %s", line)
        if not candidates:
            if files:
                logger.warning(
                    "[RESUME] no eligible RL checkpoint in %s (%d file(s) rejected); "
                    "starting from the warm-start base instead",
                    self.checkpoint_dir,
                    len(rejected),
                )
            return None
        best = max(candidates, key=lambda c: (c["step"], c["mtime"]))
        logger.info(
            "[RESUME] selected %s (step=%s, type=%s, optimizer=%s) — %d eligible candidate(s)",
            best["path"],
            best["step"],
            best.get("checkpoint_type") or "unstamped(SGD-evidence)",
            best["optimizer_class"],
            len(candidates),
        )
        return best["path"]

    def _init_eval_csv(self) -> None:
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
            # Stage A §7 provenance columns (appended; older files keep theirs).
            "model_sha256",
            "stockfish_sha256_16",
            "commit",
            "partial",
            # Opponent provenance + search-aware puzzle metric (EV-01/EV-04,
            # carried from the strength-remediation branch; appended).
            "sf_engine",
            "sf_elo",
            "sf_elo_verified",
            "mcts_puzzle_sims",
            "mcts_puzzle_solved",
            "mcts_puzzle_total",
            "mcts_puzzle_rate",
            # Promotion evidence (audit CKP-15, appended 2026-10-31).
            "promotion_decision",
            "promotion_reason",
            "promotion_generation",
        ]
        need_write = True
        if self.eval_csv_path.exists():
            with open(self.eval_csv_path, "r", newline="", encoding="utf-8") as f:
                first_line = f.readline().strip()
                has_data_rows = bool(f.readline().strip())
            if first_line == ",".join(headers):
                need_write = False
            elif has_data_rows:
                raise RuntimeError(
                    f"{self.eval_csv_path} has data rows with an older column "
                    "schema; archive it (or point --log-dir elsewhere) so "
                    "incompatible eval schemas are not mixed silently."
                )
            else:
                logger.info(
                    "[EVAL] Upgrading header-only %s with new provenance columns",
                    self.eval_csv_path,
                )
        if need_write:
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                f.flush()
                os.fsync(f.fileno())
        # Per-game durability (F-15): every completed match game is persisted
        # the moment it finishes, so an interrupted evaluation keeps its
        # evidence and a resumed run can deduplicate by (step, seed, color).
        #
        # Audit RL-21: the previous code wrote a 14-column header and then a
        # dead second block re-wrote a 16-column header for a file that by
        # then always existed, while _persist_match_game appended 16-column
        # rows — a permanently ragged ledger. One authoritative header now,
        # upgraded in place only while header-only (eval-history policy);
        # files with data rows in an older schema must be archived instead so
        # per-game evidence is never mixed across schemas. The trailing
        # ``adjudicated`` column carries the RL-20 value-cutoff provenance.
        self.match_games_csv_path = self.log_dir / "match_games.csv"
        match_headers = [
            "timestamp",
            "step",
            "model_version",
            "checkpoint_file",
            "model_sha256",
            "opening_idx",
            "engine_is_white",
            "result",
            "seed",
            "stockfish_elo",
            "stockfish_nodes",
            "mcts_sims",
            "search_mode",
            "plies",
            "commit",
            "partial",
            "adjudicated",
        ]
        match_need_write = True
        if self.match_games_csv_path.exists():
            with open(self.match_games_csv_path, "r", newline="", encoding="utf-8") as f:
                first_line = f.readline().strip()
                has_data_rows = bool(f.readline().strip())
            if first_line == ",".join(match_headers):
                match_need_write = False
            elif has_data_rows:
                raise RuntimeError(
                    f"{self.match_games_csv_path} has data rows with an older column "
                    "schema; archive it (or point --log-dir elsewhere) so per-game "
                    "match evidence is not mixed across schemas."
                )
            else:
                logger.info(
                    "[EVAL] Upgrading header-only %s with new provenance columns",
                    self.match_games_csv_path,
                )
        if match_need_write:
            with open(self.match_games_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(match_headers)
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def _git_commit() -> str:
        """Short commit hash of the running tree ('unknown' outside a repo)."""
        import subprocess

        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return out.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _current_model_sha(self) -> str:
        """SHA-256 of the exact weights being evaluated (first 16 hex chars)."""
        path = getattr(self, "_last_saved_ckpt_path", None)
        if not path:
            return "unknown"
        try:
            from python.model.checkpoint_registry import compute_sha256

            return compute_sha256(path)[:16]
        except Exception as exc:
            logger.warning("[EVAL] Could not hash checkpoint for provenance: %s", exc)
            return "unknown"

    def _append_eval_row(self, row: list) -> None:
        """Append one eval_history row and fsync so interruption keeps it.

        Held under the artifact lock so two processes appending to the same
        history cannot interleave partial rows (SUS-02); the wait is short
        because appends are rare compared with training steps.
        """
        from python.common.artifacts import artifact_lock

        with artifact_lock(self.eval_csv_path, timeout=15.0, owner="phase4-eval-history"):
            with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
                f.flush()
                os.fsync(f.fileno())

    def _persist_puzzle_result(self, solve_rate: float, solved: int, total: int) -> None:
        """Persist the completed puzzle-phase result before the engine match starts."""
        record = {
            "type": "puzzle_policy",
            "timestamp": time.time(),
            "step": self.trainer.step_num,
            "model_version": self.server.model_version,
            "checkpoint_file": getattr(self, "_last_saved_ckpt_name", None),
            "model_sha256": self._current_model_sha(),
            "solve_rate": solve_rate,
            "solved": solved,
            "total": total,
            "commit": self._run_commit,
        }
        with open(self.puzzle_eval_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _append_game_record(self, summary: GameSummary, num_records: int) -> None:
        """Append one per-game provenance row to games.jsonl (audit SR-08).

        Runs on game threads, so the append is lock-protected and flushed (not
        fsynced) to keep the sink off the critical path while remaining durable
        across a clean stop. Every row carries the run id, the producing game
        id and model version, the termination reason and result, the discarded/
        aborted flags, and the buffer insertion statistics observed right after
        this game's records were committed.
        """
        try:
            row = {
                "run_id": self.run_id,
                "timestamp": time.time(),
                "game_id": int(getattr(summary, "game_id", 0) or 0),
                "model_version": int(getattr(summary, "model_version", 0) or 0),
                "termination_reason": getattr(summary, "termination_reason", "unknown"),
                "result": getattr(summary, "result", None),
                "aborted": bool(getattr(summary, "aborted", False)),
                "adjudicated": bool(getattr(summary, "adjudicated", False)),
                "resigned": bool(getattr(summary, "resigned", False)),
                "resigned_side_white": getattr(summary, "resigned_side_white", None),
                "would_have_resigned": bool(getattr(summary, "would_have_resigned", False)),
                "is_false_resign": bool(getattr(summary, "is_false_resign", False)),
                "cut_ply": getattr(summary, "cut_ply", None),
                "cut_piece_count": getattr(summary, "cut_piece_count", None),
                "raw_wdl": getattr(summary, "raw_wdl", None),
                "plies": int(getattr(summary, "length_plies", 0) or 0),
                "duration_sec": float(getattr(summary, "duration_sec", 0.0) or 0.0),
                "total_simulations": int(getattr(summary, "total_simulations", 0) or 0),
                "records": int(num_records),
                "buffer_size_after": int(getattr(self.buffer, "size", 0)),
                "buffer_write_head_after": int(getattr(self.buffer, "write_head", 0)),
                "buffer_total_added_after": int(getattr(self.buffer, "total_added", 0)),
                "buffer_generation": int(getattr(self.buffer, "generation", 0)),
            }
            with self._game_log_lock:
                with open(self.games_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    f.flush()
        except OSError as exc:
            logger.warning("[GAMES] failed to append game record: %s", exc)

    def _write_run_manifest(self, *, resuming_from: Optional[str]) -> None:
        """Persist this run's identity and resolved configuration (audit SR-08).

        Written once at startup (atomic tmp+replace). Records the run id, the
        master seed, the active checkpoint and its sha256 (when verified), the
        buffer identity/capacity/schema/generation, the resolved settings, and
        the run's commit — the durable "what exactly ran" record whose absence
        made the historical campaign unreproducible.
        """
        payload = {
            "run_id": self.run_id,
            "started_at": time.time(),
            "commit": self._run_commit,
            "seed": self.seed,
            "resumed_from": resuming_from,
            "active_checkpoint": self._active_checkpoint_path,
            "active_checkpoint_sha256": getattr(self, "_active_checkpoint_sha256", None),
            "warm_start_record": getattr(self, "_warm_start_record", None),
            "buffer": {
                "dir": str(self.buffer.buffer_dir),
                "buffer_id": getattr(self.buffer, "buffer_id", None),
                "capacity": self.buffer.capacity,
                "schema_version": getattr(self.buffer, "schema_version", None),
                "generation": getattr(self.buffer, "generation", None),
                "size": self.buffer.size,
                "write_head": self.buffer.write_head,
                "total_added": self.buffer.total_added,
                "selfplay_sampling_policy": getattr(
                    self.buffer, "selfplay_sampling_policy", None
                ),
            },
            "config": {
                "max_hours": self.max_hours,
                "eval_interval_sec": self.eval_interval_sec,
                "num_concurrent_games": self.num_concurrent_games,
                "full_search_sims": self.full_search_sims,
                "fast_search_sims": self.fast_search_sims,
                "base_lr": self.base_lr,
                "min_lr": self.min_lr,
                "tactical_mix": self.tactical_mix,
                "policy_target_temp": self.policy_target_temp,
                "entropy_bonus": self.entropy_bonus,
                "max_reuse_ratio": self.max_reuse_ratio,
                "human_data_dir": getattr(self, "human_data_dir", None),
                # Audit RL-03: True when a resumable checkpoint let this run
                # proceed without the (gated) optional human dataset.
                "human_data_disabled_for_recovery": bool(
                    getattr(self, "human_data_disabled_for_recovery", False)
                ),
                "data_role": self.data_role,
                "allow_uncertified_data": self.allow_uncertified_data,
                "device": self.device,
                "train_device": self.train_device,
                "infer_device": self.infer_device,
                "tablebase_path": self.tablebase_path,
                "stockfish_nodes": self.stockfish_nodes,
                "stockfish_elo": self.stockfish_elo,
            },
        }
        try:
            tmp = self.run_manifest_path.with_name(self.run_manifest_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.run_manifest_path)
        except OSError as exc:
            logger.warning("[RUN] failed to write run manifest: %s", exc)

    def _write_run_summary(self, status: str) -> None:
        """Persist shutdown diagnostics and counters (audit SR-08)."""
        payload = {
            "run_id": self.run_id,
            "stopped_at": time.time(),
            "status": status,
            "exit_reason": getattr(self, "_exit_reason", "unknown"),
            "games": {
                "completed": int(getattr(self.metrics_tracker, "total_games", 0)),
                "aborted": int(getattr(self.worker, "aborted_games", 0))
                + int(self._aborted_games_seen),
                "invalid": int(getattr(self.worker, "invalid_games", 0))
                + int(self._invalid_games_seen),
                "records_rejected": int(getattr(self.buffer, "rejected_records", 0))
                + int(self._records_rejected),
                "positions_inserted": int(getattr(self.buffer, "total_added", 0)),
            },
            "worker": {
                "inference_failures": int(
                    getattr(self.worker.aggregator, "inference_failure_count", 0)
                ),
                "failed_flushes": int(getattr(self.worker.aggregator, "failed_flushes", 0)),
                "failed_eval_requests": int(
                    getattr(self.worker.aggregator, "failed_eval_requests", 0)
                ),
                "staleness_fallbacks": int(
                    getattr(self.buffer, "staleness_fallback_count", 0)
                ),
            },
            "trainer": {
                "rl_steps": int(getattr(self.trainer, "step_num", 0)),
                "run_steps": int(self._steps_this_run),
                "push_success": int(getattr(self.trainer, "push_success_count", 0)),
                "push_failure": int(getattr(self.trainer, "push_failure_count", 0)),
            },
            "buffer": {
                "buffer_id": getattr(self.buffer, "buffer_id", None),
                "size": int(getattr(self.buffer, "size", 0)),
                "write_head": int(getattr(self.buffer, "write_head", 0)),
                "total_added": int(getattr(self.buffer, "total_added", 0)),
                "generation": int(getattr(self.buffer, "generation", 0)),
            },
            "checkpoint": {
                "last_saved": getattr(self, "_last_saved_ckpt_name", None),
                "final_path": str(self.checkpoint_dir / "phase4_final_checkpoint.pt"),
            },
        }
        try:
            tmp = self.run_summary_path.with_name(self.run_summary_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.run_summary_path)
        except OSError as exc:
            logger.warning("[RUN] failed to write run summary: %s", exc)

    def _persist_match_game(self, game: dict, partial: bool = False) -> None:
        """Persist one completed match game (Stage A §7 incremental durability)."""
        row = [
            time.time(),
            self.trainer.step_num,
            self.server.model_version,
            getattr(self, "_last_saved_ckpt_name", None),
            self._current_model_sha(),
            game.get("opening_idx"),
            int(bool(game.get("engine_is_white"))),
            game.get("result"),
            game.get("seed"),
            self.stockfish_elo if self.stockfish_elo else "full",
            self.stockfish_nodes,
            self.eval_sims if self.eval_sims is not None else (10 if self.is_verify_mode else 400),
            "serial",
            len(game.get("moves", []) or []),
            self._run_commit,
            int(partial),
            # Audit RL-20: value-cutoff adjudication provenance.
            int(bool(game.get("adjudicated", False))),
        ]
        with open(self.match_games_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def _promotion_gate(self, match_res: dict, puzzle_rate: Optional[float] = None) -> tuple[bool, str]:
        """Strength gate for best_eval.pt promotion (R-07, hardened per EV-01/EV-04).

        A 100-game match has an Elo confidence interval of roughly ±90-140 Elo, and the
        composite score weights it at 0.7, so a purely luckier match can out-score a
        stronger engine. Require a real sample, a real score, a COMPLETE match record
        (an interrupted/failed eval must never read as a promotion signal), and that
        the candidate still retains baseline supervised puzzle strength.
        """
        if match_res.get("interrupted"):
            return (
                False,
                f"match interrupted after {match_res.get('completed_games', 0)} games "
                "(partial record is never a promotion signal)",
            )
        games = int(match_res.get("total_games", 0) or 0)
        score = float(match_res.get("engine_score", 0.0) or 0.0)
        if games < self.promotion_min_games:
            return (
                False,
                f"insufficient sample ({games} < {self.promotion_min_games} games)",
            )
        if score < self.promotion_min_score:
            return (
                False,
                f"engine score {score:.3f} < required {self.promotion_min_score:.3f}",
            )
        if match_res.get("interrupted"):
            return (False, "match interrupted before completion (record incomplete)")
        if match_res.get("completed_games", games) < games:
            return (
                False,
                f"match record incomplete ({match_res.get('completed_games')} of {games} games persisted)",
            )
        # Baseline retention (EV-01/B4): a candidate that loses the warm-start's
        # supervised puzzle strength must not promote on a lucky match score.
        baseline = self._baseline_puzzle_rate
        if baseline is not None and puzzle_rate is not None:
            retention_band = max(2.0 * self.puzzle_noise_band, 0.03)
            if puzzle_rate < baseline - retention_band:
                return (
                    False,
                    f"puzzle solve rate {puzzle_rate:.3f} below baseline retention "
                    f"({baseline:.3f} - band {retention_band:.3f})",
                )
        return True, f"{games} games, engine score {score:.3f}"

    def _run_promotion_sprt(self, candidate_ckpt: Path, openings: int) -> tuple[str, str]:
        """Bounded SPRT of the candidate against the current promotion baseline.

        The baseline is the current ``best_eval.pt`` when one exists, otherwise
        the run's warm-start checkpoint. Returns ``(decision, detail)``; only
        ``ACCEPT_H1`` promotes (EV-03: threshold passes can be noise, and an
        INCONCLUSIVE result must not promote either).
        """
        import shutil
        import tempfile

        base_path = self.checkpoint_dir / "best_eval.pt"
        baseline_label = "best_eval"
        if not base_path.exists():
            ws = str(getattr(self, "_active_checkpoint_path", "") or "")
            if not ws or ws == "random_init":
                return "UNAVAILABLE", "no best_eval and no warm-start baseline to test against"
            base_path = Path(ws)
            baseline_label = "warm-start"
        if not base_path.exists():
            return "UNAVAILABLE", f"baseline checkpoint missing: {base_path}"

        with tempfile.TemporaryDirectory(prefix="sprt_promotion_") as td:
            cand_copy = Path(td) / "candidate.pt"
            base_copy = Path(td) / "base.pt"
            shutil.copyfile(candidate_ckpt, cand_copy)
            shutil.copyfile(base_path, base_copy)
            try:
                from scripts.run_sprt import SprtMatchRunner

                runner = SprtMatchRunner(
                    candidate_checkpoint=str(cand_copy),
                    base_checkpoint=str(base_copy),
                    device=self.device,
                    sims=self.eval_sims if self.eval_sims else 400,
                    workers=8,
                )
                result = runner.run_sprt(
                    f"promotion_step{self.trainer.step_num}",
                    {"c_puct": 2.0, "fpu_reduction": 0.2},
                    {"c_puct": 2.0, "fpu_reduction": 0.2},
                    Path(td) / "out",
                    max_games=self.promotion_sprt_max_games,
                )
            except Exception as e:
                logger.exception("Promotion SPRT failed; refusing to promote on failure")
                return "FAILED", f"{type(e).__name__}: {e}"
        s = result["sprt"]
        detail = (
            f"{s['n']} games W{s['wins']}/D{s['draws']}/L{s['losses']} vs {baseline_label}, "
            f"LLR {s['llr']:+.2f} bounds [{s['lower_bound']:.2f}, {s['upper_bound']:.2f}]"
        )
        logger.info("[PROMOTION SPRT] %s — %s", result["final_status"], detail)
        self._event("PROMOTION_SPRT", f"{result['final_status']}: {detail}")
        return result["final_status"], detail

    def _save_best_eval_state(self, **fields: object) -> None:
        payload = dict(fields)
        payload["saved_at"] = time.time()
        tmp = self._best_eval_state_path.with_name(self._best_eval_state_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self._best_eval_state_path)

    def _report_puzzle_certification(self) -> bool:
        """Verify the eval puzzle set's certification before use (audit F-10).

        Delegates to :func:`python.data.eval_certification.verify_eval_certification`,
        which re-checks the certification manifest against the CURRENT bytes of the
        eval CSV and every recorded training source. A stale or incomplete
        certification is reported as NOT certified; when ``require_certified_eval``
        is set (default), the run refuses to start rather than evaluating against
        an uncertified set.
        """
        from python.data.eval_certification import verify_eval_certification

        certified, reasons, data = verify_eval_certification(self.puzzle_csv_path)
        if not certified:
            logger.warning(
                "[EVAL] Puzzle set %s is NOT certified: %s. Solve rates may measure "
                "memorisation; re-certify with scripts/check_puzzle_overlap.py.",
                self.puzzle_csv_path,
                "; ".join(reasons),
            )
            return False
        logger.info(
            "[EVAL] Puzzle set %s certified: %s puzzles, overlap_in_output=%s, "
            "source_overlap=%s/%s, output hash verified",
            self.puzzle_csv_path,
            data.get("output_puzzles"),
            data.get("overlap_in_output"),
            data.get("overlap_either"),
            data.get("candidate_puzzles"),
        )
        return True

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
        self._last_saved_ckpt_name = ckpt_name
        self._last_saved_ckpt_path = str(self.checkpoint_dir / ckpt_name)
        ckpt_path = self.checkpoint_dir / ckpt_name
        self.trainer.save_checkpoint(ckpt_path, keep_last_n=self.keep_checkpoints)
        self._known_ckpt_names.add(ckpt_name)
        self._event("CHECKPOINT", f"saved {ckpt_path} sha={short_hash(str(ckpt_path))} (eval)")

        # DI-17: Re-verify puzzle evaluation set certification per eval cycle.
        # Catches mid-run eval CSV swap or corruption across long-running cycles.
        self.puzzle_certified = self._report_puzzle_certification()
        if (
            not self.puzzle_certified
            and self.require_certified_eval
            and not self.allow_uncertified_eval
            and not self.is_verify_mode
        ):
            raise RuntimeError(
                f"[EVAL CYCLE] Refusing evaluation cycle: puzzle set {self.puzzle_csv_path} "
                f"failed certification check. Manifest is missing, stale, or hash mismatched."
            )

        # 1. Evaluate Puzzles (policy top-1) — a policy metric, never a rating.
        logger.info(
            f"[EVAL CYCLE] Evaluating {len(self.puzzles) if not max_puzzles else max_puzzles} puzzles..."
        )
        puzzles_to_run = self.puzzles[:max_puzzles] if max_puzzles else self.puzzles
        evaluator = PuzzleEvaluator(
            model=self.trainer.model, device=self.device, puzzles=puzzles_to_run
        )
        puzzle_res = evaluator.evaluate_policy(batch_size=128)
        solve_rate = puzzle_res["solve_rate"]
        # 1b. Search-aware puzzle evaluation (EV-04): the policy head alone is
        # insensitive to search regressions; the MCTS path measures the whole
        # play pipeline on the same fixed certified set.
        mcts_eval_sims = 50 if self.is_verify_mode else 200
        try:
            mcts_res = evaluator.evaluate_mcts(num_simulations=mcts_eval_sims)
            mcts_rate = float(mcts_res.get("solve_rate", 0.0))
            mcts_solved = int(mcts_res.get("solved", 0))
            mcts_total = int(mcts_res.get("total", 0))
            logger.info(
                f"[EVAL CYCLE] MCTS-{mcts_eval_sims} puzzle solve rate: {mcts_rate * 100:.2f}% "
                f"({mcts_solved}/{mcts_total})"
            )
        except Exception as e:
            logger.exception("MCTS puzzle evaluation failed; recording as unavailable")
            mcts_rate, mcts_solved, mcts_total = None, 0, 0
        self._event(
            "EVAL",
            f"puzzles solve_rate={solve_rate * 100:.2f}% "
            f"({puzzle_res['solved']}/{puzzle_res['total']}) "
            f"mcts_{mcts_eval_sims}={mcts_rate if mcts_rate is not None else 'unavailable'}",
        )
        logger.info(
            f"[EVAL CYCLE] Puzzle solve rate: {solve_rate * 100:.2f}% ({puzzle_res['solved']}/{puzzle_res['total']})"
        )

        if self._baseline_puzzle_rate is None:
            # First completed eval of this run anchors baseline retention: at
            # run start (or --skip-baseline-eval absent) this IS the warm
            # start's supervised strength.
            self._baseline_puzzle_rate = solve_rate
            logger.info(f"[EVAL CYCLE] Baseline puzzle rate anchored at {solve_rate:.4f}")

        # Persist the completed puzzle phase BEFORE the potentially long engine
        # match (Stage A §7): an interrupted match must not lose the puzzle row.
        try:
            self._persist_puzzle_result(solve_rate, puzzle_res["solved"], puzzle_res["total"])
        except OSError:
            logger.exception("Failed to persist puzzle eval result; continuing eval cycle")

        self.puzzle_solve_history.append(solve_rate)

        # 2. Evaluate Match vs Stockfish
        logger.info(
            f"[EVAL CYCLE] Running {openings * 2}-game match vs Stockfish ({self.stockfish_nodes} nodes)..."
        )
        # R-08: at 200 simulations the engine is well below its policy-value strength and
        # the match floors out against Stockfish(1000 nodes), hiding real improvement.
        # 400 sims keeps the 100-game eval affordable while restoring discrimination.
        eval_sims = (
            self.eval_sims if self.eval_sims is not None else (10 if self.is_verify_mode else 400)
        )
        elo_note = f", UCI_Elo {self.stockfish_elo}" if self.stockfish_elo else ", full strength"
        logger.info(
            f"[EVAL CYCLE] Running {openings * 2}-game match vs Stockfish "
            f"({self.stockfish_nodes} nodes{elo_note})..."
        )
        match_runner = MatchRunner(
            model=self.trainer.model,
            device=self.device,
            stockfish_path=self.stockfish_path,
            stockfish_nodes=self.stockfish_nodes,
            stockfish_elo=self.stockfish_elo,
            mcts_sims=eval_sims,
            c_puct=2.0,
            openings_path=self.openings_path,
            num_workers=8,
            search_mode="serial",
            # Audit RL-27: eval games are seeded from the run's --seed when
            # one is given (previously the match always used the MatchRunner
            # default regardless of the run seed), and the eval-history row
            # below records THIS seed rather than the compile-time MCTS
            # default.
            seed=int(self.seed) if self.seed is not None else 42,
        )
        try:
            match_pgn_dir = self.log_dir / "matches"
            match_pgn_dir.mkdir(parents=True, exist_ok=True)
            match_res = match_runner.run_match(
                num_openings=openings,
                pgn_path=match_pgn_dir / f"eval_step{self.trainer.step_num}.pgn",
                on_game_result=self._persist_match_game,
            )
        finally:
            match_runner.close()

        if match_res.get("interrupted"):
            logger.error(
                "[EVAL CYCLE] Match was INTERRUPTED after %d/%s games (%s); the eval row "
                "below is marked partial and the promotion gate will refuse it.",
                match_res.get("completed_games"),
                openings * 2,
                match_res.get("error"),
            )
            self._event(
                "EVAL_INTERRUPTED",
                f"match partial: {match_res.get('completed_games')} games; "
                f"error={match_res.get('error')}",
            )

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
        model_sha = self._current_model_sha()
        partial = bool(match_res.get("interrupted"))
        engine_info = match_res.get("engine") or {}
        stockfish_sha16 = (engine_info.get("sha256") or "unknown")[:16]

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
            # Audit RL-27: the seed the match actually ran with (the compile-time
            # MCTS default was recorded here before, even when the run seeded
            # the match from --seed).
            int(match_runner.seed),
            eval_config_hash(
                self.stockfish_nodes,
                eval_sims,
                2.0,
                self.openings_path,
                8,
                policy_target_temp=self.policy_target_temp,
                entropy_bonus=self.entropy_bonus,
                stockfish_elo=self.stockfish_elo,
                seed=int(match_runner.seed),
            ),
            match_res.get("search_mode", "serial"),
            0,
        ]
        # P0-5: signature is (wins, draws, losses) and the return is a dict —
        # the previous call passed (wins, losses, draws), inverting the sign of
        # every recorded Elo, and the 3-tuple unpacking of the dict raised, so
        # the except wrote "n/a" into every row. On an interrupted match the
        # tally covers exactly the completed games, so the interval is computed
        # from actual games only.
        row.extend(
            eval_elo_columns(
                match_res["engine_wins"], match_res["engine_draws"], match_res["engine_losses"]
            )
        )
        row.extend([model_sha, stockfish_sha16, self._run_commit, int(partial)])
        # Opponent provenance + search-aware puzzle metric (append-only).
        row.extend(
            [
                engine_info.get("engine_name", "unknown"),
                self.stockfish_elo if self.stockfish_elo else "full",
                int(bool(engine_info.get("elo_verified"))),
                mcts_eval_sims,
                mcts_solved,
                mcts_total,
                round(mcts_rate, 4) if mcts_rate is not None else "unavailable",
            ]
        )
        # Check safety trigger: puzzle solve rate falls for 3 consecutive checks
        # (noise band: each step must drop by MORE than the band to count, so
        # jitter at the 1e-4 level cannot trip a critical stop; semantics
        # otherwise unchanged). Runs before promotion so a tripping run never
        # promotes.
        if puzzle_decline_trips(self.puzzle_solve_history, self.puzzle_noise_band):
            last4 = self.puzzle_solve_history[-4:]
            self._append_eval_row(
                row + ["not_evaluated", "safety stop triggered before promotion", ""]
            )
            raise SafetyTriggerException(
                f"CRITICAL STOP TRIGGER: Puzzle solve rate fell for 3 consecutive checks: "
                f"{[round(x, 4) for x in last4]}"
            )

        # Promotion gate: save models/checkpoints_rl/best_eval.pt only when the candidate
        # is demonstrably strong, then only when it improves on the previous best score.
        comp_score = float(match_res["engine_score"]) * 0.7 + float(solve_rate) * 0.3
        eligible, gate_reason = self._promotion_gate(match_res, puzzle_rate=solve_rate)
        if eligible and self.promotion_sprt_max_games > 0:
            # Sequential-testing protocol (EV-03): a threshold pass alone can be
            # noise; when enabled, promotion additionally requires an SPRT
            # decision (never an INCONCLUSIVE or REJECT_H1 result).
            sprt_decision, sprt_detail = self._run_promotion_sprt(ckpt_path, openings)
            if sprt_decision != "ACCEPT_H1":
                eligible = False
                gate_reason = f"SPRT {sprt_decision}: {sprt_detail}"
        elif eligible:
            # F-11 remediation: With SPRT disabled, require explicit opt-in via allow_unverified_promotion
            if not self.allow_unverified_promotion:
                eligible = False
                gate_reason = (
                    "SPRT disabled (promotion_sprt_max_games=0); promotion refused without statistical validation. "
                    "Enable --promotion-sprt-max-games >= 100 or pass --allow-unverified-promotion."
                )
            else:
                adjudicated = int(match_res.get("adjudicated_games", 0) or 0)
                logger.warning(
                    "[PROMOTION] PROVISIONAL: promoting on threshold evidence only — SPRT is "
                    "disabled and --allow-unverified-promotion was passed. A %d-game match has an Elo CI of "
                    "roughly ±90-140.%s",
                    int(match_res.get("total_games", 0) or 0),
                    f" {adjudicated} of the completed games were value-cutoff adjudicated "
                    "(see the match_games.csv adjudicated column)."
                    if adjudicated
                    else "",
                )
        previous_best = self._best_eval_comp_score
        promotion_decision = "skipped"
        promotion_generation = ""
        promotion_reason = gate_reason
        if not eligible:
            logger.info(
                "[PROMOTION] Not eligible (%s); composite score %.4f not promoted%s",
                gate_reason,
                comp_score,
                "" if previous_best is None else f" (best {previous_best:.4f})",
            )
            self._event("PROMOTION_SKIPPED", f"{gate_reason}; comp_score={comp_score:.4f}")
        elif previous_best is None or comp_score > previous_best:
            promoted, promotion_generation, promote_detail = self._promote_best_eval(
                Path(ckpt_path),
                ckpt_name=ckpt_name,
                comp_score=comp_score,
                match_res=match_res,
                solve_rate=solve_rate,
                gate_reason=gate_reason,
            )
            if promoted:
                promotion_decision = "promoted"
            else:
                promotion_decision = "failed"
                promotion_reason = promote_detail
                self._event("PROMOTION_FAILED", promote_detail)
        else:
            promotion_decision = "no_improvement"
            logger.info(
                "[PROMOTION] No improvement (composite %.4f <= best %.4f)",
                comp_score,
                previous_best,
            )

        # The eval row is written only after the promotion decision so the
        # decision and its reason are durable evidence in the same record as the
        # measurements (audit CKP-15: the promotion path previously left no
        # trace on disk).
        self._append_eval_row(
            row + [promotion_decision, promotion_reason, str(promotion_generation or "")]
        )

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

        return {
            "puzzle": puzzle_res,
            "match": match_res,
            "checkpoint": ckpt_name,
            "promotion": promotion_decision,
        }

    def _promote_best_eval(
        self,
        candidate_ckpt: Path,
        *,
        ckpt_name: str,
        comp_score: float,
        match_res: dict,
        solve_rate: float,
        gate_reason: str,
    ) -> "tuple[bool, str, str]":
        """Atomically promote ``candidate_ckpt`` to ``best_eval.pt`` (finding CKP-03).

        Requirements implemented here (audit roadmap P1-2):

        1. the source checkpoint is hashed up front so the promotion record and
           the promoted bytes are provably the same artifact;
        2. the candidate is copied to a temporary file in the same directory,
           flushed and fsynced, then hash-verified before ``os.replace`` makes it
           visible (the previous ``best_eval.pt`` survives until then);
        3. the containing directory is fsynced after the rename;
        4. ``best_eval.json`` is written only after the artifact is in place and
           carries the same ``generation`` id and ``checkpoint_sha256``, so a
           torn promotion can never pair a checkpoint with unrelated evaluation
           results;
        5. failure is reported to the caller (and the run's event stream) instead
           of being a silent ``logger.warning``.

        Returns ``(promoted, generation, detail)``.
        """
        import uuid

        from python.common.artifacts import assert_safe_artifact_destination, artifact_lock
        from python.model.checkpoint_registry import compute_sha256

        best_eval_path = self.checkpoint_dir / "best_eval.pt"
        ticket = getattr(self, "artifact_ticket", None)
        try:
            src = Path(candidate_ckpt)
            if not src.is_file():
                return False, "", f"candidate checkpoint missing: {src}"
            candidate_sha = compute_sha256(src)
            best_eval_path = assert_safe_artifact_destination(
                best_eval_path, ticket=ticket, what="best_eval promotion target"
            )
            generation = f"promo-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            tmp_path = best_eval_path.with_name(best_eval_path.name + ".tmp")
            with artifact_lock(best_eval_path, timeout=0.0, owner="phase4-promotion"):
                with open(src, "rb") as fsrc, open(tmp_path, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                    fdst.flush()
                    os.fsync(fdst.fileno())
                tmp_sha = compute_sha256(tmp_path)
                if tmp_sha != candidate_sha:
                    tmp_path.unlink(missing_ok=True)
                    return (
                        False,
                        "",
                        f"promotion copy verification failed: source {candidate_sha[:16]}... "
                        f"!= temp {tmp_sha[:16]}...",
                    )
                os.replace(tmp_path, best_eval_path)
                try:
                    dir_fd = os.open(str(best_eval_path.parent), os.O_RDONLY)
                except OSError:
                    dir_fd = None
                if dir_fd is not None:
                    try:
                        os.fsync(dir_fd)
                    except OSError:
                        logger.debug("best_eval dir fsync failed", exc_info=True)
                    finally:
                        os.close(dir_fd)
                # Metadata is written only after the artifact is safely in place,
                # and carries the same generation id + hash.
                self._best_eval_comp_score = comp_score
                self._save_best_eval_state(
                    comp_score=comp_score,
                    match_score=float(match_res["engine_score"]),
                    puzzle_rate=float(solve_rate),
                    step=self.trainer.step_num,
                    checkpoint=ckpt_name,
                    checkpoint_sha256=candidate_sha,
                    generation=generation,
                    reason=gate_reason,
                )
            logger.info(
                "[PROMOTION] best_eval.pt updated atomically (score=%.3f, %s, gen=%s, "
                "sha256=%s): %s",
                comp_score,
                gate_reason,
                generation,
                candidate_sha[:16],
                best_eval_path,
            )
            self._event(
                "PROMOTION",
                f"best_eval.pt updated (score={comp_score:.4f}, gen={generation}, "
                f"sha256={candidate_sha[:16]})",
            )
            return True, generation, f"promoted {candidate_sha[:16]}..."
        except Exception as exc:  # promotion failure must never crash the run
            # A failure we caught here proves the temp file is incomplete or
            # unverified, so it is removed (a hard crash still leaves it as
            # evidence, and cleanup.protect keeps it from being pruned blindly).
            try:
                (self.checkpoint_dir / "best_eval.pt.tmp").unlink(missing_ok=True)
            except OSError:
                logger.debug("could not remove failed promotion temp", exc_info=True)
            logger.error("Failed to promote best_eval.pt: %s", exc, exc_info=True)
            return False, "", f"promotion failed: {type(exc).__name__}: {exc}"

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
                f"(LOG-ONLY; measured on games where resignation was disabled for calibration)",
            )
            logger.warning(
                f"SAFETY TRIGGER (LOG-ONLY): False-resign rate exceeded 5% (currently {false_resign_rate * 100:.1f}% "
                f"over {self.metrics_tracker.total_would_have_resigned} calibration games — resignation "
                f"runs disabled in {int(round(getattr(self, 'disable_resign_prob', 0.15) * 100))}% "
                "of games for exactly this measurement)."
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

    def _event(self, tag: str, *messages: str) -> None:
        message = " | ".join(str(m) for m in messages if m)
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
            (
                "RESIGN",
                f"on t={self.worker.resign_threshold} k={self.worker.consecutive_resign_moves} "
                f"ply>={self.worker.min_resign_ply}; "
                f"{int(round((1 - self.disable_resign_prob) * 100))}% of games resign, "
                f"{int(round(self.disable_resign_prob * 100))}% run disabled for "
                "false-resign calibration (RL-02)",
            ),
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
        self._write_run_manifest(resuming_from=getattr(self, "_resumed_from", None))
        self._print_startup_banner()
        self._event(
            "START",
            f"run_id={self.run_id} buffer_id={getattr(self.buffer, 'buffer_id', '?')}",
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
                self._event(
                    "EVAL_START",
                    # Audit RL-27: report the model version actually being
                    # evaluated (v0 was only correct for a random-init run).
                    f"baseline Evaluation Cycle 0 (Model v{self.server.model_version})",
                )
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

                # Pacing: Avoid runaway training steps that starve self-play workers.
                # The gate compares THIS RUN's RL steps (trainer.step_num, reset to 0
                # on warm start) against buffer production of THIS RUN — an inherited
                # supervised counter must never hold the gate closed (PR-03: the Oct 26
                # run idled ~100 minutes before its first step on exactly this bug).
                target_max_step = pacing_target_step(
                    self.buffer.total_added, self.trainer.batch_size, self.max_reuse_ratio
                )
                if self.trainer.step_num > target_max_step:
                    self._maybe_status()
                    time.sleep(0.02)  # Give self-play game threads CPU/GPU time to advance
                    continue

                # RL Training Step (busy-time measured for the duty fraction)
                busy_t0 = time.perf_counter()
                try:
                    train_metrics = self.trainer.train_step()
                    self._consecutive_train_failures = 0
                except Exception as exc:
                    self._consecutive_train_failures += 1
                    logger.error(
                        f"[TRAIN ERROR] step failed ({self._consecutive_train_failures}/{self._max_consecutive_train_failures}): {exc}",
                        exc_info=True,
                    )
                    self._event("TRAIN_STEP_FAILED", f"error={exc} consecutive={self._consecutive_train_failures}")
                    if self._consecutive_train_failures >= self._max_consecutive_train_failures:
                        raise RuntimeError(
                            f"Phase 4 trainer encountered {self._consecutive_train_failures} consecutive train_step failures; stopping: {exc}"
                        ) from exc
                    time.sleep(0.5)
                    continue

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

                # Periodic Telemetry Flush every 50 steps (or every 5 steps in verify mode) or on step 1 or upon reaching max_steps
                if (
                    self._steps_this_run == 1
                    or step % 50 == 0
                    or (self.is_verify_mode and step % 5 == 0)
                    or (self.max_steps is not None and self._steps_this_run >= self.max_steps)
                ):
                    self.metrics_tracker.log_step(
                        step=step,
                        buffer_size=len(self.buffer),
                        model_version=self.server.model_version,
                        buffer_total_added=self.buffer.total_added,
                        buffer_write_head=self.buffer.write_head,
                        buffer_schema_version=getattr(self.buffer, "schema_version", None),
                        games_aborted=getattr(self.worker, "aborted_games", 0)
                        + self._aborted_games_seen,
                        games_invalid=getattr(self.worker, "invalid_games", 0)
                        + self._invalid_games_seen,
                        records_rejected=getattr(self.buffer, "rejected_records", 0)
                        + self._records_rejected,
                        checkpoint_file=self._last_saved_ckpt_name or None,
                        sampling={
                            "sample_source": train_metrics.get("sample_source"),
                            "sampled_selfplay_rows": train_metrics.get("sampled_selfplay_rows"),
                            "sampled_unique_games": train_metrics.get("sampled_unique_games"),
                            "staleness_fallbacks": train_metrics.get("staleness_fallbacks"),
                        },
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
            self._status_for_checkpoint = "SAFETY_STOP"
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
            self._status_for_checkpoint = "INTERRUPTED"
            self.stop()
            self._print_summary("INTERRUPTED")
            self._exit_reason = "interrupted"
            return 0
        except Exception as e:
            return self._handle_unexpected_error(e)
        else:
            self._status_for_checkpoint = "SUCCESS"
            self.stop()
            self._print_summary("SUCCESS")
            self._exit_reason = "ok"
            return 0
        finally:
            self.stop()

    def _handle_unexpected_error(self, e: Exception) -> int:
        """Clean up after an unhandled exception and report it (always logs the traceback)."""
        logger.exception("Phase 4 run failed: %s", e)
        self._status_for_checkpoint = "FAILED"
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

    def _print_summary(self, status: str) -> None:
        """Exit/SIGINT summary: totals from counters + standard footer."""
        elapsed = time.time() - self.start_time
        games = self.metrics_tracker.total_games
        # Counters are labeled explicitly (PR-04): "steps" alone was ambiguous
        # after a warm start (Run B logged steps=6000 — an inherited supervised
        # counter — and looked like progress).
        fields = [
            ("games", games),
            ("positions", self.buffer.total_added),
            ("rl_steps", self.trainer.step_num),
            ("run_steps", self._steps_this_run),
            ("model_v", self.server.model_version),
            ("evals", len(self.puzzle_solve_history)),
        ]
        if getattr(self.trainer, "inherited_step_num", 0):
            fields.append(("inherited_supervised_steps", self.trainer.inherited_step_num))
        self._emit(status_line(fields))
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
        # Stamp the termination status into the FINAL CHECKPOINT (not only the
        # trainer's in-memory dict) so a structurally valid checkpoint from a
        # failed run can never masquerade as a completed training run. The
        # ledger is advanced to a terminal state from RUN_STATES and the final
        # save below writes it into the artifact (finding CKP-01; the Oct-26 Run
        # B failure mode — the ledger existed but never reached disk).
        try:
            from python.model.checkpoint_schema import (
                RUN_STATE_CANCELLED,
                RUN_STATE_COMPLETED,
                RUN_STATE_FAILED,
                RUN_STATE_INTERRUPTED,
                advance_run_state_inplace,
            )

            status = str(getattr(self, "_status_for_checkpoint", "") or "")
            exit_reason = str(getattr(self, "_exit_reason", "") or "")
            state = {
                "SUCCESS": RUN_STATE_COMPLETED,
                "INTERRUPTED": RUN_STATE_INTERRUPTED,
                "CANCELLED": RUN_STATE_CANCELLED,
                "FAILED": RUN_STATE_FAILED,
                # A safety stop means the run did NOT complete; it is recorded as
                # a failure with the reason, never as success.
                "SAFETY_STOP": RUN_STATE_FAILED,
            }.get(status)
            if state is None:
                # No explicit status (e.g. a zero-step run that hit its duration
                # limit, or an early return): fall back to the recorded exit
                # reason, and treat a truly unknown termination as FAILED — an
                # unobservable outcome is never reported as success.
                state = {
                    "ok": RUN_STATE_COMPLETED,
                    "interrupted": RUN_STATE_INTERRUPTED,
                }.get(exit_reason, RUN_STATE_FAILED)
            reason = None
            if status == "SAFETY_STOP":
                reason = "safety trigger stopped the run"
            elif state == RUN_STATE_FAILED:
                reason = f"terminated with status {status or exit_reason or 'unknown'}"
            advance_run_state_inplace(
                self.trainer.run_metadata,
                state,
                step=int(getattr(self.trainer, "step_num", 0) or 0),
                reason=reason,
            )
            self.trainer.run_metadata["run_status"] = status  # legacy field, kept in sync
            self.trainer.run_metadata["terminated_at_unix"] = time.time()
            logger.info(
                "[RUN-STATUS] final ledger state=%s (cli status %s) run_id=%s",
                self.trainer.run_metadata.get("run_state"),
                status,
                self.trainer.run_metadata.get("run_id"),
            )
        except Exception as exc:  # ledger stamping must never block shutdown
            logger.error("Failed to stamp run status into the final checkpoint: %s", exc)
        try:
            self.worker.stop()
            final_ckpt = self.trainer.save_checkpoint(
                self.checkpoint_dir / "phase4_final_checkpoint.pt",
                keep_last_n=self.keep_checkpoints,
            )
            self._known_ckpt_names.add(Path(final_ckpt).name)
            # Audit RL-27: record the final checkpoint so the shutdown banner
            # reports the real artifact instead of a permanent 'unknown'.
            self._last_final_ckpt = str(final_ckpt)
            self._event("CHECKPOINT", f"saved {final_ckpt} sha={short_hash(final_ckpt)} (final)")
        except Exception as exc:
            logger.error("Final checkpoint save failed during shutdown: %s", exc)
            self._event("SHUTDOWN", f"final checkpoint failed: {exc}")
        try:
            # F-15: Ensure at least one telemetry row is logged and durably flushed before close
            if getattr(self, "metrics_tracker", None) is not None:
                step_val = int(getattr(self.trainer, "step_num", 0) or 0)
                buf_size = len(self.buffer) if hasattr(self, "buffer") and self.buffer is not None else 0
                buf_total = getattr(self.buffer, "total_added", 0) if hasattr(self, "buffer") and self.buffer is not None else 0
                buf_head = getattr(self.buffer, "write_head", 0) if hasattr(self, "buffer") and self.buffer is not None else 0
                buf_schema = getattr(self.buffer, "schema_version", None) if hasattr(self, "buffer") and self.buffer is not None else None
                m_ver = getattr(self.server, "model_version", 0) if hasattr(self, "server") and self.server is not None else 0
                self.metrics_tracker.flush_final(
                    step=step_val,
                    buffer_size=buf_size,
                    model_version=m_ver,
                    buffer_total_added=buf_total,
                    buffer_write_head=buf_head,
                    buffer_schema_version=buf_schema,
                    games_aborted=getattr(self.worker, "aborted_games", 0) + getattr(self, "_aborted_games_seen", 0),
                    games_invalid=getattr(self.worker, "invalid_games", 0) + getattr(self, "_invalid_games_seen", 0),
                    records_rejected=getattr(self.buffer, "rejected_records", 0) + getattr(self, "_records_rejected", 0) if hasattr(self, "buffer") and self.buffer is not None else 0,
                    checkpoint_file=getattr(self, "_last_saved_ckpt_name", None) or getattr(self, "_last_final_ckpt", None),
                )
                logger.info("[TELEMETRY] flush_final completed (total rows written: %d)", getattr(self.metrics_tracker, "rows_written", 0))
        except Exception as exc:
            logger.error("Telemetry final flush failed during shutdown: %s", exc)
        try:
            self.metrics_tracker.close()
        except Exception as exc:
            logger.error("Metrics close failed during shutdown: %s", exc)
        if hasattr(self, "buffer") and self.buffer is not None:
            try:
                self.buffer.close()
            except Exception as exc:
                logger.error("Buffer close failed during shutdown: %s", exc)
        # Audit S4-06: release the Syzygy tablebase file handles on shutdown.
        # The adjudicator is a lazily-created process-wide singleton (the
        # worker creates it on the first probe); reset_instance() closes it
        # when it exists, drops it, and is a no-op when it was never created.
        # get_instance() must NOT be used here: it would CONSTRUCT a disabled
        # adjudicator at shutdown when none exists.
        try:
            from python.selfplay.tb_adjudicate import TablebaseAdjudicator

            TablebaseAdjudicator.reset_instance()
            logger.info("Tablebase adjudicator released (shutdown).")
        except Exception as exc:  # never let tablebase cleanup block shutdown
            logger.error("Tablebase adjudicator close failed during shutdown: %s", exc)
        # Shutdown diagnostics (audit SR-08): counters and provenance in one
        # durable JSON record, written before the janitor prunes anything.
        try:
            self._write_run_summary(getattr(self, "_status_for_checkpoint", "unknown"))
        except Exception as exc:  # never let reporting break shutdown
            logger.warning("Run summary write failed: %s", exc)
        self._event(
            "SHUTDOWN",
            "telemetry flushed "
            f"(games_completed={getattr(self.metrics_tracker, 'total_games', 0)} "
            f"aborted={getattr(self.worker, 'aborted_games', 0) + self._aborted_games_seen} "
            f"invalid={getattr(self.worker, 'invalid_games', 0) + self._invalid_games_seen} "
            f"records_rejected={getattr(self.buffer, 'rejected_records', 0) + self._records_rejected})",
        )
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


def build_parser() -> argparse.ArgumentParser:
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
        default="default",
        help=(
            "Base warm-start checkpoint path ('default' resolves canonically via "
            "docs/checkpoint_registry.json; 'none' or '' forces random initialization). "
            "Must exist, be readable, match the 12x128 architecture and be an eligible warm start."
        ),
    )
    parser.add_argument(
        "--pipelined",
        action="store_true",
        default=False,
        help=(
            "Enable pipelined dispatch between MCTS and GPU inference (F-01). "
            "Default: False (disabled until a >10% speedup bar is proven)."
        ),
    )
    # Audit RL-37: the shutdown banner has advertised `--resume` for a long
    # time while the standalone parser never defined it — explicit crash
    # recovery had to rely on auto-resume alone. This is the full-resume path
    # (model + SGD optimizer + scheduler + RNG + buffer state), distinct from
    # the weights-only warm start of --base-checkpoint.
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Full resume from an RL checkpoint (model, SGD optimizer, scheduler, RNG and "
            "buffer state). Takes precedence over --base-checkpoint, which is ignored when "
            "this is given. The checkpoint must pass resume eligibility (quarantine, "
            "optimizer family, scheduler geometry, buffer association; see "
            "--allow-buffer-mismatch / --allow-scheduler-geometry-mismatch for the explicit "
            "overrides)."
        ),
    )
    parser.add_argument(
        "--allow-scheduler-geometry-mismatch",
        action="store_true",
        help=(
            "Audit RL-07 override: on full resume, adopt the checkpoint's scheduler geometry "
            "(warmup/total steps) even when it differs from this run's configuration instead "
            "of refusing. Without it a foreign-geometry checkpoint fails closed with an "
            "actionable error."
        ),
    )
    # Audit SR-02 / SLQ-07 (fixed independently on both branches): this option
    # was registered TWICE (duplicate argparse registration -> ArgumentError on
    # every invocation, including --help). Single definition retained below.
    parser.add_argument(
        "--allow-unverified-warm-start",
        action="store_true",
        help=(
            "Explicit operator override to warm-start from a checkpoint that is "
            "not a designated registry entry (logged as an override; use only "
            "with a documented provisional selection). The decision is "
            "logged loudly; without this flag such starts are rejected (F-11/F-12)."
        ),
    )
    parser.add_argument(
        "--allow-buffer-mismatch",
        action="store_true",
        help=(
            "Operator override: allow resuming from a checkpoint whose stored buffer "
            "snapshot contradicts the live replay buffer (different buffer directory, "
            "capacity, generation, or pointer/count arithmetic). The live buffer state "
            "is kept as-is (never rewound) and the override is logged loudly."
        ),
    )
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=1500,
        help=(
            "Limit the eval/promotion opponent with UCI_LimitStrength at this UCI_Elo "
            "(1320-3190). Default 1500 = the calibrated baseline cell; the match score "
            "is otherwise a floor against full-strength Stockfish."
        ),
    )
    parser.add_argument(
        "--stockfish-full-strength",
        action="store_true",
        help="Explicitly disable the UCI_Elo limit (legacy full-strength opponent).",
    )
    parser.add_argument(
        "--promotion-sprt-max-games",
        type=int,
        default=0,
        help=(
            "Bounded SPRT games required before promotion (0 = threshold gate only). "
            "When >0, promotion additionally requires an ACCEPT_H1 decision vs "
            "best_eval.pt (or the warm start) at H0=0/H1=+30."
        ),
    )
    parser.add_argument(
        "--allow-unverified-promotion",
        action="store_true",
        default=False,
        help="Allow provisional promotion when SPRT is disabled (not statistically validated).",
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
        help="MCTS simulations per move during match evaluation (default: 400)",
    )
    parser.add_argument(
        "--promotion-min-games",
        type=int,
        default=100,
        help="Minimum match games before best_eval.pt may be promoted (default: 100)",
    )
    parser.add_argument(
        "--promotion-min-score",
        type=float,
        default=0.55,
        help="Minimum match score before best_eval.pt may be promoted (default: 0.55)",
    )
    parser.add_argument(
        "--data-role",
        choices=("games_wdl", "policy_tactics", "endgame_syzygy", "unified"),
        default=None,
        help="Explicit role for the human replay dataset (default: from dataset metadata).",
    )
    parser.add_argument(
        "--allow-uncertified-data",
        action="store_true",
        help="Consume human/tactic datasets without valid validation receipts "
        "(loud UNCERTIFIED warning; role/label invariants still apply).",
    )
    parser.add_argument(
        "--allow-uncertified-eval",
        action="store_true",
        help="Run with an uncertified evaluation puzzle set (results are NOT certified).",
    )
    parser.add_argument(
        "--no-require-certified-eval",
        dest="require_certified_eval",
        action="store_false",
        default=True,
        help="Do not abort when the evaluation puzzle set is uncertified "
        "(equivalent to --allow-uncertified-eval for the startup gate).",
    )
    return parser


def main() -> None:
    parser = build_parser()
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
            pipelined=args.pipelined,
            resume_checkpoint=args.resume,
            allow_unverified_warm_start=args.allow_unverified_warm_start,
            allow_buffer_mismatch=args.allow_buffer_mismatch,
            allow_scheduler_geometry_mismatch=args.allow_scheduler_geometry_mismatch,
            device=args.device,
            train_device=args.train_device,
            infer_device=args.infer_device,
            watchdog_rss_gb=args.watchdog_rss_gb,
            watchdog_free_ram_gb=args.watchdog_free_ram_gb,
            max_wait_ms=args.max_wait_ms,
            stockfish_nodes=args.stockfish_nodes,
            stockfish_elo=None if args.stockfish_full_strength else args.stockfish_elo,
            promotion_sprt_max_games=args.promotion_sprt_max_games,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            tactical_mix=args.tactical_mix,
            tactical_csv_path=args.tactical_csv_path,
            max_steps=args.max_steps,
            policy_target_temp=args.policy_target_temp,
            entropy_bonus=args.entropy_bonus,
            eval_openings=args.eval_openings,
            eval_sims=args.eval_sims,
            promotion_min_games=args.promotion_min_games,
            promotion_min_score=args.promotion_min_score,
            data_role=args.data_role,
            allow_uncertified_data=args.allow_uncertified_data,
            allow_uncertified_eval=args.allow_uncertified_eval,
            require_certified_eval=args.require_certified_eval,
            allow_unverified_promotion=args.allow_unverified_promotion,
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
            pipelined=args.pipelined,
            resume_checkpoint=args.resume,
            allow_unverified_warm_start=args.allow_unverified_warm_start,
            allow_buffer_mismatch=args.allow_buffer_mismatch,
            allow_scheduler_geometry_mismatch=args.allow_scheduler_geometry_mismatch,
            device=args.device,
            train_device=args.train_device,
            infer_device=args.infer_device,
            watchdog_rss_gb=args.watchdog_rss_gb,
            watchdog_free_ram_gb=args.watchdog_free_ram_gb,
            max_wait_ms=args.max_wait_ms,
            stockfish_nodes=args.stockfish_nodes,
            stockfish_elo=None if args.stockfish_full_strength else args.stockfish_elo,
            promotion_sprt_max_games=args.promotion_sprt_max_games,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            tactical_mix=args.tactical_mix,
            tactical_csv_path=args.tactical_csv_path,
            max_steps=args.max_steps,
            policy_target_temp=args.policy_target_temp,
            entropy_bonus=args.entropy_bonus,
            eval_openings=args.eval_openings,
            eval_sims=args.eval_sims,
            promotion_min_games=args.promotion_min_games,
            promotion_min_score=args.promotion_min_score,
            data_role=args.data_role,
            allow_uncertified_data=args.allow_uncertified_data,
            allow_uncertified_eval=args.allow_uncertified_eval,
            require_certified_eval=args.require_certified_eval,
            allow_unverified_promotion=args.allow_unverified_promotion,
        )
        sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
