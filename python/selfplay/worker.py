"""Self-Play Worker for Concurrent AlphaZero-style Game Generation.

Contract Specifications:
- Each worker runs many concurrent games using C++ search (virtual loss, FPU, NN cache).
- Playout cap randomization (full-search e.g. 800 with p=0.25 vs fast-search e.g. 150 with p=0.75).
- Temperature: sample by visit count for the first 15-30 plies, then greedy (argmax).
- Root Dirichlet noise (alpha=0.3, weight=0.25 for chess; alpha=0.8 for connect4).
- Resignation: root value < -0.9 for consecutive moves; disabled in 10% of games to measure false-resigns.
- Correct draw rules (50-move rule, 3-fold repetition, stalemate, insufficient material).
- Saved position schema:
  planes (19,8,8), visits (4672,), WDL outcome from THAT position's side to move,
  is_full_search (bool), model_version (int), game_id (uint64).
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import zeropawn_cpp
from python.selfplay.client import ShmInferenceClient
from python.selfplay.ipc import SharedMemoryPool
from python.selfplay.tb_adjudicate import TablebaseAdjudicator

logger = logging.getLogger(__name__)

# Track books that have already logged warnings to prevent spamming across concurrent workers
_WARNED_BOOK_PATHS: set[str] = set()


def resolve_opening_book_path(book_path: Optional[Union[str, Path]] = None) -> str:
    """Find the best available opening book file across CWD and REPO_ROOT."""
    if book_path:
        p = Path(book_path)
        if p.is_file():
            return str(p)
        try:
            from python.cli.common import REPO_ROOT

            if (REPO_ROOT / p).is_file():
                return str(REPO_ROOT / p)
        except Exception:
            pass
        return str(book_path)

    candidates = [
        "data/openings/curated_book.json",
        "data/openings/opening_book.bin.json",
    ]
    for cand in candidates:
        if Path(cand).is_file():
            return cand
    try:
        from python.cli.common import REPO_ROOT

        for cand in candidates:
            if (REPO_ROOT / cand).is_file():
                return str(REPO_ROOT / cand)
    except Exception:
        pass
    return "data/openings/curated_book.json"


@dataclass
class SavedPosition:
    """Record of a single position visited during self-play."""

    planes: np.ndarray  # shape: (C, H, W) float32
    visits: np.ndarray  # shape: (num_actions,) float32
    outcome: int  # 0=Win, 1=Draw, 2=Loss (from side-to-move's perspective)
    is_full_search: bool  # bool
    model_version: int  # int
    game_id: int  # uint64


@dataclass
class GameSummary:
    """Telemetry summary of a completed self-play game."""

    game_id: int
    length_plies: int
    result: str  # "1-0", "0-1", "1/2-1/2"
    resigned: bool
    resigned_side_white: Optional[bool]
    would_have_resigned: (
        bool  # True if resign condition was met in a game where resign was disabled
    )
    is_false_resign: bool  # True if side met resign condition but ended up drawing or winning
    duration_sec: float
    total_simulations: int
    model_version: int
    moves: List[str] = None
    termination_reason: str = "unknown"
    adjudicated: bool = False
    cut_ply: Optional[int] = None
    cut_piece_count: Optional[int] = None
    raw_wdl: Optional[int] = None


class WorkerBatchAggregator:
    """Thread-safe batch aggregator within a single worker process.

    Collects leaf evaluation requests from multiple concurrent game threads,
    batches them, and sends them to the InferenceClient.
    """

    def __init__(
        self,
        client: Any,
        max_batch_size: int = 64,
        max_wait_ms: float = 2.0,
        pipelined: bool = False,
    ) -> None:
        self.client = client
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_ms / 1000.0
        self.pipelined = pipelined
        self.lock = threading.Lock()
        self.pending: List[Tuple[np.ndarray, List[Any], threading.Event]] = []
        self.cond = threading.Condition(self.lock)
        self.first_arrival_time: Optional[float] = None
        self.running = True
        self.last_model_version = 0
        self._window_lock = threading.Lock()
        self._window_wait_time_sec = 0.0
        self._window_wait_count = 0
        self.flush_timeout_count = 0
        self.flush_capacity_count = 0
        self.total_gil_eval_sec = 0.0
        self.total_gil_wait_sec = 0.0
        self.total_gil_dispatch_sec = 0.0
        self.total_eval_calls = 0

        # Pipelined dispatcher queues and threads
        if self.pipelined:
            self.gpu_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
            self.delivery_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
            self._gpu_thread: Optional[threading.Thread] = None
            self._delivery_thread: Optional[threading.Thread] = None

    def evaluate_game_state(self, game: Any) -> Tuple[List[float], float]:
        """Evaluator callback conforming to zeropawn::Evaluator."""
        t0 = time.perf_counter()
        # 1. Encode state
        if hasattr(game, "num_planes") and game.num_planes() == 3:
            # Connect4
            state = game.encode_state()
        else:
            # ChessGame
            state = zeropawn_cpp.encode_state(game)

        res_box: List[Any] = [None, None]  # [policy_list, scalar_val]
        evt = threading.Event()

        need_flush = False
        batch_to_dispatch = None
        with self.lock:
            if not self.pending:
                self.first_arrival_time = time.perf_counter()
            self.pending.append((state, res_box, evt))
            need_flush = len(self.pending) >= self.max_batch_size
            if need_flush:
                if self.pipelined:
                    batch_to_dispatch = self.pending
                    self.pending = []
                self.first_arrival_time = None
            elif len(self.pending) == 1:
                # Notify dispatcher so it can compute remaining wait time
                self.cond.notify()
        if need_flush:
            with self._window_lock:
                self.flush_capacity_count += 1
            if self.pipelined and batch_to_dispatch is not None:
                self.gpu_queue.put(batch_to_dispatch)
            else:
                self._flush_locked()

        t_before_wait = time.perf_counter()
        if not evt.wait(timeout=60):
            raise TimeoutError("Timed out waiting for batched inference result")
        t_after_wait = time.perf_counter()
        res0, res1 = res_box[0], res_box[1]
        t_end = time.perf_counter()

        wait_elapsed = t_after_wait - t0
        gil_eval = (t_before_wait - t0) + (t_end - t_after_wait)
        with self._window_lock:
            self._window_wait_time_sec += wait_elapsed
            self._window_wait_count += 1
            self.total_gil_eval_sec += gil_eval
            self.total_gil_wait_sec += t_after_wait - t_before_wait
            self.total_eval_calls += 1
        return res0, res1

    def get_window_queue_wait_ms(self) -> float:
        """Returns and resets average queue wait time in ms since last call."""
        with self._window_lock:
            count = self._window_wait_count
            time_sec = self._window_wait_time_sec
            self._window_wait_count = 0
            self._window_wait_time_sec = 0.0
        if count == 0:
            return 0.0
        return (time_sec / count) * 1000.0

    def _flush_locked(self) -> None:
        """Flush all pending requests to inference client synchronously.

        Takes a copy of the batch under lock, then releases the lock
        before performing the blocking ``evaluate_leaves`` RPC so game
        threads can keep enqueueing. Kept name for API compatibility;
        callers must NOT hold ``self.lock`` when calling.
        """
        t_flush0 = time.perf_counter()
        with self.lock:
            if not self.pending:
                return
            batch = self.pending
            self.pending = []
            self.first_arrival_time = None

        try:
            states = np.stack([item[0] for item in batch], axis=0)
            t_flush_pack = time.perf_counter() - t_flush0
            p_logits, w_logits, version = self.client.evaluate_leaves(states)
            t_unpack0 = time.perf_counter()
        except Exception:
            logger.exception("Batched inference RPC failed; waking %d waiters", len(batch))
            for item in batch:
                item[1][0] = np.full(4672, 1.0 / 4672, dtype=np.float32)
                item[1][1] = 0.0
                item[2].set()
            return
        self.last_model_version = version

        # Check if already softmax probabilities or compute stable softmax
        w_max = np.max(w_logits, axis=-1, keepdims=True)
        w_exp = np.exp(w_logits - w_max)
        w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)

        # Stage-1B prior fix (Python only, C++ untouched): softmax over the
        # full 4672 policy logits. After the C++ mask+renormalize over legal
        # moves this is EXACTLY softmax-over-legal (common denominator
        # cancels), matching training and the UCI/match/puzzle paths. The C++
        # uniform fallback (p_sum <= 1e-8) can no longer fire on real
        # outputs (softmax entries are all positive).
        p_max = np.max(p_logits, axis=-1, keepdims=True)
        p_exp = np.exp(p_logits - p_max)
        p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)

        scalar_vals = w_probs[:, 0] - w_probs[:, 2]

        for i, item in enumerate(batch):
            policy = np.ascontiguousarray(p_probs[i], dtype=np.float32)
            item[1][0] = policy
            item[1][1] = float(scalar_vals[i])
            item[2].set()
        t_flush_unpack = time.perf_counter() - t_unpack0
        with self._window_lock:
            self.total_gil_dispatch_sec += t_flush_pack + t_flush_unpack

    def _gpu_worker_loop(self) -> None:
        """Stage 2: Batched GPU inference worker loop."""
        while self.running:
            try:
                batch = self.gpu_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if batch is None:
                self.gpu_queue.task_done()
                break

            t0 = time.perf_counter()
            try:
                states = np.stack([item[0] for item in batch], axis=0)
                t_pack = time.perf_counter() - t0
                p_logits, w_logits, version = self.client.evaluate_leaves(states)
                self.last_model_version = version
                self.delivery_queue.put((batch, p_logits, w_logits, t_pack))
            except Exception:
                logger.exception("Batched inference failed in GPU worker loop; waking waiters")
                for item in batch:
                    item[1][0] = [1.0 / 4672] * 4672
                    item[1][1] = 0.0
                    item[2].set()
            finally:
                self.gpu_queue.task_done()

    def _delivery_worker_loop(self) -> None:
        """Stage 3: CPU unboxing and result delivery worker loop."""
        while self.running:
            try:
                item_pkg = self.delivery_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item_pkg is None:
                self.delivery_queue.task_done()
                break

            batch, p_logits, w_logits, t_pack = item_pkg
            t_unpack0 = time.perf_counter()
            try:
                w_max = np.max(w_logits, axis=-1, keepdims=True)
                w_exp = np.exp(w_logits - w_max)
                w_probs = w_exp / np.sum(w_exp, axis=-1, keepdims=True)

                # Stage-1B prior fix (pipelined path): see _flush_locked.
                p_max = np.max(p_logits, axis=-1, keepdims=True)
                p_exp = np.exp(p_logits - p_max)
                p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)

                scalar_vals = w_probs[:, 0] - w_probs[:, 2]

                for i, item in enumerate(batch):
                    policy = np.ascontiguousarray(p_probs[i], dtype=np.float32)
                    item[1][0] = policy
                    item[1][1] = float(scalar_vals[i])
                    item[2].set()
            except Exception:
                logger.exception("Unboxing failed in delivery worker loop; waking waiters")
                for item in batch:
                    item[1][0] = np.full(4672, 1.0 / 4672, dtype=np.float32)
                    item[1][1] = 0.0
                    item[2].set()
            finally:
                t_unpack = time.perf_counter() - t_unpack0
                with self._window_lock:
                    self.total_gil_dispatch_sec += t_pack + t_unpack
                self.delivery_queue.task_done()

    def start_pipelined_workers(self) -> List[threading.Thread]:
        """Start Stage 2 (GPU) and Stage 3 (Delivery) background threads."""
        threads: List[threading.Thread] = []
        if self.pipelined:
            self._gpu_thread = threading.Thread(
                target=self._gpu_worker_loop, daemon=True, name="aggregator_gpu"
            )
            self._delivery_thread = threading.Thread(
                target=self._delivery_worker_loop, daemon=True, name="aggregator_delivery"
            )
            self._gpu_thread.start()
            self._delivery_thread.start()
            threads.extend([self._gpu_thread, self._delivery_thread])
        return threads

    def dispatcher_loop(self) -> None:
        """Background thread loop flushing batches on timeout."""
        while self.running:
            batch_to_dispatch = None
            needs_flush = False
            with self.lock:
                while self.running and not self.pending:
                    self.cond.wait(timeout=self.max_wait_sec)

                if not self.running:
                    break

                if self.pending:
                    elapsed = (
                        time.perf_counter() - self.first_arrival_time
                        if self.first_arrival_time is not None
                        else self.max_wait_sec
                    )
                    remaining = self.max_wait_sec - elapsed
                    if remaining > 0 and len(self.pending) < self.max_batch_size:
                        self.cond.wait(timeout=remaining)
                        if not self.running:
                            break

                    if self.pending:
                        needs_flush = True
                        if self.pipelined:
                            batch_to_dispatch = self.pending
                            self.pending = []
                        self.first_arrival_time = None

            if needs_flush:
                with self._window_lock:
                    self.flush_timeout_count += 1
                try:
                    if self.pipelined and batch_to_dispatch is not None:
                        self.gpu_queue.put(batch_to_dispatch)
                    else:
                        self._flush_locked()
                except Exception:
                    logger.exception("Dispatcher flush failed")

    def stop(self) -> None:
        self.running = False
        if self.pipelined:
            try:
                self.gpu_queue.put_nowait(None)
            except Exception:
                pass
            try:
                self.delivery_queue.put_nowait(None)
            except Exception:
                pass
        with self.lock:
            self.first_arrival_time = None
            for item in self.pending:
                item[1][0] = np.full(4672, 1.0 / 4672, dtype=np.float32)
                item[1][1] = 0.0
                item[2].set()
            self.pending.clear()
            self.cond.notify_all()


class SelfPlayGame:
    """Simulates a single full self-play game with MCTS, temperature, and resignation."""

    def __init__(
        self,
        game_id: int,
        aggregator: WorkerBatchAggregator,
        game_type: str = "chess",
        full_search_sims: int = 400,
        fast_search_sims: int = 80,
        full_search_prob: float = 0.25,
        temp_cutoff_range: Tuple[int, int] = (15, 30),
        resign_threshold: float = -0.95,
        consecutive_resign_moves: int = 4,
        disable_resign_prob: float = 0.10,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        c_puct: float = 2.0,
        fpu_reduction: float = 0.2,
        virtual_loss: int = 1,
        max_plies: int = 400,
        ply_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        tablebase_path: Optional[str] = None,
        tablebase_max_pieces: int = 5,
        allow_resignation: Optional[bool] = None,
        min_resign_ply: int = 30,
        on_sims: Optional[Callable[[int], None]] = None,
        seed: Optional[int] = None,
        search_mode: str = "serial",
        use_book: bool = True,
        book_path: Optional[str] = None,
        book_policy: str = "weighted_blend",
        book_max_ply: int = 30,
    ) -> None:
        self.game_id = game_id
        self.aggregator = aggregator
        self.game_type = game_type
        from python.eval.match_runner import check_search_mode

        self.search_mode = check_search_mode(search_mode)
        self.full_search_sims = full_search_sims
        self.fast_search_sims = fast_search_sims
        self.full_search_prob = full_search_prob
        self.seed = seed
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)
            mcts_seed = seed
        else:
            self._rng = random
            self._np_rng = np.random
            mcts_seed = random.randint(1, 2**31 - 1)

        self.temp_cutoff = self._rng.randint(temp_cutoff_range[0], temp_cutoff_range[1])
        self.resign_threshold = resign_threshold
        self.consecutive_resign_moves = consecutive_resign_moves
        self.min_resign_ply = min_resign_ply
        if allow_resignation is not None:
            self.allow_resignation = allow_resignation
        else:
            self.allow_resignation = self._rng.random() >= disable_resign_prob
        self.max_plies = max_plies
        self.ply_hook = ply_hook
        self.tablebase_path = tablebase_path
        self.tablebase_max_pieces = tablebase_max_pieces
        self.on_sims = on_sims
        self.adjudicator = (
            TablebaseAdjudicator.get_instance(tablebase_path, max_pieces=tablebase_max_pieces)
            if tablebase_path
            else None
        )

        # Opening book initialization (similar to uci_engine.py)
        self.use_book = use_book
        self.book_path = resolve_opening_book_path(book_path)
        self.book_policy = book_policy
        self.book_max_ply = book_max_ply
        self.book_plies = 0
        self.opening_book: Optional[Any] = None
        if self.use_book:
            try:
                from python.data.opening_book import OpeningBook

                book_seed = self.seed if self.seed is not None else random.randint(1, 2**31 - 1)
                self.opening_book = OpeningBook(
                    book_path=self.book_path,
                    policy=self.book_policy,
                    max_depth_plies=self.book_max_ply,
                    seed=book_seed,
                    enabled=self.use_book,
                )
                if len(getattr(self.opening_book, "positions", {})) == 0:
                    if self.book_path not in _WARNED_BOOK_PATHS:
                        _WARNED_BOOK_PATHS.add(self.book_path)
                        logger.warning(
                            "Opening book at %s contains 0 positions; self-play starting from startpos",
                            self.book_path,
                        )
            except Exception as e:
                if str(self.book_path) not in _WARNED_BOOK_PATHS:
                    _WARNED_BOOK_PATHS.add(str(self.book_path))
                    logger.warning(
                        "Opening book initialization failed (%s); self-play starting from startpos",
                        e,
                    )
                self.opening_book = None

        # MCTS Config
        self.mcts_config = zeropawn_cpp.MCTSConfig()
        self.mcts_config.c_puct = c_puct
        self.mcts_config.fpu_reduction = fpu_reduction
        self.mcts_config.virtual_loss = virtual_loss
        self.mcts_config.add_noise = True
        self.mcts_config.dirichlet_alpha = dirichlet_alpha
        self.mcts_config.dirichlet_weight = dirichlet_weight
        self.mcts_config.seed = mcts_seed
        self.mcts = zeropawn_cpp.MCTS(self.mcts_config)

        # Game state
        if game_type == "chess":
            self.game = zeropawn_cpp.ChessGame()
        elif game_type == "connect4":
            self.game = zeropawn_cpp.Connect4()
        else:
            raise ValueError(f"Unknown game type: {game_type}")

        # Trajectory buffer
        self.raw_positions: List[Dict[str, Any]] = []
        self.ply = 0
        self.total_simulations = 0
        self.consecutive_low_root_white = 0
        self.consecutive_low_root_black = 0
        self.would_have_resigned = False
        self.would_have_resigned_side_white: Optional[bool] = None
        self.moves_uci: List[str] = []

    def play_game(self) -> Tuple[List[SavedPosition], GameSummary]:
        """Play the full game to completion or resignation."""
        from python.eval.match_runner import require_serial_search

        require_serial_search(self.search_mode, "SelfPlayGame.play_game")
        start_time = time.perf_counter()
        resigned = False
        resigned_side_white = None
        adjudicated = False
        cut_ply: Optional[int] = None
        cut_piece_count: Optional[int] = None
        raw_wdl: Optional[int] = None
        game_result = "1/2-1/2"
        white_won: Optional[bool] = None

        while not self.game.is_terminal() and self.ply < self.max_plies and self.aggregator.running:
            is_white = (
                self.game.side_to_move()
                if self.game_type == "chess"
                else (self.game.current_player() == 1)
            )

            # Opening book lookup (before MCTS search; similar to uci_engine.py)
            # When ply < max_depth_plies and the book has a legal move, select it directly.
            # lookup() returns (uci_str, meta_dict) — NOT (chess.Move, meta).
            # We do NOT use get_move() (which returns chess.Move); instead we extract
            # the UCI string directly from the tuple and apply it.
            if (
                self.opening_book is not None
                and self.opening_book.enabled
                and self.ply < self.opening_book.max_depth_plies
            ):
                # Pass the exact ply: the C++ ChessGame FEN cannot reproduce the
                # move history, so deriving the ply from the position alone would
                # misreport depth for mid-game starts.
                book_res = self.opening_book.lookup(self.game, current_ply=self.ply)
                if book_res is not None:
                    book_uci_str, book_meta = book_res  # (uci_str, meta_dict)
                    # Verify the book move is still legal in the C++ game state
                    if self.game.is_legal(book_uci_str):
                        book_action = self.game.string_to_action(book_uci_str)
                        self.moves_uci.append(book_uci_str)
                        self.game.apply_action(book_action)
                        self.ply += 1
                        self.book_plies += 1

                        # Clean interaction with tb_adjudicate (book does not interfere)
                        if (
                            self.adjudicator is not None
                            and self.game_type == "chess"
                            and not self.game.is_terminal()
                        ):
                            fen = self.game.get_fen()
                            probe_res, probe_meta = self.adjudicator.probe(fen)
                            if probe_res is not None:
                                adjudicated = True
                                cut_ply = self.ply
                                cut_piece_count = probe_meta.get("piece_count")
                                raw_wdl = probe_meta.get("raw_wdl")
                                stm_white = self.game.side_to_move()
                                if probe_res == 1:
                                    white_won = True if stm_white else False
                                    game_result = "1-0" if stm_white else "0-1"
                                elif probe_res == -1:
                                    white_won = False if stm_white else True
                                    game_result = "0-1" if stm_white else "1-0"
                                else:
                                    white_won = None
                                    game_result = "1/2-1/2"
                                termination_reason = "tablebase_adjudication"

                        # Bound MCTS cache growth (same as normal path)
                        if hasattr(self.mcts, "cache_size") and hasattr(self.mcts, "clear"):
                            try:
                                if self.mcts.cache_size() > 2048:
                                    self.mcts.clear()
                            except Exception:
                                pass

                        if adjudicated:
                            break
                        # Skip MCTS / temperature / recording for this ply; continue loop
                        continue

            # 1. State encoding
            if self.game_type == "connect4":
                planes = self.game.encode_state()
            else:
                planes = zeropawn_cpp.encode_state(self.game)

            # 2. Playout cap randomization
            is_full_search = self._rng.random() < self.full_search_prob
            num_sims = self.full_search_sims if is_full_search else self.fast_search_sims
            self.total_simulations += num_sims
            if self.on_sims is not None:
                self.on_sims(num_sims)

            # 3. MCTS Search (respect configured search_mode)
            if self.search_mode == "batched" and hasattr(self.mcts, "search_batched"):
                search_result = self.mcts.search_batched(
                    self.game, num_sims, self.aggregator.evaluate_game_state
                )
            else:
                search_result = self.mcts.search(
                    self.game, num_sims, self.aggregator.evaluate_game_state
                )
            visit_distribution = np.array(search_result.action_visits, dtype=np.float32)
            root_value = search_result.root_value  # [-1, 1] from side-to-move's perspective
            # 4. Check resignation
            if root_value < self.resign_threshold:
                if is_white:
                    self.consecutive_low_root_white += 1
                else:
                    self.consecutive_low_root_black += 1
            else:
                if is_white:
                    self.consecutive_low_root_white = 0
                else:
                    self.consecutive_low_root_black = 0

            consecutive_low = (
                self.consecutive_low_root_white if is_white else self.consecutive_low_root_black
            )

            if consecutive_low >= self.consecutive_resign_moves and self.ply >= self.min_resign_ply:
                if self.allow_resignation:
                    resigned = True
                    resigned_side_white = is_white
                    white_won = not is_white
                    game_result = "0-1" if is_white else "1-0"
                    break
                else:
                    # Resignation disabled; record that resign threshold was reached
                    if not self.would_have_resigned:
                        self.would_have_resigned = True
                        self.would_have_resigned_side_white = is_white

            # 5. Move selection via Temperature
            legal_actions = self.game.legal_actions()
            if not legal_actions:
                break

            # Temperature exploration plies apply to searched moves (not burned by book moves)
            if (self.ply - self.book_plies) < self.temp_cutoff:
                # Sample proportionally to visits
                legal_probs = visit_distribution[legal_actions]
                sum_p = legal_probs.sum()
                if sum_p > 1e-6:
                    legal_probs = legal_probs / sum_p
                    chosen_action = int(self._np_rng.choice(legal_actions, p=legal_probs))
                else:
                    chosen_action = search_result.best_action
            else:
                # Greedy: argmax visits
                chosen_action = search_result.best_action

            # 6. Apply action and record position
            current_version = self.aggregator.last_model_version
            self.raw_positions.append(
                {
                    "planes": planes,
                    "visits": visit_distribution,
                    "is_white": is_white,
                    "is_full_search": is_full_search,
                    "model_version": current_version,
                }
            )
            uci_str = self.game.action_to_string(chosen_action)
            self.moves_uci.append(uci_str)

            if self.ply_hook is not None:
                self.ply_hook(
                    {
                        "game_id": self.game_id,
                        "ply": self.ply,
                        "side_to_move": "white" if is_white else "black",
                        "root_value": float(root_value),
                        "search_type": "full" if is_full_search else "fast",
                        "num_sims": int(num_sims),
                        "uci_move": uci_str,
                    }
                )

            self.game.apply_action(chosen_action)
            self.ply += 1

            # Bound C++ MCTS cache growth during long games
            if hasattr(self.mcts, "cache_size") and hasattr(self.mcts, "clear"):
                try:
                    if self.mcts.cache_size() > 2048:
                        self.mcts.clear()
                except Exception:
                    pass

            # Tablebase adjudication check (after move applied)
            if (
                self.adjudicator is not None
                and self.game_type == "chess"
                and not self.game.is_terminal()
            ):
                fen = self.game.get_fen()
                probe_res, probe_meta = self.adjudicator.probe(fen)
                if probe_res is not None:
                    adjudicated = True
                    cut_ply = self.ply
                    cut_piece_count = probe_meta.get("piece_count")
                    raw_wdl = probe_meta.get("raw_wdl")
                    stm_white = self.game.side_to_move()
                    if probe_res == 1:
                        white_won = True if stm_white else False
                        game_result = "1-0" if stm_white else "0-1"
                    elif probe_res == -1:
                        white_won = False if stm_white else True
                        game_result = "0-1" if stm_white else "1-0"
                    else:
                        white_won = None
                        game_result = "1/2-1/2"
                    termination_reason = "tablebase_adjudication"
                    break

        # Determine terminal result if not resigned and not adjudicated
        if resigned:
            termination_reason = "resignation"
        elif adjudicated:
            termination_reason = "tablebase_adjudication"
        else:
            termination_reason = "unknown"
            if self.game_type == "chess":
                if self.game.is_checkmate():
                    # Checkmate: side that just moved delivered checkmate
                    # If current side to move is White, Black just checkmated White (Black won)
                    side_to_move = self.game.side_to_move()
                    white_won = not side_to_move
                    game_result = "0-1" if side_to_move else "1-0"
                    termination_reason = "checkmate"
                elif self.ply >= self.max_plies:
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "max_plies_cap"
                elif self.game.is_stalemate():
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "stalemate"
                elif self.game.is_50_move_rule():
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "50_move_rule"
                elif self.game.is_repetition(2):
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "repetition"
                elif self.game.is_insufficient_material():
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "insufficient_material"
                else:
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "draw_other"
            else:
                # Connect 4
                w = self.game.winner()
                if w == 1:
                    white_won = True
                    game_result = "1-0"
                    termination_reason = "connect4_win_1"
                elif w == 2:
                    white_won = False
                    game_result = "0-1"
                    termination_reason = "connect4_win_2"
                else:
                    white_won = None
                    game_result = "1/2-1/2"
                    termination_reason = "connect4_draw"

        # Check false-resignation rate
        is_false_resign = False
        if self.would_have_resigned:
            # If the side that would have resigned ended up drawing or winning
            if self.would_have_resigned_side_white:
                # White would have resigned
                if white_won is True or white_won is None:
                    is_false_resign = True
            else:
                # Black would have resigned
                if white_won is False or white_won is None:
                    is_false_resign = True

        # Assign outcome to each position from THAT position's side-to-move perspective
        # 0 = Win, 1 = Draw, 2 = Loss
        saved_positions: List[SavedPosition] = []
        for raw in self.raw_positions:
            pos_is_white = raw["is_white"]
            if white_won is None:
                outcome = 1  # Draw
            elif white_won == pos_is_white:
                outcome = 0  # Win for side to move
            else:
                outcome = 2  # Loss for side to move

            saved_positions.append(
                SavedPosition(
                    planes=raw["planes"],
                    visits=raw["visits"],
                    outcome=outcome,
                    is_full_search=raw["is_full_search"],
                    model_version=raw["model_version"],
                    game_id=self.game_id,
                )
            )

        duration = time.perf_counter() - start_time
        summary = GameSummary(
            game_id=self.game_id,
            length_plies=self.ply,
            result=game_result,
            resigned=resigned,
            resigned_side_white=resigned_side_white,
            would_have_resigned=self.would_have_resigned,
            is_false_resign=is_false_resign,
            duration_sec=duration,
            total_simulations=self.total_simulations,
            model_version=self.aggregator.last_model_version,
            moves=self.moves_uci,
            termination_reason=termination_reason,
            adjudicated=adjudicated,
            cut_ply=cut_ply,
            cut_piece_count=cut_piece_count,
            raw_wdl=raw_wdl,
        )

        return saved_positions, summary

    def close(self) -> None:
        """Release C++ MCTS tree and NN evaluation cache."""
        if hasattr(self, "mcts") and hasattr(self.mcts, "clear"):
            try:
                self.mcts.clear()
            except Exception:
                pass


class SelfPlayWorker:
    """Manages many concurrent games within a worker process."""

    def __init__(
        self,
        worker_id: int,
        client: Any,
        num_concurrent_games: int = 24,
        game_type: str = "chess",
        full_search_sims: int = 400,
        fast_search_sims: int = 80,
        full_search_prob: float = 0.25,
        temp_cutoff_range: Tuple[int, int] = (15, 30),
        resign_threshold: float = -0.95,
        consecutive_resign_moves: int = 4,
        disable_resign_prob: float = 0.10,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        c_puct: float = 2.0,
        fpu_reduction: float = 0.2,
        virtual_loss: int = 1,
        max_plies: int = 400,
        position_sink: Optional[Callable[[List[SavedPosition], GameSummary], None]] = None,
        ply_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        tablebase_path: Optional[str] = None,
        tablebase_max_pieces: int = 5,
        allow_resignation: bool = False,
        min_resign_ply: int = 30,
        max_wait_ms: float = 1.5,
        pipelined: bool = False,
        search_mode: str = "serial",
        seed: Optional[int] = None,
        use_book: bool = True,
        book_path: Optional[str] = None,
        book_policy: str = "weighted_blend",
        book_max_ply: int = 30,
    ) -> None:
        self.search_mode = search_mode
        self.seed = seed
        self.worker_id = worker_id
        self.client = client
        self.max_wait_ms = max_wait_ms
        self.pipelined = pipelined
        from python.eval.match_runner import check_search_mode

        self.search_mode = check_search_mode(search_mode)
        # Task 3(a): Enforce concurrency cap at 24 unless free RAM stays above 3.0 GB
        if num_concurrent_games > 24:
            try:
                import psutil

                free_ram_gb = psutil.virtual_memory().available / (1024.0**3)
                if free_ram_gb < 3.0:
                    logger.warning(
                        "Concurrency cap of 24 enforced (requested %d, free RAM %.2f GB < 3.0 GB)",
                        num_concurrent_games,
                        free_ram_gb,
                    )
                    num_concurrent_games = 24
            except Exception:
                num_concurrent_games = 24
        self.num_concurrent_games = num_concurrent_games
        self.game_type = game_type
        self.full_search_sims = full_search_sims
        self.fast_search_sims = fast_search_sims
        self.full_search_prob = full_search_prob
        self.temp_cutoff_range = temp_cutoff_range
        self.resign_threshold = resign_threshold
        self.consecutive_resign_moves = consecutive_resign_moves
        self.disable_resign_prob = disable_resign_prob
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.virtual_loss = virtual_loss
        self.max_plies = max_plies
        self.position_sink = position_sink
        self.ply_hook = ply_hook
        self.tablebase_path = tablebase_path
        self.tablebase_max_pieces = tablebase_max_pieces
        self.allow_resignation = allow_resignation
        self.min_resign_ply = min_resign_ply
        self.use_book = use_book
        self.book_path = book_path
        self.book_policy = book_policy
        self.book_max_ply = book_max_ply

        self.aggregator = WorkerBatchAggregator(
            client=client,
            max_batch_size=num_concurrent_games,
            max_wait_ms=self.max_wait_ms,
            pipelined=self.pipelined,
        )

        self.running = False
        self.game_counter = self.worker_id * 1_000_000
        self.threads: List[threading.Thread] = []
        self.total_games_played = 0
        self.total_positions_generated = 0
        self.total_simulations = 0
        self.live_simulations = 0
        self.counter_lock = threading.Lock()
        self.active_games: Dict[int, SelfPlayGame] = {}

    def get_live_cache_stats(self) -> Dict[str, Any]:
        """Query real-time C++ NN cache entries across all active games."""
        with self.counter_lock:
            games = list(self.active_games.values())
        cache_sizes = [g.mcts.cache_size() for g in games]
        total_entries = sum(cache_sizes)
        est_bytes = total_entries * 18744
        return {
            "live_games": len(games),
            "total_cache_entries": total_entries,
            "max_game_cache_entries": max(cache_sizes) if cache_sizes else 0,
            "mean_game_cache_entries": float(np.mean(cache_sizes)) if cache_sizes else 0.0,
            "est_cache_mb": est_bytes / (1024.0 * 1024.0),
        }

    def _game_seed(self, game_id: int) -> Optional[int]:
        """Seed for one game.

        Every concurrent game gets its own seed (base seed + game id) so that a seeded
        run produces independent games instead of N copies of the same Dirichlet noise
        and temperature draws, while staying reproducible run-to-run.
        """
        if self.seed is None:
            return None
        return self.seed + game_id

    def _add_sims(self, n: int) -> None:
        with self.counter_lock:
            self.live_simulations += n

    def _game_runner_loop(self) -> None:
        """Loop run by each concurrent game thread."""
        thread_id = threading.get_ident()
        while self.running:
            with self.counter_lock:
                self.game_counter += 1
                game_id = self.game_counter
            game = SelfPlayGame(
                game_id=game_id,
                aggregator=self.aggregator,
                game_type=self.game_type,
                full_search_sims=self.full_search_sims,
                fast_search_sims=self.fast_search_sims,
                full_search_prob=self.full_search_prob,
                temp_cutoff_range=self.temp_cutoff_range,
                resign_threshold=self.resign_threshold,
                consecutive_resign_moves=self.consecutive_resign_moves,
                disable_resign_prob=self.disable_resign_prob,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_weight=self.dirichlet_weight,
                c_puct=self.c_puct,
                fpu_reduction=self.fpu_reduction,
                virtual_loss=self.virtual_loss,
                max_plies=self.max_plies,
                ply_hook=self.ply_hook,
                tablebase_path=self.tablebase_path,
                tablebase_max_pieces=self.tablebase_max_pieces,
                allow_resignation=self.allow_resignation,
                min_resign_ply=self.min_resign_ply,
                on_sims=self._add_sims,
                search_mode=self.search_mode,
                seed=self._game_seed(game_id),
                use_book=self.use_book,
                book_path=self.book_path,
                book_policy=self.book_policy,
                book_max_ply=self.book_max_ply,
            )

            with self.counter_lock:
                self.active_games[thread_id] = game

            try:
                positions, summary = game.play_game()
            except TimeoutError:
                if not self.running:
                    break
                raise
            finally:
                with self.counter_lock:
                    self.active_games.pop(thread_id, None)
                try:
                    game.close()
                except Exception:
                    pass

            with self.counter_lock:
                self.total_games_played += 1
                self.total_positions_generated += len(positions)
                self.total_simulations += summary.total_simulations

            if self.position_sink is not None:
                self.position_sink(positions, summary)

    def start(self) -> None:
        """Start batch aggregator and all concurrent game threads."""
        from python.eval.match_runner import require_serial_search

        require_serial_search(self.search_mode, "SelfPlayWorker.start")
        logger.info(
            "SelfPlayWorker %d starting: %d games [search_mode=%s]",
            self.worker_id,
            self.num_concurrent_games,
            self.search_mode,
        )
        self.running = True
        if self.aggregator.pipelined:
            self.threads.extend(self.aggregator.start_pipelined_workers())

        disp_thread = threading.Thread(target=self.aggregator.dispatcher_loop, daemon=True)
        disp_thread.start()
        self.threads.append(disp_thread)

        for _ in range(self.num_concurrent_games):
            t = threading.Thread(target=self._game_runner_loop, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        """Signal all threads to stop."""
        self.running = False
        self.aggregator.stop()
        for t in self.threads:
            t.join(timeout=1.0)
        self.threads.clear()


def run_worker_process(
    worker_id: int,
    shm_prefix: str,
    capacity: int,
    request_queue: Any,
    completion_event: Any,
    slot_range: range,
    position_queue: Any,
    stop_event: Any,
    num_concurrent_games: int = 24,
    game_type: str = "chess",
    full_search_sims: int = 800,
    fast_search_sims: int = 150,
    full_search_prob: float = 0.25,
) -> None:
    """Multiprocessing entry point for a self-play worker process."""
    shm_pool = SharedMemoryPool(name_prefix=shm_prefix, capacity=capacity, create=False)
    client = ShmInferenceClient(
        worker_id=worker_id,
        shm_pool=shm_pool,
        request_queue=request_queue,
        completion_event=completion_event,
        slot_range=slot_range,
    )

    def sink(positions: List[SavedPosition], summary: GameSummary):
        position_queue.put((positions, summary))

    worker = SelfPlayWorker(
        worker_id=worker_id,
        client=client,
        num_concurrent_games=num_concurrent_games,
        game_type=game_type,
        full_search_sims=full_search_sims,
        fast_search_sims=fast_search_sims,
        full_search_prob=full_search_prob,
        position_sink=sink,
    )

    worker.start()
    while not stop_event.is_set():
        time.sleep(0.1)

    worker.stop()
    shm_pool.close()
