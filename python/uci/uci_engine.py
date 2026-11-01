"""Universal Chess Interface (UCI) Protocol Engine Implementation.

Provides standard UCI protocol compliance for ZeroPawn to play against chess GUIs
such as Arena, Lucas Chess, cutechess-cli, and standard chess tools.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from python.cli.common import (
    get_default_checkpoint_path,
    load_engine_model,
    resolve_device,
    setup_environment,
)
from python.model.checkpoint_registry import DefaultCheckpointError
from python.model.chess_net import ChessNet
from python.uci.time_management import SearchTimeManager, ThroughputKey, allocate_clock_ms

logger = logging.getLogger(__name__)


class UCIEngine:
    """Implements Universal Chess Interface (UCI) protocol for ZeroPawn."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        default_sims: int = 400,
        c_puct: float = 2.0,
        search_mode: str = "serial",
        batch_size: Union[int, str] = 1,
        use_book: bool = True,
        book_path: Optional[Union[str, Path]] = None,
        book_policy: str = "weighted_blend",
        book_max_ply: int = 30,
        fpu_reduction: float = 0.2,
        reuse_tree: bool = False,
    ) -> None:
        setup_environment()
        import zeropawn_cpp
        from python.eval.match_runner import check_search_mode, require_serial_search
        from python.hardware.profile import resolve_batch_size

        self.zeropawn_cpp = zeropawn_cpp
        self.device = resolve_device(device)
        if checkpoint_path:
            # "" is the explicit "no model" signal; any non-empty path is the
            # operator's explicit choice (provenance warnings are emitted by
            # load_engine_model).  None means "use the designated default".
            self.checkpoint_path = checkpoint_path
        else:
            try:
                self.checkpoint_path = get_default_checkpoint_path()
            except DefaultCheckpointError as e:
                # UCI stays usable in protocol/introspection mode without a
                # model (uniform-policy evaluator), but the reason must be
                # visible in the log — never a silent weak-checkpoint swap.
                logger.warning(
                    "No default checkpoint available; UCI engine starts WITHOUT a "
                    "model (uniform-policy evaluator). Pass an explicit checkpoint "
                    "path or restore/designate the registry default. Detail: %s",
                    e,
                )
                self.checkpoint_path = None
        self.default_sims = default_sims
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.reuse_tree = reuse_tree
        self.search_mode = check_search_mode(search_mode)
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            self.batch_size = resolve_batch_size("auto", workload="uci", device=self.device)
        else:
            self.batch_size = max(1, int(batch_size))
        self._require_serial_search = require_serial_search
        self.use_book = use_book
        self.book_path = str(book_path) if book_path else "data/openings/curated_book.json"
        self.book_policy = book_policy
        self.book_max_ply = book_max_ply
        self.opening_book: Optional[Any] = None
        self._init_opening_book()
        logger.info(
            "UCIEngine search_mode=%s (batch_size=%d), use_book=%s, reuse_tree=%s",
            self.search_mode,
            self.batch_size,
            self.use_book,
            self.reuse_tree,
        )
        self.model: Optional[ChessNet] = None
        self.model_info: dict[str, Any] = {}

        # MCTS Config
        self.mcts_config = self.zeropawn_cpp.MCTSConfig()
        self.mcts_config.c_puct = self.c_puct
        self.mcts_config.fpu_reduction = self.fpu_reduction
        self.mcts_config.reuse_tree = self.reuse_tree
        self.mcts_config.batch_size = self.batch_size
        self.mcts_config.virtual_loss = 1
        self.mcts_config.add_noise = False  # Deterministic / no root noise for play mode per PRD
        # Long interactive sessions accumulate positions; LRU-bound the cache
        # (~16k entries ~= 300 MB) instead of the 100k default (~1.9 GB).
        self.mcts_config.nn_cache_cap = 16384
        self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)

        # Game state and subtree reuse tracking
        self.game = self.zeropawn_cpp.ChessGame()
        self._lock = threading.Lock()
        # Mode-aware time management (audit H-3): throughput observations are
        # EWMA'd per (device, search_mode, batch_size); cold start uses a
        # conservative per-configuration default instead of a hardcoded rate.
        self.time_manager = SearchTimeManager()
        self._last_search_ran = False
        self.last_engine_action: Optional[int] = None
        self.last_engine_move: Optional[str] = None
        self.prev_moves_at_search: Optional[List[str]] = None
        self.prev_moves: List[str] = []
        # Ply of the position BEFORE any replayed "moves" suffix, derived from
        # the validated FEN metadata (remediation CR-07): startpos is ply 0,
        # a FEN at fullmove N with black to move sits at (N-1)*2 + 1 plies.
        # history_len() alone under-counts for FEN-loaded games (it only
        # counts moves replayed inside this UCI session), which let book
        # moves be offered beyond max_depth_plies.
        self.prev_base_ply: int = 0
        self.prev_fen: str = "startpos"

        # Real-time search control (F-3): stop event + abort deadline shared
        # with the in-search abort predicate, a rolling simulation-rate
        # estimate used for clock allocation, and the running search thread.
        self._stop_event = threading.Event()
        self._search_thread: Optional[threading.Thread] = None
        self._search_state_lock = threading.Lock()
        # Conservative initial sims/s estimate; measured serial rate on this
        # hardware is 167-250 sims/s (docs/perf/BATCH_SWEEP_EXTENDED.json).
        self._sims_per_sec: float = 200.0
        # Search-stall safety (remediation F-03): while a search thread is
        # alive, the native MCTS/Game objects must not be touched from this
        # thread — the search walks them with the GIL released. If a stop
        # request does not end the search within ``_stall_timeout`` seconds
        # (e.g. an evaluator stuck inside a GIL-releasing native call, where
        # the in-search abort predicate cannot fire until it returns), the
        # engine enters a stalled state and every state-mutating command is
        # refused until the search thread actually finishes.
        self._stall_timeout: float = 10.0
        self._search_stalled: bool = False

        # Load model if checkpoint exists
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self._load_model(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Time management (F-3)
    # ------------------------------------------------------------------

    def allocate_time(
        self, arg_dict: dict, side_is_white: bool
    ) -> Tuple[int, Optional[float], bool]:
        """Convert ``go`` arguments into a (simulation cap, deadline, infinite).

        The simulation cap is a soft budget derived from the rolling measured
        simulation rate; the deadline is an absolute monotonic-clock timestamp
        in **seconds** (``time.perf_counter()`` domain) enforced inside the
        search every ~32 simulations. For ``go nodes`` the simulation cap is
        exact and no deadline is set (stop is still honored). ``go infinite``
        returns a huge cap and no deadline.
        """
        if arg_dict.get("infinite"):
            return (2_000_000_000, None, True)

        if "nodes" in arg_dict:
            try:
                return (max(10, int(arg_dict["nodes"])), None, False)
            except (TypeError, ValueError):
                return (max(10, self.default_sims), None, False)

        # Mode-aware rate (audit H-3): prefer the per-configuration EWMA from
        # SearchTimeManager; engines and tests without one fall back to the
        # rolling _sims_per_sec estimate.
        tm = getattr(self, "time_manager", None)
        tp_key = None
        rate = None
        if tm is not None:
            try:
                tp_key = ThroughputKey.from_engine(
                    self.device, self.search_mode, getattr(self, "batch_size", 1)
                )
                rate = max(1.0, float(tm.estimate_rate(tp_key)))
            except Exception:
                tp_key, rate = None, None
        if rate is None:
            rate = max(1.0, float(self._sims_per_sec))
        now = time.perf_counter()

        if "movetime" in arg_dict:
            try:
                movetime_ms = max(1.0, float(arg_dict["movetime"]))
            except (TypeError, ValueError):
                movetime_ms = 1000.0
            # Reserve a small margin for move selection, info/bestmove output,
            # and UCI write latency.
            margin_ms = min(50.0, movetime_ms * 0.05)
            budget_ms = max(1.0, movetime_ms - margin_ms)
            if tm is not None and tp_key is not None:
                # SearchTimeManager applies its own movetime margin (0.9).
                sims = max(10, min(10000, tm.sims_for_budget(tp_key, movetime_ms, "movetime")))
            else:
                sims = max(10, int((budget_ms / 1000.0) * rate))
            return (sims, now + budget_ms / 1000.0, False)

        # Clock play: wtime/btime with optional increment. Keep the historical
        # allocation policy (wtime/30 + inc, clamped) as the per-move budget —
        # what changed is that it is now enforced against the real clock.
        time_key = "wtime" if side_is_white else "btime"
        inc_key = "winc" if side_is_white else "binc"
        if time_key in arg_dict:
            try:
                total_time_ms = max(0.0, float(arg_dict[time_key]))
                inc_ms = max(0.0, float(arg_dict.get(inc_key, 0.0)))
            except (TypeError, ValueError):
                return (max(10, self.default_sims), None, False)
            mtg = None
            if "movestogo" in arg_dict:
                try:
                    mtg = max(1, int(float(arg_dict["movestogo"])))
                except (TypeError, ValueError):
                    mtg = None
            if tm is not None and tp_key is not None:
                # allocate_clock_ms preserves the established /30 + inc policy
                # with the /40 movestogo clamp; SearchTimeManager applies the
                # soft 0.8 clock margin.
                allocated_ms = allocate_clock_ms(total_time_ms, inc_ms, mtg)
                sims = max(10, min(10000, tm.sims_for_budget(tp_key, allocated_ms, "clock")))
            else:
                allocated_ms = (total_time_ms / 30.0) + (inc_ms * 0.9)
                allocated_ms = max(100.0, min(allocated_ms, total_time_ms * 0.4, 10_000.0))
                margin_ms = min(100.0, allocated_ms * 0.05)
                allocated_ms = max(1.0, allocated_ms - margin_ms)
                sims = max(20, int((allocated_ms / 1000.0) * rate))
            return (sims, now + allocated_ms / 1000.0, False)

        # No clock information: fixed simulation budget (legacy behavior).
        return (max(10, self.default_sims), None, False)

    def _make_abort_fn(self, deadline_ms: Optional[float]):
        """Build the in-search abort predicate.

        Called by the C++ search approximately every 32 simulations. Returns
        True when a stop was requested or the wall-clock deadline passed.
        """
        stop_event = self._stop_event

        def abort() -> bool:
            if stop_event.is_set():
                return True
            if deadline_ms is not None and time.perf_counter() >= deadline_ms:
                return True
            return False

        return abort

    def _update_rate_estimate(self, sims: int, elapsed_sec: float) -> None:
        """EWMA-update the rolling sims/s estimate used for clock allocation."""
        if elapsed_sec <= 0.0 or sims <= 0:
            return
        measured = sims / elapsed_sec
        # Guard against absurd one-off spikes (e.g., an aborted 1-sim search).
        if measured <= 0 or measured > 100_000:
            return
        self._sims_per_sec = 0.7 * self._sims_per_sec + 0.3 * measured

    def _wait_for_running_search(self, timeout: Optional[float] = None) -> bool:
        """Block until any in-flight search finishes (state-mutation safety).

        Returns **True** when it is safe to mutate ``self.mcts``/``self.game``.
        If the search thread is still alive after ``timeout`` seconds (default
        ``self._stall_timeout``) despite a stop request, the engine is marked
        **stalled** and False is returned. Remediation F-03: the previous
        implementation logged "proceeding unsafely" and returned anyway, which
        let the main thread mutate the native MCTS/Game objects while the
        search thread was still walking them with the GIL released
        (use-after-free window). While stalled, every state-mutating command is
        refused; the stalled state clears automatically once the search thread
        finishes, because a finished thread short-circuits the wait below.
        """
        if timeout is None:
            timeout = self._stall_timeout
        thread = self._search_thread
        if thread is None or not thread.is_alive():
            self._search_stalled = False
            return True
        logger.warning(
            "Command arrived during active search; requesting stop and waiting up to %.1fs",
            timeout,
        )
        self._stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            self._search_stalled = True
            logger.error(
                "Search thread did not finish after %.1fs stop request; the engine will "
                "REFUSE state-mutating commands (ucinewgame/position/setoption/go/model "
                "load) until the search completes. Never mutating mcts/game while a "
                "GIL-released search may reference them (remediation F-03).",
                timeout,
            )
            sys.stdout.write(
                "info string engine busy: previous search has not finished; "
                "state-mutating commands are refused until it completes\n"
            )
            sys.stdout.flush()
            return False
        self._search_stalled = False
        return True

    def _load_model(self, path: Union[str, Path]) -> bool:
        try:
            new_model, new_info = load_engine_model(path, device=self.device)
            new_model.eval()
            self.model = new_model
            self.model_info = new_info
            self.checkpoint_path = path
            if hasattr(self, "mcts") and self.mcts is not None:
                self.mcts.clear()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            logger.info("Loaded checkpoint %s on %s", path, self.device)
            return True
        except Exception as e:
            logger.error("Failed to load model from %s: %s", path, e)
            return False

    def _eval_callback(self, game: Any) -> Tuple[Any, float]:
        if self.model is None:
            num_actions = game.num_actions()
            return (np.full(num_actions, 1.0 / num_actions, dtype=np.float32), 0.0)

        # Encode the passed C++ game directly: it is a ChessGame carrying the
        # real move history, so the repetition plane matches training semantics.
        # The former FEN round-trip (ChessGame(game.get_fen())) erased history
        # and pinned plane 18 to zero at play time (audit H-2).
        planes = self.zeropawn_cpp.encode_state(game)
        t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with self._lock:
                if self.device == "cuda":
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = self.model(t)
                else:
                    p_logits, v_wdl = self.model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])

        # Return a float32 ndarray (not a Python list): the native binding has
        # a memcpy fast path for arrays; .tolist() serialization cost the
        # batched play-time path a measured ~2.4x in throughput (audit H-4).
        return (p_probs.astype(np.float32), scalar_val)

    def _batched_eval_callback(self, batch_planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(batch_planes.shape[0])
        if n == 0 or self.model is None:
            return (np.zeros((n, 4672), dtype=np.float32), np.zeros((n,), dtype=np.float32))

        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32))
        t = t.view(n, 19, 8, 8).to(self.device, non_blocking=True)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with self._lock:
                if self.device == "cuda":
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = self.model(t)
                else:
                    p_logits, v_wdl = self.model(t)
            policy = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            wdl = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)

        return (policy.astype(np.float32), values)

    def _eval_batch_callback(self, games: List[Any]) -> List[Tuple[Any, float]]:
        if not games:
            return []
        if self.model is None:
            num_actions = games[0].num_actions()
            uniform_policy = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)
            return [(uniform_policy, 0.0) for _ in games]

        # Encode the passed games directly (audit H-2): they are ChessGame
        # objects carrying real move history — a FEN round-trip would erase it.
        batch_planes = [self.zeropawn_cpp.encode_state(g) for g in games]

        batch_arr = np.stack(batch_planes)
        t = torch.from_numpy(batch_arr).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with self._lock:
                if self.device == "cuda":
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = self.model(t)
                else:
                    p_logits, v_wdl = self.model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            scalar_vals = (v_probs[:, 0] - v_probs[:, 2]).astype(float)

        # Array returns (not .tolist() lists): the binding fast-paths numpy
        # arrays; list serialization cost ~2.4x throughput at B=16 (audit H-4).
        return [(p_probs[i], float(scalar_vals[i])) for i in range(len(games))]

    def _init_opening_book(self) -> None:
        try:
            from python.data.opening_book import OpeningBook

            self.opening_book = OpeningBook(
                book_path=self.book_path,
                policy=self.book_policy,
                max_depth_plies=self.book_max_ply,
                enabled=self.use_book,
            )
        except Exception as e:
            logger.debug("Opening book initialization skipped: %s", e)
            self.opening_book = None

    def search_and_respond(
        self, sims: int, deadline_ms: Optional[float] = None
    ) -> str:
        """Run MCTS search on current game state and output bestmove.

        ``deadline_ms`` is a monotonic-clock timestamp; when set, the search
        aborts at the next ~32-simulation poll after the deadline passes (or
        immediately once :attr:`_stop_event` is set). The best move is chosen
        from whatever tree was built, so a partial search still answers.
        """
        self._require_serial_search(self.search_mode, "UCIEngine.search_and_respond")
        start = time.perf_counter()
        abort_fn = self._make_abort_fn(deadline_ms)

        if self.game.is_game_over():
            self.mcts.reset_root()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()
            return "0000"

        legal_moves = self.game.legal_moves_uci()
        if not legal_moves:
            self.mcts.reset_root()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()
            return "0000"

        # Check opening book first if enabled
        if self.use_book and self.opening_book is not None:
            # True game ply (remediation CR-07): the bare position's own ply
            # (startpos -> 0; a FEN at fullmove N -> (N-1)*2 [+1 for black])
            # plus the moves replayed after it. history_len() alone reports 0
            # for FEN-loaded games and under-counts book depth. Book moves
            # remain legality-gated below either way.
            current_ply = self.prev_base_ply + len(self.prev_moves)
            book_res = self.opening_book.lookup(self.game, current_ply=current_ply)
            if book_res is not None:
                book_move, _ = book_res
                if book_move in legal_moves:
                    self.mcts.reset_root()
                    self.last_engine_action = None
                    self.last_engine_move = None
                    self.prev_moves_at_search = None
                    sys.stdout.write(f"info depth 1 nodes 1 score cp 0 pv {book_move}\n")
                    sys.stdout.write(f"bestmove {book_move}\n")
                    sys.stdout.flush()
                    return book_move

        # Search with MCTS. The abort predicate enforces the wall-clock
        # deadline and stop requests from inside the C++ search loop.
        self._last_search_ran = True
        self.prev_moves_at_search = list(self.prev_moves)
        res = self.mcts.search(
            self.game, sims, self._eval_callback, self._eval_batch_callback, abort_fn
        )
        best_action = res.best_action
        best_move = self.game.action_to_string(best_action)
        root_val = res.root_value
        self.last_engine_action = best_action
        self.last_engine_move = best_move

        actual_sims = int(getattr(res, "new_simulations", sims))
        elapsed = time.perf_counter() - start
        self._update_rate_estimate(actual_sims, elapsed)

        # Score in centipawns (AlphaZero approx: score_cp = int(root_val * 400))
        score_cp = int(root_val * 400)
        sys.stdout.write(f"info depth 10 nodes {actual_sims} score cp {score_cp} pv {best_move}\n")
        sys.stdout.write(f"bestmove {best_move}\n")
        sys.stdout.flush()
        return best_move

    def _search_worker(self, sims: int, deadline_ms: Optional[float]) -> None:
        """Thread body for ``go``: run the search and emit bestmove.

        Stdin stays responsive while this runs, so ``stop`` and ``quit`` are
        processed immediately. Any failure must still emit a bestmove so the
        GUI never waits forever.
        """
        try:
            self.search_and_respond(sims, deadline_ms)
        except Exception as e:
            logger.exception("Search failed during go: %s", e)
            sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()

    def _start_search(self, sims: int, deadline_ms: Optional[float]) -> None:
        # Never overlap two searches: the previous one must have emitted its
        # bestmove before a new tree/state is used. If the previous search is
        # stalled (F-03), refuse to start another one instead of touching the
        # state it still references.
        if not self._wait_for_running_search():
            logger.error("Refusing 'go infinite': previous search has not finished")
            return
        # Fresh stop event per search: a "stop" that arrived before this "go"
        # belonged to the previous search and must not abort this one, while a
        # "stop" arriving after this point is guaranteed to be seen by the
        # in-search abort predicate (the event object is already installed).
        self._stop_event = threading.Event()
        thread = threading.Thread(
            target=self._search_worker, args=(sims, deadline_ms), daemon=True
        )
        self._search_thread = thread
        thread.start()

    def handle_line(self, line: str) -> bool:
        """Handle a single UCI protocol command line. Returns False if 'quit'."""
        line = line.strip()
        if not line:
            return True

        tokens = line.split()
        cmd = tokens[0].lower()

        if cmd == "uci":
            sys.stdout.write("id name ZeroPawn\n")
            sys.stdout.write("id author ZeroPawn Team\n")
            sys.stdout.write("option name Hash type spin default 64 min 1 max 2048\n")
            ckpt_str = str(self.checkpoint_path) if self.checkpoint_path else "none"
            sys.stdout.write(f"option name WeightsFile type string default {ckpt_str}\n")
            sys.stdout.write(
                f"option name Simulations type spin default {self.default_sims} min 10 max 10000\n"
            )
            sys.stdout.write(f"option name CP_UCT type string default {self.c_puct}\n")
            sys.stdout.write(f"option name FPU type string default {self.fpu_reduction}\n")
            sys.stdout.write(
                f"option name ReuseTree type check default {'true' if self.reuse_tree else 'false'}\n"
            )
            sys.stdout.write(f"option name Device type string default {self.device}\n")
            sys.stdout.write(
                f"option name SearchMode type combo default {self.search_mode} var serial var batched\n"
            )
            sys.stdout.write(
                f"option name BatchSize type spin default {self.batch_size} min 1 max 64\n"
            )
            sys.stdout.write(
                f"option name OwnBook type check default {'true' if self.use_book else 'false'}\n"
            )
            sys.stdout.write(f"option name BookFile type string default {self.book_path}\n")
            sys.stdout.write(
                f"option name BookPolicy type combo default {self.book_policy} var frequency var score var weighted_blend var uniform var best\n"
            )
            sys.stdout.write(
                f"option name BookMaxPly type spin default {self.book_max_ply} min 2 max 100\n"
            )
            sys.stdout.write("uciok\n")
            sys.stdout.flush()

        elif cmd == "isready":
            if self.model is None and self.checkpoint_path:
                # Model load clears the MCTS tree; refuse it while a search is
                # still running (F-03). readyok is still emitted so the GUI's
                # handshake proceeds; the engine simply stays without weights
                # until a later isready.
                if self._wait_for_running_search():
                    self._load_model(self.checkpoint_path)
                else:
                    logger.error("Refusing model load during unfinished search (isready)")
            sys.stdout.write("readyok\n")
            sys.stdout.flush()

        elif cmd == "setoption":
            # Several options rebuild the MCTS object or reload weights; those
            # must not race an in-flight search. If the search is stalled
            # (F-03), the whole command is refused.
            if not self._wait_for_running_search():
                logger.error("Refusing 'setoption' while a search has not finished")
                return True
            try:
                name_idx = tokens.index("name") + 1
                val_idx = tokens.index("value") + 1 if "value" in tokens else None
                if val_idx is not None:
                    opt_name = " ".join(tokens[name_idx : val_idx - 1]).lower()
                    opt_val = " ".join(tokens[val_idx:])
                else:
                    opt_name = " ".join(tokens[name_idx:]).lower()
                    opt_val = ""

                if opt_name == "weightsfile":
                    self._load_model(opt_val)
                elif opt_name == "simulations":
                    self.default_sims = int(opt_val)
                elif opt_name in ("cp_uct", "cpuct", "c_puct"):
                    self.c_puct = float(opt_val)
                    self.mcts_config.c_puct = self.c_puct
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                elif opt_name in ("fpu", "fpu_reduction", "fpureduction"):
                    self.fpu_reduction = float(opt_val)
                    self.mcts_config.fpu_reduction = self.fpu_reduction
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                elif opt_name in ("reusetree", "reuse_tree"):
                    self.reuse_tree = opt_val.lower() in ("true", "1", "yes", "on")
                    self.mcts_config.reuse_tree = self.reuse_tree
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                    self.last_engine_action = None
                    self.last_engine_move = None
                    self.prev_moves_at_search = None
                elif opt_name in (
                    "batchsize",
                    "batch_size",
                    "inferencebatchsize",
                    "inference_batch_size",
                ):
                    if opt_val.strip().lower() == "auto":
                        from python.hardware.profile import resolve_batch_size

                        self.batch_size = resolve_batch_size(
                            "auto", workload="uci", device=self.device
                        )
                    else:
                        self.batch_size = max(1, min(64, int(opt_val)))
                    self.mcts_config.batch_size = self.batch_size
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                elif opt_name == "device":
                    self.device = resolve_device(opt_val)
                    if self.checkpoint_path:
                        self._load_model(self.checkpoint_path)
                elif opt_name == "searchmode":
                    from python.eval.match_runner import check_search_mode

                    self.search_mode = check_search_mode(opt_val.lower())
                elif opt_name == "ownbook":
                    self.use_book = opt_val.lower() in ("true", "1", "yes", "on")
                    if self.opening_book:
                        self.opening_book.enabled = self.use_book
                    else:
                        self._init_opening_book()
                elif opt_name == "bookfile":
                    self.book_path = opt_val
                    self._init_opening_book()
                elif opt_name == "bookpolicy":
                    self.book_policy = opt_val.lower()
                    if self.opening_book:
                        self.opening_book.policy = self.book_policy
                elif opt_name == "bookmaxply":
                    try:
                        self.book_max_ply = max(2, min(100, int(opt_val)))
                        if self.opening_book:
                            self.opening_book.max_depth_plies = self.book_max_ply
                        else:
                            self._init_opening_book()
                    except ValueError:
                        logger.debug("Invalid BookMaxPly value: %s", opt_val)
            except Exception as e:
                logger.debug("Failed parsing setoption: %s", e)

        elif cmd == "ucinewgame":
            if not self._wait_for_running_search():
                logger.error("Refusing 'ucinewgame' while a search has not finished")
                return True
            self.mcts.clear()
            self.mcts.reset_root()
            self.game = self.zeropawn_cpp.ChessGame()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            self.prev_moves = []
            self.prev_base_ply = 0
            self.prev_fen = "startpos"

        elif cmd == "position":
            if not self._wait_for_running_search():
                logger.error("Refusing 'position' while a search has not finished")
                return True
            idx = 1
            moves_idx = None
            if "moves" in tokens:
                moves_idx = tokens.index("moves")

            current_fen = "startpos"
            if idx < len(tokens) and tokens[idx] == "startpos":
                new_game = self.zeropawn_cpp.ChessGame()
                current_fen = "startpos"
            elif idx < len(tokens) and tokens[idx] == "fen":
                end_fen = moves_idx if moves_idx is not None else len(tokens)
                fen_str = " ".join(tokens[idx + 1 : end_fen])
                try:
                    new_game = self.zeropawn_cpp.ChessGame(fen_str)
                except ValueError as e:
                    # Malformed or rule-illegal FEN (e.g. side not to move in
                    # check): the native layer rejects it with ValueError
                    # instead of crashing on king-capture move generation.
                    # Keep the previous game state and stay alive.
                    logger.error("Rejected invalid position FEN (%s): %s", e, fen_str)
                    return True
                current_fen = fen_str
            else:
                new_game = self.zeropawn_cpp.ChessGame()

            # Base ply from the bare position's own metadata, BEFORE any
            # moves suffix is replayed (remediation CR-07): fullmove N with
            # white to move is (N-1)*2 plies deep, black to move (N-1)*2 + 1.
            # This reflects the FEN's move number without inventing history;
            # replayed moves are added on top at book-lookup time.
            base_ply = (new_game.full_move_number() - 1) * 2 + (
                0 if new_game.side_to_move() else 1
            )

            new_moves: List[str] = []
            if moves_idx is not None:
                for move_uci in tokens[moves_idx + 1 :]:
                    if new_game.is_legal(move_uci):
                        new_game.make_move_uci(move_uci)
                        new_moves.append(move_uci)
                    else:
                        logger.warning(
                            "Illegal move in position command: '%s', stopping move replay", move_uci
                        )
                        break

            # Check if subtree reuse can advance root to resulting position
            reused = False
            if (
                self.reuse_tree
                and self.last_engine_action is not None
                and self.last_engine_move is not None
                and self.prev_moves_at_search is not None
                and current_fen == self.prev_fen
                and len(new_moves) == len(self.prev_moves_at_search) + 2
                and new_moves[: len(self.prev_moves_at_search)] == self.prev_moves_at_search
                and new_moves[-2] == self.last_engine_move
            ):
                opp_move_str = new_moves[-1]
                temp_g = (
                    self.zeropawn_cpp.ChessGame()
                    if current_fen == "startpos"
                    else self.zeropawn_cpp.ChessGame(current_fen)
                )
                for m in self.prev_moves_at_search:
                    temp_g.make_move_uci(m)
                temp_g.make_move_uci(self.last_engine_move)
                opp_action = temp_g.string_to_action(opp_move_str)
                if opp_action >= 0:
                    reused = self.mcts.advance_root(self.last_engine_action, opp_action, new_game)

            if not reused:
                self.mcts.reset_root()

            self.game = new_game
            self.prev_fen = current_fen
            self.prev_moves = new_moves
            self.prev_base_ply = base_ply
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None

        elif cmd == "go":
            side = self.game.side_to_move()

            args = tokens[1:]
            arg_dict = {}
            i = 0
            while i < len(args):
                k = args[i]
                if (
                    i + 1 < len(args)
                    and not args[i + 1].startswith("-")
                    and args[i + 1] not in ["infinite"]
                ):
                    arg_dict[k] = args[i + 1]
                    i += 2
                else:
                    arg_dict[k] = True
                    i += 1

            sims, deadline_ms, infinite = self.allocate_time(arg_dict, side)
            if infinite:
                # Only "go infinite" runs asynchronously: it has no budget, so
                # stdin must stay responsive for "stop"/"quit" to end it.
                logger.info("go infinite: searching until stop")
                self._start_search(sims, deadline_ms)
            else:
                # Budgeted searches run synchronously — the simulation budget
                # itself bounds latency. A fresh stop event guards against a
                # stale "stop" from a previous search aborting this one.
                if not self._wait_for_running_search():
                    logger.error(
                        "Refusing 'go': previous search has not finished; its bestmove "
                        "will be emitted when it completes"
                    )
                    return True
                self._stop_event = threading.Event()
                t_search0 = time.perf_counter()
                self.search_and_respond(sims)
                if getattr(self, "_last_search_ran", False):
                    tm = getattr(self, "time_manager", None)
                    if tm is not None:
                        try:
                            tp_key = ThroughputKey.from_engine(
                                self.device, self.search_mode, getattr(self, "batch_size", 1)
                            )
                            tm.record(tp_key, sims, time.perf_counter() - t_search0)
                        except Exception:
                            logger.debug("throughput record failed", exc_info=True)

        elif cmd == "stop":
            # Honor stop promptly: the in-search abort predicate polls this
            # event every ~32 simulations; the search thread then emits
            # bestmove for the tree built so far.
            self._stop_event.set()

        elif cmd == "quit":
            self._stop_event.set()
            thread = self._search_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
            return False

        return True

    def run_stdio_loop(self) -> None:
        """Main stdio loop reading UCI protocol commands."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                if not self.handle_line(line):
                    break
            except (EOFError, KeyboardInterrupt):
                break
        # Clean shutdown: don't leave a search thread writing to a dead pipe.
        self._stop_event.set()
        thread = self._search_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def main() -> int:
    """Entry point for standalone UCI engine executable."""
    engine = UCIEngine()
    engine.run_stdio_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
