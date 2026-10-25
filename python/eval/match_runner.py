"""100-Game Match Runner against Fixed Reference Opponent (Stockfish).

Contract:
- Fixed reference opponent: tools/stockfish.exe at fixed node limit (go nodes 1000).
- Fixed opening set: data/openings/eval_openings_50.epd (50 openings).
- Exactly 100 games: each opening played twice with colors alternated
  (Game 1: Engine White / Stockfish Black, Game 2: Stockfish White / Engine Black).
- Play mode for Engine: NO Dirichlet noise, NO temperature, argmax visit selection.

Optimizations:
- Persistent Stockfish process pool (processes kept open across games).
- C++ MCTS transposition cache retained across moves of the same game.
- Fast search budget (50 simulations per move).
- Standard draw and resignation adjudication for barren or decisive endgames.
- Real-time progress updates logged after every opening pair (every 2 games).
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from contextlib import nullcontext
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Optional, Union
import numpy as np
import torch

import zeropawn_cpp
from python.model.chess_net import ChessNet

logger = logging.getLogger("match_runner")

VALID_SEARCH_MODES = ("serial", "batched")


def check_search_mode(search_mode: str) -> str:
    """Validate an explicit search-mode setting (default ``serial``).

    The ``batched`` C++ path does not exist on this branch; requesting it
    raises at use time. Unknown values raise immediately.
    """
    if search_mode not in VALID_SEARCH_MODES:
        raise ValueError(
            f"Unknown search_mode {search_mode!r}; expected one of {VALID_SEARCH_MODES}."
        )
    return search_mode


def require_serial_search(search_mode: str, caller: str) -> None:
    """Validate search mode support."""
    if search_mode == "batched":
        if not hasattr(zeropawn_cpp.MCTS, "search_batched"):
            raise RuntimeError(
                f"{caller}: search_mode='batched' requested but zeropawn_cpp does not export search_batched."
            )


def engine_outcome(board_result: str, engine_is_white: bool) -> str:
    """Map a BOARD-relative result ("1-0"/"1/2-1/2"/"0-1") to engine-relative.

    Returns one of "win", "draw", "loss". Anything that is not an explicit
    white-win or draw counts as a loss (matches historical tally behavior).
    """
    if board_result == "1/2-1/2":
        return "draw"
    white_won = board_result == "1-0"
    return "win" if white_won == engine_is_white else "loss"


def score_board_results(results_white: list[str], results_black: list[str]) -> dict[str, int]:
    """Tally engine W/D/L from board-relative results split by engine color.

    ``results_white``: board-relative results of games where the engine was
    White; ``results_black``: where the engine was Black.
    """
    wins = draws = losses = 0
    for res, is_white in [(r, True) for r in results_white] + [(r, False) for r in results_black]:
        outcome = engine_outcome(res, is_white)
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
    return {"engine_wins": wins, "engine_draws": draws, "engine_losses": losses}


def eval_config_hash(
    stockfish_nodes: int,
    mcts_sims: int,
    c_puct: float,
    openings_path: Union[str, Path],
    num_workers: int,
    policy_target_temp: Optional[float] = None,
    entropy_bonus: Optional[float] = None,
) -> str:
    """Short stable hash of the eval-match configuration for CSV provenance."""
    import hashlib

    canonical = f"{stockfish_nodes}|{mcts_sims}|{c_puct}|{Path(openings_path).name}|{num_workers}"
    if policy_target_temp is not None:
        canonical += f"|temp={policy_target_temp}"
    if entropy_bonus is not None:
        canonical += f"|ent={entropy_bonus}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def default_mcts_seed() -> int:
    """MCTS seed actually used by eval match workers (config default; noise off)."""
    return int(zeropawn_cpp.MCTSConfig().seed)


def elo_interval(wins: int, draws: int, losses: int) -> dict[str, Optional[float]]:
    """Elo difference (engine − opponent) from W/D/L with Wald 95% interval.

    Formula: score s = (W + D/2) / N; diff = -400 * log10(1/s - 1) (logistic).
    Interval: s ± 1.96 * sqrt(s(1-s)/N), endpoints clamped to (0.001, 0.999)
    and transformed by the same logistic. Returns None endpoints when the
    score is exactly 0 or 1 (unbounded). This is approximate: game pairs share
    openings (not independent) and 100 games give wide intervals (±~70+ Elo
    near 50%).
    """
    import math

    n = wins + draws + losses
    if n <= 0:
        return {"elo_diff": None, "lo": None, "hi": None, "score": 0.0}
    s = (wins + 0.5 * draws) / n

    def _to_elo(x: float) -> Optional[float]:
        if x <= 0.0 or x >= 1.0:
            return None
        return -400.0 * math.log10(1.0 / x - 1.0)

    se = math.sqrt(max(s * (1.0 - s), 0.0) / n)
    lo_s = max(s - 1.96 * se, 0.001)
    hi_s = min(s + 1.96 * se, 0.999)
    return {"elo_diff": _to_elo(s), "lo": _to_elo(lo_s), "hi": _to_elo(hi_s), "score": s}


def write_match_pgn(games: list[dict], pgn_path: Union[str, Path]) -> str:
    """Append match games as PGN (board-relative results). Returns path str.

    Each game dict: opening_idx, opening_fen, engine_is_white, result, moves,
    seed. Headers include engine color, seed, and opening index for audit.
    """
    import chess
    import chess.pgn

    pgn_path = Path(pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pgn_path, "a", encoding="utf-8") as f:
        for game in games:
            board = chess.Board(game["opening_fen"])
            for mv in game["moves"]:
                try:
                    board.push_uci(mv)
                except ValueError:
                    break
            pgn_game = chess.pgn.Game.from_board(board)
            pgn_game.headers["Event"] = "ZeroPawn calibrated baseline"
            pgn_game.headers["White"] = (
                "ZeroPawn-engine" if game["engine_is_white"] else "Stockfish"
            )
            pgn_game.headers["Black"] = (
                "Stockfish" if game["engine_is_white"] else "ZeroPawn-engine"
            )
            pgn_game.headers["Result"] = game["result"]
            pgn_game.headers["EngineColor"] = "white" if game["engine_is_white"] else "black"
            pgn_game.headers["OpeningIdx"] = str(game["opening_idx"])
            pgn_game.headers["MCTSSeed"] = str(game["seed"])
            pgn_game.headers["FEN"] = game["opening_fen"]
            print(pgn_game, file=f, end="\n\n")
    return str(pgn_path)


def rescore_pgn(pgn_path: Union[str, Path]) -> dict:
    """Independently re-score a match PGN with python-chess (not match_runner).

    Replays every game: flags illegal moves, and flags checkmate results that
    contradict the final board (loser must be the side to move). Adjudicated
    decisive results (resignation/value cutoffs) cannot be re-derived from the
    board and are tallied from the recorded strings via an independent
    (engine-color-aware) count. Returns agreement summary + problem rows.
    """
    import chess
    import chess.pgn

    rows = []
    with open(pgn_path, encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            illegal = 0
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    illegal += 1
                    break
                board.push(move)
            recorded = game.headers.get("Result", "*")
            contradiction = False
            if board.is_checkmate():
                truth = "0-1" if board.turn == chess.WHITE else "1-0"
                contradiction = recorded != truth
            engine_white = game.headers.get("EngineColor", "white") == "white"
            rows.append(
                {
                    "engine_is_white": engine_white,
                    "recorded": recorded,
                    "illegal": illegal,
                    "mate_contradiction": contradiction,
                }
            )
    ok = [r for r in rows if r["illegal"] == 0 and not r["mate_contradiction"]]
    wins = sum(1 for r in ok if engine_outcome(r["recorded"], r["engine_is_white"]) == "win")
    draws = sum(1 for r in ok if r["recorded"] == "1/2-1/2")
    losses = len(ok) - wins - draws
    return {
        "games": len(rows),
        "clean_games": len(ok),
        "engine_wins": wins,
        "engine_draws": draws,
        "engine_losses": losses,
        "problem_rows": [r for r in rows if r["illegal"] or r["mate_contradiction"]],
    }


class StockfishPlayer:
    """Wrapper around Stockfish engine subprocess communicating via UCI."""

    def __init__(
        self, stockfish_path: Union[str, Path] = "tools/stockfish.exe", nodes: int = 1000
    ) -> None:
        s = str(stockfish_path)
        if "\x00" in s:
            raise ValueError("Invalid stockfish path: NUL byte not allowed.")
        self.stockfish_path = Path(s)
        if not self.stockfish_path.is_file():
            raise FileNotFoundError(f"Stockfish binary not found at {self.stockfish_path}")
        if sys.platform == "win32" and self.stockfish_path.suffix.lower() != ".exe":
            raise ValueError(
                f"Stockfish binary on Windows must end with .exe: {self.stockfish_path}"
            )
        self.nodes = nodes
        self.proc: Optional[subprocess.Popen] = None
        self._line_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._start_process()

    def _reader_worker(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        try:
            for line in iter(self.proc.stdout.readline, ""):
                self._line_queue.put(line.strip())
        except (OSError, ValueError):
            pass
        self._line_queue.put(None)

    def _start_process(self) -> None:
        if self.proc is not None:
            self.close()
        self._line_queue = queue.Queue()
        self.proc = subprocess.Popen(
            [str(self.stockfish_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self._reader_thread.start()

        self.send("uci")
        while True:
            line = self.readline(timeout=10.0)
            if line == "uciok" or not line:
                break
        self.send("isready")
        while True:
            line = self.readline(timeout=10.0)
            if line == "readyok" or not line:
                break

    def reset_game(self) -> None:
        """Reset internal engine state between games."""
        self.send("ucinewgame")
        self.send("isready")
        while True:
            line = self.readline(timeout=10.0)
            if line == "readyok" or not line:
                break

    def set_elo(self, elo: int) -> None:
        """Enable UCI_LimitStrength at the given UCI_Elo (e.g. 1320/1500/1700).

        Approximate strength levels (Stockfish docs); not a certified rating.
        """
        if not 1320 <= elo <= 3190:
            raise ValueError(f"UCI_Elo {elo} outside Stockfish range 1320-3190.")
        self.send("setoption name UCI_LimitStrength value true")
        self.send(f"setoption name UCI_Elo value {elo}")
        self.send("isready")
        while True:
            line = self.readline(timeout=10.0)
            if line == "readyok" or not line:
                break
        self.elo = elo

    def send(self, cmd: str) -> None:
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(f"{cmd}\n")
                self.proc.stdin.flush()
            except (OSError, BrokenPipeError):
                logger.warning("Failed to send command %r to Stockfish", cmd)

    def readline(self, timeout: float = 10.0) -> str:
        if not self.proc:
            return ""
        try:
            item = self._line_queue.get(timeout=timeout)
            return item if item is not None else ""
        except queue.Empty:
            logger.warning("Stockfish readline timed out after %.1fs", timeout)
            return ""

    def get_move(self, fen: str) -> str:
        """Query Stockfish for best move from FEN with fixed node limit."""
        self.send(f"position fen {fen}")
        self.send(f"go nodes {self.nodes}")

        best_move = ""
        while True:
            line = self.readline(timeout=30.0)
            if not line:
                break
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                break

        return best_move

    def close(self) -> None:
        if self.proc:
            try:
                self.send("quit")
                self.proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.debug("Stockfish quit failed or timed out: %s; killing", e)
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1)
                except (subprocess.SubprocessError, OSError):
                    pass
            finally:
                self.proc = None


class MatchWorkerContext:
    """Thread-local context holding persistent Stockfish and MCTS instances."""

    def __init__(
        self,
        stockfish_path: Path,
        stockfish_nodes: int,
        c_puct: float,
        seed: int = 42,
        stockfish_elo: Optional[int] = None,
        reuse_tree: bool = False,
        batch_size: int = 1,
    ) -> None:
        self.stockfish = StockfishPlayer(stockfish_path, nodes=stockfish_nodes)
        if stockfish_elo is not None:
            self.stockfish.set_elo(stockfish_elo)
        mcts_cfg = zeropawn_cpp.MCTSConfig()
        mcts_cfg.c_puct = c_puct
        mcts_cfg.add_noise = False  # NO noise during match evaluation
        mcts_cfg.virtual_loss = 1
        mcts_cfg.fpu_reduction = 0.2
        mcts_cfg.seed = seed
        mcts_cfg.reuse_tree = reuse_tree
        mcts_cfg.batch_size = max(1, int(batch_size))
        self.mcts = zeropawn_cpp.MCTS(mcts_cfg)

    def close(self) -> None:
        self.stockfish.close()


class MatchRunner:
    """Manages the 100-game match between ChessNet MCTS and Stockfish reference."""

    use_book: bool = True
    book_path: Optional[str] = "data/openings/curated_book.json"
    book_policy: str = "weighted_blend"
    opening_book: Optional[Any] = None

    def __init__(
        self,
        model: ChessNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        stockfish_path: Union[str, Path] = "tools/stockfish.exe",
        stockfish_nodes: int = 1000,
        mcts_sims: int = 50,
        c_puct: float = 2.0,
        openings_path: Union[str, Path] = "data/openings/eval_openings_50.epd",
        num_workers: int = 8,
        search_mode: str = "serial",
        seed: int = 42,
        stockfish_elo: Optional[int] = None,
        use_book: bool = True,
        book_path: Optional[Union[str, Path]] = "data/openings/curated_book.json",
        book_policy: str = "weighted_blend",
        batch_size: Union[int, str] = 1,
        reuse_tree: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        self.search_mode = check_search_mode(search_mode)
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            from python.hardware.profile import resolve_batch_size

            self.batch_size = resolve_batch_size(batch_size, workload="match", default=16)
        else:
            self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.stockfish_elo = stockfish_elo
        self.reuse_tree = reuse_tree

        self.stockfish_path = Path(stockfish_path)
        self.stockfish_nodes = stockfish_nodes
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.openings = self._load_openings(Path(openings_path))
        self.num_workers = num_workers
        self._lock = threading.Lock()

        # Opening book support (defaults to 'weighted_blend' to align with UCI default policy)
        self.use_book = use_book
        self.book_path = str(book_path) if book_path else None
        self.book_policy = book_policy
        self.opening_book = None
        if self.use_book and self.book_path and Path(self.book_path).is_file():
            try:
                from python.data.opening_book import OpeningBook

                self.opening_book = OpeningBook(
                    book_path=self.book_path,
                    policy=self.book_policy,
                    enabled=True,
                )
            except Exception as e:
                logger.debug("Opening book disabled in MatchRunner: %s", e)
                self.opening_book = None

    def _load_openings(self, path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Openings file not found: {path}")
        fens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fen = line.split(";")[0].strip()
                if fen:
                    fens.append(fen)
        return fens

    def _eval_callback(self, game: zeropawn_cpp.Game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with self._lock:
                with autocast_ctx:
                    p_logits, v_wdl = self.model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])

        return (p_probs, scalar_val)

    def _batched_eval_callback(self, batch_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(batch_planes.shape[0])
        if n == 0:
            return (np.zeros((0, 4672), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32))
        t = t.view(n, 19, 8, 8).to(self.device, non_blocking=True)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with self._lock:
                with autocast_ctx:
                    p_logits, v_wdl = self.model(t)
            policy = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            wdl = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)

        return policy.astype(np.float32), values

    def _eval_batch_callback(self, games: list) -> list[tuple[list[float], float]]:
        if not games:
            return []
        batch_planes = []
        for g in games:
            if hasattr(g, "get_fen"):
                cg = zeropawn_cpp.ChessGame(g.get_fen())
                batch_planes.append(zeropawn_cpp.encode_state(cg))
            else:
                batch_planes.append(zeropawn_cpp.encode_state(g))

        batch_arr = np.stack(batch_planes)
        t = torch.from_numpy(batch_arr).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with self._lock:
                with autocast_ctx:
                    p_logits, v_wdl = self.model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            scalar_vals = (v_probs[:, 0] - v_probs[:, 2]).astype(float)

        results = []
        for i in range(len(games)):
            results.append((p_probs[i].tolist(), float(scalar_vals[i])))
        return results

    def _play_game_worker(
        self,
        opening_fen: str,
        engine_is_white: bool,
        ctx: MatchWorkerContext,
        max_plies: int = 150,
        game_seed: Optional[int] = None,
        move_log: Optional[list] = None,
    ) -> str:
        """Play single game between engine and Stockfish worker with cache reuse and adjudication.

        Returns a BOARD-relative result string: "1-0" (White won), "0-1"
        (Black won), or "1/2-1/2" (draw), regardless of engine color.
        If ``game_seed`` is given the worker MCTS is re-seeded for this game
        (deterministic per game). If ``move_log`` (a list) is given, the
        game's UCI moves are appended to it.
        """
        result, moves = self._play_game_impl(
            opening_fen, engine_is_white, ctx, max_plies, game_seed
        )
        if move_log is not None:
            move_log.extend(moves)
        return result

    def _play_game_impl(
        self,
        opening_fen: str,
        engine_is_white: bool,
        ctx: MatchWorkerContext,
        max_plies: int = 150,
        game_seed: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        """Play single game; return (board-relative result, UCI moves)."""
        require_serial_search(self.search_mode, "MatchRunner._play_game_worker")
        game = zeropawn_cpp.ChessGame(opening_fen)
        ctx.stockfish.reset_game()
        # Retain MCTS cache within game, clear only at start of game
        ctx.mcts.clear()
        if game_seed is not None:
            ctx.mcts.set_seed(game_seed)
        moves: list[str] = []
        last_engine_action: Optional[int] = None

        def _finish(result: str) -> tuple[str, list[str]]:
            return (result, moves)

        plies = 0
        consecutive_hopeless = 0
        consecutive_winning = 0

        while plies < max_plies:
            if game.is_game_over():
                break

            # Adjudication: 50-move countdown reaches >= 60 plies (30 full moves no progress)
            if game.half_move_clock() >= 60:
                return _finish("1/2-1/2")

            # NOTE: ChessGame.side_to_move() returns bool (True = White to move).
            # Never compare it with == 0 / == 1 (True == 0 is False).
            side = game.side_to_move()
            white_to_move = bool(side)
            is_engine_turn = white_to_move == engine_is_white

            if is_engine_turn:
                # Check opening book first if enabled (default policy aligned with UCI: weighted_blend)
                book_move = None
                book = getattr(self, "opening_book", None)
                if book is not None and getattr(self, "use_book", False):
                    try:
                        import chess

                        py_board = chess.Board(game.get_fen())
                        book_move = book.get_move(py_board)
                    except Exception:
                        book_move = None

                if book_move:
                    move_uci = book_move.uci()
                    root_val = 0.0
                    last_engine_action = None
                    if getattr(self, "reuse_tree", False):
                        ctx.mcts.reset_root()
                else:
                    # Do NOT call clear() here — retain transposition cache across plies!
                    if (
                        self.search_mode == "batched"
                        and hasattr(ctx.mcts, "search_batched")
                        and not getattr(self, "reuse_tree", False)
                    ):
                        res = ctx.mcts.search_batched(
                            game, self.mcts_sims, self.batch_size, self._batched_eval_callback
                        )
                    else:
                        res = ctx.mcts.search(
                            game, self.mcts_sims, self._eval_callback, self._eval_batch_callback
                        )
                    move_uci = game.action_to_string(res.best_action)
                    root_val = res.root_value
                    last_engine_action = res.best_action

                # Adjudication checks based on root value (engine-relative value:
                # negative = engine losing, positive = engine winning). Returns
                # are BOARD-relative: map the engine outcome to White/Black.
                if root_val < -0.92:
                    consecutive_hopeless += 1
                    if consecutive_hopeless >= 4:
                        # Engine forfeits: engine loses board-relatively.
                        return _finish("0-1" if engine_is_white else "1-0")
                else:
                    consecutive_hopeless = 0

                if root_val > 0.92:
                    consecutive_winning += 1
                    if consecutive_winning >= 4:
                        # Engine adjudicated winner board-relatively.
                        return _finish("1-0" if engine_is_white else "0-1")
                else:
                    consecutive_winning = 0
            else:
                move_uci = ctx.stockfish.get_move(game.get_fen())
                if (
                    getattr(self, "reuse_tree", False)
                    and last_engine_action is not None
                    and move_uci
                    and game.is_legal(move_uci)
                ):
                    opp_action = game.string_to_action(move_uci)
                    if opp_action >= 0:
                        temp_next = game.clone()
                        temp_next.apply_action(opp_action)
                        ctx.mcts.advance_root(last_engine_action, opp_action, temp_next)
                    else:
                        ctx.mcts.reset_root()
                elif getattr(self, "reuse_tree", False):
                    ctx.mcts.reset_root()
                last_engine_action = None

            if not move_uci or not game.is_legal(move_uci):
                # Side to move forfeits (illegal/missing move): loser is the
                # mover, board-relatively (True = White to move = White loses).
                return _finish("0-1" if game.side_to_move() else "1-0")

            game.make_move_uci(move_uci)
            moves.append(move_uci)
            plies += 1

        if game.is_checkmate():
            # Loser is the side to move (True = White mated => Black won).
            # Board-relative: loser White -> "0-1", loser Black -> "1-0".
            loser_is_white = bool(game.side_to_move())
            return _finish("0-1" if loser_is_white else "1-0")
        else:
            return _finish("1/2-1/2")

    def run_match(
        self,
        num_openings: int = 50,
        pgn_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Union[int, float]]:
        """Run 2 * num_openings games in parallel across opening pairs with persistent workers.

        Game ``i`` (opening index ``idx``, engine-white first) uses MCTS seed
        ``self.seed + 2*idx`` / ``+ 2*idx+1`` — deterministic per game. If
        ``pgn_path`` is given, all games are appended as PGN (board-relative).
        Returns the tally plus a ``games`` detail list and ``pgn_path``.
        """
        selected_openings = self.openings[:num_openings]
        engine_wins = 0
        engine_draws = 0
        engine_losses = 0
        total_games = len(selected_openings) * 2
        elo_str = f"UCI_Elo {self.stockfish_elo}" if self.stockfish_elo else "full strength"

        logger.info(
            f"Starting {total_games}-game match vs Stockfish ({self.stockfish_nodes} nodes, {elo_str}, "
            f"{self.mcts_sims} sims) [search_mode={self.search_mode}] [seed={self.seed}] "
            f"using {self.num_workers} persistent parallel workers..."
        )
        start_time = time.time()

        # Create persistent worker contexts
        contexts = [
            MatchWorkerContext(
                self.stockfish_path,
                self.stockfish_nodes,
                self.c_puct,
                seed=self.seed,
                stockfish_elo=self.stockfish_elo,
                reuse_tree=self.reuse_tree,
                batch_size=self.batch_size,
            )
            for _ in range(self.num_workers)
        ]
        ctx_queue: queue.Queue[MatchWorkerContext] = queue.Queue()
        for ctx in contexts:
            ctx_queue.put(ctx)

        def worker_task(idx_fen: tuple[int, str]) -> tuple[dict, dict]:
            idx, fen = idx_fen
            worker_ctx = ctx_queue.get()
            try:
                moves_w: list[str] = []
                res_w = self._play_game_worker(
                    fen,
                    engine_is_white=True,
                    ctx=worker_ctx,
                    game_seed=self.seed + 2 * idx,
                    move_log=moves_w,
                )
                moves_b: list[str] = []
                res_b = self._play_game_worker(
                    fen,
                    engine_is_white=False,
                    ctx=worker_ctx,
                    game_seed=self.seed + 2 * idx + 1,
                    move_log=moves_b,
                )
                return (
                    {
                        "opening_idx": idx,
                        "opening_fen": fen,
                        "engine_is_white": True,
                        "result": res_w,
                        "moves": moves_w,
                        "seed": self.seed + 2 * idx,
                    },
                    {
                        "opening_idx": idx,
                        "opening_fen": fen,
                        "engine_is_white": False,
                        "result": res_b,
                        "moves": moves_b,
                        "seed": self.seed + 2 * idx + 1,
                    },
                )
            finally:
                ctx_queue.put(worker_ctx)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(worker_task, (idx, fen))
                    for idx, fen in enumerate(selected_openings)
                ]

                completed_pairs = 0
                results_as_white: list[str] = []
                results_as_black: list[str] = []
                games: list[dict] = []
                for future in concurrent.futures.as_completed(futures):
                    game_w, game_b = future.result()
                    games.extend([game_w, game_b])
                    # Board-relative strings tallied per engine color.
                    results_as_white.append(game_w["result"])
                    results_as_black.append(game_b["result"])
                    tally = score_board_results(results_as_white, results_as_black)
                    engine_wins, engine_draws, engine_losses = (
                        tally["engine_wins"],
                        tally["engine_draws"],
                        tally["engine_losses"],
                    )

                    completed_pairs += 1
                    games_done = completed_pairs * 2
                    score = (engine_wins + 0.5 * engine_draws) / games_done
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Match Progress: {games_done:3d}/{total_games} games | "
                        f"W: {engine_wins:2d} D: {engine_draws:2d} L: {engine_losses:2d} | "
                        f"Score: {score * 100:5.1f}% | Elapsed: {elapsed:5.1f}s "
                        f"({elapsed / games_done:.1f}s/game)"
                    )
        finally:
            for ctx in contexts:
                ctx.close()

        score_pct = (engine_wins + 0.5 * engine_draws) / total_games if total_games > 0 else 0.0
        draw_rate = engine_draws / total_games if total_games > 0 else 0.0
        games.sort(key=lambda g: (g["opening_idx"], not g["engine_is_white"]))

        pgn_saved: Optional[str] = None
        if pgn_path is not None:
            pgn_saved = write_match_pgn(games, pgn_path)

        return {
            "total_games": total_games,
            "engine_wins": engine_wins,
            "engine_draws": engine_draws,
            "engine_losses": engine_losses,
            "engine_score": score_pct,
            "draw_rate": draw_rate,
            "search_mode": self.search_mode,
            "seed": self.seed,
            "stockfish_elo": self.stockfish_elo,
            "games": games,
            "pgn_path": pgn_saved,
            "elapsed_sec": round(time.time() - start_time, 2),
        }

    def close(self) -> None:
        pass
