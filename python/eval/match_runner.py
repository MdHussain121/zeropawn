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
import hashlib
import logging
from pathlib import Path
from contextlib import nullcontext
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional, Union
import numpy as np
import torch

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.eval.stats import (
    elo_confidence_interval,
    paired_score_and_ci,
)

logger = logging.getLogger("match_runner")


def _compute_material_balance(opening_fen: str, moves: list[str], engine_is_white: bool) -> int:
    """Compute material balance (engine - opponent) from pawn values (P=1, N=3, B=3, R=5, Q=9)."""
    try:
        import chess

        board = chess.Board(opening_fen)
        for m in moves:
            board.push_uci(m)
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        w_mat = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in vals.items())
        b_mat = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in vals.items())
        eng_mat = w_mat if engine_is_white else b_mat
        opp_mat = b_mat if engine_is_white else w_mat
        return eng_mat - opp_mat
    except Exception:
        return 0

VALID_SEARCH_MODES = ("serial", "batched")


def check_search_mode(search_mode: str) -> str:
    """Validate an explicit search-mode setting (default ``serial``).

    ``serial`` runs one leaf evaluation per simulation; ``batched`` routes
    through the legacy ``MCTS.search_batched`` entry point (intra-search
    batching, no subtree reuse — see audit M-2). Unknown values raise
    immediately.
    """
    if search_mode not in VALID_SEARCH_MODES:
        raise ValueError(
            f"Unknown search_mode {search_mode!r}; expected one of {VALID_SEARCH_MODES}."
        )
    return search_mode


def assert_search_mode_supported(search_mode: str, caller: str) -> None:
    """Assert the configured search mode is SUPPORTED by this build.

    Audit SR-14: the previous name (``require_serial_search``) implied an
    enforcement it never performed — it only checked that ``batched`` is
    backed by ``zeropawn_cpp.search_batched`` and accepted ``serial``
    unconditionally. Callers that need the mode to be supported (self-play,
    match, puzzle, UCI paths) now say so explicitly.
    """
    if search_mode == "batched":
        if not hasattr(zeropawn_cpp.MCTS, "search_batched"):
            raise RuntimeError(
                f"{caller}: search_mode='batched' requested but zeropawn_cpp does not export search_batched."
            )


def enforce_serial_search(search_mode: str, caller: str) -> None:
    """Enforce the Stage-2 locked serial contract: raise unless mode is 'serial'.

    Provided for callers that genuinely require serial search (the locked
    Stage-2 configuration, and any future path where batching is known to be
    unsupported at runtime). No current production caller enforces serial —
    ``serial`` remains the default everywhere, and ``batched`` is an explicit
    opt-in whose support is checked by :func:`assert_search_mode_supported`.
    """
    if search_mode != "serial":
        raise RuntimeError(
            f"{caller}: serial search is required by this code path, got search_mode={search_mode!r}."
        )


def require_serial_search(search_mode: str, caller: str) -> None:
    """Backward-compatible alias for :func:`assert_search_mode_supported`.

    Kept because external callers (``python/cli/commands_match.py``,
    ``python/uci/uci_engine.py``, tests) import this name; its behavior is
    unchanged from before the SR-14 split (support check only). New code should
    call ``assert_search_mode_supported`` (support check) or
    ``enforce_serial_search`` (actual serial enforcement).
    """
    assert_search_mode_supported(search_mode, caller)


def engine_outcome(board_result: str, engine_is_white: bool) -> str:
    """Map a BOARD-relative result ("1-0"/"1/2-1/2"/"0-1"/"*") to engine-relative.

    Returns one of "win", "draw", "loss", or "capped" (Audit F-01).
    A game reaching the ply cap without a decisive or rule-based draw outcome
    is mapped to "capped" rather than being silently counted as a draw or loss.
    """
    if board_result == "1/2-1/2":
        return "draw"
    if board_result in ("*", "inconclusive", "capped"):
        return "capped"
    white_won = board_result == "1-0"
    return "win" if white_won == engine_is_white else "loss"


# Board-relative result strings recognized by the shared game classifier.
# "*"/"inconclusive"/"capped" all mean "no genuine chess result" (ply cap).
_CAP_RESULT_STRINGS = ("*", "inconclusive", "capped")
_GENUINE_RESULT_STRINGS = ("1-0", "0-1", "1/2-1/2")

_RECORD_OUTCOME_COUNTERS = {
    "decisive_win": "decisive_wins",
    "decisive_loss": "decisive_losses",
    "forfeit_win": "forfeit_wins",
    "forfeit_loss": "forfeit_losses",
    "genuine_draw": "genuine_draws",
    "capped": "capped_games",
    "uncounted": "uncounted_games",
}


def classify_game_record(record: dict) -> str:
    """Classify ONE completed game record into a fine-grained engine-relative outcome.

    Shared classification contract (Audit F-01, S5-05): this is the single
    implementation used by BOTH ``MatchRunner.run_match`` and
    ``scripts/run_sprt.py`` so that a game reaching the ply cap classifies as
    "capped" — never as a genuine draw — in every match/SPRT harness.

    Accepts MatchRunner-style records (``result`` / ``engine_is_white``) and
    SPRT-style records (``board_result`` / ``cand_white``), plus the common
    provenance fields ``outcome_class`` ("decisive"|"genuine_draw"|"capped"|
    "forfeit"|...) and ``termination_reason`` (e.g. "checkmate"|
    "max_plies_cap"|"rule_draw").

    Returns one of:
      "decisive_win", "decisive_loss", "forfeit_win", "forfeit_loss",
      "genuine_draw", "capped", "uncounted".

    Semantics:
      - A ply-cap termination ("*"-family result, outcome_class "capped", or
        termination_reason "max_plies_cap") is "capped" and must NOT be
        scored as a genuine draw (cap_policy="exclude" default).
      - Forfeits (illegal move / timeout) count as wins/losses for the
        engine, tracked separately from decisive results.
      - Records with a missing or unrecognized result, or without a color,
        are "uncounted": they are reported but never scored.
    """
    res = record.get("result", record.get("board_result"))
    outcome_class = record.get("outcome_class")
    term_reason = record.get("termination_reason") or ""
    eng_white = record.get("engine_is_white", record.get("cand_white"))

    if res is None or eng_white is None:
        return "uncounted"
    if res in _CAP_RESULT_STRINGS:
        return "capped"
    if res not in _GENUINE_RESULT_STRINGS:
        return "uncounted"
    if outcome_class == "capped" or term_reason == "max_plies_cap":
        # e.g. cap_policy="draw" produced a "1/2-1/2" board string for a cap.
        return "capped"
    if res == "1/2-1/2":
        return "genuine_draw"
    engine_won = (res == "1-0") == bool(eng_white)
    if outcome_class == "forfeit":
        return "forfeit_win" if engine_won else "forfeit_loss"
    return "decisive_win" if engine_won else "decisive_loss"


def is_capped_game_record(record: dict) -> bool:
    """True when the record is a ply-cap termination (Audit F-01 / S5-05).

    Convenience predicate over the same shared classification used for the
    score; used for per-color capped accounting in SPRT summaries.
    """
    return classify_game_record(record) == "capped"


def classify_game_records(games: list) -> dict[str, int]:
    """Tally completed game records into outcome-class counters (shared contract).

    Single implementation backing ``MatchRunner.run_match`` and
    ``scripts/run_sprt.py`` — see :func:`classify_game_record` for the record
    contract and the capped-game semantics (Audit F-01 / S5-05).
    """
    counts = {
        "decisive_wins": 0,
        "decisive_losses": 0,
        "genuine_draws": 0,
        "capped_games": 0,
        "forfeit_wins": 0,
        "forfeit_losses": 0,
        "uncounted_games": 0,
    }
    for g in games:
        counts[_RECORD_OUTCOME_COUNTERS[classify_game_record(g)]] += 1
    return counts


def score_board_results(results_white: list[str], results_black: list[str]) -> dict[str, int]:
    """Tally engine W/D/L from board-relative results split by engine color.

    ``results_white``: board-relative results of games where the engine was
    White; ``results_black``: where the engine was Black.
    """
    wins = draws = losses = capped = 0
    for res, is_white in [(r, True) for r in results_white] + [(r, False) for r in results_black]:
        outcome = engine_outcome(res, is_white)
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        elif outcome == "capped":
            capped += 1
        else:
            losses += 1
    tally = {"engine_wins": wins, "engine_draws": draws, "engine_losses": losses}
    if capped > 0:
        tally["capped_games"] = capped
    return tally


def eval_config_hash(
    stockfish_nodes: int,
    mcts_sims: int,
    c_puct: float,
    openings_path: Union[str, Path],
    num_workers: int,
    policy_target_temp: Optional[float] = None,
    entropy_bonus: Optional[float] = None,
    stockfish_elo: Optional[int] = None,
    seed: Optional[int] = None,
    fpu_reduction: Optional[float] = None,
    use_book: Optional[bool] = None,
    book_hash: Optional[str] = None,
    book_policy: Optional[str] = None,
    max_plies: Optional[int] = None,
) -> str:
    """Short stable hash of the eval-match configuration for CSV provenance.

    Audit DI-15 & F-15: binds actual openings file content SHA-256, stockfish_elo,
    match seed, fpu_reduction, opening book configuration/hash, and max_plies.
    """
    import hashlib

    p = Path(openings_path)
    if p.is_file():
        try:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            openings_id = h.hexdigest()[:12]
        except Exception:
            openings_id = p.name
    else:
        openings_id = p.name

    canonical = f"{stockfish_nodes}|{mcts_sims}|{c_puct}|{openings_id}|{num_workers}"
    if stockfish_elo is not None:
        canonical += f"|sf_elo={stockfish_elo}"
    if seed is not None:
        canonical += f"|seed={seed}"
    if fpu_reduction is not None:
        canonical += f"|fpu={fpu_reduction}"
    if use_book is not None:
        canonical += f"|book={use_book}"
    if book_hash is not None:
        canonical += f"|book_hash={book_hash}"
    if book_policy is not None:
        canonical += f"|book_pol={book_policy}"
    if max_plies is not None:
        canonical += f"|max_plies={max_plies}"
    if policy_target_temp is not None:
        canonical += f"|temp={policy_target_temp}"
    if entropy_bonus is not None:
        canonical += f"|ent={entropy_bonus}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def default_mcts_seed() -> int:
    """MCTS seed actually used by eval match workers (config default; noise off)."""
    return int(zeropawn_cpp.MCTSConfig().seed)


def elo_interval(wins: int, draws: int, losses: int) -> dict[str, Optional[float]]:
    """Elo difference (engine - opponent) from W/D/L with canonical Agresti-Caffo 95% interval.

    Replaces the defective clamped-Wald variant with authoritative Agresti-Caffo (Audit F-03).
    Prevents artificial finite bounds at shutouts while matching logistic difference.
    Returns {"elo_diff": ..., "lo": ..., "hi": ..., "score": ...}.
    """
    res = elo_confidence_interval(wins, draws, losses, confidence=0.95, method="agresti_caffo")
    return {
        "elo_diff": res["elo_diff"],
        "lo": res["lo"],
        "hi": res["hi"],
        "score": res["score"],
    }


def engine_identity(stockfish_path: Union[str, Path]) -> dict:
    """Binary identity of the reference engine used for a match (provenance).

    Every persisted eval result should be able to name the exact opponent
    binary it was produced against (Stage A §7); hashing is done once per
    match, not per game.
    """
    p = Path(stockfish_path)
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"path": str(p), "sha256": h.hexdigest(), "size_bytes": p.stat().st_size}
    except OSError as e:
        logger.warning("Could not hash reference engine binary %s: %s", p, e)
        return {"path": str(p), "sha256": None, "size_bytes": None}


def write_match_pgn(games: list[dict], pgn_path: Union[str, Path]) -> str:
    """Append match games as PGN (board-relative results). Returns path str.

    Each game dict: opening_idx, opening_fen, engine_is_white, result, moves,
    seed. Headers include engine color, seed, opening index, termination, and
    material balance for independent verification (Audit F-01).
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
            if game.get("termination_reason"):
                pgn_game.headers["Termination"] = str(game["termination_reason"])
            if game.get("outcome_class"):
                pgn_game.headers["OutcomeClass"] = str(game["outcome_class"])
            if "final_material_balance" in game:
                pgn_game.headers["FinalMaterialBalance"] = str(game["final_material_balance"])
            if game.get("book_moves_count"):
                pgn_game.headers["BookMoves"] = str(game["book_moves_count"])
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
    capped = sum(1 for r in ok if r["recorded"] == "*")
    losses = sum(1 for r in ok if engine_outcome(r["recorded"], r["engine_is_white"]) == "loss")
    return {
        "games": len(rows),
        "clean_games": len(ok),
        "engine_wins": wins,
        "engine_draws": draws,
        "engine_losses": losses,
        "capped_games": capped,
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
    book_sha256: Optional[str] = None
    opening_book: Optional[Any] = None
    enable_value_adjudication: bool = True  # Class default enables _bare_runner unit tests
    cap_policy: str = "exclude"  # Audit F-01: exclude cap games from primary WDL tally

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
        enable_value_adjudication: bool = False,
        cap_policy: str = "exclude",
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
        self.enable_value_adjudication = enable_value_adjudication
        self.cap_policy = cap_policy

        self.stockfish_path = Path(stockfish_path)
        self.stockfish_nodes = stockfish_nodes
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.openings = self._load_openings(Path(openings_path))
        self.num_workers = num_workers
        self._lock = threading.Lock()

        # Opening book support with provenance (Audit F-05)
        self.use_book = use_book
        self.book_path = str(book_path) if book_path else None
        self.book_policy = book_policy
        self.book_sha256 = None
        self.opening_book = None
        if self.use_book and self.book_path:
            p_book = Path(self.book_path)
            if p_book.is_file():
                try:
                    h = hashlib.sha256()
                    with open(p_book, "rb") as fh:
                        for chunk in iter(lambda: fh.read(65536), b""):
                            h.update(chunk)
                    self.book_sha256 = h.hexdigest()
                    from python.data.opening_book import OpeningBook

                    self.opening_book = OpeningBook(
                        book_path=self.book_path,
                        policy=self.book_policy,
                        enabled=True,
                    )
                except Exception as e:
                    logger.warning("Opening book load failed in MatchRunner: %s; disabling book", e)
                    self.opening_book = None
                    self.use_book = False
            else:
                logger.warning(
                    "Opening book requested at '%s' but file does not exist; opening book is DISABLED.",
                    self.book_path,
                )
                self.use_book = False

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
        # Encode the passed C++ game directly (audit H-2): it carries the real
        # move history; the former FEN round-trip zeroed the repetition plane.
        planes = zeropawn_cpp.encode_state(game)
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
            # Array return (audit H-4): the binding fast-paths numpy arrays;
            # .tolist() serialization cost ~2.4x at B=16.
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])

        return (p_probs.astype(np.float32), scalar_val)

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
            # Encode the passed games directly (audit H-2): they are ChessGame
            # objects with real move history — no FEN round-trip.
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

        # Array returns (audit H-4): binding fast-paths numpy arrays; no
        # per-leaf .tolist() serialization.
        return [(p_probs[i], float(scalar_vals[i])) for i in range(len(games))]

    def _play_game_worker(
        self,
        opening_fen: str,
        engine_is_white: bool,
        ctx: MatchWorkerContext,
        max_plies: int = 150,
        game_seed: Optional[int] = None,
        move_log: Optional[list] = None,
        info_log: Optional[dict] = None,
    ) -> str:
        """Play single game between engine and Stockfish worker with cache reuse and adjudication.

        Returns a BOARD-relative result string: "1-0" (White won), "0-1"
        (Black won), or "1/2-1/2" (draw), regardless of engine color.
        If ``game_seed`` is given the worker MCTS is re-seeded for this game
        (deterministic per game). If ``move_log`` (a list) is given, the
        game's UCI moves are appended to it. If ``info_log`` (a dict) is
        given, it records whether the result was adjudicated and why
        (audit RL-20: value-cutoff adjudication must be measurable so an
        uncalibrated value head cannot silently fabricate match wins).
        """
        result, moves = self._play_game_impl(
            opening_fen, engine_is_white, ctx, max_plies, game_seed, info=info_log
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
        info: Optional[dict] = None,
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
        book_moves_count = 0
        # Audit RL-20 & F-01: record every adjudicated and terminated ending with
        # explicit outcome classifications, reasons, and material balance.
        adjudication_info: dict = {}

        def _finish(
            result: str,
            reason: Optional[str] = None,
            outcome_class: Optional[str] = None,
        ) -> tuple[str, list[str]]:
            if reason is not None:
                adjudication_info["termination_reason"] = reason
            if outcome_class is not None:
                adjudication_info["outcome_class"] = outcome_class
            else:
                if result in ("1-0", "0-1"):
                    adjudication_info["outcome_class"] = "decisive"
                elif result == "1/2-1/2":
                    adjudication_info["outcome_class"] = "genuine_draw"
                else:
                    adjudication_info["outcome_class"] = "capped"

            is_adjudicated = reason in (
                "value_cutoff_loss",
                "value_cutoff_win",
                "fifty_move_countdown",
                "max_plies_cap",
            )
            adjudication_info["adjudicated"] = is_adjudicated
            if is_adjudicated:
                adjudication_info["adjudication_reason"] = reason

            adjudication_info["final_fen"] = game.get_fen()
            adjudication_info["final_material_balance"] = _compute_material_balance(
                opening_fen, moves, engine_is_white
            )
            adjudication_info["book_moves_count"] = book_moves_count
            if info is not None:
                info.update(adjudication_info)
            return (result, moves)

        plies = 0
        consecutive_hopeless = 0
        consecutive_winning = 0

        while plies < max_plies:
            if game.is_game_over():
                break

            # Adjudication: 50-move countdown reaches >= 60 plies (30 full moves no progress)
            if game.half_move_clock() >= 60:
                return _finish(
                    "1/2-1/2",
                    reason="fifty_move_countdown",
                    outcome_class="genuine_draw",
                )

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
                    book_moves_count += 1
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
                # Audit F-04: only enabled if explicitly configured.
                if getattr(self, "enable_value_adjudication", False):
                    if root_val < -0.92:
                        consecutive_hopeless += 1
                        if consecutive_hopeless >= 4:
                            # Engine forfeits: engine loses board-relatively.
                            return _finish(
                                "0-1" if engine_is_white else "1-0",
                                reason="value_cutoff_loss",
                                outcome_class="decisive",
                            )
                    else:
                        consecutive_hopeless = 0

                    if root_val > 0.92:
                        consecutive_winning += 1
                        if consecutive_winning >= 4:
                            # Engine adjudicated winner board-relatively.
                            return _finish(
                                "1-0" if engine_is_white else "0-1",
                                reason="value_cutoff_win",
                                outcome_class="decisive",
                            )
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
                mover_is_white = bool(game.side_to_move())
                reason = "forfeit_illegal_move" if move_uci else "forfeit_timeout"
                return _finish(
                    "0-1" if mover_is_white else "1-0",
                    reason=reason,
                    outcome_class="forfeit",
                )

            game.make_move_uci(move_uci)
            moves.append(move_uci)
            plies += 1

        if game.is_checkmate():
            # Loser is the side to move (True = White mated => Black won).
            # Board-relative: loser White -> "0-1", loser Black -> "1-0".
            loser_is_white = bool(game.side_to_move())
            return _finish(
                "0-1" if loser_is_white else "1-0",
                reason="checkmate",
                outcome_class="decisive",
            )
        elif plies >= max_plies:
            # Audit F-01: Maximum ply cap reached without a decisive or rule-based draw outcome.
            # Must not be treated as a genuine chess draw by default.
            cap_pol = getattr(self, "cap_policy", "exclude")
            if cap_pol == "draw":
                return _finish("1/2-1/2", reason="max_plies_cap", outcome_class="capped")
            else:
                return _finish("*", reason="max_plies_cap", outcome_class="capped")
        else:
            return _finish("1/2-1/2", reason="rule_draw", outcome_class="genuine_draw")

    def run_match(
        self,
        num_openings: int = 50,
        pgn_path: Optional[Union[str, Path]] = None,
        on_game_result: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Union[int, float]]:
        """Run 2 * num_openings games in parallel across opening pairs with persistent workers.

        Game ``i`` (opening index ``idx``, engine-white first) uses MCTS seed
        ``self.seed + 2*idx`` / ``+ 2*idx+1`` — deterministic per game. If
        ``pgn_path`` is given, every game is appended to it the moment it
        completes (interrupted matches keep their finished games). If
        ``on_game_result`` is given it is invoked with each game dict as soon
        as that game completes, so callers can persist results incrementally.

        Returns the tally plus a ``games`` detail list and ``pgn_path``. On
        interruption (including worker failures) the tally covers exactly the
        completed games, ``interrupted`` is True with the error recorded, and
        ``total_games`` never pretends unplayed games were played.
        """
        selected_openings = self.openings[:num_openings]
        engine_wins = 0
        engine_draws = 0
        engine_losses = 0
        total_games = len(selected_openings) * 2
        elo_str = f"UCI_Elo {self.stockfish_elo}" if self.stockfish_elo else "full strength"
        engine_info = engine_identity(self.stockfish_path)

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
                info_w: dict = {}
                res_w = self._play_game_worker(
                    fen,
                    engine_is_white=True,
                    ctx=worker_ctx,
                    game_seed=self.seed + 2 * idx,
                    move_log=moves_w,
                    info_log=info_w,
                )
                moves_b: list[str] = []
                info_b: dict = {}
                res_b = self._play_game_worker(
                    fen,
                    engine_is_white=False,
                    ctx=worker_ctx,
                    game_seed=self.seed + 2 * idx + 1,
                    move_log=moves_b,
                    info_log=info_b,
                )
                return (
                    {
                        "opening_idx": idx,
                        "opening_fen": fen,
                        "engine_is_white": True,
                        "result": res_w,
                        "moves": moves_w,
                        "seed": self.seed + 2 * idx,
                        # Audit RL-20 & F-01: adjudication and termination provenance.
                        "adjudicated": bool(info_w.get("adjudicated", False)),
                        "adjudication_reason": info_w.get("adjudication_reason"),
                        "outcome_class": info_w.get("outcome_class", "unknown"),
                        "termination_reason": info_w.get("termination_reason", "unknown"),
                        "final_material_balance": info_w.get("final_material_balance", 0),
                        "final_fen": info_w.get("final_fen", ""),
                        "book_moves_count": info_w.get("book_moves_count", 0),
                    },
                    {
                        "opening_idx": idx,
                        "opening_fen": fen,
                        "engine_is_white": False,
                        "result": res_b,
                        "moves": moves_b,
                        "seed": self.seed + 2 * idx + 1,
                        "adjudicated": bool(info_b.get("adjudicated", False)),
                        "adjudication_reason": info_b.get("adjudication_reason"),
                        "outcome_class": info_b.get("outcome_class", "unknown"),
                        "termination_reason": info_b.get("termination_reason", "unknown"),
                        "final_material_balance": info_b.get("final_material_balance", 0),
                        "final_fen": info_b.get("final_fen", ""),
                        "book_moves_count": info_b.get("book_moves_count", 0),
                    },
                )
            finally:
                ctx_queue.put(worker_ctx)

        completed_pairs = 0
        results_as_white: list[str] = []
        results_as_black: list[str] = []
        games: list[dict] = []
        interrupted = False
        interrupt_error: Optional[str] = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(worker_task, (idx, fen))
                    for idx, fen in enumerate(selected_openings)
                ]

                try:
                    for future in concurrent.futures.as_completed(futures):
                        game_w, game_b = future.result()
                        games.extend([game_w, game_b])
                        # Persist each game the moment it completes (Stage A §7):
                        # an interrupted match must not lose finished games.
                        if pgn_path is not None:
                            try:
                                write_match_pgn([game_w], pgn_path)
                                write_match_pgn([game_b], pgn_path)
                            except OSError:
                                logger.exception(
                                    "Incremental PGN write to %s failed; continuing match",
                                    pgn_path,
                                )
                        if on_game_result is not None:
                            for completed_game in (game_w, game_b):
                                try:
                                    on_game_result(completed_game)
                                except Exception:
                                    logger.exception(
                                        "on_game_result callback failed; continuing match"
                                    )
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
                        eff_done = engine_wins + engine_draws + engine_losses
                        score = (
                            (engine_wins + 0.5 * engine_draws) / eff_done
                            if eff_done > 0
                            else 0.0
                        )
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Match Progress: {games_done:3d}/{total_games} games | "
                            f"W: {engine_wins:2d} D: {engine_draws:2d} L: {engine_losses:2d} | "
                            f"Score: {score * 100:5.1f}% (effective N={eff_done}) | "
                            f"Elapsed: {elapsed:5.1f}s ({elapsed / games_done:.1f}s/game)"
                        )
                except (KeyboardInterrupt, Exception) as exc:
                    # Interruption or worker failure: keep every completed game
                    # and never fabricate a full-match result from a partial
                    # sample. The traceback stays in the log; the error text
                    # travels with the result.
                    interrupted = True
                    interrupt_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Match interrupted after %d/%d games; preserving completed results",
                        completed_pairs * 2,
                        total_games,
                    )
                    logger.exception("Match interruption diagnostics")
                    for pending in futures:
                        pending.cancel()
        finally:
            for ctx in contexts:
                ctx.close()

        # Outcome classification & statistical integrity (Audit F-01, F-03, F-16)
        # Shared record classifier (S5-05): one implementation for the match
        # runner and the SPRT harness (scripts/run_sprt.py).
        counts = classify_game_records(games)
        decisive_wins = counts["decisive_wins"]
        decisive_losses = counts["decisive_losses"]
        genuine_draws = counts["genuine_draws"]
        capped_games = counts["capped_games"]
        forfeit_wins = counts["forfeit_wins"]
        forfeit_losses = counts["forfeit_losses"]

        engine_wins = decisive_wins + forfeit_wins
        engine_losses = decisive_losses + forfeit_losses
        engine_draws = genuine_draws
        effective_games = engine_wins + engine_draws + engine_losses

        completed_total = len(games)
        denominator = completed_total if interrupted else total_games
        primary_score = (
            (engine_wins + 0.5 * engine_draws) / effective_games
            if effective_games > 0
            else 0.0
        )
        draw_rate = engine_draws / effective_games if effective_games > 0 else 0.0
        capped_rate = capped_games / completed_total if completed_total > 0 else 0.0

        primary_elo_res = elo_confidence_interval(
            engine_wins, engine_draws, engine_losses, confidence=0.95, method="agresti_caffo"
        )

        # Cluster-adjusted paired analysis across complete opening pairs (Audit F-16)
        pairs_dict: dict[int, dict[str, float]] = {}
        for g in games:
            idx = g["opening_idx"]
            res = g["result"]
            eng_white = g["engine_is_white"]
            if res == "1-0":
                s = 1.0 if eng_white else 0.0
            elif res == "0-1":
                s = 0.0 if eng_white else 1.0
            elif res == "1/2-1/2" and g.get("outcome_class") != "capped":
                s = 0.5
            else:
                s = None  # capped or incomplete
            if s is not None:
                if idx not in pairs_dict:
                    pairs_dict[idx] = {}
                pairs_dict[idx]["white" if eng_white else "black"] = s

        valid_pairs = [
            (p["white"], p["black"])
            for p in pairs_dict.values()
            if "white" in p and "black" in p
        ]
        pairing_res = paired_score_and_ci(valid_pairs, confidence=0.95)

        # Sensitivity analyses (Audit F-01):
        # 1. Capped scored as draw (legacy comparison)
        cd_draws = engine_draws + capped_games
        cd_elo = elo_confidence_interval(engine_wins, cd_draws, engine_losses)

        # 2. Capped scored as loss (worst-case sensitivity)
        cl_losses = engine_losses + capped_games
        cl_elo = elo_confidence_interval(engine_wins, engine_draws, cl_losses)

        # 3. Capped adjudicated by material balance (>= +2 pawns win, <= -2 pawns loss, else draw)
        cm_wins = engine_wins
        cm_losses = engine_losses
        cm_draws = engine_draws
        for g in games:
            if (
                g.get("outcome_class") == "capped"
                or g.get("termination_reason") == "max_plies_cap"
                or g.get("result") == "*"
            ):
                mat = g.get("final_material_balance", 0)
                if mat >= 2:
                    cm_wins += 1
                elif mat <= -2:
                    cm_losses += 1
                else:
                    cm_draws += 1
        cm_elo = elo_confidence_interval(cm_wins, cm_draws, cm_losses)

        games.sort(key=lambda g: (g["opening_idx"], not g["engine_is_white"]))

        pgn_saved: Optional[str] = str(pgn_path) if (pgn_path is not None and games) else None

        return {
            "total_games": denominator if interrupted else total_games,
            "completed_games": completed_total,
            "effective_games": effective_games,
            "engine_wins": engine_wins,
            "engine_draws": engine_draws,
            "engine_losses": engine_losses,
            "capped_games": capped_games,
            "forfeit_wins": forfeit_wins,
            "forfeit_losses": forfeit_losses,
            "engine_score": primary_score,
            "primary_score": primary_score,
            "draw_rate": draw_rate,
            "capped_rate": capped_rate,
            "primary_elo": primary_elo_res,
            "elo_diff": primary_elo_res["elo_diff"],
            "elo_lo": primary_elo_res["lo"],
            "elo_hi": primary_elo_res["hi"],
            "search_mode": self.search_mode,
            "seed": self.seed,
            "stockfish_elo": self.stockfish_elo,
            "engine": engine_info,
            "interrupted": interrupted,
            "error": interrupt_error,
            "games": games,
            "pgn_path": pgn_saved,
            "elapsed_sec": round(time.time() - start_time, 2),
            "adjudicated_games": sum(1 for g in games if g.get("adjudicated")),
            "adjudicated_value_cutoff_wins": sum(
                1 for g in games if g.get("adjudication_reason") == "value_cutoff_win"
            ),
            "adjudicated_value_cutoff_losses": sum(
                1 for g in games if g.get("adjudication_reason") == "value_cutoff_loss"
            ),
            "use_book": self.use_book,
            "book_path": self.book_path,
            "book_sha256": self.book_sha256,
            "book_policy": self.book_policy,
            "book_moves_total": sum(int(g.get("book_moves_count", 0)) for g in games),
            "book_games_total": sum(1 for g in games if int(g.get("book_moves_count", 0)) > 0),
            "pairing_analysis": pairing_res,
            "sensitivity_analysis": {
                "capped_as_draw": cd_elo,
                "capped_as_loss": cl_elo,
                "capped_by_material": cm_elo,
            },
        }

    def close(self) -> None:
        pass
