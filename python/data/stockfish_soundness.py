"""Stockfish Soundness Evaluation for Candidate Opening Moves.

Evaluates candidate moves with normalized perspective:
- Position P: side to move is M.
- Candidate move m produces child position P' where opponent O is to move.
- When Stockfish evaluates child P', Stockfish returns evaluation from O's perspective.
- Moving player's evaluation is:
    score_mover = -score_child
- Mate scores:
    If child position P' has mate in +k (O mates M): score_mover = -10000.0 (mover lost)
    If child position P' has mate in -k (M mates O): score_mover = +10000.0 (mover won)
- Soundness threshold:
    score_mover >= -30.0 centipawns (-0.30 pawns)

Includes a disk-backed resumable cache to eliminate redundant engine queries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import chess
from python.data.stockfish_eval import DEFAULT_STOCKFISH_PATH, StockfishEvaluator

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/openings/stockfish_eval_cache.json")


class StockfishSoundnessEvaluator:
    """Evaluates opening candidate moves against the >= -0.30 pawn soundness criterion."""

    def __init__(
        self,
        stockfish_path: Union[str, Path] = DEFAULT_STOCKFISH_PATH,
        soundness_threshold_cp: float = -30.0,
        nodes: int = 25000,
        threads: int = 2,
        hash_mb: int = 32,
        cache_path: Optional[Union[str, Path]] = DEFAULT_CACHE_PATH,
    ) -> None:
        self.stockfish_path = Path(stockfish_path)
        self.soundness_threshold_cp = soundness_threshold_cp
        self.nodes = nodes
        self.threads = threads
        self.hash_mb = hash_mb
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

        self._evaluator: Optional[StockfishEvaluator] = None

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.is_file():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(
                    "Loaded %d evaluations from Stockfish cache at %s",
                    len(self.cache),
                    self.cache_path,
                )
            except Exception as e:
                logger.warning("Could not load Stockfish cache: %s", e)
                self.cache = {}

    def _save_cache(self) -> None:
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                logger.warning("Failed to save Stockfish cache: %s", e)

    def _get_evaluator(self) -> StockfishEvaluator:
        if self._evaluator is None:
            self._evaluator = StockfishEvaluator(
                stockfish_path=self.stockfish_path,
                threads=self.threads,
                hash_mb=self.hash_mb,
            )
        return self._evaluator

    def evaluate_candidate_move(
        self,
        parent_board: chess.Board,
        move: chess.Move,
        nodes: Optional[int] = None,
        timeout: float = 15.0,
    ) -> Tuple[Optional[float], Optional[int], bool]:
        """Evaluate a candidate move from parent_board.

        Returns:
            (score_mover_cp, mate_dist_for_mover, is_sound)
            If Stockfish evaluation fails (timeout/crash/no bestmove),
            score_mover_cp is None, mate_dist_for_mover is None, and
            is_sound is False. Caching is skipped for failed evaluations.
        """
        if move not in parent_board.legal_moves:
            raise ValueError(f"Move {move.uci()} is not legal in {parent_board.fen()}")

        nodes_budget = nodes or self.nodes
        cache_key = f"{parent_board.epd(en_passant='pawns')}_{move.uci()}_{nodes_budget}"

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            cp = entry["cp"]
            mate = entry.get("mate")
            sound = entry["sound"]
            return cp, mate, sound

        # Construct child board
        child = parent_board.copy(stack=False)
        child.push(move)
        child_fen = child.fen()

        # If child board is checkmate, mover wins immediately!
        if child.is_checkmate():
            cp = 10000.0
            mate = 1
            sound = True
            self._cache_result(cache_key, cp, mate, sound)
            return cp, mate, sound

        if child.is_stalemate() or child.is_insufficient_material():
            cp = 0.0
            mate = None
            sound = cp >= self.soundness_threshold_cp
            self._cache_result(cache_key, cp, mate, sound)
            return cp, mate, sound

        # Query Stockfish on child position
        evaluator = self._get_evaluator()
        # Flush any stale output (e.g., leftover bestmove from timeout) before issuing new commands
        evaluator.flush_queue()
        evaluator._send_cmd(f"position fen {child_fen}")
        evaluator._send_cmd(f"go nodes {nodes_budget}")

        child_cp: Optional[float] = None
        child_mate: Optional[int] = None
        bestmove_received = False
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            line_str = evaluator._readline(timeout=max(0.1, timeout - (time.monotonic() - start)))
            if line_str is None:
                break
            if "score cp" in line_str:
                parts = line_str.split()
                try:
                    idx = parts.index("cp")
                    child_cp = float(parts[idx + 1])
                    child_mate = None
                except (ValueError, IndexError):
                    pass
            elif "score mate" in line_str:
                parts = line_str.split()
                try:
                    idx = parts.index("mate")
                    child_mate = int(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            if line_str.startswith("bestmove"):
                bestmove_received = True
                break

        # If evaluation failed (timeout/crash/no bestmove), do NOT default to 0.0.
        # Set failure flag, skip caching, and return a sound=False verdict.
        if not bestmove_received:
            return None, None, False

        # Convert perspective: mover score is -child score!
        if child_mate is not None:
            # If child has mate in +k, opponent mates mover: mover mate is -k!
            # If child has mate in -k, mover mates opponent: mover mate is +k!
            mate_for_mover = -child_mate
            score_mover_cp = 10000.0 if mate_for_mover > 0 else -10000.0
            is_sound = score_mover_cp >= self.soundness_threshold_cp
        else:
            if child_cp is None:
                # Defensive: if no score line was received despite bestmove,
                # treat as failure rather than defaulting to -0.0.
                return None, None, False
            mate_for_mover = None
            score_mover_cp = -child_cp
            is_sound = score_mover_cp >= self.soundness_threshold_cp

        self._cache_result(cache_key, score_mover_cp, mate_for_mover, is_sound)
        return score_mover_cp, mate_for_mover, is_sound

    def _cache_result(self, key: str, cp: float, mate: Optional[int], sound: bool) -> None:
        self.cache[key] = {
            "cp": cp,
            "mate": mate,
            "sound": sound,
            "timestamp": time.time(),
        }
        # Periodically persist cache
        if len(self.cache) % 10 == 0:
            self._save_cache()

    def evaluate_corpus_positions(
        self,
        corpus: Any,
        max_positions: Optional[int] = None,
        max_moves_per_pos: int = 5,
        nodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate unevaluated candidate moves in the opening corpus."""
        evaluated_moves = 0
        sound_moves_found = 0
        positions_processed = 0

        for key, pos in corpus.positions.items():
            if max_positions is not None and positions_processed >= max_positions:
                break

            board = chess.Board(pos.fen)
            moves_to_eval = list(pos.moves.keys())[:max_moves_per_pos]

            for uci in moves_to_eval:
                ev = pos.moves[uci]
                if ev.stockfish_evaluated:
                    continue

                try:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        cp, mate, is_sound = self.evaluate_candidate_move(board, move, nodes=nodes)
                        ev.stockfish_cp = cp
                        ev.stockfish_mate = mate
                        ev.stockfish_evaluated = True
                        evaluated_moves += 1
                        if is_sound:
                            sound_moves_found += 1
                except Exception as e:
                    logger.warning("Failed to evaluate %s in %s: %s", uci, pos.fen, e)

            positions_processed += 1

        self._save_cache()
        corpus.update_qualification_status()

        return {
            "positions_processed": positions_processed,
            "moves_evaluated": evaluated_moves,
            "sound_moves_found": sound_moves_found,
            "cache_size": len(self.cache),
        }

    def close(self) -> None:
        self._save_cache()
        if self._evaluator is not None:
            self._evaluator.close()
            self._evaluator = None

    def __enter__(self) -> StockfishSoundnessEvaluator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
