"""Syzygy tablebase adjudication module for self-play games.

Contract specifications:
- python-chess (chess.syzygy) is used ONLY inside this module, between plies from the worker.
- Never called inside search or C++ code.
- Fast string filter on FEN: piece count <= max_pieces and castling == "-".
- python-chess board constructed and probed ONLY when the filter passes.
- Thread-safe probing with lock wait-time telemetry.
- Result mapping:
    Win  (+2 in syzygy) -> +1 (decisive win for side to move)
    Loss (-2 in syzygy) -> -1 (decisive loss for side to move)
    Draw ( 0), Cursed Win (+1), Blessed Loss (-1) -> 0 (draw)
- Missing table falls through (returns None) and is counted.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import chess
import chess.syzygy

logger = logging.getLogger(__name__)


class TablebaseAdjudicator:
    """Thread-safe singleton tablebase adjudicator per worker process."""

    _instance: Optional["TablebaseAdjudicator"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(self, tablebase_path: Optional[str] = None, max_pieces: int = 5) -> None:
        self.tablebase_path = tablebase_path
        self.max_pieces = max_pieces
        self.tablebase: Optional[chess.syzygy.Tablebase] = None
        self._probe_lock: threading.Lock = threading.Lock()

        # Telemetry counters
        self.filter_checks: int = 0
        self.filter_passed: int = 0
        self.probes_total: int = 0
        self.probes_decisive: int = 0
        self.probes_draw: int = 0
        self.missing_tables: int = 0

        # Timing telemetry (cumulative seconds)
        self.total_filter_time_sec: float = 0.0
        self.total_probe_time_sec: float = 0.0
        self.total_lock_wait_sec: float = 0.0

        if tablebase_path and os.path.isdir(tablebase_path):
            try:
                self.tablebase = chess.syzygy.open_tablebase(tablebase_path)
                logger.info(
                    "Loaded Syzygy tablebases from %s (max_pieces=%d)", tablebase_path, max_pieces
                )
            except Exception as e:
                logger.error("Failed to open tablebase at %s: %s", tablebase_path, e)
                self.tablebase = None

    @classmethod
    def get_instance(
        cls,
        tablebase_path: Optional[str] = None,
        max_pieces: int = 5,
    ) -> "TablebaseAdjudicator":
        """Get or initialize process-wide singleton."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(tablebase_path=tablebase_path, max_pieces=max_pieces)
            elif tablebase_path is not None and cls._instance.tablebase_path != tablebase_path:
                # Update tablebase path if changed
                cls._instance = cls(tablebase_path=tablebase_path, max_pieces=max_pieces)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful for testing)."""
        with cls._init_lock:
            if cls._instance is not None and cls._instance.tablebase is not None:
                try:
                    cls._instance.tablebase.close()
                except Exception:
                    pass
            cls._instance = None

    @staticmethod
    def fast_filter(fen: str, max_pieces: int = 5) -> Tuple[bool, int]:
        """Fast Python string filter on raw FEN string.

        Checks:
        1. Piece count in placement field <= max_pieces.
        2. Castling field == "-".

        Returns: (passed, piece_count)
        """
        # FEN format: "<placement> <stm> <castling> <ep> <halfmove> <fullmove>"
        parts = fen.split()
        if len(parts) < 3:
            return False, 32

        # 1. Castling rights must be "-"
        has_no_castling = parts[2] == "-"

        # 2. Count piece letters in placement field
        placement = parts[0]
        pc = sum(1 for ch in placement if ch.isalpha())

        qualifies = has_no_castling and (pc <= max_pieces)
        return qualifies, pc

    def probe(self, fen: str) -> Tuple[Optional[int], Dict[str, Any]]:
        """Probe tablebase for position FEN if it qualifies.

        Returns:
            outcome:
                +1 if side to move has decisive win
                -1 if side to move has decisive loss
                 0 if draw (including cursed-win / blessed-loss)
                None if position doesn't qualify, tablebase disabled, or missing table.
            metadata: dict with timing and telemetry.
        """
        if self.tablebase is None:
            return None, {"qualified": False, "reason": "no_tablebase"}

        t0 = time.perf_counter()
        qualifies, pc = self.fast_filter(fen, self.max_pieces)
        t_filter = time.perf_counter() - t0

        self.filter_checks += 1
        self.total_filter_time_sec += t_filter

        if not qualifies:
            return None, {"qualified": False, "piece_count": pc, "filter_time_us": t_filter * 1e6}

        self.filter_passed += 1

        # Construct python-chess board ONLY after filter passes
        t_lock_start = time.perf_counter()
        with self._probe_lock:
            t_lock_acquired = time.perf_counter()
            lock_wait = t_lock_acquired - t_lock_start
            self.total_lock_wait_sec += lock_wait

            t_probe_start = time.perf_counter()
            try:
                board = chess.Board(fen)
                # Probe WDL
                # Syzygy WDL returns:
                #  2: win
                #  1: cursed win (win within 50 moves? Drawn by 50-move rule)
                #  0: draw
                # -1: blessed loss (loss saved by 50-move rule)
                # -2: loss
                raw_wdl = self.tablebase.probe_wdl(board)
                self.probes_total += 1

                if raw_wdl == 2:
                    outcome = 1
                    self.probes_decisive += 1
                elif raw_wdl == -2:
                    outcome = -1
                    self.probes_decisive += 1
                else:
                    outcome = 0
                    self.probes_draw += 1

                t_probe = time.perf_counter() - t_probe_start
                self.total_probe_time_sec += t_probe

                return outcome, {
                    "qualified": True,
                    "piece_count": pc,
                    "raw_wdl": raw_wdl,
                    "outcome": outcome,
                    "filter_time_us": t_filter * 1e6,
                    "probe_time_us": t_probe * 1e6,
                    "lock_wait_us": lock_wait * 1e6,
                }

            except (KeyError, chess.syzygy.MissingTableError):
                self.missing_tables += 1
                t_probe = time.perf_counter() - t_probe_start
                self.total_probe_time_sec += t_probe
                return None, {
                    "qualified": True,
                    "piece_count": pc,
                    "missing_table": True,
                    "filter_time_us": t_filter * 1e6,
                    "probe_time_us": t_probe * 1e6,
                    "lock_wait_us": lock_wait * 1e6,
                }
            except Exception as e:
                logger.warning("Tablebase probe unexpected error for FEN %s: %s", fen, e)
                return None, {"qualified": True, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative performance and counter telemetry."""
        avg_filter_us = (
            (self.total_filter_time_sec / self.filter_checks * 1e6)
            if self.filter_checks > 0
            else 0.0
        )
        avg_probe_us = (
            (self.total_probe_time_sec / self.probes_total * 1e6) if self.probes_total > 0 else 0.0
        )
        avg_lock_wait_us = (
            (self.total_lock_wait_sec / self.probes_total * 1e6) if self.probes_total > 0 else 0.0
        )

        return {
            "filter_checks": self.filter_checks,
            "filter_passed": self.filter_passed,
            "probes_total": self.probes_total,
            "probes_decisive": self.probes_decisive,
            "probes_draw": self.probes_draw,
            "missing_tables": self.missing_tables,
            "avg_filter_time_us": avg_filter_us,
            "avg_probe_time_us": avg_probe_us,
            "avg_lock_wait_us": avg_lock_wait_us,
            "total_lock_wait_sec": self.total_lock_wait_sec,
        }
