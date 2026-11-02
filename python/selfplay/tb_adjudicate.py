"""Syzygy tablebase adjudication module for self-play games.

Contract specifications:
- python-chess (chess.syzygy) is used ONLY inside this module, between plies from the worker.
- Never called inside search or C++ code.
- Fast string filter on FEN: piece count <= max_pieces and castling == "-".
- python-chess board constructed and probed ONLY when the filter passes.
- Strict board validity check (board.is_valid()) rejects impossible/corrupt states (SY-06).
- Thread-safe probing with lock wait-time telemetry.
- Resource management: singleton path changes and explicit close release file handles (SY-05).
- Result mapping:
    Win  (+2 in syzygy, halfmove_clock == 0) -> +1 (decisive win for side to move)
    Loss (-2 in syzygy, halfmove_clock == 0) -> -1 (decisive loss for side to move)
    Decisive (+2 / -2) with halfmove_clock > 0 -> None (ineligible decisive; no fabricated draw) (SY-01)
    Draw ( 0), Cursed Win (+1), Blessed Loss (-1) -> 0 (draw under FIDE rule-50)
- Missing table or ineligible state falls through (returns None) and is counted.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import chess
import chess.syzygy

logger = logging.getLogger(__name__)


class TablebaseAdjudicator:
    """Thread-safe singleton tablebase adjudicator per worker process."""

    _instance: Optional["TablebaseAdjudicator"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(self, tablebase_path: Optional[str] = None, max_pieces: int = 5) -> None:
        if tablebase_path:
            p = Path(tablebase_path)
            if not p.is_absolute() and not p.is_dir():
                candidate = Path(__file__).resolve().parent.parent.parent / p
                if candidate.is_dir():
                    tablebase_path = str(candidate)
        self.tablebase_path = tablebase_path
        self.max_pieces = max_pieces
        self.tablebase: Optional[chess.syzygy.Tablebase] = None
        self._probe_lock: threading.Lock = threading.Lock()
        # Audit S4-04: telemetry counters are updated from up to 24 game
        # threads; CPython ``+=`` on an attribute is a non-atomic
        # read-modify-write, so cumulative counters need their own lock. This
        # is deliberately NOT the probe lock: the fast filter must stay
        # lock-free, only the counter updates serialize (µs-class hold).
        self._telemetry_lock: threading.Lock = threading.Lock()

        # Telemetry counters
        self.filter_checks: int = 0
        self.filter_passed: int = 0
        self.probes_total: int = 0
        self.probes_decisive: int = 0
        self.probes_draw: int = 0
        self.probes_halfmove_ineligible: int = 0
        self.invalid_positions: int = 0
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

    def close(self) -> None:
        """Close opened tablebase file handles cleanly (SY-05)."""
        with self._probe_lock:
            if self.tablebase is not None:
                try:
                    self.tablebase.close()
                except Exception as e:
                    logger.debug("Error closing tablebase: %s", e)
                self.tablebase = None

    def __enter__(self) -> "TablebaseAdjudicator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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
                # Update tablebase path if changed: safely close old handles to prevent leaks (SY-05)
                if cls._instance is not None:
                    cls._instance.close()
                cls._instance = cls(tablebase_path=tablebase_path, max_pieces=max_pieces)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful for testing)."""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance.close()
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
                +1 if side to move has decisive win (halfmove_clock == 0)
                -1 if side to move has decisive loss (halfmove_clock == 0)
                 0 if draw (including cursed-win / blessed-loss under rule 50)
                None if position doesn't qualify, tablebase disabled, missing table,
                     invalid board state, or decisive position with halfmove_clock > 0.
            metadata: dict with timing and telemetry.
        """
        if self.tablebase is None:
            return None, {"qualified": False, "reason": "no_tablebase"}

        t0 = time.perf_counter()
        qualifies, pc = self.fast_filter(fen, self.max_pieces)
        t_filter = time.perf_counter() - t0

        # Audit S4-04: counter updates hold only the dedicated telemetry lock
        # (never the probe lock), so the fast filter stays fast and the
        # cumulative counters no longer lose updates under concurrency.
        with self._telemetry_lock:
            self.filter_checks += 1
            self.total_filter_time_sec += t_filter

        if not qualifies:
            return None, {"qualified": False, "piece_count": pc, "filter_time_us": t_filter * 1e6}

        with self._telemetry_lock:
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
            except Exception as e:
                self.invalid_positions += 1
                t_probe = time.perf_counter() - t_probe_start
                self.total_probe_time_sec += t_probe
                return None, {
                    "qualified": True,
                    "invalid": True,
                    "error": f"malformed_fen: {e}",
                    "piece_count": pc,
                    "filter_time_us": t_filter * 1e6,
                    "probe_time_us": t_probe * 1e6,
                    "lock_wait_us": lock_wait * 1e6,
                }

            # SY-06: Strict board validity check
            if not board.is_valid():
                self.invalid_positions += 1
                t_probe = time.perf_counter() - t_probe_start
                self.total_probe_time_sec += t_probe
                return None, {
                    "qualified": True,
                    "invalid": True,
                    "error": "invalid_board_state",
                    "status": int(board.status()),
                    "piece_count": pc,
                    "filter_time_us": t_filter * 1e6,
                    "probe_time_us": t_probe * 1e6,
                    "lock_wait_us": lock_wait * 1e6,
                }

            try:
                # Probe WDL
                # Syzygy WDL returns:
                #  2: win
                #  1: cursed win (win requires >50 moves without capture/pawn move; drawn under rule 50)
                #  0: draw
                # -1: blessed loss (loss defended by 50-move rule; drawn under rule 50)
                # -2: loss
                raw_wdl = self.tablebase.probe_wdl(board)
                self.probes_total += 1
                halfmove_clock = board.halfmove_clock

                # Rule-50 safeguard (SY-01):
                # Decisive adjudication (+1/-1) is strictly valid when halfmove_clock == 0.
                # If halfmove_clock > 0, decisive outcome cannot be guaranteed before the 50-move limit
                # without DTZ tables. Ineligible decisive positions return None (do not fabricate draw).
                if raw_wdl in (2, -2):
                    if halfmove_clock == 0:
                        outcome = 1 if raw_wdl == 2 else -1
                        self.probes_decisive += 1
                    else:
                        self.probes_halfmove_ineligible += 1
                        t_probe = time.perf_counter() - t_probe_start
                        self.total_probe_time_sec += t_probe
                        return None, {
                            "qualified": True,
                            "piece_count": pc,
                            "raw_wdl": raw_wdl,
                            "halfmove_ineligible": True,
                            "halfmove_clock": halfmove_clock,
                            "filter_time_us": t_filter * 1e6,
                            "probe_time_us": t_probe * 1e6,
                            "lock_wait_us": lock_wait * 1e6,
                        }
                else:
                    # raw_wdl in (1, 0, -1):
                    # 0 is a theoretical draw.
                    # +1 (cursed win) / -1 (blessed loss) cannot be won within 50 moves even with clock 0,
                    # so under FIDE rules they are draws regardless of halfmove clock.
                    outcome = 0
                    self.probes_draw += 1

                t_probe = time.perf_counter() - t_probe_start
                self.total_probe_time_sec += t_probe

                return outcome, {
                    "qualified": True,
                    "piece_count": pc,
                    "raw_wdl": raw_wdl,
                    "outcome": outcome,
                    "halfmove_clock": halfmove_clock,
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
            "probes_halfmove_ineligible": self.probes_halfmove_ineligible,
            "invalid_positions": self.invalid_positions,
            "missing_tables": self.missing_tables,
            "avg_filter_time_us": avg_filter_us,
            "avg_probe_time_us": avg_probe_us,
            "avg_lock_wait_us": avg_lock_wait_us,
            "total_lock_wait_sec": self.total_lock_wait_sec,
        }
