"""Runtime Opening Book Implementation for ZeroPawn.

Provides fast, in-memory opening book lookups with configurable selection policies:
- `frequency`: Sample proportionally to master-game frequency.
- `score`: Sample or select according to Stockfish soundness evaluations.
- `weighted_blend`: Balanced combination of frequency and engine score.
- `uniform`: Uniform random selection among all qualified book moves.
- `best`: Deterministic selection of the top-ranked qualified move.

Invariants:
- Selected move is strictly validated against current legal moves.
- Transpositions with identical board states resolve to the same canonical book entry.
- Clean fallback to MCTS / neural search when position is outside book or beyond max depth.
- Reproducible seeded random generator for matches and tournaments.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess

logger = logging.getLogger(__name__)

DEFAULT_BOOK_PATH = Path("data/openings/curated_book.json")

# In-memory cache of parsed (positions, metadata) keyed by (resolved_path, mtime)
# Prevents duplicate disk reads and redundant log spam across concurrent workers
_BOOK_CACHE: Dict[Tuple[str, float], Tuple[Dict[str, Any], Dict[str, Any]]] = {}


class OpeningBook:
    """In-memory opening book index with configurable move selection."""

    def __init__(
        self,
        book_path: Union[str, Path] = DEFAULT_BOOK_PATH,
        policy: str = "frequency",
        max_depth_plies: int = 24,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        self.book_path = Path(book_path)
        if not self.book_path.is_file() and not self.book_path.is_absolute():
            try:
                from python.cli.common import REPO_ROOT

                if (REPO_ROOT / self.book_path).is_file():
                    self.book_path = REPO_ROOT / self.book_path
            except Exception:
                pass
        self.policy = policy.lower()
        self.max_depth_plies = max_depth_plies
        self.temperature = max(1e-4, float(temperature))
        self.enabled = enabled
        self.rng = random.Random(seed)

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        if self.book_path.is_file():
            self.load(self.book_path)
        else:
            logger.debug(
                "Opening book file not found at %s (book disabled until loaded)", self.book_path
            )

    def load(self, book_path: Union[str, Path]) -> None:
        """Load curated opening book from JSON file.

        Supports both the `positions` schema (canonical position -> moves dict)
        and the `lines` / line-catalog schema (list of opening sequences indexed
        by replaying from the initial position).
        """
        self.book_path = Path(book_path)
        if not self.book_path.is_file() and not self.book_path.is_absolute():
            try:
                from python.cli.common import REPO_ROOT

                if (REPO_ROOT / self.book_path).is_file():
                    self.book_path = REPO_ROOT / self.book_path
            except Exception:
                pass
        if self.book_path.is_file():
            try:
                cache_key = (str(self.book_path.resolve()), self.book_path.stat().st_mtime)
                if cache_key in _BOOK_CACHE:
                    self.positions, self.metadata = _BOOK_CACHE[cache_key]
                    return
            except Exception:
                cache_key = None
        else:
            cache_key = None

        with open(self.book_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalise top-level list to a dict with a `lines` key.
        if isinstance(data, list):
            data = {"lines": data}

        if "lines" in data:
            # Build in-memory index from line-catalog by replaying each line
            # from the initial board position.
            self.positions = {}
            for item in data.get("lines", []):
                uci_moves = item.get("uci", [])
                eco = item.get("eco")
                name = item.get("name")
                count = item.get("count", 1)
                board = chess.Board()
                for ply, uci_str in enumerate(uci_moves):
                    try:
                        move = chess.Move.from_uci(uci_str)
                    except Exception:
                        # Skip invalid UCI strings in the catalog.
                        break
                    key = board.epd(en_passant="pawns")
                    if key not in self.positions:
                        self.positions[key] = {
                            "canonical_key": key,
                            "fen": board.fen(),
                            "side_to_move": bool(board.turn == chess.WHITE),
                            "ply": ply,
                            "eco": eco,
                            "opening_name": name,
                            "moves": {},
                        }
                    # Preserve or accumulate frequency across lines.
                    moves_dict = self.positions[key]["moves"]
                    existing = moves_dict.get(uci_str)
                    if existing is None:
                        moves_dict[uci_str] = {
                            "uci": uci_str,
                            "frequency": int(count),
                            "stockfish_cp": item.get("stockfish_cp"),
                            "qualification": "frequency",
                        }
                    else:
                        existing["frequency"] = max(existing.get("frequency", 0), int(count))
                        if item.get("stockfish_cp") is not None:
                            existing["stockfish_cp"] = item["stockfish_cp"]
                    board.push(move)
            self.metadata = data.get("summary", {})
        elif "positions" in data:
            self.metadata = data.get("summary", {})
            self.positions = data.get("positions", {})
        else:
            # Unknown schema — attempt to interpret dict keys as position keys
            # if they contain FEN-like strings.
            self.metadata = data.get("summary", {})
            self.positions = data.get("positions", {})

        if cache_key is not None:
            _BOOK_CACHE[cache_key] = (self.positions, self.metadata)

        logger.info(
            "Loaded OpeningBook from %s (%d positions indexed)",
            self.book_path,
            len(self.positions),
        )

    def is_in_book(self, board_or_fen: Union[chess.Board, str, Any]) -> bool:
        """Check whether the given position exists in the book."""
        if not self.enabled:
            return False
        key = self._canonical_key(board_or_fen)
        return key in self.positions and bool(self.positions[key].get("moves"))

    def _canonical_key(self, board_or_fen: Union[chess.Board, str, Any]) -> str:
        """Extract canonical position key from chess.Board, FEN string, or C++ ChessGame."""
        if isinstance(board_or_fen, chess.Board):
            return board_or_fen.epd(en_passant="pawns")
        if isinstance(board_or_fen, str):
            board = chess.Board(board_or_fen)
            return board.epd(en_passant="pawns")
        if hasattr(board_or_fen, "get_fen"):
            board = chess.Board(board_or_fen.get_fen())
            return board.epd(en_passant="pawns")
        raise TypeError(f"Unsupported board representation: {type(board_or_fen)}")

    def _get_board_and_ply(
        self, board_or_fen: Union[chess.Board, str, Any]
    ) -> Tuple[chess.Board, int]:
        """Convert input to a python-chess Board and determine current ply count."""
        if isinstance(board_or_fen, chess.Board):
            return board_or_fen, board_or_fen.ply()
        if isinstance(board_or_fen, str):
            board = chess.Board(board_or_fen)
            return board, board.ply()
        if hasattr(board_or_fen, "get_fen"):
            board = chess.Board(board_or_fen.get_fen())
            # For C++ ChessGame, attempt history length if available
            ply = getattr(board_or_fen, "history_len", lambda: board.ply())()
            return board, ply
        raise TypeError(f"Unsupported board representation: {type(board_or_fen)}")

    def get_move(self, board_or_fen: Union[chess.Board, str, Any]) -> Optional[chess.Move]:
        """Look up opening move and return a python-chess Move (or None)."""
        res = self.lookup(board_or_fen)
        if res is None:
            return None
        chosen_uci, _ = res
        return chess.Move.from_uci(chosen_uci)

    def lookup(
        self,
        board_or_fen: Union[chess.Board, str, Any],
        current_ply: Optional[int] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Look up opening move for position.

        Returns:
            (best_move_uci, metadata) or None if no book move qualifies.
        """
        if not self.enabled or not self.positions:
            return None

        board, ply = self._get_board_and_ply(board_or_fen)
        ply_to_check = current_ply if current_ply is not None else ply
        if ply_to_check >= self.max_depth_plies:
            return None

        key = board.epd(en_passant="pawns")
        if key not in self.positions:
            return None

        pos_data = self.positions[key]
        moves_dict = pos_data.get("moves", {})
        if not moves_dict:
            return None

        # Filter strictly for currently legal moves
        legal_ucis = {m.uci(): m for m in board.legal_moves}
        candidate_moves: List[Tuple[str, Dict[str, Any]]] = []

        for uci, ev in moves_dict.items():
            if uci in legal_ucis:
                candidate_moves.append((uci, ev))

        if not candidate_moves:
            return None

        # Apply selection policy
        chosen_uci, chosen_meta = self._select_move(candidate_moves)

        result_meta = {
            "source": "opening_book",
            "policy": self.policy,
            "eco": pos_data.get("eco"),
            "opening_name": pos_data.get("opening_name"),
            "candidate_count": len(candidate_moves),
            "move_metadata": chosen_meta,
        }
        return chosen_uci, result_meta

    def _select_move(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Select a move from legal candidates according to self.policy."""
        if len(candidates) == 1:
            return candidates[0]

        policy = self.policy

        if policy == "best":
            # Highest frequency, tiebreak by stockfish_cp
            candidates_sorted = sorted(
                candidates,
                key=lambda x: (
                    x[1].get("frequency", 0),
                    x[1].get("stockfish_cp") if x[1].get("stockfish_cp") is not None else -999.0,
                ),
                reverse=True,
            )
            return candidates_sorted[0]

        if policy == "uniform":
            return self.rng.choice(candidates)

        if policy == "score":
            # Softmax over Stockfish scores
            scores = []
            for _, ev in candidates:
                cp = ev.get("stockfish_cp")
                if cp is None:
                    cp = 0.0
                scores.append(cp / (100.0 * self.temperature))
            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) for s in scores]
            total_exp = sum(exp_scores)
            weights = [s / total_exp for s in exp_scores]
            return self.rng.choices(candidates, weights=weights, k=1)[0]

        if policy == "weighted_blend":
            # 50% frequency weight, 50% score weight
            freqs = [float(ev.get("frequency", 1)) for _, ev in candidates]
            tot_freq = sum(freqs) or 1.0
            norm_freqs = [f / tot_freq for f in freqs]

            scores = [float(ev.get("stockfish_cp", 0.0) or 0.0) for _, ev in candidates]
            max_s = max(scores)
            exp_scores = [math.exp((s - max_s) / (100.0 * self.temperature)) for s in scores]
            tot_exp = sum(exp_scores) or 1.0
            norm_scores = [s / tot_exp for s in exp_scores]

            blended_weights = [0.5 * nf + 0.5 * ns for nf, ns in zip(norm_freqs, norm_scores)]
            return self.rng.choices(candidates, weights=blended_weights, k=1)[0]

        # Default: frequency-weighted selection
        freqs = [float(ev.get("frequency", 1)) for _, ev in candidates]
        tot_f = sum(freqs)
        if tot_f <= 0:
            return self.rng.choice(candidates)
        weights = [f / tot_f for f in freqs]
        return self.rng.choices(candidates, weights=weights, k=1)[0]
