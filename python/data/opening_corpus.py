"""Unified Opening Data Ingestion, Canonicalization, and Corpus Pipeline.

Ingests, validates, normalizes, deduplicates, and indexes opening data from:
1. EPD files (e.g. data/openings/eval_openings_50.epd, book_selfplay.epd)
2. JSON opening catalogs (e.g. book_selfplay.json, eco_openings.json)
3. PGN game archives (e.g. data/elite_2500.pgn, master collections)
4. Raw FEN lists and SAN/UCI move sequences.

Canonical Representation:
- Canonical position key is generated using `board.epd(en_passant="pawns")`,
  ensuring transpositions with identical piece placement, side to move,
  castling rights, and valid en-passant square map to the same key.
- Tracks:
  - distinct games
  - unique canonical positions
  - unique position-move pairs
  - total occurrences across sources
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import chess
import chess.pgn

logger = logging.getLogger(__name__)


def canonical_position_key(board: chess.Board) -> str:
    """Compute normalized canonical key for board state.

    Includes piece placement, side to move, castling rights, and en-passant
    availability (only if an en-passant capture is legally possible).
    """
    return board.epd(en_passant="pawns")


@dataclass
class MoveEvidence:
    """Evidence record for a specific move in a canonical position."""

    uci: str
    san: str
    frequency: int = 0
    distinct_game_ids: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    stockfish_cp: Optional[float] = None
    stockfish_mate: Optional[int] = None
    stockfish_evaluated: bool = False
    qualification: str = "pending"  # "frequency", "stockfish", "both", "rejected", "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uci": self.uci,
            "san": self.san,
            "frequency": self.frequency,
            "distinct_games_count": len(self.distinct_game_ids),
            "sources": sorted(list(self.sources)),
            "stockfish_cp": self.stockfish_cp,
            "stockfish_mate": self.stockfish_mate,
            "stockfish_evaluated": self.stockfish_evaluated,
            "qualification": self.qualification,
        }


@dataclass
class PositionRecord:
    """Normalized position record in the opening corpus."""

    canonical_key: str
    fen: str
    side_to_move: bool  # True = White, False = Black
    ply: int = 0
    moves: Dict[str, MoveEvidence] = field(default_factory=dict)
    eco: Optional[str] = None
    opening_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "ply": self.ply,
            "eco": self.eco,
            "opening_name": self.opening_name,
            "moves": {uci: ev.to_dict() for uci, ev in self.moves.items()},
        }


class OpeningCorpus:
    """Curated corpus of opening positions, candidate moves, and evidence."""

    def __init__(
        self, min_game_frequency: int = 5, sf_soundness_threshold_cp: float = -30.0
    ) -> None:
        self.min_game_frequency = min_game_frequency
        self.sf_soundness_threshold_cp = sf_soundness_threshold_cp
        self.positions: Dict[str, PositionRecord] = {}

        # Statistics & telemetry
        self.records_processed: int = 0
        self.records_accepted: int = 0
        self.records_rejected: int = 0
        self.validation_errors: List[str] = []
        self.quarantined_items: List[Dict[str, Any]] = []
        self.sources_manifest: Dict[str, Dict[str, Any]] = {}

    def get_or_create_position(
        self,
        board: chess.Board,
        ply: int = 0,
        eco: Optional[str] = None,
        name: Optional[str] = None,
    ) -> PositionRecord:
        """Retrieve existing PositionRecord or initialize a new canonical record."""
        key = canonical_position_key(board)
        if key not in self.positions:
            self.positions[key] = PositionRecord(
                canonical_key=key,
                fen=board.fen(),
                side_to_move=bool(board.turn == chess.WHITE),
                ply=ply,
                eco=eco,
                opening_name=name,
            )
        pos = self.positions[key]
        if eco and not pos.eco:
            pos.eco = eco
        if name and not pos.opening_name:
            pos.opening_name = name
        return pos

    def add_position_move(
        self,
        board: chess.Board,
        move: chess.Move,
        game_id: Optional[str] = None,
        source: str = "unknown",
        ply: int = 0,
        eco: Optional[str] = None,
        name: Optional[str] = None,
    ) -> bool:
        """Add a legal move observation to the canonical corpus."""
        if move not in board.legal_moves:
            err = f"Illegal move {move.uci()} in position {board.fen()}"
            self.validation_errors.append(err)
            self.quarantined_items.append(
                {"source": source, "fen": board.fen(), "move": move.uci(), "reason": err}
            )
            self.records_rejected += 1
            return False

        pos = self.get_or_create_position(board, ply=ply, eco=eco, name=name)
        uci = move.uci()
        if uci not in pos.moves:
            try:
                san = board.san(move)
            except Exception:
                san = uci
            pos.moves[uci] = MoveEvidence(uci=uci, san=san)

        ev = pos.moves[uci]
        ev.frequency += 1
        ev.sources.add(source)
        if game_id:
            ev.distinct_game_ids.add(game_id)
        else:
            # Fallback game id per occurrence if not supplied
            ev.distinct_game_ids.add(f"{source}_{pos.canonical_key}_{ev.frequency}")

        self.records_accepted += 1
        return True

    def ingest_epd(self, epd_path: Union[str, Path], source_name: str = "epd") -> int:
        """Ingest positions from an EPD file."""
        epd_path = Path(epd_path)
        if not epd_path.is_file():
            raise FileNotFoundError(f"EPD file not found: {epd_path}")

        count = 0
        with open(epd_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                self.records_processed += 1
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    # Support both standard EPD and FEN followed by EPD operations (e.g. ; id "name";)
                    fen_part = line
                    op_part = ""
                    if ";" in line:
                        fen_part, op_part = line.split(";", 1)

                    tokens = fen_part.strip().split()
                    if len(tokens) == 4:
                        fen_str = " ".join(tokens) + " 0 1"
                    elif len(tokens) >= 6:
                        fen_str = " ".join(tokens[:6])
                    else:
                        raise ValueError(f"Too few fields for position: {fen_part}")

                    board = chess.Board(fen_str)
                    id_name = None
                    if 'id "' in op_part:
                        id_name = op_part.split('id "', 1)[1].split('"', 1)[0]
                    elif "id " in op_part:
                        id_name = op_part.split("id ", 1)[1].strip().strip('"')

                    self.get_or_create_position(board, name=id_name)
                    count += 1
                except Exception as e:
                    err = f"Line {line_idx} invalid EPD: {e}"
                    self.validation_errors.append(err)
                    self.quarantined_items.append(
                        {"source": str(epd_path), "line": line, "reason": err}
                    )
                    self.records_rejected += 1

        self._record_source_manifest(source_name, epd_path, count)
        return count

    def ingest_json_catalog(
        self, json_path: Union[str, Path], source_name: str = "json_catalog"
    ) -> int:
        """Ingest openings from JSON catalog (book_selfplay.json or eco_openings.json)."""
        json_path = Path(json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"JSON catalog not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines: List[Dict[str, Any]] = []
        if isinstance(data, list):
            lines = data
        elif isinstance(data, dict) and "lines" in data:
            lines = data["lines"]

        count = 0
        for idx, item in enumerate(lines):
            self.records_processed += 1
            uci_moves = item.get("uci", [])
            eco = item.get("eco")
            name = item.get("name")
            reported_count = item.get("count", 1)

            board = chess.Board()
            valid = True
            for ply, uci_str in enumerate(uci_moves):
                try:
                    move = chess.Move.from_uci(uci_str)
                    if move in board.legal_moves:
                        # Feed synthetic game occurrences if reported
                        for rep in range(min(reported_count, 10)):
                            gid = f"catalog_{idx}_{rep}" if reported_count > 1 else f"catalog_{idx}"
                            self.add_position_move(
                                board=board,
                                move=move,
                                game_id=gid,
                                source=source_name,
                                ply=ply,
                                eco=eco,
                                name=name,
                            )
                        board.push(move)
                    else:
                        valid = False
                        break
                except Exception as e:
                    err = f"Catalog item {idx} invalid move {uci_str}: {e}"
                    self.validation_errors.append(err)
                    self.quarantined_items.append(
                        {"source": str(json_path), "item": item, "reason": err}
                    )
                    self.records_rejected += 1
                    valid = False
                    break

            if valid:
                count += 1

        self._record_source_manifest(source_name, json_path, count)
        return count

    def ingest_pgn(
        self,
        pgn_path: Union[str, Path],
        max_games: Optional[int] = None,
        max_plies: int = 24,
        min_elo: Optional[int] = 2200,
        source_name: str = "pgn",
    ) -> int:
        """Ingest master games from PGN file up to max_plies."""
        pgn_path = Path(pgn_path)
        if not pgn_path.is_file():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        games_ingested = 0
        with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                if max_games is not None and games_ingested >= max_games:
                    break
                self.records_processed += 1
                try:
                    game = chess.pgn.read_game(f)
                except Exception as e:
                    err = f"Malformed PGN game around game {games_ingested}: {e}"
                    self.validation_errors.append(err)
                    self.records_rejected += 1
                    continue

                if game is None:
                    break

                # Filter by rating if required
                if min_elo is not None:
                    try:
                        w_elo = int(game.headers.get("WhiteElo", "0"))
                        b_elo = int(game.headers.get("BlackElo", "0"))
                        if max(w_elo, b_elo) < min_elo and min(w_elo, b_elo) < (min_elo - 100):
                            continue
                    except ValueError:
                        pass

                game_id = game.headers.get("Site", "")
                if not game_id or not game_id.startswith("http"):
                    w = game.headers.get("White", "W")
                    b = game.headers.get("Black", "B")
                    d = game.headers.get("Date", "D")
                    game_id = f"{d}_{w}_{b}_{games_ingested}"

                eco = game.headers.get("ECO")
                name = game.headers.get("Opening")

                board = game.board()
                ply = 0
                for node in game.mainline():
                    if ply >= max_plies:
                        break
                    move = node.move
                    if move in board.legal_moves:
                        self.add_position_move(
                            board=board,
                            move=move,
                            game_id=game_id,
                            source=source_name,
                            ply=ply,
                            eco=eco,
                            name=name,
                        )
                        board.push(move)
                        ply += 1
                    else:
                        break

                games_ingested += 1

        self._record_source_manifest(source_name, pgn_path, games_ingested)
        return games_ingested

    def _record_source_manifest(self, source_name: str, file_path: Path, count: int) -> None:
        """Update manifest metadata for an imported source."""
        try:
            h = hashlib.sha256(file_path.read_bytes()[:1_000_000]).hexdigest()
            size = file_path.stat().st_size
        except Exception:
            h = "unknown"
            size = 0
        self.sources_manifest[source_name] = {
            "source_name": source_name,
            "path": str(file_path),
            "size_bytes": size,
            "sha256_prefix": h[:16],
            "records_imported": count,
        }

    def update_qualification_status(self) -> Dict[str, int]:
        """Apply the union qualification rule:

        Move qualifies if:
          distinct_games >= min_game_frequency OR stockfish_cp >= sf_soundness_threshold_cp
        """
        counts = {
            "frequency_only": 0,
            "stockfish_only": 0,
            "both": 0,
            "rejected": 0,
            "pending": 0,
            "total_pairs": 0,
        }

        for pos in self.positions.values():
            for ev in pos.moves.values():
                counts["total_pairs"] += 1
                freq_ok = len(ev.distinct_game_ids) >= self.min_game_frequency
                sf_ok = False
                if ev.stockfish_evaluated:
                    if ev.stockfish_mate is not None:
                        sf_ok = ev.stockfish_mate > 0
                    elif ev.stockfish_cp is not None:
                        sf_ok = ev.stockfish_cp >= self.sf_soundness_threshold_cp

                if freq_ok and sf_ok:
                    ev.qualification = "both"
                    counts["both"] += 1
                elif freq_ok and not ev.stockfish_evaluated:
                    ev.qualification = "frequency"
                    counts["frequency_only"] += 1
                elif freq_ok and not sf_ok:
                    # Qualified by frequency despite suboptimal engine evaluation
                    ev.qualification = "frequency"
                    counts["frequency_only"] += 1
                elif not freq_ok and sf_ok:
                    ev.qualification = "stockfish"
                    counts["stockfish_only"] += 1
                elif not freq_ok and ev.stockfish_evaluated and not sf_ok:
                    ev.qualification = "rejected"
                    counts["rejected"] += 1
                else:
                    ev.qualification = "pending"
                    counts["pending"] += 1

        return counts

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return comprehensive audit numbers."""
        total_positions = len(self.positions)
        total_moves = sum(len(p.moves) for p in self.positions.values())
        unique_games = set()
        for p in self.positions.values():
            for ev in p.moves.values():
                unique_games.update(ev.distinct_game_ids)

        qual_counts = self.update_qualification_status()

        return {
            "unique_canonical_positions": total_positions,
            "unique_position_move_pairs": total_moves,
            "distinct_games_indexed": len(unique_games),
            "records_processed": self.records_processed,
            "records_accepted": self.records_accepted,
            "records_rejected": self.records_rejected,
            "qualification_breakdown": qual_counts,
            "sources": self.sources_manifest,
        }

    def export_curated_db(self, output_path: Union[str, Path], include_all: bool = False) -> Path:
        """Export curated database to JSON.

        If include_all is False, only includes moves satisfying the union qualification rule.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.update_qualification_status()

        export_data: Dict[str, Any] = {
            "version": "1.0.0",
            "min_game_frequency": self.min_game_frequency,
            "sf_soundness_threshold_cp": self.sf_soundness_threshold_cp,
            "summary": self.get_summary_stats(),
            "positions": {},
        }

        for key, pos in self.positions.items():
            qual_moves = {}
            for uci, ev in pos.moves.items():
                # Qualification gates lookup when include_all is False.
                if not include_all and ev.qualification not in ("frequency", "stockfish", "both"):
                    continue
                qual_moves[uci] = ev.to_dict()

            if qual_moves or include_all:
                export_data["positions"][key] = {
                    "canonical_key": pos.canonical_key,
                    "fen": pos.fen,
                    "side_to_move": pos.side_to_move,
                    "ply": pos.ply,
                    "eco": pos.eco,
                    "opening_name": pos.opening_name,
                    "moves": qual_moves,
                }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        logger.info(
            "Exported curated opening DB to %s with %d positions",
            output_path,
            len(export_data["positions"]),
        )
        return output_path
