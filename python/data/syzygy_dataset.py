"""Offline Syzygy Endgame Supervision Pipeline.

Extracts tablebase-eligible endgame positions (<= 5 pieces, castling "-"),
computes exact WDL value targets and optimal tablebase move sets, and saves
supervised training datasets in the canonical memory-mapped binary format.

Perspective and Value Target Invariants:
- All labels are strictly relative to the player whose turn it is to move.
- 5-class outcome labels kept separate (not collapsed to Draw):
  0 = Win (+2 raw_wdl)        -> WDL target [1.0, 0.0, 0.0]
  1 = Cursed Win (+1)           -> WDL target [1.0, 0.0, 0.0]
  2 = Draw (0)                  -> WDL target [0.0, 1.0, 0.0]
  3 = Blessed Loss (-1)         -> WDL target [0.0, 0.0, 1.0]
  4 = Loss (-2 raw_wdl)        -> WDL target [0.0, 0.0, 1.0]

Tablebase Policy Targets:
- From eligible position P (mover M), enumerate legal candidate moves m -> P'.
- Evaluate child outcome from M's perspective: -WDL(P').
- Winning moves: child has WDL == -2 (opponent loses).
- Best tablebase moves are encoded into 4,672-action policy targets.

Separation of Concerns:
- Operates strictly offline or between games.
- Never called inside C++ MCTS simulation loop.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import chess
import chess.syzygy
import numpy as np

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp
from python.selfplay.tb_adjudicate import TablebaseAdjudicator

logger = logging.getLogger(__name__)

DEFAULT_SYZYGY_PATH = Path("data/syzygy")


class SyzygyPositionSample:
    """A verified tablebase endgame training sample."""

    def __init__(
        self,
        fen: str,
        planes: np.ndarray,
        best_move_idx: int,
        best_move_uci: str,
        outcome_class: int,
        wdl_target: np.ndarray,
        optimal_action_indices: List[int],
        optimal_policy_target: Optional[np.ndarray] = None,
        ply: int = 50,
        game_id: int = 0,
        dtz_target: Optional[int] = None,
    ) -> None:
        self.fen = fen
        self.planes = planes
        self.best_move_idx = best_move_idx
        self.best_move_uci = best_move_uci
        self.outcome_class = outcome_class
        self.wdl_target = wdl_target
        self.optimal_action_indices = optimal_action_indices
        self.optimal_policy_target = optimal_policy_target
        self.ply = ply
        self.game_id = game_id
        self.dtz_target = dtz_target


class SyzygySupervisionGenerator:
    """Generates endgame training datasets from Syzygy WDL tablebases."""

    def __init__(
        self,
        tablebase_path: Union[str, Path] = DEFAULT_SYZYGY_PATH,
        max_pieces: int = 5,
    ) -> None:
        self.tablebase_path = Path(tablebase_path)
        self.max_pieces = max_pieces
        self.tb: Optional[chess.syzygy.Tablebase] = None

        if self.tablebase_path.is_dir():
            try:
                self.tb = chess.syzygy.open_tablebase(str(self.tablebase_path))
                logger.info("Opened Syzygy tablebase from %s", self.tablebase_path)
            except Exception as e:
                logger.error("Failed to open Syzygy tablebase at %s: %s", self.tablebase_path, e)
                self.tb = None
        else:
            logger.warning("Syzygy tablebase directory %s does not exist", self.tablebase_path)

    def is_eligible(self, board_or_fen: Union[chess.Board, str]) -> Tuple[bool, int]:
        """Check if board satisfies Syzygy probing criteria."""
        if isinstance(board_or_fen, str):
            fen = board_or_fen
            return TablebaseAdjudicator.fast_filter(fen, self.max_pieces)
        else:
            board = board_or_fen
            has_no_castling = board.castling_rights == chess.BB_EMPTY
            piece_count = len(board.piece_map())
            return (has_no_castling and piece_count <= self.max_pieces), piece_count

    def probe_dtz(self, board: chess.Board) -> Optional[int]:
        """Probe raw Syzygy DTZ (distance-to-zero) value.

        Returns positive value for winning side to move, 0 for draw,
        negative value for losing side. Requires both WDL and DTZ tables.
        """
        if self.tb is None:
            return None
        eligible, _ = self.is_eligible(board)
        if not eligible:
            return None
        try:
            return self.tb.probe_dtz(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return None
        except Exception as e:
            logger.debug("Syzygy DTZ probe error: %s", e)
            return None

    def probe_dtm(self, board: chess.Board) -> Optional[int]:
        """DTM probing is not provided by python-chess syzygy Tablebase interface."""
        logger.debug("probe_dtm is unavailable in current python-chess syzygy bindings.")
        return None

    def probe_wdl(self, board: chess.Board) -> Optional[int]:
        """Probe raw Syzygy WDL: +2 (win), +1 (cursed win), 0 (draw), -1 (blessed loss), -2 (loss).

        Score is from the perspective of the side to move in `board`.
        """
        if self.tb is None:
            return None
        eligible, _ = self.is_eligible(board)
        if not eligible:
            return None
        try:
            return self.tb.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return None
        except Exception as e:
            logger.debug("Syzygy probe error: %s", e)
            return None

    def probe_position_and_moves(
        self,
        board: chess.Board,
    ) -> Optional[Tuple[int, int, np.ndarray, List[Tuple[chess.Move, int]]]]:
        """Probe position outcome and classify all legal candidate moves.

        Returns:
            (raw_wdl, outcome_class, wdl_target, list_of_moves_with_child_scores)
            where child score is from moving player's perspective:
              +2 = win for mover
               0 = draw for mover
              -2 = loss for mover
        """
        raw_wdl = self.probe_wdl(board)
        if raw_wdl is None:
            return None

        # Outcome class from moving player's perspective (5-class, not collapsed):
        # 0 = Win (+2), 1 = Cursed Win (+1), 2 = Draw (0),
        # 3 = Blessed Loss (-1), 4 = Loss (-2)
        if raw_wdl == 2:
            outcome_class = 0
            wdl_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif raw_wdl == 1:
            outcome_class = 1
            wdl_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif raw_wdl == 0:
            outcome_class = 2
            wdl_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif raw_wdl == -1:
            outcome_class = 3
            wdl_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif raw_wdl == -2:
            outcome_class = 4
            wdl_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            # Fallback for unexpected raw_wdl values
            outcome_class = 2
            wdl_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Enumerate legal moves and determine child outcome
        move_scores: List[Tuple[chess.Move, int]] = []
        for m in board.legal_moves:
            child = board.copy(stack=False)
            child.push(m)

            if child.is_checkmate():
                # Direct checkmate: decisive win for mover
                move_scores.append((m, 2))
                continue
            if child.is_stalemate() or child.is_insufficient_material():
                # Direct stalemate / draw
                move_scores.append((m, 0))
                continue

            child_wdl = self.probe_wdl(child)
            if child_wdl is None:
                continue

            # Child WDL is from opponent's perspective!
            # Keep separate classes (+1 / -1) instead of collapsing to draw.
            # Mover value mapping (inverse of child perspective):
            #  -2 -> +2  (opponent loss -> mover win)
            #  -1 -> +1  (opponent blessed loss -> mover cursed win)
            #   0 ->  0  (draw)
            #  +1 -> -1  (opponent cursed win -> mover blessed loss)
            #  +2 -> -2  (opponent win -> mover loss)
            if child_wdl == -2:
                mover_val = 2
            elif child_wdl == -1:
                mover_val = 1
            elif child_wdl == 0:
                mover_val = 0
            elif child_wdl == 1:
                mover_val = -1
            elif child_wdl == 2:
                mover_val = -2
            else:
                mover_val = 0
            move_scores.append((m, mover_val))

        return raw_wdl, outcome_class, wdl_target, move_scores

    def create_training_sample(
        self,
        board: chess.Board,
        game_id: int = 0,
        ply: Optional[int] = None,
    ) -> Optional[SyzygyPositionSample]:
        """Create a complete Syzygy training sample for the position."""
        res = self.probe_position_and_moves(board)
        if res is None:
            return None

        if ply is None:
            # Read ply from FEN/fullmove number instead of hard-coding 50
            ply = board.fullmove_number * 2 - (1 if board.turn == chess.WHITE else 0)

        raw_wdl, outcome_class, wdl_target, move_scores = res
        if not move_scores:
            return None

        # Best available outcome among legal moves
        best_val = max(s for _, s in move_scores)
        optimal_moves = [m for m, s in move_scores if s == best_val]
        if not optimal_moves:
            optimal_moves = [m for m, _ in move_scores]

        # Use C++ game to encode state and moves
        fen = board.fen()
        cpp_game = zeropawn_cpp.ChessGame(fen)
        planes = zeropawn_cpp.encode_state(cpp_game)

        optimal_indices = []
        for m in optimal_moves:
            try:
                idx = zeropawn_cpp.move_to_index(cpp_game, m.uci())
                if 0 <= idx < 4672:
                    optimal_indices.append(idx)
            except Exception:
                pass

        if not optimal_indices:
            # Fallback to any legal move
            legal_ucis = cpp_game.legal_moves_uci()
            if not legal_ucis:
                return None
            idx = zeropawn_cpp.move_to_index(cpp_game, legal_ucis[0])
            optimal_indices = [idx]

        primary_best_move_idx = optimal_indices[0]
        primary_best_move_uci = optimal_moves[0].uci()

        # Probe DTZ target if available (preserves distance-to-zero targets)
        dtz_target = self.probe_dtz(board)

        # Build normalized optimal policy distribution target
        policy_target = np.zeros(4672, dtype=np.float32)
        prob_per_move = 1.0 / len(optimal_indices)
        for idx in optimal_indices:
            policy_target[idx] = prob_per_move

        return SyzygyPositionSample(
            fen=fen,
            planes=planes,
            best_move_idx=primary_best_move_idx,
            best_move_uci=primary_best_move_uci,
            outcome_class=outcome_class,
            wdl_target=wdl_target,
            optimal_action_indices=optimal_indices,
            optimal_policy_target=policy_target,
            ply=ply,
            game_id=game_id,
            dtz_target=dtz_target,
        )

    def generate_synthetic_endgame_positions(
        self,
        count: int = 500,
        seed: int = 42,
    ) -> List[chess.Board]:
        """Generate legal 3, 4, and 5-piece endgame positions."""
        rng = random.Random(seed)
        boards: List[chess.Board] = []
        seen_fens: Set[str] = set()

        # Piece configurations for 3, 4, 5 pieces
        configs = [
            # 3 pieces
            ([chess.PAWN], []),
            ([chess.ROOK], []),
            ([chess.QUEEN], []),
            ([chess.BISHOP], []),
            ([chess.KNIGHT], []),
            # 4 pieces
            ([chess.PAWN], [chess.PAWN]),
            ([chess.ROOK], [chess.PAWN]),
            ([chess.BISHOP, chess.BISHOP], []),
            ([chess.BISHOP, chess.KNIGHT], []),
            ([chess.ROOK], [chess.ROOK]),
            ([chess.QUEEN], [chess.QUEEN]),
            # 5 pieces
            ([chess.ROOK, chess.PAWN], [chess.ROOK]),
            ([chess.QUEEN, chess.PAWN], [chess.QUEEN]),
            ([chess.PAWN, chess.PAWN], [chess.PAWN]),
            ([chess.BISHOP, chess.PAWN], [chess.BISHOP]),
            ([chess.KNIGHT, chess.PAWN], [chess.KNIGHT]),
            ([chess.ROOK, chess.BISHOP], [chess.ROOK]),
        ]

        squares = list(chess.SQUARES)

        trials = 0
        max_trials = count * 20
        while len(boards) < count and trials < max_trials:
            trials += 1
            w_pieces, b_pieces = rng.choice(configs)
            turn = rng.choice([chess.WHITE, chess.BLACK])

            board = chess.Board(None)
            board.turn = turn
            board.castling_rights = chess.BB_EMPTY

            # Place White King and Black King
            occupied: Set[int] = set()
            wk_sq = rng.choice(squares)
            occupied.add(wk_sq)
            board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))

            # Black King cannot be adjacent to White King
            valid_bk_sqs = [
                sq for sq in squares if sq not in occupied and chess.square_distance(wk_sq, sq) > 1
            ]
            if not valid_bk_sqs:
                continue
            bk_sq = rng.choice(valid_bk_sqs)
            occupied.add(bk_sq)
            board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

            valid_config = True
            for pt in w_pieces:
                valid_sqs = [
                    sq
                    for sq in squares
                    if sq not in occupied
                    and not (pt == chess.PAWN and chess.square_rank(sq) in (0, 7))
                ]
                if not valid_sqs:
                    valid_config = False
                    break
                sq = rng.choice(valid_sqs)
                occupied.add(sq)
                board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))

            if not valid_config:
                continue

            for pt in b_pieces:
                valid_sqs = [
                    sq
                    for sq in squares
                    if sq not in occupied
                    and not (pt == chess.PAWN and chess.square_rank(sq) in (0, 7))
                ]
                if not valid_sqs:
                    valid_config = False
                    break
                sq = rng.choice(valid_sqs)
                occupied.add(sq)
                board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))

            if not valid_config:
                continue

            # Must be a legal, non-terminal chess board
            if not board.is_valid():
                continue
            if board.is_checkmate() or board.is_stalemate():
                continue
            if not list(board.legal_moves):
                continue

            board.fen()
            epd = board.epd(en_passant="pawns")
            if epd in seen_fens:
                continue
            seen_fens.add(epd)
            boards.append(board)

        logger.info(
            "Generated %d legal synthetic endgame positions after %d trials", len(boards), trials
        )
        return boards

    def build_endgame_dataset(
        self,
        output_dir: Union[str, Path],
        positions: Optional[List[chess.Board]] = None,
        synthetic_count: int = 500,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Build binary memmap dataset with Syzygy supervision."""
        output_dir = Path(output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        if positions is None:
            positions = self.generate_synthetic_endgame_positions(count=synthetic_count, seed=seed)

        samples: List[SyzygyPositionSample] = []
        for idx, board in enumerate(positions):
            s = self.create_training_sample(board, game_id=idx + 1000)
            if s is not None:
                samples.append(s)

        if not samples:
            raise RuntimeError("No valid Syzygy training samples could be generated.")

        # Split train and validation
        rng = random.Random(seed)
        rng.shuffle(samples)
        num_val = max(1, int(len(samples) * val_ratio))
        val_samples = samples[:num_val]
        train_samples = samples[num_val:]

        self._write_split(train_dir, train_samples, "train")
        self._write_split(val_dir, val_samples, "val")

        summary = {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "output_dir": str(output_dir),
            "wins": sum(1 for s in samples if s.outcome_class in (0, 1)),
            "draws": sum(1 for s in samples if s.outcome_class == 2),
            "losses": sum(1 for s in samples if s.outcome_class in (3, 4)),
        }
        logger.info("Built endgame Syzygy dataset: %s", summary)
        return summary

    def _write_split(
        self, split_dir: Path, samples: List[SyzygyPositionSample], split_name: str
    ) -> None:
        num = len(samples)
        planes_path = split_dir / "planes.bin"
        moves_path = split_dir / "moves.bin"
        outcomes_path = split_dir / "outcomes.bin"
        game_ids_path = split_dir / "game_ids.bin"
        plies_path = split_dir / "plies.bin"
        val_targets_path = split_dir / "value_targets.bin"
        policy_targets_path = split_dir / "policy_targets.bin"
        dtz_targets_path = split_dir / "dtz_targets.bin"
        meta_path = split_dir / "metadata.json"

        planes_mm = np.memmap(planes_path, dtype=np.float32, mode="w+", shape=(num, 19, 8, 8))
        moves_mm = np.memmap(moves_path, dtype=np.int16, mode="w+", shape=(num,))
        outcomes_mm = np.memmap(outcomes_path, dtype=np.uint8, mode="w+", shape=(num,))
        game_ids_mm = np.memmap(game_ids_path, dtype=np.uint64, mode="w+", shape=(num,))
        plies_mm = np.memmap(plies_path, dtype=np.int16, mode="w+", shape=(num,))
        val_targets_mm = np.memmap(val_targets_path, dtype=np.float32, mode="w+", shape=(num, 3))
        policy_targets_mm = np.memmap(
            policy_targets_path, dtype=np.float32, mode="w+", shape=(num, 4672)
        )
        dtz_mm = np.memmap(dtz_targets_path, dtype=np.float32, mode="w+", shape=(num,))

        for i, s in enumerate(samples):
            planes_mm[i] = s.planes
            moves_mm[i] = np.int16(s.best_move_idx)
            outcomes_mm[i] = np.uint8(s.outcome_class)
            game_ids_mm[i] = np.uint64(s.game_id)
            plies_mm[i] = np.int16(s.ply)
            val_targets_mm[i] = s.wdl_target
            policy_targets_mm[i] = (
                s.optimal_policy_target
                if s.optimal_policy_target is not None
                else np.zeros(4672, dtype=np.float32)
            )
            dtz_mm[i] = np.float32(s.dtz_target) if s.dtz_target is not None else np.nan

        planes_mm.flush()
        moves_mm.flush()
        outcomes_mm.flush()
        game_ids_mm.flush()
        plies_mm.flush()
        val_targets_mm.flush()
        policy_targets_mm.flush()
        dtz_mm.flush()

        metadata = {
            "num_samples": num,
            "planes_shape": [19, 8, 8],
            "split": split_name,
            "kind": "endgame_syzygy",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def close(self) -> None:
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass
            self.tb = None

    def __enter__(self) -> SyzygySupervisionGenerator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
