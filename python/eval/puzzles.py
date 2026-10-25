"""Puzzle evaluation benchmark on a fixed set of 1,000 Lichess puzzles (1200-1600).

Contract:
- Fixed set of 1,000 Lichess puzzles rated 1200-1600 (do not change the set).
- Reusable across checkpoints every 6 hours.
- Computes solve rate (fraction of puzzles where engine's selected move matches solution).
"""

from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from python.model.chess_net import ChessNet

try:
    import zeropawn_cpp
except ImportError:
    zeropawn_cpp = None


from python.eval.match_runner import check_search_mode, require_serial_search


@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    opponent_move: str
    solution_move: str
    rating: int


def load_puzzles(
    csv_path: Union[str, Path] = "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
) -> list[Puzzle]:
    """Load puzzles from the CSV file, or all CSV files if a directory is given."""
    csv_path = Path(csv_path)
    if csv_path.is_dir():
        all_puzzles: list[Puzzle] = []
        for f in sorted(csv_path.glob("*.csv")):
            all_puzzles.extend(load_puzzles(f))
        return all_puzzles
    if not csv_path.exists():
        raise FileNotFoundError(f"Puzzle file not found at {csv_path}")

    puzzles: list[Puzzle] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 4:
                continue
            puzzle_id = row[0]
            fen = row[1]
            moves = row[2].split()
            if len(moves) < 2:
                continue
            opp_move = moves[0]
            sol_move = moves[1]
            rating = int(row[3])
            puzzles.append(
                Puzzle(
                    puzzle_id=puzzle_id,
                    fen=fen,
                    opponent_move=opp_move,
                    solution_move=sol_move,
                    rating=rating,
                )
            )
    return puzzles


class PuzzleEvaluator:
    """Evaluates a neural network model on the fixed puzzle benchmark."""

    def __init__(
        self,
        model: ChessNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        puzzles: Optional[list[Puzzle]] = None,
        puzzle_csv_path: Union[str, Path] = "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        search_mode: str = "serial",
        batch_size: Union[int, str] = 16,
    ) -> None:
        if zeropawn_cpp is None:
            raise RuntimeError(
                "zeropawn_cpp extension module is not installed or compiled. "
                "Compile it first using 'python cli.py build compile' or 'cmake --build build'."
            )
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        self.puzzles = puzzles if puzzles is not None else load_puzzles(puzzle_csv_path)
        self.search_mode = check_search_mode(search_mode)
        from python.hardware.profile import resolve_batch_size

        self.batch_size = resolve_batch_size(batch_size, workload="puzzles", default=16)

    @torch.no_grad()
    def evaluate_policy(self, batch_size: int = 128) -> dict[str, float]:
        """Evaluate top-1 move match from the policy head (masked with legal moves)."""
        solved = 0
        total = len(self.puzzles)
        if total == 0:
            return {"solve_rate": 0.0, "total": 0, "solved": 0}

        # Process in batches
        for i in range(0, total, batch_size):
            batch_puzzles = self.puzzles[i : i + batch_size]
            games: list[zeropawn_cpp.ChessGame] = []
            valid_puzzles: list[Puzzle] = []

            for p in batch_puzzles:
                g = zeropawn_cpp.ChessGame(p.fen)
                # Apply opponent move to reach puzzle position
                if g.is_legal(p.opponent_move):
                    g.make_move_uci(p.opponent_move)
                    games.append(g)
                    valid_puzzles.append(p)

            if not games:
                continue

            # Encode states
            planes_list = [zeropawn_cpp.encode_state(g) for g in games]
            planes_tensor = torch.from_numpy(np.stack(planes_list)).to(self.device)
            if self.device == "cuda":
                planes_tensor = planes_tensor.to(memory_format=torch.channels_last)

            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                policy_logits, _ = self.model(planes_tensor)

            policy_probs = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()

            for j, (g, p) in enumerate(zip(games, valid_puzzles)):
                legal_moves = g.legal_moves_uci()
                if not legal_moves:
                    continue

                best_move = None
                best_prob = -1.0

                for m in legal_moves:
                    idx = zeropawn_cpp.move_to_index(g, m)
                    if 0 <= idx < 4672:
                        prob = policy_probs[j, idx]
                        if prob > best_prob:
                            best_prob = prob
                            best_move = m

                if best_move == p.solution_move:
                    solved += 1

        solve_rate = solved / total if total > 0 else 0.0
        return {
            "solve_rate": solve_rate,
            "solved": solved,
            "total": total,
        }

    def evaluate_mcts(
        self,
        num_simulations: int = 50,
        c_puct: float = 2.0,
        max_puzzles: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> dict[str, float]:
        """Evaluate puzzles using C++ MCTS search (deterministic, no noise)."""
        require_serial_search(self.search_mode, "PuzzleEvaluator.evaluate_mcts")
        solved = 0
        puzzles_to_eval = self.puzzles[:max_puzzles] if max_puzzles else self.puzzles
        total = len(puzzles_to_eval)
        effective_batch_size = max(
            1, int(batch_size if batch_size is not None else self.batch_size)
        )

        mcts_config = zeropawn_cpp.MCTSConfig()
        mcts_config.c_puct = c_puct
        mcts_config.add_noise = False
        mcts_config.virtual_loss = 1
        mcts_config.fpu_reduction = 0.2
        mcts = zeropawn_cpp.MCTS(mcts_config)

        def eval_fn(game: zeropawn_cpp.Game):
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
                with autocast_ctx:
                    p_logits, v_wdl = self.model(t)
                p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
                v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
                scalar_val = float(v_probs[0] - v_probs[2])
            return (p_probs, scalar_val)

        def batched_eval_fn(batch_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                with autocast_ctx:
                    p_logits, v_wdl = self.model(t)
                policy = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
                wdl = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
                values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)
            return policy.astype(np.float32), values

        for p in puzzles_to_eval:
            g = zeropawn_cpp.ChessGame(p.fen)
            if not g.is_legal(p.opponent_move):
                continue
            g.make_move_uci(p.opponent_move)
            mcts.clear()
            if self.search_mode == "batched":
                res = mcts.search_batched(g, num_simulations, effective_batch_size, batched_eval_fn)
            else:
                res = mcts.search(g, num_simulations, eval_fn)
            best_uci = g.action_to_string(res.best_action)
            if best_uci == p.solution_move:
                solved += 1

        solve_rate = solved / total if total > 0 else 0.0
        return {
            "solve_rate": solve_rate,
            "solved": solved,
            "total": total,
            "search_mode": self.search_mode,
            "batch_size": effective_batch_size if self.search_mode == "batched" else 1,
        }
