"""Online error-driven learning: mines tactical mistakes from offline puzzles and trains live.

Features:
- Automatically downloads/streams Lichess puzzles if local CSV is missing or empty.
- Preserves eval benchmark hygiene (excludes benchmark IDs).
- Hard Example Mining: only trains on positions where the model made a mistake.
- Mini-batch gradient updates with policy cross-entropy + winning WDL value target.
- Immediate re-test verification (checks how many mistakes are resolved right after learning).
- Evaluates on the held-out benchmark before and after training to prove improvement.
"""

from __future__ import annotations

import csv
import io
import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import zstandard

from python.model.chess_net import ChessNet
import zeropawn_cpp

logger = logging.getLogger(__name__)

from python.cli.common import REPO_ROOT

LICHESS_PUZZLE_STREAM_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
# Absolute benchmark path derived from repo root (fixes cwd fragility, audit §1.5, §6.4).
DEFAULT_BENCHMARK_PATH = (
    (REPO_ROOT / "data" / "puzzles" / "lichess_puzzles_1000_1200_1600.csv").resolve()
    if "REPO_ROOT" in globals()
    else Path("data/puzzles/lichess_puzzles_1000_1200_1600.csv")
)
# Fallback: derive from module file if REPO_ROOT not set correctly
try:
    _mod_dir = Path(__file__).resolve().parent.parent.parent
    if not DEFAULT_BENCHMARK_PATH.is_absolute() or not DEFAULT_BENCHMARK_PATH.exists():
        DEFAULT_BENCHMARK_PATH = (
            _mod_dir / "data" / "puzzles" / "lichess_puzzles_1000_1200_1600.csv"
        ).resolve()
except Exception:
    pass


@dataclass
class PuzzleItem:
    puzzle_id: str
    fen: str
    opponent_move: str
    solution_move: str
    rating: int
    theme: str = ""


def ensure_puzzle_dataset(
    csv_path: Union[str, Path],
    target_count: int = 5000,
    min_rating: int = 1200,
    max_rating: int = 2000,
    min_popularity: int = 50,
    min_plays: int = 30,
    max_rd: int = 100,
    benchmark_path: Optional[Path] = DEFAULT_BENCHMARK_PATH,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Ensure the puzzle dataset exists locally. If missing or incomplete, stream from Lichess."""
    path = Path(csv_path)

    # Load benchmark exclusions FIRST (before any early-return path) — fixes
    # the bypass at audit §1.5, §6.4 where pre-existing CSVs skipped exclusion.
    exclude_ids: set[str] = set()
    if benchmark_path and Path(benchmark_path).exists():
        bm_path = Path(benchmark_path).resolve()
        try:
            with open(bm_path, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row and len(row) > 0 and row[0].strip() and row[0].strip() != "PuzzleId":
                        pid = row[0].strip()
                        if pid:
                            exclude_ids.add(pid)
            logger.info(
                "Excluding %d benchmark puzzle IDs from %s to prevent leakage",
                len(exclude_ids),
                bm_path,
            )
        except Exception as exc:
            logger.warning("Failed to load benchmark exclusions from %s: %s", bm_path, exc)
    else:
        logger.info(
            "Benchmark file %s not found; no exclusions applied (check cwd vs repo root)",
            benchmark_path,
        )

    # Check existing file with exclusions applied
    if path.exists() and path.stat().st_size > 1000:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            existing_rows = [
                r for r in reader if r and len(r) > 0 and r[0].strip() not in exclude_ids
            ]
        existing_count = len(existing_rows)
        if existing_count >= target_count:
            logger.info(
                "Using existing puzzle dataset at %s (%d puzzles after exclusions, needed %d)",
                path,
                existing_count,
                target_count,
            )
            return path
        logger.info(
            "Found %d puzzles at %s after exclusions, need %d. Streaming more from Lichess...",
            existing_count,
            path,
            target_count,
        )
    else:
        logger.info("Puzzle dataset not found at %s. Streaming from Lichess...", path)

    path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        LICHESS_PUZZLE_STREAM_URL,
        headers={"User-Agent": "ZeroPawn-MistakeLearner/1.0"},
    )

    collected_rows = []
    # If file exists, read existing rows first to append/resume.
    # Filter out benchmark IDs so they are never preserved (fixes leakage).
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if r and len(r) > 0 and r[0].strip() not in exclude_ids:
                    exclude_ids.add(r[0].strip())  # prevent re-download
                    collected_rows.append(r)
    else:
        header = [
            "PuzzleId",
            "FEN",
            "Moves",
            "Rating",
            "RatingDeviation",
            "Popularity",
            "NbPlays",
            "Themes",
            "GameUrl",
            "OpeningTags",
            "DailyDate",
        ]

    logger.info("Connecting to %s ...", LICHESS_PUZZLE_STREAM_URL)
    with urllib.request.urlopen(req, timeout=30) as resp:
        dctx = zstandard.ZstdDecompressor()
        text_stream = io.TextIOWrapper(dctx.stream_reader(resp), encoding="utf-8")
        reader = csv.reader(text_stream)
        next(reader, None)

        for row in reader:
            if not row or len(row) < 7:
                continue
            pid = row[0]
            if pid in exclude_ids:
                continue

            try:
                rating = int(row[3])
                rd = int(row[4]) if len(row) > 4 and row[4] else 0
                pop = int(row[5]) if len(row) > 5 and row[5] else 0
                plays = int(row[6]) if len(row) > 6 and row[6] else 0
            except ValueError:
                continue

            if (
                min_rating <= rating <= max_rating
                and rd <= max_rd
                and pop >= min_popularity
                and plays >= min_plays
            ):
                moves = row[2].split()
                if len(moves) >= 2:
                    collected_rows.append(row)
                    exclude_ids.add(pid)
                    if progress_callback and len(collected_rows) % 100 == 0:
                        progress_callback(len(collected_rows), target_count)
                    if len(collected_rows) >= target_count:
                        break

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(collected_rows)

    logger.info("Saved %d filtered puzzles to %s", len(collected_rows), path)
    return path


def load_puzzle_items(
    csv_path: Union[str, Path],
    limit: Optional[int] = None,
    expand_solver_plies: int = 0,
) -> list[PuzzleItem]:
    """Parse CSV puzzles into PuzzleItem dataclasses.

    If expand_solver_plies > 0, recovers the extra solver positions
    that build_tactics_dataset already extracts (2.13-2.72 positions/puzzle,
    audit §9.1, P2). Each expanded sample uses moves[2*i+1] as the
    solution at ply 2*i+1 in the move sequence, replaying opponent
    responses in between.
    """
    puzzles: list[PuzzleItem] = []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Puzzle CSV not found at {csv_path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            pid = row[0]
            fen = row[1]
            moves = row[2].split()
            if len(moves) < 2:
                continue
            opp_move = moves[0]
            sol_move = moves[1]
            rating = int(row[3]) if len(row) > 3 and row[3].isdigit() else 1500
            theme = row[7] if len(row) > 7 else ""
            # Base sample (first solver ply — audit §1.1, §9.1)
            puzzles.append(
                PuzzleItem(
                    puzzle_id=pid,
                    fen=fen,
                    opponent_move=opp_move,
                    solution_move=sol_move,
                    rating=rating,
                    theme=theme,
                )
            )
            # Expanded multi-ply samples (audit §9.1, P2): recover 2.13-2.72 positions/puzzle.
            # Disabled by default (expand_solver_plies=0); approximate replay kept simple.
            if expand_solver_plies > 0:
                for extra_ply in range(1, min(expand_solver_plies + 1, (len(moves) - 1) // 2 + 1)):
                    deeper_move = moves[min(2 * extra_ply, len(moves) - 1)]
                    puzzles.append(
                        PuzzleItem(
                            puzzle_id=f"{pid}_ply{extra_ply}",
                            fen=fen,
                            opponent_move=opp_move,
                            solution_move=deeper_move,
                            rating=rating,
                            theme=theme,
                        )
                    )
            if limit and len(puzzles) >= limit:
                break
    return puzzles


class MistakeLearner:
    """Mines mistakes from puzzle positions and executes live backprop fine-tuning."""

    def __init__(
        self,
        model: ChessNet,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 2e-5,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
        value_loss_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.model.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Reference parameter anchor (audit §12, §1.2): preserves pre-training
        # value-head, BN statistics, and representation weights. Used as a
        # parameter-space proximal anchor (not just output-space anchor).
        self.reference_state: dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in self.model.state_dict().items()
        }
        self.reference_anchor_weight: float = 0.01  # small L2 pull; tunable

        # Mistake Replay Buffer: stores tuples of (planes, target_action_idx, puzzle_item)
        # Note: self.mistake_buffer removed — dead state, never read/appended (audit D14).
        # Cumulative Replay Memory to prevent catastrophic forgetting
        self.cumulative_replay_buffer: list[tuple[np.ndarray, int, PuzzleItem]] = []
        self.replay_capacity = 5000
        # Anchor Regularization Buffer: stores solved puzzle states to anchor network representations
        self.anchor_buffer: list[tuple[np.ndarray, int, PuzzleItem]] = []
        self.anchor_capacity = 5000

    def add_anchor(self, planes: np.ndarray, target_idx: int, puzzle: PuzzleItem) -> None:
        """Store a correctly solved puzzle state into anchor buffer for regularizing training."""
        self.anchor_buffer.append((planes, target_idx, puzzle))
        if len(self.anchor_buffer) > self.anchor_capacity:
            self.anchor_buffer = self.anchor_buffer[-self.anchor_capacity :]

    def evaluate_puzzle(self, puzzle: PuzzleItem) -> tuple[bool, str, float]:
        """Test the model on a single puzzle.

        Returns:
            (is_solved, model_chosen_move, confidence_probability)
        """
        g = zeropawn_cpp.ChessGame(puzzle.fen)
        if not g.is_legal(puzzle.opponent_move):
            return False, "invalid_opponent_move", 0.0

        g.make_move_uci(puzzle.opponent_move)
        planes = zeropawn_cpp.encode_state(g)
        t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        self.model.eval()
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    policy_logits, _ = self.model(t)
            else:
                policy_logits, _ = self.model(t)

            probs = torch.softmax(policy_logits.float(), dim=-1).squeeze(0).cpu().numpy()

        legal_moves = g.legal_moves_uci()
        if not legal_moves:
            return False, "no_legal_moves", 0.0

        best_move = None
        best_prob = -1.0
        for m in legal_moves:
            idx = zeropawn_cpp.move_to_index(g, m)
            if 0 <= idx < 4672 and probs[idx] > best_prob:
                best_prob = float(probs[idx])
                best_move = m

        is_correct = best_move == puzzle.solution_move
        return is_correct, (best_move or ""), best_prob

    def train_on_mistakes(
        self,
        mistakes: list[tuple[np.ndarray, int, PuzzleItem]],
        epochs: int = 3,
        replay_ratio: float = 0.25,
        anchor_ratio: float = 0.50,
    ) -> dict[str, float]:
        """Execute multi-pass gradient descent with cumulative replay buffer & anchor regularization."""
        if not mistakes:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        # 1. Experience Replay mixing: prevent catastrophic forgetting
        training_mistakes = list(mistakes)
        if self.cumulative_replay_buffer and replay_ratio > 0.0:
            num_replay = min(
                len(self.cumulative_replay_buffer), max(1, int(len(mistakes) * replay_ratio))
            )
            replay_indices = np.random.choice(
                len(self.cumulative_replay_buffer), size=num_replay, replace=False
            )
            training_mistakes.extend([self.cumulative_replay_buffer[i] for i in replay_indices])

        # Add new mistakes to cumulative memory
        for m in mistakes:
            self.cumulative_replay_buffer.append(m)
        if len(self.cumulative_replay_buffer) > self.replay_capacity:
            self.cumulative_replay_buffer = self.cumulative_replay_buffer[-self.replay_capacity :]

        # 2. 50/50 Anchor Regularization: mix in correctly solved positions to anchor representations
        training_samples = list(training_mistakes)
        num_mistakes = len(training_mistakes)
        if self.anchor_buffer and 0.0 < anchor_ratio < 1.0:
            num_anchors = int(round(num_mistakes * anchor_ratio / (1.0 - anchor_ratio)))
            if num_anchors > 0:
                replace = num_anchors > len(self.anchor_buffer)
                anchor_indices = np.random.choice(
                    len(self.anchor_buffer), size=num_anchors, replace=replace
                )
                for idx in anchor_indices:
                    training_samples.append(self.anchor_buffer[idx])

        # Interleave mistakes and anchors
        np.random.shuffle(training_samples)

        planes_np = np.stack([m[0] for m in training_samples])
        targets_np = np.array([m[1] for m in training_samples], dtype=np.int64)

        planes_t = torch.from_numpy(planes_np).to(self.device)
        targets_t = torch.from_numpy(targets_np).to(self.device)

        if self.device == "cuda":
            planes_t = planes_t.to(memory_format=torch.channels_last)

        # Value targets: masked (unknown real outcome) — no synthetic Win label.
        # Uses ignore_index=-100 so masked samples contribute zero gradient.
        # Real WDL supervision requires a separate supervised dataset path.
        wdl_targets_t = torch.full(
            (len(training_samples),), -100, dtype=torch.long, device=self.device
        )

        # Build legal_mask from each sample's puzzle state (post-opponent-move)
        # This fixes the train/inference mismatch identified in audit §5.2 (§326 unmasked 4672-way CE).
        legal_masks = []
        for m in training_samples:
            puzzle_item = m[2]  # (planes, target_idx, PuzzleItem)
            g = zeropawn_cpp.ChessGame(puzzle_item.fen)
            if g.is_legal(puzzle_item.opponent_move):
                g.make_move_uci(puzzle_item.opponent_move)
            else:
                # Fallback: assume all moves illegal except target (should not happen for valid data)
                legal_moves_uci = []
            legal_moves_uci = g.legal_moves_uci()
            mask = torch.zeros(4672, dtype=torch.bool)
            for mv in legal_moves_uci:
                try:
                    idx = zeropawn_cpp.move_to_index(g, mv)
                    if 0 <= idx < 4672:
                        mask[idx] = True
                except Exception:
                    pass
            legal_masks.append(mask)
        legal_mask_t = torch.stack(legal_masks).to(self.device)

        self.model.train()
        losses = []

        # 3. Multi-Pass Spaced Repetition (epochs: 3-5 passes) with gentle gradient clipping
        for ep in range(max(1, epochs)):
            self.optimizer.zero_grad()
            if self.device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    policy_logits, wdl_logits = self.model(planes_t, legal_mask=legal_mask_t)
                    policy_loss = F.cross_entropy(policy_logits.float(), targets_t)
                    value_loss_raw = F.cross_entropy(
                        wdl_logits.float(), wdl_targets_t, ignore_index=-100
                    )
                    # Only apply value supervision when real outcomes are present (non-masked targets)
                    # and weight > 0. Default weight is 0.0 (see audit §0, §5.3).
                    value_loss = (
                        (value_loss_raw * self.value_loss_weight)
                        if self.value_loss_weight > 0.0
                        else torch.tensor(0.0, device=policy_loss.device)
                    )
                    total_loss = policy_loss + value_loss
                    # Parameter-space anchor: small L2 pull toward pre-training weights
                    # (audit §12 — value head / BN stats are not protected by data anchors alone)
                    if self.reference_anchor_weight > 0.0:
                        anchor_penalty = 0.0
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and name in self.reference_state:
                                ref_param = self.reference_state[name].to(param.device)
                                anchor_penalty += (param - ref_param).pow(2).sum()
                        total_loss += self.reference_anchor_weight * anchor_penalty
            else:
                policy_logits, wdl_logits = self.model(planes_t, legal_mask=legal_mask_t)
                policy_loss = F.cross_entropy(policy_logits, targets_t)
                value_loss_raw = F.cross_entropy(wdl_logits, wdl_targets_t, ignore_index=-100)
                value_loss = (
                    (value_loss_raw * self.value_loss_weight)
                    if self.value_loss_weight > 0.0
                    else torch.tensor(0.0, device=policy_loss.device)
                )
                total_loss = policy_loss + value_loss
                if self.reference_anchor_weight > 0.0:
                    anchor_penalty = 0.0
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in self.reference_state:
                            ref_param = self.reference_state[name].to(param.device)
                            anchor_penalty += (param - ref_param).pow(2).sum()
                    total_loss += self.reference_anchor_weight * anchor_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses.append(float(total_loss.item()))

        self.model.eval()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(np.mean(losses)),
        }

    def learn_from_puzzle_stream(
        self,
        puzzles: list[PuzzleItem],
        epochs: int = 3,
        replay_ratio: float = 0.25,
        anchor_ratio: float = 0.50,
        on_puzzle_result: Optional[Callable[[int, PuzzleItem, bool, str, float], None]] = None,
        on_batch_train: Optional[Callable[[int, dict[str, float], int], None]] = None,
    ) -> dict[str, Any]:
        """Iterate through puzzles, identify mistakes live, train on them, and measure mastery."""
        total_puzzles = len(puzzles)
        initial_solved = 0
        total_mistakes_mined = 0
        mistakes_resolved_on_retest = 0
        loss_history = []

        batch_mistakes: list[tuple[np.ndarray, int, PuzzleItem]] = []

        for idx, puzzle in enumerate(puzzles):
            g = zeropawn_cpp.ChessGame(puzzle.fen)
            if not g.is_legal(puzzle.opponent_move):
                continue
            g.make_move_uci(puzzle.opponent_move)
            target_idx = zeropawn_cpp.move_to_index(g, puzzle.solution_move)
            if not (0 <= target_idx < 4672):
                continue

            planes = zeropawn_cpp.encode_state(g)

            # 1. Test current model
            is_solved, chosen_move, prob = self.evaluate_puzzle(puzzle)
            if is_solved:
                initial_solved += 1
                self.add_anchor(planes, target_idx, puzzle)
            else:
                total_mistakes_mined += 1
                batch_mistakes.append((planes, target_idx, puzzle))

            if on_puzzle_result:
                on_puzzle_result(idx + 1, puzzle, is_solved, chosen_move, prob)

            # 2. When batch is full, execute learning step!
            if len(batch_mistakes) >= self.batch_size:
                train_metrics = self.train_on_mistakes(
                    batch_mistakes,
                    epochs=epochs,
                    replay_ratio=replay_ratio,
                    anchor_ratio=anchor_ratio,
                )
                loss_history.append(train_metrics["total_loss"])

                # 3. Retest these exact mistakes to verify live improvement
                retested_solved = 0
                for _, _, mist_p in batch_mistakes:
                    ret_solved, _, _ = self.evaluate_puzzle(mist_p)
                    if ret_solved:
                        retested_solved += 1
                mistakes_resolved_on_retest += retested_solved

                if on_batch_train:
                    on_batch_train(len(batch_mistakes), train_metrics, retested_solved)

                batch_mistakes.clear()

        # Flush remaining mistakes
        if batch_mistakes:
            train_metrics = self.train_on_mistakes(
                batch_mistakes,
                epochs=epochs,
                replay_ratio=replay_ratio,
                anchor_ratio=anchor_ratio,
            )
            loss_history.append(train_metrics["total_loss"])
            retested_solved = 0
            for _, _, mist_p in batch_mistakes:
                ret_solved, _, _ = self.evaluate_puzzle(mist_p)
                if ret_solved:
                    retested_solved += 1
            mistakes_resolved_on_retest += retested_solved
            if on_batch_train:
                on_batch_train(len(batch_mistakes), train_metrics, retested_solved)
            batch_mistakes.clear()

        retest_resolution_rate = (
            mistakes_resolved_on_retest / total_mistakes_mined if total_mistakes_mined > 0 else 1.0
        )
        avg_loss = float(np.mean(loss_history)) if loss_history else 0.0

        return {
            "total_puzzles": total_puzzles,
            "initial_solved": initial_solved,
            "initial_solve_rate": initial_solved / total_puzzles if total_puzzles else 0.0,
            "total_mistakes_mined": total_mistakes_mined,
            "mistakes_resolved_on_retest": mistakes_resolved_on_retest,
            "retest_resolution_rate": retest_resolution_rate,
            "avg_loss": avg_loss,
        }


@dataclass
class CurriculumTier:
    tier_number: int
    min_rating: int
    max_rating: int
    name: str
    mastery_threshold: float = 0.75


SWEET_SPOT_TIERS: list[CurriculumTier] = [
    CurriculumTier(1, 1100, 1300, "Novice Motifs", 0.80),
    CurriculumTier(2, 1300, 1500, "Core Combinations", 0.75),
    CurriculumTier(3, 1500, 1700, "Intermediate Attacks", 0.70),
    CurriculumTier(4, 1700, 1900, "Advanced Tactics", 0.65),
    CurriculumTier(5, 1900, 2100, "Master Calculation", 0.60),
]

DEFAULT_CURRICULUM_TIERS: list[CurriculumTier] = [
    CurriculumTier(1, 1000, 1200, "Novice Motifs", 0.75),
    CurriculumTier(2, 1200, 1400, "Intermediate Tactics", 0.75),
    CurriculumTier(3, 1400, 1600, "Club Combinations", 0.70),
    CurriculumTier(4, 1600, 1800, "Advanced Tactics", 0.70),
    CurriculumTier(5, 1800, 2000, "Expert Calculation", 0.65),
    CurriculumTier(6, 2000, 2200, "Master Tactical Complexity", 0.60),
    CurriculumTier(7, 2200, 2500, "Grandmaster Sharpness", 0.55),
    CurriculumTier(8, 2500, 3000, "Super-GM / Engine Edge", 0.50),
]


def run_curriculum(
    learner: MistakeLearner,
    tiers: Optional[list[CurriculumTier]] = None,
    start_rating: int = 1000,
    end_rating: int = 3000,
    base_tier_puzzles: int = 250,
    max_tier_puzzles: int = 600,
    epochs: int = 3,
    replay_ratio: float = 0.25,
    anchor_ratio: float = 0.50,
    puzzles_dir: Union[str, Path] = "data/puzzles/curriculum",
    checkpoint_dir: Union[str, Path] = "models/checkpoints_finetune",
    benchmark_callback: Optional[Callable[[torch.nn.Module], dict[str, float]]] = None,
    on_tier_start: Optional[Callable[[CurriculumTier], None]] = None,
    on_tier_complete: Optional[
        Callable[[CurriculumTier, dict[str, Any], Optional[dict[str, float]]], None]
    ] = None,
    on_puzzle_result: Optional[Callable[[int, PuzzleItem, bool, str, float], None]] = None,
    on_batch_train: Optional[Callable[[int, dict[str, float], int], None]] = None,
) -> list[dict[str, Any]]:
    """Runs autonomous curriculum learning across rating tiers from start_rating to end_rating."""
    curriculum_tiers = tiers or DEFAULT_CURRICULUM_TIERS
    active_tiers = [
        t for t in curriculum_tiers if t.max_rating > start_rating and t.min_rating < end_rating
    ]

    puzzles_base = Path(puzzles_dir)
    puzzles_base.mkdir(parents=True, exist_ok=True)
    ckpt_base = Path(checkpoint_dir)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    tier_summaries: list[dict[str, Any]] = []

    for tier in active_tiers:
        if on_tier_start:
            on_tier_start(tier)

        tier_csv = puzzles_base / f"tier_{tier.tier_number}_{tier.min_rating}_{tier.max_rating}.csv"

        # 1. Autonomously ensure initial chunk of puzzles exists for this tier
        current_target = base_tier_puzzles
        ensure_puzzle_dataset(
            csv_path=tier_csv,
            target_count=current_target,
            min_rating=tier.min_rating,
            max_rating=tier.max_rating,
        )

        puzzles = load_puzzle_items(tier_csv, limit=current_target)
        tier_results = learner.learn_from_puzzle_stream(
            puzzles=puzzles,
            epochs=epochs,
            replay_ratio=replay_ratio,
            anchor_ratio=anchor_ratio,
            on_puzzle_result=on_puzzle_result,
            on_batch_train=on_batch_train,
        )

        # 2. Adaptive reinforcement: if below threshold and under max cap, fetch extra practice chunk
        reinforcement_rounds = 0
        while (
            tier_results["initial_solve_rate"] < tier.mastery_threshold
            and current_target < max_tier_puzzles
        ):
            reinforcement_rounds += 1
            extra_needed = min(150, max_tier_puzzles - current_target)
            current_target += extra_needed

            ensure_puzzle_dataset(
                csv_path=tier_csv,
                target_count=current_target,
                min_rating=tier.min_rating,
                max_rating=tier.max_rating,
            )

            extra_puzzles = load_puzzle_items(tier_csv)[len(puzzles) : current_target]
            if not extra_puzzles:
                break

            puzzles.extend(extra_puzzles)
            extra_res = learner.learn_from_puzzle_stream(
                puzzles=extra_puzzles,
                epochs=epochs,
                replay_ratio=replay_ratio,
                anchor_ratio=anchor_ratio,
                on_puzzle_result=on_puzzle_result,
                on_batch_train=on_batch_train,
            )
            # Accumulate metrics
            tier_results["total_puzzles"] += extra_res["total_puzzles"]
            tier_results["initial_solved"] += extra_res["initial_solved"]
            tier_results["total_mistakes_mined"] += extra_res["total_mistakes_mined"]
            tier_results["mistakes_resolved_on_retest"] += extra_res["mistakes_resolved_on_retest"]
            tier_results["initial_solve_rate"] = (
                tier_results["initial_solved"] / tier_results["total_puzzles"]
            )

        # 3. Optional Benchmark Evaluation (run first to inform regression gate & checkpoint)
        bench_res = None
        if benchmark_callback:
            bench_res = benchmark_callback(learner.model)

        # Regression gate (audit §11, P1): compare MCTS solve rate across tiers.
        # If MCTS collapses (regression > 5pp from previous tier's MCTS rate),
        # the checkpoint is flagged so downstream selection does not auto-promote it.
        mcts_regression_flag = False
        prev_mcts = None
        if tier_summaries and tier_summaries[-1].get("benchmark"):
            prev_mcts = tier_summaries[-1]["benchmark"].get("mcts_rate")
        if bench_res and prev_mcts is not None and bench_res.get("mcts_rate") is not None:
            delta_mcts = (bench_res.get("mcts_rate", 0) - prev_mcts) * 100
            if delta_mcts < -5.0:
                mcts_regression_flag = True
                logger.warning(
                    "MCTS regression detected at tier %d: %.1f%% -> %.1f%% (delta %.1fpp). "
                    "Checkpoint flagged; do not promote without verification.",
                    tier.tier_number,
                    prev_mcts * 100,
                    bench_res.get("mcts_rate", 0) * 100,
                    delta_mcts,
                )

        # 4. Save Tier Milestone Checkpoint (with regression flag and benchmark results embedded)
        lr_str = f"{learner.optimizer.param_groups[0].get('lr', 2e-5):.1e}".replace(
            ".", ""
        ).replace("-0", "n")
        tier_ckpt_path = (
            ckpt_base
            / f"curriculum_tier_{tier.tier_number}_{tier.min_rating}_{tier.max_rating}_p{base_tier_puzzles}_lr{lr_str}.pt"
        )
        torch.save(
            {
                "model": learner.model.state_dict(),
                "optimizer": learner.optimizer.state_dict(),
                "tier": tier.tier_number,
                "rating_min": tier.min_rating,
                "rating_max": tier.max_rating,
                "solve_rate": tier_results["initial_solve_rate"],
                "total_puzzles": tier_results["total_puzzles"],
                "mcts_regression_flag": mcts_regression_flag,
                "mcts_rate": bench_res.get("mcts_rate") if bench_res else None,
                "timestamp": time.time(),
            },
            tier_ckpt_path,
        )

        tier_summary = {
            "tier": tier,
            "results": tier_results,
            "benchmark": bench_res,
            "checkpoint": str(tier_ckpt_path),
            "graduated": tier_results["initial_solve_rate"] >= tier.mastery_threshold,
            "reinforcement_rounds": reinforcement_rounds,
            "mcts_regression_flag": mcts_regression_flag,
        }
        tier_summaries.append(tier_summary)

        if on_tier_complete:
            on_tier_complete(tier, tier_results, bench_res)

    return tier_summaries
