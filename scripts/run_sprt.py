"""SPRT matches: search-parameter screens and candidate-vs-base promotion tests.

Ported from the validated Stage 1 screen harness (``scripts/run_sprt_screen.py``
on branch ``search-study``, commit 71cc6a5 lineage) with three changes:

1. The SPRT math lives in ``python/eval/sprt.py`` (shared, unit-tested).
2. Evaluation encodes the live game state (repetition plane faithful) instead
   of a FEN-rebuilt ChessGame.
3. Candidate and base may be DIFFERENT checkpoints (promotion testing), not
   only one model with different search parameters.

Protocol defaults: H0 = +0 Elo, H1 = +30 Elo, alpha = beta = 0.05 (bounds
±2.944), engine-vs-engine at equal simulation budgets, colors alternated per
pair, resumable state, PGNs and JSON summary persisted. INCONCLUSIVE is a
valid outcome and never promotes anything.

Cap semantics (Audit S5-05, porting the MatchRunner F-01 contract): a game
that reaches the 150-ply cap without a decisive or rule-based outcome is
recorded with board result "*" and outcome_class "capped" (default
``cap_policy="exclude"``) and is EXCLUDED from the SPRT score — it is never
counted as a candidate draw. Capped games are counted per candidate color
and in total in the state/summary output so the bias is visible, and the
summary carries capped-as-draw / capped-as-loss / capped-by-material
sensitivity decisions that are reported alongside — never instead of — the
primary (exclude) decision. Resume compatibility: state files written by
the pre-S5-05 schema (records without ``outcome_class`` /
``termination_reason`` / ``plies``) load unchanged; such legacy records are
treated as UNCAPPED and keep their recorded result (the historical
cap-as-draw residue in a legacy state is visible via its
``adjudication_reason == "max_plies_cap"`` but is not reinterpreted).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
import zeropawn_cpp
from python.cli.common import get_default_checkpoint_path
from python.eval.sprt import compute_sprt_llr
from python.model.chess_net import ChessNet

# Shared F-01/S5-05 classification contract: ONE implementation of the
# capped-game (cap != draw) semantics for MatchRunner and this SPRT harness.
from python.eval.match_runner import (
    _compute_material_balance,
    classify_game_record,
    classify_game_records,
    is_capped_game_record,
)

VALID_CAP_POLICIES = ("exclude", "draw")

# Fine-grained shared-classifier outcomes -> candidate-relative SPRT outcome.
_RECORD_OUTCOME_TO_SPRT = {
    "decisive_win": "win",
    "forfeit_win": "win",
    "decisive_loss": "loss",
    "forfeit_loss": "loss",
    "genuine_draw": "draw",
    "capped": "capped",
    "uncounted": "uncounted",
}


def classify_sprt_record(record: dict) -> str:
    """Candidate-relative outcome of one SPRT game record (shared contract).

    Uses MatchRunner's F-01 classifier: a ply-cap termination is "capped"
    (never "draw"), forfeits count as win/loss, and incomplete/invalid
    records are "uncounted" (reported, never scored).
    """
    return _RECORD_OUTCOME_TO_SPRT[classify_game_record(record)]


def sprt_tally(games: list) -> dict:
    """Tally SPRT game records through the shared F-01 classifier (S5-05).

    Returns the shared outcome-class counters plus convenience keys:
      wins/draws/losses  — the SPRT score inputs (capped games EXCLUDED)
      capped_games       — total capped (ply-cap) games
      capped_candidate_white / capped_candidate_black — per-color cap counts
      uncounted_games    — incomplete/invalid records (never scored)
    """
    counts = classify_game_records(games)
    counts["wins"] = counts["decisive_wins"] + counts["forfeit_wins"]
    counts["draws"] = counts["genuine_draws"]
    counts["losses"] = counts["decisive_losses"] + counts["forfeit_losses"]
    capped_white = sum(
        1
        for g in games
        if is_capped_game_record(g) and bool(g.get("cand_white", g.get("engine_is_white")))
    )
    counts["capped_candidate_white"] = capped_white
    counts["capped_candidate_black"] = counts["capped_games"] - capped_white
    return counts


def capped_sensitivity(
    games: list,
    elo0: float = 0.0,
    elo1: float = 30.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict:
    """Sensitivity of the SPRT decision to the capped-game scoring policy (S5-05).

    The PRIMARY decision (produced by ``SprtMatchRunner.run_sprt``) always
    EXCLUDES capped games. This reports what the decision would have been if
    capped games were scored as draws (the legacy pre-F-01 bias), as losses
    (worst case), or adjudicated by material balance (>= +2 pawns win,
    <= -2 pawns loss, else draw — capped records without a recorded material
    balance count as draws there). Sensitivity entries never replace the
    primary decision.
    """
    counts = sprt_tally(games)
    wins = counts["wins"]
    draws = counts["draws"]
    losses = counts["losses"]
    capped = counts["capped_games"]

    primary = compute_sprt_llr(wins, draws, losses, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)
    as_draw = compute_sprt_llr(
        wins, draws + capped, losses, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta
    )
    as_loss = compute_sprt_llr(
        wins, draws, losses + capped, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta
    )

    cm_wins, cm_draws, cm_losses = wins, draws, losses
    for g in games:
        if is_capped_game_record(g):
            mat = int(g.get("final_material_balance", 0) or 0)
            if mat >= 2:
                cm_wins += 1
            elif mat <= -2:
                cm_losses += 1
            else:
                cm_draws += 1
    by_material = compute_sprt_llr(
        cm_wins, cm_draws, cm_losses, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta
    )

    return {
        "primary": {"llr": primary["llr"], "decision": primary["decision"]},
        "capped_games": capped,
        "capped_candidate_white": counts["capped_candidate_white"],
        "capped_candidate_black": counts["capped_candidate_black"],
        "capped_as_draw": {"llr": as_draw["llr"], "decision": as_draw["decision"]},
        "capped_as_loss": {"llr": as_loss["llr"], "decision": as_loss["decision"]},
        "capped_by_material": {
            "llr": by_material["llr"],
            "decision": by_material["decision"],
        },
    }


def _file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


# Pre-registered search-parameter candidates from the 2026-10-16 screen
# (locked baseline: c_puct=2.0, fpu_reduction=0.2).
CANDIDATE_SEARCH_PARAMS = {
    "cpuct_1.5": {"c_puct": 1.5, "fpu_reduction": 0.2},
    "cpuct_2.5": {"c_puct": 2.5, "fpu_reduction": 0.2},
    "cpuct_3.0": {"c_puct": 3.0, "fpu_reduction": 0.2},
    "fpu_0.0": {"c_puct": 2.0, "fpu_reduction": 0.0},
    "fpu_0.1": {"c_puct": 2.0, "fpu_reduction": 0.1},
    "alphazero_dynamic": {"c_puct": 1.26, "fpu_reduction": 0.2},
}
BASELINE_SEARCH_PARAMS = {"c_puct": 2.0, "fpu_reduction": 0.2}


def check_ram_watchdog(max_sys_percent: float = 90.0, max_proc_mb: float = 4096.0) -> None:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    if vm.percent > max_sys_percent or rss_mb > max_proc_mb:
        print(
            f"[RAM WATCHDOG ALERT] System RAM {vm.percent:.1f}%, "
            f"Process RSS {rss_mb:.1f}MB! Triggering cleanup...",
            flush=True,
        )
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model(path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


class SprtMatchRunner:
    """Engine-vs-engine MCTS match between a candidate and a base setup."""

    def __init__(
        self,
        candidate_checkpoint: str,
        base_checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sims: int = 200,
        workers: int = 8,
        cap_policy: str = "exclude",
    ):
        if cap_policy not in VALID_CAP_POLICIES:
            raise ValueError(
                f"Unknown cap_policy {cap_policy!r}; expected one of {VALID_CAP_POLICIES}."
            )
        self.device = device
        self.sims = sims
        self.workers = workers
        self.cap_policy = cap_policy
        self.candidate_checkpoint = candidate_checkpoint
        self.base_checkpoint = base_checkpoint

        self._model_lock = threading.Lock()
        self.candidate_model = _load_model(candidate_checkpoint, device)
        self.base_model = (
            self.candidate_model
            if base_checkpoint == candidate_checkpoint
            else _load_model(base_checkpoint, device)
        )

        self.openings = self._load_openings(Path("data/openings/eval_openings_50.epd"))

    @staticmethod
    def _load_openings(path: Path) -> list[str]:
        fens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    fen = line.split(";")[0].strip()
                    if fen:
                        fens.append(fen)
        return fens

    def _make_eval_fn(self, model: torch.nn.Module):
        def eval_fn(game: Any) -> tuple[Any, float]:
            # Live-game encoding: a FEN round-trip would drop the repetition
            # plane (plane 18) and mismatch the training input distribution.
            planes = zeropawn_cpp.encode_state(game)
            planes_t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
            if self.device == "cuda":
                planes_t = planes_t.to(memory_format=torch.channels_last)
            with torch.no_grad():
                with self._model_lock:
                    if self.device == "cuda":
                        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                            p_logits, v_wdl = model(planes_t)
                    else:
                        p_logits, v_wdl = model(planes_t)
                probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy()
                v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
                scalar_val = float(v_probs[0] - v_probs[2])
            import numpy as np

            return np.ascontiguousarray(probs, dtype=np.float32), scalar_val

        return eval_fn

    def play_single_game(
        self,
        opening_fen: str,
        cand_white: bool,
        cand_cfg_dict: dict[str, float],
        base_cfg_dict: dict[str, float],
        game_seed: int,
        max_plies: int = 150,
        info: Optional[dict] = None,
    ) -> tuple[str, list[str]]:
        cfg_cand = zeropawn_cpp.MCTSConfig()
        cfg_cand.c_puct = cand_cfg_dict.get("c_puct", 2.0)
        cfg_cand.fpu_reduction = cand_cfg_dict.get("fpu_reduction", 0.2)
        cfg_cand.add_noise = False
        cfg_cand.virtual_loss = 1
        cfg_cand.seed = game_seed
        mcts_cand = zeropawn_cpp.MCTS(cfg_cand)

        cfg_base = zeropawn_cpp.MCTSConfig()
        cfg_base.c_puct = base_cfg_dict.get("c_puct", 2.0)
        cfg_base.fpu_reduction = base_cfg_dict.get("fpu_reduction", 0.2)
        cfg_base.add_noise = False
        cfg_base.virtual_loss = 1
        cfg_base.seed = game_seed + 10000
        mcts_base = zeropawn_cpp.MCTS(cfg_base)

        cand_eval = self._make_eval_fn(self.candidate_model)
        base_eval = self._make_eval_fn(self.base_model)

        game = zeropawn_cpp.ChessGame(opening_fen)
        moves: list[str] = []
        plies = 0
        consecutive_hopeless = 0
        consecutive_winning = 0
        # Audit RL-20 & F-01/S5-05: record every terminated ending with explicit
        # outcome classifications, reasons, and material balance — mirroring
        # MatchRunner._play_game_worker's provenance contract so both harnesses
        # classify endings identically.
        adj: dict[str, object] = {"adjudicated": False, "adjudication_reason": None}

        def _finish(
            result: str, reason: Optional[str] = None, outcome_class: Optional[str] = None
        ) -> tuple[str, list[str]]:
            if reason is not None:
                adj["termination_reason"] = reason
            if outcome_class is not None:
                adj["outcome_class"] = outcome_class
            else:
                if result in ("1-0", "0-1"):
                    adj["outcome_class"] = "decisive"
                elif result == "1/2-1/2":
                    adj["outcome_class"] = "genuine_draw"
                else:
                    adj["outcome_class"] = "capped"
            # "Adjudicated" = ended by a proactive adjudication or the ply cap
            # rather than by natural completion (MatchRunner's RL-20 semantics:
            # value cutoffs, the 50-move countdown, and the ply cap — NOT
            # checkmate/stalemate/rule draws).
            is_adjudicated = reason in (
                "value_cutoff_loss",
                "value_cutoff_win",
                "fifty_move_countdown",
                "max_plies_cap",
            )
            adj["adjudicated"] = is_adjudicated
            if is_adjudicated:
                adj["adjudication_reason"] = reason
            adj["plies"] = plies
            adj["final_fen"] = game.get_fen()
            adj["final_material_balance"] = _compute_material_balance(
                opening_fen, moves, cand_white
            )
            if info is not None:
                info.update(adj)
            return result, moves

        while plies < max_plies:
            if game.is_game_over():
                break
            if game.half_move_clock() >= 60:
                return _finish(
                    "1/2-1/2", reason="fifty_move_countdown", outcome_class="genuine_draw"
                )

            white_to_move = bool(game.side_to_move())
            cand_turn = white_to_move == cand_white
            mcts = mcts_cand if cand_turn else mcts_base
            eval_fn = cand_eval if cand_turn else base_eval

            res = mcts.search(game, self.sims, eval_fn)
            move_uci = game.action_to_string(res.best_action)
            root_val = res.root_value

            if not move_uci or not game.is_legal(move_uci):
                loser_is_white = white_to_move
                return _finish(
                    "0-1" if loser_is_white else "1-0",
                    reason="forfeit_illegal_move",
                    outcome_class="forfeit",
                )

            # Root-value adjudication: 4 consecutive plies beyond ±0.92 ends the
            # game (both sides see the same evidence; symmetric between arms).
            if root_val < -0.92:
                consecutive_hopeless += 1
                if consecutive_hopeless >= 4:
                    loser_is_white = white_to_move
                    return _finish(
                        "0-1" if loser_is_white else "1-0",
                        reason="value_cutoff_loss",
                        outcome_class="decisive",
                    )
            else:
                consecutive_hopeless = 0

            if root_val > 0.92:
                consecutive_winning += 1
                if consecutive_winning >= 4:
                    winner_is_white = white_to_move
                    return _finish(
                        "1-0" if winner_is_white else "0-1",
                        reason="value_cutoff_win",
                        outcome_class="decisive",
                    )
            else:
                consecutive_winning = 0

            game.make_move_uci(move_uci)
            moves.append(move_uci)
            plies += 1

        if game.is_checkmate():
            loser_is_white = bool(game.side_to_move())
            return _finish(
                "0-1" if loser_is_white else "1-0", reason="checkmate", outcome_class="decisive"
            )
        if plies >= max_plies:
            # Audit F-01 / S5-05: the ply cap reached without a decisive or
            # rule-based outcome must NOT be treated as a genuine chess draw.
            # Default policy "exclude": board result "*" -> classified "capped"
            # and excluded from the SPRT score. cap_policy="draw" keeps the
            # legacy "1/2-1/2" board string but STILL records outcome_class
            # "capped", so the classification (and any resumed tally) is
            # policy-independent.
            cap_pol = getattr(self, "cap_policy", "exclude")
            if cap_pol == "draw":
                return _finish("1/2-1/2", reason="max_plies_cap", outcome_class="capped")
            return _finish("*", reason="max_plies_cap", outcome_class="capped")
        # Natural game-over without checkmate (stalemate, insufficient material,
        # repetition, ...): a genuine rule draw. (The pre-S5-05 harness labeled
        # this ending "max_plies_cap"; that mislabel is corrected here.)
        return _finish("1/2-1/2", reason="rule_draw", outcome_class="genuine_draw")

    def run_sprt(
        self,
        name: str,
        cand_cfg: dict[str, float],
        base_cfg: dict[str, float],
        outdir: Path,
        max_games: int = 200,
        elo0: float = 0.0,
        elo1: float = 30.0,
    ) -> dict[str, Any]:
        outdir.mkdir(parents=True, exist_ok=True)
        state_file = outdir / f"{name}_state.json"
        pgn_file = outdir / f"{name}.pgn"
        cap_policy = getattr(self, "cap_policy", "exclude")

        games_played: list[dict[str, Any]] = []
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    games_played = saved.get("games", [])
                    print(f"Resuming {name}: loaded {len(games_played)} existing games")
            except Exception as e:
                print(f"Warning: could not load state file {state_file}: {e}")

        def _tally() -> dict[str, Any]:
            # Audit S5-05: reclassify EVERY record through the shared F-01
            # classifier on each call (never trust the stored cand_outcome).
            # Capped games are EXCLUDED from the SPRT score; legacy pre-S5-05
            # records (no outcome_class/termination_reason) classify from
            # their board result alone and default to uncapped.
            return sprt_tally(games_played)

        counts = _tally()
        cand_wins, cand_draws, cand_losses = counts["wins"], counts["draws"], counts["losses"]
        sprt = compute_sprt_llr(cand_wins, cand_draws, cand_losses, elo0=elo0, elo1=elo1)

        if len(games_played) >= max_games or sprt["decision"] != "CONTINUE":
            final_status = sprt["decision"] if sprt["decision"] != "CONTINUE" else "INCONCLUSIVE"
            print(
                f"Candidate {name} already complete: {final_status} after {len(games_played)} games."
            )
            return {
                "name": name,
                "cand_cfg": cand_cfg,
                "cap_policy": cap_policy,
                "sprt": sprt,
                "final_status": final_status,
                "games_count": len(games_played),
                "capped_games": counts["capped_games"],
                "capped_candidate_white": counts["capped_candidate_white"],
                "capped_candidate_black": counts["capped_candidate_black"],
                "uncounted_games": counts["uncounted_games"],
                "capped_sensitivity": capped_sensitivity(games_played, elo0=elo0, elo1=elo1),
                "pgn_path": str(pgn_file),
            }

        print(
            f"\n{'=' * 70}\nSPRT: {name}\n"
            f"cand_cfg={cand_cfg} vs base_cfg={base_cfg} "
            f"(H0=+{elo0} H1=+{elo1}, max {max_games} games)\n{'=' * 70}",
            flush=True,
        )

        start_pair_idx = len(games_played) // 2
        total_pairs = max_games // 2
        pair_indices = list(range(start_pair_idx, total_pairs))
        batch_size_pairs = max(1, self.workers // 2)

        for b_start in range(0, len(pair_indices), batch_size_pairs):
            check_ram_watchdog()
            batch_pairs = pair_indices[b_start : b_start + batch_size_pairs]
            tasks = []
            for p_idx in batch_pairs:
                opening_idx = p_idx % len(self.openings)
                fen = self.openings[opening_idx]
                tasks.append((p_idx, opening_idx, fen, True, 100000 + p_idx * 2))
                tasks.append((p_idx, opening_idx, fen, False, 100000 + p_idx * 2 + 1))

            t0 = time.time()
            batch_results: list[dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {}
                for p_idx, opening_idx, fen, cand_white, seed in tasks:
                    # Audit RL-20: per-game adjudication provenance.
                    game_info: dict[str, object] = {}
                    futures[
                        executor.submit(
                            self.play_single_game,
                            fen,
                            cand_white,
                            cand_cfg,
                            base_cfg,
                            seed,
                            150,
                            game_info,
                        )
                    ] = (p_idx, opening_idx, fen, cand_white, seed, game_info)
                for fut in concurrent.futures.as_completed(futures):
                    p_idx, opening_idx, fen, cand_white, seed, game_info = futures[fut]
                    board_res, moves = fut.result()
                    record = {
                        "game_idx": len(games_played) + len(batch_results) + 1,
                        "pair_idx": p_idx,
                        "opening_idx": opening_idx,
                        "opening_fen": fen,
                        "cand_white": cand_white,
                        "board_result": board_res,
                        "moves": moves,
                        "seed": seed,
                        "adjudicated": bool(game_info.get("adjudicated", False)),
                        "adjudication_reason": game_info.get("adjudication_reason"),
                        # Audit F-01/S5-05: full termination provenance so an
                        # interrupted SPRT can resume and reclassify
                        # consistently (capped flag via outcome_class /
                        # termination_reason, plus plies and material balance
                        # for the sensitivity analyses).
                        "termination_reason": game_info.get("termination_reason"),
                        "outcome_class": game_info.get("outcome_class"),
                        "plies": int(game_info.get("plies", len(moves))),
                        "final_material_balance": game_info.get("final_material_balance", 0),
                    }
                    # Shared F-01 classification: a ply cap is "capped",
                    # never "draw" (cap_policy="exclude" default).
                    record["cand_outcome"] = classify_sprt_record(record)
                    batch_results.append(record)

            batch_results.sort(key=lambda r: r["game_idx"])
            games_played.extend(batch_results)
            dt = time.time() - t0

            with open(pgn_file, "a", encoding="utf-8") as pf:
                for gr in batch_results:
                    white_name = f"{name}" if gr["cand_white"] else "Base"
                    black_name = "Base" if gr["cand_white"] else f"{name}"
                    pf.write(f'[Event "SPRT: {name} vs Base"]\n')
                    pf.write(f'[White "{white_name}"]\n')
                    pf.write(f'[Black "{black_name}"]\n')
                    pf.write(f'[Result "{gr["board_result"]}"]\n')
                    pf.write(f'[FEN "{gr["opening_fen"]}"]\n')
                    pf.write(f'[CandidateWhite "{gr["cand_white"]}"]\n\n')
                    pf.write(" ".join(gr["moves"]) + f" {gr['board_result']}\n\n")

            counts = _tally()
            cand_wins, cand_draws, cand_losses = counts["wins"], counts["draws"], counts["losses"]
            sprt = compute_sprt_llr(cand_wins, cand_draws, cand_losses, elo0=elo0, elo1=elo1)
            elo_str = f"{sprt['elo_diff']:+.1f}" if sprt["elo_diff"] is not None else "N/A"
            lo_str = f"{sprt['elo_lo']:+.1f}" if sprt["elo_lo"] is not None else "N/A"
            hi_str = f"{sprt['elo_hi']:+.1f}" if sprt["elo_hi"] is not None else "N/A"
            print(
                f"[{name}] Games: {sprt['n']:3d}/{max_games} | W: {sprt['wins']:2d} D: {sprt['draws']:2d} "
                f"L: {sprt['losses']:2d} | Capped(excluded): {counts['capped_games']} "
                f"(cand-W: {counts['capped_candidate_white']}, cand-B: {counts['capped_candidate_black']}) | "
                f"Score: {sprt['score']:.3f} | Elo: {elo_str} [{lo_str}, {hi_str}] | "
                f"LLR: {sprt['llr']:+.2f} (bounds: [{sprt['lower_bound']:.2f}, {sprt['upper_bound']:.2f}]) | "
                f"Batch {len(batch_results)} games took {dt:.1f}s",
                flush=True,
            )

            with open(state_file, "w", encoding="utf-8") as sf:
                json.dump(
                    {
                        "name": name,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "git_commit": _git_commit(),
                        "cand_cfg": cand_cfg,
                        "base_cfg": base_cfg,
                        "cap_policy": cap_policy,
                        "candidate_checkpoint": self.candidate_checkpoint,
                        "candidate_checkpoint_sha256": _file_sha256(self.candidate_checkpoint),
                        "base_checkpoint": self.base_checkpoint,
                        "base_checkpoint_sha256": _file_sha256(self.base_checkpoint),
                        "device": str(self.device),
                        "sims": self.sims,
                        "openings_file": "data/openings/eval_openings_50.epd",
                        "openings_sha256": _file_sha256("data/openings/eval_openings_50.epd"),
                        "sprt": sprt,
                        # Audit S5-05: explicit cap accounting so the
                        # capped-excluded share of the sample is visible.
                        "capped_games": counts["capped_games"],
                        "capped_candidate_white": counts["capped_candidate_white"],
                        "capped_candidate_black": counts["capped_candidate_black"],
                        "uncounted_games": counts["uncounted_games"],
                        "games": games_played,
                    },
                    sf,
                    indent=2,
                )

            if sprt["decision"] != "CONTINUE":
                print(
                    f"--> SPRT STOP for {name}: {sprt['decision']} at {sprt['n']} games!",
                    flush=True,
                )
                break

        final_status = sprt["decision"] if sprt["decision"] != "CONTINUE" else "INCONCLUSIVE"
        return {
            "name": name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_commit(),
            "candidate_checkpoint": self.candidate_checkpoint,
            "candidate_checkpoint_sha256": _file_sha256(self.candidate_checkpoint),
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_sha256": _file_sha256(self.base_checkpoint),
            "cand_cfg": cand_cfg,
            "base_cfg": base_cfg,
            "cap_policy": cap_policy,
            "device": str(self.device),
            "sims": self.sims,
            "openings_file": "data/openings/eval_openings_50.epd",
            "openings_sha256": _file_sha256("data/openings/eval_openings_50.epd"),
            "sprt": sprt,
            "final_status": final_status,
            "games_count": len(games_played),
            "pgn_path": str(pgn_file),
            # Audit S5-05: explicit cap accounting (per candidate color and
            # total) plus capped-as-draw/loss/by-material sensitivity
            # decisions — reported alongside, never replacing, the primary
            # exclude-based decision.
            "capped_games": counts["capped_games"],
            "capped_candidate_white": counts["capped_candidate_white"],
            "capped_candidate_black": counts["capped_candidate_black"],
            "uncounted_games": counts["uncounted_games"],
            "capped_sensitivity": capped_sensitivity(games_played, elo0=elo0, elo1=elo1),
            # Audit RL-20: adjudication provenance for the SPRT evidence
            # (value-cutoff counts keyed on the recorded reason, matching
            # MatchRunner's summary accounting).
            "adjudicated_games": sum(1 for g in games_played if g.get("adjudicated")),
            "adjudicated_value_cutoff_wins": sum(
                1 for g in games_played if g.get("adjudication_reason") == "value_cutoff_win"
            ),
            "adjudicated_value_cutoff_losses": sum(
                1 for g in games_played if g.get("adjudication_reason") == "value_cutoff_loss"
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="SPRT match runner (screen or promotion test)")
    parser.add_argument(
        "--candidate",
        choices=list(CANDIDATE_SEARCH_PARAMS) + ["all", "promotion"],
        default="all",
        help="Pre-registered search-parameter candidate, 'all' of them, or a "
        "candidate-CHECKPOINT-vs-base promotion test",
    )
    parser.add_argument(
        "--candidate-checkpoint", default=None, help="Candidate weights (promotion test)"
    )
    parser.add_argument(
        "--base-checkpoint", default=None, help="Base weights (default: same as candidate)"
    )
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cap-policy",
        choices=list(VALID_CAP_POLICIES),
        default="exclude",
        help="Scoring of games that reach the 150-ply cap without a decisive "
        "or rule-based outcome (Audit F-01/S5-05): 'exclude' (default) "
        "records them as capped and EXCLUDES them from the SPRT score; "
        "'draw' keeps the legacy draw board string but still classifies "
        "them as capped. The cap count is always reported separately.",
    )
    parser.add_argument("--outdir", default="runs/sprt")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=30.0)
    parser.add_argument("--name", default=None, help="Override the result name (promotion tests)")
    args = parser.parse_args()

    try:
        default_ckpt = str(get_default_checkpoint_path())
    except Exception:
        default_ckpt = "models/checkpoints_scaled/best_model.pt"

    base_ckpt = args.base_checkpoint or args.candidate_checkpoint or default_ckpt
    cand_ckpt = args.candidate_checkpoint or base_ckpt

    runner = SprtMatchRunner(
        candidate_checkpoint=cand_ckpt,
        base_checkpoint=base_ckpt,
        sims=args.sims,
        workers=args.workers,
        cap_policy=args.cap_policy,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.candidate == "promotion":
        if not args.candidate_checkpoint:
            parser.error("promotion test requires --candidate-checkpoint")
        name = args.name or f"promotion_{Path(cand_ckpt).stem}"
        result = runner.run_sprt(
            name,
            dict(BASELINE_SEARCH_PARAMS),
            dict(BASELINE_SEARCH_PARAMS),
            outdir,
            max_games=args.max_games,
            elo0=args.elo0,
            elo1=args.elo1,
        )
        results = [result]
    else:
        to_run = list(CANDIDATE_SEARCH_PARAMS) if args.candidate == "all" else [args.candidate]
        results = []
        for cand_name in to_run:
            results.append(
                runner.run_sprt(
                    cand_name,
                    dict(CANDIDATE_SEARCH_PARAMS[cand_name]),
                    dict(BASELINE_SEARCH_PARAMS),
                    outdir,
                    max_games=args.max_games,
                    elo0=args.elo0,
                    elo1=args.elo1,
                )
            )

    print("\n" + "=" * 90)
    print("SPRT FINAL SUMMARY")
    print("=" * 90)
    for r in results:
        s = r["sprt"]
        wdl = f"{s['wins']}/{s['draws']}/{s['losses']}"
        elo_str = (
            f"{s['elo_diff']:+.1f} [{s['elo_lo']:+.1f}, {s['elo_hi']:+.1f}]"
            if s["elo_diff"] is not None
            else "N/A"
        )
        capped = r.get("capped_games", 0)
        sens = r.get("capped_sensitivity") or {}
        sens_str = ""
        if sens.get("capped_games"):
            sens_str = (
                f" | sens: as-draw={sens['capped_as_draw']['decision']},"
                f" as-loss={sens['capped_as_loss']['decision']}"
            )
        print(
            f"{r['name']:<24} | {wdl:<12} | {s['score']:.3f} | {elo_str:<22} | "
            f"LLR {s['llr']:+.2f} | capped(excluded): {capped:<4} | {r['final_status']}"
            f"{sens_str}"
        )
    print("=" * 90)

    summary_file = outdir / "sprt_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"matches": results}, f, indent=2)
    print(f"Saved SPRT results to {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
