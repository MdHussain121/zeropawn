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
from python.eval.sprt import compute_sprt_llr, wilson_interval
from python.model.chess_net import ChessNet


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
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
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
    ):
        self.device = device
        self.sims = sims
        self.workers = workers
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
        # Audit RL-20: record adjudicated endings so SPRT evidence can be
        # audited for value-cutoff-decided games (symmetric between arms, but
        # still value-head-dependent).
        adj: dict[str, object] = {"adjudicated": False, "adjudication_reason": None}

        def _done(result: str, reason: Optional[str] = None) -> tuple[str, list[str]]:
            if reason is not None:
                adj["adjudicated"] = True
                adj["adjudication_reason"] = reason
            if info is not None:
                info.update(adj)
            return result, moves

        while plies < max_plies:
            if game.is_game_over():
                break
            if game.half_move_clock() >= 60:
                return _done("1/2-1/2", reason="fifty_move_countdown")

            white_to_move = bool(game.side_to_move())
            cand_turn = white_to_move == cand_white
            mcts = mcts_cand if cand_turn else mcts_base
            eval_fn = cand_eval if cand_turn else base_eval

            res = mcts.search(game, self.sims, eval_fn)
            move_uci = game.action_to_string(res.best_action)
            root_val = res.root_value

            if not move_uci or not game.is_legal(move_uci):
                loser_is_white = white_to_move
                return _done("0-1" if loser_is_white else "1-0")

            # Root-value adjudication: 4 consecutive plies beyond ±0.92 ends the
            # game (both sides see the same evidence; symmetric between arms).
            if root_val < -0.92:
                consecutive_hopeless += 1
                if consecutive_hopeless >= 4:
                    loser_is_white = white_to_move
                    return _done(
                        "0-1" if loser_is_white else "1-0", reason="value_cutoff_loss"
                    )
            else:
                consecutive_hopeless = 0

            if root_val > 0.92:
                consecutive_winning += 1
                if consecutive_winning >= 4:
                    winner_is_white = white_to_move
                    return _done("1-0" if winner_is_white else "0-1", reason="value_cutoff_win")
            else:
                consecutive_winning = 0

            game.make_move_uci(move_uci)
            moves.append(move_uci)
            plies += 1

        if game.is_checkmate():
            loser_is_white = bool(game.side_to_move())
            return _done("0-1" if loser_is_white else "1-0", reason="checkmate")
        return _done("1/2-1/2", reason="max_plies_cap")

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

        games_played: list[dict[str, Any]] = []
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    games_played = saved.get("games", [])
                    print(f"Resuming {name}: loaded {len(games_played)} existing games")
            except Exception as e:
                print(f"Warning: could not load state file {state_file}: {e}")

        def _tally() -> tuple[int, int, int]:
            return (
                sum(1 for g in games_played if g["cand_outcome"] == "win"),
                sum(1 for g in games_played if g["cand_outcome"] == "draw"),
                sum(1 for g in games_played if g["cand_outcome"] == "loss"),
            )

        cand_wins, cand_draws, cand_losses = _tally()
        sprt = compute_sprt_llr(cand_wins, cand_draws, cand_losses, elo0=elo0, elo1=elo1)

        if len(games_played) >= max_games or sprt["decision"] != "CONTINUE":
            final_status = sprt["decision"] if sprt["decision"] != "CONTINUE" else "INCONCLUSIVE"
            print(f"Candidate {name} already complete: {final_status} after {len(games_played)} games.")
            return {
                "name": name,
                "cand_cfg": cand_cfg,
                "sprt": sprt,
                "final_status": final_status,
                "games_count": len(games_played),
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
                for (p_idx, opening_idx, fen, cand_white, seed) in tasks:
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
                    if board_res == "1/2-1/2":
                        cand_outcome = "draw"
                    elif (board_res == "1-0" and cand_white) or (board_res == "0-1" and not cand_white):
                        cand_outcome = "win"
                    else:
                        cand_outcome = "loss"
                    batch_results.append(
                        {
                            "game_idx": len(games_played) + len(batch_results) + 1,
                            "pair_idx": p_idx,
                            "opening_idx": opening_idx,
                            "opening_fen": fen,
                            "cand_white": cand_white,
                            "board_result": board_res,
                            "cand_outcome": cand_outcome,
                            "moves": moves,
                            "seed": seed,
                            "adjudicated": bool(game_info.get("adjudicated", False)),
                            "adjudication_reason": game_info.get("adjudication_reason"),
                        }
                    )

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
                    pf.write(" ".join(gr["moves"]) + f' {gr["board_result"]}\n\n')

            cand_wins, cand_draws, cand_losses = _tally()
            sprt = compute_sprt_llr(cand_wins, cand_draws, cand_losses, elo0=elo0, elo1=elo1)
            elo_str = f"{sprt['elo_diff']:+.1f}" if sprt["elo_diff"] is not None else "N/A"
            lo_str = f"{sprt['elo_lo']:+.1f}" if sprt["elo_lo"] is not None else "N/A"
            hi_str = f"{sprt['elo_hi']:+.1f}" if sprt["elo_hi"] is not None else "N/A"
            print(
                f"[{name}] Games: {sprt['n']:3d}/{max_games} | W: {sprt['wins']:2d} D: {sprt['draws']:2d} "
                f"L: {sprt['losses']:2d} | Score: {sprt['score']:.3f} | Elo: {elo_str} [{lo_str}, {hi_str}] | "
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
                        "candidate_checkpoint": self.candidate_checkpoint,
                        "candidate_checkpoint_sha256": _file_sha256(self.candidate_checkpoint),
                        "base_checkpoint": self.base_checkpoint,
                        "base_checkpoint_sha256": _file_sha256(self.base_checkpoint),
                        "device": str(self.device),
                        "sims": self.sims,
                        "openings_file": "data/openings/eval_openings_50.epd",
                        "openings_sha256": _file_sha256("data/openings/eval_openings_50.epd"),
                        "sprt": sprt,
                        "games": games_played,
                    },
                    sf,
                    indent=2,
                )

            if sprt["decision"] != "CONTINUE":
                print(f"--> SPRT STOP for {name}: {sprt['decision']} at {sprt['n']} games!", flush=True)
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
            "device": str(self.device),
            "sims": self.sims,
            "openings_file": "data/openings/eval_openings_50.epd",
            "openings_sha256": _file_sha256("data/openings/eval_openings_50.epd"),
            "sprt": sprt,
            "final_status": final_status,
            "games_count": len(games_played),
            "pgn_path": str(pgn_file),
            # Audit RL-20: adjudication provenance for the SPRT evidence.
            "adjudicated_games": sum(1 for g in games_played if g.get("adjudicated")),
            "adjudicated_value_cutoff_wins": sum(
                1
                for g in games_played
                if g.get("adjudicated") and g.get("cand_outcome") == "win"
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
    parser.add_argument("--candidate-checkpoint", default=None, help="Candidate weights (promotion test)")
    parser.add_argument("--base-checkpoint", default=None, help="Base weights (default: same as candidate)")
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
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
        elo_str = f"{s['elo_diff']:+.1f} [{s['elo_lo']:+.1f}, {s['elo_hi']:+.1f}]" if s["elo_diff"] is not None else "N/A"
        print(
            f"{r['name']:<24} | {wdl:<12} | {s['score']:.3f} | {elo_str:<22} | "
            f"LLR {s['llr']:+.2f} | {r['final_status']}"
        )
    print("=" * 90)

    summary_file = outdir / "sprt_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"matches": results}, f, indent=2)
    print(f"Saved SPRT results to {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
