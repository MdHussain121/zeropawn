"""Engine-vs-engine head-to-head under equal, recorded conditions.

One harness, two registered uses:
1. Baseline designation (F-11/F-12): candidate A vs candidate B, equal sims,
   fresh shared openings, balanced colors — the evidence a warm-start
   designation requires (docs/checkpoint_registry.json designation_procedure).
2. The pre-registered FPU confirmation experiment: same model, fpu_reduction
   0.2 (default) vs 0.0, fresh openings; adopt fpu 0.0 ONLY if the trinomial
   95% confidence interval's lower bound for A-B exceeds zero.

Every game is appended to a per-game CSV as it finishes (F-15 durability) and
the summary records both model hashes, seeds, openings file, and settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import python  # noqa: F401
import zeropawn_cpp
from python.model.checkpoint_registry import compute_sha256


def load_model(ckpt: str, device: str):
    from python.model.chess_net import ChessNet

    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model") or ck.get("model_state_dict"))
    model.eval()
    return model


def load_openings(path: str, n: int) -> list[str]:
    fens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fen = line.split(";")[0].strip()
            if fen:
                fens.append(fen)
    if not fens:
        raise ValueError(f"no openings in {path}")
    out = []
    for i in range(n):
        out.append(fens[i % len(fens)])
    return out


class PlayerContext:
    """Per-thread pair of engines (own model copies + own MCTS trees)."""

    def __init__(
        self,
        ckpt_a: str,
        ckpt_b: str,
        device: str,
        sims: int,
        c_puct: float,
        fpu_a: float,
        fpu_b: float,
    ):
        self.model_a = load_model(ckpt_a, device)
        self.model_b = load_model(ckpt_b, device)
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.fpu_a = fpu_a
        self.fpu_b = fpu_b

    def _make_mcts(self, fpu: float, seed: int) -> zeropawn_cpp.MCTS:
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.c_puct = self.c_puct
        cfg.fpu_reduction = fpu
        cfg.add_noise = False
        cfg.virtual_loss = 1
        cfg.seed = seed
        cfg.nn_cache_cap = 10000
        return zeropawn_cpp.MCTS(cfg)

    def _eval_factory(self, model):
        device = self.device
        lock = threading.Lock()

        def eval_fn(game):
            planes = zeropawn_cpp.encode_state(game)
            t = torch.from_numpy(planes).unsqueeze(0).to(device)
            if device == "cuda":
                t = t.to(memory_format=torch.channels_last)
            with torch.no_grad():
                with lock:
                    if device == "cuda":
                        with torch.amp.autocast(device, dtype=torch.bfloat16):
                            p_logits, v_wdl = model(t)
                    else:
                        p_logits, v_wdl = model(t)
                p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy()
                v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
                val = float(v_probs[0] - v_probs[2])
            return (np.ascontiguousarray(p_probs, dtype=np.float32), val)

        return eval_fn

    def play_game(self, fen: str, a_is_white: bool, seed: int) -> str:
        """Play one game; returns board-relative result '1-0'/'0-1'/'1/2-1/2'."""
        eval_a = self._eval_factory(self.model_a)
        eval_b = self._eval_factory(self.model_b)
        mcts_a = self._make_mcts(self.fpu_a, seed)
        mcts_b = self._make_mcts(self.fpu_b, seed + 1)
        g = zeropawn_cpp.ChessGame(fen)
        max_plies = 400
        ply = 0
        search_failed = False
        failed_side_white = False

        while not g.is_game_over() and ply < max_plies:
            if hasattr(g, "half_move_clock") and g.half_move_clock() >= 100:
                return ("1/2-1/2", "genuine_draw", "fifty_move_rule")
            white_to_move = g.side_to_move()
            if white_to_move == a_is_white:
                res = mcts_a.search(g, self.sims, eval_a)
            else:
                res = mcts_b.search(g, self.sims, eval_b)
            act = res.best_action
            if act < 0:
                search_failed = True
                failed_side_white = bool(white_to_move)
                break
            g.apply_action(act)
            ply += 1

        if search_failed:
            # F-07: Engine search failure is a forfeit loss for the failing engine, NEVER a draw
            board_res = "0-1" if failed_side_white else "1-0"
            return (board_res, "engine_failure", "search_failure_forfeit")

        if g.is_checkmate():
            loser_is_white = bool(g.side_to_move())
            board_res = "0-1" if loser_is_white else "1-0"
            return (board_res, "decisive", "checkmate")

        if g.is_game_over() or (hasattr(g, "is_draw") and g.is_draw()):
            return ("1/2-1/2", "genuine_draw", "chess_rules_draw")

        # Reached max_plies without terminal state: evaluation cap
        return ("*", "capped", "max_plies_cap")


def trinomial_ci_95(w: int, l: int, d: int) -> tuple[float | None, float | None, float | None]:
    """Elo difference and 95% Agresti-Caffo CI (engine A − engine B)."""
    from python.eval.stats import elo_confidence_interval

    res = elo_confidence_interval(w, d, l)
    return res["elo_diff"], res["lo"], res["hi"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True, help="checkpoint for engine A")
    ap.add_argument("--engine-b", required=True, help="checkpoint for engine B")
    ap.add_argument("--fpu-a", type=float, default=0.2)
    ap.add_argument("--fpu-b", type=float, default=0.2)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--num-openings", type=int, default=20, help="pairs: 2 games per opening")
    ap.add_argument(
        "--openings",
        default=str(_repo_root / "data" / "openings" / "eval_openings_50.epd"),
    )
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="head_to_head")
    ap.add_argument("--output-dir", default="reports/matches")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    openings = load_openings(args.openings, args.num_openings)

    out_dir = _repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    games_csv = out_dir / f"{args.label}_{stamp}_games.csv"
    summary_json = out_dir / f"{args.label}_{stamp}_summary.json"

    with open(games_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "timestamp",
                "opening_idx",
                "a_is_white",
                "result_board",
                "a_result",
                "outcome_class",
                "termination_reason",
                "seed",
                "sims",
                "fpu_a",
                "fpu_b",
            ]
        )

    sha_a = compute_sha256(args.engine_a)
    sha_b = compute_sha256(args.engine_b)
    print(f"Engine A: {args.engine_a} sha={sha_a[:16]} fpu={args.fpu_a}")
    print(f"Engine B: {args.engine_b} sha={sha_b[:16]} fpu={args.fpu_b}")
    print(f"{2 * len(openings)} games, {args.sims} sims, {args.workers} workers on {device}")

    ctx_queue: queue.Queue[PlayerContext] = queue.Queue()
    lock = threading.Lock()
    rows: list[list] = []
    tally = {"a_wins": 0, "draws": 0, "b_wins": 0, "capped": 0, "failures": 0}
    done = {"n": 0}
    pairs_results: dict[int, list[tuple[bool, str]]] = {}
    start = time.time()

    def game_task(idx: int, fen: str):
        ctx = ctx_queue.get()
        try:
            results = []
            for a_is_white, seed in ((True, args.seed + 2 * idx), (False, args.seed + 2 * idx + 1)):
                board_res, outcome_class, term_reason = ctx.play_game(fen, a_is_white, seed)
                if board_res == "*":
                    a_res = "capped"
                elif board_res == ("1-0" if a_is_white else "0-1"):
                    a_res = "win"
                elif board_res == ("0-1" if a_is_white else "1-0"):
                    a_res = "loss"
                else:
                    a_res = "draw"
                results.append((a_is_white, board_res, a_res, outcome_class, term_reason, seed))
            return idx, results
        finally:
            ctx_queue.put(ctx)

    def first_ctx():
        return PlayerContext(
            args.engine_a,
            args.engine_b,
            device,
            args.sims,
            args.c_puct,
            args.fpu_a,
            args.fpu_b,
        )

    import concurrent.futures
    from python.eval.stats import paired_score_and_ci

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        # Prime one context before starting (model load takes seconds).
        for _ in range(args.workers):
            ctx_queue.put(first_ctx())
        futures = [ex.submit(game_task, i, fen) for i, fen in enumerate(openings)]
        try:
            for fut in concurrent.futures.as_completed(futures):
                idx, results = fut.result()
                with lock:
                    pairs_results[idx] = []
                    for a_is_white, board_res, a_res, outcome_class, term_reason, seed in results:
                        row = [
                            time.time(),
                            idx,
                            int(a_is_white),
                            board_res,
                            a_res,
                            outcome_class,
                            term_reason,
                            seed,
                            args.sims,
                            args.fpu_a,
                            args.fpu_b,
                        ]
                        rows.append(row)
                        with open(games_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(row)
                            f.flush()
                        if a_res == "win":
                            tally["a_wins"] += 1
                        elif a_res == "loss":
                            tally["b_wins"] += 1
                        elif a_res == "draw":
                            tally["draws"] += 1
                        elif a_res == "capped":
                            tally["capped"] += 1
                        if outcome_class == "engine_failure":
                            tally["failures"] += 1
                        done["n"] += 1
                        pairs_results[idx].append((a_is_white, a_res))
                    n = done["n"]
                    total = 2 * len(openings)
                    n_eff = tally["a_wins"] + tally["b_wins"] + tally["draws"]
                    score = (tally["a_wins"] + 0.5 * tally["draws"]) / n_eff if n_eff else 0.0
                    print(
                        f"Progress {n:3d}/{total} | A: +{tally['a_wins']} ={tally['draws']} "
                        f"-{tally['b_wins']} (capped={tally['capped']}, fail={tally['failures']}) | "
                        f"score {score * 100:.1f}% | {time.time() - start:.0f}s"
                    )
        except KeyboardInterrupt:
            print("Interrupted; partial tally returned")

    n = done["n"]
    n_eff = tally["a_wins"] + tally["b_wins"] + tally["draws"]
    score = (tally["a_wins"] + 0.5 * tally["draws"]) / n_eff if n_eff else 0.0
    elo_diff, lo, hi = trinomial_ci_95(tally["a_wins"], tally["b_wins"], tally["draws"])

    # Compute paired opening statistics
    pair_scores: list[tuple[float, float]] = []
    for idx, games in pairs_results.items():
        if len(games) == 2:
            w_score = None
            b_score = None
            for a_white, res in games:
                score_val = 1.0 if res == "win" else (0.5 if res == "draw" else 0.0)
                if a_white:
                    w_score = score_val
                else:
                    b_score = score_val
            if w_score is not None and b_score is not None:
                pair_scores.append((w_score, b_score))
    paired_analysis = paired_score_and_ci(pair_scores)

    summary = {
        "label": args.label,
        "date": stamp,
        "engine_a": {"path": args.engine_a, "sha256": sha_a, "fpu": args.fpu_a},
        "engine_b": {"path": args.engine_b, "sha256": sha_b, "fpu": args.fpu_b},
        "settings": {
            "sims": args.sims,
            "c_puct": args.c_puct,
            "openings_file": args.openings,
            "num_openings": args.num_openings,
            "seed": args.seed,
            "device": device,
            "noise": False,
            "temperature": "greedy (argmax visits)",
        },
        "completed_games": n,
        "effective_games": n_eff,
        "planned_games": 2 * len(openings),
        "a_wins": tally["a_wins"],
        "draws": tally["draws"],
        "b_wins": tally["b_wins"],
        "capped_games": tally["capped"],
        "engine_failures": tally["failures"],
        "a_score": round(score, 4),
        "elo_diff": round(elo_diff, 1) if not math.isinf(elo_diff) else None,
        "elo_diff_a_minus_b_95ci": [
            round(lo, 1) if not math.isinf(lo) else None,
            round(hi, 1) if not math.isinf(hi) else None,
        ],
        "pairing_analysis": paired_analysis,
        "games_csv": str(games_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
