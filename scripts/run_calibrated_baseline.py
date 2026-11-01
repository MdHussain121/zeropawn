"""Calibrated strength baseline: engine vs Stockfish UCI_Elo levels.

One setting per invocation (ckpt x elo); schtasks chains them for long runs.
Serial search, fixed sims, no noise/temperature, no tablebase/book for engine.
Saves PGNs + independent python-chess rescore + full provenance row.

Usage:
  python scripts/run_calibrated_baseline.py --ckpt data/models/checkpoints_finetune/best_model.pt
      --elo 1500 --openings 50 --outdir runs/baseline [--smoke] [--workers 8]
Smoke: first 5 openings (10 games).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "build/cpp")
sys.path.insert(0, ".")

from python.model.chess_net import ChessNet
from python.eval.match_runner import MatchRunner, elo_interval, rescore_pgn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def stockfish_version(sf_path: str) -> str:
    p = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate("uci\nquit\n", timeout=30)
    name = next((l for l in out.splitlines() if l.startswith("id name")), "?")
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--elo", type=int, required=True, choices=[1320, 1500, 1700])
    ap.add_argument("--openings", type=int, default=50)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/baseline")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stockfish", default="tools/stockfish.exe")
    ap.add_argument("--book", default=None, help="Path to curated opening book JSON")
    ap.add_argument("--no-book", action="store_true", default=True, help="Disable opening book (default: True)")
    ap.add_argument("--enable-book", dest="no_book", action="store_false", help="Enable opening book")
    ap.add_argument("--cap-policy", default="exclude", choices=["exclude", "draw", "loss", "adjudicate"])
    args = ap.parse_args()

    n_open = 5 if args.smoke else args.openings
    tag = f"{Path(args.ckpt).parent.name}_{Path(args.ckpt).stem}_elo{args.elo}"
    tag += "_smoke" if args.smoke else "_full"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pgn_path = outdir / f"{tag}.pgn"
    if pgn_path.exists():
        pgn_path.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    model.load_state_dict(sd)
    model.eval()

    import zeropawn_cpp

    print(f"pyd: {zeropawn_cpp.__file__}", flush=True)
    use_book = not args.no_book
    runner = MatchRunner(
        model=model,
        device=device,
        stockfish_path=args.stockfish,
        stockfish_nodes=args.nodes,
        mcts_sims=args.sims,
        c_puct=2.0,
        openings_path="data/openings/eval_openings_50.epd",
        num_workers=args.workers,
        search_mode="serial",
        seed=args.seed,
        stockfish_elo=args.elo,
        use_book=use_book,
        book_path=args.book or "data/openings/curated_book.json",
        cap_policy=args.cap_policy,
        enable_value_adjudication=False,
    )
    t0 = time.time()
    try:
        res = runner.run_match(num_openings=n_open, pgn_path=pgn_path)
    finally:
        runner.close()
    wall = time.time() - t0

    w, d, l = res["engine_wins"], res["engine_draws"], res["engine_losses"]
    genuine_d = res.get("genuine_draws", d)
    capped = res.get("capped_games", 0)
    eff_n = res.get("effective_games", w + d + l)

    elo_diff = res.get("elo_diff", 0.0)
    elo_lo, elo_hi = res.get("elo_ci_95", (None, None))
    resc = rescore_pgn(pgn_path)
    row = {
        "tag": tag,
        "checkpoint": args.ckpt,
        "checkpoint_sha256": sha256_file(Path(args.ckpt)),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "stockfish": args.stockfish,
        "stockfish_version": stockfish_version(args.stockfish),
        "stockfish_elo": args.elo,
        "stockfish_nodes": args.nodes,
        "openings_file": "data/openings/eval_openings_50.epd",
        "openings_sha256": sha256_file(Path("data/openings/eval_openings_50.epd")),
        "num_openings": n_open,
        "total_games": res["total_games"],
        "effective_games": eff_n,
        "engine_wins": w,
        "engine_draws": d,
        "genuine_draws": genuine_d,
        "capped_games": capped,
        "engine_losses": l,
        "score": round(res["engine_score"], 4),
        "elo_diff": elo_diff,
        "elo_lo95": elo_lo,
        "elo_hi95": elo_hi,
        "pairing_analysis": res.get("pairing_analysis", {}),
        "sensitivity_analysis": res.get("sensitivity_analysis", {}),
        "elo_formula": "diff=-400*log10(1/s-1); Agresti-Caffo logit 95% CI + cluster-robust paired SE",
        "search_mode": res["search_mode"],
        "mcts_sims": args.sims,
        "seed": args.seed,
        "c_puct": 2.0,
        "num_workers": args.workers,
        "noise": False,
        "temperature": False,
        "tablebase": False,
        "book": use_book,
        "book_sha256": res.get("book_sha256"),
        "book_policy": res.get("book_policy"),
        "book_moves_played": res.get("book_moves_played", 0),
        "cap_policy": args.cap_policy,
        "commit": git_head(),
        "pyd_sha256": sha256_file(Path(zeropawn_cpp.__file__)),
        "pgn": str(pgn_path),
        "rescore": {
            "games": resc["games"],
            "clean_games": resc["clean_games"],
            "w": resc["engine_wins"],
            "d": resc["engine_draws"],
            "l": resc["engine_losses"],
            "capped": resc.get("capped_games", 0),
            "problems": resc["problem_rows"],
        },
        "rescore_agrees": (resc["engine_wins"], resc["engine_draws"], resc["engine_losses"])
        == (w, d, l),
        "wall_sec": round(wall, 1),
        "crashes": 0,
        "illegal_moves": "see pgn/problems (forfeit=loss)",
        "time_forfeits": "n/a (no clocks in runner)",
    }
    (outdir / f"{tag}.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    print(json.dumps(row, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
