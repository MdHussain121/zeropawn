"""Optional diagnostic: value-head sign and perspective across 20 positions.

This is a DIAGNOSTIC (not part of pytest discovery).  The mandatory,
assertion-bearing regression coverage for prediction semantics lives in
``tests/test_model_prediction_semantics.py`` (audit I-1): WDL class ordering,
side-to-move perspective, single-softmax chains, and the canonical scalar
conversion are pinned there with controlled deterministic models and
skip-if-absent real-checkpoint checks.

What this script adds: a broader 20-position material/quality sweep on the
current default checkpoint, exiting non-zero when the sign accuracy is below
the expected bar, so CI/operator runs cannot mistake a mis-signed value head
for success.  (A checkpoint can be semantically consistent and still weakly
calibrated — the designated 2026-10-31 baseline is; see audit I-5.)"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_build_cpp = _repo_root / "build" / "cpp"
if _build_cpp.is_dir() and str(_build_cpp) not in sys.path:
    sys.path.insert(0, str(_build_cpp))

_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    try:
        os.add_dll_directory(_winlibs_bin)
    except (AttributeError, OSError):
        pass

import zeropawn_cpp
from python.model.chess_net import ChessNet


TEST_POSITIONS = [
    # (name, fen, expected_sign: "+", "-", "0")
    (
        "01 Startpos (White to move, equal)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "0",
    ),
    (
        "02 Startpos after 1.e4 (Black to move, equal)",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "0",
    ),
    (
        "03 White up Queen (White to move, winning)",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "+",
    ),
    (
        "04 White up Queen (Black to move, losing)",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "-",
    ),
    (
        "05 Black up Queen (Black to move, winning)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
        "+",
    ),
    (
        "06 Black up Queen (White to move, losing)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "-",
    ),
    ("07 White up Rook (White to move, winning)", "1k6/8/8/8/8/8/4R3/4K3 w - - 0 1", "+"),
    ("08 White up Rook (Black to move, losing)", "1k6/8/8/8/8/8/4R3/4K3 b - - 0 1", "-"),
    # Rook on a7 (not attacking the white king on e1) so the side NOT to
    # move is never in check — the previous e7-rook FENs made 'white in
    # check, black to move', rule-illegal and rejected by the C-1 FEN
    # hardening, which is why this script must construct only legal
    # positions.
    ("09 Black up Rook (Black to move, winning)", "1k6/r7/8/8/8/8/8/4K3 b - - 0 1", "+"),
    ("10 Black up Rook (White to move, losing)", "1k6/r7/8/8/8/8/8/4K3 w - - 0 1", "-"),
    (
        "11 White Queen+King vs King (White to move, winning)",
        "1k6/8/8/8/8/8/4Q3/4K3 w - - 0 1",
        "+",
    ),
    ("12 White Queen+King vs King (Black to move, losing)", "1k6/8/8/8/8/8/4Q3/4K3 b - - 0 1", "-"),
    # Queen on a7 (not attacking e1) so the side NOT to move is not in check
    # (the e7-queen '...b' FEN was rule-illegal: white in check, black to
    # move — rejected by the C-1 FEN hardening).
    (
        "13 Black Queen+King vs King (Black to move, winning)",
        "1k6/q7/8/8/8/8/8/4K3 b - - 0 1",
        "+",
    ),
    ("14 Black Queen+King vs King (White to move, losing)", "1k6/4q3/8/8/8/8/8/4K3 w - - 0 1", "-"),
    (
        "15 Scholar's Mate (Black is checkmated, terminal)",
        "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        "-",
    ),
    (
        "16 Fool's Mate (White is checkmated, terminal)",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "-",
    ),
    ("17 White pawn on 7th rank (White to move, winning)", "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "+"),
    ("18 Black pawn on 2nd rank (Black to move, winning)", "4k3/8/8/8/8/8/1p6/4K3 b - - 0 1", "+"),
    ("19 Opposite bishops draw (White to move, equal)", "4k3/8/2b5/8/8/2B5/8/4K3 w - - 0 1", "0"),
    ("20 Opposite bishops draw (Black to move, equal)", "4k3/8/2b5/8/8/2B5/8/4K3 b - - 0 1", "0"),
]


def evaluate_checkpoint(ckpt_path: str, device: str = "cuda"):
    print(f"\n{'=' * 75}")
    print(f"EVALUATING VALUE SIGN TEST ON: {ckpt_path}")
    print(f"{'=' * 75}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    num_blocks = ckpt.get("num_blocks", 12)
    num_filters = ckpt.get("num_filters", 128)

    model = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    correct_sign = 0
    total = len(TEST_POSITIONS)

    header = f"{'Idx':<4} {'Position':<42} {'STM':<5} {'Exp':<5} {'P(Win)':<8} {'P(Draw)':<8} {'P(Loss)':<8} {'Value':<8} {'Match':<6}"
    print(header)
    print("-" * len(header))

    results = []

    for idx, (name, fen, exp_sign) in enumerate(TEST_POSITIONS, 1):
        game = zeropawn_cpp.ChessGame(fen)
        planes_np = zeropawn_cpp.encode_state(game)
        planes_t = torch.from_numpy(planes_np).unsqueeze(0).to(device)

        with torch.inference_mode():
            p_logits, w_logits = model(planes_t)
            w_probs = torch.softmax(w_logits, dim=-1).squeeze(0).cpu().numpy()

        p_win, p_draw, p_loss = float(w_probs[0]), float(w_probs[1]), float(w_probs[2])
        scalar_val = p_win - p_loss  # side to move perspective

        stm = "White" if game.side_to_move() else "Black"

        if exp_sign == "+":
            ok = scalar_val > 0.05
        elif exp_sign == "-":
            ok = scalar_val < -0.05
        else:
            ok = abs(scalar_val) < 0.40

        if ok:
            correct_sign += 1

        match_str = "PASS" if ok else "FAIL"
        print(
            f"{idx:<4} {name[:40]:<42} {stm:<5} {exp_sign:<5} {p_win:<8.4f} {p_draw:<8.4f} {p_loss:<8.4f} {scalar_val:<+8.4f} {match_str:<6}"
        )

        results.append(
            {
                "idx": idx,
                "name": name,
                "fen": fen,
                "stm": stm,
                "exp_sign": exp_sign,
                "p_win": p_win,
                "p_draw": p_draw,
                "p_loss": p_loss,
                "value": scalar_val,
                "pass": ok,
            }
        )

    print("-" * len(header))
    print(f"SIGN TEST ACCURACY: {correct_sign}/{total} ({correct_sign / total * 100:.1f}%)")
    print(f"{'=' * 75}\n")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Value-head sign diagnostic (optional; pytest coverage is in "
        "tests/test_model_prediction_semantics.py)"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Checkpoint to probe (default: the registry-designated default).",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.75,
        help="Minimum sign accuracy for exit code 0 (default: 0.75). A weakly "
        "calibrated but semantically correct checkpoint may legitimately fail "
        "this quality bar (audit I-5) — lower it consciously, do not ignore it.",
    )
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        from python.model.checkpoint_registry import (
            DefaultCheckpointError,
            resolve_default_checkpoint,
        )

        try:
            ckpt_path = str(resolve_default_checkpoint(repo_root=_repo_root).path)
        except DefaultCheckpointError as e:
            print("No designated default checkpoint is available.\n")
            print(e)
            raise SystemExit(2)

    results = evaluate_checkpoint(ckpt_path, dev)
    total = len(results)
    correct = sum(1 for r in results if r["pass"])
    accuracy = (correct / total) if total else 0.0
    ok = accuracy >= args.min_accuracy
    print(
        f"RESULT: {correct}/{total} ({accuracy:.1%}) sign accuracy; "
        f"threshold {args.min_accuracy:.0%} -> {'PASS' if ok else 'FAIL'}"
    )
    raise SystemExit(0 if ok else 1)
