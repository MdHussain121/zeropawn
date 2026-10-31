"""Independent sample-decode verification for a rebuilt dataset split.

For N random rows: decode planes -> FEN using the stored side to move,
rebuild a C++ ChessGame, and verify that
  (a) the packed legal mask exactly matches the position's legal actions,
  (b) the stored move label is legal in that position,
  (c) the stored outcome is a valid WDL class (or the masked sentinel),
  (d) ply and stm are consistent for PGN-rooted rows (stm flips each ply).

Usage:
    python scripts/remediation_sample_decode_check.py --split data/.../train --rows 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
import zeropawn_cpp
from python.data.dataset import MASK_PACKED_BYTES
from python.data.fen_utils import decode_planes_to_fen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    split = Path(args.split)
    meta = (split / "metadata.json")
    import json

    m = json.loads(meta.read_text())
    n = int(m["num_samples"])
    planes = np.memmap(split / "planes.bin", dtype=np.float32, mode="r", shape=(n, 19, 8, 8))
    moves = np.fromfile(split / "moves.bin", dtype=np.int16)
    outcomes = np.fromfile(split / "outcomes.bin", dtype=np.uint8)
    plies = np.fromfile(split / "plies.bin", dtype=np.int16)
    masks = np.memmap(split / "masks.bin", dtype=np.uint8, mode="r", shape=(n, MASK_PACKED_BYTES))
    stm_path = split / "stm.bin"
    stms = np.fromfile(stm_path, dtype=np.uint8) if stm_path.is_file() else None

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n, size=min(args.rows, n), replace=False)

    mask_mismatch = 0
    illegal_label = 0
    bad_outcome = 0
    ply_stm_mismatch = 0
    checked = 0
    for i in idxs:
        i = int(i)
        is_white = bool(stms[i]) if stms is not None else (int(plies[i]) % 2 == 0)
        fen = decode_planes_to_fen(
            np.asarray(planes[i], dtype=np.float32), is_white_to_move=is_white, ply=int(plies[i])
        )
        try:
            g = zeropawn_cpp.ChessGame(fen)
        except Exception as e:
            print(f"row {i}: FEN decode failed: {e}")
            return 1
        legal = set(int(a) for a in g.legal_actions())
        stored_mask = set(
            j for j, bit in enumerate(np.unpackbits(np.asarray(masks[i]), count=4672)) if bit
        )
        if legal != stored_mask:
            mask_mismatch += 1
            if mask_mismatch <= 3:
                only_legal = legal - stored_mask
                only_mask = stored_mask - legal
                print(
                    f"row {i}: mask mismatch (legal-only {list(only_legal)[:5]}, "
                    f"mask-only {list(only_mask)[:5]})"
                )
        if int(moves[i]) not in legal:
            illegal_label += 1
            if illegal_label <= 3:
                print(f"row {i}: stored move {int(moves[i])} not legal")
        if outcomes[i] not in (0, 1, 2, 255):
            bad_outcome += 1
        if stms is not None and int(plies[i]) > 0:
            expected_stm = 1 if (int(plies[i]) % 2 == 0) else 0
            if int(stms[i]) != expected_stm:
                ply_stm_mismatch += 1
        checked += 1

    print(
        f"[{split}] checked {checked}/{n} rows: mask_mismatch={mask_mismatch} "
        f"illegal_label={illegal_label} bad_outcome={bad_outcome} "
        f"ply_stm_mismatch={ply_stm_mismatch}"
    )
    ok = mask_mismatch == 0 and illegal_label == 0 and bad_outcome == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
