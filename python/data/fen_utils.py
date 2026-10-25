"""FEN decoding utilities for converting 19-plane tensors back to FEN strings.

Follows the exact orientation specification in docs/ENCODING.md:
- White to move: Row 0 is Rank 1, Row 7 is Rank 8.
- Black to move: Board vertically flipped; Row 0 is Rank 8, Row 7 is Rank 1.
"""

from __future__ import annotations

import numpy as np


def decode_planes_to_fen(
    planes: np.ndarray,
    is_white_to_move: bool,
    ply: int = 0,
) -> str:
    """Decode an (8x8x19) or (19x8x8) state array into a standard FEN string."""
    if planes.ndim != 3:
        raise ValueError(f"Expected 3D planes array, got shape {planes.shape}")

    # Ensure channels-first (19, 8, 8)
    if planes.shape[-1] == 19 and planes.shape[0] == 8:
        planes = np.transpose(planes, (2, 0, 1))

    piece_names = ["P", "N", "B", "R", "Q", "K"]
    board = [["." for _ in range(8)] for _ in range(8)]

    if is_white_to_move:
        # Own pieces are White (uppercase), Opponent pieces are Black (lowercase)
        for p_idx in range(6):
            r_idx, c_idx = np.where(planes[p_idx] > 0.5)
            for r, c in zip(r_idx, c_idx):
                board[r][c] = piece_names[p_idx]
        for p_idx in range(6):
            r_idx, c_idx = np.where(planes[6 + p_idx] > 0.5)
            for r, c in zip(r_idx, c_idx):
                board[r][c] = piece_names[p_idx].lower()
    else:
        # Own pieces are Black (lowercase), Opponent pieces are White (uppercase)
        # Black board is vertically flipped: row 0 in planes is rank 7 (rank 8 in chess)
        for p_idx in range(6):
            r_idx, c_idx = np.where(planes[p_idx] > 0.5)
            for r, c in zip(r_idx, c_idx):
                actual_r = 7 - r
                board[actual_r][c] = piece_names[p_idx].lower()
        for p_idx in range(6):
            r_idx, c_idx = np.where(planes[6 + p_idx] > 0.5)
            for r, c in zip(r_idx, c_idx):
                actual_r = 7 - r
                board[actual_r][c] = piece_names[p_idx]

    # Build piece placement string (Rank 8 down to Rank 1)
    fen_ranks = []
    for r in range(7, -1, -1):
        rank_str = ""
        empty_count = 0
        for c in range(8):
            sq = board[r][c]
            if sq == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += sq
        if empty_count > 0:
            rank_str += str(empty_count)
        fen_ranks.append(rank_str)
    placement = "/".join(fen_ranks)

    # Active color
    color_str = "w" if is_white_to_move else "b"

    # Castling rights
    castling = ""
    if is_white_to_move:
        if planes[12, 0, 0] > 0.5:
            castling += "K"
        if planes[13, 0, 0] > 0.5:
            castling += "Q"
        if planes[14, 0, 0] > 0.5:
            castling += "k"
        if planes[15, 0, 0] > 0.5:
            castling += "q"
    else:
        if planes[14, 0, 0] > 0.5:
            castling += "K"
        if planes[15, 0, 0] > 0.5:
            castling += "Q"
        if planes[12, 0, 0] > 0.5:
            castling += "k"
        if planes[13, 0, 0] > 0.5:
            castling += "q"
    if not castling:
        castling = "-"

    # En passant square
    ep_r, ep_c = np.where(planes[16] > 0.5)
    if len(ep_r) > 0:
        r, c = int(ep_r[0]), int(ep_c[0])
        file_char = chr(ord("a") + c)
        actual_rank = (r + 1) if is_white_to_move else (8 - r)
        ep_str = f"{file_char}{actual_rank}"
    else:
        ep_str = "-"

    # Halfmove clock
    halfmove = int(round(float(planes[17, 0, 0]) * 100.0))
    fullmove = (ply // 2) + 1

    return f"{placement} {color_str} {castling} {ep_str} {halfmove} {fullmove}"
