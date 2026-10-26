"""Tests for the binary-shard contamination check (UQ-3 / C2).

The certified eval set must be checked against the FULL training corpus,
including binary shards that carry no FENs (dataset_tactics). Positions are
matched by a digest of the color-invariant planes (0-16).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
import zeropawn_cpp
from scripts.check_puzzle_overlap import _plane_key, eval_plane_keys, load_shard_keys


def _write_shard(tmp_path: Path, planes: np.ndarray) -> Path:
    d = tmp_path / "shard_train"
    d.mkdir()
    (d / "planes.bin").write_bytes(planes.astype(np.float32).tobytes())
    (d / "metadata.json").write_text(json.dumps({"num_samples": int(planes.shape[0])}))
    return d


def _encode_fen(fen: str) -> np.ndarray:
    return zeropawn_cpp.encode_state(zeropawn_cpp.ChessGame(fen))


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_shard_mode_detects_and_clears_overlap(tmp_path):
    # A "training shard" containing the start position and the position after e4.
    shard_planes = np.stack([_encode_fen(STARTPOS), _encode_fen(AFTER_E4)])
    shard_dir = _write_shard(tmp_path, shard_planes)

    eval_rows = [
        {"PuzzleId": "inTrain", "FEN": STARTPOS, "Rating": "1500"},
        {"PuzzleId": "afterE4", "FEN": AFTER_E4, "Rating": "1500"},
        {"PuzzleId": "clean", "FEN": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Rating": "1500"},
    ]
    keys = load_shard_keys([shard_dir])
    assert len(keys) == 2

    ekeys = eval_plane_keys(eval_rows)
    hits = [k in keys for k in ekeys]
    assert hits == [True, True, False]


def test_plane_key_ignores_halfmove_and_repetition_planes(tmp_path):
    """Halfmove-clock and repetition differences must NOT create false
    negatives: a position's identity is its pieces/castling/EP, not its clocks."""
    same_position_different_history = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 17 30"
    planes_a = _encode_fen(STARTPOS)
    planes_b = _encode_fen(same_position_different_history)
    assert not np.array_equal(planes_a, planes_b)  # they differ somewhere...
    assert _plane_key(planes_a) == _plane_key(planes_b)  # ...only on masked planes


def test_shard_mode_handles_missing_metadata(tmp_path):
    planes = np.stack([_encode_fen(STARTPOS)])
    d = tmp_path / "shard_nometa"
    d.mkdir()
    (d / "planes.bin").write_bytes(planes.astype(np.float32).tobytes())
    keys = load_shard_keys([d])
    assert len(keys) == 1


def test_shard_mode_reports_metadata_size_mismatch(tmp_path, capsys):
    planes = np.stack([_encode_fen(STARTPOS), _encode_fen(AFTER_E4)])
    d = _write_shard(tmp_path, planes)
    (d / "metadata.json").write_text(json.dumps({"num_samples": 99}))
    keys = load_shard_keys([d])
    assert len(keys) == 2  # file size wins over the wrong metadata
    assert "file size implies" in capsys.readouterr().err
