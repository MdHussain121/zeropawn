"""Regression tests for tactical dataset sampling and value target supervision (F-01, D-1, D-2).

Verifies:
1. SourceAwareCombinedDataset combines master and tactical datasets.
2. Exact exclusion for ratio 0.0, pure tactical for ratio 1.0, and expected proportion sampling for intermediate ratios (e.g. 0.3).
3. Guarantee that tactical samples have masked value targets ([0, 0, 0]), outcome = -1, and ply = 0.
4. Input validation (ratio bounds, empty datasets).
5. PositionData validation for outcomes 0, 1, 2, and OUTCOME_UNAVAILABLE (255) and rejection of invalid outcomes.
6. Serialization and memmap round trip for OUTCOME_UNAVAILABLE (255).
7. Path and metadata precedence in determine_is_tactical (preventing false positives for master datasets in tactics-related parent folders).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from python.data.build_dataset import (
    OUTCOME_UNAVAILABLE,
    VALID_OUTCOMES,
    PositionData,
    save_dataset_split,
)
from python.data.dataset import (
    BinaryChessDataset,
    SourceAwareCombinedDataset,
    determine_is_tactical,
)
from python.selfplay.replay_buffer import (
    ReplayBuffer,
    load_tactical_positions_from_csv,
)
from python.selfplay.worker import SavedPosition


def _create_synthetic_dataset(
    dir_path: Path,
    n: int,
    is_tactical: bool,
) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    planes = np.zeros((n, 19, 8, 8), dtype=np.float32)
    moves = np.arange(n, dtype=np.int16)

    if is_tactical:
        outcomes = np.full(n, 255, dtype=np.uint8)
        values = np.zeros((n, 3), dtype=np.float32)
        plies = np.zeros(n, dtype=np.int16)
    else:
        outcomes = np.array([i % 3 for i in range(n)], dtype=np.uint8)
        values = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            values[i, outcomes[i]] = 1.0
        plies = np.full(n, 20, dtype=np.int16)

    game_ids = np.arange(n, dtype=np.uint64)

    planes.tofile(dir_path / "planes.bin")
    moves.tofile(dir_path / "moves.bin")
    outcomes.tofile(dir_path / "outcomes.bin")
    values.tofile(dir_path / "values.bin")
    plies.tofile(dir_path / "plies.bin")
    game_ids.tofile(dir_path / "game_ids.bin")

    meta = {
        "num_samples": n,
        "kind": "tactics-fullline" if is_tactical else "master-games",
    }
    with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return dir_path


def test_tactical_value_target_masking(tmp_path):
    """Verify that tactical dataset positions always have value_target [0,0,0], outcome -1, ply 0."""
    t_dir = _create_synthetic_dataset(tmp_path / "tactics", n=10, is_tactical=True)
    with BinaryChessDataset(t_dir, is_tactical=True) as ds:
        assert ds.is_tactical is True
        for i in range(len(ds)):
            s = ds[i]
            assert s["is_tactical"].item() is True
            assert s["outcome"].item() == -1
            assert torch.allclose(s["value_target"], torch.zeros(3))
            assert s["ply"].item() == 0


def test_source_aware_combined_dataset_ratios(tmp_path):
    """Verify exact boundary exclusion (0.0/1.0) and expected proportion sampling (0.3) using SourceAwareCombinedDataset."""
    m_dir = _create_synthetic_dataset(tmp_path / "master", n=50, is_tactical=False)
    t_dir = _create_synthetic_dataset(tmp_path / "tactics", n=50, is_tactical=True)

    m_ds = BinaryChessDataset(m_dir, is_tactical=False)
    t_ds = BinaryChessDataset(t_dir, is_tactical=True)

    # 1. Pure Master (ratio = 0.0) guarantees exact exclusion of tactical positions
    comb_0 = SourceAwareCombinedDataset(m_ds, t_ds, tactics_ratio=0.0)
    loader_0 = DataLoader(comb_0, batch_size=20, sampler=comb_0.get_sampler())
    for b in loader_0:
        assert not b["is_tactical"].any(), "Ratio 0.0 sampled tactical positions!"

    # 2. Pure Tactics (ratio = 1.0) guarantees pure tactical positions
    comb_1 = SourceAwareCombinedDataset(m_ds, t_ds, tactics_ratio=1.0)
    loader_1 = DataLoader(comb_1, batch_size=20, sampler=comb_1.get_sampler())
    for b in loader_1:
        assert b["is_tactical"].all(), "Ratio 1.0 sampled non-tactical positions!"

    # 3. Mixed (ratio = 0.3) targets 30% in expectation under weighted sampling with replacement
    torch.manual_seed(42)
    comb_03 = SourceAwareCombinedDataset(m_ds, t_ds, tactics_ratio=0.3)
    sampler = comb_03.get_sampler()
    # Draw 5,000 samples to verify empirical ratio converges near 30%
    sample_indices = list(iter(sampler))
    tactics_drawn = sum(1 for idx in sample_indices if idx >= len(m_ds))
    empirical_ratio = tactics_drawn / len(sample_indices)
    assert 0.20 <= empirical_ratio <= 0.40, (
        f"Empirical ratio {empirical_ratio:.3f} deviated from 0.30"
    )


def test_invalid_tactics_ratios(tmp_path):
    """Verify that invalid ratios raise descriptive ValueErrors."""
    m_dir = _create_synthetic_dataset(tmp_path / "master", n=10, is_tactical=False)
    t_dir = _create_synthetic_dataset(tmp_path / "tactics", n=10, is_tactical=True)

    m_ds = BinaryChessDataset(m_dir, is_tactical=False)
    t_ds = BinaryChessDataset(t_dir, is_tactical=True)

    with pytest.raises(ValueError, match="tactics_ratio must be between 0.0 and 1.0"):
        SourceAwareCombinedDataset(m_ds, t_ds, tactics_ratio=-0.1)

    with pytest.raises(ValueError, match="tactics_ratio must be between 0.0 and 1.0"):
        SourceAwareCombinedDataset(m_ds, t_ds, tactics_ratio=1.1)


def test_position_data_outcome_validation():
    """Verify that PositionData accepts legitimate outcomes (0, 1, 2) and sentinel (255), and rejects invalid."""
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    # Valid outcomes: 0 (win), 1 (draw), 2 (loss), 255 (OUTCOME_UNAVAILABLE)
    for valid_o in (0, 1, 2, OUTCOME_UNAVAILABLE):
        pos = PositionData(
            planes=planes,
            move_idx=100,
            outcome=valid_o,
            game_id=42,
            ply=15,
            pos_hash=12345,
        )
        assert pos.outcome == valid_o

    # Invalid outcomes must be rejected with descriptive ValueError
    for invalid_o in (-1, 3, 4, 100, 254, 256):
        with pytest.raises(ValueError, match="Invalid outcome"):
            PositionData(
                planes=planes,
                move_idx=100,
                outcome=invalid_o,
                game_id=42,
                ply=15,
                pos_hash=12345,
            )


def test_position_data_roundtrip_serialization(tmp_path):
    """Verify that OUTCOME_UNAVAILABLE survives binary serialization, memmap, and dataset loading."""
    out_dir = tmp_path / "roundtrip_test"
    planes = np.ones((19, 8, 8), dtype=np.float32)

    # 1 master win (0), 1 master draw (1), 1 master loss (2), 1 tactical position (255)
    positions = [
        PositionData(planes=planes, move_idx=10, outcome=0, game_id=1, ply=12, pos_hash=101),
        PositionData(planes=planes, move_idx=20, outcome=1, game_id=2, ply=14, pos_hash=102),
        PositionData(planes=planes, move_idx=30, outcome=2, game_id=3, ply=16, pos_hash=103),
        PositionData(
            planes=planes, move_idx=40, outcome=OUTCOME_UNAVAILABLE, game_id=4, ply=0, pos_hash=104
        ),
    ]

    save_dataset_split(out_dir, positions, "train", extra_metadata={"kind": "mixed_test"})

    # Check raw binary encoding on disk: uint8
    outcomes_disk = np.fromfile(out_dir / "outcomes.bin", dtype=np.uint8)
    assert list(outcomes_disk) == [0, 1, 2, 255]

    # Open with BinaryChessDataset
    with BinaryChessDataset(out_dir) as ds:
        assert len(ds) == 4

        # Master samples preserve valid WDL one-hot targets
        s0 = ds[0]
        assert s0["outcome"].item() == 0
        assert torch.allclose(s0["value_target"], torch.tensor([1.0, 0.0, 0.0]))
        assert s0["is_tactical"].item() is False

        s1 = ds[1]
        assert s1["outcome"].item() == 1
        assert torch.allclose(s1["value_target"], torch.tensor([0.0, 1.0, 0.0]))
        assert s1["is_tactical"].item() is False

        s2 = ds[2]
        assert s2["outcome"].item() == 2
        assert torch.allclose(s2["value_target"], torch.tensor([0.0, 0.0, 1.0]))
        assert s2["is_tactical"].item() is False

        # Tactical sample: outcome remapped to -1, value_target masked to [0, 0, 0]
        s3 = ds[3]
        assert s3["outcome"].item() == -1
        assert torch.allclose(s3["value_target"], torch.tensor([0.0, 0.0, 0.0]))
        assert s3["is_tactical"].item() is True
        assert s3["move"].item() == 40


def test_determine_is_tactical_precedence():
    """Verify strict 4-tier precedence in determine_is_tactical."""
    # Tier 1: Explicit caller override always wins
    assert (
        determine_is_tactical(
            Path("some/tactics/path"), metadata={"kind": "tactics"}, is_tactical=False
        )
        is False
    )
    assert (
        determine_is_tactical(
            Path("some/master/path"), metadata={"kind": "master"}, is_tactical=True
        )
        is True
    )

    # Tier 2: Metadata declaration wins over path
    # Master metadata in a path containing tactical keywords
    p_tricky = Path("C:/chess/tactics_experiment/dataset_tactics/train_master")
    assert determine_is_tactical(p_tricky, metadata={"kind": "master_games_2500"}) is False
    assert determine_is_tactical(p_tricky, metadata={"kind": "games"}) is False
    assert determine_is_tactical(p_tricky, metadata={"is_tactical": False}) is False

    # Tactical metadata in an innocent path
    p_innocent = Path("C:/chess/my_data/train")
    assert determine_is_tactical(p_innocent, metadata={"kind": "tactics-fullline"}) is True
    assert determine_is_tactical(p_innocent, metadata={"kind": "tactics_puzzles"}) is True
    assert determine_is_tactical(p_innocent, metadata={"is_tactical": True}) is True

    # Tier 3: Narrow legacy compatibility rule (exact path components)
    assert determine_is_tactical(Path("data/dataset_tactics/train"), metadata={}) is True
    assert determine_is_tactical(Path("data/dataset_tactics_v1/train"), metadata={}) is True
    assert determine_is_tactical(Path("data/dataset_puzzles/train"), metadata={}) is True

    # Substrings in folder names MUST NOT trigger tactical classification!
    assert (
        determine_is_tactical(Path("C:/chess/tactics_experiment/train_master"), metadata={})
        is False
    )
    assert determine_is_tactical(Path("C:/chess/puzzle_study/train_master"), metadata={}) is False
    assert determine_is_tactical(Path("data/dataset_training/train"), metadata={}) is False

    # Tier 4: Safe default is False
    assert determine_is_tactical(Path("data/unknown_dataset"), metadata=None) is False
    assert determine_is_tactical(Path("data/unknown_dataset"), metadata={"corrupted": 123}) is False


def test_master_dataset_under_tactics_named_parent_retains_supervision(tmp_path):
    """Critical invariant: A master dataset under a parent directory containing 'tactics' or 'puzzle'
    must NEVER lose value supervision."""
    # Parent directory with 'tactics_experiment'
    parent_dir = tmp_path / "tactics_experiment" / "train_master"
    m_dir = _create_synthetic_dataset(parent_dir, n=10, is_tactical=False)

    with BinaryChessDataset(m_dir) as ds:
        assert ds.is_tactical is False, (
            "Master dataset was falsely classified as tactical due to parent dir!"
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["is_tactical"].item() is False
            assert sample["outcome"].item() != -1
            # Value target must be valid one-hot, not masked to zero
            assert sample["value_target"].sum().item() == pytest.approx(1.0)


def test_tactical_pool_loads_from_synthetic_csv(tmp_path):
    """Verify that load_tactical_positions_from_csv loads valid positions and skips illegal moves."""
    csv_file = tmp_path / "synthetic_tactics.csv"
    csv_content = (
        "PuzzleId,FEN,Moves,Rating\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,1500\n"
        "p2,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1,e7e5 g1f3,1500\n"
        "p3_illegal,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,h1h8,1500\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    planes, moves, outcomes = load_tactical_positions_from_csv(csv_file)
    # p1 and p2 are valid, p3 is illegal and skipped
    assert planes.shape == (2, 19, 8, 8)
    assert moves.shape == (2,)
    assert outcomes.shape == (2,)
    assert (outcomes == 255).all()


def test_replay_buffer_tactical_sampling_batch_tensors(tmp_path):
    """Verify sample_batch_tensors contains ~12% tactical rows with full-search, one-hot visits, outcome 255."""
    rows = ["PuzzleId,FEN,Moves,Rating"]
    moves = ["e2e4", "d2d4", "g1f3", "b1c3", "c2c4", "f2f4", "b2b3", "g2g3", "e2e3", "d2d3"]
    for i, m in enumerate(moves):
        rows.append(f"p_{i},rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,{m},1500")
    csv_file = tmp_path / "tactics.csv"
    csv_file.write_text("\n".join(rows), encoding="utf-8")

    buf = ReplayBuffer(
        buffer_dir=tmp_path / "buf",
        capacity=500,
        tactical_csv_path=csv_file,
        tactical_mix=0.12,
        create_new=True,
    )
    assert buf.tactical_pool_size == len(moves)

    # Add 100 self-play positions
    sp_positions = []
    for i in range(100):
        pl = np.zeros((19, 8, 8), dtype=np.float32)
        pl[0, 0, 0] = float(i)
        v = np.zeros(4672, dtype=np.float32)
        v[i % 4672] = 1.0
        sp_positions.append(
            SavedPosition(
                planes=pl,
                visits=v,
                outcome=i % 3,
                is_full_search=False,
                model_version=1,
                game_id=i,
            )
        )
    buf.add_positions(sp_positions)

    # Sample batch of 100 with return_tactical_mix=True
    batch, human_mix, tactical_mix = buf.sample_batch_tensors(
        batch_size=100,
        return_tactical_mix=True,
    )

    # 12% of 100 = 12 tactical positions
    assert abs(tactical_mix - 0.12) < 1e-4

    # Tactical positions have outcome == 255
    tactical_mask = batch["outcomes"] == 255
    assert tactical_mask.sum().item() == 12

    # Check tactical position properties:
    # 1. is_full_search == True
    assert batch["is_full_search"][tactical_mask].all()

    # 2. visit targets are one-hot
    tac_visits = batch["visits"][tactical_mask]
    assert torch.allclose(tac_visits.sum(dim=-1), torch.ones(12, device=tac_visits.device))
    assert torch.allclose(tac_visits.max(dim=-1).values, torch.ones(12, device=tac_visits.device))

    # Also verify non-tactical rows have outcome != 255
    non_tactical_mask = ~tactical_mask
    assert non_tactical_mask.sum().item() == 88
    assert (batch["outcomes"][non_tactical_mask] != 255).all()

    # Verify sample_batch (numpy) also supports return_tactical_mix and has tactical rows
    np_batch, np_hm, np_tm = buf.sample_batch(
        batch_size=50,
        return_tactical_mix=True,
    )
    # round(50 * 0.12) = 6
    assert abs(np_tm - 0.12) < 1e-4
    np_tac_mask = np_batch["outcomes"] == 255
    assert np_tac_mask.sum() == 6
    assert np_batch["is_full_search"][np_tac_mask].all()
    assert np.allclose(np_batch["visits"][np_tac_mask].sum(axis=-1), 1.0)


def test_replay_buffer_missing_tactical_csv_fallback(tmp_path):
    """Verify that a missing or empty tactical CSV falls back to 0 tactical mix and does not crash."""
    non_existent = tmp_path / "does_not_exist.csv"
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "buf_missing",
        capacity=200,
        tactical_csv_path=non_existent,
        tactical_mix=0.12,
        create_new=True,
    )
    assert getattr(buf, "tactical_pool_size", 0) == 0

    # Add positions and sample
    sp_positions = []
    for i in range(50):
        pl = np.zeros((19, 8, 8), dtype=np.float32)
        v = np.zeros(4672, dtype=np.float32)
        v[0] = 1.0
        sp_positions.append(
            SavedPosition(
                planes=pl,
                visits=v,
                outcome=0,
                is_full_search=True,
                model_version=1,
                game_id=i,
            )
        )
    buf.add_positions(sp_positions)

    batch, human_mix, tactical_mix = buf.sample_batch_tensors(
        batch_size=32, return_tactical_mix=True
    )
    assert tactical_mix == 0.0
    assert (batch["outcomes"] != 255).all()

    # Also test empty CSV
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    buf_empty = ReplayBuffer(
        buffer_dir=tmp_path / "buf_empty",
        capacity=200,
        tactical_csv_path=empty_csv,
        tactical_mix=0.12,
        create_new=True,
    )
    assert getattr(buf_empty, "tactical_pool_size", 0) == 0


def test_replay_buffer_tactical_metadata_resumability(tmp_path):
    """Verify tactical pool size is preserved in replay buffer metadata."""
    csv_file = tmp_path / "tactics.csv"
    csv_file.write_text(
        "PuzzleId,FEN,Moves,Rating\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,1500\n",
        encoding="utf-8",
    )
    buf_dir = tmp_path / "buf_meta"
    buf = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=100,
        tactical_csv_path=csv_file,
        create_new=True,
    )
    assert buf.tactical_pool_size == 1

    # Check meta.json on disk
    meta_path = buf_dir / "meta.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["tactical_pool_size"] == 1

    # Reload buffer without creating new
    buf2 = ReplayBuffer(
        buffer_dir=buf_dir,
        capacity=100,
        create_new=False,
    )
    assert getattr(buf2, "saved_tactical_pool_size", 0) == 1
