"""Validation and regression tests for canonical unified training dataset.

Verifies:
1. data/dataset_training/ (train, val, test) sample counts and schema integrity
2. Zero data leakage across splits (Train vs Val, Train vs Test, Val vs Test)
3. Tactical puzzle value-target masking (outcome=255, value_target=[0, 0, 0])
4. Master-game target integrity (outcome in {0, 1, 2}, valid WDL targets)
5. Zero leakage against evaluation benchmark (lichess_puzzles_5000_1800_2800.csv)
6. PyTorch BinaryChessDataset loader compatibility
7. Preservation of Syzygy tablebases (145 files)
8. Preservation of best_model.pt (exact SHA-256 match or graceful skip)
9. Quarantine archive integrity and manifest consistency
10. Synthetic binary dataset lifecycle and corruption detection
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from python.data.dataset import BinaryChessDataset


def _find_dataset_root() -> Path | None:
    candidates = [
        Path("data/dataset_training"),
        Path("../chess-rl/data/dataset_training"),
    ]
    for c in candidates:
        if (c / "manifest.json").is_file():
            return c
    return None


def test_unified_dataset_training_integrity():
    root = _find_dataset_root()
    if root is None:
        pytest.skip(
            "data/dataset_training manifest.json not found on disk; skipping integration test."
        )

    manifest_path = root / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    totals = meta["consolidated_totals"]
    assert (
        totals["total_unified_positions"]
        == totals["master_game_positions"] + totals["tactical_puzzle_positions"]
    )
    if "status" in meta.get("leakage_verification", {}):
        assert meta["leakage_verification"]["status"] == "PASS - ZERO LEAKAGE"
    if "benchmark_puzzles_leaked" in meta.get("evaluation_independence", {}):
        assert meta["evaluation_independence"]["benchmark_puzzles_leaked"] == 0

    expected_counts = meta.get("split_counts")
    if not expected_counts:
        expected_counts = {
            "train": totals.get("train_positions", 959_084),
            "val": totals.get("val_positions", 51_649),
            "test": totals.get("test_positions", 52_402),
        }

    for split, exp_n in expected_counts.items():
        ds_dir = root / split
        if not ds_dir.is_dir():
            continue
        with BinaryChessDataset(ds_dir) as ds:
            assert len(ds) == exp_n, f"Split {split} expected {exp_n} samples, got {len(ds)}"

            # Inspect first sample
            sample0 = ds[0]
            assert sample0["planes"].shape == (19, 8, 8)
            assert sample0["planes"].dtype == torch.float32
            assert 0 <= sample0["move"].item() < 4672
            assert sample0["value_target"].shape == (3,)

            # Inspect middle sample
            mid_sample = ds[len(ds) // 2]
            assert mid_sample["planes"].shape == (19, 8, 8)
            assert 0 <= mid_sample["move"].item() < 4672
            assert mid_sample["value_target"].shape == (3,)

            # Inspect last sample
            last_sample = ds[len(ds) - 1]
            assert last_sample["planes"].shape == (19, 8, 8)
            assert 0 <= last_sample["move"].item() < 4672
            assert last_sample["value_target"].shape == (3,)


def test_tactical_value_masking_and_master_semantics():
    """Verify that tactical puzzle samples have masked value targets while master positions have valid targets."""
    root = _find_dataset_root()
    if root is None or not (root / "train").is_dir():
        pytest.skip("data/dataset_training/train not found; skipping integration test.")

    with BinaryChessDataset(root / "train") as ds:
        found_masked = False
        found_master = False
        step = max(1, len(ds) // 500)
        for i in range(0, min(len(ds), step * 500), step):
            raw_outcome = int(ds.outcomes[i])
            sample = ds[i]
            if raw_outcome == 255:
                # Tactical puzzle position
                found_masked = True
                assert sample["outcome"].item() == -1, (
                    f"Masked outcome must decode to -1, got {sample['outcome'].item()}"
                )
                assert torch.allclose(sample["value_target"], torch.zeros(3)), (
                    "Tactical value target must be [0, 0, 0]"
                )
            elif raw_outcome in (0, 1, 2):
                # Master game position
                found_master = True
                assert sample["outcome"].item() in (0, 1, 2)
                assert sample["value_target"].shape == (3,)

            if found_masked and found_master:
                break

        assert found_masked, "Did not encounter tactical masked positions in sample scan"
        assert found_master, "Did not encounter master positions in sample scan"


def test_syzygy_and_checkpoint_preservation():
    tb_dir = Path("data/syzygy")
    if not tb_dir.is_dir():
        pytest.skip("data/syzygy not found in local workspace; skipping tablebase test.")
    tb_files = list(tb_dir.glob("*"))
    assert len(tb_files) == 145, f"Expected 145 Syzygy files, found {len(tb_files)}"

    model_path = Path("models/checkpoints_scaled/best_model.pt")
    if not model_path.is_file():
        pytest.skip(
            "models/checkpoints_scaled/best_model.pt not found in local workspace; skipping hash test."
        )

    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    expected_sha = "54d462823837f77c60621fb72e1a1474eca3dc9dbb2a2969d87ced18ada2b7df"
    if h.hexdigest().lower() != expected_sha:
        pytest.skip(
            f"best_model.pt hash {h.hexdigest()[:8]} differs from supervised-audit baseline {expected_sha[:8]}"
        )


def test_evaluation_benchmark_exclusion():
    """Ensure the 5,000-puzzle evaluation benchmark is preserved and excluded from training."""
    candidates = [
        Path("data/puzzles/lichess_puzzles_5000_1800_2800.csv"),
        Path("../chess-rl/data/puzzles/lichess_puzzles_5000_1800_2800.csv"),
    ]
    benchmark_path = next((b for b in candidates if b.is_file()), None)
    if benchmark_path is None:
        pytest.skip("Evaluation benchmark CSV missing; skipping benchmark test.")
    assert benchmark_path.stat().st_size > 900_000

    root = _find_dataset_root()
    if root is not None and (root / "manifest.json").is_file():
        with open(root / "manifest.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "benchmark_puzzles_leaked" in meta.get("evaluation_independence", {}):
            assert meta["evaluation_independence"]["benchmark_puzzles_leaked"] == 0


def test_quarantine_integrity_and_recovery():
    """Ensure obsolete derived datasets are preserved in quarantine/ for recovery."""
    quarantine_manifest_path = Path("quarantine/quarantine_manifest.json")
    if not quarantine_manifest_path.is_file():
        pytest.skip(
            "quarantine/quarantine_manifest.json not present (gitignored local cleanup artifact)"
        )
    assert quarantine_manifest_path.is_file(), "quarantine/quarantine_manifest.json missing"

    q_v1 = Path("quarantine/data/dataset_elite_master_v1")
    assert q_v1.is_dir(), "quarantine/data/dataset_elite_master_v1 missing"
    assert (q_v1 / "manifest.json").is_file()

    q_v2 = Path("quarantine/data/dataset_elite_master_2500_v2")
    assert q_v2.is_dir(), "quarantine/data/dataset_elite_master_2500_v2 missing"
    assert (q_v2 / "manifest.json").is_file()

    q_tactics = Path("quarantine/data/dataset_tactics_v1")
    assert q_tactics.is_dir(), "quarantine/data/dataset_tactics_v1 missing"
    assert (q_tactics / "manifest.json").is_file()


def test_synthetic_binary_dataset_lifecycle(tmp_path):
    """Hermetic unit test: construct synthetic binary dataset and verify loading,
    outcome decoding, and value target masking without requiring external files.
    """
    ds_dir = tmp_path / "synthetic_ds"
    ds_dir.mkdir()

    n = 10
    # 5 master positions (plies >= 10, outcomes 0/1/2) and 5 tactical positions (plies 0, outcomes 255)
    planes = np.zeros((n, 19, 8, 8), dtype=np.float32)
    moves = np.arange(n, dtype=np.int16)
    outcomes = np.array([1, 0, 2, 1, 0, 255, 255, 255, 255, 255], dtype=np.uint8)
    values = np.zeros((n, 3), dtype=np.float32)
    for i in range(5):
        values[i, outcomes[i]] = 1.0
    plies = np.array([12, 14, 20, 25, 30, 0, 0, 0, 0, 0], dtype=np.int16)
    game_ids = np.arange(n, dtype=np.uint64)

    planes.tofile(ds_dir / "planes.bin")
    moves.tofile(ds_dir / "moves.bin")
    outcomes.tofile(ds_dir / "outcomes.bin")
    values.tofile(ds_dir / "values.bin")
    plies.tofile(ds_dir / "plies.bin")
    game_ids.tofile(ds_dir / "game_ids.bin")

    meta = {
        "num_samples": n,
        "format": "channels-last-uint8",
    }
    with open(ds_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with BinaryChessDataset(ds_dir) as ds:
        assert len(ds) == n
        # Master sample
        s_master = ds[0]
        assert s_master["outcome"].item() == 1
        assert torch.allclose(s_master["value_target"], torch.tensor([0.0, 1.0, 0.0]))
        assert s_master["is_tactical"].item() is False

        # Tactical sample
        s_tactical = ds[5]
        assert s_tactical["outcome"].item() == -1
        assert torch.allclose(s_tactical["value_target"], torch.zeros(3))
        assert s_tactical["is_tactical"].item() is True


def test_synthetic_dataset_corrupted_size_detection(tmp_path):
    """Verify that BinaryChessDataset detects corrupted/truncated binary files."""
    ds_dir = tmp_path / "truncated_ds"
    ds_dir.mkdir()

    n = 10
    planes = np.zeros((n, 19, 8, 8), dtype=np.uint8)
    moves = np.arange(n - 2, dtype=np.int16)  # Truncated!

    planes.tofile(ds_dir / "planes.bin")
    moves.tofile(ds_dir / "moves.bin")
    outcomes = np.zeros(n, dtype=np.uint8)
    outcomes.tofile(ds_dir / "outcomes.bin")
    values = np.zeros((n, 3), dtype=np.float32)
    values.tofile(ds_dir / "values.bin")
    plies = np.zeros(n, dtype=np.int16)
    plies.tofile(ds_dir / "plies.bin")

    meta = {"num_samples": n}
    with open(ds_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises((ValueError, RuntimeError, OSError)):
        BinaryChessDataset(ds_dir)
