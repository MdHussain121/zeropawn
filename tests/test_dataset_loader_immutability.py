"""Regression tests: dataset loading must never mutate source dataset files.

Guards audit finding V-1: GpuResidentDataLoader used to reopen the master
``outcomes.bin`` in ``r+`` mode and overwrite every stored label with the 255
sentinel whenever ``tactics_dir`` was set and ``value_targets.bin`` was absent,
destroying the canonical master-game training shard on disk (2,490,528 labels
lost). These tests snapshot every file of a synthetic dataset tree and require
byte-identical content after loader passes with tactics enabled and disabled,
while asserting the in-memory label semantics (valid W/D/L decode, sentinel
masked) survive the fix.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from python.data.build_dataset import OUTCOME_UNAVAILABLE
from python.data.dataset import BinaryChessDataset, GpuResidentDataLoader

PLANES_SHAPE = (19, 8, 8)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_tree(directory: Path) -> dict[str, str]:
    """Map of relative file name -> SHA-256 for every file under directory."""
    return {
        str(p.relative_to(directory)): _sha256_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _write_dataset(
    directory: Path,
    outcomes: list[int],
    kind: str,
    with_outcomes: bool = True,
) -> Path:
    """Create a minimal binary dataset shard; ``directory`` must not exist yet."""
    directory.mkdir(parents=True)
    n = len(outcomes)
    metadata = {"num_samples": n, "planes_shape": list(PLANES_SHAPE), "kind": kind}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rng = np.random.default_rng(1234)
    planes = rng.standard_normal((n, *PLANES_SHAPE)).astype(np.float32)
    planes.tofile(directory / "planes.bin")
    moves = np.arange(n, dtype=np.int16) % 100
    moves.tofile(directory / "moves.bin")
    plies = np.arange(4, 4 + n, dtype=np.int16)
    plies.tofile(directory / "plies.bin")
    game_ids = np.arange(n, dtype=np.uint64) + 1
    game_ids.tofile(directory / "game_ids.bin")
    if with_outcomes:
        np.array(outcomes, dtype=np.uint8).tofile(directory / "outcomes.bin")
    return directory


def _cpu_loader(dataset_dir: Path, tactics_dir: Path | None) -> GpuResidentDataLoader:
    loader = GpuResidentDataLoader(
        dataset_dir,
        batch_size=3,
        shuffle=False,
        device="cpu",
        storage_dtype=torch.float32,
        tactics_dir=tactics_dir,
    )
    # Consume one full pass through the real sampling path (RAM-resident on CPU).
    batches = list(loader)
    assert len(batches) >= 1
    first = batches[0]
    assert set(("planes", "move", "value_target", "ply")) <= set(first.keys())
    return loader


def _targets_from(loader: GpuResidentDataLoader) -> np.ndarray:
    assert loader.ram_targets is not None
    return loader.ram_targets.float().numpy()


MASTER_OUTCOMES = [0, 1, 2, OUTCOME_UNAVAILABLE, 1, 0, 2, OUTCOME_UNAVAILABLE]


def _expected_onehot(outcome: int) -> list[float]:
    if outcome == OUTCOME_UNAVAILABLE:
        return [0.0, 0.0, 0.0]
    one_hot = [0.0, 0.0, 0.0]
    one_hot[int(outcome)] = 1.0
    return one_hot


@pytest.fixture()
def dataset_pair(tmp_path: Path) -> tuple[Path, Path]:
    master = _write_dataset(tmp_path / "master", MASTER_OUTCOMES, kind="master-games")
    tactics = _write_dataset(
        tmp_path / "tactics",
        [OUTCOME_UNAVAILABLE] * 4,
        kind="tactics",
    )
    return master, tactics


def test_gpu_loader_with_tactics_leaves_sources_byte_identical(dataset_pair):
    master, tactics = dataset_pair
    before_master = _snapshot_tree(master)
    before_tactics = _snapshot_tree(tactics)
    assert "outcomes.bin" in before_master  # the destroyed file must be present

    loader = _cpu_loader(master, tactics_dir=tactics)
    loader.close()

    assert _snapshot_tree(master) == before_master, (
        "GpuResidentDataLoader modified the master dataset tree "
        "(audit V-1: outcomes.bin used to be overwritten with 255)"
    )
    assert _snapshot_tree(tactics) == before_tactics, (
        "GpuResidentDataLoader modified the tactics dataset tree"
    )


def test_gpu_loader_without_tactics_leaves_sources_byte_identical(dataset_pair):
    master, _ = dataset_pair
    before = _snapshot_tree(master)

    loader = _cpu_loader(master, tactics_dir=None)
    loader.close()

    assert _snapshot_tree(master) == before


def test_valid_outcomes_decode_and_sentinel_rows_masked(dataset_pair):
    master, _ = dataset_pair
    loader = _cpu_loader(master, tactics_dir=None)
    try:
        targets = _targets_from(loader)
        assert targets.shape == (len(MASTER_OUTCOMES), 3)
        for idx, outcome in enumerate(MASTER_OUTCOMES):
            np.testing.assert_allclose(
                targets[idx], _expected_onehot(outcome), atol=1e-6,
                err_msg=f"row {idx} (stored outcome {outcome}) decoded wrongly",
            )
    finally:
        loader.close()


def test_tactics_concat_preserves_master_labels_and_masks_tactics(dataset_pair):
    master, tactics = dataset_pair
    loader = _cpu_loader(master, tactics_dir=tactics)
    try:
        targets = _targets_from(loader)
        assert loader.num_samples == len(MASTER_OUTCOMES) + 4
        for idx, outcome in enumerate(MASTER_OUTCOMES):
            np.testing.assert_allclose(
                targets[idx], _expected_onehot(outcome), atol=1e-6,
                err_msg=f"master row {idx} lost its label after tactics concat",
            )
        for idx in range(len(MASTER_OUTCOMES), loader.num_samples):
            np.testing.assert_allclose(
                targets[idx], [0.0, 0.0, 0.0], atol=1e-6,
                err_msg=f"tactics row {idx} must carry a masked (zero) value target",
            )
    finally:
        loader.close()


def test_syzygy_five_class_labels_map_to_three_class_targets(tmp_path):
    """5-class Syzygy outcomes map 0/1->W, 2->D, 3/4->L; 255 stays masked."""
    outcomes = [0, 1, 2, 3, 4, OUTCOME_UNAVAILABLE]
    shard = _write_dataset(tmp_path / "tb", outcomes, kind="endgame_syzygy")
    expected = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    loader = GpuResidentDataLoader(
        shard, batch_size=3, shuffle=False, device="cpu", storage_dtype=torch.float32
    )
    try:
        np.testing.assert_allclose(_targets_from(loader), expected, atol=1e-6)
    finally:
        loader.close()
    # The non-GPU loader must agree on the same mapping.
    ds = BinaryChessDataset(shard)
    for idx, exp in enumerate(expected):
        torch.testing.assert_close(ds[idx]["value_target"], torch.tensor(exp))


def test_binary_dataset_source_immutability_and_label_semantics(dataset_pair):
    master, _ = dataset_pair
    before = _snapshot_tree(master)

    ds = BinaryChessDataset(master)
    assert len(ds) == len(MASTER_OUTCOMES)
    for idx, outcome in enumerate(MASTER_OUTCOMES):
        sample = ds[idx]
        if outcome == OUTCOME_UNAVAILABLE:
            assert int(sample["outcome"]) == -1
            torch.testing.assert_close(
                sample["value_target"], torch.zeros(3, dtype=torch.float32)
            )
            assert bool(sample["is_tactical"]) is True
        else:
            assert int(sample["outcome"]) == int(outcome)
            expected = torch.zeros(3, dtype=torch.float32)
            expected[int(outcome)] = 1.0
            torch.testing.assert_close(sample["value_target"], expected)
            assert bool(sample["is_tactical"]) is False
        assert int(sample["ply"]) > 0 or outcome == OUTCOME_UNAVAILABLE

    assert _snapshot_tree(master) == before


def test_non_tactical_dataset_without_outcomes_fails_closed(tmp_path):
    shard = _write_dataset(tmp_path / "master", MASTER_OUTCOMES, kind="master-games",
                           with_outcomes=False)
    before = _snapshot_tree(shard)

    with pytest.raises(FileNotFoundError):
        GpuResidentDataLoader(
            shard, batch_size=2, device="cpu", storage_dtype=torch.float32
        )
    with pytest.raises(ValueError):
        BinaryChessDataset(shard)

    assert _snapshot_tree(shard) == before, "failed load must not write anything"


def test_tactical_dataset_without_outcomes_loads_masked_in_memory(tmp_path):
    """Tactical shards may lack outcome labels entirely: mask in memory, never write."""
    shard = _write_dataset(tmp_path / "tactics", [0] * 4, kind="tactics",
                           with_outcomes=False)
    before = _snapshot_tree(shard)

    loader = GpuResidentDataLoader(
        shard, batch_size=2, shuffle=False, device="cpu", storage_dtype=torch.float32
    )
    try:
        targets = _targets_from(loader)
        np.testing.assert_allclose(targets, np.zeros((4, 3)), atol=1e-6)
    finally:
        loader.close()

    after = _snapshot_tree(shard)
    assert after == before, "outcomes.bin must not be synthesized on disk"
    assert "outcomes.bin" not in after
