import json
from pathlib import Path
import numpy as np
import pytest

from python.selfplay.worker import SavedPosition
from python.selfplay.replay_buffer import ReplayBuffer


@pytest.fixture
def synthetic_human_dir(tmp_path):
    """Create a lightweight synthetic BinaryChessDataset for replay buffer tests."""
    target_dir = tmp_path / "synthetic_human"
    target_dir.mkdir(parents=True, exist_ok=True)
    num_samples = 300
    metadata = {
        "num_samples": num_samples,
        "planes_shape": [19, 8, 8],
        "split": "train",
    }
    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    planes = np.zeros((num_samples, 19, 8, 8), dtype=np.float32)
    moves = np.zeros(num_samples, dtype=np.int16)
    outcomes = np.zeros(num_samples, dtype=np.uint8)
    game_ids = np.zeros(num_samples, dtype=np.uint64)
    plies = np.zeros(num_samples, dtype=np.uint16)

    planes.tofile(target_dir / "planes.bin")
    moves.tofile(target_dir / "moves.bin")
    outcomes.tofile(target_dir / "outcomes.bin")
    game_ids.tofile(target_dir / "game_ids.bin")
    plies.tofile(target_dir / "plies.bin")
    return target_dir


def _create_dummy_pos(pos_id: int, full_search: bool = True) -> SavedPosition:
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    planes[0, 0, 0] = float(pos_id)
    visits = np.zeros(4672, dtype=np.float32)
    visits[0] = 1.0
    return SavedPosition(
        planes=planes,
        visits=visits,
        outcome=pos_id % 3,
        is_full_search=full_search,
        model_version=1,
        game_id=pos_id,
    )


def test_replay_buffer_ring_overwrite(tmp_path):
    """Verify that buffer acts as a fixed-capacity ring buffer and overwrites oldest entries."""
    capacity = 100
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_ring",
        capacity=capacity,
        create_new=True,
    )

    # Add 60 positions
    positions_1 = [_create_dummy_pos(i) for i in range(60)]
    buf.add_positions(positions_1)
    assert len(buf) == 60
    assert buf.write_head == 60

    # Add another 60 positions (total 120, exceeds capacity 100)
    positions_2 = [_create_dummy_pos(i + 60) for i in range(60)]
    buf.add_positions(positions_2)
    assert len(buf) == capacity
    assert buf.write_head == 20  # (60 + 60) % 100

    # Positions 0..19 should now contain IDs 100..119
    assert buf.planes[0, 0, 0, 0] == 100.0
    assert buf.planes[19, 0, 0, 0] == 119.0
    # Position 20 should contain ID 20
    assert buf.planes[20, 0, 0, 0] == 20.0


def test_human_data_taper_schedule(tmp_path, synthetic_human_dir):
    """Verify linear taper schedule from 25% down to 5% over taper_steps."""
    default_p = Path("data/dataset_scaled/train")
    human_dir = default_p if (default_p / "metadata.json").is_file() else synthetic_human_dir
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_taper",
        capacity=1000,
        initial_human_mix=0.25,
        final_human_mix=0.05,
        taper_steps=50_000,
        human_data_dir=human_dir,
        create_new=True,
    )

    assert abs(buf.get_target_human_mix(step=0) - 0.25) < 1e-4
    assert abs(buf.get_target_human_mix(step=25_000) - 0.15) < 1e-4
    assert abs(buf.get_target_human_mix(step=50_000) - 0.05) < 1e-4
    assert abs(buf.get_target_human_mix(step=100_000) - 0.05) < 1e-4


def test_sampling_and_mix_ratio_logging(tmp_path, synthetic_human_dir):
    """Verify uniform batch sampling and actual human mix ratio logging."""
    default_p = Path("data/dataset_scaled/train")
    human_dir = default_p if (default_p / "metadata.json").is_file() else synthetic_human_dir
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_sample",
        capacity=1000,
        initial_human_mix=0.25,
        final_human_mix=0.05,
        taper_steps=10_000,
        human_data_dir=human_dir,
        create_new=True,
    )

    # Insert 200 self-play positions
    positions = [_create_dummy_pos(i) for i in range(200)]
    buf.add_positions(positions)

    batch_size = 64
    batch, mix_ratio = buf.sample_batch(batch_size=batch_size, step=0)

    assert batch["planes"].shape == (batch_size, 19, 8, 8)
    assert batch["visits"].shape == (batch_size, 4672)
    assert batch["outcomes"].shape == (batch_size,)
    assert batch["is_full_search"].shape == (batch_size,)

    # Target mix is 25%, so approximately 16 out of 64 are human samples
    expected_human = int(round(64 * 0.25))
    expected_ratio = expected_human / 64.0
    assert abs(mix_ratio - expected_ratio) < 1e-4


def test_replay_buffer_resumability(tmp_path):
    """Verify that buffer size and pointers persist across restarts."""
    dir_path = tmp_path / "test_resume"
    buf1 = ReplayBuffer(
        buffer_dir=dir_path,
        capacity=500,
        create_new=True,
    )
    positions = [_create_dummy_pos(i) for i in range(75)]
    buf1.add_positions(positions)
    assert len(buf1) == 75

    # Re-open buffer without create_new
    buf2 = ReplayBuffer(
        buffer_dir=dir_path,
        capacity=500,
        create_new=False,
    )
    assert len(buf2) == 75
    assert buf2.write_head == 75
    assert buf2.planes[0, 0, 0, 0] == 0.0
    assert buf2.planes[74, 0, 0, 0] == 74.0


def test_replay_buffer_in_memory_mode_and_flush(tmp_path):
    """Verify that in_memory mode runs in RAM and can flush to / load from disk."""
    dir_path = tmp_path / "test_in_mem"
    buf = ReplayBuffer(
        buffer_dir=dir_path,
        capacity=100,
        in_memory=True,
        create_new=True,
    )
    assert buf.in_memory is True
    # Planes should be regular numpy array in RAM, not memmap
    assert not isinstance(buf.planes, np.memmap)

    positions = [_create_dummy_pos(i) for i in range(40)]
    buf.add_positions(positions)
    assert len(buf) == 40
    assert buf.write_head == 40

    # Flush to disk
    buf.flush_to_disk()
    assert (dir_path / "planes.bin").exists()

    # Re-open in in_memory mode and verify loaded from disk
    buf2 = ReplayBuffer(
        buffer_dir=dir_path,
        capacity=100,
        in_memory=True,
        create_new=False,
    )
    assert len(buf2) == 40
    assert buf2.write_head == 40
    assert buf2.planes[0, 0, 0, 0] == 0.0
    assert buf2.planes[39, 0, 0, 0] == 39.0


def test_gpu_resident_replay_cache_operations():
    """Verify circular GPU tensor cache insertion and sampling."""
    import torch
    from python.selfplay.replay_buffer import GpuResidentReplayCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = GpuResidentReplayCache(capacity=50, device=device)
    assert len(cache) == 0

    # Insert 30 positions
    planes = np.zeros((30, 19, 8, 8), dtype=np.float32)
    planes[:, 0, 0, 0] = np.arange(30)
    visits = np.zeros((30, 4672), dtype=np.float32)
    outcomes = np.arange(30) % 3
    full_search = np.ones(30, dtype=bool)

    cache.add_positions(planes, visits, outcomes, full_search)
    assert len(cache) == 30
    assert cache.write_head == 30

    # Insert another 30 positions (total 60, wraps around 50 capacity)
    planes2 = np.zeros((30, 19, 8, 8), dtype=np.float32)
    planes2[:, 0, 0, 0] = np.arange(30, 60)
    cache.add_positions(planes2, visits, outcomes, full_search)
    assert len(cache) == 50
    assert cache.write_head == 10

    # Sample batch directly on device
    batch = cache.sample_batch(batch_size=16)
    assert batch["planes"].shape == (16, 19, 8, 8)
    assert batch["planes"].device.type == device.type
    assert batch["visits"].shape == (16, 4672)
    assert batch["outcomes"].shape == (16,)
    assert batch["is_full_search"].shape == (16,)


def test_sample_batch_tensors_tiered(tmp_path):
    """Verify sample_batch_tensors returns torch Tensors directly on target device with zero PCIe overhead."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_path = tmp_path / "test_tiered"
    buf = ReplayBuffer(
        buffer_dir=dir_path,
        capacity=200,
        gpu_cache_capacity=100,
        gpu_device=device,
        create_new=True,
    )
    assert buf.gpu_cache is not None

    positions = [_create_dummy_pos(i) for i in range(50)]
    buf.add_positions(positions)
    assert len(buf.gpu_cache) == 50

    batch, mix = buf.sample_batch_tensors(batch_size=32, step=0, device=device)
    assert isinstance(batch["planes"], torch.Tensor)
    assert batch["planes"].shape == (32, 19, 8, 8)
    assert batch["planes"].device.type == device.type
    assert isinstance(batch["visits"], torch.Tensor)
    assert isinstance(batch["outcomes"], torch.Tensor)
    assert batch["is_full_search"].shape == (32,)


def test_replay_buffer_write_head_modulo_wrap(tmp_path):
    """Verify write_head property clamps modulo capacity and handles slice wrap-around safely."""
    capacity = 100
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_wrap_clamp",
        capacity=capacity,
        create_new=True,
    )
    # Setting an out-of-range write_head (e.g. from a checkpoint saved with larger capacity)
    buf.write_head = 350
    assert buf.write_head == 50  # 350 % 100

    # Add positions that wrap around the capacity boundary
    positions = [_create_dummy_pos(i) for i in range(80)]
    buf.add_positions(positions)
    # 50 + 80 = 130 -> write_head should be 30
    assert buf.write_head == 30
    assert len(buf) == 80
    # Slices verify:
    # positions 0..49 were from start=50 part1=50
    # positions 50..79 were from start=0 part2=30
    assert buf.planes[50, 0, 0, 0] == 0.0
    assert buf.planes[99, 0, 0, 0] == 49.0
    assert buf.planes[0, 0, 0, 0] == 50.0
    assert buf.planes[29, 0, 0, 0] == 79.0

    # Add 30 more positions to reach capacity 100
    positions_extra = [_create_dummy_pos(i + 80) for i in range(30)]
    buf.add_positions(positions_extra)
    assert buf.write_head == 60
    assert len(buf) == capacity


def test_gpu_resident_human_cache_sampling(tmp_path, synthetic_human_dir):
    """Verify GPU-resident human cache placement, host RAM freeing, and zero-PCIe sampling."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_p = Path("data/dataset_scaled/train")
    human_dir = default_p if (default_p / "metadata.json").is_file() else synthetic_human_dir
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "test_gpu_human",
        capacity=500,
        gpu_cache_capacity=100,
        gpu_device=device,
        human_data_dir=human_dir,
        human_cache_size=200,
        create_new=True,
    )

    if device.type == "cuda":
        # Host RAM should be freed
        assert buf._human_cache_planes is None
        assert buf._human_gpu_planes is not None
        assert buf._human_gpu_planes.device.type == "cuda"
        assert len(buf._human_gpu_planes) == 200

    # Add 50 self-play positions
    positions = [_create_dummy_pos(i) for i in range(50)]
    buf.add_positions(positions)

    # Sample tensors directly on device
    batch_t, mix_t = buf.sample_batch_tensors(batch_size=32, step=0, device=device)
    assert isinstance(batch_t["planes"], torch.Tensor)
    assert batch_t["planes"].shape == (32, 19, 8, 8)
    assert batch_t["planes"].device.type == device.type
    assert isinstance(batch_t["visits"], torch.Tensor)
    assert batch_t["visits"].shape == (32, 4672)
    assert isinstance(batch_t["outcomes"], torch.Tensor)
    assert batch_t["outcomes"].shape == (32,)

    # Sample numpy batch (compatibility verification)
    batch_np, mix_np = buf.sample_batch(batch_size=32, step=0)
    assert isinstance(batch_np["planes"], np.ndarray)
    assert batch_np["planes"].shape == (32, 19, 8, 8)
    assert isinstance(batch_np["visits"], np.ndarray)
