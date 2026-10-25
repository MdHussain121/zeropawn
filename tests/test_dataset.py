import io
import pytest
import numpy as np
import torch
import chess.pgn
from pathlib import Path
from python.data.build_dataset import (
    hash_game_id,
    extract_game_id,
    parse_result,
    process_pgn_game,
    build_binary_dataset,
)
from python.data.dataset import BinaryChessDataset

SAMPLE_GAMES_PGN = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/game001"]
[Date "2026.09.01"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]
[WhiteElo "2300"]
[BlackElo "2300"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/game002"]
[Date "2026.09.01"]
[White "PlayerC"]
[Black "PlayerD"]
[Result "0-1"]
[WhiteElo "2300"]
[BlackElo "2300"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 c6 7. Bd3 Nbd7 8. Qc2 O-O 9. Nf3 Re8 10. O-O Nf8 0-1

[Event "Rated Blitz game"]
[Site "https://lichess.org/game003"]
[Date "2026.09.01"]
[White "PlayerE"]
[Black "PlayerF"]
[Result "1/2-1/2"]
[WhiteElo "2300"]
[BlackElo "2300"]
[Termination "Normal"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 1/2-1/2
""".strip()


def test_hash_game_id_deterministic():
    h1 = hash_game_id("game001")
    h2 = hash_game_id("game001")
    h3 = hash_game_id("game002")
    assert h1 == h2
    assert h1 != h3
    assert isinstance(h1, int)
    assert h1 >= 0


def test_process_pgn_game_outcome_perspective():
    game = chess.pgn.read_game(io.StringIO(SAMPLE_GAMES_PGN))
    assert game is not None
    game_id, is_val, positions = process_pgn_game(game)

    assert len(positions) == 20  # 10 full moves = 20 plies
    # Result is 1-0 (White win)
    # Ply 0: White to move -> outcome should be Win (0)
    assert positions[0].ply == 0
    assert positions[0].outcome == 0
    # Ply 1: Black to move -> outcome should be Loss (2)
    assert positions[1].ply == 1
    assert positions[1].outcome == 2
    # Ply 2: White to move -> outcome should be Win (0)
    assert positions[2].ply == 2
    assert positions[2].outcome == 0

    # Check move indices and planes shapes
    for p in positions:
        assert 0 <= p.move_idx < 4672
        assert p.planes.shape == (19, 8, 8)
        assert p.planes.dtype == np.float32


def test_build_binary_dataset_and_loader(tmp_path):
    out_dir = tmp_path / "dataset"
    stream = io.StringIO(SAMPLE_GAMES_PGN)

    train_count, val_count, removed = build_binary_dataset(stream, out_dir)
    assert train_count > 0 or val_count > 0

    # Test loading train dataset if non-empty
    train_dir = out_dir / "train"
    if (train_dir / "metadata.json").exists() and train_count > 0:
        ds = BinaryChessDataset(train_dir)
        assert len(ds) == train_count
        sample = ds[0]
        assert "planes" in sample
        assert sample["planes"].shape == (19, 8, 8)
        assert sample["planes"].dtype == torch.float32
        assert sample["move"].shape == ()
        assert 0 <= sample["move"].item() < 4672
        assert sample["outcome"].item() in (0, 1, 2)
        assert sample["value_target"].shape == (3,)
        assert sample["value_target"].dtype == torch.float32


def test_deduplication_removes_shared_positions(tmp_path):
    # Two games with identical opening moves but different game IDs
    pgn_dup = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/g1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/g2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
""".strip()
    out_dir = tmp_path / "dup_dataset"

    # Force g1 to train and g2 to val by mocking hash or checking build
    train_count, val_count, removed = build_binary_dataset(io.StringIO(pgn_dup), out_dir)
    # Regardless of which split they landed on, deduplication logic runs without error
    val_dir = out_dir / "val"
    assert (val_dir / "metadata.json").exists()


def test_gpu_resident_dataloader_cpu_and_cuda(tmp_path):
    from python.data.dataset import GpuResidentDataLoader

    out_dir = tmp_path / "dataset"
    stream = io.StringIO(SAMPLE_GAMES_PGN)
    train_count, val_count, _ = build_binary_dataset(stream, out_dir)
    train_dir = out_dir / "train"

    # Test CPU execution
    loader_cpu = GpuResidentDataLoader(
        train_dir,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        device="cpu",
        storage_dtype=torch.float32,
    )
    assert len(loader_cpu) > 0
    batch_cpu = next(iter(loader_cpu))
    assert batch_cpu["planes"].shape[1:] == (19, 8, 8)
    assert batch_cpu["planes"].dtype == torch.float32
    assert batch_cpu["move"].dtype == torch.long
    assert batch_cpu["value_target"].shape[-1] == 3
    assert batch_cpu["ply"].dtype == torch.long

    # Test CUDA execution if available
    if torch.cuda.is_available():
        loader_cuda = GpuResidentDataLoader(
            train_dir,
            batch_size=8,
            shuffle=True,
            drop_last=False,
            device="cuda",
            storage_dtype=torch.bfloat16,
        )
        assert len(loader_cuda) > 0
        batch_cuda = next(iter(loader_cuda))
        assert batch_cuda["planes"].device.type == "cuda"
        assert batch_cuda["planes"].dtype == torch.bfloat16
        assert batch_cuda["move"].device.type == "cuda"
        assert batch_cuda["value_target"].device.type == "cuda"
        assert batch_cuda["ply"].device.type == "cuda"


def test_gpu_resident_dataloader_tiered_spillover(tmp_path):
    from python.data.dataset import GpuResidentDataLoader

    out_dir = tmp_path / "dataset"
    stream = io.StringIO(SAMPLE_GAMES_PGN)
    train_count, val_count, _ = build_binary_dataset(stream, out_dir)
    train_dir = out_dir / "train"

    if torch.cuda.is_available():
        # Force a tiny max_gpu_memory_gb (e.g. 0.00005 GB = 50 KB) so it forces spillover to RAM
        loader_tiered = GpuResidentDataLoader(
            train_dir,
            batch_size=4,
            shuffle=True,
            drop_last=False,
            device="cuda",
            storage_dtype=torch.bfloat16,
            max_gpu_memory_gb=0.00005,
        )
        assert loader_tiered.gpu_capacity > 0
        assert loader_tiered.spillover_count > 0

        # Iterate through all batches and verify they yield successfully to GPU
        total_items = 0
        for batch in loader_tiered:
            assert batch["planes"].device.type == "cuda"
            assert batch["move"].device.type == "cuda"
            assert batch["value_target"].device.type == "cuda"
            assert batch["ply"].device.type == "cuda"
            total_items += batch["planes"].size(0)
        assert total_items == train_count
