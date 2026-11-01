"""Regression tests for DI-03: Legal-move mask emission in GpuResidentDataLoader.

Verifies:
1. decode_packed_masks matches np.unpackbits bit-for-bit across all 4672 actions.
2. GpuResidentDataLoader emits 'legal_mask' in every batch when masks.bin is present.
3. Batch 'legal_mask' is bool dtype and shape (batch_size, 4672).
4. Policy loss with legal masking operates correctly and trains without warning.
5. Operates seamlessly on both CPU and CUDA devices.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
import torch

from python.data.build_dataset import build_binary_dataset
from python.data.dataset import GpuResidentDataLoader, decode_packed_masks, MASK_PACKED_BYTES
from python.train.train_supervised import legal_masked_policy_loss

SAMPLE_PGN = """
[Event "Rated Classical"]
[Site "https://lichess.org/test001"]
[Date "2026.10.01"]
[White "Master1"]
[Black "Master2"]
[Result "1-0"]
[WhiteElo "2600"]
[BlackElo "2600"]
[Termination "Normal"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Classical"]
[Site "https://lichess.org/test002"]
[Date "2026.10.01"]
[White "Master3"]
[Black "Master4"]
[Result "0-1"]
[WhiteElo "2600"]
[BlackElo "2600"]
[Termination "Normal"]
[TimeControl "600+0"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 c6 7. Bd3 Nbd7 8. Qc2 O-O 9. Nf3 Re8 10. O-O Nf8 0-1

[Event "Rated Classical"]
[Site "https://lichess.org/test003"]
[Date "2026.10.01"]
[White "Master5"]
[Black "Master6"]
[Result "1/2-1/2"]
[WhiteElo "2600"]
[BlackElo "2600"]
[Termination "Normal"]
[TimeControl "600+0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 1/2-1/2

[Event "Rated Classical"]
[Site "https://lichess.org/test004"]
[Date "2026.10.01"]
[White "Master7"]
[Black "Master8"]
[Result "1-0"]
[WhiteElo "2600"]
[BlackElo "2600"]
[Termination "Normal"]
[TimeControl "600+0"]

1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 6. Bg2 Nb6 7. O-O Be7 8. a3 O-O 9. b4 Be6 10. d3 f6 1-0
""".strip()


def test_decode_packed_masks_matches_numpy():
    """Ensure decode_packed_masks is bit-identical to np.unpackbits."""
    rng = np.random.default_rng(42)
    packed_np = rng.integers(0, 256, size=(16, MASK_PACKED_BYTES), dtype=np.uint8)
    packed_torch = torch.from_numpy(packed_np)

    decoded_torch = decode_packed_masks(packed_torch).cpu().numpy()
    decoded_np = np.unpackbits(packed_np, axis=1)[:, :4672].astype(bool)

    np.testing.assert_array_equal(decoded_torch, decoded_np)


def test_gpu_resident_loader_emits_legal_masks(tmp_path: Path):
    """Test GpuResidentDataLoader emits legal_mask on CPU and CUDA."""
    data_dir = tmp_path / "dataset"
    build_binary_dataset(io.StringIO(SAMPLE_PGN), data_dir, force=True)

    train_dir = data_dir / "train"
    assert (train_dir / "masks.bin").is_file()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    for device in devices:
        loader = GpuResidentDataLoader(
            data_dir=str(train_dir),
            batch_size=4,
            shuffle=False,
            device=device,
        )
        assert loader.has_masks is True

        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert "legal_mask" in batch
            mask = batch["legal_mask"]
            assert mask.dtype == torch.bool
            assert mask.ndim == 2
            assert mask.shape[1] == 4672
            assert mask.device.type == device.type

            # Check that the played move is legal in every row of the batch
            moves = batch["move"]
            for i in range(len(moves)):
                mv = moves[i].item()
                assert mask[i, mv].item() is True, f"Move {mv} was not marked legal"

            # Check legal_masked_policy_loss runs without error
            logits = torch.randn(len(moves), 4672, device=device)
            loss = legal_masked_policy_loss(logits, moves, mask)
            assert torch.isfinite(loss)
            assert loss.item() > 0

        assert batch_count > 0
        loader.close()
