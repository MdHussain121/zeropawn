"""Tests for DI-04 / P1-5: Binding dataset identity into checkpoints."""

import logging
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from python.train.train_supervised import save_checkpoint, train_supervised


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.num_blocks = 2
        self.num_filters = 16


def test_checkpoint_stores_dataset_identity(tmp_path: Path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class DummyScheduler:
        def state_dict(self):
            return {"step": 1}

    ckpt_path = tmp_path / "checkpoints" / "test_ckpt.pt"
    run_meta = {
        "run_id": "test-run-123",
        "dataset": {
            "train": {
                "receipt_id": "sha256:abcd1234abcd1234",
                "role": "games_wdl",
                "certified": True,
            }
        },
    }

    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=DummyScheduler(),
        step=1,
        epoch=1,
        best_val_loss=1.5,
        num_blocks=2,
        num_filters=16,
        run_metadata=run_meta,
    )

    loaded = torch.load(ckpt_path, weights_only=False)
    assert "run_metadata" in loaded
    meta = loaded["run_metadata"]
    assert "dataset" in meta
    assert meta["dataset"]["train"]["receipt_id"] == "sha256:abcd1234abcd1234"
    assert meta["dataset"]["train"]["certified"] is True

    # Provenance copy
    assert "provenance" in loaded
    prov = loaded["provenance"]
    assert "dataset" in prov
    assert prov["dataset"]["train"]["receipt_id"] == "sha256:abcd1234abcd1234"
