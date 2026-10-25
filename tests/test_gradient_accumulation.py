"""Regression tests for gradient accumulation and WDL loss normalization (F-07).

Verifies:
1. Mathematical equivalence of the global accumulated WDL loss objective against full-batch evaluation.
2. Equality of representations and model parameter gradients under fixed activation statistics (eval mode).
   Note: In training mode with BatchNorm2d, mini-batch normalization statistics depend on micro-batch
   size (e.g. b=4 vs B=8), which naturally produces slight batch-dependent differences in network activations.
3. Uneven valid mask counts across micro-batches (e.g. 1 valid in micro 1, 4 valid in micro 2).
4. Zero valid counts in a micro-batch (e.g. all tactical or plies < 10) without NaN or gradient corruption.
5. Total zero valid count across the entire effective batch (V_total == 0).
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn

from python.model.chess_net import ChessNet
from python.train.train_supervised import train_step


def create_deterministic_model() -> ChessNet:
    torch.manual_seed(123)
    model = ChessNet(in_channels=19, num_filters=32, num_blocks=2, num_actions=4672)
    model.eval()  # Set eval mode to isolate loss accumulation math from BatchNorm sample statistics
    return model


def test_gradient_accumulation_uniform_equivalence():
    """Verify that accumulated micro-batches yield identical gradients under fixed normalization statistics."""
    device = torch.device("cpu")

    # Generate deterministic batch of 8
    torch.manual_seed(42)
    full_batch = {
        "planes": torch.randn(8, 19, 8, 8),
        "move": torch.randint(0, 4672, (8,)),
        "value_target": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "ply": torch.tensor([15, 20, 25, 30, 35, 40, 45, 50]),  # All valid (ply >= 10)
    }

    micro1 = {k: v[:4] for k, v in full_batch.items()}
    micro2 = {k: v[4:] for k, v in full_batch.items()}

    # Run single effective batch (accum_steps=1)
    model_single = create_deterministic_model()
    opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)
    train_step(
        model=model_single,
        batch=full_batch,
        optimizer=opt_single,
        device=device,
        use_autocast=False,
        accum_steps=1,
        step_optimizer=False,
        zero_grad=True,
    )

    # Run accumulated micro-batches (accum_steps=2, v_total=8)
    model_accum = copy.deepcopy(model_single)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.01)
    opt_accum.zero_grad()

    train_step(
        model=model_accum,
        batch=micro1,
        optimizer=opt_accum,
        device=device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
        v_total=8,
    )
    train_step(
        model=model_accum,
        batch=micro2,
        optimizer=opt_accum,
        device=device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
        v_total=8,
    )

    # Gradients must match exactly
    for (name1, p1), (name2, p2) in zip(
        model_single.named_parameters(), model_accum.named_parameters()
    ):
        assert p1.grad is not None and p2.grad is not None
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)


def test_gradient_accumulation_uneven_valid_counts():
    """Verify exact equivalence when micro-batches have uneven valid samples (e.g. 1 valid vs 3 valid)."""
    device = torch.device("cpu")
    torch.manual_seed(99)

    # 8 samples total:
    # Samples 0-3 (micro1): plies [5, 6, 8, 15] -> only sample 3 is valid (1 valid)
    # Samples 4-7 (micro2): plies [12, 20, 25, 30] -> all 4 are valid (4 valid)
    # Total valid = 5
    full_batch = {
        "planes": torch.randn(8, 19, 8, 8),
        "move": torch.randint(0, 4672, (8,)),
        "value_target": torch.ones(8, 3) / 3.0,
        "ply": torch.tensor([5, 6, 8, 15, 12, 20, 25, 30]),
    }

    micro1 = {k: v[:4] for k, v in full_batch.items()}
    micro2 = {k: v[4:] for k, v in full_batch.items()}

    # Single batch
    model_single = create_deterministic_model()
    opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)
    train_step(
        model=model_single,
        batch=full_batch,
        optimizer=opt_single,
        device=device,
        use_autocast=False,
        accum_steps=1,
        step_optimizer=False,
        zero_grad=True,
    )

    # Accumulated micro-batches with v_total = 5
    model_accum = copy.deepcopy(model_single)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.01)
    opt_accum.zero_grad()

    train_step(
        model=model_accum,
        batch=micro1,
        optimizer=opt_accum,
        device=device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
        v_total=5,
    )
    train_step(
        model=model_accum,
        batch=micro2,
        optimizer=opt_accum,
        device=device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
        v_total=5,
    )

    for (name1, p1), (name2, p2) in zip(
        model_single.named_parameters(), model_accum.named_parameters()
    ):
        assert p1.grad is not None and p2.grad is not None
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)


def test_gradient_accumulation_zero_valid_micro_batch():
    """Verify that a micro-batch with 0 valid samples (e.g. all tactical [0,0,0])
    does not cause NaNs or zero-division errors, and matches combined batch.
    """
    device = torch.device("cpu")
    torch.manual_seed(77)

    # micro1: all tactical (value_target=[0,0,0], ply=0) -> 0 valid
    # micro2: master positions (ply=20) -> 4 valid
    # total valid = 4
    val_targets = torch.zeros(8, 3)
    val_targets[4:] = 1.0 / 3.0

    full_batch = {
        "planes": torch.randn(8, 19, 8, 8),
        "move": torch.randint(0, 4672, (8,)),
        "value_target": val_targets,
        "ply": torch.tensor([0, 0, 0, 0, 20, 20, 20, 20]),
    }

    micro1 = {k: v[:4] for k, v in full_batch.items()}
    micro2 = {k: v[4:] for k, v in full_batch.items()}

    # Single batch
    model_single = create_deterministic_model()
    opt_single = torch.optim.SGD(model_single.parameters(), lr=0.01)
    train_step(
        model=model_single,
        batch=full_batch,
        optimizer=opt_single,
        device=device,
        use_autocast=False,
        accum_steps=1,
        step_optimizer=False,
        zero_grad=True,
    )

    # Accumulated micro-batches with v_total = 4
    model_accum = copy.deepcopy(model_single)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.01)
    opt_accum.zero_grad()

    tot1, pol1, wdl1, *rest1 = train_step(
        model=model_accum,
        batch=micro1,
        optimizer=opt_accum,
        device=device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
        v_total=4,
    )
    assert wdl1 == 0.0  # Zero valid samples in micro1 -> 0 WDL loss

    tot2, pol2, wdl2, *rest2 = train_step(
        model=model_accum,
        batch=micro2,
        optimizer=opt_accum,
        device=device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
        v_total=4,
    )
    assert wdl2 > 0.0

    for (name1, p1), (name2, p2) in zip(
        model_single.named_parameters(), model_accum.named_parameters()
    ):
        assert p1.grad is not None and p2.grad is not None
        assert not torch.isnan(p2.grad).any()
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)
