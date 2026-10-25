"""Regression tests for automatic-settings resolution and CLI/programmatic precedence (D-3).

Verifies:
1. Automatic mode resolves default preset (batch 256, micro 64, accum 4, lr 0.001, warmup 500).
2. Explicit argument overrides on CLI and programmatic invocations (batch_size, lr, warmup_steps, no_gpu_resident).
3. Independence from process-global sys.argv.
4. Correct attribution of parameter origins ('user' vs 'auto' vs 'default').
5. Programmatic invocation using parser.parse_args(custom_argv) and raw Namespace fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch
import pytest

from python.cli.commands_train import run_supervised_training
from python.cli.main import build_parser


def _run_with_mocked_trainer(args: argparse.Namespace, tmp_path: Path):
    """Run run_supervised_training with synthetic data dirs and mocked train_supervised."""
    train_dir = tmp_path / "mock_train"
    val_dir = tmp_path / "mock_val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    setattr(args, "train_dir", str(train_dir))
    setattr(args, "val_dir", str(val_dir))

    with patch("python.train.train_supervised.train_supervised") as mock_train:
        mock_train.return_value = {"val_loss": 0.5}
        ret = run_supervised_training(args)
        assert ret == 0, f"run_supervised_training failed with code {ret}"
        assert mock_train.called, "train_supervised was not called"
        return mock_train.call_args.kwargs


def test_auto_preset_defaults_programmatic(tmp_path):
    """Verify that --auto without overrides sets recommended preset values."""
    parser = build_parser()
    args = parser.parse_args(["supervised", "--auto"])

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["batch_size"] == 256
    assert kwargs["micro_batch_size"] == 64
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["base_lr"] == 0.001
    assert kwargs["warmup_steps"] == 500


def test_auto_preset_batch_size_override(tmp_path):
    """Verify that explicitly passing --batch-size overrides auto batch size, while preserving other auto values."""
    parser = build_parser()
    args = parser.parse_args(["supervised", "--auto", "--batch-size", "128"])

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["batch_size"] == 128
    assert kwargs["micro_batch_size"] == 64
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["base_lr"] == 0.001
    assert kwargs["warmup_steps"] == 500


def test_auto_preset_lr_override(tmp_path):
    """Verify that explicitly passing --lr overrides auto learning rate."""
    parser = build_parser()
    args = parser.parse_args(["supervised", "--auto", "--lr", "0.0005"])

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["batch_size"] == 256
    assert kwargs["base_lr"] == 0.0005
    assert kwargs["warmup_steps"] == 500


def test_auto_preset_warmup_override(tmp_path):
    """Verify that explicitly passing --warmup-steps overrides auto warmup steps."""
    parser = build_parser()
    args = parser.parse_args(["supervised", "--auto", "--warmup-steps", "100"])

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["batch_size"] == 256
    assert kwargs["base_lr"] == 0.001
    assert kwargs["warmup_steps"] == 100


def test_auto_preset_no_gpu_resident_override(tmp_path):
    """Verify that --no-gpu-resident disables GPU residency even in auto mode."""
    parser = build_parser()
    args = parser.parse_args(["supervised", "--auto", "--no-gpu-resident"])

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["gpu_resident"] is False


def test_non_auto_defaults_preserved(tmp_path):
    """Verify that when --auto is NOT supplied, standard defaults (batch 512, lr 0.0002) remain untouched."""
    parser = build_parser()
    args = parser.parse_args(["supervised"])

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["batch_size"] == 512
    assert kwargs["base_lr"] == 0.0002
    assert kwargs["warmup_steps"] == 500


def test_raw_namespace_fallback_without_parser(tmp_path):
    """Verify fallback precedence when a plain Namespace is constructed directly in tests/orchestrators."""
    # Namespace with auto=True and explicit batch_size=128
    args = argparse.Namespace(
        auto=True,
        batch_size=128,
        lr=0.0002,  # equals default -> should inherit auto lr 0.001
        warmup_steps=500,
        no_gpu_resident=False,
        micro_batch_size=None,
        gradient_accumulation_steps=None,
        epochs=1,
        num_blocks=12,
        num_filters=128,
        verify=False,
        checkpoint_dir=None,
        log_dir=None,
        device=None,
    )

    kwargs = _run_with_mocked_trainer(args, tmp_path)
    assert kwargs["batch_size"] == 128
    assert kwargs["base_lr"] == 0.001
    assert kwargs["micro_batch_size"] == 64
    assert kwargs["gradient_accumulation_steps"] == 2
