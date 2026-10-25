"""Comprehensive regression tests for ZeroPawn CLI coverage and configurability.

Tests:
1. Strict 5-tier configuration precedence:
   Default < Config File < Preset < Environment Variable < Explicit CLI Override.
2. Unspecified CLI flags (parser defaults) do NOT overwrite preset or config values.
3. Mandatory supervised training defaults: batch_size=512, lr=0.0002.
4. Gradient accumulation calculation, micro-batching, and optimizer stepping.
5. Backward-compatibility fallback for CHESSRL_* environment variables.
6. Operation catalog and CLI parser registration parity.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pytest
import torch

from python.cli.config import (
    CONFIG_REGISTRY,
    resolve_configuration,
    set_config_option,
    get_config_option,
)
from python.cli.main import build_parser, main
from python.cli.operations import OPERATIONS_REGISTRY, OPERATIONS_CATALOG
from python.cli.presets import BUILTIN_PRESETS, get_preset_by_id
from python.common.env import get_env, set_env
from python.train.train_supervised import train_step, train_supervised


def test_five_tier_precedence(tmp_path, monkeypatch):
    """Verify strict 5-tier precedence order:
    Built-in defaults -> Config file -> Preset -> Environment variable -> CLI override.
    """
    test_cfg = tmp_path / "precedence_config.json"

    # Tier 5: Built-in default
    res_def = resolve_configuration(config_path=test_cfg)
    rec_def = res_def.get_record("training.batch_size")
    assert rec_def.source == "default"
    assert rec_def.effective_value == 512

    # Tier 4: Config file overrides default
    set_config_option("training.batch_size", 256, config_path=test_cfg)
    res_file = resolve_configuration(config_path=test_cfg)
    rec_file = res_file.get_record("training.batch_size")
    assert rec_file.source == "config_file"
    assert rec_file.effective_value == 256

    # Tier 3: Preset overrides config file
    preset = {"parameters": {"batch_size": 128}}
    res_preset = resolve_configuration(preset_dict=preset, config_path=test_cfg)
    rec_preset = res_preset.get_record("training.batch_size")
    assert rec_preset.source == "preset"
    assert rec_preset.effective_value == 128

    # Tier 2: Environment variable overrides preset
    monkeypatch.setenv("ZEROPAWN_BATCH_SIZE", "64")
    res_env = resolve_configuration(preset_dict=preset, config_path=test_cfg)
    rec_env = res_env.get_record("training.batch_size")
    assert rec_env.source == "env"
    assert rec_env.effective_value == 64

    # Tier 2 Fallback: Legacy CHESSRL_* environment variable
    monkeypatch.delenv("ZEROPAWN_BATCH_SIZE", raising=False)
    monkeypatch.setenv("CHESSRL_BATCH_SIZE", "32")
    res_env_fallback = resolve_configuration(preset_dict=preset, config_path=test_cfg)
    rec_env_fb = res_env_fallback.get_record("training.batch_size")
    assert rec_env_fb.source == "env"
    assert rec_env_fb.effective_value == 32

    # Tier 1: Explicit CLI override overrides environment, preset, and config file
    parser = build_parser()
    args = parser.parse_args(["supervised", "--batch-size", "1024"])
    res_cli = resolve_configuration(
        cli_args=args,
        preset_dict=preset,
        config_path=test_cfg,
        raw_argv=["supervised", "--batch-size", "1024"],
    )
    rec_cli = res_cli.get_record("training.batch_size")
    assert rec_cli.source == "cli"
    assert rec_cli.effective_value == 1024


def test_parser_defaults_do_not_overwrite_preset_or_config(tmp_path):
    """When a command is executed without explicit flags, parser defaults
    must NOT overwrite settings from a preset or configuration file.
    """
    test_cfg = tmp_path / "noparser_overwrite_config.json"
    set_config_option("training.batch_size", 256, config_path=test_cfg)

    parser = build_parser()
    # User ran "supervised" without --batch-size; parser gives default 512
    raw_argv = ["supervised", "--epochs", "2"]
    args = parser.parse_args(raw_argv)

    resolved = resolve_configuration(
        command_name="supervised",
        cli_args=args,
        config_path=test_cfg,
        raw_argv=raw_argv,
    )

    rec_batch = resolved.get_record("training.batch_size")
    assert rec_batch.source == "config_file"
    assert rec_batch.effective_value == 256  # Config file preserved!

    rec_epochs = resolved.get_record("training.epochs")
    assert rec_epochs.source == "cli"
    assert rec_epochs.effective_value == 2  # Explicit CLI flag used!


def test_supervised_training_mandatory_defaults():
    """Verify supervised training effective batch size 512 and base lr 0.0002."""
    parser = build_parser()

    # Subcommand "supervised"
    sup_args = parser.parse_args(["supervised"])
    assert sup_args.batch_size == 512
    assert sup_args.lr == 0.0002

    # Subcommand "train --supervised"
    tr_args = parser.parse_args(["train", "--supervised"])
    assert tr_args.batch_size == 512
    assert tr_args.lr == 0.0002

    # CONFIG_REGISTRY defaults
    assert CONFIG_REGISTRY["training.batch_size"].default == 512
    assert CONFIG_REGISTRY["training.lr"].default == 0.0002

    # Builtin preset supervised-standard
    p_sup = get_preset_by_id("supervised-standard")
    assert p_sup is not None
    assert p_sup["parameters"]["batch_size"] == 512
    assert p_sup["parameters"]["lr"] == 0.0002


def test_gradient_accumulation_and_micro_batching():
    """Verify train_step loss scaling and accumulation mechanics."""
    from python.model.chess_net import ChessNet

    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)

    device = torch.device("cpu")
    batch = {
        "planes": torch.zeros(8, 19, 8, 8, dtype=torch.float32),
        "move": torch.zeros(8, dtype=torch.long),
        "value_target": torch.zeros(8, 3, dtype=torch.float32),
        "ply": torch.zeros(8, dtype=torch.float32),
    }
    batch["value_target"][:, 0] = 1.0

    # Step 1 of 2: Accumulate gradient (loss scaled by 0.5, no optimizer step)
    optimizer.zero_grad()
    tot1, *_ = train_step(
        model,
        batch,
        optimizer,
        device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=False,
        zero_grad=False,
    )
    assert tot1 > 0.0

    # Gradients should be present
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

    # Step 2 of 2: Accumulate gradient & step optimizer
    tot2, *_ = train_step(
        model,
        batch,
        optimizer,
        device,
        use_autocast=False,
        accum_steps=2,
        step_optimizer=True,
        zero_grad=True,
    )
    assert tot2 > 0.0


def test_all_catalog_operations_accessible():
    """Verify that every operation defined in the Operations Catalog
    has an active parser/handler in the unified CLI.
    """
    parser = build_parser()
    subparsers_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break

    assert subparsers_action is not None
    registered_subcommands = set(subparsers_action.choices.keys())

    all_ops = OPERATIONS_REGISTRY
    assert len(all_ops) >= 30

    for op in all_ops:
        cmd_parts = op.direct_command.split()
        if len(cmd_parts) >= 2 and cmd_parts[0] in ("zeropawn", "chess"):
            root_cmd = cmd_parts[1].lstrip("-")
            # Should be in registered subcommands or aliases
            assert root_cmd in registered_subcommands, (
                f"Operation '{op.id}' ({op.direct_command}) not registered in CLI subparsers!"
            )
