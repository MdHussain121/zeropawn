"""Comprehensive test suite for the Universal ZeroPawn CLI overhaul.

Validates:
1. 4-tier configuration precedence resolution and provenance.
2. `chess config` subcommands (show, get, set, reset, validate, diff, export).
3. `chess presets diff` comparison.
4. `chess data` inspection, validation, and leakage detection.
5. `chess models` validation, compatibility checking, and export.
6. `chess runs` and `chess logs` lifecycle operations.
7. `chess diagnose` env, health, and parity subcommands.
8. `chess build` contracts and native C++ test dispatch.
9. Conflicting initialization rejection (--checkpoint vs --random-init vs --resume).
10. Strict UCI stdout purity (no ANSI codes).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from python.cli.config import (
    CONFIG_REGISTRY,
    ResolvedConfiguration,
    get_config_option,
    load_config_file,
    reset_config_option,
    resolve_configuration,
    set_config_option,
    validate_config_dict,
)
from python.cli.main import build_parser, main
from python.cli.presets import get_all_presets, get_preset_by_id


def test_config_registry_integrity():
    """Verify all configuration domains and fields are registered properly."""
    assert len(CONFIG_REGISTRY) >= 30
    for key, cf in CONFIG_REGISTRY.items():
        assert cf.key == key
        assert cf.alias
        assert cf.type_name in ("int", "float", "bool", "str", "choice", "path")
        assert cf.description
        assert cf.runtime_dest


def test_four_tier_precedence_resolution(tmp_path):
    """Test resolution order: CLI args > Preset > Config file > Default."""
    test_cfg_path = tmp_path / "test_config.json"

    # Tier 4: Default
    resolved_default = resolve_configuration(config_path=test_cfg_path)
    rec = resolved_default.get_record("training.max_hours")
    assert rec is not None
    assert rec.source == "default"
    assert rec.effective_value == 72.0
    assert not rec.is_override

    # Tier 3: Config file overrides default
    set_config_option("training.max_hours", 24.0, config_path=test_cfg_path)
    resolved_file = resolve_configuration(config_path=test_cfg_path)
    rec_file = resolved_file.get_record("training.max_hours")
    assert rec_file.source == "config_file"
    assert rec_file.effective_value == 24.0
    assert rec_file.is_override

    # Tier 2: Preset overrides config file
    preset_dict = {"max_hours": 12.0}
    resolved_preset = resolve_configuration(preset_dict=preset_dict, config_path=test_cfg_path)
    rec_preset = resolved_preset.get_record("training.max_hours")
    assert rec_preset.source == "preset"
    assert rec_preset.effective_value == 12.0

    # Tier 1: CLI args overrides preset and config file
    cli_args = {"max_hours": 4.5}
    resolved_cli = resolve_configuration(
        cli_args=cli_args,
        preset_dict=preset_dict,
        config_path=test_cfg_path,
    )
    rec_cli = resolved_cli.get_record("training.max_hours")
    assert rec_cli.source == "cli"
    assert rec_cli.effective_value == 4.5


def test_config_crud_and_validation(tmp_path):
    """Test get, set, reset, and validation of configuration values."""
    cfg_path = tmp_path / "custom_config.json"

    # Set valid option
    ok, msg = set_config_option("selfplay.concurrent_games", 32, config_path=cfg_path)
    assert ok
    val, cf = get_config_option("selfplay.concurrent_games", config_path=cfg_path)
    assert val == 32
    assert cf.alias == "concurrent_games"

    # Set invalid option (out of bounds)
    ok_bad, msg_bad = set_config_option("selfplay.concurrent_games", 500, config_path=cfg_path)
    assert not ok_bad
    assert "exceeds maximum" in msg_bad

    # Validation errors
    errors = validate_config_dict({"selfplay.concurrent_games": 999, "unknown_key": "val"})
    assert len(errors) == 2

    # Reset option
    ok_reset, _ = reset_config_option("selfplay.concurrent_games", config_path=cfg_path)
    assert ok_reset
    val_reset, _ = get_config_option("selfplay.concurrent_games", config_path=cfg_path)
    assert val_reset == 24  # Default (R-13: aligned with the standalone runner)


def test_presets_diff():
    """Verify preset diffing correctly highlights parameter differences."""
    p1 = get_preset_by_id("quick-smoke")
    p2 = get_preset_by_id("fast-tests")
    assert p1 is not None and p2 is not None

    parser = build_parser()
    args = parser.parse_args(["presets", "diff", "quick-smoke", "fast-tests"])
    ret = args.func(args)
    assert ret == 0


def _write_hermetic_dataset(root: Path) -> Path:
    """Build a tiny games_wdl dataset the data CLI can validate end-to-end.

    Mirrors the split layout of tests/test_data_integrity.py::make_split so the
    CLI's validate/inspect/check-leakage paths all find what they expect. The
    dataset lives entirely under tmp_path — running the data CLI against the
    production tree from a test is forbidden (a pre-2026-11-03 version of this
    test validated data/dataset_training directly, and the then-default
    --write-receipt rewrote the production receipt on every run).
    """
    import numpy as np

    ds = root / "dataset_hermetic"
    base_ids = {"train": 1000, "val": 5000, "test": 9000}  # disjoint game-id ranges
    n = 24
    rng = np.random.default_rng(11)
    for split, count in (("train", n), ("val", 8), ("test", 8)):
        sd = ds / split
        sd.mkdir(parents=True, exist_ok=True)
        planes = rng.random((count, 19, 8, 8), dtype=np.float32).round()
        planes.tofile(sd / "planes.bin")
        np.asarray([10] * count, dtype=np.int16).tofile(sd / "moves.bin")
        np.zeros(count, dtype=np.uint8).tofile(sd / "outcomes.bin")  # all wins: real WDL labels
        np.asarray(
            [base_ids[split] + i for i in range(count)], dtype=np.uint64
        ).tofile(sd / "game_ids.bin")
        np.asarray([10 + i for i in range(count)], dtype=np.int16).tofile(sd / "plies.bin")
        (sd / "metadata.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "num_samples": count,
                    "role": "games_wdl",
                    "kind": "games",
                    "label_semantics": {"role": "games_wdl", "wdl_supervised": True},
                }
            ),
            encoding="utf-8",
        )
    (ds / "metadata.json").write_text(
        json.dumps(
            {
                "name": "dataset_hermetic",
                "kind": "games_wdl",
                "role": "games_wdl",
                "num_samples": n + 16,
                "splits": ["train", "val", "test"],
                "build_record": {
                    "builder": "test_cli_universal hermetic fixture",
                    "builder_version": "1",
                    "source": "synthetic",
                    "created_at": "2026-11-03T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    outputs = {}
    for split in ("train", "val", "test"):
        sd = ds / split
        outputs[split] = {
            name: {"sha256": _sha(sd / name)}
            for name in ("planes.bin", "moves.bin", "outcomes.bin", "game_ids.bin", "plies.bin")
        }
    (ds / "build_record.json").write_text(
        json.dumps(
            {
                "build_record_version": "1.0",
                "builder": "test_cli_universal hermetic fixture",
                "builder_version": "1",
                "source": "synthetic",
                "created_at": "2026-11-03T00:00:00Z",
                "counts": {"train": n, "val": 8, "test": 8},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return ds


def test_data_subcommands_registration(tmp_path):
    """Verify extended dataset operations are executable without crashing.

    Fully hermetic: operates on a synthetic dataset under tmp_path. The
    validate leg runs an exhaustive (full) scan so the exit code exercises the
    success path, and passes --write-receipt explicitly to prove certification
    writes land in the temp dataset only.
    """
    parser = build_parser()

    # list
    args_list = parser.parse_args(["data", "list"])
    assert args_list.func(args_list) == 0

    ds = _write_hermetic_dataset(tmp_path)

    # inspect
    args_insp = parser.parse_args(["data", "inspect", str(ds)])
    assert args_insp.func(args_insp) == 0

    # validate (full scan → definitive pass; receipt written into tmp dataset)
    args_val = parser.parse_args(
        ["data", "validate", "--dir", str(ds), "--scan-planes", "full", "--write-receipt"]
    )
    assert args_val.func(args_val) == 0
    assert (ds / "validation_receipt.json").is_file()

    # leakage check (--max-positions makes it a bounded scan; --allow-bounded
    # accepts the bounded result so the leg asserts executability, exit 0)
    args_leak = parser.parse_args(
        [
            "data",
            "check-leakage",
            "--train-dir",
            str(ds / "train"),
            "--val-dir",
            str(ds / "val"),
            "--max-positions",
            "100",
            "--allow-bounded",
        ]
    )
    assert args_leak.func(args_leak) == 0


def test_models_subcommands_registration(tmp_path):
    """Verify model inspection, validation, and compatibility commands."""
    parser = build_parser()

    # list
    args_list = parser.parse_args(["models", "list"])
    assert args_list.func(args_list) == 0

    # Ensure model checkpoint exists for inspection, compat, and validate
    model_path = Path("models/checkpoints_scaled/best_model.pt")
    if not model_path.is_file():
        from python.model.chess_net import ChessNet
        import torch

        dummy = ChessNet(num_blocks=12, num_filters=128)
        dummy_path = tmp_path / "dummy_scaled_best.pt"
        torch.save(
            {
                "model": dummy.state_dict(),
                "model_state_dict": dummy.state_dict(),
                "epoch": 1,
                "loss": 0.5,
            },
            dummy_path,
        )
        test_path_str = str(dummy_path)
    else:
        test_path_str = str(model_path)

    # inspect
    args_insp = parser.parse_args(["models", "inspect", test_path_str])
    assert args_insp.func(args_insp) == 0

    # check-compat
    args_compat = parser.parse_args(["models", "check-compat", test_path_str])
    assert args_compat.func(args_compat) == 0

    # validate numerical stability
    args_val = parser.parse_args(["models", "validate", test_path_str])
    assert args_val.func(args_val) == 0


def test_runs_and_logs_subcommands():
    """Verify run history and log inspection commands."""
    parser = build_parser()

    # runs list
    args_runs = parser.parse_args(["runs", "list"])
    assert args_runs.func(args_runs) == 0

    # runs interrupted
    args_intr = parser.parse_args(["runs", "interrupted"])
    assert args_intr.func(args_intr) == 0

    # logs tail (10 lines)
    args_logs = parser.parse_args(["logs", "tail", "--lines", "10"])
    assert args_logs.func(args_logs) == 0


def test_diagnose_subcommands():
    """Verify diagnostic subcommands."""
    parser = build_parser()

    # diagnose env
    args_env = parser.parse_args(["diagnose", "env"])
    assert args_env.func(args_env) == 0

    # diagnose health
    args_hlth = parser.parse_args(["diagnose", "health"])
    ret_hlth = args_hlth.func(args_hlth)
    # On clean CI checkouts, missing pre-downloaded baseline weights produces attention required (code 1)
    assert ret_hlth in (0, 1)

    # diagnose parity
    args_par = parser.parse_args(["diagnose", "parity"])
    assert args_par.func(args_par) == 0


def test_build_subcommands():
    """Verify C++ build inspection and contracts."""
    parser = build_parser()

    # build info / status
    args_info = parser.parse_args(["build", "info"])
    assert args_info.func(args_info) == 0

    # build contracts / compat
    args_cntr = parser.parse_args(["build", "contracts"])
    assert args_cntr.func(args_cntr) == 0

    # build test
    args_test = parser.parse_args(["build", "test"])
    assert args_test.func(args_test) == 0


def test_conflicting_init_rejected():
    """Verify that specifying multiple initialization modes is rejected."""
    parser = build_parser()
    args = parser.parse_args(
        ["train", "--checkpoint", "models/checkpoints_scaled/best_model.pt", "--random-init"]
    )
    ret = args.func(args)
    assert ret != 0


def test_uci_stdout_zero_ansi():
    """Verify strict UCI protocol output without any ANSI escape sequences."""
    import subprocess
    import sys
    from python.cli.common import REPO_ROOT

    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "cli.py"), "uci"],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout, stderr = proc.communicate(input="uci\nisready\nquit\n", timeout=10)

    assert "id name ZeroPawn" in stdout
    assert "uciok" in stdout
    assert "readyok" in stdout

    # ANSI regex check
    ansi_pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    assert not ansi_pattern.search(stdout), f"Found ANSI escape in UCI stdout: {stdout!r}"
