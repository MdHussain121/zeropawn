"""Comprehensive test suite for the redesigned CLI, Rich UI, presets, models, and operations."""

import json
from pathlib import Path
import pytest
import torch
from unittest.mock import patch

from python.cli.main import build_parser, execute_operation_from_interactive, launch_preset_quick
from python.cli.models import (
    compare_models,
    discover_checkpoints,
    get_default_model_path,
    inspect_model,
    validate_checkpoint_file,
)
from python.cli.operations import (
    CATEGORIES,
    OPERATIONS_REGISTRY,
    get_operation_by_id,
    get_operations_by_category,
)
from python.cli.presets import (
    BUILTIN_PRESETS,
    delete_preset,
    duplicate_preset,
    export_preset,
    get_all_presets,
    get_preset_by_id,
    import_preset,
    load_user_presets,
    save_preset,
    save_user_presets,
)
from python.cli.ui import (
    get_hardware_metrics,
    is_interactive,
    render_hardware_panel,
    render_operation_header,
    render_operation_summary,
)
from python.cli.wizard import enter_custom_arguments_mode, run_configuration_wizard


# ---------------- 1. Operation Registry & Parity Tests ----------------


def test_operation_registry_completeness():
    """Verify that all 42 operations are defined across all 10 categories."""
    assert len(OPERATIONS_REGISTRY) >= 42
    assert len(CATEGORIES) == 10

    # Ensure every category has at least one operation
    for cat_id in range(1, 11):
        ops = get_operations_by_category(cat_id)
        assert len(ops) > 0, f"Category {cat_id} ({CATEGORIES[cat_id]}) is empty!"

    # Ensure every operation has valid identifiers and schemas
    seen_ids = set()
    for op in OPERATIONS_REGISTRY:
        assert op.id not in seen_ids, f"Duplicate operation ID: {op.id}"
        seen_ids.add(op.id)
        assert op.name
        assert op.description
        assert op.direct_command
        assert op.requires_model in ("none", "optional", "required")
        assert 1 <= op.category_id <= 10

        # Verify arguments schema
        for arg in op.arguments:
            assert arg.name
            if not getattr(arg, "is_positional", False):
                assert arg.flag.startswith("-")
            assert arg.type in ("int", "float", "str", "bool", "choice", "path")


def test_operation_lookup():
    """Verify lookup helpers."""
    op = get_operation_by_id("train_rl")
    assert op is not None
    assert op.name == "Reinforcement Learning Self-Play Continuous Training"
    assert op.category_id == 1

    missing = get_operation_by_id("nonexistent_xyz")
    assert missing is None


# ---------------- 2. Model Discovery, Inspection & Comparison ----------------


def test_model_discovery():
    """Verify model discovery discovers existing .pt files in repo."""
    ckpts = discover_checkpoints()
    assert isinstance(ckpts, list)
    if ckpts:
        first = ckpts[0]
        assert "name" in first
        assert "path" in first
        assert "size_mb" in first
        assert first["path"].is_file()


def test_default_model_path():
    """Verify default model resolver."""
    p = get_default_model_path()
    if p is not None:
        assert p.is_file()
        assert p.suffix == ".pt"


def test_validate_checkpoint_file(tmp_path):
    """Verify checkpoint file validation accepts valid torch files and rejects bad ones."""
    # Non-existent file
    ok, err = validate_checkpoint_file(tmp_path / "missing.pt")
    assert not ok
    assert "not found" in err.lower()

    # Invalid extension
    bad_ext = tmp_path / "model.bin"
    bad_ext.write_text("dummy")
    ok, err = validate_checkpoint_file(bad_ext)
    assert not ok
    assert "must be .pt" in err

    # Valid dummy torch dict
    valid_pt = tmp_path / "valid_model.pt"
    torch.save({"step": 100, "num_blocks": 12, "num_filters": 128}, valid_pt)
    ok, err = validate_checkpoint_file(valid_pt)
    assert ok
    assert "valid" in err.lower()


def test_inspect_and_compare_models(tmp_path):
    """Verify model inspection and comparison functions execute without error."""
    m1 = tmp_path / "m1.pt"
    m2 = tmp_path / "m2.pt"

    torch.save(
        {
            "step": 1000,
            "best_val_loss": 2.5,
            "num_blocks": 12,
            "num_filters": 128,
            "model": {"conv.weight": torch.zeros((128, 19, 3, 3))},
        },
        m1,
    )

    torch.save(
        {
            "step": 2000,
            "best_val_loss": 2.1,
            "num_blocks": 12,
            "num_filters": 128,
            "model": {"conv.weight": torch.zeros((128, 19, 3, 3))},
        },
        m2,
    )

    assert inspect_model(m1) == 0
    assert compare_models(m1, m2) == 0


# ---------------- 3. Presets System & Persistence ----------------


def test_builtin_presets():
    """Verify built-in presets are valid and non-empty."""
    assert len(BUILTIN_PRESETS) >= 12
    for pid, p in BUILTIN_PRESETS.items():
        assert p["id"] == pid
        assert p["name"]
        assert p["operation"]
        assert p["is_builtin"] is True
        assert "tradeoff" in p


def test_user_presets_crud_and_persistence(tmp_path, monkeypatch):
    """Test full CRUD lifecycle for user presets with atomic persistence."""
    fake_presets_file = tmp_path / "configs" / "presets.json"
    import python.cli.presets as pr_module

    monkeypatch.setattr(pr_module, "PRESETS_FILE", fake_presets_file)

    # Initially empty user presets
    assert load_user_presets() == {}

    # Create preset
    ok = save_preset(
        preset_id="test-custom-1",
        name="Custom Test Preset",
        description="Testing preset",
        operation="test",
        args=["--suite", "fast", "-k", "import"],
        parameters={"suite": "fast", "filter": "import"},
    )
    assert ok is True
    assert fake_presets_file.is_file()

    # Retrieve preset
    loaded = get_preset_by_id("test-custom-1")
    assert loaded is not None
    assert loaded["name"] == "Custom Test Preset"
    assert loaded["is_builtin"] is False

    # Cannot overwrite built-in
    assert save_preset("quick-smoke", "Hacked", "", "test", []) is False

    # Duplicate preset
    ok_dup = duplicate_preset("test-custom-1", "test-custom-2", "Duplicated Preset")
    assert ok_dup is True
    dup = get_preset_by_id("test-custom-2")
    assert dup is not None
    assert dup["name"] == "Duplicated Preset"

    # Export preset
    exp_file = tmp_path / "exported.json"
    assert export_preset("test-custom-1", exp_file) is True
    assert exp_file.is_file()

    # Import preset
    assert import_preset(exp_file) is True

    # Delete preset
    assert delete_preset("test-custom-2") is True
    assert get_preset_by_id("test-custom-2") is None

    # Cannot delete built-in preset
    assert delete_preset("quick-smoke") is False


# ---------------- 4. Wizard & Custom Arguments Mode ----------------


def test_custom_arguments_parsing(monkeypatch):
    """Test Path C custom arguments parser with spaces and flags."""
    op = get_operation_by_id("train_supervised")
    assert op is not None

    initial_params = {"epochs": 5, "batch_size": 256, "train_dir": "data/default"}

    # Mock user typing custom CLI args
    custom_str = '--epochs 10 --batch-size 128 --train-dir "data/my dataset/train"'
    monkeypatch.setattr("rich.console.Console.input", lambda self, *args, **kwargs: custom_str)

    res = enter_custom_arguments_mode(op, initial_params)
    assert res is not None
    assert res["epochs"] == 10
    assert res["batch_size"] == 128
    assert res["train_dir"] == "data/my dataset/train"


# ---------------- 5. UI & Hardware Telemetry ----------------


def test_hardware_telemetry_honest_measurement():
    """Verify hardware telemetry returns honest metrics or N/A."""
    hw = get_hardware_metrics()
    assert "cpu_percent" in hw
    assert "ram_used_gb" in hw
    assert "ram_total_gb" in hw
    assert "gpu_name" in hw

    # Verify panel renders without error
    panel = render_hardware_panel()
    assert panel is not None


def test_standard_output_contract():
    """Verify pre-run and post-run contract rendering."""
    # Should render cleanly without exception
    render_operation_header(
        op_name="unit-test-smoke",
        description="Testing output contract",
        parameters={"epochs": 5, "lr": 0.001},
        device="cpu",
    )

    render_operation_summary(
        op_name="unit-test-smoke",
        status="SUCCESS",
        duration_sec=1.23,
        metrics={"accuracy": "95.5%", "loss": "0.123"},
        warnings_count=0,
        exit_code=0,
    )


# ---------------- 6. Direct CLI Parser Parity Tests ----------------


def test_cli_subcommands_registration():
    """Verify all new and existing subcommands are registered in the top-level parser."""
    parser = build_parser()

    # Test all command namespaces
    expected_cmds = [
        "test",
        "train",
        "supervised",
        "match",
        "puzzles",
        "add-puzzles",
        "status",
        "uci",
        "cleanup",
        "finetune",
        "data",
        "eval",
        "bench",
        "diagnose",
        "tablebase",
        "profile",
        "models",
        "presets",
        "interactive",
    ]

    for cmd in expected_cmds:
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([cmd, "--help"])
        assert exc_info.value.code == 0


def test_all_catalog_operations_interactive_dispatch():
    """Verify all catalog operations can be successfully parsed and dispatched via interactive mode."""
    assert len(OPERATIONS_REGISTRY) >= 80
    model_path = Path("models/checkpoints_scaled/best_model.pt")

    def dummy_func(args):
        return 0

    with patch("python.cli.ui.console.print"), patch("rich.console.Console.print"):
        for op in OPERATIONS_REGISTRY:
            default_params = {a.name: a.default for a in op.arguments}

            real_parser = build_parser()
            orig_parse_args = real_parser.parse_args

            def hooked_parse(tokens):
                args = orig_parse_args(tokens)
                args.func = dummy_func
                return args

            with (
                patch("python.cli.main.build_parser", return_value=real_parser),
                patch.object(real_parser, "parse_args", side_effect=hooked_parse),
            ):
                code = execute_operation_from_interactive(op, default_params, model_path)
                assert code == 0, f"Operation {op.id} failed interactive dispatch with code {code}"


def test_interactive_positional_arguments_integrity():
    """Verify that operations with positional arguments pass them positionally without flags."""
    positional_ops = [
        "config_get",
        "config_set",
        "config_export",
        "presets_diff",
        "runs_inspect",
        "runs_recover",
        "logs_export",
        "models_compare",
    ]
    for op_id in positional_ops:
        op = get_operation_by_id(op_id)
        assert op is not None, f"Operation {op_id} not found"
        for arg in op.arguments:
            if arg.is_positional:
                assert not arg.flag.startswith("-"), (
                    f"Positional argument {arg.name} in {op_id} should not start with '-'"
                )


def test_all_subcommands_registration_exhaustive():
    """Exhaustively verify that all 52 registered subcommands and aliases answer to --help cleanly."""
    import argparse

    parser = build_parser()
    subaction = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert len(subaction.choices) >= 50

    for cmd in subaction.choices.keys():
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([cmd, "--help"])
        assert exc_info.value.code == 0, (
            f"Command '{cmd} --help' failed with code {exc_info.value.code}"
        )


def test_intelligent_checkpoint_appending_rules():
    """Verify exact token transformations for each intelligent checkpoint injection rule."""
    model_path = Path("models/checkpoints_scaled/best_model.pt")
    captured_tokens = {}

    def dummy_func(args):
        return 0

    with patch("python.cli.ui.console.print"), patch("rich.console.Console.print"):
        test_ops = [
            "train_rl",  # accepts --checkpoint
            "eval_baseline",  # accepts --checkpoint
            "train_finetune",  # accepts --resume
            "models_validate",  # accepts positional checkpoint
            "models_inspect",  # accepts --checkpoint
            "eval_all_baseline",  # rejects checkpoint
            "bench_pipelined",  # rejects checkpoint
            "diagnose_lag",  # rejects checkpoint
            "test_guardrails",  # rejects checkpoint
        ]
        for op_id in test_ops:
            op = get_operation_by_id(op_id)
            default_params = {a.name: a.default for a in op.arguments}
            real_parser = build_parser()
            orig_parse_args = real_parser.parse_args

            def make_hook(current_op_id):
                def hooked(tokens):
                    captured_tokens[current_op_id] = list(tokens)
                    args = orig_parse_args(tokens)
                    args.func = dummy_func
                    return args

                return hooked

            with (
                patch("python.cli.main.build_parser", return_value=real_parser),
                patch.object(real_parser, "parse_args", side_effect=make_hook(op_id)),
            ):
                code = execute_operation_from_interactive(op, default_params, model_path)
                assert code == 0, f"{op_id} failed"

    # 1. Accepts --checkpoint
    assert "--checkpoint" in captured_tokens["train_rl"]
    assert "--checkpoint" in captured_tokens["eval_baseline"]
    assert "--checkpoint" in captured_tokens["models_inspect"]

    # 2. Accepts positional checkpoint
    assert str(model_path) in captured_tokens["models_validate"]
    assert "--checkpoint" not in captured_tokens["models_validate"]

    # 3. Accepts --resume
    assert "--resume" in captured_tokens["train_finetune"]
    assert "--checkpoint" not in captured_tokens["train_finetune"]

    # 4. Rejects checkpoint
    assert "--checkpoint" not in captured_tokens["eval_all_baseline"]
    assert str(model_path) not in captured_tokens["eval_all_baseline"]
    assert "--checkpoint" not in captured_tokens["bench_pipelined"]
    assert "--checkpoint" not in captured_tokens["diagnose_lag"]
    assert "--checkpoint" not in captured_tokens["test_guardrails"]


def test_launch_preset_quick_dispatch():
    """Verify Path A preset quick execution parses and dispatches all 12 built-in presets without token collision."""
    from python.cli.presets import BUILTIN_PRESETS

    def dummy_func(args):
        return 0

    with patch("python.cli.ui.console.print"), patch("rich.console.Console.print"):
        for preset_id in BUILTIN_PRESETS:
            real_parser = build_parser()
            orig_parse_args = real_parser.parse_args

            def hooked(tokens):
                args = orig_parse_args(tokens)
                args.func = dummy_func
                return args

            with (
                patch("python.cli.main.build_parser", return_value=real_parser),
                patch.object(real_parser, "parse_args", side_effect=hooked),
            ):
                code = launch_preset_quick(preset_id)
                assert code == 0, f"Preset '{preset_id}' failed quick run dispatch with code {code}"


def test_configuration_wizard_rendering():
    """Verify Path B Configuration Wizard renders without NameError across all operations."""
    fetch_elite_op = get_operation_by_id("data_fetch_elite")
    assert fetch_elite_op is not None

    with (
        patch("python.cli.ui.console.print"),
        patch("rich.console.Console.print"),
        patch("rich.console.Console.input", return_value="C"),
    ):
        res = run_configuration_wizard(fetch_elite_op)
        assert res is None

    # Test all operations in the registry
    with (
        patch("python.cli.ui.console.print"),
        patch("rich.console.Console.print"),
        patch("rich.console.Console.input", return_value="C"),
    ):
        for op in OPERATIONS_REGISTRY:
            res = run_configuration_wizard(op)
            assert res is None
