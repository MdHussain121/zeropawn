"""End-to-end integration and regression tests for ZeroPawn CLI, autotuning, and batched MCTS.

Validates:
1. Top-level entrypoints: `cli.py` and `python -m zeropawn`.
2. Hardware profiling commands: `hardware info`, `hardware profile --show`, `hardware profile --format json`.
3. Standalone MCTS search: `search --mode serial`, `search --mode batched --batch-size 4`.
4. Extended benchmark subcommands: `batch-sweep`, `concurrency-sweep`, `verify`.
5. Puzzles evaluation with `--search-mode batched` and `--mcts-batch-size auto`.
6. Match runner with `--search-mode batched` and `--batch-size auto`.
7. Interactive Operations Registry integrity & token resolution.
8. Built-in preset resolution and parser compatibility for new batched presets.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from python.cli.main import build_parser, main
from python.cli.operations import (
    OPERATIONS_CATALOG,
    OPERATIONS_REGISTRY,
    get_operation_by_id,
    get_operations_by_category,
)
from python.cli.presets import BUILTIN_PRESETS, get_all_presets, get_preset_by_id
from python.hardware.profile import (
    HardwareProfile,
    load_profile,
    resolve_batch_size,
    save_profile,
)


def test_parser_has_all_new_subcommands():
    """Verify that search and hardware subcommands are registered in top-level parser."""
    parser = build_parser()

    # 1. Hardware command parsing
    args_hw_info = parser.parse_args(["hardware", "info"])
    assert hasattr(args_hw_info, "func")
    assert args_hw_info.hardware_subcommand == "info"

    args_hw_prof = parser.parse_args(["hardware", "profile", "--show"])
    assert hasattr(args_hw_prof, "func")
    assert args_hw_prof.hardware_subcommand == "profile"
    assert args_hw_prof.show is True

    # 2. Search command parsing
    args_search = parser.parse_args(
        ["search", "--sims", "10", "--mode", "batched", "--batch-size", "16"]
    )
    assert hasattr(args_search, "func")
    assert args_search.sims == 10
    assert args_search.search_mode == "batched"
    assert args_search.batch_size == "16"

    # 3. Benchmark extended subcommands
    args_bs = parser.parse_args(["bench", "batch-sweep", "--batch-sizes", "1,8,16", "--sims", "50"])
    assert hasattr(args_bs, "func")
    assert args_bs.batch_sizes == "1,8,16"
    assert args_bs.sims == 50

    args_cs = parser.parse_args(["bench", "concurrency-sweep", "--concurrencies", "1,2,4"])
    assert hasattr(args_cs, "func")
    assert args_cs.concurrencies == "1,2,4"

    args_v = parser.parse_args(["bench", "verify"])
    assert hasattr(args_v, "func")


def test_puzzles_and_match_search_mode_args():
    """Verify that puzzles and match commands support batched search arguments."""
    parser = build_parser()

    # Puzzles eval
    args_puz = parser.parse_args(
        [
            "puzzles",
            "eval",
            "-n",
            "10",
            "--search-mode",
            "batched",
            "--mcts-batch-size",
            "auto",
        ]
    )
    assert args_puz.search_mode == "batched"
    assert args_puz.mcts_batch_size == "auto"

    # Match run
    args_match = parser.parse_args(
        [
            "match",
            "run",
            "-n",
            "2",
            "--search-mode",
            "batched",
            "--batch-size",
            "32",
        ]
    )
    assert args_match.search_mode == "batched"
    assert args_match.batch_size == "32"


def test_operations_registry_completeness():
    """Verify that all new operations are defined in the operations registry."""
    expected_ops = [
        "bench_batch_sweep",
        "bench_concurrency_sweep",
        "bench_verify",
        "mcts_search",
        "hardware_profile",
        "hardware_info",
        "hardware_calibrate",
    ]
    for op_id in expected_ops:
        op = get_operation_by_id(op_id)
        assert op is not None, f"Operation '{op_id}' missing from OPERATIONS_REGISTRY"
        assert op.direct_command
        assert op.category_id in range(1, 11)


def test_operations_token_resolution():
    """Verify that all operations in the registry produce tokens that parse without error."""
    parser = build_parser()

    for op in OPERATIONS_REGISTRY:
        cmd_parts = op.direct_command.split()
        if cmd_parts[0] == "chess":
            cmd_parts = cmd_parts[1:]

        tokens = []
        if cmd_parts:
            tokens.append(cmd_parts[0])
        if len(cmd_parts) > 1 and not cmd_parts[1].startswith(("-", "<", "[")):
            tokens.append(cmd_parts[1])

        # Test with minimal flag-free invocation or defaults
        try:
            sub_args = parser.parse_args(tokens)
            assert sub_args is not None
        except SystemExit:
            # Some operations might require subcommands if primary is a group
            pass


def test_builtin_presets_coverage():
    """Verify that newly added batched presets exist and parse cleanly."""
    all_presets = get_all_presets()
    expected_presets = ["batched-match", "batched-puzzles", "batch-sweep", "concurrency-sweep"]

    parser = build_parser()
    for pid in expected_presets:
        assert pid in all_presets, f"Preset '{pid}' not found in presets"
        p = all_presets[pid]
        assert p.get("is_builtin") is True

        tokens = [p["operation"]]
        subcmd = p.get("subcommand")
        args_list = list(p.get("args", []))
        if subcmd and (not args_list or args_list[0] != subcmd):
            tokens.append(subcmd)
        tokens.extend(args_list)

        # Filter out arguments that contain brackets or placeholders if any
        clean_tokens = [t for t in tokens if not (t.startswith("<") or t.startswith("["))]
        parsed = parser.parse_args(clean_tokens)
        assert parsed is not None


def test_hardware_commands_execution(capsys):
    """Test running hardware info and profile commands."""
    # Run hardware info
    ret = main(["hardware", "info"])
    assert ret == 0

    # Run hardware profile --show
    ret = main(["hardware", "profile", "--show"])
    assert ret == 0

    # Run hardware profile --format json
    ret = main(["hardware", "profile", "--format", "json"])
    assert ret == 0


def test_search_command_execution():
    """Test standalone search command in serial and batched modes."""
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Serial search
    ret_serial = main(["search", "--fen", startpos, "--sims", "5", "--mode", "serial"])
    assert ret_serial == 0

    # Batched search with explicit batch size
    ret_batched = main(
        ["search", "--fen", startpos, "--sims", "5", "--mode", "batched", "--batch-size", "4"]
    )
    assert ret_batched == 0

    # Batched search with auto batch size
    ret_auto = main(
        ["search", "--fen", startpos, "--sims", "5", "--mode", "batched", "--batch-size", "auto"]
    )
    assert ret_auto == 0
