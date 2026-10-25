"""Exhaustive test suite for ZeroPawn CLI argument completeness and combination coverage."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from python.cli.combinations import (
    ArgumentCombination,
    CombinationEngine,
    SafeParserValidator,
    evaluate_combination_constraints,
    get_combination_engine,
)
from python.cli.discovery import (
    ArgumentMetadata,
    CommandCatalog,
    CommandNode,
    get_command_catalog,
)
from python.cli.main import build_parser, handle_interactive_command, main
from python.cli.operations import CATEGORIES, OPERATIONS_REGISTRY


class TestCommandDiscoveryCompleteness:
    """Verify that every executable command in the ZeroPawn parser is discovered."""

    def test_catalog_discovers_all_executable_commands(self):
        catalog = get_command_catalog()
        commands = catalog.all_leaf_commands()
        assert len(commands) >= 110, (
            f"Expected at least 110 executable leaf commands, found {len(commands)}"
        )

        # Verify key baseline and advanced commands are all present
        expected_paths = [
            "train",
            "train connect4",
            "supervised",
            "finetune",
            "match",
            "match run",
            "match live",
            "match human",
            "eval baseline",
            "eval all-baseline",
            "eval heldout",
            "eval reliability",
            "bench selfplay",
            "bench batch-sizes",
            "bench pipelined",
            "bench dispatcher",
            "bench sweep-concurrency",
            "puzzles",
            "puzzles eval",
            "puzzles solve",
            "puzzles learn",
            "data download",
            "data filter",
            "data build",
            "data validate",
            "data check-leakage",
            "data merge",
            "data split",
            "models list",
            "models inspect",
            "models compare",
            "models validate",
            "models export",
            "models check-compat",
            "models download",
            "config show",
            "config get",
            "config set",
            "config validate",
            "config diff",
            "config export",
            "presets list",
            "presets show",
            "presets diff",
            "runs list",
            "runs inspect",
            "runs recover",
            "logs tail",
            "logs export",
            "diagnose env",
            "diagnose health",
            "diagnose lag",
            "tablebase download",
            "tablebase validate",
            "book build",
            "book evaluate",
            "build test",
            "build compile",
            "hardware profile",
            "hardware calibrate",
            "status",
            "uci",
            "cleanup",
            "search",
            "test",
        ]

        for p in expected_paths:
            node = catalog.get_command(p)
            assert node is not None, f"Expected command '{p}' not found in catalog!"
            assert node.is_leaf, f"Command '{p}' is not marked as executable leaf!"

    def test_all_10_categories_represented(self):
        catalog = get_command_catalog()
        for cat_id in range(1, 11):
            nodes = catalog.filter_by_category(cat_id)
            assert len(nodes) > 0, (
                f"Category {cat_id} ({CATEGORIES.get(cat_id)}) has 0 discovered commands!"
            )

    def test_command_aliases_and_classifications(self):
        catalog = get_command_catalog()

        # Test aliases
        t_node = catalog.get_command("test")
        assert t_node is not None
        assert "tests" in t_node.aliases or "t" in t_node.aliases

        train_node = catalog.get_command("train")
        assert train_node is not None
        assert "tr" in train_node.aliases or "selfplay" in train_node.aliases

        # Test classifications
        exp_node = catalog.get_command("diagnose lag")
        if exp_node:
            assert exp_node.classification == "experimental"

        adv_node = catalog.get_command("data check-leakage")
        if adv_node:
            assert adv_node.classification == "advanced"


class TestArgumentInventoryCompleteness:
    """Verify that all arguments, options, flags, and constraints are exposed."""

    def test_argument_counts_and_attributes(self):
        catalog = get_command_catalog()
        total_args = catalog.total_arguments_count()
        assert total_args >= 800, (
            f"Expected at least 800 total argument instances across commands, found {total_args}"
        )

        for node in catalog.all_leaf_commands():
            for arg in node.arguments:
                assert arg.name, f"Unnamed argument in {node.full_command}"
                assert arg.type in ("bool", "int", "float", "str", "choice", "path"), (
                    f"Invalid type {arg.type} in {node.full_command}"
                )
                if arg.is_flag_only:
                    assert arg.type == "bool" or arg.action_type in (
                        "store_true",
                        "store_false",
                        "count",
                    )

    def test_train_command_arguments_exhaustive(self):
        catalog = get_command_catalog()
        train_node = catalog.get_command("train")
        assert train_node is not None

        # Verify critical training options are discovered
        arg_names = {a.name for a in train_node.arguments}
        critical_flags = [
            "mode",
            "supervised",
            "checkpoint",
            "random_init",
            "resume",
            "max_hours",
            "concurrent_games",
            "buffer_capacity",
            "full_sims",
            "fast_sims",
            "batch_size",
            "lr",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "warmup_steps",
            "device",
            "train_device",
            "infer_device",
            "verify",
            "train_dir",
            "val_dir",
            "epochs",
            "tactics_ratio",
            "tactics_dir",
            "num_blocks",
            "num_filters",
        ]
        for f in critical_flags:
            assert f in arg_names, f"Expected flag '{f}' missing from train command inventory!"

    def test_finite_choices_domain_completeness(self):
        catalog = get_command_catalog()
        train_node = catalog.get_command("train")
        mode_arg = train_node.get_argument("mode")
        assert mode_arg is not None
        assert mode_arg.choices == ["rl", "supervised"]

        match_node = catalog.get_command("match live")
        if match_node:
            color_arg = match_node.get_argument("engine_color")
            assert color_arg is not None
            assert set(color_arg.choices) == {"white", "black"}

        test_node = catalog.get_command("test")
        if test_node:
            suite_arg = test_node.get_argument("suite")
            assert suite_arg is not None
            assert {
                "fast",
                "all",
                "cpp",
                "mcts",
                "parity",
                "guardrails",
            }.issubset(set(suite_arg.choices))


class TestConstraintEngine:
    """Verify constraint derivation and semantic rule evaluation."""

    def test_mutually_exclusive_random_init_and_checkpoint(self):
        catalog = get_command_catalog()
        train_node = catalog.get_command("train")
        assert train_node is not None

        # Conflicting state
        is_valid, reason = evaluate_combination_constraints(
            train_node,
            parameters={"random_init": True, "checkpoint": "best_model.pt"},
            explicit_flags={"random_init", "checkpoint"},
        )
        assert not is_valid
        assert "--random-init" in reason and "--checkpoint" in reason

        # Valid states
        valid_a, _ = evaluate_combination_constraints(
            train_node,
            parameters={"random_init": True},
            explicit_flags={"random_init"},
        )
        assert valid_a

        valid_b, _ = evaluate_combination_constraints(
            train_node,
            parameters={"checkpoint": "best_model.pt"},
            explicit_flags={"checkpoint"},
        )
        assert valid_b

    def test_dependency_tactics_ratio_requires_tactics_dir(self):
        catalog = get_command_catalog()
        train_node = catalog.get_command("train")
        assert train_node is not None

        # Invalid: ratio specified without dir
        is_valid, reason = evaluate_combination_constraints(
            train_node,
            parameters={"tactics_ratio": 0.2, "tactics_dir": None},
            explicit_flags={"tactics_ratio"},
        )
        assert not is_valid
        assert "--tactics-ratio requires --tactics-dir" in reason

        # Valid: both specified
        is_valid, _ = evaluate_combination_constraints(
            train_node,
            parameters={"tactics_ratio": 0.2, "tactics_dir": "data/puzzles"},
            explicit_flags={"tactics_ratio", "tactics_dir"},
        )
        assert is_valid


class TestCombinationEnumeration:
    """Verify combination generation, determinism, pagination, and parser validation."""

    def test_combination_space_counting(self):
        catalog = get_command_catalog()
        engine = get_combination_engine()

        status_node = catalog.get_command("status")
        total, valid, invalid = engine.get_command_combination_space_size(status_node)
        assert total == 1
        assert valid == 1
        assert invalid == 0

        eval_node = catalog.get_command("eval baseline")
        total_e, valid_e, invalid_e = engine.get_command_combination_space_size(eval_node)
        assert total_e >= 100
        assert valid_e >= 100

    def test_deterministic_enumeration_uniqueness(self):
        catalog = get_command_catalog()
        engine = get_combination_engine()

        for test_path in ("eval baseline", "bench selfplay", "models list", "data filter"):
            node = catalog.get_command(test_path)
            assert node is not None
            combos = list(engine.enumerate_combinations(node, max_limit=100, include_invalid=False))
            assert len(combos) > 0, f"No combinations generated for {test_path}"

            # Check uniqueness of generated command lines
            cmd_lines = [c.to_command_line() for c in combos]
            assert len(cmd_lines) == len(set(cmd_lines)), (
                f"Duplicate combinations detected in {test_path}!"
            )

    def test_parser_acceptance_of_generated_combinations(self):
        """Verify that generated valid combinations parse cleanly with build_parser()."""
        catalog = get_command_catalog()
        engine = get_combination_engine()

        sampled_commands = [
            "status",
            "eval all-baseline",
            "eval baseline",
            "bench pipelined",
            "bench dispatcher",
            "data list",
            "models list",
            "presets list",
            "diagnose env",
            "diagnose health",
            "cleanup list",
            "tablebase feasibility",
        ]

        for p in sampled_commands:
            node = catalog.get_command(p)
            assert node is not None
            combos = list(engine.enumerate_combinations(node, max_limit=10, include_invalid=False))
            for c in combos:
                ok, err = SafeParserValidator.validate_tokens(c.to_argv())
                assert ok, (
                    f"Parser rejected valid combination for {p}: tokens={c.to_argv()} error={err}"
                )

    def test_paginated_combinations_browsing(self):
        catalog = get_command_catalog()
        engine = get_combination_engine()
        eval_node = catalog.get_command("eval baseline")
        assert eval_node is not None

        page1, total1, pages1 = engine.get_paginated_combinations(eval_node, page=1, page_size=10)
        page2, total2, pages2 = engine.get_paginated_combinations(eval_node, page=2, page_size=10)

        assert total1 == total2
        assert pages1 == pages2
        assert len(page1) == 10
        assert len(page2) == 10
        # Disjoint indices
        indices_p1 = {c.index for c in page1}
        indices_p2 = {c.index for c in page2}
        assert not (indices_p1 & indices_p2), "Overlap detected between page 1 and page 2!"


class TestInteractiveAndEntryPoints:
    """Verify interactive CLI options, argument builder previews, and entrypoint routing."""

    def test_interactive_subcommand_flags(self):
        parser = build_parser()

        # Test --options flag
        args_opt = parser.parse_args(
            ["interactive", "--options", "data filter", "--non-interactive"]
        )
        assert args_opt.options == "data filter"
        with patch("rich.console.Console.print"):
            code = args_opt.func(args_opt)
            assert code == 0

        # Test --combinations flag
        args_combo = parser.parse_args(
            [
                "interactive",
                "--combinations",
                "eval baseline",
                "--non-interactive",
                "--page",
                "1",
                "--page-size",
                "5",
            ]
        )
        assert args_combo.combinations == "eval baseline"
        with patch("rich.console.Console.print"):
            code = args_combo.func(args_combo)
            assert code == 0

    def test_argument_combination_command_line_rendering(self):
        catalog = get_command_catalog()
        engine = get_combination_engine()
        node = catalog.get_command("eval baseline")
        assert node is not None

        combos = list(engine.enumerate_combinations(node, max_limit=3))
        for c in combos:
            cmd_line = c.to_command_line()
            assert cmd_line.startswith("zeropawn eval baseline")
            # Ensure proper shlex splitting
            import shlex

            tokens = shlex.split(cmd_line)
            assert tokens[0] == "zeropawn"
            assert tokens[1:3] == ["eval", "baseline"]
