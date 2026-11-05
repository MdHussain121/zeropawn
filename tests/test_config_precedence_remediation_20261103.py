"""Post-merge production remediation regression tests (2026-11-03).

Covers the config/CLI lane findings:

* S5-01 / S8-05 — explicit CLI overrides must ALWAYS win over the env,
  preset and config-file tiers, for every command (the field's
  applicable_workflows list must never demote an explicitly-passed flag).
  Full precedence matrix per field, plus propagation-level tests through
  the real ``main()`` chain.
* S5-08 (CFG-01/CFG-02) — invalid env/config-file/preset values warn loudly
  (stderr) naming the field and fall back to the field default; a malformed
  config file is reported instead of silently ignored; an explicitly-passed
  missing --config is reported.  Never fatal.
* S3-03 / S5-09 — the seven dead CONFIG_REGISTRY fields are removed; the
  module documents the 5-tier engine; eval.games' runtime_dest is truthful.
* S5-03 — wizard-saved presets store real CLI command tokens; saving an
  unlaunchable preset is refused; all built-in presets parse through the
  shared launch-token builder; the stale ``run1-512`` user preset is gone.
* S5-04 — interactive model injection targets the operation's ACTUAL model
  flag (``--checkpoint-path`` for bench selfplay) and tells the user when a
  selection cannot apply.
* S3-04/S3-05 — the bootstrap build-dir loop iterates reversed; both
  checkpoint scanners share one directory list and agree on a tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import python.cli.common as cli_common
from python.cli.common import get_available_checkpoints
from python.cli.combinations import POSITIONAL_MODEL, find_op_model_target
from python.cli.config import (
    CONFIG_REGISTRY,
    load_config_file,
    resolve_configuration,
    set_config_option,
)
from python.cli.main import build_parser, execute_operation_from_interactive, main
from python.cli.models import discover_checkpoints
from python.cli.operations import get_operation_by_id
from python.cli.presets import (
    BUILTIN_PRESETS,
    get_all_presets,
    get_preset_by_id,
    preset_launch_tokens,
    save_preset,
    split_preset_tokens,
    validate_preset_launchable,
)
from python.cli.wizard import build_preset_tokens


# ────────────────────────────── helpers ──────────────────────────────


def _resolve(argv, tmp_path, monkeypatch=None, env=None, preset_dict=None):
    """Parse argv with the real parser and resolve with an isolated config file."""
    if env and monkeypatch is not None:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    parser = build_parser()
    args = parser.parse_args(argv)
    return (
        resolve_configuration(
            command_name=getattr(args, "command", ""),
            cli_args=args,
            preset_dict=preset_dict,
            config_path=tmp_path / "resolve_config.json",
            raw_argv=list(argv),
        ),
        args,
    )


def _propagate(resolved, args):
    """Run main.py's REAL propagation loop (SA1-01/SA1-02 remediation moved
    it out of main() into config.propagate_resolved_to_args; this helper now
    delegates so the tests exercise the production code, not a stale copy)."""
    from python.cli.config import propagate_resolved_to_args

    propagate_resolved_to_args(resolved, args, parser=build_parser())
    return args


def _all_parser_flags() -> set:
    parser = build_parser()
    flags: set = set()

    def walk(p):
        for action in p._actions:
            flags.update(action.option_strings)
            if isinstance(action.choices, dict):
                for sub in action.choices.values():
                    walk(sub)

    walk(parser)
    return flags


# ─────────────────────── S5-01 / S8-05: precedence matrix ───────────────────────


class TestPrecedenceMatrix:
    """Per-field precedence matrix over the 5-tier engine."""

    def test_explicit_cli_only(self, tmp_path):
        resolved, _ = _resolve(["supervised", "--batch-size", "128"], tmp_path)
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "cli"
        assert rec.effective_value == 128

    def test_env_only(self, tmp_path, monkeypatch):
        resolved, _ = _resolve(
            ["supervised"], tmp_path, monkeypatch, env={"ZEROPAWN_BATCH_SIZE": "64"}
        )
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "env"
        assert rec.effective_value == 64

    def test_config_file_only(self, tmp_path):
        cfg = tmp_path / "c.json"
        set_config_option("training.batch_size", 256, config_path=cfg)
        parser = build_parser()
        argv = ["supervised"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="supervised", cli_args=args, config_path=cfg, raw_argv=argv
        )
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "config_file"
        assert rec.effective_value == 256

    def test_default_only(self, tmp_path):
        resolved, _ = _resolve(["supervised"], tmp_path)
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "default"
        assert rec.effective_value == 512

    def test_cli_beats_env(self, tmp_path, monkeypatch):
        resolved, _ = _resolve(
            ["supervised", "--batch-size", "128"],
            tmp_path,
            monkeypatch,
            env={"ZEROPAWN_BATCH_SIZE": "64"},
        )
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "cli"
        assert rec.effective_value == 128

    def test_cli_beats_config_file(self, tmp_path):
        cfg = tmp_path / "c.json"
        set_config_option("training.batch_size", 256, config_path=cfg)
        parser = build_parser()
        argv = ["supervised", "--batch-size", "1024"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="supervised", cli_args=args, config_path=cfg, raw_argv=argv
        )
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "cli"
        assert rec.effective_value == 1024

    def test_env_beats_config_file(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.json"
        set_config_option("training.batch_size", 256, config_path=cfg)
        monkeypatch.setenv("ZEROPAWN_BATCH_SIZE", "64")
        parser = build_parser()
        argv = ["supervised"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="supervised", cli_args=args, config_path=cfg, raw_argv=argv
        )
        rec = resolved.get_record("training.batch_size")
        assert rec.source == "env"
        assert rec.effective_value == 64

    def test_multiple_applicable_workflows(self, tmp_path, monkeypatch):
        """system.device is applicable to train/supervised/match/uci alike —
        the explicit CLI tier must hold on each of them."""
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "cuda")
        for cmd in ("train", "supervised", "match", "uci"):
            resolved, args = _resolve([cmd, "--device", "cpu"], tmp_path)
            rec = resolved.get_record("system.device")
            assert rec.source == "cli", cmd
            assert rec.effective_value == "cpu", cmd
            assert args.device == "cpu", cmd

    def test_non_applicable_workflow_device_s5_01(self, tmp_path, monkeypatch):
        """THE S5-01 case: `mcts` (alias of search) is NOT in system.device's
        applicable_workflows, yet an explicit --device cpu must beat
        ZEROPAWN_SYSTEM_DEVICE=cuda."""
        assert "search" not in CONFIG_REGISTRY["system.device"].applicable_workflows
        resolved, args = _resolve(
            ["mcts", "--device", "cpu"],
            tmp_path,
            monkeypatch,
            env={"ZEROPAWN_SYSTEM_DEVICE": "cuda"},
        )
        rec = resolved.get_record("system.device")
        assert rec.source == "cli"
        assert rec.effective_value == "cpu"
        assert args.device == "cpu"

    def test_non_applicable_workflow_sims(self, tmp_path, monkeypatch):
        """eval.mcts_sims (--sims) on the mcts command with an env value set:
        the explicit CLI sims must win."""
        assert "search" not in CONFIG_REGISTRY["eval.mcts_sims"].applicable_workflows
        resolved, args = _resolve(
            ["mcts", "--sims", "500"],
            tmp_path,
            monkeypatch,
            env={"ZEROPAWN_EVAL_MCTS_SIMS": "42"},
        )
        rec = resolved.get_record("eval.mcts_sims")
        assert rec.source == "cli"
        assert rec.effective_value == 500
        assert args.sims == 500

    def test_non_applicable_workflow_games_flag(self, tmp_path, monkeypatch):
        """eval.games (--games/-n) on the `test` command with an env value:
        the explicit CLI value must win."""
        assert "test" not in CONFIG_REGISTRY["eval.games"].applicable_workflows
        resolved, args = _resolve(
            ["test", "--games", "50"],
            tmp_path,
            monkeypatch,
            env={"ZEROPAWN_EVAL_GAMES": "999"},
        )
        rec = resolved.get_record("eval.games")
        assert rec.source == "cli"
        assert rec.effective_value == 50
        assert args.games == 50

    def test_cli_beats_preset_for_non_applicable_workflow(self, tmp_path):
        """Even a preset value must not overwrite an explicit CLI flag on a
        non-applicable workflow."""
        preset = {"parameters": {"device": "cuda"}}
        parser = build_parser()
        argv = ["mcts", "--device", "cpu"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="mcts",
            cli_args=args,
            preset_dict=preset,
            config_path=tmp_path / "c.json",
            raw_argv=list(argv),
        )
        rec = resolved.get_record("system.device")
        assert rec.source == "cli"
        assert rec.effective_value == "cpu"

    def test_cli_flag_equals_form(self, tmp_path, monkeypatch):
        """`--flag=value` tokens are detected as explicit CLI passes."""
        resolved, _ = _resolve(
            ["mcts", "--device=cpu"],
            tmp_path,
            monkeypatch,
            env={"ZEROPAWN_SYSTEM_DEVICE": "cuda"},
        )
        rec = resolved.get_record("system.device")
        assert rec.source == "cli"
        assert rec.effective_value == "cpu"

    def test_unspecified_flag_still_allows_env(self, tmp_path, monkeypatch):
        """Without an explicit flag the env tier still applies (unchanged)."""
        resolved, _ = _resolve(
            ["mcts"], tmp_path, monkeypatch, env={"ZEROPAWN_SYSTEM_DEVICE": "cuda"}
        )
        rec = resolved.get_record("system.device")
        assert rec.source == "env"
        assert rec.effective_value == "cuda"

    def test_numeric_override(self, tmp_path):
        resolved, _ = _resolve(["match", "run", "--mcts-sims", "77"], tmp_path)
        rec = resolved.get_record("eval.mcts_sims")
        assert rec.source == "cli"
        assert rec.effective_value == 77

    def test_string_override(self, tmp_path):
        """mcts.batch_size is a str-typed field ('auto' or integer string)."""
        resolved, _ = _resolve(["mcts", "--batch-size", "24"], tmp_path)
        rec = resolved.get_record("mcts.batch_size")
        assert rec.source == "cli"
        assert rec.effective_value == "24"

    def test_boolean_override_true(self, tmp_path):
        """mcts.reuse_tree is a bool field; --reuse-tree exists on the uci parser."""
        resolved, args = _resolve(["uci", "--reuse-tree"], tmp_path)
        rec = resolved.get_record("mcts.reuse_tree")
        assert rec.source == "cli"
        assert rec.effective_value is True
        assert args.reuse_tree is True

    def test_boolean_explicit_false(self, tmp_path, monkeypatch):
        """`--no-book` is an explicit boolean-family CLI pass and must be
        attributed to the eval.use_book record (source=cli), beating a truthy
        env value; the record's effective value is book-disabled (False) and
        the parsed arg survives.  NOTE (pre-existing quirk, unchanged here):
        the match parser encodes the negation as dest `no_book` (store_true)
        and ALSO carries a `--book <path>` flag whose dest `book` collides
        with this field's flag list — the handler derives use_book from
        args.no_book, and source=cli propagation correctly leaves it alone."""
        resolved, args = _resolve(
            ["match", "run", "--no-book"],
            tmp_path,
            monkeypatch,
            env={"ZEROPAWN_EVAL_USE_BOOK": "true"},
        )
        rec = resolved.get_record("eval.use_book")
        assert rec.source == "cli"
        assert rec.effective_value is False  # book usage disabled
        assert args.no_book is True  # explicit CLI pass survived the env tier

    def test_boolean_env_only(self, tmp_path, monkeypatch):
        resolved, _ = _resolve(
            ["uci"], tmp_path, monkeypatch, env={"ZEROPAWN_MCTS_REUSE_TREE": "true"}
        )
        rec = resolved.get_record("mcts.reuse_tree")
        assert rec.source == "env"
        assert rec.effective_value is True


class TestPropagationThroughMain:
    """Propagation-level proof: the REAL main() parse→resolve→propagate→dispatch
    chain delivers the right value to the command handler."""

    def test_mcts_device_cpu_with_cuda_env(self, monkeypatch):
        """S5-01 end-to-end: with ZEROPAWN_SYSTEM_DEVICE=cuda, the `mcts`
        handler must receive device == 'cpu' (the explicit CLI value)."""
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "cuda")
        seen = {}

        def fake_run_search(args):
            seen["device"] = getattr(args, "device", None)
            return 0

        import python.cli.commands_search as commands_search

        monkeypatch.setattr(commands_search, "run_search", fake_run_search)
        rc = main(["mcts", "--device", "cpu", "--sims", "4", "--search-mode", "serial"])
        assert rc == 0
        assert seen["device"] == "cpu", (
            f"handler received device={seen.get('device')!r} — explicit CLI "
            f"override was demoted below the env tier (S5-01 regression)"
        )

    def test_train_device_cpu_control(self, monkeypatch):
        """Control (applicable workflow): `train` resolves the CLI device."""
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "cuda")
        seen = {}

        def fake_run_training(args):
            seen["device"] = getattr(args, "device", None)
            return 0

        import python.cli.commands_train as commands_train

        monkeypatch.setattr(commands_train, "run_training", fake_run_training)
        rc = main(["train", "--device", "cpu"])
        assert rc == 0
        assert seen["device"] == "cpu"

    def test_env_still_propagates_without_explicit_flag(self, monkeypatch):
        """Without an explicit flag, env values still propagate to args."""
        monkeypatch.delenv("ZEROPAWN_EVAL_MCTS_SIMS", raising=False)
        monkeypatch.setenv("ZEROPAWN_EVAL_MCTS_SIMS", "42")
        seen = {}

        def fake_run_search(args):
            seen["sims"] = getattr(args, "sims", None)
            return 0

        import python.cli.commands_search as commands_search

        monkeypatch.setattr(commands_search, "run_search", fake_run_search)
        rc = main(["mcts", "--search-mode", "serial"])
        assert rc == 0
        assert seen["sims"] == 42

    def test_propagation_replicated_inline(self, tmp_path, monkeypatch):
        """Inline replication of main.py's propagation loop (no dispatch):
        the explicit CLI value survives; the env value would have overwritten
        it under the demoted record."""
        monkeypatch.setenv("ZEROPAWN_EVAL_GAMES", "999")
        parser = build_parser()
        argv = ["test", "--games", "50"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="test", cli_args=args, config_path=tmp_path / "c.json", raw_argv=argv
        )
        assert resolved.get_record("eval.games").source == "cli"
        args = _propagate(resolved, args)
        assert args.games == 50


# ─────────────────────── S5-08 (CFG-01/CFG-02): validation gaps ───────────────────────


class TestCoercionValidation:
    def test_invalid_env_int_warns_and_falls_back(self, tmp_path, monkeypatch, capfd):
        monkeypatch.setenv("ZEROPAWN_EVAL_GAMES", "not_a_number")
        parser = build_parser()
        argv = ["match", "run"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="match", cli_args=args, config_path=tmp_path / "c.json", raw_argv=argv
        )
        rec = resolved.get_record("eval.games")
        assert rec.effective_value == CONFIG_REGISTRY["eval.games"].default
        err = capfd.readouterr().err
        assert "eval.games" in err
        assert "not_a_number" in err

    def test_invalid_env_choice_warns_and_falls_back(self, tmp_path, monkeypatch, capfd):
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "banana")
        resolved, _ = _resolve(["mcts"], tmp_path, monkeypatch)
        rec = resolved.get_record("system.device")
        assert rec.effective_value == "auto"
        assert "system.device" in capfd.readouterr().err

    def test_invalid_env_bool_warns_and_falls_back(self, tmp_path, monkeypatch, capfd):
        monkeypatch.setenv("ZEROPAWN_SELFPLAY_PIPELINED", "banana")
        resolved, _ = _resolve(["train"], tmp_path, monkeypatch)
        rec = resolved.get_record("selfplay.pipelined")
        assert rec.effective_value is False
        assert "selfplay.pipelined" in capfd.readouterr().err

    def test_invalid_config_file_value_warns_and_falls_back(self, tmp_path, capfd):
        cfg = tmp_path / "bad_value.json"
        cfg.write_text(json.dumps({"eval.games": "not_a_number"}), encoding="utf-8")
        parser = build_parser()
        argv = ["match", "run"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="match", cli_args=args, config_path=cfg, raw_argv=argv
        )
        rec = resolved.get_record("eval.games")
        assert rec.effective_value == CONFIG_REGISTRY["eval.games"].default
        assert "eval.games" in capfd.readouterr().err

    def test_malformed_config_file_warns_not_fatal(self, tmp_path, capfd):
        cfg = tmp_path / "garbage.json"
        cfg.write_text("{not valid json!!", encoding="utf-8")
        settings = load_config_file(cfg)
        assert settings == {}
        err = capfd.readouterr().err
        assert "could not be read" in err
        # Resolution continues with defaults (non-fatal)
        parser = build_parser()
        argv = ["match", "run"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="match", cli_args=args, config_path=cfg, raw_argv=argv
        )
        assert resolved.get_record("eval.games").source == "default"

    def test_valid_config_file_stays_silent(self, tmp_path, capfd):
        cfg = tmp_path / "ok.json"
        set_config_option("training.batch_size", 128, config_path=cfg)
        load_config_file(cfg)
        assert capfd.readouterr().err == ""

    def test_missing_default_config_file_stays_silent(self, tmp_path, capfd):
        load_config_file(tmp_path / "does_not_exist.json")
        assert capfd.readouterr().err == ""

    def test_missing_explicit_config_reported_by_main(self, tmp_path, monkeypatch, capfd):
        """An explicitly-passed --config that does not exist is loudly reported."""
        missing = tmp_path / "nope" / "missing.json"
        seen = {}

        def fake_run_status(args):
            seen["called"] = True
            return 0

        import python.cli.commands_status as commands_status

        monkeypatch.setattr(commands_status, "run_status", fake_run_status)
        rc = main(["--config", str(missing), "status"])
        assert rc == 0
        assert seen.get("called") is True
        err = capfd.readouterr().err
        assert "Configuration file not found" in err
        assert str(missing) in err

    def test_invalid_preset_value_warns_and_falls_back(self, tmp_path, monkeypatch, capfd):
        preset = {"parameters": {"games": "not_a_number"}}
        parser = build_parser()
        argv = ["match", "run"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="match",
            cli_args=args,
            preset_dict=preset,
            config_path=tmp_path / "c.json",
            raw_argv=argv,
        )
        rec = resolved.get_record("eval.games")
        assert rec.effective_value == CONFIG_REGISTRY["eval.games"].default
        assert "eval.games" in capfd.readouterr().err


# ─────────────────────── S3-03 / S5-09: dead fields & docs ───────────────────────


class TestRegistryDeadFieldRemoval:
    REMOVED = [
        "system.precision",
        "system.channels_last",
        "mcts.dirichlet_alpha",
        "mcts.dirichlet_weight",
        "mcts.virtual_loss",
        "mcts.fpu_reduction",
        "training.push_freq",
    ]

    def test_dead_fields_removed(self):
        for key in self.REMOVED:
            assert key not in CONFIG_REGISTRY, f"dead field {key} still declared"

    def test_docstring_says_5_tier(self):
        import python.cli.config as config_module

        assert "5-Tier" in config_module.__doc__
        assert "4-Tier" not in config_module.__doc__

    def test_eval_games_runtime_dest_truthful(self):
        dest = CONFIG_REGISTRY["eval.games"].runtime_dest
        assert "MatchRunner.num_games" not in dest
        assert "run_match(num_openings=" in dest

    def test_registry_integrity_unchanged_for_live_fields(self):
        assert len(CONFIG_REGISTRY) == 49
        for key, cf in CONFIG_REGISTRY.items():
            assert cf.key == key
            assert cf.alias
            assert cf.type_name in ("int", "float", "bool", "str", "choice", "path")
            assert cf.description
            assert cf.runtime_dest

    def test_removed_field_flags_absent_from_parsers(self):
        """None of the removed fields' flags exist on any parser (they never
        did — which is exactly why the fields were dead)."""
        flags = _all_parser_flags()
        for flag in (
            "--precision",
            "--channels-last",
            "--no-channels-last",
            "--dirichlet-alpha",
            "--dirichlet-weight",
            "--virtual-loss",
            "--fpu-reduction",
            "--push-freq",
        ):
            assert flag not in flags, f"parser defines {flag} though field was removed"


# ─────────────────────── S5-03: wizard presets launchable ───────────────────────


class TestWizardPresetsLaunchable:
    @pytest.fixture()
    def presets_file(self, tmp_path, monkeypatch):
        import python.cli.presets as pr_module

        fake = tmp_path / "configs" / "presets.json"
        monkeypatch.setattr(pr_module, "PRESETS_FILE", fake)
        return fake

    def test_all_16_builtin_presets_parse(self):
        assert len(BUILTIN_PRESETS) == 16
        for pid, p in BUILTIN_PRESETS.items():
            ok, err = validate_preset_launchable(p)
            assert ok, f"builtin preset {pid} does not parse: {err}"

    def test_preset_launch_tokens_dedupes_subcommand(self):
        """Built-in presets carry a legacy args shape repeating the subcommand;
        the shared builder normalizes it (previously `presets run` emitted
        e.g. `match run run -n 10` and failed argparse for 10 of 16)."""
        assert preset_launch_tokens(BUILTIN_PRESETS["match-short"]) == [
            "match",
            "run",
            "-n",
            "10",
            "--mcts-sims",
            "30",
        ]

    def test_wizard_save_produces_parseable_preset(self, presets_file):
        """The wizard save path (build_preset_tokens + split_preset_tokens +
        save_preset) yields a preset that the launch token builder reproduces
        verbatim."""
        op = get_operation_by_id("train_verify")  # direct_command: zeropawn train --verify
        params = {a.name: a.default for a in op.arguments}
        tokens, note = build_preset_tokens(op, params, None)
        assert tokens == ["train", "--verify"]
        assert note is None

        operation, subcommand, args = split_preset_tokens(tokens)
        ok = save_preset(
            preset_id="wizard-made",
            name="Wizard Made",
            description="from the wizard path",
            operation=operation,
            args=args,
            parameters=params,
            subcommand=subcommand,
        )
        assert ok is True
        p = get_preset_by_id("wizard-made")
        assert p is not None
        assert p["operation"] == "train", "wizard preset must store a CLI command token"
        assert validate_preset_launchable(p)[0] is True
        assert preset_launch_tokens(p) == tokens

    def test_wizard_save_with_changed_params_round_trips(self, presets_file):
        """direct_command placeholder flags (e.g. "--epochs <N>") must not
        duplicate the parameter flags, and the split must round-trip."""
        op = get_operation_by_id("train_supervised")  # zeropawn supervised --epochs <N>
        params = {a.name: a.default for a in op.arguments}
        params["epochs"] = 7
        tokens, note = build_preset_tokens(op, params, None)
        assert tokens[0] == "supervised"
        assert tokens.count("--epochs") == 1, tokens
        assert "7" in tokens
        operation, subcommand, args = split_preset_tokens(tokens)
        ok = save_preset(
            preset_id="wizard-epochs",
            name="Wizard Epochs",
            description="7 epochs",
            operation=operation,
            args=args,
            parameters=params,
            subcommand=subcommand,
        )
        assert ok is True
        p = get_preset_by_id("wizard-epochs")
        assert p is not None
        assert preset_launch_tokens(p) == tokens
        assert validate_preset_launchable(p)[0] is True

    def test_wizard_save_with_model_uses_real_model_flag(self, presets_file):
        """A bench selfplay wizard save must inject --checkpoint-path (the
        op's actual model flag), not --checkpoint."""
        op = get_operation_by_id("bench_selfplay")
        params = {a.name: a.default for a in op.arguments}
        model = Path("models/checkpoints_scaled/latest_checkpoint.pt")
        tokens, note = build_preset_tokens(op, params, model)
        assert note is None
        assert "--checkpoint-path" in tokens
        assert "--checkpoint" not in tokens
        assert str(model) in tokens
        operation, subcommand, args = split_preset_tokens(tokens)
        ok = save_preset(
            preset_id="wizard-bench",
            name="Wizard Bench",
            description="bench selfplay with model",
            operation=operation,
            args=args,
            parameters=params,
            model=str(model),
            subcommand=subcommand,
        )
        assert ok is True
        p = get_preset_by_id("wizard-bench")
        assert p is not None
        assert validate_preset_launchable(p)[0] is True
        assert preset_launch_tokens(p) == tokens

    def test_wizard_save_without_model_flag_notes_user(self):
        """test_guardrails has requires_model=required but NO model flag — the
        selection cannot apply and the builder says so."""
        op = get_operation_by_id("test_guardrails")
        params = {a.name: a.default for a in op.arguments}
        tokens, note = build_preset_tokens(
            op, params, Path("models/checkpoints_scaled/latest_checkpoint.pt")
        )
        assert note is not None
        assert "--checkpoint" not in tokens
        assert "--checkpoint-path" not in tokens

    def test_unlaunchable_preset_refused(self, presets_file, capsys):
        """operation=op.id (not a CLI command) must be refused with a clear
        error and NOT written."""
        ok = save_preset(
            preset_id="legacy-bad",
            name="Legacy Bad",
            description="stores op.id like the old wizard did",
            operation="train_supervised",
            args=["--epochs", "4"],
        )
        assert ok is False
        assert "not launchable" in capsys.readouterr().out
        assert get_preset_by_id("legacy-bad") is None

    def test_unknown_flag_preset_refused(self, presets_file):
        ok = save_preset(
            preset_id="bad-flag",
            name="Bad Flag",
            description="unknown flag",
            operation="match",
            subcommand="run",
            args=["--definitely-not-a-flag", "1"],
        )
        assert ok is False
        assert get_preset_by_id("bad-flag") is None

    def test_stale_run1_512_removed(self):
        """The audit's stale persisted artifact must be gone from the tracked
        presets file."""
        assert "run1-512" not in get_all_presets()
        import python.cli.presets as pr_module

        assert pr_module.PRESETS_FILE.is_file()
        data = json.loads(pr_module.PRESETS_FILE.read_text(encoding="utf-8"))
        assert "run1-512" not in data.get("presets", {})

    def test_duplicate_of_builtin_still_launchable(self, presets_file):
        from python.cli.presets import duplicate_preset

        assert duplicate_preset("match-short", "match-short-copy", "Copy") is True
        p = get_preset_by_id("match-short-copy")
        assert p is not None
        assert validate_preset_launchable(p)[0] is True


# ─────────────────────── S5-04: interactive model injection ───────────────────────


class TestInteractiveModelInjection:
    MODEL = Path("models/checkpoints_scaled/best_model.pt")

    def _capture_tokens(self, op_id):
        op = get_operation_by_id(op_id)
        captured = {}

        def dummy_func(args):
            return 0

        real_parser = build_parser()
        orig_parse_args = real_parser.parse_args

        def hooked(tokens):
            captured["tokens"] = list(tokens)
            args = orig_parse_args(tokens)
            args.func = dummy_func
            return args

        # execute_operation_from_interactive's finally block calls
        # torch.cuda.empty_cache()/ipc_collect() which initialize the CUDA
        # context on machines with a GPU — keep the test CPU-pure.
        with (
            patch("python.cli.main.build_parser", return_value=real_parser),
            patch.object(real_parser, "parse_args", side_effect=hooked),
            patch("python.cli.ui.console.print"),
            patch("rich.console.Console.print"),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            code = execute_operation_from_interactive(
                op, {a.name: a.default for a in op.arguments}, self.MODEL
            )
        return code, captured.get("tokens", [])

    def test_bench_selfplay_injects_checkpoint_path(self):
        """THE S5-04 case: bench selfplay's model flag is --checkpoint-path."""
        code, tokens = self._capture_tokens("bench_selfplay")
        assert code == 0
        assert "--checkpoint-path" in tokens
        assert str(self.MODEL) in tokens
        assert "--checkpoint" not in tokens

    def test_checkpoint_flag_ops_unchanged(self):
        code, tokens = self._capture_tokens("train_rl")
        assert code == 0
        assert "--checkpoint" in tokens
        assert str(self.MODEL) in tokens

    def test_positional_model_op_unchanged(self):
        code, tokens = self._capture_tokens("models_validate")
        assert code == 0
        assert str(self.MODEL) in tokens
        assert "--checkpoint" not in tokens

    def test_resume_model_op_unchanged(self):
        code, tokens = self._capture_tokens("train_finetune")
        assert code == 0
        assert "--resume" in tokens
        assert str(self.MODEL) in tokens

    def test_no_model_flag_op_informs_and_skips(self):
        """test_guardrails has no model flag: nothing is injected (the run
        keeps its own default resolution) and the user is informed."""
        code, tokens = self._capture_tokens("test_guardrails")
        assert code == 0
        assert str(self.MODEL) not in tokens
        assert "--checkpoint" not in tokens
        assert "--checkpoint-path" not in tokens

    def test_find_op_model_target_priorities(self):
        expectations = {
            "bench_selfplay": "--checkpoint-path",
            "bench_pipelined": None,
            "test_soak": None,
            "test_guardrails": None,
            "train_rl": "--checkpoint",
            "train_finetune": "--resume",
            "models_validate": POSITIONAL_MODEL,
            "match_run": "--checkpoint",
        }
        for op_id, expected in expectations.items():
            assert find_op_model_target(get_operation_by_id(op_id)) == expected, op_id


# ─────────────────────── S3-04/S3-05: bootstrap + scanners ───────────────────────


class TestBootstrapAndScanners:
    def test_setup_environment_priority_build_cpp_first(self):
        """S3-04: after the (reversed) bootstrap loop, the deployed build/cpp
        must rank above the stale build/cmake/cpp."""
        candidates = [
            cli_common.REPO_ROOT / "build" / "cpp",
            cli_common.REPO_ROOT / "build",
            cli_common.REPO_ROOT / "build" / "Release",
            cli_common.REPO_ROOT / "build" / "Debug",
            cli_common.REPO_ROOT / "build" / "branch" / "cpp",
            cli_common.REPO_ROOT / "build" / "cmake" / "cpp",
        ]
        existing = [str(c) for c in candidates if c.is_dir() and str(c) in sys.path]
        for s in existing:
            sys.path.remove(s)
        try:
            cli_common.setup_environment()
        finally:
            # setup_environment re-inserted them in the fixed order; restore
            # any it could not (defensive only).
            for s in existing:
                if s not in sys.path:
                    sys.path.insert(0, s)
        build_cpp = str(cli_common.REPO_ROOT / "build" / "cpp")
        cmake_cpp = str(cli_common.REPO_ROOT / "build" / "cmake" / "cpp")
        if build_cpp in sys.path and cmake_cpp in sys.path:
            assert sys.path.index(build_cpp) < sys.path.index(cmake_cpp)

    def test_both_scanners_share_directory_constant(self, monkeypatch, tmp_path):
        """S3-05: with the shared directory list pointed at a synthetic tree,
        both scanners must report exactly the same set of checkpoint files."""
        rl = tmp_path / "models" / "checkpoints_rl"
        sup = tmp_path / "models" / "checkpoints_supervised"
        root = tmp_path / "models"
        rl.mkdir(parents=True)
        sup.mkdir(parents=True)
        torch.save({"step": 10, "num_blocks": 2, "num_filters": 32}, rl / "ckpt_a.pt")
        torch.save({"step": 4, "epoch": 2, "num_blocks": 2, "num_filters": 32}, sup / "ckpt_b.pt")
        torch.save({"step": 1, "num_blocks": 2, "num_filters": 32}, root / "loose.pt")
        # a non-checkpoint file must not crash either scanner
        (sup / "notes.txt").write_text("hello", encoding="utf-8")

        # Both scanners derive their directories from the shared list at call
        # time against the module REPO_ROOT — patch it to the synthetic tree.
        monkeypatch.setattr(cli_common, "REPO_ROOT", tmp_path)

        common_paths = {str(p["path"].resolve()) for p in get_available_checkpoints()}
        model_paths = {str(p["path"].resolve()) for p in discover_checkpoints()}
        expected = {
            str((rl / "ckpt_a.pt").resolve()),
            str((sup / "ckpt_b.pt").resolve()),
            str((root / "loose.pt").resolve()),
        }
        assert common_paths == expected
        assert model_paths == expected
        assert common_paths == model_paths

    def test_shared_constant_is_the_union_list(self):
        names = cli_common.CHECKPOINT_SCAN_DIR_NAMES
        assert {
            "models/checkpoints_baseline_20261031",
            "models/checkpoints_scaled",
            "models/checkpoints_supervised",
            "models/checkpoints_rl",
            "models/checkpoints_finetune",
            "models/checkpoints",
            "models",
        }.issubset(set(names))
        # the models root is scanned last
        assert names[-1] == "models"
        assert cli_common.checkpoint_scan_dirs()[-1] == cli_common.REPO_ROOT / "models"
