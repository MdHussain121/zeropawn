"""Configuration-precedence matrix + SA1-01/SA1-02 regression tests (2026-11-05).

Remediation suite for the 2026-11-05 pre-push audit findings SA1-01 and
SA1-02 (section 4 SA-1 of ZERO_PAWN_PRE_PUSH_AUDIT_20261105.md):

* SA1-01 (Medium) — ``eval.use_book`` configuration/environment propagation
  INVERTED the effective setting when the value came from a config file or
  environment variable, and clobbered the ``--book <path>`` argument with a
  boolean.  Root cause: the propagation loop in ``python/cli/main.py``
  blindly ``setattr``-ed every alias / key / flag-derived attribute name,
  including the match parser's ``no_book`` (store_true) dest, whose truth
  the handler negates (``use_book=not args.no_book``).
* SA1-02 (Medium) — ``--batch-size`` collided between the training tier
  (``training.batch_size``, int) and the MCTS/match inference tier
  (``mcts.batch_size``, str): a config-file/env training batch size leaked
  into the search-side namespace (and the string-typed MCTS value leaked
  into ``train`` in the other direction).

The suite pins the full 5-tier precedence matrix (defaults < config file <
presets < environment < explicit CLI) with deliberately conflicting
combinations for the two audited keys plus representative keys that flow
through the same propagation machinery (``system.device``,
``eval.mcts_sims``, ``mcts.reuse_tree``, ``eval.book_path``), and asserts
per-command namespace isolation (a training batch size must reach ONLY its
training component; an MCTS batch size ONLY its search component).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.cli.config import (
    CONFIG_REGISTRY,
    propagate_resolved_to_args,
    resolve_configuration,
)
from python.cli.main import build_parser, main


# ────────────────────────────── helpers ──────────────────────────────

# Every env suffix the matrix touches, in both modern and legacy spellings.
# Cleared before each e2e run so host environments cannot contaminate proofs.
_ENV_SUFFIXES = [
    "EVAL_USE_BOOK",
    "USE_BOOK",
    "TRAINING_BATCH_SIZE",
    "BATCH_SIZE",
    "MCTS_BATCH_SIZE",
    "SYSTEM_DEVICE",
    "DEVICE",
    "EVAL_MCTS_SIMS",
    "MCTS_SIMS",
    "MCTS_REUSE_TREE",
    "REUSE_TREE",
    "EVAL_BOOK_PATH",
    "BOOK_PATH",
]


@pytest.fixture()
def clean_env(monkeypatch):
    """Remove all matrix-relevant env vars (both ZEROPAWN_* and CHESSRL_*)."""
    for suffix in _ENV_SUFFIXES:
        monkeypatch.delenv(f"ZEROPAWN_{suffix}", raising=False)
        monkeypatch.delenv(f"CHESSRL_{suffix}", raising=False)
    return monkeypatch


def _capture_handler(monkeypatch, module_name, handler_name):
    """Patch <module>.<handler> with a capturer; returns the seen-args dict."""
    import importlib

    mod = importlib.import_module(module_name)
    seen: dict = {}

    def _fake(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(mod, handler_name, _fake)
    return seen


def _run_main(monkeypatch, argv, config_file=None):
    """Run the real main() chain and return (rc, argv_used)."""
    argv_full = list(argv)
    if config_file is not None:
        argv_full = ["--config", str(config_file)] + argv_full
    rc = main(argv_full)
    return rc, argv_full


def _write_config(tmp_path: Path, settings: dict) -> Path:
    cfg = tmp_path / "matrix_config.json"
    cfg.write_text(json.dumps({"settings": settings}), encoding="utf-8")
    return cfg


def _propagate_inline(resolved, args):
    """Run main.py's REAL propagation loop (extracted to
    config.propagate_resolved_to_args by the SA1-01/SA1-02 remediation)."""
    propagate_resolved_to_args(resolved, args, parser=build_parser())
    return args


def _resolve_and_propagate(argv, tmp_path, config_file=None, env=None, preset_dict=None):
    """Unit-level: parse with the real parser, resolve, run the propagation loop."""
    parser = build_parser()
    args = parser.parse_args(list(argv))
    resolved = resolve_configuration(
        command_name=getattr(args, "command", "") or (argv[0] if argv else ""),
        cli_args=args,
        preset_dict=preset_dict,
        config_path=config_file if config_file is not None else tmp_path / "unused.json",
        raw_argv=list(argv),
    )
    return _propagate_inline(resolved, args), resolved


# ════════════════════════ SA1-01: eval.use_book inversion ════════════════════════


class TestSA101UseBookInversion:
    """`eval.use_book` via config file / env must set book usage, not invert it.

    The match handler derives book usage as ``use_book = not args.no_book``
    (python/cli/commands_match.py).  The buggy propagation wrote the RAW
    boolean into ``args.no_book``, inverting it; it also clobbered
    ``args.book`` (the book PATH) with the boolean.
    """

    @pytest.mark.parametrize(
        ("cfg_value", "expected_use_book"),
        [(False, False), (True, True)],
        ids=["config-disables-book", "config-enables-book"],
    )
    def test_config_file_use_book_reaches_match_verbatim(
        self, clean_env, tmp_path, monkeypatch, cfg_value, expected_use_book
    ):
        cfg = _write_config(tmp_path, {"eval.use_book": cfg_value})
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        args = seen["args"]
        # The handler's exact expression: book usage must match the config value.
        assert hasattr(args, "no_book")  # sanity: the negated dest exists
        assert (not getattr(args, "no_book", False)) is expected_use_book, (
            f"SA1-01 regression: config eval.use_book={cfg_value} propagated to "
            f"args.no_book={getattr(args, 'no_book', None)!r} → handler sees "
            f"use_book={not getattr(args, 'no_book', False)} (inverted)"
        )
        # The book PATH attribute must never be overwritten with a boolean.
        assert isinstance(getattr(args, "book", None), str), (
            f"SA1-01 regression: args.book was clobbered with "
            f"{getattr(args, 'book', None)!r} (must stay the book path string)"
        )

    @pytest.mark.parametrize(
        ("env_value", "expected_use_book"),
        [("true", True), ("false", False)],
        ids=["env-enables-book", "env-disables-book"],
    )
    def test_env_use_book_reaches_match_verbatim(
        self, clean_env, tmp_path, monkeypatch, env_value, expected_use_book
    ):
        monkeypatch.setenv("ZEROPAWN_EVAL_USE_BOOK", env_value)
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"])
        assert rc == 0
        args = seen["args"]
        assert (not getattr(args, "no_book", False)) is expected_use_book, (
            f"SA1-01 regression: ZEROPAWN_EVAL_USE_BOOK={env_value} reached the "
            f"match handler inverted (args.no_book={getattr(args, 'no_book', None)!r})"
        )
        assert isinstance(getattr(args, "book", None), str)

    def test_explicit_no_book_cli_still_beats_env_true(self, clean_env, tmp_path, monkeypatch):
        """Explicit CLI usage must remain unaffected (audit: 'Explicit CLI
        usage is unaffected') — --no-book beats ZEROPAWN_EVAL_USE_BOOK=true."""
        monkeypatch.setenv("ZEROPAWN_EVAL_USE_BOOK", "true")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run", "--no-book"])
        assert rc == 0
        args = seen["args"]
        assert args.no_book is True
        assert (not args.no_book) is False

    def test_explicit_book_path_not_clobbered_by_config_use_book(
        self, clean_env, tmp_path, monkeypatch
    ):
        """`--book <custom path>` (explicit CLI) + config use_book=false: the
        path flag must survive untouched AND the book must stay disabled."""
        cfg = _write_config(tmp_path, {"eval.use_book": False})
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(
            monkeypatch,
            ["match", "run", "--book", "data/openings/custom_book.json"],
            config_file=cfg,
        )
        assert rc == 0
        args = seen["args"]
        assert args.book == "data/openings/custom_book.json", (
            f"SA1-01 regression: explicit --book path was clobbered by the "
            f"config-file use_book propagation (args.book={args.book!r})"
        )
        assert (not getattr(args, "no_book", False)) is False

    def test_unit_level_propagation_no_inversion(self, clean_env, tmp_path):
        """Unit-level pin (no dispatch): config use_book=false must land as
        args.no_book=True so the handler's `not args.no_book` yields False."""
        cfg = _write_config(tmp_path, {"eval.use_book": False})
        args, resolved = _resolve_and_propagate(["match", "run"], tmp_path, config_file=cfg)
        rec = resolved.get_record("eval.use_book")
        assert rec.source == "config_file"
        assert rec.effective_value is False
        assert getattr(args, "no_book", False) is True, (
            f"SA1-01 regression: propagation wrote args.no_book="
            f"{getattr(args, 'no_book', None)!r}; the handler negates it, so the "
            f"config value reached the runtime inverted"
        )


# ════════════════════════ SA1-02: batch-size namespace collision ════════════════════════


class TestSA102BatchSizeCollision:
    """`--batch-size` training↔MCTS isolation.

    ``training.batch_size`` (int, default 512) must reach ONLY the training
    components (train / supervised / finetune).  ``mcts.batch_size`` (str,
    default 'auto') must reach ONLY the search components (match / uci /
    puzzles-mcts / search / bench) — and on `puzzles` ONLY the dedicated
    ``mcts_batch_size`` dest, never the policy-eval ``batch_size`` dest.
    """

    def test_training_batch_config_does_not_leak_into_match(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"training.batch_size": 64})
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        args = seen["args"]
        assert getattr(args, "batch_size", "auto") == "auto", (
            f"SA1-02 regression: training.batch_size=64 (config file) leaked into "
            f"the match/MCTS namespace (args.batch_size={args.batch_size!r}); the "
            f"search side must keep its own 'auto' default"
        )
        rec = None
        # the training tier record still resolves (for `config show` etc.)
        _, resolved_args = _resolve_and_propagate(["match", "run"], tmp_path, config_file=cfg)
        rec = resolved_args.get_record("training.batch_size")
        assert rec.source == "config_file" and rec.effective_value == 64

    def test_training_batch_env_does_not_leak_into_match(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_BATCH_SIZE", "64")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"])
        assert rc == 0
        args = seen["args"]
        assert getattr(args, "batch_size", "auto") == "auto", (
            f"SA1-02 regression: ZEROPAWN_BATCH_SIZE=64 (training tier) leaked "
            f"into the match/MCTS namespace (args.batch_size={args.batch_size!r})"
        )

    def test_mcts_batch_config_does_not_leak_into_train(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"mcts.batch_size": "64"})
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_training")
        rc, _ = _run_main(monkeypatch, ["train"], config_file=cfg)
        assert rc == 0
        args = seen["args"]
        assert args.batch_size == 512 and isinstance(args.batch_size, int), (
            f"SA1-02 regression: mcts.batch_size='64' (config file, string-typed) "
            f"leaked into the train namespace (args.batch_size={args.batch_size!r})"
        )

    def test_mcts_batch_env_does_not_leak_into_train(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_MCTS_BATCH_SIZE", "64")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_training")
        rc, _ = _run_main(monkeypatch, ["train"])
        assert rc == 0
        args = seen["args"]
        assert args.batch_size == 512 and isinstance(args.batch_size, int), (
            f"SA1-02 regression: ZEROPAWN_MCTS_BATCH_SIZE=64 leaked into the "
            f"train namespace (args.batch_size={args.batch_size!r})"
        )

    def test_mcts_batch_env_reaches_match(self, clean_env, tmp_path, monkeypatch):
        """Positive control: the MCTS tier must still reach the search side."""
        monkeypatch.setenv("ZEROPAWN_MCTS_BATCH_SIZE", "32")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"])
        assert rc == 0
        assert seen["args"].batch_size == "32"

    def test_training_batch_env_reaches_supervised(self, clean_env, tmp_path, monkeypatch):
        """Positive control: the training tier must still reach its component."""
        monkeypatch.setenv("ZEROPAWN_TRAINING_BATCH_SIZE", "128")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_supervised_training")
        rc, _ = _run_main(monkeypatch, ["supervised"])
        assert rc == 0
        assert seen["args"].batch_size == 128

    def test_mcts_batch_on_puzzles_targets_only_mcts_dest(self, clean_env, tmp_path, monkeypatch):
        """On `puzzles eval` the search-side dest is --mcts-batch-size; the
        int policy-eval --batch-size dest (default 128) must never receive
        the string-typed MCTS value."""
        cfg = _write_config(tmp_path, {"mcts.batch_size": "64"})
        seen = _capture_handler(monkeypatch, "python.cli.commands_puzzles", "run_eval_puzzles")
        rc, _ = _run_main(monkeypatch, ["puzzles", "eval"], config_file=cfg)
        assert rc == 0
        args = seen["args"]
        assert getattr(args, "mcts_batch_size", None) == "64"
        assert getattr(args, "batch_size", None) == 128, (
            f"SA1-02 regression: mcts.batch_size leaked into the puzzles "
            f"policy-eval batch_size dest (args.batch_size={args.batch_size!r})"
        )

    def test_explicit_cli_batch_sizes_unaffected(self, clean_env, tmp_path, monkeypatch):
        """Explicit CLI usage must remain unaffected (audit note)."""
        seen_match = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run", "--batch-size", "8"])
        assert rc == 0
        assert seen_match["args"].batch_size == "8"

        seen_train = _capture_handler(
            monkeypatch, "python.cli.commands_train", "run_supervised_training"
        )
        rc, _ = _run_main(monkeypatch, ["supervised", "--batch-size", "256"])
        assert rc == 0
        assert seen_train["args"].batch_size == 256


# ════════════════════════ precedence matrix (conflicting combos) ════════════════════════


class TestPrecedenceMatrixConflicts:
    """Full 5-tier matrix with deliberately conflicting tiers.

    Precedence (highest wins): explicit CLI > env > preset > config file >
    built-in defaults.  Every conflict below stacks tiers with OPPOSITE
    values and asserts the strongest tier survives — for the two audited
    keys plus representative keys sharing the propagation machinery.
    """

    # ── eval.use_book ──
    def test_use_book_conflict_config_false_env_true(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.use_book": False})
        monkeypatch.setenv("ZEROPAWN_EVAL_USE_BOOK", "true")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert (not seen["args"].no_book) is True, "env tier must beat config file"

    def test_use_book_conflict_config_true_env_false(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.use_book": True})
        monkeypatch.setenv("ZEROPAWN_EVAL_USE_BOOK", "false")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert (not seen["args"].no_book) is False, "env tier must beat config file"

    def test_use_book_conflict_all_three_tiers(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.use_book": True})
        monkeypatch.setenv("ZEROPAWN_EVAL_USE_BOOK", "false")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run", "--no-book"], config_file=cfg)
        assert rc == 0
        assert seen["args"].no_book is True and (not seen["args"].no_book) is False, (
            "explicit CLI --no-book must beat env=false and config=true"
        )

    def test_use_book_default_when_no_tiers_set(self, clean_env, tmp_path, monkeypatch):
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"])
        assert rc == 0
        assert (not seen["args"].no_book) is True, "built-in default is book-enabled"
        assert seen["args"].book == "data/openings/curated_book.json"

    # ── training.batch_size (train/supervised side) ──
    def test_train_batch_conflict_config_env(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"training.batch_size": 256})
        monkeypatch.setenv("ZEROPAWN_TRAINING_BATCH_SIZE", "128")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_supervised_training")
        rc, _ = _run_main(monkeypatch, ["supervised"], config_file=cfg)
        assert rc == 0
        assert seen["args"].batch_size == 128, "env must beat config file"

    def test_train_batch_conflict_config_env_cli(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"training.batch_size": 256})
        monkeypatch.setenv("ZEROPAWN_BATCH_SIZE", "128")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_supervised_training")
        rc, _ = _run_main(monkeypatch, ["supervised", "--batch-size", "64"], config_file=cfg)
        assert rc == 0
        assert seen["args"].batch_size == 64, "explicit CLI must beat env and config"

    def test_train_batch_config_only(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"training.batch_size": 256})
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_supervised_training")
        rc, _ = _run_main(monkeypatch, ["supervised"], config_file=cfg)
        assert rc == 0
        assert seen["args"].batch_size == 256

    def test_train_batch_default_only(self, clean_env, tmp_path, monkeypatch):
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_supervised_training")
        rc, _ = _run_main(monkeypatch, ["supervised"])
        assert rc == 0
        assert seen["args"].batch_size == 512

    # ── mcts.batch_size (search side) ──
    def test_mcts_batch_conflict_config_env(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"mcts.batch_size": "32"})
        monkeypatch.setenv("ZEROPAWN_MCTS_BATCH_SIZE", "48")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert seen["args"].batch_size == "48", "env must beat config file"

    def test_mcts_batch_conflict_config_env_cli(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"mcts.batch_size": "32"})
        monkeypatch.setenv("ZEROPAWN_MCTS_BATCH_SIZE", "48")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run", "--batch-size", "16"], config_file=cfg)
        assert rc == 0
        assert seen["args"].batch_size == "16", "explicit CLI must beat env and config"

    def test_mcts_batch_cross_tier_isolation(self, clean_env, tmp_path, monkeypatch):
        """THE SA1-02 conflict: training tier set high, MCTS tier set low,
        both in the SAME config file — each side must see ONLY its own value."""
        cfg = _write_config(tmp_path, {"training.batch_size": 4096, "mcts.batch_size": "8"})
        seen_match = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert seen_match["args"].batch_size == "8"

        seen_train = _capture_handler(
            monkeypatch, "python.cli.commands_train", "run_supervised_training"
        )
        rc, _ = _run_main(monkeypatch, ["supervised"], config_file=cfg)
        assert rc == 0
        assert seen_train["args"].batch_size == 4096

    # ── system.device (choice; multi-workflow representative) ──
    def test_device_conflict_config_env(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"system.device": "cuda"})
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "cpu")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert seen["args"].device == "cpu", "env must beat config file"

    def test_device_conflict_config_env_cli(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"system.device": "cuda"})
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "cpu")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run", "--device", "cuda"], config_file=cfg)
        assert rc == 0
        assert seen["args"].device == "cuda", "explicit CLI must beat env and config"

    # ── eval.mcts_sims (int; reaches match AND search) ──
    def test_mcts_sims_conflict_config_env_cli(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.mcts_sims": 100})
        monkeypatch.setenv("ZEROPAWN_EVAL_MCTS_SIMS", "42")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run", "--mcts-sims", "77"], config_file=cfg)
        assert rc == 0
        assert seen["args"].mcts_sims == 77, "explicit CLI must beat env and config"

    def test_mcts_sims_env_beats_config(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.mcts_sims": 100})
        monkeypatch.setenv("ZEROPAWN_EVAL_MCTS_SIMS", "42")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert seen["args"].mcts_sims == 42

    def test_mcts_sims_config_file_only(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.mcts_sims": 100})
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        assert seen["args"].mcts_sims == 100

    # ── mcts.reuse_tree (bool store_true with DIRECT dest mapping) ──
    def test_reuse_tree_env_reaches_uci(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_MCTS_REUSE_TREE", "true")
        seen = _capture_handler(monkeypatch, "python.cli.commands_uci", "run_uci")
        rc, _ = _run_main(monkeypatch, ["uci"])
        assert rc == 0
        assert seen["args"].reuse_tree is True

    def test_reuse_tree_config_env_cli_conflict(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"mcts.reuse_tree": True})
        monkeypatch.setenv("ZEROPAWN_MCTS_REUSE_TREE", "false")
        seen = _capture_handler(monkeypatch, "python.cli.commands_uci", "run_uci")
        rc, _ = _run_main(monkeypatch, ["uci", "--reuse-tree"], config_file=cfg)
        assert rc == 0
        assert seen["args"].reuse_tree is True, "explicit CLI store_true must win"

    # ── eval.book_path (path; shares the --book flag family with use_book) ──
    def test_book_path_config_reaches_match(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.book_path": "data/openings/other_book.json"})
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"], config_file=cfg)
        assert rc == 0
        args = seen["args"]
        assert args.book == "data/openings/other_book.json"
        # book usage must remain at its default (enabled) — path tier is separate
        assert (not args.no_book) is True

    def test_book_path_conflict_config_env_cli(self, clean_env, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {"eval.book_path": "data/openings/cfg_book.json"})
        monkeypatch.setenv("ZEROPAWN_EVAL_BOOK_PATH", "data/openings/env_book.json")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(
            monkeypatch,
            ["match", "run", "--book", "data/openings/cli_book.json"],
            config_file=cfg,
        )
        assert rc == 0
        assert seen["args"].book == "data/openings/cli_book.json", (
            "explicit CLI --book path must beat env and config tiers"
        )


# ════════════════════════ preset tier through the propagation loop ════════════════════════


class TestPresetTierPropagation:
    """Preset parameters flow through the same loop; the same isolation and
    no-inversion guarantees must hold for the preset tier."""

    def test_preset_use_book_no_inversion(self, clean_env, tmp_path):
        args, resolved = _resolve_and_propagate(
            ["match", "run"], tmp_path, preset_dict={"parameters": {"use_book": False}}
        )
        rec = resolved.get_record("eval.use_book")
        assert rec.source == "preset" and rec.effective_value is False
        assert getattr(args, "no_book", False) is True

    def test_preset_training_batch_does_not_leak_into_match(self, clean_env, tmp_path):
        args, resolved = _resolve_and_propagate(
            ["match", "run"], tmp_path, preset_dict={"parameters": {"batch_size": 64}}
        )
        assert getattr(args, "batch_size", "auto") == "auto", (
            "preset training batch_size must not leak into the match/MCTS namespace"
        )

    def test_preset_training_batch_reaches_supervised(self, clean_env, tmp_path):
        args, _ = _resolve_and_propagate(
            ["supervised"], tmp_path, preset_dict={"parameters": {"batch_size": 64}}
        )
        assert args.batch_size == 64

    def test_preset_mcts_batch_does_not_stringify_train(self, clean_env, tmp_path):
        args, _ = _resolve_and_propagate(
            ["train"], tmp_path, preset_dict={"parameters": {"mcts_batch_size": "64"}}
        )
        assert args.batch_size == 512 and isinstance(args.batch_size, int)


# ════════════════════════ namespace isolation matrix (unit level) ════════════════════════


class TestNamespaceIsolationMatrix:
    """Parametrized per-command isolation for the shared --batch-size flag.

    For EVERY command whose parser exposes a ``batch_size``/``mcts_batch_size``
    dest, a config file carrying BOTH batch tiers must deliver exactly the
    owning tier's value — the same bug class SA1-02 was filed for, pinned
    across the whole command surface (unit level: parse → resolve → real
    propagation loop, no dispatch).
    """

    # Commands whose search-side dest is `batch_size` (on `puzzles eval` the
    # search value lands in `mcts_batch_size` instead — covered separately).
    SEARCH_COMMANDS = ["match run", "search", "uci"]
    TRAIN_COMMANDS = ["train", "supervised", "finetune"]

    def _isolate(self, tmp_path, argv, settings):
        cfg = _write_config(tmp_path, settings)
        args, _ = _resolve_and_propagate(argv.split(), tmp_path, config_file=cfg)
        return args

    @pytest.mark.parametrize("cmd", SEARCH_COMMANDS)
    def test_search_side_gets_only_mcts_tier(self, clean_env, tmp_path, cmd):
        args = self._isolate(tmp_path, cmd, {"training.batch_size": 4096, "mcts.batch_size": "8"})
        assert args.batch_size == "8", (
            f"{cmd}: search-side batch dest must receive mcts.batch_size only "
            f"(got {args.batch_size!r} with training.batch_size=4096 also set)"
        )

    @pytest.mark.parametrize("cmd", TRAIN_COMMANDS)
    def test_train_side_gets_only_training_tier(self, clean_env, tmp_path, cmd):
        args = self._isolate(tmp_path, cmd, {"training.batch_size": 4096, "mcts.batch_size": "8"})
        assert args.batch_size == 4096 and isinstance(args.batch_size, int), (
            f"{cmd}: training batch dest must receive training.batch_size only "
            f"(got {args.batch_size!r} with mcts.batch_size='8' also set)"
        )

    def test_puzzles_policy_batch_is_neither_tier(self, clean_env, tmp_path):
        """puzzles' int policy-eval batch (default 128) is owned by NEITHER
        registry tier and must stay at its parser default."""
        args = self._isolate(
            tmp_path, "puzzles eval", {"training.batch_size": 4096, "mcts.batch_size": "8"}
        )
        assert args.batch_size == 128
        assert args.mcts_batch_size == "8"


# ════════════════════════ latent same-class collision regressions ════════════════════════


class TestLatentCollisionRegressions:
    """Same bug class as SA1-02, other flag collisions the per-command
    disambiguation now blocks.  Each pair names the env tier and the command
    whose same-named dest has a DIFFERENT meaning."""

    def test_mcts_search_mode_env_does_not_corrupt_train_mode(
        self, clean_env, tmp_path, monkeypatch
    ):
        """train's --mode selects rl|supervised; mcts.search_mode ('serial'/
        'batched') must never reach it (the old loop wrote args.mode='serial')."""
        monkeypatch.setenv("ZEROPAWN_MCTS_SEARCH_MODE", "serial")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_training")
        rc, _ = _run_main(monkeypatch, ["train"])
        assert rc == 0
        assert seen["args"].mode == "rl", (
            "mcts.search_mode leaked into train's rl|supervised --mode dest"
        )

    def test_eval_games_env_does_not_leak_into_test_command(self, clean_env, tmp_path, monkeypatch):
        """test's --games counts differential random games, not match games."""
        monkeypatch.setenv("ZEROPAWN_EVAL_GAMES", "999")
        seen = _capture_handler(monkeypatch, "python.cli.commands_test", "run_tests")
        rc, _ = _run_main(monkeypatch, ["test"])
        assert rc == 0
        assert seen["args"].games is None

    def test_eval_workers_env_does_not_leak_into_train_dataloader(
        self, clean_env, tmp_path, monkeypatch
    ):
        """train's --workers counts DataLoader processes; eval.workers counts
        match threads.  Different settings sharing a dest name."""
        monkeypatch.setenv("ZEROPAWN_EVAL_WORKERS", "16")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_supervised_training")
        rc, _ = _run_main(monkeypatch, ["supervised"])
        assert rc == 0
        assert seen["args"].workers == 2

    def test_eval_workers_env_still_reaches_match(self, clean_env, tmp_path, monkeypatch):
        """Positive control: eval.workers still reaches its own workflow."""
        monkeypatch.setenv("ZEROPAWN_EVAL_WORKERS", "16")
        seen = _capture_handler(monkeypatch, "python.cli.commands_match", "run_batch_match")
        rc, _ = _run_main(monkeypatch, ["match", "run"])
        assert rc == 0
        assert seen["args"].workers == 16

    # ── preserved propagations onto commands outside the display list ──
    # (authorized via ConfigField.propagate_attrs; the S5-01 precondition
    # tests pin these commands' absence from applicable_workflows)

    def test_mcts_sims_env_reaches_uci(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_EVAL_MCTS_SIMS", "42")
        seen = _capture_handler(monkeypatch, "python.cli.commands_uci", "run_uci")
        rc, _ = _run_main(monkeypatch, ["uci"])
        assert rc == 0
        assert seen["args"].sims == 42

    def test_device_env_reaches_search(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_SYSTEM_DEVICE", "cpu")
        seen = _capture_handler(monkeypatch, "python.cli.commands_search", "run_search")
        rc, _ = _run_main(monkeypatch, ["mcts", "--search-mode", "serial"])
        assert rc == 0
        assert seen["args"].device == "cpu"

    def test_stockfish_nodes_env_reaches_train(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_EVAL_STOCKFISH_NODES", "2500")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_training")
        rc, _ = _run_main(monkeypatch, ["train"])
        assert rc == 0
        assert seen["args"].stockfish_nodes == 2500

    def test_epochs_env_reaches_train(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.setenv("ZEROPAWN_TRAINING_EPOCHS", "7")
        seen = _capture_handler(monkeypatch, "python.cli.commands_train", "run_training")
        rc, _ = _run_main(monkeypatch, ["train"])
        assert rc == 0
        assert seen["args"].epochs == 7


# ════════════════════════ propagation function unit pins ════════════════════════


class TestPropagationFunctionUnit:
    """Direct pins on config.propagate_resolved_to_args behavior."""

    def test_command_alias_canonicalized_for_authorization(self, clean_env, tmp_path):
        """`mcts` is the registered alias of `search`: the propagation loop
        must canonicalize the command name before the workflow check, so env
        eval.mcts_sims still reaches args.sims through `zeropawn mcts`."""
        clean_env.setenv("ZEROPAWN_EVAL_MCTS_SIMS", "42")
        parser = build_parser()
        args = parser.parse_args(["mcts"])
        resolved = resolve_configuration(
            command_name="mcts",
            cli_args=args,
            config_path=tmp_path / "unused.json",
            raw_argv=["mcts"],
        )
        propagate_resolved_to_args(resolved, args, parser=parser)
        assert args.sims == 42

    def test_use_book_propagate_map_shape(self):
        targets = CONFIG_REGISTRY["eval.use_book"].propagate_attrs
        assert targets == {"*": (("use_book", "set"), ("no_book", "negate"))}

    def test_mcts_batch_size_propagate_map_shape(self):
        targets = CONFIG_REGISTRY["mcts.batch_size"].propagate_attrs
        assert targets["puzzles"] == (("mcts_batch_size", "set"),)
        assert targets["match"] == (("batch_size", "set"),)
        assert "train" not in targets and "supervised" not in targets
        assert "finetune" not in targets
        assert "*" in targets

    def test_mcts_search_mode_has_no_train_authorization(self):
        targets = CONFIG_REGISTRY["mcts.search_mode"].propagate_attrs
        assert not targets or "train" not in targets

    def test_all_propagate_attr_transforms_valid(self):
        """Structural guard: transforms are 'set'/'negate'; 'negate' targets
        only exist on bool fields (a negated write of a non-bool is a bug)."""
        for key, cf in CONFIG_REGISTRY.items():
            for base, spec in (cf.propagate_attrs or {}).items():
                assert spec, f"{key}: empty propagation spec for {base!r}"
                for attr, transform in spec:
                    assert transform in ("set", "negate"), f"{key}/{base}: {attr}"
                    if transform == "negate":
                        assert cf.type_name == "bool", (
                            f"{key}: 'negate' transform on non-bool field "
                            f"(type_name={cf.type_name})"
                        )

    def test_negation_aware_cli_tier_extraction(self, tmp_path):
        """`match run --no-book` must attribute use_book=False to the CLI tier
        deterministically (not by the old accident of coercing the book path
        string), and `--book <path>` must NOT toggle the use_book tier."""
        parser = build_parser()
        argv = ["match", "run", "--no-book"]
        args = parser.parse_args(argv)
        resolved = resolve_configuration(
            command_name="match",
            cli_args=args,
            config_path=tmp_path / "unused.json",
            raw_argv=argv,
        )
        rec = resolved.get_record("eval.use_book")
        assert rec.source == "cli"
        assert rec.effective_value is False

        argv2 = ["match", "run", "--book", "data/openings/custom.json"]
        args2 = parser.parse_args(argv2)
        resolved2 = resolve_configuration(
            command_name="match",
            cli_args=args2,
            config_path=tmp_path / "unused.json",
            raw_argv=argv2,
        )
        rec2 = resolved2.get_record("eval.use_book")
        assert rec2.source == "default", (
            "--book <path> belongs to eval.book_path and must not toggle the eval.use_book tier"
        )
        assert resolved2.get_record("eval.book_path").effective_value == (
            "data/openings/custom.json"
        )


# ════════════════════════ registry invariants ════════════════════════


class TestRegistryInvariants:
    """Registry-level pins that make the bug classes structurally visible."""

    def test_use_book_registry_shape(self):
        """eval.use_book is a bool field whose only CLI toggle is --no-book
        (the --book flag belongs to eval.book_path — claiming it here is what
        let the propagation clobber the path dest with a boolean)."""
        cf = CONFIG_REGISTRY["eval.use_book"]
        assert cf.type_name == "bool"
        assert "--no-book" in cf.cli_flags
        assert "--book" not in cf.cli_flags, (
            "--book sets the book PATH (eval.book_path); eval.use_book claiming "
            "it makes the propagation write a boolean into the path dest"
        )

    def test_batch_size_fields_have_distinct_aliases(self):
        assert CONFIG_REGISTRY["training.batch_size"].alias == "batch_size"
        assert CONFIG_REGISTRY["mcts.batch_size"].alias == "mcts_batch_size"
        assert (
            CONFIG_REGISTRY["training.batch_size"].alias != CONFIG_REGISTRY["mcts.batch_size"].alias
        )
