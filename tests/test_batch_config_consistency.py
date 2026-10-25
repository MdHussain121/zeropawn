"""Regression test suite for Stage 2.4 Batch Configuration Consistency Fixes.

Validates the three audited configuration inconsistency fixes:
- F-01: UCIEngine defaults to batch_size=1 and advertises default 1 in UCI protocol.
- F-02: uci CLI subcommand exposes --batch-size (default 1) and propagates to UCIEngine.
- F-03: match live subcommand accepts --batch-size and --search-mode with default 1 / serial,
        explicitly rejects unsupported batching (>1 or 'auto') or batched search mode with
        actionable errors, and sets mcts_cfg.batch_size = 1.
"""

from __future__ import annotations

import argparse
import io
import sys
from unittest.mock import MagicMock, patch

import pytest

import zeropawn_cpp
from python.cli.commands_match import run_live_game
from python.cli.commands_uci import run_uci
from python.cli.main import build_parser
from python.uci.uci_engine import UCIEngine


def test_f01_uci_engine_batch_size_default():
    """F-01: UCIEngine must default to batch_size=1 and configure MCTSConfig with 1."""
    engine = UCIEngine(checkpoint_path=None, use_book=False)
    assert engine.batch_size == 1
    assert engine.mcts_config.batch_size == 1

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_buf
        engine.handle_line("uci")
        out = stdout_buf.getvalue()
        assert "option name BatchSize type spin default 1 min 1 max 64" in out
        assert "option name SearchMode type combo default serial var serial var batched" in out
    finally:
        sys.stdout = old_stdout

    # Verify setoption dynamically updates batch_size
    assert engine.handle_line("setoption name BatchSize value 8") is True
    assert engine.batch_size == 8
    assert engine.mcts_config.batch_size == 8


def test_f02_uci_cli_parser_and_propagation():
    """F-02: CLI uci command must expose --batch-size (default 1) and propagate it to UCIEngine."""
    parser = build_parser()

    # Test default
    args_default = parser.parse_args(["uci"])
    assert getattr(args_default, "batch_size", None) == "1"

    # Test explicit override
    args_custom = parser.parse_args(["uci", "--batch-size", "8"])
    assert args_custom.batch_size == "8"

    # Test propagation in run_uci
    with patch("python.cli.commands_uci.UCIEngine") as mock_engine_cls:
        mock_instance = MagicMock()
        mock_engine_cls.return_value = mock_instance
        ret = run_uci(args_custom)
        assert ret == 0
        mock_engine_cls.assert_called_once()
        _, kwargs = mock_engine_cls.call_args
        assert kwargs["batch_size"] == "8"
        mock_instance.run_stdio_loop.assert_called_once()


def test_f02_uci_cli_validation_rejects_invalid_batch_sizes(capsys):
    """F-02: run_uci must cleanly reject out-of-range or non-numeric batch sizes."""
    parser = build_parser()

    # Out of range (< 1)
    args_zero = parser.parse_args(["uci", "--batch-size", "0"])
    ret = run_uci(args_zero)
    assert ret == 1
    cap = capsys.readouterr()
    out = (cap.out + cap.err).replace("\n", " ")
    assert "--batch-size must be in 1..64 or 'auto'" in out

    # Out of range (> 64)
    args_high = parser.parse_args(["uci", "--batch-size", "128"])
    ret = run_uci(args_high)
    assert ret == 1
    cap = capsys.readouterr()
    out = (cap.out + cap.err).replace("\n", " ")
    assert "--batch-size must be in 1..64 or 'auto'" in out

    # Non-numeric
    args_invalid = parser.parse_args(["uci", "--batch-size", "xyz"])
    ret = run_uci(args_invalid)
    assert ret == 1
    cap = capsys.readouterr()
    out = (cap.out + cap.err).replace("\n", " ")
    assert "--batch-size must be an integer (1..64) or 'auto'" in out


def test_f03_match_live_cli_parser_defaults():
    """F-03: match live CLI parser must provide --batch-size 1 and --search-mode serial by default."""
    parser = build_parser()
    args = parser.parse_args(["match", "live"])
    assert args.batch_size == "1"
    assert args.search_mode == "serial"

    # Accepts explicit --batch-size 8 for parsing, but run_live_game will reject it
    args_explicit = parser.parse_args(["match", "live", "--batch-size", "8"])
    assert args_explicit.batch_size == "8"


def test_f03_match_live_rejects_unsupported_batching(capsys):
    """F-03: run_live_game must explicitly reject --batch-size > 1 or 'auto' or batched mode."""
    # Test batch_size > 1
    args = argparse.Namespace(
        search_mode="serial",
        batch_size="8",
        checkpoint=None,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        stockfish_path=None,
        stockfish_nodes=1000,
        mcts_sims=50,
        delay=0.0,
        device="cpu",
    )
    with (
        patch("python.cli.commands_match.get_default_checkpoint_path", return_value="mock.pt"),
        patch("python.cli.commands_match.safe_resolve", return_value="mock.pt"),
        patch("python.cli.commands_match.resolve_stockfish", return_value="tools/stockfish.exe"),
        patch("pathlib.Path.is_file", return_value=True),
        patch("python.cli.commands_match.load_engine_model", return_value=(MagicMock(), {})),
    ):
        ret = run_live_game(args)
        assert ret == 1
        cap = capsys.readouterr()
        out = " ".join((cap.out + cap.err).split())
        assert (
            "match live only supports serial search with batch_size=1; received batch_size=8" in out
        )

    # Test batch_size='auto'
    args.batch_size = "auto"
    with (
        patch("python.cli.commands_match.get_default_checkpoint_path", return_value="mock.pt"),
        patch("python.cli.commands_match.safe_resolve", return_value="mock.pt"),
        patch("python.cli.commands_match.resolve_stockfish", return_value="tools/stockfish.exe"),
        patch("pathlib.Path.is_file", return_value=True),
        patch("python.cli.commands_match.load_engine_model", return_value=(MagicMock(), {})),
    ):
        ret = run_live_game(args)
        assert ret == 1
        cap = capsys.readouterr()
        out = " ".join((cap.out + cap.err).split())
        assert (
            "match live only supports serial search with batch_size=1; received batch_size='auto'"
            in out
        )

    # Test search_mode='batched'
    args.batch_size = "1"
    args.search_mode = "batched"
    with (
        patch("python.cli.commands_match.get_default_checkpoint_path", return_value="mock.pt"),
        patch("python.cli.commands_match.safe_resolve", return_value="mock.pt"),
        patch("python.cli.commands_match.resolve_stockfish", return_value="tools/stockfish.exe"),
        patch("pathlib.Path.is_file", return_value=True),
        patch("python.cli.commands_match.load_engine_model", return_value=(MagicMock(), {})),
    ):
        ret = run_live_game(args)
        assert ret == 1
        cap = capsys.readouterr()
        out = " ".join((cap.out + cap.err).split())
        assert (
            "match live only supports serial search (search_mode='serial'); received search_mode='batched'"
            in out
        )


def test_f03_match_live_configures_mcts_batch_size_one():
    """F-03: When batch_size=1, run_live_game must configure MCTSConfig with batch_size=1."""
    args = argparse.Namespace(
        search_mode="serial",
        batch_size="1",
        checkpoint=None,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        stockfish_path=None,
        stockfish_nodes=1000,
        mcts_sims=50,
        delay=0.0,
        step=False,
        engine_color="white",
        reuse_tree=False,
        device="cpu",
    )

    created_configs = []
    real_mcts_config_cls = zeropawn_cpp.MCTSConfig

    def mock_mcts_config_factory():
        cfg = real_mcts_config_cls()
        created_configs.append(cfg)
        return cfg

    # Mock game to terminate immediately so run_live_game exits after setup
    mock_game = MagicMock()
    mock_game.is_game_over.return_value = True
    mock_game.get_fen.return_value = args.fen

    with (
        patch("python.cli.commands_match.get_default_checkpoint_path", return_value="mock.pt"),
        patch("python.cli.commands_match.safe_resolve", return_value="mock.pt"),
        patch("python.cli.commands_match.resolve_stockfish", return_value="tools/stockfish.exe"),
        patch("pathlib.Path.is_file", return_value=True),
        patch("python.cli.commands_match.load_engine_model", return_value=(MagicMock(), {})),
        patch("python.eval.match_runner.StockfishPlayer"),
        patch.object(zeropawn_cpp, "MCTSConfig", side_effect=mock_mcts_config_factory),
        patch.object(zeropawn_cpp, "ChessGame", return_value=mock_game),
        patch("python.cli.commands_match.console.print"),
    ):
        ret = run_live_game(args)
        assert ret == 0
        assert len(created_configs) == 1
        assert created_configs[0].batch_size == 1
