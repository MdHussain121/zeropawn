"""Unit tests for the unified ZeroPawn CLI."""

import argparse
from pathlib import Path
import pytest

from python.cli.main import build_parser
from python.cli.common import (
    REPO_ROOT,
    get_available_checkpoints,
    get_default_checkpoint_path,
    get_default_stockfish_path,
    render_ascii_board,
    resolve_path,
)
from python.cli.commands_add_puzzles import run_manual_add, run_list_puzzles
from python.cli.commands_status import run_status


def test_cli_parser_build():
    """Verify parser construction and subcommand structure."""
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    # Test subcommand parsing
    args = parser.parse_args(["test", "--suite", "fast"])
    assert args.command == "test"
    assert args.suite == "fast"

    args = parser.parse_args(["train", "--max-hours", "10", "--verify"])
    assert args.command in ["train", "tr"]
    assert args.max_hours == 10.0
    assert args.verify is True
    assert args.concurrent_games == 48

    args = parser.parse_args(["match", "-n", "10", "--mcts-sims", "30"])
    assert args.command in ["match", "play", "stockfish", "vs"]
    assert args.games == 10
    assert args.mcts_sims == 30

    args = parser.parse_args(["puzzles", "eval", "-n", "25"])
    assert args.command in ["puzzles", "puzzle", "p"]
    assert args.puzzle_subcommand == "eval"
    assert args.limit == 25

    args = parser.parse_args(["add-puzzles", "list"])
    assert args.command in ["add-puzzles", "add-puzzle", "fetch-puzzles", "puzzle-tool"]
    assert args.add_action == "list"


def test_checkpoint_discovery():
    """Verify that model checkpoints are discovered and sorted correctly."""
    ckpts = get_available_checkpoints()
    default_ckpt = get_default_checkpoint_path()
    if not ckpts:
        assert default_ckpt is None
    else:
        assert len(ckpts) > 0
        assert default_ckpt is not None
        assert default_ckpt.is_file()
        assert default_ckpt.suffix == ".pt"


def test_stockfish_discovery():
    """Verify that Stockfish executable is discovered if present."""
    sf_path = get_default_stockfish_path()
    if sf_path is None:
        pytest.skip("Stockfish executable not found in tools/ or system PATH.")
    assert sf_path.is_file()
    assert "stockfish" in sf_path.name.lower()


def test_board_rendering():
    """Verify ASCII board table rendering."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    table = render_ascii_board(fen)
    assert table is not None
    assert "Chess Board" in table.title


def test_status_command():
    """Verify status command executes cleanly."""
    parser = build_parser()
    args = parser.parse_args(["status"])
    ret = run_status(args)
    assert ret == 0


def test_puzzle_list_command(tmp_path):
    """Verify add-puzzles list executes cleanly."""
    parser = build_parser()
    p_dir = tmp_path / "puzzles"
    p_dir.mkdir(exist_ok=True)
    args = parser.parse_args(["add-puzzles", "list", "--dir", str(p_dir)])
    ret = run_list_puzzles(args)
    assert ret == 0


def test_manual_puzzle_validation(tmp_path):
    """Verify that legal puzzles are accepted and illegal ones are rejected."""
    target_csv = tmp_path / "custom.csv"

    # Illegal move should fail
    illegal_args = argparse.Namespace(
        id="bad1",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        opponent_move="e2e5",  # Illegal for white to move to e5 on move 1
        solution="e7e5",
        rating=1200,
        themes="test",
        target_file=str(target_csv),
    )
    assert run_manual_add(illegal_args) == 1

    # Legal move should succeed
    legal_args = argparse.Namespace(
        id="good1",
        fen="rnbqkbnr/pppp1ppp/8/4p3/5P2/8/PPPPP1PP/RNBQKBNR w KQkq - 0 2",
        opponent_move="g2g4",
        solution="d8h4",
        rating=1300,
        themes="mate mateIn1",
        target_file=str(target_csv),
    )
    assert run_manual_add(legal_args) == 0
    assert target_csv.is_file()


def test_cli_supervised_parsing():
    """Verify supervised training CLI flags and subcommands."""
    parser = build_parser()

    # train --supervised
    args = parser.parse_args(["train", "--supervised"])
    assert args.command in ["train", "tr"]
    assert args.supervised is True
    assert args.num_blocks == 12
    assert args.num_filters == 128
    assert args.batch_size == 512

    # train --mode supervised with custom options
    args = parser.parse_args(
        [
            "train",
            "--mode",
            "supervised",
            "--epochs",
            "3",
            "--batch-size",
            "128",
            "--lr",
            "0.0005",
            "--steps",
            "500",
        ]
    )
    assert args.mode == "supervised"
    assert args.epochs == 3
    assert args.batch_size == 128
    assert args.lr == 0.0005
    assert args.max_steps == 500

    # dedicated supervised subcommand
    args = parser.parse_args(["supervised", "--epochs", "2", "--verify"])
    assert args.command in ["supervised", "train-supervised", "sl"]
    assert args.epochs == 2
    assert args.verify is True

    # train-supervised alias
    args = parser.parse_args(["train-supervised", "--batch-size", "64"])
    assert args.command in ["supervised", "train-supervised", "sl"]
    assert args.batch_size == 64

    # train --supervised --auto --epochs 10
    args = parser.parse_args(["train", "--supervised", "--auto", "--epochs", "10"])
    assert args.supervised is True
    assert args.auto is True
    assert args.epochs == 10
    assert args.batch_size == 512
    assert args.lr == 0.0002


def test_cli_supervised_validation():
    """Verify numeric and path validation in run_supervised_training."""
    from python.cli.commands_train import run_supervised_training

    # Invalid batch size
    bad_batch = argparse.Namespace(
        train_dir=None,
        val_dir=None,
        batch_size=-10,
        lr=1e-3,
        epochs=1,
        num_blocks=12,
        num_filters=128,
        checkpoint_dir=None,
        log_dir=None,
        device=None,
        verify=False,
        max_steps=None,
        warmup_steps=500,
        resume=None,
    )
    assert run_supervised_training(bad_batch) == 1

    # Invalid train_dir
    bad_dir = argparse.Namespace(
        train_dir="nonexistent_folder_xyz_123",
        val_dir=None,
        batch_size=256,
        lr=1e-3,
        epochs=1,
        num_blocks=12,
        num_filters=128,
        checkpoint_dir=None,
        log_dir=None,
        device=None,
        verify=False,
        max_steps=None,
        warmup_steps=500,
        resume=None,
    )
    assert run_supervised_training(bad_dir) == 1
