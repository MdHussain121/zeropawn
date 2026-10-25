"""Unit and integration tests for CLI non-TTY resilience and path containment."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.cli.common import (
    REPO_ROOT,
    _is_subpath_of,
    resolve_path,
    safe_resolve,
    wait_for_enter,
)
from python.cli.commands_puzzles import run_solve_puzzles


def test_wait_for_enter_non_tty_returns_immediately(monkeypatch):
    """Verify wait_for_enter returns immediately when sys.stdin is not a TTY."""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    mock_input = MagicMock()
    monkeypatch.setattr("rich.console.Console.input", mock_input)

    wait_for_enter("Press [Enter] to continue...")
    mock_input.assert_not_called()


def test_wait_for_enter_tty_prompts_user(monkeypatch):
    """Verify wait_for_enter calls console.input when sys.stdin is a TTY."""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    mock_input = MagicMock(return_value="")
    monkeypatch.setattr("rich.console.Console.input", mock_input)

    wait_for_enter("Custom prompt...")
    assert mock_input.call_count == 1
    assert mock_input.call_args[0][0] == "Custom prompt..."


def test_wait_for_enter_handles_eof_and_interrupt(monkeypatch):
    """Verify wait_for_enter catches EOFError and KeyboardInterrupt without bubbling up."""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    # Test EOFError
    monkeypatch.setattr("rich.console.Console.input", MagicMock(side_effect=EOFError))
    wait_for_enter()  # should not raise

    # Test KeyboardInterrupt
    monkeypatch.setattr("rich.console.Console.input", MagicMock(side_effect=KeyboardInterrupt))
    wait_for_enter()  # should not raise


def test_human_guess_fails_fast_in_non_tty(monkeypatch, capsys):
    """Verify run_solve_puzzles rejects human_guess=True if stdin is not a TTY."""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", mock_stdin)

    args = argparse.Namespace(
        human_guess=True,
        puzzles="data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        start_index=0,
        checkpoint=None,
        device="cpu",
        mcts_sims=10,
    )

    exit_code = run_solve_puzzles(args)
    assert exit_code == 1


def test_path_containment_strict_bounds():
    """Verify strict subpath containment and rejection of prefix/parent traversal escapes."""
    # Strict containment under parent
    assert _is_subpath_of(REPO_ROOT / "python" / "cli", REPO_ROOT) is True
    assert _is_subpath_of(REPO_ROOT, REPO_ROOT) is True

    # Sibling directory is NOT a subpath of REPO_ROOT
    sibling = REPO_ROOT.parent / "search-study"
    assert _is_subpath_of(sibling, REPO_ROOT) is False

    # Prefix collision (e.g., chess-rl-malicious is not under chess-rl)
    fake_prefix = REPO_ROOT.parent / f"{REPO_ROOT.name}-fake"
    assert _is_subpath_of(fake_prefix, REPO_ROOT) is False

    # Grandparent escape attempt
    outside = Path("C:/Windows")
    assert _is_subpath_of(outside, REPO_ROOT) is False

    # Traversal rejection in safe_resolve
    with pytest.raises(ValueError, match="escapes allowed directories|outside allowed"):
        safe_resolve("../../Windows/System32", base=REPO_ROOT, allowed_roots=[REPO_ROOT])
