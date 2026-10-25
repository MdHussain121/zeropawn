"""Tests for UCI Protocol Engine implementation."""

import io
import sys

from python.uci.uci_engine import UCIEngine


def test_uci_protocol_handshake():
    """Verify standard UCI handshake (uci -> uciok, isready -> readyok)."""
    engine = UCIEngine()

    # Capture stdout
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_buf
        assert engine.handle_line("uci") is True
        output = stdout_buf.getvalue()
        assert "id name ZeroPawn" in output
        assert "option name Hash" in output
        assert "option name Simulations" in output
        assert "uciok" in output

        stdout_buf.truncate(0)
        stdout_buf.seek(0)
        assert engine.handle_line("isready") is True
        assert "readyok" in stdout_buf.getvalue()
    finally:
        sys.stdout = old_stdout


def test_uci_position_and_go():
    """Verify setting positions and searching with nodes limit."""
    engine = UCIEngine()

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_buf
        # Initialize new game and play 2 plies
        assert engine.handle_line("ucinewgame") is True
        assert engine.handle_line("position startpos moves e2e4 e7e5") is True
        assert engine.game.get_fen().split()[1] == "w"  # White to move after e2e4 e7e5

        # Search with a small node limit
        stdout_buf.truncate(0)
        stdout_buf.seek(0)
        assert engine.handle_line("go nodes 15") is True
        output = stdout_buf.getvalue()
        assert "bestmove" in output
        assert "info depth" in output

        # Test quit
        assert engine.handle_line("quit") is False
    finally:
        sys.stdout = old_stdout


def test_uci_batched_search_options():
    """Verify SearchMode and BatchSize options and batched search execution."""
    engine = UCIEngine()

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_buf
        engine.handle_line("uci")
        out = stdout_buf.getvalue()
        assert "option name SearchMode type combo default serial var serial var batched" in out
        assert "option name BatchSize type spin default 1 min 1 max 64" in out

        # Configure batched search
        assert engine.handle_line("setoption name SearchMode value batched") is True
        assert engine.search_mode == "batched"
        assert engine.handle_line("setoption name BatchSize value 8") is True
        assert engine.batch_size == 8

        # Run batched search
        stdout_buf.truncate(0)
        stdout_buf.seek(0)
        assert engine.handle_line("ucinewgame") is True
        assert engine.handle_line("position startpos") is True
        assert engine.handle_line("go nodes 20") is True
        go_out = stdout_buf.getvalue()
        assert "bestmove" in go_out
        assert "info depth" in go_out
    finally:
        sys.stdout = old_stdout


def test_uci_auto_batch_size():
    """Verify UCIEngine supports batch_size='auto' and setoption BatchSize value auto."""
    engine = UCIEngine(batch_size="auto")
    assert engine.batch_size in (4, 16, 24, 32, 64)

    assert engine.handle_line("setoption name BatchSize value 32") is True
    assert engine.batch_size == 32

    assert engine.handle_line("setoption name BatchSize value auto") is True
    assert engine.batch_size in (4, 16, 24, 32, 64)
