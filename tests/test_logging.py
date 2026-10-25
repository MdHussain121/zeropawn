"""Unit tests for setup_logging in python/cli/common.py."""

import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import pytest

from python.cli.common import setup_logging


def test_setup_logging_handlers_and_levels(tmp_path, monkeypatch):
    test_log_dir = tmp_path / "test_logs"
    test_log_file = "test_zeropawn.log"

    # Capture stderr
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    # Clean existing handlers on root logger for isolation
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    root.handlers.clear()

    try:
        # First call
        logger1 = setup_logging(
            log_dir=test_log_dir,
            log_filename=test_log_file,
            level=logging.INFO,
            console_level=logging.WARNING,
        )

        # Check handlers
        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        stream_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        ]

        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.INFO
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.WARNING
        assert stream_handlers[0].stream is sys.stderr

        # Second call: must NOT add duplicate handlers
        logger2 = setup_logging(
            log_dir=test_log_dir,
            log_filename=test_log_file,
            level=logging.INFO,
            console_level=logging.WARNING,
        )

        file_handlers_after = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        stream_handlers_after = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers_after) == 1
        assert len(stream_handlers_after) == 1

        # Test message filtering
        logger1.info("This is an info message.")
        # stderr must NOT contain info message
        assert "This is an info message." not in stderr_buf.getvalue()

        logger1.warning("This is a warning message.")
        # stderr MUST contain warning message
        assert "This is a warning message." in stderr_buf.getvalue()

        # Check that warning message appears exactly once on stderr (no duplicates)
        count_warning = stderr_buf.getvalue().count("This is a warning message.")
        assert count_warning == 1

        # Check file output
        log_file_path = test_log_dir / test_log_file
        assert log_file_path.exists()
        file_content = log_file_path.read_text(encoding="utf-8")
        assert "This is an info message." in file_content
        assert "This is a warning message." in file_content
        assert file_content.count("This is a warning message.") == 1

    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)


def test_no_basicConfig_in_source_tree():
    """Regression: every entry point uses setup_logging; no basicConfig."""
    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for base in ("python", "scripts", "cli.py"):
        target = repo_root / base
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in files:
            if ".venv" in path.parts or "site-packages" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "logging.basicConfig(" in text:
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []


def test_third_party_logs_do_not_flood_terminal(tmp_path, monkeypatch):
    """INFO from third-party loggers (torch/matplotlib/urllib3) stays off stderr."""
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    root.handlers.clear()
    # Simulate the real hijack: `import torch` installs a DEBUG stderr
    # handler with propagate=False (verified on torch 2.13).
    torch_logger = logging.getLogger("torch")
    old_propagate = torch_logger.propagate
    old_third_handlers = list(torch_logger.handlers)
    hijack = logging.StreamHandler(sys.stderr)
    hijack.setLevel(logging.DEBUG)
    torch_logger.handlers.clear()
    torch_logger.addHandler(hijack)
    torch_logger.propagate = False
    try:
        setup_logging(log_dir=tmp_path / "logs", log_filename="thirdparty.log")
        # Hijack reclaimed: foreign handler gone, propagation restored.
        assert torch_logger.propagate is True
        assert hijack not in torch_logger.handlers
        for name in ("torch", "matplotlib", "urllib3", "PIL"):
            logging.getLogger(name).info("third-party info %s", name)
            logging.getLogger(name).warning("third-party warning %s", name)
        captured = stderr_buf.getvalue()
        assert "third-party info" not in captured
        for name in ("torch", "matplotlib", "urllib3", "PIL"):
            assert captured.count(f"third-party warning {name}") == 1
    finally:
        torch_logger.handlers.clear()
        torch_logger.handlers.extend(old_third_handlers)
        torch_logger.propagate = old_propagate
        root.handlers.clear()
        root.handlers.extend(old_handlers)


def test_basicConfig_after_setup_adds_no_duplicate_lines(tmp_path, monkeypatch):
    """Documents the run_phase4_rl ordering: basicConfig on a configured root is a no-op."""
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    root.handlers.clear()
    try:
        setup_logging(log_dir=tmp_path / "logs", log_filename="ordering.log")
        before = len(root.handlers)
        logging.basicConfig(
            level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
        )
        assert len(root.handlers) == before
        logging.getLogger("ordering.probe").warning("ordering probe msg")
        assert stderr_buf.getvalue().count("ordering probe msg") == 1
    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)
