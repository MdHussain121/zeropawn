"""Tests for Unified OutputContext and OutputMode behavior."""

import io
import os
import sys
import threading
from unittest import mock

import pytest
from python.cli.output_context import (
    OutputContext,
    OutputMode,
    get_output_mode,
    set_output_mode,
)
from python.common.progress import emit_error, emit_report


def test_output_mode_defaults_and_env():
    """Verify default mode resolution and ZEROPAWN_OUTPUT_MODE environment variable."""
    with mock.patch.dict(os.environ, {}, clear=True):
        ctx = OutputContext()
        assert ctx.mode == OutputMode.PLAIN
        assert ctx.is_plain()
        assert not ctx.is_rich()
        assert not ctx.is_protocol()

    with mock.patch.dict(os.environ, {"ZEROPAWN_OUTPUT_MODE": "rich"}):
        ctx = OutputContext()
        assert ctx.mode == OutputMode.RICH
        assert ctx.is_rich()

    with mock.patch.dict(os.environ, {"ZEROPAWN_OUTPUT_MODE": "protocol"}):
        ctx = OutputContext()
        assert ctx.mode == OutputMode.PROTOCOL
        assert ctx.is_protocol()


def test_output_context_thread_scoping():
    """Verify thread-local isolation of output contexts."""
    main_ctx = OutputContext(OutputMode.PLAIN)
    OutputContext.set_current(main_ctx)
    assert OutputContext.get_current().mode == OutputMode.PLAIN

    worker_modes = []

    def worker():
        worker_ctx = OutputContext(OutputMode.RICH)
        OutputContext.set_current(worker_ctx)
        worker_modes.append(OutputContext.get_current().mode)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert worker_modes == [OutputMode.RICH]
    assert OutputContext.get_current().mode == OutputMode.PLAIN


def test_protocol_mode_stdout_isolation():
    """Verify protocol mode ensures stdout is never polluted with ANSI escapes or reports."""
    proto_ctx = OutputContext(OutputMode.PROTOCOL)
    OutputContext.set_current(proto_ctx)

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    with mock.patch("sys.stdout", out_buf), mock.patch("sys.stderr", err_buf):
        # 1. Normal protocol line
        proto_ctx.emit_line("bestmove e2e4")
        # 2. Report emitted by progress layer
        emit_report("Progress report: iteration 10")
        # 3. Error emitted by progress layer
        emit_error("Warning: buffer low")

    stdout_val = out_buf.getvalue()
    stderr_val = err_buf.getvalue()

    # stdout must contain ONLY the protocol line, strictly clean
    assert stdout_val.strip() == "bestmove e2e4"
    assert "\x1b[" not in stdout_val

    # reports and errors must be routed to stderr to protect protocol stream
    assert "Progress report: iteration 10" in stderr_val
    assert "Warning: buffer low" in stderr_val


def test_plain_mode_clean_emission():
    """Verify plain mode emits cp437-safe plain-text to stdout without Rich wraps."""
    plain_ctx = OutputContext(OutputMode.PLAIN)
    OutputContext.set_current(plain_ctx)

    out_buf = io.StringIO()
    with mock.patch("sys.stdout", out_buf):
        emit_report("Status: OK")

    assert out_buf.getvalue().strip() == "Status: OK"
    assert "\x1b[" not in out_buf.getvalue()


def test_set_output_mode_helper():
    """Verify set_output_mode modifies active context and environment."""
    set_output_mode(OutputMode.RICH)
    assert OutputContext.get_current().mode == OutputMode.RICH
    assert os.environ.get("ZEROPAWN_OUTPUT_MODE") == "rich"

    set_output_mode(OutputMode.PLAIN)
    assert OutputContext.get_current().mode == OutputMode.PLAIN
    assert os.environ.get("ZEROPAWN_OUTPUT_MODE") == "plain"
