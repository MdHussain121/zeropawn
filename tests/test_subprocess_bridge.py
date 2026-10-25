"""Tests for Subprocess Output Bridge and Concurrent Stream Consumption."""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
from python.cli.output_context import OutputContext, OutputMode
from python.cli.runner import SubprocessOutputBridge, execute_subcommand, terminate_process_tree


def test_subprocess_bridge_exit_code_and_capture():
    """Verify exit code propagation and output capture for a quick command."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('hello stdout'); sys.stderr.write('hello stderr\\n'); sys.exit(7)",
    ]
    bridge = SubprocessOutputBridge(
        cmd=cmd,
        op_name="test_exit_code",
        output_context=OutputContext(OutputMode.PLAIN),
        buffer_lines=50,
    )
    code = bridge.run()
    assert code == 7

    history = bridge.get_history()
    assert any("hello stdout" in line for line in history)
    assert any("hello stderr" in line for line in history)


def test_subprocess_bridge_high_volume_deadlock_freedom():
    """Verify child producing large volume on stdout and stderr does not deadlock pipes."""
    # Write 10,000 lines to stdout and stderr simultaneously
    code_snippet = (
        "import sys\n"
        "for i in range(10000):\n"
        "    sys.stdout.write(f'stdout line {i}\\n')\n"
        "    sys.stderr.write(f'stderr line {i}\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.flush()\n"
    )
    cmd = [sys.executable, "-c", code_snippet]
    bridge = SubprocessOutputBridge(
        cmd=cmd,
        op_name="test_high_volume",
        output_context=OutputContext(OutputMode.PLAIN),
        buffer_lines=100,  # Circular buffer must not exceed 100 items
    )
    t0 = time.time()
    code = bridge.run()
    duration = time.time() - t0

    assert code == 0
    # Must finish quickly without stalling on OS pipe buffer limit
    assert duration < 10.0

    history = bridge.get_history()
    assert len(history) <= 100
    # Last lines should be from the end of the loop
    assert any("line 999" in line for line in history)


def test_subprocess_bridge_disk_logging():
    """Verify stdout and stderr lines are written to log file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_bridge.log"
        cmd = [
            sys.executable,
            "-c",
            "import sys; print('disk line 1'); sys.stderr.write('disk line 2\\n')",
        ]
        bridge = SubprocessOutputBridge(
            cmd=cmd,
            op_name="test_disk",
            log_path=str(log_file),
            output_context=OutputContext(OutputMode.PLAIN),
        )
        code = bridge.run()
        assert code == 0
        assert log_file.exists()
        log_content = log_file.read_text(encoding="utf-8")
        assert "disk line 1" in log_content
        assert "disk line 2" in log_content


def test_subprocess_bridge_cancellation():
    """Verify process cancellation safely terminates the running process tree."""
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    bridge = SubprocessOutputBridge(
        cmd=cmd,
        op_name="test_cancellation",
        output_context=OutputContext(OutputMode.PLAIN),
    )

    import threading

    worker = threading.Thread(target=bridge.run)
    worker.start()

    time.sleep(0.5)
    assert bridge._proc is not None
    assert bridge._proc.poll() is None

    # Cancel process
    bridge.cancel()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert bridge._proc.poll() is not None
