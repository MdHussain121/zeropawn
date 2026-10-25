"""End-to-End subprocess tests verifying strict UCI protocol isolation and zero ANSI escapes."""

import subprocess
import sys
import time

from python.cli.common import REPO_ROOT


def test_uci_subprocess_protocol_isolation():
    """Verify that launching 'chess uci' in a subprocess yields clean UCI protocol lines without ANSI escapes."""
    cmd = [sys.executable, str(REPO_ROOT / "cli.py"), "uci"]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    commands = [
        "uci\n",
        "isready\n",
        "ucinewgame\n",
        "position startpos moves e2e4 e7e5\n",
        "go nodes 20\n",
        "quit\n",
    ]

    stdout_lines = []
    stderr_lines = []

    try:
        # Send commands
        for c in commands:
            proc.stdin.write(c)
            proc.stdin.flush()
            time.sleep(0.05)

        stdout_data, stderr_data = proc.communicate(timeout=30.0)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0, f"Process exited with {proc.returncode}, stderr: {stderr_data}"

    # Verify no ANSI escape codes on stdout
    assert "\x1b[" not in stdout_data, f"ANSI escape sequence detected on stdout: {stdout_data}"
    assert "\x1b" not in stdout_data, f"Escape character detected on stdout: {stdout_data}"

    # Verify standard UCI protocol markers
    assert "id name ZeroPawn" in stdout_data
    assert "id author" in stdout_data
    assert "uciok" in stdout_data
    assert "readyok" in stdout_data
    assert "bestmove" in stdout_data
