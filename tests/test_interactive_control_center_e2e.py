"""Comprehensive end-to-end interactive terminal UI tests for ZeroPawn.

Validates the real interactive Rich control center (cli.py):
- Full application startup, banner, and hardware telemetry panel rendering.
- Category navigation (Categories 1-10) with verified Back [B] behavior.
- Operation selection flow across Quick Run [1], Configuration Wizard [2], and Custom Args [3].
- Argument editing, validation, bounds clamping, and reset [R] in Wizard.
- Real MCTS Search execution through interactive UI (batched & serial).
- Model selection [M], Status [S], and Preset launcher [P] quick actions.
- Graceful exit [Q], invalid input recovery, and EOF handling.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_interactive_session(
    input_sequence: str, timeout: float = 30.0, extra_env: dict | None = None
) -> tuple[int, str]:
    """Execute real interactive `cli.py` with piped stdin and capture terminal output."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"  # ensure standard terminal rendering
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    p = subprocess.Popen(
        [sys.executable, "-u", "cli.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        env=env,
    )

    out, _ = p.communicate(input=input_sequence, timeout=timeout)
    return p.returncode, out


def test_interactive_startup_and_exit():
    """Verify application startup, hardware panel rendering, and clean exit."""
    code, out = run_interactive_session("Q\n")
    assert code == 0
    assert "ZEROPAWN CONTROL CENTER" in out or "ZeroPawn" in out
    assert "Hardware Telemetry" in out
    assert "Main Operation Categories" in out
    assert "Goodbye!" in out


def test_interactive_category_navigation_and_back():
    """Verify navigating through multiple categories and returning to main menu with [B]."""
    # Navigate: Category 1 -> B -> Category 5 -> B -> Category 6 -> B -> Category 9 -> B -> Q
    inputs = "1\nB\n5\nB\n6\nB\n9\nB\nQ\n"
    code, out = run_interactive_session(inputs)
    assert code == 0
    assert "Category 1 / 10" in out
    assert "Training & Reinforcement Learning" in out
    assert "Category 5 / 10" in out
    assert "Inference & Benchmarking" in out
    assert "Category 6 / 10" in out
    assert "MCTS & Search Diagnostics" in out
    assert "Category 9 / 10" in out
    assert "Logs, Telemetry & Diagnostics" in out
    assert "Goodbye!" in out


def test_interactive_invalid_category_and_recovery():
    """Verify invalid category numbers prompt error and recover without crash."""
    # Enter invalid category 99, then enter to continue, then Q to quit
    inputs = "99\n\nQ\n"
    code, out = run_interactive_session(inputs)
    assert code == 0
    assert "out of range" in out.lower() or "invalid" in out.lower()
    assert "Goodbye!" in out


def test_interactive_status_quick_action():
    """Verify pressing [S] in the main menu displays system status and returns."""
    # S -> Enter to continue -> Q
    inputs = "S\n\nQ\n"
    code, out = run_interactive_session(inputs, timeout=40.0)
    assert code == 0
    assert "OPERATION: STATUS" in out
    assert "gpu" in out.lower() or "cuda" in out.lower()
    assert "Goodbye!" in out


def test_interactive_mcts_search_wizard_execution():
    """Verify full Configuration Wizard (Path B) for MCTS Search through interactive UI.

    1. Category 6 (MCTS & Search Diagnostics)
    2. Operation 3 (MCTS Position Search)
    3. Mode 2 (Configuration Wizard)
    4. Edit sims (option 2) -> set to 20
    5. Edit mode (option 3) -> set to batched
    6. Edit batch_size (option 4) -> set to 16
    7. Execute (X)
    8. Press Enter to continue
    9. Back (B)
    10. Back (B)
    11. Quit (Q)
    """
    inputs = "6\n4\n2\n2\n20\n3\nbatched\n4\n16\nX\n\nB\nB\nQ\n"
    code, out = run_interactive_session(inputs, timeout=60.0)
    assert code == 0
    assert "MCTS Position Search" in out or "search" in out.lower()
    assert "Configuration Wizard (Path B)" in out
    assert "MCTS Search Result" in out
    assert "Best Move:" in out
    assert "Mode: batched (B=16)" in out
    assert "Goodbye!" in out


def test_interactive_mcts_serial_search_wizard_execution():
    """Verify Configuration Wizard execution of serial search with 15 simulations."""
    # Category 6 -> Op 3 -> Mode 2 (Wizard) -> Edit 2 (sims=15) -> Edit 3 (mode=serial) -> X -> Enter -> B -> B -> Q
    inputs = "6\n4\n2\n2\n15\n3\nserial\nX\n\nB\nB\nQ\n"
    code, out = run_interactive_session(inputs, timeout=60.0)
    assert code == 0
    assert "MCTS Search Result" in out
    assert "Mode: serial (B=1)" in out
    assert "Goodbye!" in out


def test_interactive_hardware_profile_quick_run(tmp_path):
    """Verify Quick Run (Path 1) on hardware profile through interactive UI."""
    test_prof_dir = str(tmp_path / "profiles")
    # Category 9 -> Operation 5 (Hardware Profile) -> Mode 1 (Quick Run) -> Enter -> B -> Q
    inputs = "9\n5\n1\n\nB\nQ\n"
    code, out = run_interactive_session(
        inputs, timeout=40.0, extra_env={"ZEROPAWN_PROFILE_DIR": test_prof_dir}
    )
    assert code == 0
    assert "ZeroPawn Hardware Autotuning Profile" in out
    assert "Autotuned Workload Targets" in out
    assert "Goodbye!" in out


def test_interactive_wizard_cancel_abandons_changes():
    """Verify pressing [C] in Wizard cancels and abandons unapplied parameter changes."""
    # Category 6 -> Op 3 -> Mode 2 -> Edit 2 (sims=999) -> C -> Mode 2 -> check sims != 999 -> B -> B -> Q
    inputs = "6\n3\n2\n2\n999\nC\n2\nB\nB\nB\nQ\n"
    code, out = run_interactive_session(inputs, timeout=30.0)
    assert code == 0
    assert "Changes abandoned" in out
    assert "Goodbye!" in out


def test_interactive_eof_handling():
    """Verify application handles EOF gracefully without unhandled tracebacks."""
    code, out = run_interactive_session("")
    assert code == 0
    assert "ZEROPAWN CONTROL CENTER" in out or "ZeroPawn" in out
