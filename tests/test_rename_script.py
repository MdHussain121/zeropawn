"""Unit tests for scripts/rename_to_zeropawn.py (Part 2).

Tests on a fixture tree:
- Every token variant (chessrl, ChessRL, CHESSRL_, chess-rl, chess_rl, "Chess RL", "chess rl", "RL Chess Engine")
- A "python-chess" line that must NOT change
- A "chess-library" line that must NOT change
- A binary file that must NOT change
- Idempotency: a second run that changes nothing
- Dry-run verification (no files modified)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest

from scripts.rename_to_zeropawn import (
    apply_text_replacements,
    REPLACEMENTS,
    is_protected_line,
)


def test_protected_line_preservation():
    """Verify lines with python-chess and chess-library are preserved."""
    assert is_protected_line('dependencies = ["python-chess>=1.999"]') is True
    assert is_protected_line('#include "chess-library/include/chess.hpp"') is True
    assert is_protected_line("import chessrl_cpp") is False


def test_token_replacements_all_variants():
    """Verify all token variants are properly replaced."""
    sample_text = """
    # Project ChessRL
    # Also variants: Chess-RL, Chess_RL, CHESS-RL, CHESS_RL, CHESSRL
    # And lowercase: chess-rl, chess_rl, chessrl
    namespace chessrl {
    class Evaluator {};
    }
    #include <chessrl/mcts.h>
    import chessrl_cpp
    import chessrl_cpp  # legacy shim
    os.environ["CHESSRL_DEVICE"] = "cuda"
    name = "chess-rl"
    logger.info("Starting Chess RL engine")
    print("Welcome to chess rl")
    title = "RL Chess Engine"
    # Third party dependencies:
    import chess  # python-chess
    # Domain classes:
    model = ChessNet()
    """
    new_text, changes = apply_text_replacements(sample_text, "test.py")

    assert "namespace zeropawn" in new_text
    assert "#include <zeropawn/mcts.h>" in new_text
    assert "import zeropawn_cpp\n" in new_text
    assert "import chessrl_cpp  # legacy shim" in new_text  # legacy shim preserved
    assert 'os.environ["ZEROPAWN_DEVICE"] = "cuda"' in new_text
    assert 'name = "zeropawn"' in new_text
    assert "ZeroPawn engine" in new_text
    assert "ZeroPawn Chess Engine" in new_text
    assert "Also variants: ZeroPawn, ZeroPawn, ZEROPAWN, ZEROPAWN, ZEROPAWN" in new_text
    assert "And lowercase: zeropawn, zeropawn, zeropawn" in new_text
    # Excluded items preserved:
    assert "python-chess" in new_text
    assert "ChessNet" in new_text
    assert len(changes) > 5


def test_fixture_tree_and_idempotency(tmp_path):
    """Test full execution on a fixture tree including dry-run, apply, binary files, and idempotency."""
    fixture_dir = tmp_path / "repo_fixture"
    fixture_dir.mkdir()

    # Create dummy files
    f1 = fixture_dir / "sample.py"
    f1.write_text(
        """
import chessrl_cpp
# ChessRL project
device = os.environ.get("CHESSRL_DEVICE", "cpu")
dependencies = ["python-chess>=1.999"]
""",
        encoding="utf-8",
    )

    f2 = fixture_dir / "header.hpp"
    f2.write_text(
        """
namespace chessrl {
void run();
}
""",
        encoding="utf-8",
    )

    # Binary file
    f_bin = fixture_dir / "dummy.pyd"
    f_bin.write_bytes(b"\x00\x01\x02\x03chessrl\x04\x05")

    # 1. Run dry-run
    cmd_dry = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts" / "rename_to_zeropawn.py"),
        "--repo-root",
        str(fixture_dir),
        "--allow-dirty",
        "--dry-run",
    ]
    res_dry = subprocess.run(cmd_dry, capture_output=True, text=True)
    assert res_dry.returncode == 0
    # Confirm no changes made yet
    assert "import chessrl_cpp" in f1.read_text(encoding="utf-8")

    # 2. Run apply
    cmd_apply = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts" / "rename_to_zeropawn.py"),
        "--repo-root",
        str(fixture_dir),
        "--allow-dirty",
        "--apply",
    ]
    res_apply = subprocess.run(cmd_apply, capture_output=True, text=True)
    assert res_apply.returncode == 0

    # Verify changes
    f1_content = f1.read_text(encoding="utf-8")
    assert "import zeropawn_cpp" in f1_content
    assert "ZeroPawn project" in f1_content
    assert "ZEROPAWN_DEVICE" in f1_content
    assert "python-chess" in f1_content  # Untouched

    f2_content = f2.read_text(encoding="utf-8")
    assert "namespace zeropawn" in f2_content

    # Binary untouched
    assert f_bin.read_bytes() == b"\x00\x01\x02\x03chessrl\x04\x05"

    # 3. Second run (idempotency test)
    res_second = subprocess.run(cmd_apply, capture_output=True, text=True)
    assert res_second.returncode == 0
    # Manifest should record 0 changes on second run
    manifest_file = fixture_dir / "docs" / "rename" / "manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["total_changes"] == 0
