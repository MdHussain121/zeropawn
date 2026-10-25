"""Unit tests verifying compatibility shims for chessrl_cpp and checkpoint loading."""

import sys
import warnings
import pytest
import torch
from pathlib import Path

from python.model.chess_net import ChessNet


def test_chessrl_cpp_shim_warning():
    """Verify importing chessrl_cpp issues a DeprecationWarning and provides full API."""
    if "chessrl_cpp" in sys.modules:
        del sys.modules["chessrl_cpp"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import python.chessrl_cpp as chessrl_cpp

        # Verify warning was emitted
        dep_warnings = [item for item in w if issubclass(item.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "chessrl_cpp" in str(dep_warnings[0].message)

        # Verify functionality via shim
        game = chessrl_cpp.ChessGame()
        assert len(game.legal_moves_uci()) == 20
        assert game.num_actions() == 4672
        assert "version" in dir(chessrl_cpp)


def test_old_checkpoint_loading_compatibility():
    """Verify legacy checkpoint loads perfectly without deserialization errors."""
    ckpt_path = Path("models/checkpoints_scaled/best_model.pt")
    if not ckpt_path.is_file():
        pytest.skip("models/checkpoints_scaled/best_model.pt not available in test environment")

    # Load state dict
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert ckpt["num_blocks"] == 12
    assert ckpt["num_filters"] == 128

    # Instantiate model and verify parameter loading
    net = ChessNet(num_blocks=12, num_filters=128)
    net.load_state_dict(ckpt["model"])
    net.eval()

    dummy = torch.zeros(1, 19, 8, 8, dtype=torch.float32)
    with torch.no_grad():
        p, v = net(dummy)
    assert p.shape == (1, 4672)
    assert v.shape == (1, 3)


def test_token_auditor_policy_allows_history_and_shims():
    """Verify that historical records, migration contracts, and shims are permitted."""
    from scripts.verify_parity import classify_token_occurrence

    # Permitted historical documents
    ok, reason = classify_token_occurrence(
        "docs/RENAME_CONTRACT_PATCH.md", 11, "Project Identifier: chess-rl / chessrl"
    )
    assert ok is True
    assert "historical" in reason

    ok, reason = classify_token_occurrence("docs/RENAME_LOG.md", 5, "Renaming chessrl to zeropawn")
    assert ok is True
    assert "historical" in reason

    # Permitted shims
    ok, reason = classify_token_occurrence("chessrl_cpp.py", 1, "import zeropawn_cpp")
    assert ok is True
    assert "compatibility shim" in reason

    ok, reason = classify_token_occurrence(
        "python/common/env.py", 10, 'os.environ.get("CHESSRL_DEVICE")'
    )
    assert ok is True
    assert "compatibility shim" in reason

    # Permitted third-party library references
    ok, reason = classify_token_occurrence(
        "pyproject.toml", 20, 'dependencies = ["python-chess>=1.999"]'
    )
    assert ok is True
    assert "third-party" in reason


def test_token_auditor_policy_rejects_active_code_leftovers():
    """Verify that genuine active code, config, or launcher leftovers are rejected."""
    from scripts.verify_parity import classify_token_occurrence

    # Active python source file with accidental legacy token
    ok, reason = classify_token_occurrence(
        "python/model/chess_net.py", 42, "from chessrl import something"
    )
    assert ok is False
    assert "unauthorized legacy token in active file" in reason

    # Active CLI file
    ok, reason = classify_token_occurrence("python/cli/main.py", 10, "import chessrl")
    assert ok is False
    assert "unauthorized legacy token in active file" in reason

    # Active user documentation
    ok, reason = classify_token_occurrence(
        "README.md", 88, "git clone https://github.com/org/chessrl.git"
    )
    assert ok is False
    assert "unauthorized legacy token in active file" in reason

    # Active configuration
    ok, reason = classify_token_occurrence(
        "configs/presets.json", 15, '"path": "models/chessrl/model.pt"'
    )
    assert ok is False
    assert "unauthorized legacy token in active file" in reason

    # Active C++ source
    ok, reason = classify_token_occurrence("cpp/src/mcts.cpp", 10, "namespace chessrl {")
    assert ok is False
    assert "unauthorized legacy token in active file" in reason
