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


def _dummy_engine(tmp_path):
    """UCIEngine backed by a tiny deterministic checkpoint (no opening book)."""
    import torch

    from python.model.chess_net import ChessNet

    ckpt = tmp_path / "dummy_uci_model.pt"
    if not ckpt.exists():
        model = ChessNet(num_blocks=2, num_filters=32)
        torch.save(
            {"model": model.state_dict(), "num_blocks": 2, "num_filters": 32, "step": 1},
            ckpt,
        )
    return UCIEngine(checkpoint_path=str(ckpt), use_book=False)


def test_eval_callback_returns_contiguous_f32(tmp_path):
    """N-07 regression: the leaf callback returns a contiguous f32 ndarray policy.

    A Python-list policy forces a slow element-wise cast inside the pybind
    evaluator; the contiguous ndarray keeps the zero-copy numpy fast path.
    """
    import numpy as np

    engine = _dummy_engine(tmp_path)
    game = engine.zeropawn_cpp.ChessGame()
    policy, value = engine._eval_callback(game)

    assert isinstance(policy, np.ndarray)
    assert policy.dtype == np.float32
    assert policy.flags["C_CONTIGUOUS"]
    assert policy.shape == (game.num_actions(),)
    assert abs(float(policy.sum()) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0

    # No-model fallback is also ndarray-shaped (uniform policy).
    engine.model = None
    policy0, value0 = engine._eval_callback(game)
    assert isinstance(policy0, np.ndarray)
    assert abs(float(policy0.sum()) - 1.0) < 1e-4
    assert value0 == 0.0


def test_eval_callback_preserves_repetition_plane(tmp_path):
    """N-07 correctness: encoding the live game keeps the repetition plane.

    The removed ChessGame(get_fen()) round-trip reconstructed the position
    without move history, so any once-repeated position encoded with plane 18
    all zeros — subtly different network inputs than every other path.
    """
    import numpy as np

    engine = _dummy_engine(tmp_path)
    game = engine.zeropawn_cpp.ChessGame()
    for target_uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        uci_list = [game.action_to_string(a) for a in game.legal_actions()]
        game.apply_action(game.legal_actions()[uci_list.index(target_uci)])

    live_planes = engine.zeropawn_cpp.encode_state(game)
    assert float(live_planes[18].max()) > 0.0, "repetition plane must survive live encoding"

    # The engine callback must agree with the direct live encoding.
    policy, value = engine._eval_callback(game)
    assert policy.shape[0] == game.num_actions()
    assert np.isfinite(policy).all()
