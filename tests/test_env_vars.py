"""Unit tests for ZeroPawn environment variable fallback and precedence."""

import os
import pytest
import warnings

from python.common.env import get_env, set_env, _WARNED_VARS


def test_zeropawn_env_precedence(monkeypatch):
    """Verify ZEROPAWN_* takes precedence over legacy CHESSRL_* without warning."""
    monkeypatch.setenv("ZEROPAWN_OUTPUT_MODE", "rich")
    monkeypatch.setenv("CHESSRL_OUTPUT_MODE", "plain")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = get_env("OUTPUT_MODE")
        assert val == "rich"
        assert len(w) == 0


def test_legacy_chessrl_fallback_warning(monkeypatch):
    """Verify CHESSRL_* is read as fallback with a one-time DeprecationWarning."""
    monkeypatch.delenv("ZEROPAWN_DEVICE", raising=False)
    monkeypatch.setenv("CHESSRL_DEVICE", "cuda")
    _WARNED_VARS.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # First read triggers warning
        val1 = get_env("DEVICE")
        assert val1 == "cuda"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "CHESSRL_DEVICE" in str(w[0].message)

        # Second read does NOT trigger duplicate warning (one-time warning)
        val2 = get_env("DEVICE")
        assert val2 == "cuda"
        assert len(w) == 1


def test_set_env_backward_compatibility(monkeypatch):
    """Verify set_env populates both ZEROPAWN_* and CHESSRL_*."""
    set_env("CUSTOM_FLAG", "enabled")
    assert os.environ.get("ZEROPAWN_CUSTOM_FLAG") == "enabled"
    assert os.environ.get("CHESSRL_CUSTOM_FLAG") == "enabled"
