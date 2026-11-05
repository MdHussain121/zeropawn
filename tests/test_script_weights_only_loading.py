"""SA3-02 regression tests: operator-script checkpoint loading must be tensor-only.

Covers the three operator scripts flagged by the 2026-11-05 pre-push audit
(``ZERO_PAWN_PRE_PUSH_AUDIT_20261105.md`` section 4, SA-3) whose ``torch.load``
calls lacked ``weights_only=True``:

- ``scripts/eval_puzzles_now.py``
- ``scripts/eval_connect4.py``
- ``scripts/verify_guardrails.py``

Contract under test:

1. Legitimate tensor-only checkpoints (primitives + state_dict) keep loading
   through each script's real ``torch.load(..., weights_only=True)`` call.
2. A payload carrying an arbitrary pickled object — or plain garbage bytes —
   is rejected by the weights_only unpickler instead of being deserialized
   unrestrictedly.

All fixtures are written to ``tmp_path``; nothing under ``models/`` or
``data/`` is ever written. The certified-artifact test is read-only and
skips when the artifact is absent.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from python.model.chess_net import ChessNet

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCALED_CKPT = _REPO_ROOT / "models" / "checkpoints_scaled" / "latest_checkpoint.pt"


class _ForeignPayload:
    """Arbitrary pickled object: never on the weights_only allowlist."""


# torch >= 2.6 raises UnpicklingError for off-allowlist globals; torch builds
# without weights_only support raise TypeError on the kwarg itself. Either way
# the contract is a hard failure, never a silent unrestricted deserialization.
_REJECTS = (pickle.UnpicklingError, TypeError, RuntimeError)


def _assert_fail_closed(excinfo: pytest.ExceptionInfo) -> None:
    message = str(excinfo.value)
    assert (
        "Weights only load failed" in message
        or "weights_only" in message
        or "Unsupported" in message
    ), f"unexpected rejection reason: {message[:200]}"


def _small_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """Write a legitimate tensor-only checkpoint (mirrors repo save shape)."""
    net = ChessNet(num_blocks=2, num_filters=32)
    state_dict = net.state_dict()
    torch.save(
        {"step": 7, "num_blocks": 2, "num_filters": 32, "model": state_dict},
        path,
    )
    return state_dict


# ---------------------------------------------------------------------------
# scripts/eval_puzzles_now.py
# ---------------------------------------------------------------------------


def test_eval_puzzles_now_loads_legitimate_checkpoint(tmp_path, monkeypatch, capsys):
    """A real tensor-only checkpoint file flows through the script's load path."""
    ckpt_path = tmp_path / "best_model.pt"
    state_dict = _small_checkpoint(ckpt_path)
    puzzles_csv = tmp_path / "puzzles.csv"
    puzzles_csv.write_text("id,fen,moves,rating\n", encoding="utf-8")

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_policy.return_value = {
        "solved": 5,
        "total": 10,
        "solve_rate": 0.5,
    }
    monkeypatch.setattr("scripts.eval_puzzles_now.PuzzleEvaluator", lambda **kwargs: mock_evaluator)
    monkeypatch.setattr("scripts.eval_puzzles_now.load_puzzles", lambda p: [MagicMock()] * 10)
    # Keep the run hermetic (log file + baseline lookup resolve from CWD).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_puzzles_now.py",
            "--weights",
            str(ckpt_path),
            "--device",
            "cpu",
            "--puzzles",
            str(puzzles_csv),
            "--baseline",
            "40.0",
            "--mcts-puzzles",
            "0",
        ],
    )

    from scripts.eval_puzzles_now import main

    main()  # must not raise, must not sys.exit

    out = capsys.readouterr().out
    assert "step=7" in out  # metadata came from the real weights_only load
    assert "SUCCESS" in out
    # The loaded state_dict reached the model: header says 2x32 (from ckpt).
    assert "net 2x32" in out.replace(" ", " ") or "2x32" in out
    assert state_dict  # fixture sanity


def test_eval_puzzles_now_rejects_arbitrary_pickled_payload(tmp_path, monkeypatch):
    """A checkpoint smuggling a pickled custom object fails closed."""
    evil_path = tmp_path / "evil.pt"
    torch.save(_ForeignPayload(), evil_path)
    puzzles_csv = tmp_path / "puzzles.csv"
    puzzles_csv.write_text("id,fen,moves,rating\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_puzzles_now.py",
            "--weights",
            str(evil_path),
            "--device",
            "cpu",
            "--puzzles",
            str(puzzles_csv),
        ],
    )

    from scripts.eval_puzzles_now import main

    with pytest.raises(_REJECTS) as excinfo:
        main()
    _assert_fail_closed(excinfo)


def test_eval_puzzles_now_rejects_garbage_bytes(tmp_path, monkeypatch):
    """A non-torch file is a hard load failure, not a silent fallback."""
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"definitely not a torch archive")
    puzzles_csv = tmp_path / "puzzles.csv"
    puzzles_csv.write_text("id,fen,moves,rating\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_puzzles_now.py",
            "--weights",
            str(garbage),
            "--device",
            "cpu",
            "--puzzles",
            str(puzzles_csv),
        ],
    )

    from scripts.eval_puzzles_now import main

    with pytest.raises(_REJECTS):
        main()


# ---------------------------------------------------------------------------
# scripts/eval_connect4.py
# ---------------------------------------------------------------------------


def test_eval_connect4_loads_legitimate_state_dict(tmp_path):
    """A real raw state_dict file loads through the script's weights_only path."""
    from python.model.connect4_net import Connect4Net
    from scripts.eval_connect4 import evaluate_vs_random

    model_path = tmp_path / "connect4_model.pt"
    torch.save(Connect4Net().state_dict(), model_path)

    wins, draws, losses = evaluate_vs_random(model_path=str(model_path), num_games=1, sims=1)
    assert wins + draws + losses == 1


def test_eval_connect4_rejects_arbitrary_pickled_payload(tmp_path):
    from scripts.eval_connect4 import evaluate_vs_random

    model_path = tmp_path / "evil.pt"
    torch.save(_ForeignPayload(), model_path)

    with pytest.raises(_REJECTS) as excinfo:
        evaluate_vs_random(model_path=str(model_path), num_games=1, sims=1)
    _assert_fail_closed(excinfo)


# ---------------------------------------------------------------------------
# scripts/verify_guardrails.py
# ---------------------------------------------------------------------------

_STUB_METRICS: Dict[str, float] = {
    "b_max_p_diff": 0.0,
    "b_top1_pct": 100.0,
    "b_max_v_diff": 0.0,
    "b_mean_v_diff": 0.0,
    "b_min_cosine": 1.0,
    "c_max_p_diff": 0.0,
    "c_top1_pct": 100.0,
    "c_max_v_diff": 0.0,
    "c_mean_v_diff": 0.0,
    "c_min_cosine": 1.0,
}


def test_verify_guardrails_loads_legitimate_checkpoint(tmp_path, monkeypatch):
    """The guardrail checkpoint loads (weights_only) and its weights reach the models."""
    import scripts.verify_guardrails as vg

    net = ChessNet(num_blocks=12, num_filters=128)
    fixture_sd = net.state_dict()
    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"model": fixture_sd}, ckpt_path)

    captured: Dict[str, Any] = {"calls": []}

    def _stub_metrics(name, test_positions, gpu_model, cpu_model, **kwargs):
        captured["calls"].append(name)
        captured.setdefault("cpu_models", []).append(cpu_model)
        return dict(_STUB_METRICS)

    monkeypatch.setattr(vg, "evaluate_dataset_metrics", _stub_metrics)
    monkeypatch.setattr(
        vg,
        "load_real_val_positions",
        lambda *a, **k: [(np.zeros((19, 8, 8), dtype=np.float32), [])] * 2,
    )

    result = vg.verify_output_differences(checkpoint_path=str(ckpt_path), num_positions=2)

    assert set(result) == {"set1", "set2"}
    assert captured["calls"] == [
        "Set 1 (10,000 Rollout Positions)",
        "Set 2 (2,000 Human Master Val Positions)",
    ]
    # The fixture weights really flowed through torch.load(weights_only=True)
    # into the models handed to the metrics stage.
    key = next(iter(fixture_sd))
    for model in captured["cpu_models"]:
        loaded = model.state_dict()[key]
        assert torch.equal(loaded, fixture_sd[key])


def test_verify_guardrails_rejects_arbitrary_pickled_payload(tmp_path):
    import scripts.verify_guardrails as vg

    evil_path = tmp_path / "evil.pt"
    torch.save(_ForeignPayload(), evil_path)

    with pytest.raises(_REJECTS) as excinfo:
        vg.verify_output_differences(checkpoint_path=str(evil_path), num_positions=2)
    _assert_fail_closed(excinfo)


# ---------------------------------------------------------------------------
# Certified artifacts (read-only, gated on presence)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SCALED_CKPT.is_file(), reason="certified scaled checkpoint not present")
def test_certified_scaled_checkpoint_loads_through_eval_puzzles_now(tmp_path, monkeypatch, capsys):
    """The certified artifact loads through the fixed script path (weights_only)."""
    puzzles_csv = tmp_path / "puzzles.csv"
    puzzles_csv.write_text("id,fen,moves,rating\n", encoding="utf-8")

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_policy.return_value = {
        "solved": 1,
        "total": 1,
        "solve_rate": 1.0,
    }
    monkeypatch.setattr("scripts.eval_puzzles_now.PuzzleEvaluator", lambda **kwargs: mock_evaluator)
    monkeypatch.setattr("scripts.eval_puzzles_now.load_puzzles", lambda p: [MagicMock()])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_puzzles_now.py",
            "--weights",
            str(_SCALED_CKPT),
            "--device",
            "cpu",
            "--puzzles",
            str(puzzles_csv),
            "--baseline",
            "40.0",
            "--mcts-puzzles",
            "0",
        ],
    )

    from scripts.eval_puzzles_now import main

    main()
    out = capsys.readouterr().out
    # Architecture metadata from the real 12x128 certified checkpoint.
    assert re.search(r"net 12x128", out) or "12x128" in out
    assert re.search(r"step=\d+", out)
    assert "SUCCESS" in out


@pytest.mark.skipif(not _SCALED_CKPT.is_file(), reason="certified scaled checkpoint not present")
def test_certified_scaled_checkpoint_is_weights_only_compatible():
    """Direct evidence: the certified artifact survives strict weights_only loading."""
    ck = torch.load(_SCALED_CKPT, map_location="cpu", weights_only=True)
    assert isinstance(ck, dict)
    state = ck.get("model")
    assert isinstance(state, dict) and state
    net = ChessNet(
        num_blocks=int(ck.get("num_blocks", 12)),
        num_filters=int(ck.get("num_filters", 128)),
    )
    net.load_state_dict(state)  # strict: full certified state must fit exactly
