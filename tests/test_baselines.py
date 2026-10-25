"""Unit tests for puzzle baseline registry and per-model baseline management."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from python.eval.baselines import (
    DEFAULT_BASELINE,
    get_baseline_for_model,
    set_baseline_for_model,
    load_baselines,
    get_baselines_file_path,
)
from python.cli.main import build_parser


import logging


def test_default_baseline():
    """Verify default baseline is None when model is not recognized."""
    rate = get_baseline_for_model("unknown_model_12345.pt")
    assert rate == DEFAULT_BASELINE
    assert rate is None


def test_set_and_get_baseline_for_model(tmp_path, monkeypatch):
    """Verify baseline can be updated and retrieved for a specific model."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    # Initially empty
    data = load_baselines()
    assert data == {}
    assert get_baseline_for_model("best_model.pt") is None

    # Update baseline for a new checkpoint
    ckpt_name = "test_dummy_model_for_baseline.pt"
    set_baseline_for_model(ckpt_name, 41.25, mcts_rate=45.0, total_puzzles=500)

    # Verify retrieval by filename
    retrieved = get_baseline_for_model(ckpt_name)
    assert retrieved == 41.25

    # Verify retrieval by path
    full_path = Path("models/checkpoints_rl") / ckpt_name
    assert get_baseline_for_model(full_path) == 41.25

    # Verify retrieval by stem (without .pt)
    assert get_baseline_for_model("test_dummy_model_for_baseline") == 41.25

    # Verify disk persistence
    assert fake_baselines.is_file()
    with open(fake_baselines, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert ckpt_name in disk_data
    assert disk_data[ckpt_name]["policy_rate"] == 41.25
    assert disk_data[ckpt_name]["mcts_rate"] == 45.0
    assert disk_data[ckpt_name]["total_puzzles"] == 500


def test_baseline_sha256_binding_and_renaming(tmp_path, monkeypatch):
    """Verify baseline binds to checkpoint SHA-256 and survives file renaming."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    # Create a real checkpoint file
    ckpt_file = tmp_path / "model_alpha.pt"
    ckpt_file.write_bytes(b"dummy checkpoint content alpha")

    set_baseline_for_model(ckpt_file, 48.5, mcts_rate=52.0, total_puzzles=1000)

    # Retrieve by file path
    assert get_baseline_for_model(ckpt_file) == 48.5

    # Rename file: hash match still finds the baseline!
    renamed_file = tmp_path / "model_renamed.pt"
    ckpt_file.rename(renamed_file)
    assert get_baseline_for_model(renamed_file) == 48.5

    # If content changes, hash changes -> baseline not returned
    renamed_file.write_bytes(b"modified checkpoint content beta")
    assert get_baseline_for_model(renamed_file) is None


def test_baseline_hash_mismatch_warning(tmp_path, monkeypatch, caplog):
    """Verify that if a file with same name has different hash than recorded, it is rejected."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    ckpt_file = tmp_path / "special_model.pt"
    ckpt_file.write_bytes(b"initial content")
    set_baseline_for_model(ckpt_file, 50.0)

    # Modify file content while keeping the same name
    ckpt_file.write_bytes(b"tampered content")

    # Lookup by path should reject because recorded sha256 does not match actual sha256
    with caplog.at_level(logging.WARNING):
        rate = get_baseline_for_model(ckpt_file)
    assert rate is None
    assert "does not match recorded" in caplog.text


def test_baseline_atomic_write(tmp_path, monkeypatch):
    """Verify baseline write is atomic and produces valid JSON."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    set_baseline_for_model("ckpt1.pt", 30.0)
    set_baseline_for_model("ckpt2.pt", 40.0)

    with open(fake_baselines, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "ckpt1.pt" in data
    assert "ckpt2.pt" in data


def test_cli_puzzles_eval_args():
    """Verify that --baseline, --update-baseline, --mcts-puzzles, and --mcts-sims parse properly."""
    parser = build_parser()

    # Test custom MCTS sims and puzzles, plus custom baseline
    args = parser.parse_args(
        [
            "puzzles",
            "eval",
            "--mcts-puzzles",
            "20",
            "--mcts-sims",
            "100",
            "--baseline",
            "42.5",
            "--update-baseline",
        ]
    )
    assert args.command in ["puzzles", "puzzle", "p"]
    assert args.mcts_puzzles == 20
    assert args.mcts_sims == 100
    assert args.baseline == 42.5
    assert args.update_baseline is True

    # Test defaults
    args_default = parser.parse_args(["puzzles", "eval"])
    assert args_default.mcts_puzzles == 50
    assert args_default.mcts_sims == 50
    assert args_default.baseline is None
    assert args_default.update_baseline is False


def test_puzzles_eval_none_baseline_formatting(monkeypatch, capsys):
    """Verify evaluation reports 'vs_baseline N/A' cleanly when no baseline exists."""
    from unittest.mock import MagicMock
    from python.cli.commands_puzzles import run_eval_puzzles
    import argparse

    monkeypatch.setattr("python.cli.commands_puzzles.get_baseline_for_model", lambda p: None)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_policy.return_value = {"solved": 5, "total": 10, "solve_rate": 0.5}
    mock_evaluator.evaluate_mcts.return_value = {"solved": 6, "total": 10, "solve_rate": 0.6}
    monkeypatch.setattr("python.eval.puzzles.PuzzleEvaluator", lambda **kwargs: mock_evaluator)
    monkeypatch.setattr("python.eval.puzzles.load_puzzles", lambda p: [MagicMock()] * 10)
    monkeypatch.setattr(
        "python.cli.commands_puzzles.load_engine_model",
        lambda p, device=None: (MagicMock(), {"step": 1000}),
    )

    args = argparse.Namespace(
        checkpoint="models/checkpoints_scaled/best_model.pt",
        device="cpu",
        puzzles_file="data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        mcts_puzzles=10,
        mcts_sims=10,
        search_mode="serial",
        batch_size=16,
        update_baseline=False,
        baseline=None,
        min_rating=None,
        max_rating=None,
        limit=None,
        batch_size_eval=128,
        threads=1,
    )

    exit_code = run_eval_puzzles(args)
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "vs_baseline N/A (no baseline recorded)" in captured


def test_eval_puzzles_now_none_baseline_formatting(monkeypatch, capsys):
    """Verify scripts/eval_puzzles_now.py reports 'vs_baseline N/A' cleanly when no baseline exists."""
    from unittest.mock import MagicMock
    from scripts.eval_puzzles_now import main
    import sys

    monkeypatch.setattr("scripts.eval_puzzles_now.get_baseline_for_model", lambda p: None)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_policy.return_value = {"solved": 5, "total": 10, "solve_rate": 0.5}
    mock_evaluator.evaluate_mcts.return_value = {"solved": 6, "total": 10, "solve_rate": 0.6}
    monkeypatch.setattr("scripts.eval_puzzles_now.PuzzleEvaluator", lambda **kwargs: mock_evaluator)
    monkeypatch.setattr("scripts.eval_puzzles_now.load_puzzles", lambda p: [MagicMock()] * 10)
    monkeypatch.setattr(
        "torch.load",
        lambda p, map_location=None: {
            "step": 1000,
            "num_blocks": 2,
            "num_filters": 32,
            "model": {},
        },
    )
    monkeypatch.setattr("python.model.chess_net.ChessNet.load_state_dict", lambda self, sd: None)
    monkeypatch.setattr("python.model.chess_net.ChessNet.to", lambda self, dev: self)

    test_args = [
        "eval_puzzles_now.py",
        "--weights",
        "models/checkpoints_scaled/best_model.pt",
        "--device",
        "cpu",
        "--puzzles",
        "data/puzzles/lichess_puzzles_1000_1200_1600.csv",
        "--mcts-puzzles",
        "10",
        "--mcts-sims",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()
    captured = capsys.readouterr().out
    assert "vs_baseline N/A (no baseline recorded)" in captured


def test_legacy_unhashed_record_rejected_for_physical_file(tmp_path, monkeypatch, caplog):
    """Verify legacy records without a SHA-256 are rejected when a physical file exists."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    fake_baselines.write_text(
        json.dumps(
            {"legacy_model.pt": {"policy_rate": 35.60, "mcts_rate": 40.0, "total_puzzles": 500}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    ckpt_file = tmp_path / "legacy_model.pt"
    ckpt_file.write_bytes(b"some physical checkpoint bytes")

    with caplog.at_level(logging.WARNING):
        rate = get_baseline_for_model(ckpt_file)

    assert rate is None
    assert "legacy record has no recorded SHA-256 hash" in caplog.text
    assert "Re-benchmark with --update-baseline" in caplog.text


def test_legacy_scalar_record_rejected_for_physical_file(tmp_path, monkeypatch, caplog):
    """Verify legacy raw scalar records are rejected when a physical file exists."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    fake_baselines.write_text(
        json.dumps({"scalar_model.pt": 38.2}),
        encoding="utf-8",
    )
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    ckpt_file = tmp_path / "scalar_model.pt"
    ckpt_file.write_bytes(b"scalar checkpoint content")

    with caplog.at_level(logging.WARNING):
        rate = get_baseline_for_model(ckpt_file)

    assert rate is None
    assert "legacy record has no recorded SHA-256 hash" in caplog.text


def test_legacy_unhashed_stem_match_rejected_for_physical_file(tmp_path, monkeypatch, caplog):
    """Verify unhashed records matched via stem are rejected when a physical file exists."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    fake_baselines.write_text(
        json.dumps({"stem_model": {"policy_rate": 42.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    ckpt_file = tmp_path / "stem_model.pt"
    ckpt_file.write_bytes(b"stem checkpoint content")

    with caplog.at_level(logging.WARNING):
        rate = get_baseline_for_model(ckpt_file)

    assert rate is None
    assert "legacy record has no recorded SHA-256 hash" in caplog.text


def test_legacy_empty_or_none_sha256_rejected_for_physical_file(tmp_path, monkeypatch, caplog):
    """Verify records with empty or null sha256 strings are rejected for physical files."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    fake_baselines.write_text(
        json.dumps(
            {
                "empty_sha.pt": {"policy_rate": 40.0, "sha256": ""},
                "none_sha.pt": {"policy_rate": 45.0, "sha256": None},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    f1 = tmp_path / "empty_sha.pt"
    f1.write_bytes(b"content 1")
    f2 = tmp_path / "none_sha.pt"
    f2.write_bytes(b"content 2")

    with caplog.at_level(logging.WARNING):
        assert get_baseline_for_model(f1) is None
        assert get_baseline_for_model(f2) is None

    assert "legacy record has no recorded SHA-256 hash" in caplog.text


def test_explicit_baseline_update_remediates_legacy_baseline(tmp_path, monkeypatch, caplog):
    """Verify --update-baseline remediates an unhashed legacy baseline and binds SHA-256."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    fake_baselines.write_text(
        json.dumps({"updatable_model.pt": {"policy_rate": 30.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    ckpt_file = tmp_path / "updatable_model.pt"
    ckpt_file.write_bytes(b"checkpoint version 1")

    # 1. Prior legacy record is rejected
    assert get_baseline_for_model(ckpt_file) is None

    # 2. Update baseline for this checkpoint -> binds sha256
    set_baseline_for_model(ckpt_file, 40.0, mcts_rate=45.0, total_puzzles=100)

    # 3. Now verified baseline is returned
    assert get_baseline_for_model(ckpt_file) == 40.0

    # 4. Modifying checkpoint invalidates baseline
    ckpt_file.write_bytes(b"checkpoint version 2 (tampered)")
    with caplog.at_level(logging.WARNING):
        assert get_baseline_for_model(ckpt_file) is None
    assert "does not match recorded" in caplog.text


def test_synthetic_unhashed_lookup_succeeds_when_file_not_on_disk(tmp_path, monkeypatch):
    """Verify synthetic / mock non-existent paths preserve unhashed lookup compatibility."""
    fake_baselines = tmp_path / "puzzle_baselines.json"
    fake_baselines.write_text(
        json.dumps(
            {
                "synthetic_model.pt": {"policy_rate": 55.0},
                "scalar_synth": 60.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("python.eval.baselines.get_baselines_file_path", lambda: fake_baselines)

    # Neither exists on disk
    assert get_baseline_for_model("synthetic_model.pt") == 55.0
    assert get_baseline_for_model("scalar_synth") == 60.0
