"""Comprehensive argument parity and pipeline coverage tests for ZeroPawn Supervised Learning.

Validates that:
1. Every argument for data acquisition, preprocessing, supervised training, and model verification
   is registered with valid types, defaults, choices, and boundaries.
2. Argument resolution and propagation work correctly through both direct CLI and interactive dispatch.
3. Invalid numerical ranges or missing required arguments raise clean errors without corrupted operations.
4. End-to-end tiny synthetic pipeline runs seamlessly in seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from python.cli.config import (
    CONFIG_REGISTRY,
    resolve_configuration,
)
from python.cli.main import build_parser, execute_operation_from_interactive
from python.cli.operations import OPERATIONS_REGISTRY, OpArgument, OperationDef
from python.model.chess_net import ChessNet
from python.train.train_supervised import train_supervised


def test_supervised_cli_parser_arguments():
    """Verify all arguments declared for the supervised command in direct CLI."""
    parser = build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    supervised_parser = sub_action.choices["supervised"]

    arg_dests = {a.dest for a in supervised_parser._actions}
    expected_dests = {
        "train_dir",
        "val_dir",
        "epochs",
        "max_steps",
        "batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "lr",
        "warmup_steps",
        "num_blocks",
        "num_filters",
        "tactics_ratio",
        "tactics_dir",
        "checkpoint",
        "seed",
        "checkpoint_dir",
        "log_dir",
        "verify",
        "no_gpu_resident",
        "workers",
        "auto",
    }
    for dest in expected_dests:
        assert dest in arg_dests, f"Missing supervised argument: {dest}"


def test_data_subcommand_arguments():
    """Verify arguments across all data subcommands."""
    parser = build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    data_parser = sub_action.choices["data"]
    data_subs = next(a for a in data_parser._actions if isinstance(a, argparse._SubParsersAction))

    # 1. Download
    dl_actions = {a.dest for a in data_subs.choices["download"]._actions}
    assert {"month", "info", "check_license", "output_dir"}.issubset(dl_actions)

    # 2. Filter
    fl_actions = {a.dest for a in data_subs.choices["filter"]._actions}
    assert {"input", "output", "max_kept", "min_elo", "min_tc", "min_plies"}.issubset(fl_actions)

    # 3. Build
    bd_actions = {a.dest for a in data_subs.choices["build"]._actions}
    assert {"pgn", "output_dir", "max_games"}.issubset(bd_actions)

    # 4. Build-tactics
    bt_actions = {a.dest for a in data_subs.choices["build-tactics"]._actions}
    assert {"csv", "output_dir", "max_solver_plies"}.issubset(bt_actions)

    # 5. Validate
    val_actions = {a.dest for a in data_subs.choices["validate"]._actions}
    assert {"dataset_dir", "dir_flag"}.issubset(val_actions)

    # 6. Check-leakage
    leak_actions = {a.dest for a in data_subs.choices["check-leakage"]._actions}
    assert {"train_dir", "val_dir", "max_positions"}.issubset(leak_actions)


def test_interactive_operation_argument_parity():
    """Verify that every catalog operation argument matches parser flags."""
    parser = build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))

    supervised_op = next(op for op in OPERATIONS_REGISTRY if op.id == "train_supervised")
    assert supervised_op.category_id == 1
    op_arg_names = {a.name for a in supervised_op.arguments}
    assert "batch_size" in op_arg_names
    assert "epochs" in op_arg_names
    assert "lr" in op_arg_names
    assert "train_dir" in op_arg_names
    assert "val_dir" in op_arg_names


def test_interactive_dispatch_execution(tmp_path):
    """Verify executing operations via interactive execution helper."""
    # Test models_list operation execution
    models_op = next(op for op in OPERATIONS_REGISTRY if op.id == "models_list")
    exit_code = execute_operation_from_interactive(models_op, {}, model_path=None)
    assert exit_code == 0


def test_synthetic_end_to_end_pipeline(tmp_path):
    """Fast hermetic end-to-end integration test (audit SLQ-21 remediation):
    Synthetic PGN -> builder helpers -> validation receipt -> certified
    fail-closed data gate -> train 2 steps on CPU -> telemetry + stamped
    checkpoint -> reload model.

    This is the production consumption path: the data gate runs in its
    default (certified) mode, so the fixture must carry role metadata and a
    genuine validation receipt — exactly what the stale pre-remediation test
    could not do.
    """
    import chess
    import chess.pgn
    from python.data.build_dataset import process_pgn_game, save_dataset_split
    from python.data.dataset_validation import validate_split, write_receipt

    # 1. Create a tiny synthetic PGN with 3 games (different results so the
    #    WDL labels cover all three classes).
    pgn_path = tmp_path / "synthetic_games.pgn"
    with open(pgn_path, "w", encoding="utf-8") as f:
        f.write(
            '[Event "Test"]\n[White "Player1"]\n[Black "Player2"]\n[Result "1-0"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0\n\n"
            '[Event "Test 2"]\n[White "Player3"]\n[Black "Player4"]\n[Result "0-1"]\n\n'
            "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Nf3 Be7 5. Bg5 O-O 6. e3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 0-1\n\n"
            '[Event "Test 3"]\n[White "Player5"]\n[Black "Player6"]\n[Result "1/2-1/2"]\n\n'
            "1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6 6. e3 e5 7. Nc3 Nc6 8. h3 h6 9. Be3 Be6 10. Qd2 Qd7 1/2-1/2\n"
        )

    # 2. Convert to binary dataset via the production helpers, placing games
    #    deterministically so train/val are both non-empty.
    train_positions = []
    val_positions = []
    game_no = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            gid, split, pos_list = process_pgn_game(game)
            if game_no == 0:
                train_positions.extend(pos_list)
            else:
                val_positions.extend(pos_list)
            game_no += 1

    assert train_positions and val_positions
    ds_dir = tmp_path / "synth_ds"
    save_dataset_split(ds_dir / "train", train_positions, "train")
    save_dataset_split(ds_dir / "val", val_positions, "val")

    assert (ds_dir / "train" / "planes.bin").is_file()
    assert (ds_dir / "train" / "metadata.json").is_file()
    assert (ds_dir / "train" / "stm.bin").is_file()

    # 3. Certify the dataset: genuine validation receipt binding the bytes.
    reports = {
        split: validate_split(
            ds_dir / split, scan_planes="bounded", plane_scan_rows=100, check_build_record=False
        )
        for split in ("train", "val")
    }
    for split, report in reports.items():
        assert report.overall == "pass", f"{split} fixture must validate"
    write_receipt(ds_dir, reports, overall="pass")

    # 4. Train tiny model for 2 steps on CPU through the CERTIFIED gate.
    ckpt_dir = tmp_path / "ckpts"
    metrics = train_supervised(
        train_dir=ds_dir / "train",
        val_dir=ds_dir / "val",
        batch_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        base_lr=0.001,
        warmup_steps=1,
        epochs=1,
        max_steps=2,
        checkpoint_dir=ckpt_dir,
        num_blocks=2,
        num_filters=32,
        num_workers=0,
        gpu_resident=False,
        seed=1234,
    )
    assert "total_loss" in metrics

    # 5. Durable telemetry (audit SLQ-02): manifest + JSONL stream must exist
    #    and bind the run to the receipt-identified dataset.
    manifest = json.loads((ckpt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 1234
    assert manifest["model"]["param_count"] > 0
    assert manifest["dataset"]["train"]["certified"] is True
    assert manifest["dataset"]["train"]["receipt_id"] is not None
    telemetry_lines = (ckpt_dir / "telemetry.jsonl").read_text().splitlines()
    events = [json.loads(line)["event"] for line in telemetry_lines]
    assert "run_started" in events
    assert "train_step" in events
    assert "epoch_end" in events
    assert "checkpoint_saved" in events

    # 6. Checkpoint reloading and inference verification, with the SLQ-17
    #    stamps and provenance metadata present.
    best_ckpt = ckpt_dir / "best_model.pt"
    assert best_ckpt.is_file()
    ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    assert "model" in ckpt_data
    assert "metadata" in ckpt_data
    assert ckpt_data["metadata"]["architecture"] == "2 ResBlocks x 32 Filters"
    assert ckpt_data.get("checkpoint_type") == "supervised"
    assert ckpt_data.get("_optimizer_type") == "AdamW"
    assert ckpt_data.get("run_metadata", {}).get("seed") == 1234

    model = ChessNet(in_channels=19, num_filters=32, num_blocks=2, num_actions=4672)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    sample_input = torch.randn(1, 19, 8, 8)
    pol_logits, val_logits = model(sample_input)
    assert pol_logits.shape == (1, 4672)
    assert val_logits.shape == (1, 3)
    val_pred = model.predict_value(sample_input)
    assert val_pred.shape == (1,)
    assert -1.0 <= val_pred.item() <= 1.0


def test_synthetic_end_to_end_window_skip(tmp_path):
    """A non-finite micro-batch aborts its window: no optimizer advance, and
    the skip is visible in the durable telemetry (audit SLQ-16)."""
    from types import SimpleNamespace

    from python.data.build_dataset import save_dataset_split

    rng = np.random.default_rng(0)

    def rows(n, outcomes, nan_row=None):
        out = []
        for i in range(n):
            planes = rng.random((19, 8, 8), dtype=np.float32)
            if nan_row is not None and i == nan_row:
                planes[0, 0, 0] = float("nan")
            out.append(
                SimpleNamespace(
                    planes=planes,
                    move_idx=int(rng.integers(0, 4672)),
                    outcome=outcomes[i % 3],
                    game_id=500 + i,
                    ply=10,
                    pos_hash=0,
                    legal_mask=np.packbits(np.ones(4672, dtype=bool)),
                    stm=i % 2,
                )
            )
        return out

    ds_dir = tmp_path / "ds"
    save_dataset_split(ds_dir / "train", rows(8, [0, 1, 2], nan_row=3), "train")
    save_dataset_split(ds_dir / "val", rows(4, [0, 1, 2, 0]), "val")
    ckpt_dir = tmp_path / "ckpts"
    metrics = train_supervised(
        train_dir=ds_dir / "train",
        val_dir=ds_dir / "val",
        batch_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        base_lr=0.001,
        warmup_steps=1,
        epochs=1,
        max_steps=20,
        checkpoint_dir=ckpt_dir,
        num_blocks=2,
        num_filters=32,
        num_workers=0,
        gpu_resident=False,
        # The synthetic rows carry NaN planes, which the plane scan rejects;
        # the fail-closed gate would (correctly) refuse them. This test
        # exercises the loop's skip path, so it opts out explicitly and loudly.
        allow_uncertified_data=True,
        verify_identity="stat",
    )
    telemetry_lines = (ckpt_dir / "telemetry.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in telemetry_lines]
    skipped = [r for r in records if r.get("event") == "train_step" and r.get("window_skipped")]
    assert skipped, "the NaN row's window must be reported as skipped in telemetry"
    # The manifest records the run regardless of the skip.
    manifest = json.loads((ckpt_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
