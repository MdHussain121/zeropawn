"""Tests for scripts/bench_selfplay.py benchmark protocol."""

import json
import logging
from pathlib import Path
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from scripts.bench_selfplay import (  # noqa: E402
    BenchmarkRunner,
    query_gpu_telemetry,
    query_system_telemetry,
    query_throttle_reasons,
)


def test_telemetry_query_functions():
    """Verify hardware telemetry query functions return non-empty data without crashing."""
    gpu = query_gpu_telemetry()
    assert isinstance(gpu, dict)
    assert "gpu_util_pct" in gpu
    assert "temp_c" in gpu

    sys_telemetry = query_system_telemetry()
    assert isinstance(sys_telemetry, dict)
    assert "avg_cpu_util_pct" in sys_telemetry
    assert "free_ram_mb" in sys_telemetry
    assert sys_telemetry["free_ram_mb"] > 0

    throttle = query_throttle_reasons()
    assert isinstance(throttle, str)


def test_benchmark_runner_dry_run(tmp_path):
    """Verify BenchmarkRunner executes warm-up + counted phases, produces CSV and JSON.

    Updated for the S4-02 fail-closed contract: the harness-mechanics dry run
    needs SOME weights, so it now opts into the explicit random-init mode
    (``allow_random_init=True``). The old ``checkpoint_path=""`` relied on the
    removed silent fallback to an untrained model.
    """
    # SA2-01 (2026-11-05 pre-push remediation): when the local production
    # dataset carries a validation receipt that predates masks.bin receipt
    # binding, the replay-buffer fail-closed gate refuses the dataset with a
    # DatasetContractError naming the exact remedy. The gate itself is pinned
    # by tests/test_sa2_remediation_20261106.py; this harness-mechanics dry
    # run skips loudly instead of failing on the environment's receipt state
    # (tests never modify anything under data/ or models/).
    local_dataset = Path(__file__).resolve().parent.parent / "data" / "dataset_training"
    masks_path = local_dataset / "train" / "masks.bin"
    if masks_path.is_file():
        from python.data.dataset_validation import verify_receipt

        bound, reasons, _ = verify_receipt(local_dataset, deep=False)
        if not bound and any("masks.bin" in r for r in reasons):
            pytest.skip(
                "local data/dataset_training receipt predates masks.bin receipt "
                "binding, so the fail-closed replay-buffer gate refuses it "
                "(DatasetContractError); re-run 'zeropawn data validate --dir "
                "data/dataset_training --write-receipt' outside the test suite"
            )
    runner = BenchmarkRunner(
        mode="selfplay",
        warmup_sec=1.0,
        counted_sec=2.0,
        sample_interval=1.0,
        trials=1,
        concurrency=2,
        full_sims=5,
        fast_sims=5,
        full_search_prob=0.35,
        tablebase_path=None,
        output_dir=str(tmp_path),
        checkpoint_path="",
        allow_random_init=True,
        watchdog_free_ram_mb=0.0,
    )

    result = runner.run_benchmark()

    assert result["mode"] == "selfplay"
    assert result["trials"] == 1
    assert "sims_per_sec" in result["aggregated"]

    csv_path = tmp_path / "selfplay_trial_1.csv"
    assert csv_path.is_file()

    json_path = tmp_path / "selfplay_summary.json"
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["mode"] == "selfplay"


# ---------------------------------------------------------------------------
# Audit S4-02 — fail-closed checkpoint resolution (no silent random-init)
# ---------------------------------------------------------------------------


class _FailingResolver:
    """Stands in for python.cli.common.get_default_checkpoint_path."""

    def __call__(self):
        from python.model.checkpoint_registry import DefaultCheckpointError

        raise DefaultCheckpointError(
            "No designated, verified default checkpoint is available (test stub)."
        )


def test_bench_checkpoint_default_missing_fails_closed(tmp_path, monkeypatch):
    """Default resolution failure must be a hard error naming the recovery
    actions — never a silent fallback to a random-init model (audit S4-02)."""
    import python.cli.common as cli_common

    monkeypatch.setattr(cli_common, "get_default_checkpoint_path", _FailingResolver())
    with pytest.raises(RuntimeError) as excinfo:
        BenchmarkRunner(
            mode="selfplay",
            output_dir=str(tmp_path),
            checkpoint_path="default",
            tablebase_path=None,
        )
    msg = str(excinfo.value)
    assert "--checkpoint-path" in msg
    assert "--allow-random-init" in msg


def test_bench_explicit_checkpoint_path_is_used(tmp_path):
    """An explicit, existing checkpoint path is used as-is (no registry round-trip)."""
    ckpt = tmp_path / "tiny_model.pt"
    torch.save({"model": {}}, ckpt)  # existence is what resolution checks
    runner = BenchmarkRunner(
        mode="selfplay",
        output_dir=str(tmp_path),
        checkpoint_path=str(ckpt),
        tablebase_path=None,
    )
    assert runner.checkpoint_path == str(ckpt)


def test_bench_allow_random_init_warns_and_proceeds(tmp_path, monkeypatch, caplog):
    """--allow-random-init proceeds with an untrained model behind a loud
    warning that the results are meaningless for strength (audit S4-02)."""
    import python.cli.common as cli_common

    monkeypatch.setattr(cli_common, "get_default_checkpoint_path", _FailingResolver())
    with caplog.at_level(logging.WARNING, logger="bench_selfplay"):
        runner = BenchmarkRunner(
            mode="selfplay",
            output_dir=str(tmp_path),
            checkpoint_path="default",
            allow_random_init=True,
            tablebase_path=None,
        )
    assert runner.checkpoint_path is None
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "meaningless for strength" in r.getMessage()
    ]
    assert warnings, "expected a loud random-init warning"
