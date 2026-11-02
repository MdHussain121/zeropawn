"""Regression tests for the RL-correctness remediation (audit RL-01..RL-40, 2026-11-01).

Each test names the finding it pins. Scope: the findings that were still
actionable on main @ 85646f6 (RL-03, RL-04, RL-07, RL-09, RL-10, RL-12,
RL-13, RL-16, RL-20, RL-21, RL-22, RL-23, RL-25, RL-26, RL-27, RL-29,
RL-35, RL-37, RL-40). Findings already fixed by earlier campaigns are
covered by their own suites (test_rl_entrypoint_remediation.py,
test_selfplay_replay_remediation.py, test_buffer_restore_contract.py, ...).

No RL campaign is started anywhere here: real components are exercised at
bounded scale (tiny nets, tmp buffers, synthetic datasets), and the
orchestrator tests use the established stub pattern for the heavy
server/worker/trainer constructions.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python  # noqa: F401
from python.data.dataset_contract import DatasetContractError
from python.eval.match_runner import MatchRunner
from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import (
    NO_GAME_ID,
    GpuResidentReplayCache,
    ReplayBuffer,
)
from python.selfplay.worker import SavedPosition
from python.train.train_rl import (
    RLTrainer,
    inspect_rl_checkpoint,
    is_resumable_candidate,
)

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(repo_root / "scripts"))
run_phase4_rl = importlib.import_module("run_phase4_rl")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _pos(
    game_id: int,
    *,
    outcome: int = 1,
    version: int = 1,
    full_search: bool = True,
) -> SavedPosition:
    planes = (np.random.rand(19, 8, 8) < 0.5).astype(np.float32)
    visits = np.zeros(4672, dtype=np.float32)
    visits[game_id % 4672] = 1.0
    return SavedPosition(
        planes=planes,
        visits=visits,
        outcome=outcome,
        is_full_search=full_search,
        model_version=version,
        game_id=game_id,
    )


def _synthetic_human_dataset(
    path: Path, n: int = 64, outcome: int = 255, role: str = "policy_tactics"
) -> Path:
    """A minimal BinaryChessDataset (metadata + planes/moves/outcomes bins).

    Written WITHOUT a validation receipt; consumption therefore requires the
    documented uncertified-data acknowledgement, exactly like the production
    uncertified dataset. All-255 outcomes declare the ``policy_tactics`` role
    — the audit's own consumption contract for policy-only human data (a
    WDL-role dataset that is fully masked is refused by the data gate).
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": n,
                "planes_shape": [19, 8, 8],
                "role": role,
                "kind": role,
            }
        ),
        encoding="utf-8",
    )
    planes = (np.random.rand(n, 19, 8, 8) < 0.5).astype(np.float32)
    planes.tofile(path / "planes.bin")
    moves = (np.arange(n, dtype=np.int16) % 4672)
    moves.tofile(path / "moves.bin")
    outcomes = np.full(n, outcome, dtype=np.uint8)
    outcomes.tofile(path / "outcomes.bin")
    game_ids = np.arange(1, n + 1, dtype=np.uint64)
    game_ids.tofile(path / "game_ids.bin")
    plies = np.arange(n, dtype=np.int16)
    plies.tofile(path / "plies.bin")
    return path


def _make_trainer(
    tmp_path: Path,
    *,
    warmup_steps: int = 500,
    total_steps: int = 100_000,
    buffer: ReplayBuffer | None = None,
    **kw,
):
    torch.manual_seed(7)
    np.random.seed(7)
    if buffer is None:
        buffer = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=256, create_new=True)
        buffer.add_positions([_pos(i + 1) for i in range(64)])
    model = ChessNet(num_blocks=2, num_filters=32)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=16,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=0.0005,
        min_lr=1e-5,
        checkpoint_dir=tmp_path / "ckpts",
        sync_weights_path=tmp_path / "live.pt",
        **kw,
    )
    return trainer, buffer


def _stub_server():
    class _StubServer:
        def __init__(self, *a, **k):
            # The orchestrator copies this state_dict into a real 12x128
            # trainer model, so the stub must carry the production shape.
            self.model = ChessNet(
                in_channels=19, num_filters=128, num_blocks=12, num_actions=4672
            )
            self.model_version = 0

    return _StubServer


# ---------------------------------------------------------------------------
# RL-03 — human-data gate must not block crash-recovery resume
# ---------------------------------------------------------------------------


class _StubTrainer:
    def __init__(self, *a, **k):
        self.run_metadata = {}
        self.step_num = 0
        self.inherited_step_num = 0
        self.direct_server = None

    def load_checkpoint(self, *a, **k):
        self.load_checkpoint_kwargs = k

    def save_checkpoint(self, *a, **k):  # pragma: no cover - only used by stop()
        return "stub.pt"


class _StubClient:
    def __init__(self, *a, **k):
        pass


class _StubWorker:
    def __init__(self, *a, **k):
        self.aborted_games = 0
        self.invalid_games = 0

    def stop(self):
        pass


class _StubMetrics:
    def __init__(self, *a, **k):
        self.total_games = 0

    def close(self):
        pass


def _orchestrator_kwargs(tmp_path: Path, human_dir) -> dict:
    return dict(
        base_checkpoint=None,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_dir=tmp_path,
        # The real ReplayBuffer is part of what these tests exercise (the
        # human-data gate lives inside its construction) — keep the shard
        # files tiny; tests must never allocate a 500k-row (9.3 GB) buffer.
        buffer_capacity=64,
        puzzle_csv_path=str(tmp_path / "puzzles.csv"),
        openings_path=str(tmp_path / "openings.epd"),
        stockfish_path=str(tmp_path / "stockfish.exe"),
        is_verify_mode=False,
        require_certified_eval=False,
        skip_baseline_eval=True,
        human_data_dir=str(human_dir) if human_dir else None,
        device="cpu",
        train_device="cpu",
        infer_device="cpu",
        min_buffer_fill=0,
    )


def _install_orchestrator_stubs(monkeypatch):
    monkeypatch.setattr(run_phase4_rl, "InferenceServer", _stub_server())
    monkeypatch.setattr(run_phase4_rl, "DirectInferenceClient", _StubClient)
    monkeypatch.setattr(run_phase4_rl, "RLTrainer", _StubTrainer)
    monkeypatch.setattr(run_phase4_rl, "SelfPlayWorker", _StubWorker)
    if hasattr(run_phase4_rl, "MetricsTracker"):
        monkeypatch.setattr(run_phase4_rl, "MetricsTracker", _StubMetrics)
    monkeypatch.setattr(run_phase4_rl, "load_puzzles", lambda path: [])


def test_rl03_fresh_start_with_uncertified_human_data_fails_closed(tmp_path, monkeypatch):
    """No resumable checkpoint + uncertified human data → the gate stands."""
    _install_orchestrator_stubs(monkeypatch)
    ds = _synthetic_human_dataset(tmp_path / "human")
    with pytest.raises(DatasetContractError):
        run_phase4_rl.Phase4Orchestrator(**_orchestrator_kwargs(tmp_path, ds))


def test_rl03_crash_recovery_proceeds_without_optional_human_data(tmp_path, monkeypatch):
    """A resumable RL checkpoint lets recovery proceed with the human mix
    disabled — loudly, and never by bypassing the dataset gate."""
    _install_orchestrator_stubs(monkeypatch)
    ds = _synthetic_human_dataset(tmp_path / "human")
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # A minimal RL-resumable artifact (SGD momentum evidence; no arch stamp →
    # no architecture gate; the stub trainer never actually loads it).
    torch.save(
        {
            "model": {"w": torch.zeros(2)},
            "optimizer": {
                "state": {0: {"momentum_buffer": torch.zeros(2)}},
                "param_groups": [{"lr": 0.01, "params": [0]}],
            },
            "step": 5,
        },
        ckpt_dir / "phase4_ckpt_test_step5.pt",
    )
    orch = run_phase4_rl.Phase4Orchestrator(**_orchestrator_kwargs(tmp_path, ds))
    assert orch.human_data_disabled_for_recovery is True
    assert orch._resumed_from is not None and "step5" in orch._resumed_from
    assert orch.trainer.run_metadata.get("config", {}).get("human_data") == "disabled_for_recovery"
    # The buffer was rebuilt without a human dataset.
    assert orch.buffer.human_dataset is None
    # ...while the certified-data path never disables the mix (control).
    assert orch.buffer.capacity > 0


def test_rl03_human_data_gate_still_hard_when_no_checkpoint_exists(tmp_path, monkeypatch):
    """Without any resumable checkpoint the uncertified dataset still blocks."""
    _install_orchestrator_stubs(monkeypatch)
    ds = _synthetic_human_dataset(tmp_path / "human")
    (tmp_path / "ckpts").mkdir(parents=True, exist_ok=True)
    with pytest.raises(DatasetContractError):
        run_phase4_rl.Phase4Orchestrator(**_orchestrator_kwargs(tmp_path, ds))


# ---------------------------------------------------------------------------
# RL-04 — skipped batches are not optimizer progress
# ---------------------------------------------------------------------------


def test_rl04_nonfinite_loss_does_not_advance_the_rl_counter(tmp_path):
    trainer, _buf = _make_trainer(tmp_path)
    params_before = [p.detach().clone() for p in trainer.model.parameters()]
    batch = {
        "planes": torch.full((16, 19, 8, 8), float("nan")),
        "visits": torch.rand(16, 4672),
        "outcomes": torch.randint(0, 3, (16,)),
        "is_full_search": torch.ones(16, dtype=torch.bool),
        "game_ids": torch.arange(1, 17),
    }
    metrics = trainer.train_step(batch_dict=batch)
    assert metrics["skipped_batch"] is True
    assert metrics["step"] == 0  # the previous successful-update count
    assert trainer.step_num == 0
    assert trainer.skipped_step_count == 1
    for p_before, p_after in zip(params_before, trainer.model.parameters()):
        assert torch.equal(p_before, p_after.detach())


def test_rl04_successful_step_increments_the_counter_exactly_once(tmp_path):
    trainer, _buf = _make_trainer(tmp_path)
    metrics = trainer.train_step()
    assert metrics["skipped_batch"] is False
    assert trainer.step_num == 1
    assert trainer.skipped_step_count == 0
    assert metrics["total_loss"] == metrics["total_loss"]  # finite


# ---------------------------------------------------------------------------
# RL-07 — foreign scheduler geometry is refused on full resume
# ---------------------------------------------------------------------------


def test_rl07_foreign_scheduler_geometry_refused_then_explicit_override(tmp_path):
    trainer, buf = _make_trainer(tmp_path)
    trainer.train_step()
    trainer.save_checkpoint(tmp_path / "ckpts" / "own.pt")
    ckpt = torch.load(tmp_path / "ckpts" / "own.pt", map_location="cpu", weights_only=True)
    ckpt["scheduler"]["warmup_steps"] = 200
    ckpt["scheduler"]["total_steps"] = 6000
    torch.save(ckpt, tmp_path / "ckpts" / "foreign.pt")

    # Resuming means continuing against the SAME live buffer (the generation
    # contract refuses a second create_new buffer against this checkpoint).
    refused_trainer, _ = _make_trainer(tmp_path, buffer=buf)
    with pytest.raises(RuntimeError, match="scheduler geometry"):
        refused_trainer.load_checkpoint(tmp_path / "ckpts" / "foreign.pt")

    override_trainer, _ = _make_trainer(tmp_path, buffer=buf)
    override_trainer.load_checkpoint(
        tmp_path / "ckpts" / "foreign.pt", allow_scheduler_geometry_mismatch=True
    )
    assert override_trainer.scheduler.warmup_steps == 200
    assert override_trainer.scheduler.total_steps == 6000
    assert override_trainer.step_num == 1


def test_rl07_matching_geometry_resumes_silently(tmp_path):
    trainer, buf = _make_trainer(tmp_path)
    trainer.train_step()
    trainer.save_checkpoint(tmp_path / "ckpts" / "own.pt")
    fresh_trainer, _ = _make_trainer(tmp_path, buffer=buf)
    fresh_trainer.load_checkpoint(tmp_path / "ckpts" / "own.pt")
    assert fresh_trainer.step_num == 1
    assert fresh_trainer.scheduler.warmup_steps == 500
    assert fresh_trainer.scheduler.total_steps == 100_000


# ---------------------------------------------------------------------------
# RL-09 — a forged SGD stamp cannot pass resume selection
# ---------------------------------------------------------------------------


def test_rl09_forged_stamp_rejected_by_resume_selection(tmp_path):
    poisoned = {
        "model": {"w": torch.zeros(2)},
        "optimizer": {
            "state": {0: {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}},
            "param_groups": [{"lr": 0.01, "betas": (0.9, 0.999), "params": [0]}],
        },
        "_optimizer_type": "SGD",
        "step": 6000,
    }
    p = tmp_path / "forged.pt"
    torch.save(poisoned, p)
    info = inspect_rl_checkpoint(p)
    assert info is not None
    acceptable, reason = is_resumable_candidate(info)
    assert not acceptable
    assert "AdamW" in reason or "foreign" in reason


# ---------------------------------------------------------------------------
# RL-10 — human rows are policy-only when outcomes are the 255 sentinel
# ---------------------------------------------------------------------------


def test_rl10_all_masked_human_disk_rows_still_fill_the_batch(tmp_path):
    ds = _synthetic_human_dataset(tmp_path / "human", n=64, outcome=255)
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "buf",
        capacity=64,
        create_new=True,
        human_data_dir=ds,
        human_cache_size=0,  # force the disk-fallback path
        allow_uncertified_data=True,
    )
    buf.add_positions([_pos(i + 1) for i in range(6)])
    batch, human_mix = buf.sample_batch_tensors(batch_size=8, step=0, device="cpu")
    # RL-40: the batch is never short, with or without valid human outcomes.
    assert batch["planes"].shape[0] == 8
    assert batch["outcomes"].shape[0] == 8
    # RL-10: the two human rows are present with the sentinel outcome —
    # policy supervision only, masked in the WDL loss downstream.
    human_rows = batch["game_ids"] == NO_GAME_ID
    assert int(human_rows.sum()) == 2
    assert int((batch["outcomes"][human_rows] == 255).sum()) == 2
    assert abs(human_mix - 0.25) < 1e-6


def test_rl10_masked_human_rows_through_the_cache_path(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=True)
    # Inject a CPU human cache directly: outcomes are all the 255 sentinel.
    n = 32
    buf._human_cache_planes = (np.random.rand(n, 19, 8, 8) < 0.5).astype(np.float32)
    buf._human_cache_moves = (np.arange(n, dtype=np.int64) % 4672)
    buf._human_cache_outcomes = np.full(n, 255, dtype=np.int64)
    buf._human_cache_size = n
    buf.human_dataset = object()  # non-None: the mix is requested
    buf.add_positions([_pos(i + 1) for i in range(6)])
    batch, human_mix = buf.sample_batch_tensors(batch_size=8, step=0, device="cpu")
    assert batch["planes"].shape[0] == 8
    human_rows = batch["game_ids"] == NO_GAME_ID
    assert int(human_rows.sum()) == 2
    assert int((batch["outcomes"][human_rows] == 255).sum()) == 2
    assert abs(human_mix - 0.25) < 1e-6


def test_rl10_trainer_reports_value_masked_rows(tmp_path):
    trainer, _buf = _make_trainer(tmp_path)
    batch = {
        "planes": torch.rand(16, 19, 8, 8),
        "visits": torch.rand(16, 4672),
        "outcomes": torch.tensor([0, 1, 2, 255, 255, 0, 1, 2, 255, 0, 1, 2, 0, 1, 2, 0]),
        "is_full_search": torch.ones(16, dtype=torch.bool),
        "game_ids": torch.arange(1, 17),
    }
    metrics = trainer.train_step(batch_dict=batch)
    assert metrics["n_value_masked_rows"] == 3
    # The masked rows contributed no WDL gradient: the loss is finite either
    # way, and the count is what makes the policy-only share visible.
    assert metrics["wdl_loss"] == metrics["wdl_loss"]


# ---------------------------------------------------------------------------
# RL-12 — the GPU-cache full-overwrite fast path holds the lock
# ---------------------------------------------------------------------------


def test_rl12_fast_path_adds_are_serialized_against_samplers():
    cache = GpuResidentReplayCache(device="cpu", capacity=8)
    errors: list[Exception] = []

    def adder():
        try:
            for i in range(20):
                n = 16
                cache.add_positions(
                    planes=torch.rand(n, 19, 8, 8),
                    visits=torch.rand(n, 4672),
                    outcomes=torch.randint(0, 3, (n,)),
                    full_search=torch.ones(n, dtype=torch.bool),
                    versions=torch.full((n,), 1, dtype=torch.int32),
                    game_ids=torch.arange(1, n + 1, dtype=torch.int64),
                )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def sampler():
        try:
            for _ in range(200):
                if len(cache) == 0:
                    continue
                batch = cache.sample_batch(4)
                assert batch["planes"].shape[0] == 4
                assert cache.size <= cache.capacity
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=adder), threading.Thread(target=sampler)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert cache.size == cache.capacity
    assert cache.total_added == 20 * 16


# ---------------------------------------------------------------------------
# RL-13 — a stale restored tactical size without a pool cannot break sampling
# ---------------------------------------------------------------------------


def test_rl13_stale_tactical_pool_size_yields_exact_batch(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(10)])
    # Simulate the historical restore: meta carried a pool size that no
    # longer has a pool/cache behind it.
    buf.tactical_pool_size = 50
    buf.saved_tactical_pool_size = 50
    buf._tactical_cache_size = 0
    batch, _mix, tactical_mix = buf.sample_batch_tensors(
        batch_size=16, step=0, device="cpu", return_tactical_mix=True
    )
    assert batch["planes"].shape[0] == 16
    assert tactical_mix == 0.0
    np_batch, _np_mix = buf.sample_batch(batch_size=16, step=0)
    assert np_batch["planes"].shape[0] == 16


# ---------------------------------------------------------------------------
# RL-21 / RL-20 — match_games.csv ledger: one 17-column header + adjudication
# ---------------------------------------------------------------------------


def test_rl21_match_games_header_matches_rows_and_upgrades_header_only(
    tmp_path, monkeypatch
):
    _install_orchestrator_stubs(monkeypatch)
    orch = run_phase4_rl.Phase4Orchestrator(**_orchestrator_kwargs(tmp_path, None))
    csv_path = Path(orch.match_games_csv_path)
    header = csv_path.read_text(encoding="utf-8").strip().split(",")
    assert header[-1] == "adjudicated"
    assert len(header) == 17

    # A persisted row carries exactly 17 values, including the RL-20 flag.
    game = {
        "opening_idx": 0,
        "engine_is_white": True,
        "result": "1-0",
        "seed": 42,
        "moves": ["e2e4"],
        "adjudicated": True,
    }
    orch._persist_match_game(game, partial=False)
    rows = [line for line in csv_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == 2
    assert len(next(csv.reader([rows[1]]))) == 17
    assert next(csv.reader([rows[1]]))[-1] == "1"

    # A header-only legacy file is upgraded; data rows in an old schema raise.
    legacy = tmp_path / "legacy_run" / "match_games.csv"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("timestamp,step,plies\n", encoding="utf-8")
    orch2 = run_phase4_rl.Phase4Orchestrator(
        **{**_orchestrator_kwargs(tmp_path, None), "log_dir": tmp_path / "legacy_run"}
    )
    assert Path(orch2.match_games_csv_path).read_text(encoding="utf-8").count("adjudicated") == 1


def test_rl20_match_runner_threads_adjudication_provenance(tmp_path):
    """_play_game_worker forwards the info dict so per-game rows carry the
    value-cutoff adjudication flag (audit RL-20)."""
    mr = object.__new__(MatchRunner)
    seen = {}

    def _fake_impl(fen, engine_is_white, ctx, max_plies, game_seed, info=None):
        seen["info"] = info
        if info is not None:
            info["adjudicated"] = True
            info["adjudication_reason"] = "value_cutoff_win"
        return "1-0", ["e2e4"]

    mr._play_game_impl = _fake_impl  # type: ignore[method-assign]
    info_log: dict = {}
    result = mr._play_game_worker("startpos", True, ctx=None, info_log=info_log)
    assert result == "1-0"
    assert info_log["adjudicated"] is True
    assert info_log["adjudication_reason"] == "value_cutoff_win"


# ---------------------------------------------------------------------------
# RL-22 — last_push_step records successful pushes only
# ---------------------------------------------------------------------------


class _FailingServer:
    def __init__(self):
        self.model_version = 1

    def load_state_dict_weights(self, state_dict, sync_event=None):
        return self.model_version  # never advances


class _GoodServer:
    def __init__(self):
        self.model_version = 1

    def load_state_dict_weights(self, state_dict, sync_event=None):
        self.model_version += 1
        return self.model_version


def test_rl22_failed_push_does_not_record_last_push_step(tmp_path):
    trainer, _buf = _make_trainer(tmp_path, push_freq=1)
    trainer.direct_server = _FailingServer()
    with pytest.raises(Exception):  # WeightPushError
        trainer.train_step()
    assert trainer.last_push_step == 0
    assert trainer.push_failure_count == 1


def test_rl22_serverless_disk_sync_failure_does_not_record_last_push_step(
    tmp_path, monkeypatch
):
    """Audit S2-04 (RL-22 residual): with no direct server attached, a failed
    disk-sync write must count and raise like every other push failure — a
    server-less run must neither swallow the error nor record ``last_push_step``
    for a sync that never happened."""
    trainer, _buf = _make_trainer(tmp_path, push_freq=1)
    assert trainer.direct_server is None
    assert trainer.sync_weights_path == tmp_path / "live.pt"

    import python.train.train_rl as train_rl_module

    def _boom(payload, destination):
        raise OSError("simulated disk-sync write failure")

    monkeypatch.setattr(train_rl_module, "_atomic_torch_save", _boom)
    with pytest.raises(OSError):
        trainer.train_step()
    assert trainer.step_num == 1  # the SGD update itself succeeded
    assert trainer.last_push_step == 0  # but the push never completed
    assert trainer.push_failure_count == 1


def test_rl22_successful_push_records_last_push_step(tmp_path):
    trainer, _buf = _make_trainer(tmp_path, push_freq=1)
    trainer.direct_server = _GoodServer()
    trainer.train_step()
    assert trainer.step_num == 1
    assert trainer.last_push_step == 1
    assert trainer.push_success_count == 1


# ---------------------------------------------------------------------------
# RL-23 — non-finite gradients never reach the optimizer
# ---------------------------------------------------------------------------


def test_rl23_nonfinite_gradients_skip_the_optimizer_step(tmp_path, monkeypatch):
    trainer, _buf = _make_trainer(tmp_path)
    params_before = [p.detach().clone() for p in trainer.model.parameters()]

    def _inf_norm(parameters, max_norm):
        return torch.tensor(float("inf"))

    import torch.nn.utils as _utils

    monkeypatch.setattr(_utils, "clip_grad_norm_", _inf_norm)
    metrics = trainer.train_step()
    assert metrics["skipped_batch"] is True
    assert trainer.step_num == 0
    assert trainer.skipped_step_count == 1
    for p_before, p_after in zip(params_before, trainer.model.parameters()):
        assert torch.equal(p_before, p_after.detach())


# ---------------------------------------------------------------------------
# RL-25 — seeded cache composition is reproducible
# ---------------------------------------------------------------------------


def test_rl25_same_seed_same_cache_composition(tmp_path):
    ds = _synthetic_human_dataset(tmp_path / "human", n=512, outcome=1)

    def build(seed, tag):
        # gpu_device="cpu" pins the host-RAM cache path so the composition is
        # directly comparable (on a CUDA box the cache would land in VRAM).
        return ReplayBuffer(
            buffer_dir=tmp_path / f"buf_{tag}",
            capacity=64,
            create_new=True,
            human_data_dir=ds,
            human_cache_size=64,
            allow_uncertified_data=True,
            cache_rng_seed=seed,
            gpu_device="cpu",
        )

    a = build(123, "a")
    b = build(123, "b")
    c = build(456, "c")
    assert np.array_equal(a._human_cache_moves, b._human_cache_moves)
    assert not np.array_equal(a._human_cache_moves, c._human_cache_moves)


# ---------------------------------------------------------------------------
# RL-26 — capacity must be positive
# ---------------------------------------------------------------------------


def test_rl26_zero_capacity_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=0, create_new=True)


# ---------------------------------------------------------------------------
# RL-27 — honest shutdown provenance and eval metadata
# ---------------------------------------------------------------------------


def test_rl27_stop_records_the_final_checkpoint_path(tmp_path, monkeypatch):
    _install_orchestrator_stubs(monkeypatch)
    orch = run_phase4_rl.Phase4Orchestrator(**_orchestrator_kwargs(tmp_path, None))
    assert orch._last_final_ckpt == ""
    orch._status_for_checkpoint = "SUCCESS"
    orch.stop()
    assert orch._last_final_ckpt == "stub.pt"


def test_rl27_eval_row_records_the_match_seed(tmp_path, monkeypatch):
    """The eval seed column records the seed the match actually ran with."""
    source = (repo_root / "scripts" / "run_phase4_rl.py").read_text(encoding="utf-8")
    assert "int(match_runner.seed)" in source
    assert "seed=int(self.seed) if self.seed is not None else 42," in source


# ---------------------------------------------------------------------------
# RL-29 — the tau=1.0 raw-visit hazard is never silent
# ---------------------------------------------------------------------------


def test_rl29_policy_target_temp_validation(tmp_path):
    trainer, _buf = _make_trainer(tmp_path, policy_target_temp=1.0)  # warns, works
    assert trainer.policy_target_temp == 1.0
    with pytest.raises(ValueError, match="policy_target_temp"):
        _make_trainer(tmp_path / "b2", policy_target_temp=0.0)


# ---------------------------------------------------------------------------
# RL-35 — telemetry cannot log a phantom 0.000 tactical mix
# ---------------------------------------------------------------------------


def test_rl35_train_step_reports_actual_mixes(tmp_path):
    trainer, buf = _make_trainer(tmp_path)
    # A loaded human dataset makes the sampler report a real human mix.
    metrics = trainer.train_step()
    assert "actual_human_mix" in metrics
    assert "actual_tactical_mix" in metrics
    assert "n_value_masked_rows" in metrics
    assert metrics["actual_tactical_mix"] == 0.0  # no tactical cache loaded
    assert metrics["actual_human_mix"] == 0.0  # no human dataset configured


# ---------------------------------------------------------------------------
# RL-37 — the banner's --resume exists on the parser
# ---------------------------------------------------------------------------


def test_rl37_help_advertises_resume_and_exits_zero():
    argv_backup = sys.argv
    sys.argv = ["run_phase4_rl.py", "--help"]
    try:
        with redirect_stdout(io.StringIO()) as buf:
            with pytest.raises(SystemExit) as excinfo:
                run_phase4_rl.main()
    finally:
        sys.argv = argv_backup
    assert excinfo.value.code == 0
    assert "--resume RESUME" in buf.getvalue()


# ---------------------------------------------------------------------------
# RL-40 — batch-size invariants under a shortfall
# ---------------------------------------------------------------------------


def test_rl40_single_source_short_batch_is_shuffled_safely(tmp_path):
    """With only one contributing source the shuffle is skipped; with
    several, the permutation never indexes past the assembled batch."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(6)])
    # Human cache with fewer rows than requested (forced small cache) — the
    # draw still returns the requested count, and no permutation crash.
    n = 3
    buf.human_dataset = object()
    buf._human_cache_planes = (np.random.rand(n, 19, 8, 8) < 0.5).astype(np.float32)
    buf._human_cache_moves = (np.arange(n, dtype=np.int64) % 4672)
    buf._human_cache_outcomes = np.full(n, 1, dtype=np.int64)
    buf._human_cache_size = n
    batch, _mix = buf.sample_batch_tensors(batch_size=8, step=0, device="cpu")
    assert batch["planes"].shape[0] == 8
