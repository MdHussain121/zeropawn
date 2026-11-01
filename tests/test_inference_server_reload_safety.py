"""L-2 / L-3 regression tests: inference-server locking and hot-reload safety.

L-2 (audit): ``InferenceServer`` allocated ``_weights_lock`` twice — the second
allocation, after construction, silently replaced the lock that had guarded
the initial checkpoint load.  One lock now, created before first use.

L-3 (audit): hot-reload failed soft — a missing, corrupt, or incompatible
candidate kept the stale weights with only a log line, invisible to any
caller.  The contract now: success is exactly "returned version increased";
failure keeps the active weights/version untouched and records
``last_reload_error`` / ``reload_failure_count``; the serving loop's wrapper
(``_hot_reload_and_report``) emits a loud ERROR when a request does not
activate.  Concurrency: a published model reference is never mutated, so
every (prediction, version) pair must be internally consistent — verified
deterministically with two bitwise-distinct checkpoint fixtures.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.model.chess_net import ChessNet  # noqa: E402
from python.selfplay.inference_server import (  # noqa: E402
    InferenceServer,
    _hot_reload_and_report,
)


def _checkpoint_payload(concentrated: bool) -> dict:
    """Tiny deterministic ChessNet whose outputs identify the weights.

    Variant A (concentrated=False): zero final biases -> uniform policy,
    uniform WDL.  Variant B: policy biased hard to action 0, WDL biased to
    'win' -> p[0] ~ 1.0.  The tag is recoverable from any single prediction.
    """
    model = ChessNet(num_blocks=1, num_filters=8)
    if concentrated:
        with torch.no_grad():
            model.policy_fc.bias[0] = 20.0
            model.value_fc2.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))
    return {"model": model.state_dict(), "num_blocks": 1, "num_filters": 8, "step": 1}


def _save(tmp_path: Path, name: str, concentrated: bool) -> str:
    p = tmp_path / name
    torch.save(_checkpoint_payload(concentrated), p)
    return str(p)


@pytest.fixture()
def server(tmp_path):
    ckpt_a = _save(tmp_path, "variant_a.pt", concentrated=False)
    s = InferenceServer(checkpoint_path=ckpt_a, device="cpu")
    return s, ckpt_a, tmp_path


def _predict(server, batch: int = 4):
    states = np.zeros((batch, 19, 8, 8), dtype=np.float32)
    policy, wdl, version = server.predict_batch(states)
    return policy, wdl, version


# ---------------------------------------------------------------------------
# L-2: a single weights lock, created before first use
# ---------------------------------------------------------------------------


def test_only_one_weights_lock_is_ever_allocated(tmp_path, monkeypatch):
    import python.selfplay.inference_server as ismod

    created = []
    real_rlock = threading.RLock

    def counting_rlock():
        lock = real_rlock()
        created.append(lock)
        return lock

    monkeypatch.setattr(ismod.threading, "RLock", counting_rlock)
    ckpt = _save(tmp_path, "a.pt", concentrated=False)
    s = InferenceServer(checkpoint_path=ckpt, device="cpu")

    assert len(created) == 1, f"expected one _weights_lock allocation, saw {len(created)}"
    assert s._weights_lock is created[0]
    # The single lock is the one that guarded the initial load and the swap.
    with s._weights_lock:  # re-entrant under load_checkpoint's own acquire
        assert s.load_checkpoint(ckpt) == s.model_version  # successful no-op reload


# ---------------------------------------------------------------------------
# L-3: failure keeps active weights, version unchanged, reason recorded
# ---------------------------------------------------------------------------


def test_successful_reload_updates_identity_and_clears_error(server):
    s, ckpt_a, tmp_path = server
    v0 = s.model_version
    assert s.active_checkpoint_path == ckpt_a
    assert s.reload_ok

    ckpt_b = _save(tmp_path, "variant_b.pt", concentrated=True)
    v1 = s.load_checkpoint(ckpt_b)
    assert v1 == v0 + 1
    assert s.active_checkpoint_path == ckpt_b
    assert s.last_reload_error is None
    assert s.reload_ok
    policy, _, version = _predict(s)
    assert version == v1
    assert policy[0, 0] > 0.5  # variant B is served


def test_missing_replacement_keeps_weights_and_records_reason(server):
    s, ckpt_a, tmp_path = server
    before_policy, before_wdl, v_before = _predict(s)

    missing = str(tmp_path / "does_not_exist.pt")
    v_after = s.load_checkpoint(missing)

    assert v_after == v_before  # success ALWAYS increments; unchanged == failure
    assert not s.reload_ok
    assert s.reload_failure_count == 1
    assert missing in s.last_reload_error and "unreadable" in s.last_reload_error
    assert s.active_checkpoint_path == ckpt_a  # identity unchanged
    after_policy, after_wdl, v_after2 = _predict(s)
    assert v_after2 == v_before
    assert np.array_equal(before_policy, after_policy)
    assert np.array_equal(before_wdl, after_wdl)


def test_corrupt_replacement_keeps_weights(server):
    s, ckpt_a, tmp_path = server
    before_policy, _, v_before = _predict(s)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"definitely not a checkpoint")
    v_after = s.load_checkpoint(str(corrupt))

    assert v_after == v_before
    assert s.reload_failure_count == 1
    assert "unreadable" in s.last_reload_error
    after_policy, _, _ = _predict(s)
    assert np.array_equal(before_policy, after_policy)


def test_incompatible_state_dict_is_rejected(server):
    s, ckpt_a, tmp_path = server
    bad = tmp_path / "wrong_arch.pt"
    torch.save({"model": {"not_a_real_layer.weight": torch.zeros(3)}}, bad)
    v_before = s.model_version

    v_after = s.load_checkpoint(str(bad))

    assert v_after == v_before
    assert "invalid checkpoint" in s.last_reload_error
    assert s.active_checkpoint_path == ckpt_a


def test_state_dict_push_rejection_is_recorded(server):
    s, ckpt_a, _ = server
    v_before = s.model_version

    v_after = s.load_state_dict_weights({"bogus": torch.zeros(2)})

    assert v_after == v_before
    assert s.reload_failure_count == 1
    assert "<state_dict push>" in s.last_reload_error
    assert s.active_checkpoint_path == ckpt_a


def test_repeated_failures_then_recovery(server):
    s, ckpt_a, tmp_path = server
    missing = str(tmp_path / "nope.pt")
    for _ in range(3):
        s.load_checkpoint(missing)
    assert s.reload_failure_count == 3
    assert not s.reload_ok

    ckpt_b = _save(tmp_path, "variant_b.pt", concentrated=True)
    v = s.load_checkpoint(ckpt_b)
    assert v == 2
    assert s.reload_ok
    assert s.last_reload_error is None
    assert s.reload_failure_count == 3  # cumulative counter, not reset
    assert s.active_checkpoint_path == ckpt_b
    policy, _, version = _predict(s)
    assert version == v and policy[0, 0] > 0.5


def test_hot_reload_and_report_logs_non_activation(tmp_path, caplog):
    class _StubServer:
        model_version = 7
        active_checkpoint_path = "old.pt"
        last_reload_error = "old.pt: unreadable checkpoint: boom"

        def load_checkpoint(self, path):  # soft-fail: version unchanged
            return self.model_version

    with caplog.at_level(logging.ERROR):
        v = _hot_reload_and_report(_StubServer(), "new.pt")
    assert v == 7
    assert any(
        "DID NOT ACTIVATE" in rec.message and "new.pt" in rec.getMessage()
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# L-3 concurrency: no torn or mixed model state during reloads
# ---------------------------------------------------------------------------


def test_concurrent_inference_never_observes_torn_model_state(tmp_path):
    """Deterministic consistency check: responses must match exactly one of
    the two fixture models.  Variant A -> policy[0] ~ 1/4672; variant B ->
    policy[0] ~ 1.0.  Reloads alternate A/B, so version parity must agree
    with the observed output for every single response, regardless of
    interleaving."""
    ckpt_a = _save(tmp_path, "variant_a.pt", concentrated=False)
    ckpt_b = _save(tmp_path, "variant_b.pt", concentrated=True)
    s = InferenceServer(checkpoint_path=ckpt_a, device="cpu")
    s.load_checkpoint(ckpt_b)  # v2 = B

    def is_b(policy_row) -> bool:
        return float(policy_row[0]) > 0.5

    # Deterministic anchors: what the current version must produce.
    policy, _, version = _predict(s)
    assert version == 2 and is_b(policy[0])

    # Stress: readers keep predicting while the model is swapped 20 times.
    stop = threading.Event()
    observed = []
    errors = []

    def reader():
        try:
            while not stop.is_set():
                policy, wdl, version = _predict(s, batch=2)
                observed.append((version, is_b(policy[0]), is_b(policy[1])))
        except Exception as e:  # pragma: no cover - failure reporting
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        for i in range(20):
            path = ckpt_a if i % 2 == 0 else ckpt_b
            v = s.load_checkpoint(path)
            assert v == 3 + i  # every successful swap bumps the version
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)

    assert not errors, errors
    assert observed, "readers must have produced responses"
    for version, row0_is_b, row1_is_b in observed:
        expected_b = (version - 1) % 2 == 1  # v1=A, v2=B, v3=A, ...
        assert row0_is_b == expected_b, (version, row0_is_b)
        assert row1_is_b == expected_b, (version, row1_is_b)  # no per-row mixing
