"""Checkpoint-to-buffer restoration contract (audit finding CKP-05).

The audited defect: resume overwrote the buffer's newer ``write_head`` with the
checkpoint's older value and never restored ``size``, so the ring continued with
``hand=15,947`` over ``size=96,067`` and mixed two policy generations (the exact
values observed on disk 2026-10-30/31).

These tests exercise every reconciliation branch on synthetic buffers only. The
production replay buffer is never opened here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from python.model.chess_net import ChessNet
from python.selfplay.replay_buffer import ReplayBuffer
from python.train.train_rl import BufferRestoreError, RLTrainer


def _positions(n: int, start: int = 0):
    from python.selfplay.worker import SavedPosition

    out = []
    for i in range(start, start + n):
        visits = np.zeros(4672, dtype=np.float32)
        visits[i % 4672] = 1.0
        out.append(
            SavedPosition(
                planes=np.random.rand(19, 8, 8).astype(np.float32),
                visits=visits,
                outcome=i % 3,
                is_full_search=True,
                model_version=1,
                game_id=i + 1,  # ring records require game_id > 0 (integrated SR-07 contract)
            )
        )
    return out


def _trainer(tmp_path: Path, name: str, capacity: int = 64, create_new: bool = True) -> RLTrainer:
    buf = ReplayBuffer(
        buffer_dir=tmp_path / f"buf_{name}", capacity=capacity, create_new=create_new
    )
    return RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        batch_size=8,
        warmup_steps=5,
        total_steps=100,
        checkpoint_dir=tmp_path / f"ckpt_{name}",
        sync_weights_path=tmp_path / f"sync_{name}.pt",
    )


def _reload(tmp_path: Path, name: str, capacity: int = 64) -> ReplayBuffer:
    """Re-open an existing buffer directory (the real restart path)."""
    return ReplayBuffer(buffer_dir=tmp_path / f"buf_{name}", capacity=capacity, create_new=False)


def _trainer_on(tmp_path: Path, buffer: ReplayBuffer, name: str) -> RLTrainer:
    return RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buffer,
        batch_size=8,
        checkpoint_dir=tmp_path / f"ckpt_{name}",
        sync_weights_path=tmp_path / f"sync_{name}.pt",
    )


# ── Buffer state triple and generation ───────────────────────────────────────


def test_buffer_state_triple_is_exposed_and_persisted(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_state", capacity=32, create_new=True)
    buf.add_positions(_positions(10))
    state = buf.buffer_state()
    assert state["write_head"] == 10
    assert state["size"] == 10
    assert state["total_added"] == 10
    assert state["capacity"] == 32
    assert state["generation"]

    meta = json.loads((tmp_path / "buf_state" / "meta.json").read_text(encoding="utf-8"))
    assert meta["buffer_generation"] == state["generation"]

    # A restart reads the same generation back (identity survives the process).
    reopened = ReplayBuffer(buffer_dir=tmp_path / "buf_state", capacity=32, create_new=False)
    assert reopened.buffer_generation == state["generation"]
    assert reopened.buffer_state() == state


def test_create_new_mints_a_new_generation(tmp_path):
    first = ReplayBuffer(buffer_dir=tmp_path / "buf_gen_mint", capacity=16, create_new=True)
    gen1 = first.buffer_generation
    second = ReplayBuffer(buffer_dir=tmp_path / "buf_gen_mint", capacity=16, create_new=True)
    assert second.buffer_generation != gen1


def test_legacy_meta_without_generation_is_treated_as_unknown(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf_legacy_meta", capacity=16, create_new=True)
    buf.add_positions(_positions(5))
    meta_path = tmp_path / "buf_legacy_meta" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("buffer_generation")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    reopened = ReplayBuffer(buffer_dir=tmp_path / "buf_legacy_meta", capacity=16, create_new=False)
    assert reopened.buffer_generation is None
    loaded = reopened.buffer_state()
    assert loaded["size"] == 5 and loaded["write_head"] == 5


# ── Reconciliation branches ──────────────────────────────────────────────────


def test_matching_snapshot_restores_exactly(tmp_path):
    trainer = _trainer(tmp_path, "match")
    trainer.replay_buffer.add_positions(_positions(12))
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "match.pt", keep_last_n=-1)
    saved = torch.load(ckpt, weights_only=False)

    live = _reload(tmp_path, "match")
    trainer2 = _trainer_on(tmp_path, live, "match2")
    trainer2.load_checkpoint(ckpt)
    decision = trainer2.run_metadata["buffer_resume_decision"]
    assert decision["decision"] == "identical_snapshot"
    assert live.write_head == saved["buffer_state"]["write_head"]
    assert live.size == saved["buffer_state"]["size"]


def test_live_buffer_newer_adopts_the_live_triple_as_a_unit(tmp_path):
    """The audited 15,947-vs-96,067 case: newest state wins, consistently."""
    trainer = _trainer(tmp_path, "newer")
    trainer.replay_buffer.add_positions(_positions(8))
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "newer.pt", keep_last_n=-1)
    saved = torch.load(ckpt, weights_only=False)
    assert saved["buffer_state"]["total_added"] == 8

    # The live buffer keeps accumulating after the checkpoint was written.
    buf = _reload(tmp_path, "newer")
    buf.add_positions(_positions(20, start=100))
    assert buf.total_added == 28

    trainer2 = _trainer_on(tmp_path, buf, "newer2")
    trainer2.load_checkpoint(ckpt)
    decision = trainer2.run_metadata["buffer_resume_decision"]
    assert decision["decision"] == "adopted_live_newer"
    # head and size are the LIVE ones — never the checkpoint's stale pointer.
    assert buf.write_head == 28 and buf.size == 28
    assert buf.write_head != saved["buffer_state"]["write_head"]
    # The checkpoint's own snapshot stays recorded as provenance.
    assert decision["checkpoint"]["total_added"] == 8


def test_head_and_size_stay_consistent_after_resume(tmp_path):
    trainer = _trainer(tmp_path, "consistent")
    trainer.replay_buffer.add_positions(_positions(6))
    ckpt = trainer.save_checkpoint(tmp_path / "consistent.pt", keep_last_n=-1)
    buf = _reload(tmp_path, "consistent")
    buf.add_positions(_positions(9, start=50))
    trainer2 = _trainer_on(tmp_path, buf, "consistent2")
    trainer2.load_checkpoint(ckpt)
    assert buf.write_head == buf.size
    assert buf.total_added == buf.size


def test_older_live_buffer_is_refused(tmp_path):
    """A checkpoint that expects more records than the live buffer holds fails closed."""
    trainer = _trainer(tmp_path, "older")
    trainer.replay_buffer.add_positions(_positions(30))
    trainer.replay_buffer.total_added = 30
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "older.pt", keep_last_n=-1)

    # Simulate a rewound/emptied live buffer: 5 records instead of 30.
    buf_path = tmp_path / "buf_older" / "meta.json"
    meta = json.loads(buf_path.read_text(encoding="utf-8"))
    meta.update({"write_head": 5, "size": 5, "total_added": 5})
    buf_path.write_text(json.dumps(meta), encoding="utf-8")
    buf = _reload(tmp_path, "older")
    trainer2 = _trainer_on(tmp_path, buf, "older2")

    with pytest.raises(BufferRestoreError, match="older than the checkpoint"):
        trainer2.load_checkpoint(ckpt)
    # Nothing was invented: the live buffer is untouched.
    assert buf.size == 5


def test_empty_live_buffer_with_checkpoint_records_is_refused(tmp_path):
    """Same generation, zero records: the checkpoint's pointer must not be adopted."""
    trainer = _trainer(tmp_path, "empty")
    trainer.replay_buffer.add_positions(_positions(12))
    ckpt = trainer.save_checkpoint(tmp_path / "empty.pt", keep_last_n=-1)
    saved = torch.load(ckpt, weights_only=False)

    # Same generation marker, but the buffer holds no records (an in-place reset
    # or a wiped array set with a surviving meta identity).
    meta_path = tmp_path / "buf_empty" / "meta.json"
    buf_dir = tmp_path / "buf_empty"
    buf_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "capacity": 64,
                "in_channels": 19,
                "height": 8,
                "width": 8,
                "num_actions": 4672,
                "write_head": 0,
                "size": 0,
                "total_added": 0,
                "tactical_pool_size": 0,
                "buffer_generation": saved["buffer_state"]["generation"],
            }
        ),
        encoding="utf-8",
    )
    fresh = ReplayBuffer(buffer_dir=buf_dir, capacity=64, create_new=False)
    assert fresh.total_added == 0
    trainer2 = _trainer_on(tmp_path, fresh, "empty2")
    with pytest.raises(BufferRestoreError, match="live replay buffer is empty"):
        trainer2.load_checkpoint(ckpt)
    assert fresh.size == 0, "no records may be invented from the checkpoint"


def test_generation_mismatch_is_refused_and_reset_is_explicit(tmp_path):
    trainer = _trainer(tmp_path, "gen_mismatch")
    trainer.replay_buffer.add_positions(_positions(10))
    ckpt = trainer.save_checkpoint(tmp_path / "gen_mismatch.pt", keep_last_n=-1)

    # A different generation in the same directory (buffer was reset).
    other = ReplayBuffer(buffer_dir=tmp_path / "buf_gen_mismatch", capacity=64, create_new=True)
    other.add_positions(_positions(3, start=900))
    trainer2 = _trainer_on(tmp_path, other, "gen_mismatch2")
    with pytest.raises(BufferRestoreError, match="generation"):
        trainer2.load_checkpoint(ckpt)

    # The explicit acknowledgement path: continue from the live buffer as-is.
    trainer2.load_checkpoint(ckpt, reset_buffer=True)
    decision = trainer2.run_metadata["buffer_resume_decision"]
    assert decision["decision"] == "reset_buffer_explicit"
    assert other.size == 3 and other.total_added == 3


def test_legacy_checkpoint_without_buffer_state_leaves_live_state_alone(tmp_path):
    trainer = _trainer(tmp_path, "legacy_buf")
    trainer.replay_buffer.add_positions(_positions(10))
    trainer.train_step()
    ckpt = trainer.save_checkpoint(tmp_path / "legacy_buf.pt", keep_last_n=-1)
    payload = torch.load(ckpt, weights_only=False)
    # Emulate a genuine legacy artifact: no buffer snapshot AND the pre-format-2
    # marker (a format-2 payload missing a required field is correctly refused).
    payload.pop("buffer_state")
    payload["checkpoint_format"] = 1
    torch.save(payload, ckpt)

    buf = _reload(tmp_path, "legacy_buf")
    trainer2 = _trainer_on(tmp_path, buf, "legacy_buf2")
    trainer2.load_checkpoint(ckpt)
    assert buf.size == 10 and buf.write_head == 10
    # The legacy head/size pair is reconciled like a snapshot, and the decision
    # record marks it as derived from legacy fields (no generation/triple).
    decision = trainer2.run_metadata["buffer_resume_decision"]
    assert decision["decision"] == "identical_snapshot"
    assert decision["checkpoint"]["legacy"] is True
    assert decision["checkpoint"]["generation"] is None


def test_legacy_pointer_only_checkpoint_cannot_rewind_a_newer_buffer(tmp_path):
    """The pre-remediation behavior (head-only override) is now impossible."""
    trainer = _trainer(tmp_path, "legacy_ptr")
    trainer.replay_buffer.add_positions(_positions(6))
    ckpt = trainer.save_checkpoint(tmp_path / "legacy_ptr.pt", keep_last_n=-1)
    payload = torch.load(ckpt, weights_only=False)
    # Emulate a legacy artifact: head/size only, head deliberately stale.
    payload.pop("buffer_state")
    payload["buffer_write_head"] = 2
    payload["buffer_size"] = 6
    payload["checkpoint_format"] = 1
    torch.save(payload, ckpt)

    buf = _reload(tmp_path, "legacy_ptr")
    buf.add_positions(_positions(10, start=200))
    assert buf.total_added == 16
    trainer2 = _trainer_on(tmp_path, buf, "legacy_ptr2")
    trainer2.load_checkpoint(ckpt)
    # The live (newer) state wins; head is NOT rewound to 2.
    assert buf.write_head == 16
    assert buf.size == 16
    assert trainer2.run_metadata["buffer_resume_decision"]["decision"] == "adopted_live_newer"


def test_wraparound_keeps_the_ring_consistent(tmp_path):
    capacity = 16
    trainer = _trainer(tmp_path, "wrap", capacity=capacity)
    trainer.replay_buffer.add_positions(_positions(10))
    ckpt = trainer.save_checkpoint(tmp_path / "wrap.pt", keep_last_n=-1)

    buf = _reload(tmp_path, "wrap", capacity=capacity)
    buf.add_positions(_positions(12, start=500))  # wraps past capacity
    assert buf.total_added == 22
    assert buf.size == capacity
    assert buf.write_head == 22 % capacity

    trainer2 = _trainer_on(tmp_path, buf, "wrap2")
    trainer2.load_checkpoint(ckpt)
    assert buf.size == capacity
    assert buf.write_head == 22 % capacity

    # A resumed train step still samples from a consistent ring.
    trainer2.replay_buffer.add_positions(_positions(1, start=999))
    assert buf.write_head == 23 % capacity
    assert buf.size == capacity


def test_full_buffer_resume(tmp_path):
    capacity = 8
    trainer = _trainer(tmp_path, "full", capacity=capacity)
    trainer.replay_buffer.add_positions(_positions(capacity))
    ckpt = trainer.save_checkpoint(tmp_path / "full.pt", keep_last_n=-1)
    buf = _reload(tmp_path, "full", capacity=capacity)
    trainer2 = _trainer_on(tmp_path, buf, "full2")
    trainer2.load_checkpoint(ckpt)
    assert buf.size == capacity and buf.write_head == 0  # exactly one full lap


def test_corrupt_live_meta_fails_closed_on_buffer_open(tmp_path):
    trainer = _trainer(tmp_path, "corrupt_meta")
    trainer.replay_buffer.add_positions(_positions(4))
    ckpt = trainer.save_checkpoint(tmp_path / "corrupt_meta.pt", keep_last_n=-1)

    meta_path = tmp_path / "buf_corrupt_meta" / "meta.json"
    meta_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt replay meta"):
        ReplayBuffer(buffer_dir=tmp_path / "buf_corrupt_meta", capacity=64, create_new=False)
    # The corrupt file was preserved as evidence, not silently discarded.
    backups = list((tmp_path / "buf_corrupt_meta").glob("meta.corrupt.*.json"))
    assert backups, "a corrupt meta must be backed up before raising"


def test_checkpoint_carries_capacity_and_generation_for_compatibility_checks(tmp_path):
    trainer = _trainer(tmp_path, "compat", capacity=40)
    trainer.replay_buffer.add_positions(_positions(7))
    ckpt = trainer.save_checkpoint(tmp_path / "compat.pt", keep_last_n=-1)
    state = torch.load(ckpt, weights_only=False)["buffer_state"]
    assert state["capacity"] == 40
    assert state["generation"] == trainer.replay_buffer.buffer_generation
    assert set(state) >= {"write_head", "size", "total_added", "capacity", "generation"}
