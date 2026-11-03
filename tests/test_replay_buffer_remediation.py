"""Regression tests for the replay-buffer remediation (SR-04..SR-13).

Every test uses temporary directories and synthetic records; the historical
production buffer is never opened, written, or migrated.

Findings covered:

- SR-04  checkpoint<->buffer snapshot contract (classification, generation).
- SR-05  unified staleness semantics across the ring and GPU-cache paths.
- SR-06  GPU-cache coverage split (ring_uniform reaches older rows;
         cache_uniform documents the historical newest-rows bias).
- SR-07  insertion-time record validation (transactional, no partial writes).
- SR-10  schema version, exact shard sizes, scalar-domain load validation,
         checksum round-trip, missing/oversized/zero-filled shard rejection.
- SR-11  oversized-flush accounting (total_added counts every offered record).
- SR-13  game-id preservation through disk ring, GPU cache and sampled batches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from python.selfplay.replay_buffer import (
    NO_GAME_ID,
    SCHEMA_VERSION,
    BufferIntegrityError,
    GpuResidentReplayCache,
    ReplayBuffer,
    ReplayRecordValidationError,
    STALENESS_WINDOW,
)
from python.selfplay.worker import SavedPosition


def _one_hot(slot: int, num_actions: int = 4672) -> np.ndarray:
    v = np.zeros(num_actions, dtype=np.float32)
    v[slot] = 1.0
    return v


def _pos(
    game_id: int,
    *,
    outcome: int = 1,
    version: int = 1,
    planes: np.ndarray | None = None,
    visits: np.ndarray | None = None,
    full_search: bool = True,
) -> SavedPosition:
    return SavedPosition(
        planes=planes if planes is not None else np.zeros((19, 8, 8), dtype=np.float32),
        visits=visits if visits is not None else _one_hot(game_id % 4672),
        outcome=outcome,
        is_full_search=full_search,
        model_version=version,
        game_id=game_id,
    )


# ---------------------------------------------------------------------------
# SR-07 — insertion validation
# ---------------------------------------------------------------------------


def test_valid_flush_is_accepted_and_meta_carries_schema_contract(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=16, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(5)])
    assert buf.size == 5 and buf.total_added == 5
    meta = json.loads((tmp_path / "b" / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["generation"] == 1
    assert meta["shards"]["planes"]["bytes"] == 16 * 19 * 8 * 8 * 4
    assert meta["buffer_id"] == buf.buffer_id
    assert buf.rejected_records == 0


@pytest.mark.parametrize(
    "mutator,description",
    [
        (lambda p: setattr(p, "planes", np.full((19, 8, 8), 2.0, np.float32)), "plane > 1"),
        (lambda p: setattr(p, "planes", np.full((19, 8, 8), np.nan, np.float32)), "NaN plane"),
        (
            lambda p: setattr(p, "visits", np.full(4672, 0.5, np.float32)),
            "visits not normalized (raw counts)",
        ),
        (
            lambda p: setattr(p, "visits", np.full(4672, np.nan, np.float32)),
            "NaN visits",
        ),
        (lambda p: setattr(p, "visits", _one_hot(3) * -1.0), "negative visits"),
        (lambda p: setattr(p, "outcome", 3), "outcome outside {0,1,2}"),
        (lambda p: setattr(p, "outcome", 255), "human sentinel in self-play ring"),
        (lambda p: setattr(p, "model_version", -1), "unknown provenance version"),
        (lambda p: setattr(p, "game_id", 0), "zero game id"),
        (lambda p: setattr(p, "planes", np.zeros((3, 6, 7), np.float32)), "wrong plane shape"),
    ],
)
def test_malformed_record_rejects_entire_flush_atomically(tmp_path, mutator, description):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=16, create_new=True)
    good = [_pos(1), _pos(2)]
    buf.add_positions(good)
    before_meta = (tmp_path / "b" / "meta.json").read_bytes()
    slot2_before = np.array(buf.planes[2])

    bad = _pos(3)
    mutator(bad)
    with pytest.raises(ReplayRecordValidationError):
        buf.add_positions([_pos(4), bad, _pos(5)])  # poison in the middle

    # Nothing was written: counters, meta and shard slots are untouched.
    assert buf.size == 2 and buf.write_head == 2 and buf.total_added == 2
    assert (tmp_path / "b" / "meta.json").read_bytes() == before_meta
    assert np.array_equal(np.array(buf.planes[2]), slot2_before)  # still the zero-fill
    assert buf.rejected_records == 3  # whole flush counted as rejected
    assert description  # parametrization id used in the report


def test_add_position_single_path_validates(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    bad = _pos(1)
    bad.outcome = 7
    with pytest.raises(ReplayRecordValidationError):
        buf.add_position(bad)
    assert buf.size == 0 and buf.rejected_records == 1
    buf.add_position(_pos(1))
    assert buf.size == 1


# ---------------------------------------------------------------------------
# SR-11 — oversized-flush accounting
# ---------------------------------------------------------------------------


def test_oversized_flush_counts_all_offered_records(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(20)])
    assert buf.size == 8
    assert buf.write_head == 0  # 8 written from slot 0 -> full wrap
    assert buf.total_added == 20  # SR-11: honest accounting, no understatement
    # Newest 8 records retained (tail-take), oldest 12 dropped.
    assert sorted(int(g) for g in buf.game_ids[:8]) == list(range(13, 21))


# ---------------------------------------------------------------------------
# SR-10 — schema version, shard sizes, scalar domains, checksums
# ---------------------------------------------------------------------------


def test_future_schema_version_is_rejected(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(1)])
    buf.close()
    meta_path = tmp_path / "b" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema_version"] = SCHEMA_VERSION + 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BufferIntegrityError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)


def test_oversized_and_truncated_shards_are_rejected(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(1)])
    buf.close()

    visits = tmp_path / "b" / "visits.bin"
    original = visits.stat().st_size
    with open(visits, "ab") as f:
        f.write(b"\0" * 16)  # oversized: previously opened silently
    with pytest.raises(BufferIntegrityError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)
    with open(visits, "r+b") as f:
        f.truncate(original - 16)  # truncated
    with pytest.raises(BufferIntegrityError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)


def test_zero_filled_scalar_shard_is_rejected(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(4)])
    buf.close()
    gids = tmp_path / "b" / "game_ids.bin"
    data = gids.read_bytes()
    gids.write_bytes(b"\0" * len(data))  # right size, wrong content
    with pytest.raises(BufferIntegrityError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)


def test_missing_shard_is_rejected(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(1)])
    buf.close()
    (tmp_path / "b" / "outcomes.bin").unlink()
    with pytest.raises(BufferIntegrityError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)


def test_checksum_round_trip_and_mismatch_detection(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(4)])
    sums = buf.write_shard_checksums()
    assert set(sums) == {"planes", "visits", "outcomes", "full_search", "versions", "game_ids"}

    report = buf.verify_shards(deep=True)
    assert report["ok"] is True, report["errors"]

    # Corrupt one byte inside the live region → deep verification catches it.
    planes_path = tmp_path / "b" / "planes.bin"
    with open(planes_path, "r+b") as f:
        f.seek(4)
        f.write(b"\x7f")
    buf.close()
    buf2 = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)
    report2 = buf2.verify_shards(deep=True)
    assert report2["ok"] is False
    assert any("checksum mismatch" in e for e in report2["errors"])
    # Load itself still succeeds (deep verification is explicit, not hot-path).
    assert report2["checks"]["checksums_verified"] is False


def test_reset_buffer_bumps_generation_and_keeps_loadable(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(4)])
    buf.close()
    assert buf.generation == 1

    buf2 = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)  # reset
    assert buf2.generation == 2
    assert buf2.size == 0 and buf2.total_added == 0
    # Reset semantics remain fast (no shard zeroing) and the stale tail is
    # reported, never sampled (bounds are [0, size)).
    report = buf2.verify_shards()
    assert report["ok"] is True
    assert report["checks"].get("tail_zero") == "non-zero-stale"
    assert any("slots at/after size" in w for w in report["warnings"])


def test_corrupt_meta_is_backed_up_and_never_silently_reset(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(4)])
    buf.close()
    meta = tmp_path / "b" / "meta.json"
    meta.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)
    backups = list((tmp_path / "b").glob("meta.corrupt.*.json"))
    assert backups, "corrupt meta must be preserved as forensic backup"
    assert backups[0].read_text(encoding="utf-8") == "{not json"
    # No silent reset: the buffer on disk still holds its records.
    assert (tmp_path / "b" / "game_ids.bin").stat().st_size == 8 * 8


def test_interrupted_meta_write_recovers_from_valid_scratch_file(tmp_path):
    """A crash between scratch write and os.replace leaves ``meta.tmp`` as the
    newest valid state; an empty meta recovers from it (existing path,
    regression-pinned against the ACTUAL scratch filename used by _save_meta).
    """
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(3)])
    buf.close()
    meta = tmp_path / "b" / "meta.json"
    payload = meta.read_text(encoding="utf-8")
    scratch = tmp_path / "b" / "meta.tmp"  # Path.with_suffix('.tmp') => meta.tmp
    scratch.write_text(payload, encoding="utf-8")
    meta.write_text("", encoding="utf-8")  # torn meta from an interrupted replace
    buf2 = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)
    assert buf2.size == 3 and buf2.total_added == 3


def test_missing_meta_on_populated_buffer_fails_closed(tmp_path):
    """REM-06: a lost meta must never silently reset a populated buffer."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(3)])
    buf.close()
    (tmp_path / "b" / "meta.json").unlink()
    with pytest.raises(BufferIntegrityError):
        ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8)
    # An explicitly requested reset is still allowed and starts generation 2.
    buf2 = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    assert buf2.generation == 2
    assert buf2.size == 0


def test_missing_meta_on_empty_buffer_initializes_fresh(tmp_path):
    """A brand-new directory (no shards, no meta) still initializes normally."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "fresh", capacity=8)
    assert buf.size == 0 and buf.generation == 1


# ---------------------------------------------------------------------------
# SR-04 — snapshot / consistency classification
# ---------------------------------------------------------------------------


def test_checkpoint_consistency_classification(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=16, create_new=True)
    buf.add_positions([_pos(i + 1) for i in range(4)])
    snap = buf.snapshot()
    assert snap["buffer_id"] == buf.buffer_id
    assert buf.checkpoint_consistency(snap)[0] == "match"

    # Appends continue after the checkpoint: live is verifiably "ahead".
    buf.add_positions([_pos(i + 10) for i in range(3)])
    verdict, detail = buf.checkpoint_consistency(snap)
    assert verdict == "ahead", detail

    # Same state again → match; then taxonomy of mismatches:
    assert buf.checkpoint_consistency(buf.snapshot())[0] == "match"
    assert buf.checkpoint_consistency({"buffer_id": "deadbeef"})[0] == "mismatch"
    assert buf.checkpoint_consistency({"capacity": 999})[0] == "mismatch"
    assert buf.checkpoint_consistency({**snap, "generation": 99})[0] == "mismatch"
    assert buf.checkpoint_consistency({})[0] == "mismatch"  # no snapshot at all
    # A rolled-back pointer (checkpoint "ahead" of live) is never a match.
    assert (
        buf.checkpoint_consistency({**snap, "size": 12, "write_head": 12, "total_added": 12})[0]
        == "mismatch"
    )
    # Legacy snapshot (no total_added): equality is required, 'ahead' impossible.
    legacy = {"size": 4, "write_head": 4}
    assert buf.checkpoint_consistency(legacy)[0] == "mismatch"


def test_legacy_checkpoint_without_buffer_snapshot_still_loads(tmp_path):
    """A checkpoint written before SR-04 (no buffer metadata at all) is not
    a mismatch — there is simply nothing to verify; the live state is kept."""
    from python.train.train_rl import RLTrainer
    from python.model.chess_net import ChessNet

    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=32, create_new=True)
    buf.add_positions([_pos(i + 1, version=1) for i in range(8)])
    model = ChessNet(num_blocks=2, num_filters=16)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        device="cpu",
        batch_size=4,
        warmup_steps=2,
        total_steps=10,
        checkpoint_dir=tmp_path / "ckpt",
        sync_weights_path=tmp_path / "sync.pt",
    )
    import torch

    ckpt = {
        "checkpoint_type": "rl",
        "model": model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "step": 3,
    }
    p = tmp_path / "legacy.pt"
    torch.save(ckpt, p)
    loaded = trainer.load_checkpoint(p)
    assert loaded == 3
    assert buf.size == 8  # live buffer untouched


# ---------------------------------------------------------------------------
# SR-05 — unified staleness semantics
# ---------------------------------------------------------------------------


def _stale_fresh_buffer(tmp_path, name="b") -> ReplayBuffer:
    buf = ReplayBuffer(buffer_dir=tmp_path / name, capacity=64, create_new=True)
    stale = [_pos(i + 1, version=1) for i in range(10)]  # game_ids 1..10
    fresh = [_pos(i + 101, version=30) for i in range(10)]  # game_ids 101..110
    buf.add_positions(stale + fresh)
    return buf


def test_ring_staleness_filter_matches_cache_contract(tmp_path):
    buf = _stale_fresh_buffer(tmp_path)
    assert int(buf.versions[: buf.size].max()) == 30
    for _ in range(20):
        batch, _mix = buf.sample_batch_tensors(batch_size=8, device="cpu")
        sampled = batch["game_ids"]
        assert int(sampled.min()) >= 101  # stale rows (v=1, window=20) never drawn
    assert buf.staleness_fallback_count == 0  # enough fresh rows existed


def test_ring_and_cache_apply_the_same_staleness_rule(tmp_path):
    buf = _stale_fresh_buffer(tmp_path, name="ring")
    cache = GpuResidentReplayCache(capacity=64, device="cpu")
    versions = np.array([1] * 10 + [30] * 10, dtype=np.int32)
    game_ids = np.array(list(range(1, 11)) + list(range(101, 111)), dtype=np.int64)
    cache.add_positions(
        planes=np.zeros((20, 19, 8, 8), np.float32),
        visits=np.zeros((20, 4672), np.float32),
        outcomes=np.ones(20, dtype=np.int64),
        full_search=np.ones(20, dtype=np.bool_),
        versions=versions,
        game_ids=game_ids,
    )
    for _ in range(10):
        cbatch = cache.sample_batch(8)
        assert int(cbatch["game_ids"].min()) >= 101
        rbatch, _ = buf.sample_batch_tensors(batch_size=8, device="cpu")
        assert int(rbatch["game_ids"].min()) >= 101
    assert cache.staleness_fallback_count == 0


def test_staleness_window_boundary_is_exact(tmp_path):
    """A row exactly STALENESS_WINDOW behind the max is eligible; one more is not."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=64, create_new=True)
    window = STALENESS_WINDOW
    rows = [_pos(1, version=100 - window)] + [_pos(2, version=100)]  # boundary row + max
    buf.add_positions(rows)
    for _ in range(10):
        batch, _ = buf.sample_batch_tensors(batch_size=1, device="cpu")
        assert int(batch["game_ids"][0]) in (1, 2)  # boundary row still eligible
    assert buf.staleness_fallback_count == 0

    buf2 = ReplayBuffer(buffer_dir=tmp_path / "b2", capacity=64, create_new=True)
    buf2.add_positions([_pos(1, version=100 - window - 1), _pos(2, version=100)])
    for _ in range(10):
        batch, _ = buf2.sample_batch_tensors(batch_size=1, device="cpu")
        assert int(batch["game_ids"][0]) == 2  # one step beyond the window: excluded


def test_staleness_fallback_is_counted_and_documented(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=64, create_new=True)
    buf.add_positions([_pos(1, version=1), _pos(2, version=100)])
    batch, _ = buf.sample_batch_tensors(batch_size=8, device="cpu")  # > fresh rows
    assert batch["planes"].shape[0] == 8
    assert buf.staleness_fallback_count >= 1
    # Fallback samples the whole live population (both rows reachable).
    seen = set()
    for _ in range(40):
        b, _ = buf.sample_batch_tensors(batch_size=4, device="cpu")
        seen.update(int(x) for x in b["game_ids"])
    assert seen == {1, 2}


def test_ring_rejects_unknown_version_records(tmp_path):
    """Version -1 is a cache-only marker; the ring refuses it at insertion."""
    buf = ReplayBuffer(buffer_dir=tmp_path / "b", capacity=8, create_new=True)
    with pytest.raises(ReplayRecordValidationError):
        buf.add_positions([_pos(1, version=-1)])


# ---------------------------------------------------------------------------
# SR-06 — cache coverage policy
# ---------------------------------------------------------------------------


def test_ring_uniform_policy_reaches_rows_older_than_the_cache(tmp_path):
    """Default policy: the combined cache+ring distribution stays uniform."""
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "b",
        capacity=100,
        gpu_cache_capacity=20,
        gpu_device="cpu",
        create_new=True,
    )
    buf.add_positions([_pos(i + 1, version=1) for i in range(100)])
    assert buf.gpu_cache is not None and len(buf.gpu_cache) == 20
    assert buf.gpu_cache_coverage == pytest.approx(0.2)

    seen = set()
    for _ in range(200):
        batch, _mix = buf.sample_batch_tensors(batch_size=64, step=0, device="cpu")
        seen.update(int(g) for g in batch["game_ids"])
    # Rows 81..100 live in the cache; rows 1..80 can only come from the ring.
    assert seen & set(range(1, 81)), "older-than-cache rows were never sampled (bias!)"
    assert seen & set(range(81, 101))


def test_cache_uniform_policy_documents_historical_bias(tmp_path):
    """`cache_uniform` reproduces the historical behavior: newest-rows-only."""
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "b",
        capacity=100,
        gpu_cache_capacity=20,
        gpu_device="cpu",
        create_new=True,
        selfplay_sampling_policy="cache_uniform",
    )
    buf.add_positions([_pos(i + 1, version=1) for i in range(100)])
    # Historical behavior: whenever the cache holds enough rows for the
    # request, every self-play row comes from it (batch 16 <= cache 20).
    seen = set()
    for _ in range(50):
        batch, _mix = buf.sample_batch_tensors(batch_size=16, step=0, device="cpu")
        seen.update(int(g) for g in batch["game_ids"])
    assert seen <= set(range(81, 101)), "cache_uniform must draw only from the cache"
    assert buf.last_selfplay_source == "gpu_cache"


def test_unknown_sampling_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ReplayBuffer(
            buffer_dir=tmp_path / "b",
            capacity=8,
            create_new=True,
            selfplay_sampling_policy="whatever",
        )


# ---------------------------------------------------------------------------
# SR-13 — game-id provenance through every path
# ---------------------------------------------------------------------------


def test_gpu_cache_wraparound_keeps_row_identity(tmp_path):
    """After cache wraparound, sampled game_ids still match their own rows."""
    cache = GpuResidentReplayCache(capacity=8, device="cpu")
    for base in (0, 6):  # 12 rows into an 8-slot cache → wraparound
        n = 6
        planes = np.zeros((n, 19, 8, 8), np.float32)
        gids = np.arange(base + 1, base + 1 + n, dtype=np.int64)
        planes[:, 0, 0, 0] = gids  # identity carrier (cache performs no validation)
        cache.add_positions(
            planes=planes,
            visits=np.zeros((n, 4672), np.float32),
            outcomes=np.ones(n, dtype=np.int64),
            full_search=np.ones(n, dtype=np.bool_),
            versions=np.ones(n, dtype=np.int32),
            game_ids=gids,
        )
    batch = cache.sample_batch(16)
    carried = batch["planes"][:, 0, 0, 0].to(torch_long := __import__("torch").long)
    assert (carried == batch["game_ids"]).all()
    assert int(batch["game_ids"].min()) >= 5  # only the newest 8 rows survived


def test_sampled_batch_preserves_game_ids_end_to_end(tmp_path):
    """Ring -> sample_batch_tensors: game_ids identify the producing rows."""
    buf = ReplayBuffer(
        buffer_dir=tmp_path / "b",
        capacity=50,
        gpu_cache_capacity=50,
        gpu_device="cpu",
        create_new=True,
    )
    # Row identity: game_id N is encoded as visits one-hot at slot N-11.
    buf.add_positions([_pos(i + 11, version=1) for i in range(50)])
    for _ in range(5):
        batch, _mix = buf.sample_batch_tensors(batch_size=16, step=0, device="cpu")
        # _pos(game_id) encodes identity as a one-hot at slot == game_id.
        decoded = batch["visits"].argmax(dim=-1)
        assert (decoded == batch["game_ids"]).all()

    # Human/tactical rows carry the documented no-game sentinel.
    batch_np, _ = buf.sample_batch(batch_size=8, step=0)
    assert set(np.unique(batch_np["game_ids"]).tolist()) <= set(range(11, 61))


def test_cache_rows_without_ids_carry_no_game_sentinel(tmp_path):
    cache = GpuResidentReplayCache(capacity=4, device="cpu")
    cache.add_positions(
        planes=np.zeros((2, 19, 8, 8), np.float32),
        visits=np.zeros((2, 4672), np.float32),
        outcomes=np.ones(2, dtype=np.int64),
        full_search=np.ones(2, dtype=np.bool_),
        # no versions / no game ids: unknown provenance markers
    )
    batch = cache.sample_batch(2)
    assert (batch["game_ids"] == NO_GAME_ID).all()
