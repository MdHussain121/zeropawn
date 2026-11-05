"""SA5-03 remediation tests: the configured seed must reach the batch samplers.

Audit SA5-03 (Low, 2026-11-05 pre-push): "batch sampling not seed-reproducible".
Before the fix, ``train_supervised`` / ``train_opening_policy`` seeded the
process-global RNGs but never handed a generator to the DataLoader or the
``WeightedRandomSampler`` — so the shuffle permutation drew its seed from the
global torch RNG at each epoch, silently depending on whatever else consumed
that RNG. ``GpuResidentDataLoader`` likewise used ``torch.randperm`` /
``torch.multinomial`` with no generator, and the finetune / mistake-learner
paths had no seed at all.

DETERMINISM CONTRACT (the actual guarantee after the fix — stated honestly):

1. Batch ORDER: same seed + same platform + same worker count ⇒ the same
   batch order, for every training epoch. The order for epoch ``e`` is a pure
   function of ``(seed, e)``; epochs are decorrelated (``seed + e`` reseed).
   Different seeds ⇒ different order (asserted on fixed synthetic data).
2. Resume: ``train_supervised`` already checkpoints/restores the torch, CUDA,
   numpy, and python global RNG states. Because batch order is derived from
   ``(seed, epoch)`` and the epoch counter is checkpointed, a resumed run
   replays the exact batch order an uninterrupted run used for the same
   epochs — verified below. Honest limitation: an epoch interrupted mid-way
   restarts from its beginning; the samples of the already-consumed partial
   epoch are not re-delivered (the permutation is drawn per whole epoch).
3. Workers: with ``num_workers > 0`` each worker's numpy/python RNG streams
   derive from the loader generator via ``seed_worker`` (base seed +
   worker_id) — identical across identical runs, distinct per worker and per
   seed. Verified with real DataLoader workers below.
4. Unseeded runs (``seed=None``) intentionally keep the legacy entropy-driven
   behavior and are NOT reproducible. No claim is made for them.
5. What is NOT guaranteed: model update numerics. cuDNN benchmark mode,
   TF32/bfloat16 reductions, fused-optimizer atomics, and CUDA scheduling
   introduce hardware-level nondeterminism in gradient/weight values that no
   sampler seeding can remove. This contract covers batch order and sampling
   composition only — bitwise-identical trained weights across machines or
   runs are explicitly NOT claimed.

All fixtures are hermetic (``tmp_path`` synthetic binary datasets); no file
under ``data/`` or ``models/`` is read or written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import python.train.train_supervised as ts_module
from python.data.dataset import (
    BinaryChessDataset,
    GpuResidentDataLoader,
    SourceAwareCombinedDataset,
    seed_worker,
)
from python.model.chess_net import ChessNet
from python.train.learn_from_mistakes import MistakeLearner, PuzzleItem

PLANES_SHAPE = (19, 8, 8)


# ───────────────────── hermetic synthetic dataset writer ─────────────────────


def _write_split(split_dir: Path, n: int, *, is_tactical: bool = False, seed: int = 0) -> Path:
    """Write a tiny valid binary split; game_ids == row index so batch order
    is fully observable through ``batch["game_id"]`` / ``batch["moves"]``."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    planes = rng.random((n, *PLANES_SHAPE), dtype=np.float32)
    planes.tofile(split_dir / "planes.bin")
    np.arange(n, dtype=np.int16).tofile(split_dir / "moves.bin")
    if is_tactical:
        outcomes = np.full(n, 255, dtype=np.uint8)
        values = np.zeros((n, 3), dtype=np.float32)
        plies = np.zeros(n, dtype=np.int16)
    else:
        outcomes = np.array([i % 3 for i in range(n)], dtype=np.uint8)
        values = np.zeros((n, 3), dtype=np.float32)
        values[np.arange(n), outcomes] = 1.0
        plies = np.full(n, 20, dtype=np.int16)
    outcomes.tofile(split_dir / "outcomes.bin")
    values.tofile(split_dir / "value_targets.bin")
    plies.tofile(split_dir / "plies.bin")
    np.arange(n, dtype=np.uint64).tofile(split_dir / "game_ids.bin")
    meta = {
        "num_samples": n,
        "split": split_dir.name,
        "planes_shape": list(PLANES_SHAPE),
        "role": "policy_tactics" if is_tactical else "games_wdl",
        "kind": "tactics-fullline" if is_tactical else "games",
        "label_semantics": {
            "role": "policy_tactics" if is_tactical else "games_wdl",
            "wdl_supervised": not is_tactical,
        },
    }
    import json

    with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return split_dir


def _flat_game_order(batches: list[dict]) -> list[int]:
    return [int(g) for b in batches for g in b["game_id"].tolist()]


# ─────────────────── 1. WeightedRandomSampler via get_sampler ───────────────────


def test_get_sampler_same_generator_seed_same_stream(tmp_path):
    """Two samplers built with identically-seeded generators draw the same
    index stream; a different seed draws a different one (SA5-03 core)."""
    master = BinaryChessDataset(_write_split(tmp_path / "m", 40, seed=1))
    tactics = BinaryChessDataset(_write_split(tmp_path / "t", 20, is_tactical=True, seed=2))
    comb = SourceAwareCombinedDataset(master, tactics, tactics_ratio=0.3)

    def stream(seed: int) -> list[int]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return [int(i) for i in iter(comb.get_sampler(generator=gen))]

    assert stream(5) == stream(5), "same generator seed must reproduce the stream"
    assert stream(5) != stream(6), "different generator seed must change the stream"


def test_get_sampler_without_generator_still_works(tmp_path):
    """generator=None keeps the legacy (unseeded) construction path alive."""
    master = BinaryChessDataset(_write_split(tmp_path / "m", 8, seed=3))
    tactics = BinaryChessDataset(_write_split(tmp_path / "t", 8, is_tactical=True, seed=4))
    comb = SourceAwareCombinedDataset(master, tactics, tactics_ratio=0.5)
    indices = [int(i) for i in iter(comb.get_sampler())]
    assert len(indices) == len(comb)


# ─────────────── 2. DataLoader shuffle + per-epoch reseed contract ───────────────


def _shuffled_loader_orders(ds: BinaryChessDataset, seed: Optional[int], epochs: int):
    """Mirror the exact per-epoch contract train_supervised uses: one seeded
    generator handed to the DataLoader, reseeded to seed+epoch each epoch."""
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        generator=gen,
        worker_init_fn=seed_worker,
        num_workers=0,
        drop_last=True,
    )
    orders = []
    for epoch in range(epochs):
        if gen is not None:
            gen.manual_seed(int(seed) + epoch)
        orders.append(_flat_game_order(list(loader)))
    return orders


def test_dataloader_shuffle_same_seed_same_order_every_epoch(tmp_path):
    ds = BinaryChessDataset(_write_split(tmp_path / "s", 16, seed=5))
    a = _shuffled_loader_orders(ds, 123, epochs=2)
    b = _shuffled_loader_orders(ds, 123, epochs=2)
    assert a == b, "same seed must give the same batch order for every epoch"
    # Every sample still appears exactly once per epoch (drop_last=True, 16/4).
    assert sorted(a[0]) == list(range(16))


def test_dataloader_shuffle_epochs_decorrelated_and_seeds_differ(tmp_path):
    ds = BinaryChessDataset(_write_split(tmp_path / "s", 16, seed=6))
    a = _shuffled_loader_orders(ds, 123, epochs=2)
    c = _shuffled_loader_orders(ds, 124, epochs=2)
    assert a[0] != a[1], "consecutive epochs must not replay the same permutation"
    assert a[0] != c[0] and a[1] != c[1], "different seed must change batch order"


def test_dataloader_unseeded_still_yields_full_permutation(tmp_path):
    ds = BinaryChessDataset(_write_split(tmp_path / "s", 16, seed=7))
    orders = _shuffled_loader_orders(ds, None, epochs=2)
    assert sorted(orders[0]) == list(range(16))


# ───────────────────── 3. per-worker seeding (real workers) ─────────────────────


class _RngProbeDataset(Dataset):
    """Returns (idx, numpy draw) — exposes each worker's numpy RNG stream."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "draw": torch.tensor(int(np.random.randint(0, 1_000_000)), dtype=torch.long),
        }


def _probe_worker_draws(seed: int) -> dict[int, int]:
    ds = _RngProbeDataset(24)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=gen,
    )
    draws: dict[int, int] = {}
    for batch in loader:
        for idx, draw in zip(batch["idx"].tolist(), batch["draw"].tolist()):
            draws[int(idx)] = int(draw)
    return draws


def test_worker_rng_streams_reproducible_with_seed():
    """With seed_worker + a seeded loader generator, the RNG draws performed
    inside worker processes are identical across identical runs and change
    with the seed (audit SA5-03 'worker behavior where applicable')."""
    a = _probe_worker_draws(7)
    b = _probe_worker_draws(7)
    c = _probe_worker_draws(8)
    assert a == b, "same seed must reproduce worker-side RNG draws"
    differing = sum(1 for k in a if a[k] != c[k])
    assert differing > 0, "different seed must change worker-side RNG draws"


# ───────────────────── 4. GpuResidentDataLoader seeding ─────────────────────


def _gpu_loader_orders(data_dir: Path, seed: Optional[int], epochs) -> list[list[int]]:
    loader = GpuResidentDataLoader(data_dir, batch_size=4, shuffle=True, device="cpu", seed=seed)
    try:
        orders = []
        for epoch in epochs:
            loader.set_epoch(epoch)
            # The GPU loader yields "move" (int16 row index == game_id here).
            orders.append([int(m) for batch in loader for m in batch["move"].tolist()])
        return orders
    finally:
        loader.close()


def test_gpu_resident_loader_seed_reproducible_per_epoch(tmp_path):
    data = _write_split(tmp_path / "g", 16, seed=8)
    a = _gpu_loader_orders(data, 123, (0, 1))
    b = _gpu_loader_orders(data, 123, (0, 1))
    c = _gpu_loader_orders(data, 124, (0, 1))
    assert a == b, "same seed must reproduce the GPU loader batch order"
    assert a[0] != a[1], "set_epoch must decorrelate epochs"
    assert a[0] != c[0] and a[1] != c[1], "different seed must change batch order"


def test_gpu_resident_loader_resume_alignment(tmp_path):
    """A fresh loader aligned to the checkpointed epoch replays the same
    order the uninterrupted run used for that epoch (resume contract)."""
    data = _write_split(tmp_path / "g", 16, seed=9)
    full = _gpu_loader_orders(data, 123, (0, 1, 2, 3))
    resumed = _gpu_loader_orders(data, 123, (2, 3))
    assert resumed == full[2:], "resumed epochs must match the uninterrupted order"


def test_gpu_resident_loader_weighted_branch_seeded(tmp_path):
    """The sample_weights (multinomial) branch honors the seed too."""
    data = _write_split(tmp_path / "g", 16, seed=10)
    weights = torch.rand(16)

    def collect(seed: int) -> list[int]:
        loader = GpuResidentDataLoader(data, batch_size=4, shuffle=True, device="cpu", seed=seed)
        loader.sample_weights = weights.clone()
        try:
            return [int(m) for batch in loader for m in batch["move"].tolist()]
        finally:
            loader.close()

    assert collect(5) == collect(5)
    assert collect(5) != collect(6)


def test_gpu_resident_loader_unseeded_full_coverage(tmp_path):
    data = _write_split(tmp_path / "g", 16, seed=11)
    orders = _gpu_loader_orders(data, None, (0,))
    assert sorted(orders[0]) == list(range(16))


# ───────────── 5. integration: real train_supervised batch order ─────────────


@pytest.fixture
def batch_order_spy(monkeypatch):
    """Record the game_id order of every micro-batch reaching train_step,
    while keeping the real optimizer step behavior intact."""
    real = ts_module.train_step
    recorded: list[list[int]] = []

    def spy(*args, **kwargs):
        batch = kwargs["batch"] if "batch" in kwargs else args[1]
        recorded.append([int(g) for g in batch["game_id"].tolist()])
        return real(*args, **kwargs)

    monkeypatch.setattr(ts_module, "train_step", spy)
    return recorded


def _run_supervised(
    root: Path,
    name: str,
    *,
    epochs: int,
    seed: Optional[int],
    spy: list,
    resume_path: Optional[Path] = None,
) -> Path:
    from python.train.train_supervised import train_supervised

    train_dir = _write_split(root / f"data_{name}" / "train", 32, seed=12)
    val_dir = _write_split(root / f"data_{name}" / "val", 8, seed=13)
    ckpt_dir = root / f"ckpt_{name}"
    train_supervised(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=8,
        micro_batch_size=8,
        base_lr=1e-4,
        warmup_steps=0,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        log_dir=root / f"runs_{name}",
        num_workers=0,
        gpu_resident=False,
        num_blocks=1,
        num_filters=8,
        seed=seed,
        allow_uncertified_data=True,
        verify_identity="none",
        resume_path=str(resume_path) if resume_path else None,
    )
    # 32 samples / batch 8, accum 1 → exactly 4 micro-batches per epoch.
    assert len(spy) % 4 == 0, f"expected whole epochs of 4 batches, got {len(spy)}"
    return ckpt_dir


def test_supervised_same_seed_same_batch_order(tmp_path, batch_order_spy):
    """SA5-03 core: the configured seed reaches the actual training sampler —
    two identical-seed runs produce identical batch order from epoch 0."""
    _run_supervised(tmp_path, "a", epochs=2, seed=123, spy=batch_order_spy)
    first = list(batch_order_spy)
    batch_order_spy.clear()
    _run_supervised(tmp_path, "b", epochs=2, seed=123, spy=batch_order_spy)
    second = list(batch_order_spy)
    assert first == second, "same seed must give identical batch order"
    assert first[0] != first[4], "epoch 1 must not replay epoch 0's order"


def test_supervised_different_seed_different_batch_order(tmp_path, batch_order_spy):
    _run_supervised(tmp_path, "a", epochs=1, seed=123, spy=batch_order_spy)
    first = list(batch_order_spy)
    batch_order_spy.clear()
    _run_supervised(tmp_path, "b", epochs=1, seed=124, spy=batch_order_spy)
    second = list(batch_order_spy)
    assert first != second, "different seed must change the training batch order"


def test_supervised_resume_replays_uninterrupted_batch_order(tmp_path, batch_order_spy):
    """Resume honesty check: batch order for epochs 2-3 after an
    interrupted-then-resumed run equals the uninterrupted run's epochs 2-3,
    because order = f(seed, epoch) and the epoch counter is checkpointed.

    Documented limitation: a partially-consumed epoch restarts from its
    beginning on resume; its leftover samples are not re-delivered.
    """
    # Uninterrupted 4-epoch run.
    _run_supervised(tmp_path, "full", epochs=4, seed=123, spy=batch_order_spy)
    full = list(batch_order_spy)
    assert len(full) == 16

    # Interrupted-style 2-epoch run (same seed): epochs 0-1 must match.
    batch_order_spy.clear()
    ckpt_short = _run_supervised(tmp_path, "short", epochs=2, seed=123, spy=batch_order_spy)
    short = list(batch_order_spy)
    assert short == full[:8]

    # Resume to 4 epochs: epochs 2-3 must match the uninterrupted run.
    batch_order_spy.clear()
    _run_supervised(
        tmp_path,
        "resumed",
        epochs=4,
        seed=123,
        spy=batch_order_spy,
        resume_path=ckpt_short / "latest_checkpoint.pt",
    )
    resumed = list(batch_order_spy)
    assert len(resumed) == 8
    assert resumed == full[8:], "resumed epochs must replay the uninterrupted order"


def test_supervised_unseeded_run_still_trains(tmp_path, batch_order_spy):
    """seed=None preserves the legacy entropy-driven path (no false claim)."""
    _run_supervised(tmp_path, "u", epochs=1, seed=None, spy=batch_order_spy)
    assert len(batch_order_spy) == 4
    seen = sorted(g for b in batch_order_spy for g in b)
    assert seen == list(range(32))


# ───────────── 6. integration: train_opening_policy batch order ─────────────


def _opening_base_checkpoint(path: Path) -> None:
    from python.train.train_supervised import _atomic_torch_save

    torch.manual_seed(0)
    model = ChessNet(num_blocks=1, num_filters=8)
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "num_blocks": 1,
            "num_filters": 8,
            "checkpoint_type": "supervised",
        },
        path,
    )


def test_opening_policy_seed_reaches_sampler(tmp_path, monkeypatch):
    """The opening-policy loaders receive the configured seed: identical-seed
    runs draw identical batch order, a different seed does not."""
    import python.train.train_opening_policy as op_module
    from python.train.train_opening_policy import train_opening_policy

    real_step = op_module.train_opening_step
    recorded: list[list[int]] = []

    def spy(*args, **kwargs):
        batch = kwargs["batch"] if "batch" in kwargs else args[1]
        recorded.append([int(g) for g in batch["game_id"].tolist()])
        return real_step(*args, **kwargs)

    monkeypatch.setattr(op_module, "train_opening_step", spy)
    base = tmp_path / "base.pt"
    _opening_base_checkpoint(base)

    def run(tag: str, seed: int) -> list[list[int]]:
        recorded.clear()
        train_opening_policy(
            opening_data_dir=tmp_path / f"open_{tag}",
            resume_checkpoint=base,
            output_checkpoint_dir=tmp_path / f"out_{tag}",
            epochs=1,
            batch_size=4,
            num_blocks=1,
            num_filters=8,
            seed=seed,
            allow_uncertified_data=True,
            allow_superseded_resume=True,
            allow_missing_run_manifest=True,
            repo_root=tmp_path,
        )
        return list(recorded)

    data_a = tmp_path / "open_a"
    _write_split(data_a / "train", 24, seed=14)
    _write_split(data_a / "val", 8, seed=15)
    # Two more hermetic copies of the same data: one for the same-seed
    # second run, one for the different-seed run.
    for tag in ("b", "c"):
        (tmp_path / f"open_{tag}").mkdir()
        for part in ("train", "val"):
            src = data_a / part
            dst = tmp_path / f"open_{tag}" / part
            dst.mkdir()
            for f in src.glob("*"):
                dst.joinpath(f.name).write_bytes(f.read_bytes())

    first = run("a", 42)
    second = run("b", 42)
    other = run("c", 43)

    assert first, "opening-policy run must train at least one batch"
    assert first == second, "same seed must reproduce opening-policy batch order"
    assert first != other, "different seed must change opening-policy batch order"


# ───────────── 7. MistakeLearner sampling composition (finetune path) ─────────────


_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _legal_target_idx() -> int:
    """A policy target that is legal after 1.e4 — otherwise the legal-masked
    logits saturate the CE loss identically for every composition."""
    import zeropawn_cpp

    g = zeropawn_cpp.ChessGame(_START_FEN)
    g.make_move_uci("e2e4")
    idx = zeropawn_cpp.move_to_index(g, "e7e5")
    assert 0 <= idx < 4672
    return int(idx)


def _mistake_tuple(i: int) -> tuple[np.ndarray, int, PuzzleItem]:
    planes = np.random.default_rng(100 + i).random(PLANES_SHAPE, dtype=np.float32)
    puzzle = PuzzleItem(
        puzzle_id=f"p{i}",
        fen=_START_FEN,
        opponent_move="e2e4",
        solution_move="e7e5",
        rating=1500,
    )
    return planes, _legal_target_idx(), puzzle


def _run_learner_once(seed: int, mistakes, epochs: int = 1) -> float:
    torch.manual_seed(0)  # identical init for every learner
    model = ChessNet(num_blocks=1, num_filters=8)
    learner = MistakeLearner(model=model, device="cpu", batch_size=8, seed=seed)
    for i in range(6):  # pre-fill the anchor buffer so anchor sampling draws
        planes, idx, puzzle = _mistake_tuple(50 + i)
        learner.add_anchor(planes, idx, puzzle)
    metrics = learner.train_on_mistakes(
        list(mistakes), epochs=epochs, replay_ratio=0.0, anchor_ratio=0.5
    )
    return float(metrics["total_loss"])


def test_mistake_learner_seed_reproducible_composition():
    """The CLI --seed now reaches the replay/anchor sampling: identical seeds
    select the identical training composition (same loss), different seeds
    select a different one."""
    mistakes = [_mistake_tuple(i) for i in range(4)]
    same_a = _run_learner_once(7, mistakes)
    same_b = _run_learner_once(7, mistakes)
    other = _run_learner_once(8, mistakes)
    assert same_a == same_b, "same seed must select the identical sample composition"
    assert same_a != other, "different seed must change the sampled composition"


# ───────────── 8. finetune CLI forwards --seed (wiring regression) ─────────────


def test_finetune_cli_exposes_seed_flag():
    import argparse

    from python.cli.commands_finetune import register_parser

    parser = argparse.ArgumentParser(prog="zp")
    sub = parser.add_subparsers()
    register_parser(sub)
    args = parser.parse_args(["finetune", "--dataset", "1", "--seed", "123"])
    assert args.seed == 123
    default_args = parser.parse_args(["finetune", "--dataset", "1"])
    assert default_args.seed is None, "unseeded must remain the finetune default"
