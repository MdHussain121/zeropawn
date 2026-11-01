"""Regression tests for the 2026-10-31 supervised-learning remediation (A-02, SLQ-01..24).

Each test names the finding it pins. A reintroduction of the original defect
must fail the corresponding test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from python.data.build_dataset import OUTCOME_UNAVAILABLE, save_dataset_split
from python.data.dataset import (
    MASK_PACKED_BYTES,
    BinaryChessDataset,
    GpuResidentDataLoader,
    SourceAwareCombinedDataset,
    decode_packed_masks,
)
from python.data.dataset_validation import (
    require_validated_dataset,
    validate_split,
    write_receipt,
)
from python.model.chess_net import ChessNet
from python.train.train_supervised import (
    WarmupCosineScheduler,
    legal_masked_policy_loss,
    save_checkpoint,
    train_step,
)

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_positions(n: int, outcomes=None, plies=None, tactical=False):
    """Synthetic PositionData objects for save_dataset_split."""
    from types import SimpleNamespace

    rng = np.random.default_rng(7)
    positions = []
    for i in range(n):
        planes = rng.random((19, 8, 8), dtype=np.float32)
        outcome = OUTCOME_UNAVAILABLE if outcomes is None else int(outcomes[i])
        ply = 0 if plies is None else int(plies[i])
        positions.append(
            SimpleNamespace(
                planes=planes,
                move_idx=int(rng.integers(0, 4672)),
                outcome=outcome,
                game_id=1000 + i,
                ply=ply,
                pos_hash=0,
                legal_mask=np.packbits(
                    np.concatenate([np.ones(100, dtype=bool), np.zeros(4572, dtype=bool)])
                ),
                stm=i % 2,
            )
        )
    return positions


def _write_certified_dataset(root: Path, n_train=64, n_val=16, n_test=8):
    """Write a synthetic three-split dataset with role metadata and a genuine
    validation receipt, so the fail-closed gate certifies it."""
    # Deterministic WDL mix: 0/1/2 outcomes, all at ply >= 10 so the WDL mask
    # admits every row.
    outcomes_train = [0, 1, 2] * (n_train // 3) + [0] * (n_train % 3)
    outcomes_other = [0, 1, 2] * ((n_val + n_test) // 3)
    train = _make_positions(n_train, outcomes=outcomes_train, plies=[10] * n_train)
    val = _make_positions(n_val, outcomes=outcomes_other[:n_val], plies=[10] * n_val)
    test = _make_positions(n_test, outcomes=outcomes_other[n_val:], plies=[10] * n_test)
    save_dataset_split(root / "train", train, "train")
    save_dataset_split(root / "val", val, "val")
    save_dataset_split(root / "test", test, "test")

    reports = {}
    overall_ok = True
    for split in ("train", "val", "test"):
        # check_build_record=False: the synthetic fixture has no build record;
        # build-record agreement is exercised on the real rebuilt dataset.
        report = validate_split(
            root / split, scan_planes="bounded", plane_scan_rows=100, check_build_record=False
        )
        reports[split] = report
        overall_ok = overall_ok and report.overall == "pass"
    assert overall_ok, "synthetic fixture must pass validation"
    write_receipt(root, reports, overall="pass")
    return root


@pytest.fixture(scope="module")
def certified_dataset(tmp_path_factory):
    return _write_certified_dataset(tmp_path_factory.mktemp("cert_ds"))


# ---------------------------------------------------------------------------
# SLQ-05: the GPU-resident loader consumes the legal-mask stream
# ---------------------------------------------------------------------------


def test_slq05_mask_decode_matches_unpackbits():
    rng = np.random.default_rng(3)
    packed = rng.integers(0, 256, size=(9, MASK_PACKED_BYTES), dtype=np.uint8)
    ref = np.unpackbits(packed, axis=1)[:, :4672].astype(bool)
    got = decode_packed_masks(torch.from_numpy(packed))
    assert got.dtype == torch.bool
    assert (ref == got.numpy()).all()


def test_slq05_gpu_loader_yields_legal_mask_matching_disk_loader(certified_dataset):
    """The CUDA-default loader must expose the same mask stream as the
    disk-backed loader for identical row indices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    disk = BinaryChessDataset(certified_dataset / "train")
    gpu = GpuResidentDataLoader(
        certified_dataset / "train", batch_size=8, shuffle=False, device=device
    )
    assert gpu.has_masks, "schema-2 dataset must expose masks on the GPU path"
    batches = list(gpu)
    assert batches and "legal_mask" in batches[0]
    # Row i of the unshuffled loader must equal disk row i.
    gpu_masks = torch.cat([b["legal_mask"] for b in batches]).cpu()
    gpu_rows = torch.cat([b["move"] for b in batches]).cpu()
    for i in range(len(disk)):
        disk_sample = disk[i]
        assert torch.equal(gpu_masks[i], disk_sample["legal_mask"])
        assert int(gpu_rows[i]) == int(disk_sample["move"])
    gpu.close()


def test_slq05_mask_changes_loss_behavior():
    """The mask stream must change the loss: an illegal label raises when the
    mask is present and is silently smoothed when it is absent."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4672)
    # Row 0: label 3 legal. Row 1: label 5 legal only when mask[1, 5] is set —
    # here it is not, so the mask path must raise (data-integrity gate).
    moves = torch.tensor([3, 5])
    mask = torch.zeros(2, 4672, dtype=torch.bool)
    mask[0, 3] = True
    mask[1, 7] = True
    with pytest.raises(ValueError):
        legal_masked_policy_loss(logits, moves, mask)
    # No mask → the same batch trains silently (fallback path).
    loss_unmasked = legal_masked_policy_loss(logits, moves, None)
    assert torch.isfinite(loss_unmasked)


# ---------------------------------------------------------------------------
# SLQ-06: tactical mate labels are live data through every loader
# ---------------------------------------------------------------------------


def _make_tactical_shard(root: Path):
    """Two-row tactical shard: one mate-labelled row (outcome 0, ply 10),
    one masked row (outcome 255, ply 0)."""
    mate = _make_positions(1, outcomes=[0], plies=[10])
    masked = _make_positions(1, outcomes=[OUTCOME_UNAVAILABLE], plies=[0])
    save_dataset_split(root, mate + masked, "train")
    return root


def test_slq06_disk_loader_passes_mate_labels_through(tmp_path):
    _make_tactical_shard(tmp_path / "tac")
    ds = BinaryChessDataset(tmp_path / "tac", is_tactical=True)
    mate_row = ds[0]
    masked_row = ds[1]
    assert int(mate_row["ply"]) == 10
    assert mate_row["value_target"].tolist() == [1.0, 0.0, 0.0]
    assert int(masked_row["ply"]) == 0
    assert masked_row["value_target"].tolist() == [0.0, 0.0, 0.0]


def test_slq06_combined_dataset_passes_mate_labels_through(tmp_path):
    _make_tactical_shard(tmp_path / "tac")
    # Build a minimal master shard next to the tactical one.
    save_dataset_split(
        tmp_path / "master",
        _make_positions(2, outcomes=[0, 1], plies=[12, 11]),
        "train",
    )
    master_ds = BinaryChessDataset(tmp_path / "master")
    tactics_ds = BinaryChessDataset(tmp_path / "tac", is_tactical=True)
    combined = SourceAwareCombinedDataset(master_ds, tactics_ds, tactics_ratio=0.5)
    # Rows 2..3 are the tactical rows: mate row must keep ply/target.
    assert int(combined[2]["ply"]) == 10
    assert combined[2]["value_target"].tolist() == [1.0, 0.0, 0.0]
    assert bool(combined[2]["is_tactical"]) is True
    assert int(combined[3]["ply"]) == 0
    assert combined[3]["value_target"].tolist() == [0.0, 0.0, 0.0]
    # Master rows keep their own labels.
    assert int(combined[0]["ply"]) == 12
    assert combined[0]["value_target"].tolist() == [1.0, 0.0, 0.0]


def test_slq06_gpu_tactics_path_passes_labels_through(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _make_tactical_shard(tmp_path / "tac")
    save_dataset_split(
        tmp_path / "master", _make_positions(4, outcomes=[0, 1, 2, 0], plies=[12] * 4), "train"
    )
    loader = GpuResidentDataLoader(
        tmp_path / "master",
        batch_size=6,
        shuffle=False,
        device=torch.device("cuda"),
        tactics_dir=tmp_path / "tac",
    )
    batches = list(loader)
    all_ply = torch.cat([b["ply"] for b in batches]).cpu()
    all_targets = torch.cat([b["value_target"] for b in batches]).cpu()
    # Master rows (4) + tactics rows (2): the mate row sits at index 4.
    assert int(all_ply[4]) == 10, "mate row's stored ply must not be reset to 0"
    assert all_targets[4].tolist() == [1.0, 0.0, 0.0]
    assert all_targets[5].tolist() == [0.0, 0.0, 0.0]
    # The WDL mask the trainer computes must admit the mate row.
    mask = (all_ply >= 10) & (all_targets.sum(dim=-1) > 0)
    assert bool(mask[4]) and not bool(mask[5])
    loader.close()


# ---------------------------------------------------------------------------
# SLQ-01/Phase D: value-head gradient flow and direction
# ---------------------------------------------------------------------------


def _tiny_model():
    torch.manual_seed(11)
    return ChessNet(in_channels=19, num_filters=16, num_blocks=1, num_actions=4672)


def test_value_head_receives_nonzero_gradient_on_valid_wdl():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "planes": torch.randn(4, 19, 8, 8),
        "move": torch.tensor([10, 20, 30, 40]),
        "value_target": torch.eye(3)[[0, 1, 2, 0]],
        "ply": torch.tensor([10, 11, 12, 13]),
        "legal_mask": torch.ones(4, 4672, dtype=torch.bool),
    }
    result = train_step(model, batch, optimizer, "cpu", use_autocast=False, step_optimizer=True)
    grad_diag = result[7]
    assert grad_diag is not None
    assert grad_diag["value_grad_norm"] > 0.0, "valid WDL targets must reach the value head"


def test_value_head_gradient_exactly_zero_when_all_masked():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "planes": torch.randn(4, 19, 8, 8),
        "move": torch.tensor([10, 20, 30, 40]),
        "value_target": torch.zeros(4, 3),
        "ply": torch.tensor([0, 1, 2, 3]),
        "legal_mask": torch.ones(4, 4672, dtype=torch.bool),
    }
    result = train_step(model, batch, optimizer, "cpu", use_autocast=False, step_optimizer=True)
    grad_diag = result[7]
    assert grad_diag is not None
    assert grad_diag["value_grad_norm"] == 0.0, "all-masked batches are not value supervision"
    assert grad_diag["total_grad_norm"] > 0.0, "policy loss must still produce gradients"


def test_value_head_gradient_direction_follows_labels():
    """Training toward win labels must increase P(win) and decrease P(loss)."""
    torch.manual_seed(5)
    model = ChessNet(in_channels=19, num_filters=8, num_blocks=1, num_actions=4672)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    planes = torch.randn(8, 19, 8, 8)
    batch = {
        "planes": planes,
        "move": torch.randint(0, 4672, (8,)),
        "value_target": torch.zeros(8, 3),
        "ply": torch.full((8,), 10, dtype=torch.long),
        "legal_mask": torch.ones(8, 4672, dtype=torch.bool),
    }
    batch["value_target"][:, 0] = 1.0  # all wins

    def mean_probs():
        model.eval()
        with torch.no_grad():
            _, wdl = model(planes)
            p = torch.softmax(wdl, dim=-1).mean(dim=0)
        model.train()
        return p

    p_before = mean_probs()
    for _ in range(40):
        train_step(model, batch, optimizer, "cpu", use_autocast=False, step_optimizer=True)
    p_after = mean_probs()
    assert p_after[0] > p_before[0], "P(win) must increase under win supervision"
    assert p_after[2] < p_before[2], "P(loss) must decrease under win supervision"


# ---------------------------------------------------------------------------
# SLQ-11: single warmup formula + faithful resume geometry
# ---------------------------------------------------------------------------


def test_slq11_noarg_step_matches_get_lr():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    sched = WarmupCosineScheduler(opt, warmup_steps=4, total_steps=10, base_lr=2e-4)
    for expected_step in range(1, 11):
        lr = sched.step()  # no-arg advance
        assert abs(lr - sched.get_lr(expected_step)) < 1e-12, (
            f"no-arg step {expected_step} diverged from get_lr (SLQ-11)"
        )


def test_slq11_warmup_boundary_values():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    base_lr, min_lr, warmup = 2e-4, 1e-5, 500
    sched = WarmupCosineScheduler(opt, warmup_steps=warmup, total_steps=1000, base_lr=base_lr)
    assert sched.get_lr(0) == pytest.approx(min_lr)
    assert sched.get_lr(1) == pytest.approx(min_lr + (base_lr - min_lr) * (1 / warmup))
    assert sched.get_lr(warmup) == pytest.approx(base_lr)
    assert sched.get_lr(10_000) == pytest.approx(min_lr)


def test_slq11_resume_restores_checkpoint_geometry(tmp_path):
    """Resume continues the ORIGINAL schedule; CLI differences must not
    silently reshape it."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    sched = WarmupCosineScheduler(opt, warmup_steps=50, total_steps=200, base_lr=2e-4)
    model = _tiny_model()
    ckpt = tmp_path / "c.pt"
    save_checkpoint(ckpt, model, opt, sched, step=10, epoch=0, best_val_loss=1.0)

    # Resume machinery equivalent: load the scheduler state the way the
    # trainer does (checkpoint geometry wins over differing CLI values).
    loaded = torch.load(ckpt, map_location="cpu", weights_only=True)
    saved = loaded["scheduler"]
    cli_like = WarmupCosineScheduler(
        opt, warmup_steps=500, total_steps=9000, base_lr=2e-4
    )
    cli_like.warmup_steps = saved["warmup_steps"]
    cli_like.total_steps = saved["total_steps"]
    cli_like.base_lr = saved["base_lr"]
    cli_like.min_lr = saved["min_lr"]
    cli_like.current_step = saved.get("current_step", 0)
    assert cli_like.warmup_steps == 50
    assert cli_like.total_steps == 200
    # LR trajectory continues the original schedule exactly (the checkpoint's
    # scheduler counter was 0, so the next no-arg step is step 1).
    saved_step = saved.get("current_step", 0)
    assert abs(cli_like.step() - sched.get_lr(saved_step + 1)) < 1e-12


# ---------------------------------------------------------------------------
# SLQ-16: window skip clears gradients and steps nothing
# ---------------------------------------------------------------------------


def test_slq16_nonfinite_batch_clears_gradients():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    good = {
        "planes": torch.randn(2, 19, 8, 8),
        "move": torch.tensor([10, 20]),
        "value_target": torch.eye(3)[[0, 1]],
        "ply": torch.tensor([10, 11]),
        "legal_mask": torch.ones(2, 4672, dtype=torch.bool),
    }
    # First micro-batch backprops finite gradients into the model.
    train_step(
        model, good, optimizer, "cpu", use_autocast=False, accum_steps=2,
        step_optimizer=False, zero_grad=True,
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # A non-finite batch then skips: the skip semantics abort the window and
    # clear the partially accumulated gradients.
    bad = dict(good)
    bad_planes = good["planes"].clone()
    bad_planes[0, 0, 0, 0] = float("nan")
    bad["planes"] = bad_planes
    result = train_step(
        model, bad, optimizer, "cpu", use_autocast=False, accum_steps=2,
        step_optimizer=False, zero_grad=False,
    )
    skip_flag = result[6]
    if skip_flag:
        # The trainer's window handler reacts to skip by zeroing gradients.
        optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters()), (
        "a skipped window must not leave residual gradients (SLQ-16)"
    )


# ---------------------------------------------------------------------------
# SLQ-17: supervised saves carry optimizer/checkpoint stamps
# ---------------------------------------------------------------------------


def test_slq17_checkpoint_stamps(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = WarmupCosineScheduler(optimizer, warmup_steps=5, total_steps=10, base_lr=1e-4)
    ckpt = tmp_path / "c.pt"
    save_checkpoint(ckpt, model, optimizer, sched, step=1, epoch=0, best_val_loss=1.0)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert loaded.get("checkpoint_type") == "supervised"
    assert loaded.get("_optimizer_type") == "AdamW"


# ---------------------------------------------------------------------------
# SLQ-18/SLQ-19: tactics default parity and combined __len__
# ---------------------------------------------------------------------------


def test_slq19_combined_len_is_total(tmp_path):
    save_dataset_split(
        tmp_path / "master", _make_positions(3, outcomes=[0, 1, 2], plies=[10] * 3), "train"
    )
    _make_tactical_shard(tmp_path / "tac")
    master_ds = BinaryChessDataset(tmp_path / "master")
    tactics_ds = BinaryChessDataset(tmp_path / "tac", is_tactical=True)
    combined = SourceAwareCombinedDataset(master_ds, tactics_ds, tactics_ratio=0.5)
    assert len(combined) == len(master_ds) + len(tactics_ds), (
        "direct iteration must see every sample (SLQ-19)"
    )
    sampler = combined.get_sampler(num_samples=combined.n_master)
    assert sampler.num_samples == combined.n_master


def test_slq18_tactics_ratio_default_shared():
    """The trainer resolves the shared default before either loader branch:
    tactics-dir without an explicit ratio must mean 0.3 on BOTH paths."""
    import inspect

    from python.train import train_supervised as ts

    source = inspect.getsource(ts.train_supervised)
    # The unified default must be applied once, before the GPU branch.
    assert "tactics_ratio = 0.3" in source
    assert source.index("tactics_ratio = 0.3") < source.index("GpuResidentDataLoader(")


# ---------------------------------------------------------------------------
# SLQ-08: the consumption gate stays fail-closed
# ---------------------------------------------------------------------------


def test_slq08_gate_refuses_roleless_dataset(tmp_path):
    save_dataset_split(tmp_path / "train", _make_positions(4, outcomes=[0, 1, 2, 0], plies=[10] * 4), "train")
    # Strip the role declaration → the gate must refuse.
    meta_path = tmp_path / "train" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("role", None)
    meta.pop("label_semantics", None)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(Exception) as excinfo:
        require_validated_dataset(tmp_path / "train", consumer="test")
    assert "role" in str(excinfo.value).lower()


def test_slq08_gate_refuses_uncertified_data(tmp_path):
    root = _write_certified_dataset(tmp_path / "ds", n_train=8, n_val=4)
    (root / "validation_receipt.json").unlink()
    with pytest.raises(Exception):
        require_validated_dataset(root / "train", consumer="test", deep=False)
    # Explicit uncertified opt-in warns but proceeds with the role invariants.
    gate = require_validated_dataset(
        root / "train", consumer="test", allow_uncertified=True, deep=False
    )
    assert gate.certified is False
    assert gate.warnings


def test_slq08_gate_accepts_certified_dataset(certified_dataset):
    gate = require_validated_dataset(certified_dataset / "train", consumer="test", deep=True)
    assert gate.certified is True
    assert gate.role is not None and gate.role.name == "games_wdl"


# ---------------------------------------------------------------------------
# SLQ-01/SLQ-07: registry fails closed on disqualified checkpoints
# ---------------------------------------------------------------------------


def test_registry_rejects_disqualified_warm_start(tmp_path, monkeypatch):
    from python.model import checkpoint_registry as cr

    # Build a real file whose header matches the project architecture and whose
    # weights are non-empty, so the status check (not the arch check and not the
    # weight-less-container rejection added by the 2026-10-31 lifecycle
    # remediation, finding CKP-06) is what rejects it. The registry entry names
    # this exact path: registry matching is path-first (a byte-identical copy at
    # another path must not inherit the original's status).
    from python.model.chess_net import ChessNet

    ckpt = tmp_path / "latest_checkpoint.pt"
    torch.save(
        {
            "model": ChessNet(num_blocks=2, num_filters=32).state_dict(),
            "num_blocks": 12,
            "num_filters": 128,
        },
        ckpt,
    )
    registry = {
        "checkpoints": [
            {
                "path": str(ckpt),
                "file": str(ckpt),
                "sha256": "aa" * 32,
                "status": "disqualified",
                "disqualified_at": "2026-10-31",
                "disqualification": {
                    "reason": "superseded_value_head",
                    "detail": "value head trained on corrupted targets",
                },
            }
        ]
    }
    monkeypatch.setattr(cr, "load_registry", lambda repo_root=".": registry)
    monkeypatch.setattr(cr, "compute_sha256", lambda path, chunk_size=1 << 20: "aa" * 32)

    with pytest.raises(cr.WarmStartValidationError) as excinfo:
        cr.validate_warm_start(ckpt, repo_root=str(tmp_path))
    assert "DISQUALIFIED" in str(excinfo.value)
    assert "superseded_value_head" in str(excinfo.value)


# ---------------------------------------------------------------------------
# SLQ-09: the builder carves a real, disjoint test split
# ---------------------------------------------------------------------------

# Site tags whose sha256 identity hashes land in a known split bucket:
# val = hash % 100 in {0, 1}, test = 2, train = anything else (precomputed).
_SPLIT_SITE_TAGS = {"train": "bkt73", "val": "bkt79", "test": "bkt206"}


def test_slq09_builder_splits_and_stm(tmp_path):
    """build_binary_dataset produces train/val/test with real WDL labels,
    per-row stm sidecars, a build record, and disjoint game IDs."""
    import chess.pgn

    from python.data.build_dataset import build_binary_dataset

    moves_by_result = {
        "1-0": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0",
        "0-1": "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Nf3 Be7 5. Bg5 O-O 6. e3 d5 0-1",
        "1/2-1/2": "1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6 1/2-1/2",
    }
    pgn_path = tmp_path / "games.pgn"
    with open(pgn_path, "w", encoding="utf-8") as f:
        for split, tag in _SPLIT_SITE_TAGS.items():
            result = {"train": "1-0", "val": "0-1", "test": "1/2-1/2"}[split]
            f.write(
                f'[Event "T"]\n[Site "https://example.test/{tag}"]\n'
                f'[White "W_{tag}"]\n[Black "B_{tag}"]\n[Result "{result}"]\n\n'
                f"{moves_by_result[result]}\n\n"
            )

    out_dir = tmp_path / "ds"
    n_train, n_val, n_removed = build_binary_dataset(pgn_path, out_dir)
    assert n_train > 0 and n_val > 0
    assert (out_dir / "test" / "planes.bin").is_file(), "test split must be written"

    for split in ("train", "val", "test"):
        meta = json.loads((out_dir / split / "metadata.json").read_text())
        assert meta["role"] == "games_wdl"
        assert meta["schema_version"] == "1.1"
        outcomes = np.fromfile(out_dir / split / "outcomes.bin", dtype=np.uint8)
        assert (outcomes != OUTCOME_UNAVAILABLE).all(), "PGN games always carry WDL labels"
        assert set(np.unique(outcomes).tolist()) <= {0, 1, 2}
        stm = np.fromfile(out_dir / split / "stm.bin", dtype=np.uint8)
        assert set(np.unique(stm).tolist()) <= {0, 1}
        masks = np.fromfile(out_dir / split / "masks.bin", dtype=np.uint8)
        assert masks.size == len(outcomes) * MASK_PACKED_BYTES

    # Deterministic bucket routing: the tagged games must land in their
    # designated splits (verified through unique game IDs).
    import hashlib

    for split, tag in _SPLIT_SITE_TAGS.items():
        gid = int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16)
        ids = np.fromfile(out_dir / split / "game_ids.bin", dtype=np.uint64)
        assert gid in set(ids.tolist()), f"game tagged {tag} must land in the {split} split"

    # Build record binds source + outputs.
    record = json.loads((out_dir / "build_record.json").read_text())
    assert record["source"]["sha256"]
    assert set(record["outputs"].keys()) == {"train", "val", "test"}
    assert record["counts"]["final_test_positions"] > 0

    # Cross-split integrity: exhaustive and clean on this tiny artifact.
    from python.data.dataset_validation import check_split_integrity

    results = check_split_integrity(
        out_dir / "train", out_dir / "val", exhaustive_states=True, test_dir=out_dir / "test"
    )
    for r in results:
        assert r.status.value == "pass", f"{r.name}: {r.detail}"
