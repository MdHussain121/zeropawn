"""Hermetic artifact paths, production-write protection, and cross-process locks.

Regression coverage for audit findings CKP-04 (a test-scale 2x32 model overwrote
the production ``live_weights.pt`` through CWD-relative defaults on 2026-10-30)
and SUS-02 (no protection against two writers on one artifact path).

Every test here uses temporary directories; none of them writes to the
production artifact tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from python.common.artifacts import (
    ArtifactLockError,
    ArtifactPathError,
    ProtectedArtifactWriteError,
    ProductionWriteTicket,
    artifact_lock,
    assert_safe_artifact_destination,
    describe_lock_owner,
    is_production_artifact,
    is_protected_artifact_name,
    production_artifact_root,
    read_checkpoint_arch,
    repo_root,
    resolve_artifact_path,
)
from python.model.chess_net import ChessNet


def _save_small_checkpoint(path: Path, num_blocks: int = 2, num_filters: int = 32) -> Path:
    model = ChessNet(num_blocks=num_blocks, num_filters=num_filters)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "num_blocks": num_blocks,
            "num_filters": num_filters,
            "step": 1,
        },
        path,
    )
    return path


# ── Path resolution is CWD-independent ───────────────────────────────────────


def test_relative_paths_resolve_against_repo_root_not_cwd(tmp_path, monkeypatch):
    """A relative artifact path means the same file from any working directory."""
    monkeypatch.chdir(tmp_path)  # an arbitrary, unrelated CWD
    resolved = resolve_artifact_path("models/checkpoints_rl/x.pt")
    assert resolved == repo_root() / "models" / "checkpoints_rl" / "x.pt"
    assert resolved.is_absolute()


def test_absolute_paths_are_preserved(tmp_path):
    target = tmp_path / "ckpt.pt"
    assert resolve_artifact_path(target) == Path(os.path.normpath(str(target)))
    assert resolve_artifact_path(str(target)) == Path(os.path.normpath(str(target)))


def test_is_production_artifact_is_scoped_to_the_models_tree(tmp_path):
    assert is_production_artifact(production_artifact_root() / "checkpoints_rl" / "live_weights.pt")
    assert not is_production_artifact(tmp_path / "live_weights.pt")
    assert not is_production_artifact(repo_root() / "runs" / "phase4_rl" / "live_weights.pt")


def test_protected_names_are_recognized():
    for name in (
        "live_weights.pt",
        "latest_checkpoint.pt",
        "best_model.pt",
        "best_eval.pt",
        "phase4_final_checkpoint.pt",
        "checkpoint_registry.json",
    ):
        assert is_protected_artifact_name(Path("x") / name)


# ── Production write protection ──────────────────────────────────────────────


def _fake_production_root(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the production artifact root into a tmp tree (no real writes)."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("python.common.artifacts.repo_root", lambda: fake_repo)
    return fake_repo / "models"


def test_protected_production_write_requires_a_ticket(tmp_path, monkeypatch):
    prod = _fake_production_root(tmp_path, monkeypatch)
    target = prod / "checkpoints_rl" / "live_weights.pt"
    with pytest.raises(ProtectedArtifactWriteError) as excinfo:
        assert_safe_artifact_destination(target)
    message = str(excinfo.value)
    assert "protected production" in message
    assert "ProductionWriteTicket" in message
    assert not target.exists(), "the guard must refuse before creating anything"


def test_protected_production_write_allowed_with_explicit_ticket(tmp_path, monkeypatch):
    prod = _fake_production_root(tmp_path, monkeypatch)
    target = prod / "checkpoints_rl" / "live_weights.pt"
    ticket = ProductionWriteTicket.issue_for_launcher("test-launcher", reason="unit test")
    assert assert_safe_artifact_destination(target, ticket=ticket) == target


def test_ticket_minted_by_explicit_path_records_the_destination():
    ticket = ProductionWriteTicket.issue_for_explicit_path("models/checkpoints_rl/live_weights.pt")
    assert "explicit-path" in ticket.issued_by
    assert "live_weights.pt" in ticket.reason


def test_unprotected_name_in_production_tree_needs_no_ticket(tmp_path, monkeypatch):
    prod = _fake_production_root(tmp_path, monkeypatch)
    target = prod / "checkpoints_rl" / "phase4_ckpt_v5_step997.pt"
    assert assert_safe_artifact_destination(target) == target


def test_protected_name_outside_production_tree_needs_no_ticket(tmp_path):
    target = tmp_path / "live_weights.pt"
    assert assert_safe_artifact_destination(target) == target


# ── The 2026-10-30 contamination signature ───────────────────────────────────


def test_architecture_change_under_an_existing_name_is_refused(tmp_path, monkeypatch):
    """The exact 2026-10-30 signature: a 2x32 model replacing a 12x128 artifact."""
    prod = _fake_production_root(tmp_path, monkeypatch)
    live_weights = prod / "checkpoints_rl" / "live_weights.pt"
    # The production-shaped artifact (header only; weights are irrelevant here).
    _save_small_checkpoint(live_weights, num_blocks=12, num_filters=128)

    ticket = ProductionWriteTicket.issue_for_launcher("test", reason="simulated production run")
    with pytest.raises(ArtifactPathError, match=r"2x32.*12x128|12x128"):
        assert_safe_artifact_destination(
            live_weights, ticket=ticket, arch=(2, 32), what="live weights"
        )
    # The existing artifact is untouched.
    assert read_checkpoint_arch(live_weights) == (12, 128)


def test_architecture_change_allowed_for_a_deliberate_migration(tmp_path):
    target = tmp_path / "migrating.pt"
    _save_small_checkpoint(target, num_blocks=2, num_filters=32)
    resolved = assert_safe_artifact_destination(
        target, arch=(12, 128), allow_arch_change=True
    )
    assert resolved == target


def test_read_checkpoint_arch_tolerates_unknown_and_corrupt_files(tmp_path):
    assert read_checkpoint_arch(tmp_path / "absent.pt") is None
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a torch archive")
    assert read_checkpoint_arch(garbage) is None
    no_header = tmp_path / "no_header.pt"
    torch.save({"model": {}}, no_header)
    assert read_checkpoint_arch(no_header) is None


def test_missing_output_directory_does_not_redirect_writes(tmp_path, monkeypatch):
    """A missing directory is created where it was named — never anywhere else."""
    monkeypatch.chdir(tmp_path)
    target = resolve_artifact_path("models/checkpoints_rl/custom_dir/ckpt.pt")
    assert target == repo_root() / "models" / "checkpoints_rl" / "custom_dir" / "ckpt.pt"
    assert not target.exists()


def test_no_path_can_escape_via_parent_segments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # "models/../models/checkpoints_rl" normalizes to the models tree; it does
    # not silently become an arbitrary directory outside it.
    resolved = resolve_artifact_path("models/../models/checkpoints_rl/x.pt")
    assert resolved == repo_root() / "models" / "checkpoints_rl" / "x.pt"


# ── Trainer-level behavior ───────────────────────────────────────────────────


def test_rl_trainer_without_checkpoint_dir_uses_a_session_temp_dir(tmp_path):
    """No CWD-relative default: an unspecified directory is a session temp dir."""
    from python.selfplay.replay_buffer import ReplayBuffer
    from python.train.train_rl import RLTrainer

    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=True)
    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
    )
    assert trainer.checkpoint_dir.is_absolute()
    assert not is_production_artifact(trainer.checkpoint_dir)
    assert trainer.checkpoint_dir != repo_root() / "models" / "checkpoints_rl"
    # The sync path follows the session directory, not the production tree.
    assert trainer.sync_weights_path.parent == trainer.checkpoint_dir


def test_rl_trainer_write_into_production_tree_is_refused_without_ticket(tmp_path, monkeypatch):
    """A test-scale trainer pointed at production paths cannot write them."""
    _fake_production_root(tmp_path, monkeypatch)
    from python.selfplay.replay_buffer import ReplayBuffer
    from python.train.train_rl import RLTrainer

    prod_dir = production_artifact_root() / "checkpoints_rl"
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf2", capacity=64, create_new=True)
    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
        checkpoint_dir=prod_dir,
        sync_weights_path=prod_dir / "live_weights.pt",
    )
    with pytest.raises(ProtectedArtifactWriteError):
        trainer.save_checkpoint(prod_dir / "live_weights.pt")
    with pytest.raises(ProtectedArtifactWriteError):
        trainer.push_weights()
    assert not (prod_dir / "live_weights.pt").exists()


def test_rl_trainer_authorized_write_still_refuses_arch_downgrade(tmp_path, monkeypatch):
    """Even with a ticket, a 2x32 model must not replace a 12x128 artifact."""
    prod = _fake_production_root(tmp_path, monkeypatch)
    prod_dir = prod / "checkpoints_rl"
    live = prod_dir / "live_weights.pt"
    _save_small_checkpoint(live, num_blocks=12, num_filters=128)

    from python.selfplay.replay_buffer import ReplayBuffer
    from python.train.train_rl import RLTrainer

    ticket = ProductionWriteTicket.issue_for_launcher("test", reason="authorized write")
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf3", capacity=64, create_new=True)
    trainer = RLTrainer(
        model=ChessNet(num_blocks=2, num_filters=32),
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
        checkpoint_dir=prod_dir,
        sync_weights_path=live,
        artifact_ticket=ticket,
    )
    with pytest.raises(ArtifactPathError, match="architecture"):
        trainer.push_weights()
    assert read_checkpoint_arch(live) == (12, 128)


def test_supervised_save_checkpoint_refuses_protected_destination_without_ticket(tmp_path, monkeypatch):
    prod = _fake_production_root(tmp_path, monkeypatch)
    from python.train.train_supervised import WarmupCosineScheduler, save_checkpoint

    model = ChessNet(num_blocks=2, num_filters=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1, total_steps=10, base_lr=1e-4)
    with pytest.raises(ProtectedArtifactWriteError):
        save_checkpoint(
            prod / "checkpoints_scaled" / "best_model.pt",
            model,
            optimizer,
            scheduler,
            step=1,
            epoch=0,
            best_val_loss=1.0,
            num_blocks=2,
            num_filters=32,
        )


# ── Cross-process locking (SUS-02) ───────────────────────────────────────────


def test_second_writer_on_a_held_artifact_is_refused_with_owner_info(tmp_path):
    target = tmp_path / "artifact.pt"
    with artifact_lock(target, timeout=0.0, owner="first-writer"):
        with pytest.raises(ArtifactLockError) as excinfo:
            with artifact_lock(target, timeout=0.0, owner="second-writer"):
                pass
        message = str(excinfo.value)
        assert "held by another process" in message
        assert "first-writer" in describe_lock_owner(target)
    # After release the lock is available again.
    with artifact_lock(target, timeout=0.0, owner="third-writer"):
        pass


def test_lock_contention_timeout_then_success(tmp_path):
    import threading
    import time as _time

    target = tmp_path / "artifact2.pt"
    released = threading.Event()

    def holder() -> None:
        with artifact_lock(target, timeout=0.0, owner="holder"):
            released.wait(timeout=5.0)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        _time.sleep(0.2)
        # A short positive timeout waits for the holder instead of failing.
        released.set()
        with artifact_lock(target, timeout=5.0, owner="waiter"):
            pass
    finally:
        released.set()
        thread.join(timeout=5.0)


def test_lock_is_released_after_an_exception(tmp_path):
    target = tmp_path / "artifact3.pt"
    with pytest.raises(RuntimeError):
        with artifact_lock(target, timeout=0.0, owner="failing-writer"):
            raise RuntimeError("boom")
    with artifact_lock(target, timeout=0.0, owner="next-writer"):
        pass


def test_two_trainers_share_a_directory_but_not_a_path(tmp_path):
    """Separate files in one directory do not block each other (per-path locks)."""
    with artifact_lock(tmp_path / "a.pt", timeout=0.0, owner="a"):
        with artifact_lock(tmp_path / "b.pt", timeout=0.0, owner="b"):
            pass


def test_cleanup_sweeps_only_released_lock_sentinels(tmp_path):
    from python.common.cleanup import sweep_stale_lock_files

    artifact = tmp_path / "rl_ckpt_step_1.pt"
    artifact.write_bytes(b"x")
    # A lock sentinel with a fresh mtime is never removed.
    lock_file = artifact.with_name(artifact.name + ".lock")
    lock_file.write_bytes(b"")
    with artifact_lock(artifact, timeout=0.0, owner="live"):
        assert sweep_stale_lock_files(artifact) == 0
    # Even released, a *fresh* sentinel is left alone (it may belong to a writer
    # that is just about to acquire it).
    assert sweep_stale_lock_files(artifact) == 0
    # Age it past the threshold: now it is provably stale and swept.
    old = lock_file.stat().st_mtime - 3600
    os.utime(lock_file, (old, old))
    assert sweep_stale_lock_files(artifact) == 1
    assert not lock_file.exists()
