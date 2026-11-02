"""S3-01 regression: the weight-push preflight must fail closed against a
contaminated (wrong-architecture) live_weights payload and pass cleanly once
the active sync path is free of it.

The 2026-10-30 contamination signature: a ChessNet(2,32) test-scale payload
left at models/checkpoints_rl/live_weights.pt hard-blocked the next RL
campaign's first weight push (ArtifactPathError from the arch guard). The
remediation (2026-11-03) quarantined the contaminated artifact to
models/checkpoints_rl/quarantined/ (see that directory's QUARANTINE.json);
these tests pin the preflight contract on temporary fixtures only — the
production artifact tree is never written.
"""

from __future__ import annotations

import torch
import pytest

from python.common.artifacts import (
    ArtifactPathError,
    ProductionWriteTicket,
    assert_safe_artifact_destination,
)


def _write_scale_checkpoint(path, blocks: int, filters: int) -> None:
    """Write a minimal weights-only payload shaped like a push_weights disk sync."""
    from python.model.chess_net import ChessNet

    net = ChessNet(blocks, filters)
    torch.save(
        {
            "model": net.state_dict(),
            "model_version": 1,
            "step": 1,
            "num_blocks": blocks,
            "num_filters": filters,
        },
        path,
    )


def _preflight(destination, arch=(12, 128)):
    ticket = ProductionWriteTicket.issue_for_explicit_path(destination)
    return assert_safe_artifact_destination(
        destination, ticket=ticket, arch=arch, what="live weights"
    )


def test_preflight_fails_closed_on_contaminated_scale(tmp_path):
    """A 2x32 payload at the sync path must refuse a 12x128 push (S3-01 signature)."""
    sync = tmp_path / "checkpoints_rl" / "live_weights.pt"
    sync.parent.mkdir(parents=True, exist_ok=True)
    _write_scale_checkpoint(sync, blocks=2, filters=32)
    with pytest.raises(ArtifactPathError) as excinfo:
        _preflight(sync)
    assert "2x32" in str(excinfo.value) and "12x128" in str(excinfo.value)


def test_preflight_passes_when_sync_path_is_free(tmp_path):
    """After quarantine the absent path resolves cleanly for a fresh 12x128 write."""
    sync = tmp_path / "checkpoints_rl" / "live_weights.pt"
    resolved = _preflight(sync)
    assert resolved == sync


def test_preflight_passes_for_matching_architecture(tmp_path):
    """A matching-arch payload at the sync path still permits an overwrite."""
    sync = tmp_path / "checkpoints_rl" / "live_weights.pt"
    sync.parent.mkdir(parents=True, exist_ok=True)
    _write_scale_checkpoint(sync, blocks=12, filters=128)
    resolved = _preflight(sync)
    assert resolved == sync
