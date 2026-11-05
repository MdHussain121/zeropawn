"""Offline tests for the S6-02 model-download lineage remediation.

Covers:
1. The download catalog derives registry_status/designated/lineage_note from
   docs/checkpoint_registry.json at runtime (no hardcoded contradiction).
2. The designated baseline is listed as a LOCAL artifact, not downloadable.
3. Downloading a superseded-lineage artifact is refused without
   ``--allow-superseded`` (no network request is even attempted) and
   attempted with it.
4. The old orphan VALID_SHAS entries map to no registry record and are gone.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.cli.commands_models import build_model_catalog, run_models_download
from python.cli.common import REPO_ROOT

ORPHAN_SHAS = (
    "54D462823837F77C60621FB72E1A1474ECA3DC9DBB2A2969D87CED18ADA2B7DF",
    "5AC334C8DC2E36A2577473514641929EDABFDB73D9B09FF19B455C36DDC0A038",
)


def _registry_json() -> dict:
    return json.loads((REPO_ROOT / "docs" / "checkpoint_registry.json").read_text(encoding="utf-8"))


def test_catalog_entries_carry_registry_lineage_fields():
    catalog = build_model_catalog()
    for key in ("finetuned", "scaled"):
        spec = catalog[key]
        assert spec["registry_id"], f"{key}: registry_id missing"
        assert isinstance(spec["registry_status"], str)
        assert spec["registry_status"] != "verified" or spec["designated"]
        assert spec["lineage_note"]
        assert "Superseded pre-remediation lineage" in spec["lineage_note"]
        assert spec["available"] is True


def test_catalog_matches_registry_records():
    """registry_status/designated equal the live registry record for each sha."""
    registry = _registry_json()
    by_sha = {
        str(rec["sha256"]).lower(): rec
        for rec in registry.get("checkpoints", [])
        if rec.get("sha256")
    }
    catalog = build_model_catalog()
    for key in ("finetuned", "scaled"):
        spec = catalog[key]
        rec = by_sha[spec["sha256"].lower()]
        assert spec["registry_id"] == rec["id"]
        assert spec["registry_status"] == rec["status"]
        assert spec["designated"] == bool(rec.get("designated_warm_start"))


def _patch_registry_with_designated_baseline(monkeypatch):
    """Fixture: the registry state the S6-02 catalog logic was written against
    — a designated, verified local baseline (ckpt-10) — injected so the
    baseline-local-only contract is tested deterministically instead of
    depending on the live registry's post-cleanup state (ckpt-10 was removed
    from local deployment by the operator-authorized cleanup of 2026-11-03,
    commit 7099019, and is status=missing there)."""
    registry = {
        "version": 1,
        "checkpoints": [
            {
                "id": "ckpt-10-supervised_baseline_20261031",
                "sha256": "01cead16a445f5a228e78ec85da6f1e68bcf1e6de15c1f3219edda3b544b74ef",
                "status": "verified",
                "designated_warm_start": True,
                "path": "models/checkpoints_baseline_20261031/best_model.pt",
                "file": "models/checkpoints_baseline_20261031/best_model.pt",
            },
        ],
    }
    monkeypatch.setattr("python.model.checkpoint_registry.load_registry", lambda _root: registry)


def test_designated_baseline_is_local_only(monkeypatch):
    _patch_registry_with_designated_baseline(monkeypatch)
    catalog = build_model_catalog()
    assert "baseline" in catalog, "designated baseline entry missing from catalog"
    spec = catalog["baseline"]
    assert spec["available"] is False
    assert spec["local_artifact"] is True
    assert spec["designated"] is True
    assert spec["registry_status"] == "verified"
    assert "LOCAL artifact only" in spec["lineage_note"]
    assert "models/checkpoints_baseline_20261031" in spec["lineage_note"]
    assert "filename" not in spec, "baseline must not expose a release filename"


def test_baseline_download_refused(tmp_path: Path, capsys, monkeypatch):
    _patch_registry_with_designated_baseline(monkeypatch)
    args = argparse.Namespace(
        model="baseline",
        output=str(tmp_path / "baseline.pt"),
        verify_only=False,
        url=None,
        allow_superseded=True,  # even explicit confirmation cannot conjure an asset
    )
    with patch("urllib.request.urlopen") as mock_urlopen:
        ret = run_models_download(args)
    assert ret == 1
    mock_urlopen.assert_not_called()
    out = " ".join(capsys.readouterr().out.split())  # rich wraps console lines
    assert "LOCAL ONLY" in out or "Not available for download" in out


def test_superseded_download_refused_without_flag(tmp_path: Path, capsys):
    """Downloading a non-designated artifact is refused before any network call."""
    dest = tmp_path / "scaled.pt"
    args = argparse.Namespace(
        model="scaled",
        output=str(dest),
        verify_only=False,
        url=None,
        allow_superseded=False,
    )
    with patch("urllib.request.urlopen") as mock_urlopen:
        ret = run_models_download(args)
    assert ret == 1
    mock_urlopen.assert_not_called()
    assert not dest.exists()
    out = " ".join(capsys.readouterr().out.split())  # rich wraps console lines
    assert "allow-superseded" in out
    assert "Refusing to download superseded-lineage artifact" in out
    assert "designated=False" in out
    assert "status=missing" in out


def test_superseded_download_allowed_with_flag(tmp_path: Path):
    """--allow-superseded lets the (mocked) network fetch proceed."""
    dest = tmp_path / "scaled_flag.pt"
    args = argparse.Namespace(
        model="scaled",
        output=str(dest),
        verify_only=False,
        url="https://mock.invalid.example/weights.pt",
        allow_superseded=True,
    )
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Network unreachable (expected in offline test)"),
    ) as mock_urlopen:
        ret = run_models_download(args)
    assert ret == 1  # network error, but the gate was passed
    mock_urlopen.assert_called_once()
    assert not dest.exists()


def test_existing_valid_file_skips_download_without_flag(tmp_path: Path, capsys):
    """Verifying an already-present, hash-valid artifact needs no gate flag."""
    from python.cli.commands_models import build_model_catalog as _b

    sha = _b()["scaled"]["sha256"]
    # Synthesize content whose sha256 matches the catalog entry via hasher patch.
    dest = tmp_path / "existing.pt"
    dest.write_bytes(b"already-here")

    mock_hasher = MagicMock()
    mock_hasher.hexdigest.return_value = sha

    args = argparse.Namespace(
        model="scaled",
        output=str(dest),
        verify_only=False,
        url=None,
        allow_superseded=False,
    )
    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("hashlib.sha256", return_value=mock_hasher),
    ):
        ret = run_models_download(args)
    assert ret == 0
    mock_urlopen.assert_not_called()
    assert dest.read_bytes() == b"already-here"  # not overwritten
    out = " ".join(capsys.readouterr().out.split())  # rich wraps console lines
    assert "already exists and verified" in out


def test_orphan_shas_are_gone():
    """The two orphan VALID_SHAS map to no registry record and no catalog sha."""
    registry = _registry_json()
    registry_shas = {str(rec.get("sha256", "")).lower() for rec in registry.get("checkpoints", [])}
    catalog = build_model_catalog()
    catalog_shas = {spec["sha256"].lower() for spec in catalog.values() if spec.get("sha256")}
    for orphan in ORPHAN_SHAS:
        assert orphan.lower() not in registry_shas
        assert orphan.lower() not in catalog_shas


def test_verify_only_still_works_for_missing_file(tmp_path: Path):
    args = argparse.Namespace(
        model="finetuned",
        output=str(tmp_path / "missing.pt"),
        verify_only=True,
        url=None,
        allow_superseded=False,
    )
    assert run_models_download(args) == 1


@pytest.mark.parametrize("key", ["finetuned", "scaled"])
def test_release_asset_urls_unchanged(key):
    catalog = build_model_catalog()
    spec = catalog[key]
    assert spec["filename"] in (
        "finetuned_best_model.pt",
        "best_model.pt",
    )
    assert spec["available"] is True
