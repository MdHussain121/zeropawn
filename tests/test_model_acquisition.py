"""Unit tests for model checkpoint acquisition, verification, and missing checkpoint handling."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.cli.commands_models import run_models_download
from python.cli.common import REPO_ROOT


def test_models_download_verify_only_nonexistent(tmp_path: Path):
    """--verify-only fails with exit code 1 if target file does not exist."""
    fake_path = tmp_path / "nonexistent.pt"
    args = argparse.Namespace(
        output=str(fake_path),
        verify_only=True,
        url=None,
    )
    ret = run_models_download(args)
    assert ret == 1


def test_models_download_verify_only_corrupted(tmp_path: Path):
    """--verify-only fails if file exists but has an invalid SHA-256 hash."""
    corrupt_file = tmp_path / "corrupt.pt"
    corrupt_file.write_bytes(b"corrupted model bytes")
    args = argparse.Namespace(
        output=str(corrupt_file),
        verify_only=True,
        url=None,
    )
    ret = run_models_download(args)
    assert ret == 1


def test_models_download_no_url_and_missing_file(tmp_path: Path):
    """Running download without URL or existing weights returns 1 with helpful guidance."""
    target_file = tmp_path / "target.pt"
    args = argparse.Namespace(
        output=str(target_file),
        verify_only=False,
        url=None,
    )
    with patch.dict("os.environ", {}, clear=False):
        ret = run_models_download(args)
    assert ret == 1
    assert not target_file.exists()


def test_models_download_already_valid_skips_download(tmp_path: Path):
    """If target file already exists and matches a known valid hash, download is skipped (exit 0)."""
    # Test scaled baseline
    scaled_ckpt = REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt"
    if scaled_ckpt.is_file():
        args_scaled = argparse.Namespace(
            model="scaled",
            output=str(scaled_ckpt),
            verify_only=False,
            url=None,
        )
        assert run_models_download(args_scaled) == 0

    # Test fine-tuned model (strongest)
    finetuned_ckpt = REPO_ROOT / "models" / "checkpoints_finetune" / "best_model.pt"
    if finetuned_ckpt.is_file():
        args_finetuned = argparse.Namespace(
            model="finetuned",
            output=str(finetuned_ckpt),
            verify_only=False,
            url=None,
        )
        assert run_models_download(args_finetuned) == 0


def test_models_download_verify_both(tmp_path: Path):
    """--model both --verify-only verifies both scaled and fine-tuned models if present."""
    args = argparse.Namespace(
        model="both",
        output=None,
        verify_only=True,
        url=None,
    )
    # When both are present locally, returns 0
    scaled_ckpt = REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt"
    finetuned_ckpt = REPO_ROOT / "models" / "checkpoints_finetune" / "best_model.pt"
    if scaled_ckpt.is_file() and finetuned_ckpt.is_file():
        assert run_models_download(args) == 0


def test_models_download_network_error_cleanup(tmp_path: Path):
    """Network failure cleans up temporary download file and returns 1."""
    dest_path = tmp_path / "failed_model.pt"
    args = argparse.Namespace(
        output=str(dest_path),
        verify_only=False,
        url="https://mock.invalid.example/weights.pt",
    )
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Host not found")):
        ret = run_models_download(args)
    assert ret == 1
    assert not dest_path.exists()
    assert not dest_path.with_suffix(".tmp").exists()


def test_models_download_mocked_success(tmp_path: Path):
    """Successful download writes file atomically and verifies hash."""
    canonical_sha = "79A72EF746F4BFDA2555FE7D0EC04970BE7CCD4A39E282E16A05594ED15BC36B"
    dest_path = tmp_path / "downloaded_model.pt"
    test_content = b"MockedModelBytes"

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(len(test_content))}
    mock_resp.read.side_effect = [test_content, b""]
    mock_resp.__enter__.return_value = mock_resp

    mock_hasher = MagicMock()
    mock_hasher.hexdigest.return_value = canonical_sha

    args = argparse.Namespace(
        output=str(dest_path),
        verify_only=False,
        url="https://mock.example.com/best_model.pt",
    )

    with (
        patch("urllib.request.urlopen", return_value=mock_resp),
        patch("hashlib.sha256", return_value=mock_hasher),
    ):
        ret = run_models_download(args)

    assert ret == 0
    assert dest_path.is_file()
    assert dest_path.read_bytes() == test_content
    assert not dest_path.with_suffix(".tmp").exists()
