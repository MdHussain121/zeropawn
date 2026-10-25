import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from python.data.download import (
    get_database_license,
    get_month_filename,
    get_month_url,
    check_size_limit,
    get_remote_file_size,
    download_file,
    LICHESS_LICENSE_NAME,
)


def test_license_report():
    lic = get_database_license()
    assert lic["license"] == LICHESS_LICENSE_NAME
    assert "CC0" in lic["license"]
    assert "public domain" in lic["summary"].lower()


def test_urls_and_filenames():
    assert get_month_filename("2026-09") == "lichess_db_standard_rated_2026-09.pgn.zst"
    assert (
        get_month_url("2026-09")
        == "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst"
    )


def test_check_size_limit():
    assert check_size_limit(29_000_000_000, max_bytes=30_000_000_000) is True
    assert check_size_limit(31_000_000_000, max_bytes=30_000_000_000) is False


def test_download_file_over_30gb_rejection(tmp_path):
    dest = tmp_path / "test.zst"
    with patch("python.data.download.get_remote_file_size", return_value=32_000_000_000):
        with pytest.raises(ValueError, match="exceeds 30 GB limit"):
            download_file(
                url="https://fake.url/test.zst",
                destination=dest,
                allow_over_30gb=False,
            )


def test_download_file_partial(tmp_path):
    dest = tmp_path / "test_download.bin"
    mock_data = b"0123456789" * 100
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.headers.get.return_value = str(len(mock_data))
    mock_resp.read.side_effect = [mock_data[:500], mock_data[500:], b""]

    with patch("python.data.download.get_remote_file_size", return_value=len(mock_data)):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            download_file(
                url="https://fake.url/test.bin",
                destination=dest,
                max_bytes=len(mock_data),
                allow_over_30gb=False,
            )

    assert dest.exists()
    assert dest.read_bytes() == mock_data
