"""Data download script for the Lichess open database (database.lichess.org).

Checks and reports database licensing (CC0 1.0 Universal) and enforces
a strict download size threshold (max 30 GB without explicit approval).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from python.cli.common import setup_logging
from python.common.progress import (
    artifacts_list,
    banner,
    emit_error,
    emit_report,
    error_block,
    file_record,
    header_block,
    is_tty,
    parameters_table,
    results_table,
    run_context,
    summary_block,
)

LICHESS_BASE_URL = "https://database.lichess.org/standard"
LICHESS_LICENSE_NAME = "Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"
LICHESS_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
LICHESS_LICENSE_SUMMARY = (
    "The Lichess game databases published at database.lichess.org are dedicated to the "
    "public domain under Creative Commons CC0 1.0 Universal. You can copy, modify, "
    "distribute, and perform the work, even for commercial purposes, without asking permission."
)
MAX_ALLOWED_BYTES_DEFAULT = 30 * 1_000_000_000  # 30 GB threshold


def get_database_license() -> dict[str, str]:
    """Return licensing details for the Lichess database."""
    return {
        "license": LICHESS_LICENSE_NAME,
        "url": LICHESS_LICENSE_URL,
        "summary": LICHESS_LICENSE_SUMMARY,
    }


def print_database_license() -> None:
    """Print licensing details for the Lichess database."""
    lic = get_database_license()
    print("=" * 70)
    print("LICHESS DATABASE LICENSE REPORT")
    print(f"License: {lic['license']}")
    print(f"Details: {lic['url']}")
    print(f"Summary: {lic['summary']}")
    print("=" * 70)


MONTH_REGEX = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")


def validate_month(month: str) -> str:
    """Validate month string format strictly as YYYY-MM."""
    m = str(month).strip()
    if not MONTH_REGEX.match(m):
        raise ValueError(f"Invalid month format {month!r}; expected YYYY-MM (e.g. '2024-08')")
    return m


def get_month_filename(month: str) -> str:
    """Format standard rated filename for a given month (YYYY-MM)."""
    m = validate_month(month)
    return f"lichess_db_standard_rated_{m}.pgn.zst"


def get_month_url(month: str) -> str:
    """Get full download URL for a given month (YYYY-MM)."""
    m = validate_month(month)
    return f"{LICHESS_BASE_URL}/{get_month_filename(m)}"


def get_remote_file_size(url: str, timeout: float = 10.0) -> int:
    """Query remote Content-Length via HTTP HEAD request."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ZeroPawn/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                return int(content_length)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to query remote file size at {url}: {e}") from e
    return -1


def check_size_limit(size_bytes: int, max_bytes: int = MAX_ALLOWED_BYTES_DEFAULT) -> bool:
    """Check if size in bytes exceeds the allowed limit."""
    if size_bytes > max_bytes:
        return False
    return True


def download_file(
    url: str,
    destination: Path,
    max_bytes: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
    allow_over_30gb: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download a file with resume support and size limit validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote_size = get_remote_file_size(url)
    print(f"Target URL: {url}")
    print(f"Remote file size: {remote_size} bytes ({remote_size / 1e9:.2f} GB)")

    if not allow_over_30gb and not check_size_limit(remote_size):
        raise ValueError(
            f"Remote file size ({remote_size / 1e9:.2f} GB) exceeds 30 GB limit. "
            "Approval required to download larger files."
        )

    # Pre-flight disk space check: ensure enough disk space for remote file + 1 GB safety buffer
    free_bytes = shutil.disk_usage(destination.parent).free
    needed_bytes = remote_size if remote_size > 0 else 1_000_000_000
    if free_bytes < (needed_bytes + 1_000_000_000):
        raise OSError(
            f"Insufficient disk space in {destination.parent}: {free_bytes / 1e9:.2f} GB free, "
            f"need at least {needed_bytes / 1e9:.2f} GB + 1 GB safety buffer."
        )

    headers = {"User-Agent": "ZeroPawn/1.0"}
    existing_bytes = 0
    if destination.exists():
        existing_bytes = destination.stat().st_size
        if remote_size > 0 and existing_bytes == remote_size:
            print(f"File already completely downloaded at {destination}")
            return destination
        if existing_bytes > 0:
            print(f"Resuming download from byte {existing_bytes}...")
            headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    total_downloaded = existing_bytes
    limit = (
        max_bytes if max_bytes is not None else (remote_size if remote_size > 0 else float("inf"))
    )

    t0 = time.time()
    last_print = t0

    mode = "ab" if existing_bytes > 0 else "wb"
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_code = getattr(resp, "code", getattr(resp, "status", None))
        if isinstance(resp_code, int) and resp_code == 200 and existing_bytes > 0:
            # Server sent full response (HTTP 200) instead of partial content (HTTP 206)
            mode = "wb"
            total_downloaded = 0
            existing_bytes = 0

        with open(destination, mode) as f_out:
            while total_downloaded < limit:
                remaining = int(limit - total_downloaded)
                read_sz = min(chunk_size, remaining)
                chunk = resp.read(read_sz)
                if not chunk:
                    break
                f_out.write(chunk)
                total_downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 2.0:
                    elapsed = now - t0
                    speed = (total_downloaded - existing_bytes) / elapsed if elapsed > 0 else 0
                    pct = (total_downloaded / remote_size * 100) if remote_size > 0 else 0
                    # In-place refresh only on a TTY; otherwise one plain line per update.
                    line = (
                        f"downloaded_mb={(total_downloaded / 1e6):.1f} "
                        f"total_mb={(remote_size / 1e6):.1f} "
                        f"pct={pct:.1f} speed_mbps={(speed / 1e6):.2f}"
                    )
                    if is_tty():
                        print(
                            f"\rDownloaded {total_downloaded / 1e6:.1f} MB / {remote_size / 1e6:.1f} MB "
                            f"({pct:.1f}%) - Speed: {speed / 1e6:.2f} MB/s",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
                    last_print = now

    print(f"\nDownload finished: {total_downloaded / 1e6:.1f} MB saved to {destination}")
    return destination


def main() -> None:
    setup_logging(log_dir="logs", log_filename="download.log")
    ctx = run_context("data-download")
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Download Lichess database monthly archive.")
    parser.add_argument(
        "--month", type=str, default="2026-09", help="Month to download in YYYY-MM format."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw", help="Directory to store downloaded files."
    )
    parser.add_argument(
        "--check-license", action="store_true", help="Print license information and exit."
    )
    parser.add_argument(
        "--info", action="store_true", help="Query and print file size without downloading."
    )
    parser.add_argument(
        "--max-bytes", type=int, default=None, help="Maximum number of bytes to download."
    )
    parser.add_argument(
        "--allow-over-30gb",
        action="store_true",
        help="Explicitly approve download exceeding 30 GB.",
    )

    args = parser.parse_args()

    emit_report(banner("data-download", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu",
            "n/a (network query)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("month", args.month, "user" if args.month != "2026-09" else "default"),
                (
                    "output_dir",
                    args.output_dir,
                    "user" if args.output_dir != "data/raw" else "default",
                ),
                ("check_license", args.check_license, "user" if args.check_license else "default"),
                ("info", args.info, "user" if args.info else "default"),
                ("max_bytes", args.max_bytes, "user" if args.max_bytes is not None else "default"),
                (
                    "allow_over_30gb",
                    args.allow_over_30gb,
                    "user" if args.allow_over_30gb else "default",
                ),
            ]
        )
    )

    try:
        print_database_license()

        if args.check_license:
            emit_report(artifacts_list([]))
            emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
            return

        url = get_month_url(args.month)
        size = get_remote_file_size(url)
        emit_report(
            results_table(
                [
                    ("month", f"{args.month} (M)"),
                    ("url", f"{url} (M)"),
                    ("size_bytes", f"{size} (M, HTTP HEAD Content-Length)"),
                    ("size_gb", f"{size / 1e9:.2f} (D)"),
                    ("size_gib", f"{size / (1024**3):.2f} (D)"),
                ]
            )
        )
        print(f"Month: {args.month}")
        print(f"URL: {url}")
        print(f"Size: {size} bytes ({size / 1e9:.2f} GB / {size / (1024**3):.2f} GiB)")

        if args.info:
            emit_report(artifacts_list([]))
            emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
            return

        dest = Path(args.output_dir) / get_month_filename(args.month)
        download_file(
            url=url,
            destination=dest,
            max_bytes=args.max_bytes,
            allow_over_30gb=args.allow_over_30gb,
        )
        emit_report(artifacts_list([file_record(str(dest))]))
        emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))
    except Exception as e:
        emit_error(
            error_block(
                f"data-download failed: {e}",
                "no network, bad --month (YYYY-MM), or server HEAD/download failed",
                "check --month and network access, then see LOG",
                "logs/download.log",
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)


if __name__ == "__main__":
    main()
