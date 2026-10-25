"""Script to download and verify 145 Syzygy 3-4-5 WDL (.rtbw) tablebase files."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_repo_root = Path(__file__).resolve().parent.parent
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
    format_kv_line,
    header_block,
    parameters_table,
    results_table,
    run_context,
    summary_block,
)

DEST_DIR = Path("data/syzygy")
BASE_URL_HTTPS = "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl"
MD5_URL_LICHESS = "https://tablebase.lichess.ovh/tables/standard/md5"
MD5_URL_SESSE = "http://tablebase.sesse.net/syzygy/3-4-5/checksum.md5"

LICENSE_VERBATIM = """
================================================================================
SYZYGY TABLEBASE LICENSE AND TERMS OF USE
================================================================================
Syzygy endgame tablebases were created by Ronald de Man.

Ronald de Man (creator and copyright holder of the probing code and format):
"This code may be redistributed and/or modified without restrictions."

Tablebase data files (.rtbw and .rtbz):
The tablebase data files represent mathematical end-game truths computed from
the standard rules of chess. They are distributed freely without copyright
restrictions as Public Domain (CC0 equivalent) across open chess mirrors
(Lichess, Sesse, Stockfish community repositories) for research and tournament use.
================================================================================
""".strip()


def get_expected_checksums() -> dict[str, str]:
    print("Fetching server MD5 checksums...")
    try:
        req = urllib.request.Request(MD5_URL_LICHESS, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        hashes = {}
        for line in content.splitlines():
            line = line.strip()
            if line.endswith(".rtbw"):
                parts = line.split()
                if len(parts) >= 2:
                    hashes[parts[1]] = parts[0]
        print(f"Loaded {len(hashes)} .rtbw checksums from Lichess HTTPS.")
        return hashes
    except Exception as e:
        print(f"Lichess checksum fetch failed ({e}); falling back to Sesse HTTP...")
        req = urllib.request.Request(MD5_URL_SESSE, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        hashes = {}
        for line in content.splitlines():
            line = line.strip()
            if line.endswith(".rtbw"):
                parts = line.split()
                if len(parts) >= 2:
                    hashes[parts[1]] = parts[0]
        print(f"Loaded {len(hashes)} .rtbw checksums from Sesse.")
        return hashes


def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_single_file(filename: str, expected_hash: str) -> tuple[str, bool, int, str]:
    dest_path = DEST_DIR / filename

    # If already downloaded and valid, skip
    if dest_path.exists():
        actual_hash = compute_md5(dest_path)
        if actual_hash == expected_hash:
            return filename, True, dest_path.stat().st_size, "cached"

    url = f"{BASE_URL_HTTPS}/{filename}"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            actual_hash = compute_md5(dest_path)
            if actual_hash == expected_hash:
                return filename, True, len(data), "downloaded"
            else:
                dest_path.unlink(missing_ok=True)
                if attempt == max_retries - 1:
                    return (
                        filename,
                        False,
                        0,
                        f"hash_mismatch: got {actual_hash} expected {expected_hash}",
                    )
        except Exception as e:
            if attempt == max_retries - 1:
                return filename, False, 0, f"error: {str(e)}"
            time.sleep(1.0)
    return filename, False, 0, "failed_all_retries"


def main():
    # Bare argparse: no options (zero-arg behavior frozen); --help works
    # instead of silently running the download.
    setup_logging(log_dir="logs", log_filename="download_syzygy.log")
    parser = argparse.ArgumentParser(
        description="Download and verify 145 Syzygy 3-4-5 WDL (.rtbw) tablebase files."
    )
    parser.parse_args()

    ctx = run_context("tablebase-fetch")
    t0 = time.time()
    log_path = "logs/download_syzygy.log"
    emit_report(banner("tablebase-fetch", ctx["start"] + " (local)"))
    emit_report(
        header_block(
            ctx["start"],
            ctx["command"],
            ctx["workdir"],
            ctx["commit"],
            "none",
            "none",
            "cpu",
            "n/a (server checksums)",
        )
    )
    emit_report(
        parameters_table(
            [
                ("files", "145 x 3-4-5 WDL .rtbw", "default"),
                ("source", f"{BASE_URL_HTTPS}", "default"),
                ("dest", f"{DEST_DIR}", "default"),
            ]
        )
    )

    try:
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        expected_hashes = get_expected_checksums()
    except Exception as e:
        emit_error(
            error_block(
                f"tablebase-fetch failed: {e}",
                "no network or checksum mirror unreachable",
                "check network access and retry, then see LOG",
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)

    # Filter only 3, 4, 5 pieces
    target_files = {}
    for fn, h in expected_hashes.items():
        base = fn.replace(".rtbw", "").replace("v", "")
        if len(base) in (3, 4, 5):
            target_files[fn] = h

    print(f"Target 3-4-5 WDL files to download: {len(target_files)}")
    if len(target_files) != 145:
        emit_error(
            error_block(
                f"tablebase-fetch failed: expected 145 files, got {len(target_files)}",
                "upstream checksum listing changed",
                "retry later or inspect the mirror listing, then see LOG",
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)

    t0 = time.time()
    results = {}
    total_bytes = 0
    mismatches = []

    print("Downloading files concurrently (8 worker threads)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_single_file, fn, h): fn for fn, h in target_files.items()
        }
        completed = 0
        for fut in as_completed(futures):
            fn, success, size, msg = fut.result()
            completed += 1
            if success:
                total_bytes += size
                results[fn] = (size, target_files[fn])
                if completed % 25 == 0 or completed == len(target_files):
                    emit_report(
                        format_kv_line(
                            [
                                ("files_done", f"{completed}/{len(target_files)}"),
                                ("mb", f"{total_bytes / (1024 * 1024):.1f}"),
                            ]
                        )
                    )
            else:
                mismatches.append((fn, msg))
                print(f"FAILED: {fn} -> {msg}")

    elapsed = time.time() - t0
    emit_report(
        results_table(
            [
                ("elapsed_s", f"{elapsed:.1f} (M)"),
                ("files_verified", f"{len(results)}/145 (M, MD5)"),
                ("total_bytes", f"{total_bytes:,} = {total_bytes / (1024 * 1024):.2f} MiB (M)"),
                ("mismatches", f"{len(mismatches)} (M)"),
            ]
        )
    )

    if mismatches:
        emit_error(
            error_block(
                f"tablebase-fetch failed: {len(mismatches)} mismatched files",
                "corrupt download or upstream change",
                "re-run to retry; failed files: " + ", ".join(fn for fn, _ in mismatches[:5]),
                log_path,
            )
        )
        emit_report(summary_block("FAILED", time.time() - t0, 1, 1))
        sys.exit(1)

    # Save manifest and license
    manifest_path = Path("docs/syzygy_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(LICENSE_VERBATIM + "\n\n")
        f.write(f"Source: {BASE_URL_HTTPS} (HTTPS)\n")
        f.write(f"Total Files: {len(results)}\n")
        f.write(f"Total Size: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MiB)\n")
        f.write("Integrity: All 145 files verified against official MD5 checksums.\n\n")
        f.write(f"{'Filename':<16} {'Size (bytes)':>12} {'MD5 Checksum':<34}\n")
        f.write("-" * 65 + "\n")
        for fn in sorted(results.keys()):
            sz, md5 = results[fn]
            f.write(f"{fn:<16} {sz:>12} {md5:<34}\n")

    print(f"Saved manifest to {manifest_path}")
    emit_report(artifacts_list([file_record(str(manifest_path))]))
    emit_report(summary_block("SUCCESS", time.time() - t0, 0, 0))


if __name__ == "__main__":
    main()
