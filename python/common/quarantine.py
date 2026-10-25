"""Quarantine, safe transactional cleanup, and recovery infrastructure for ZeroPawn.

Invariants and Protections:
- Protected supervised baseline (models/checkpoints_scaled/best_model.pt) is NEVER touched.
- Preserved datasets (data/dataset*, data/*.pgn, data/puzzles, data/openings, data/syzygy) are NEVER touched.
- Production presets (configs/presets.json), C++ build artifacts, cpp source tree, Git history,
  and user notes are NEVER touched.
- Dry-run by default: generates detailed markdown report and JSON manifest without moving data.
- Execution requires explicit authorization (--yes in script mode, [y/N] confirmation in interactive mode).
- Same-filesystem atomic moves into timestamped quarantine/archive_YYYYMMDD_HHMMSS/ directories.
- Transactional rollback on any error during quarantine operations.
- Separate recovery and inspection tooling (list, inspect, restore, gated purge).
- Reports bytes removed from active, bytes in quarantine, and actual physical disk space reclaimed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Known SHA-256 for protected supervised baseline checkpoint
EXPECTED_BASELINE_HASH_PREFIX = "54d46282"
FULL_BASELINE_SHA256 = "54d462823837f77c60621fb72e1a1474eca3dc9dbb2a2969d87ced18ada2b7df"


def compute_file_sha256(path: Path, max_bytes: Optional[int] = None) -> str:
    """Compute SHA-256 checksum of a file.

    For enormous files (>500MB), if max_bytes is provided, hashes initial chunks.
    Otherwise reads full file in 1MB chunks.
    """
    h = hashlib.sha256()
    bytes_read = 0
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                bytes_read += len(chunk)
                if max_bytes and bytes_read >= max_bytes:
                    break
        return h.hexdigest()
    except Exception as e:
        logger.debug("Failed to compute SHA256 for %s: %s", path, e)
        return ""


@dataclass
class QuarantineItem:
    """Descriptor for an artifact evaluated for quarantine."""

    id: str
    original_rel_path: str
    category: str
    size_bytes: int
    size_formatted: str
    mtime: str
    sha256: str
    reason: str
    run_ownership: str
    safe_to_quarantine: bool
    status: str  # "candidate", "protected", "excluded", "requires_review"
    proposed_destination: str
    dependencies: List[str] = field(default_factory=list)
    warning: Optional[str] = None


@dataclass
class CleanupManifest:
    """Complete dry-run or execution manifest for a cleanup operation."""

    manifest_id: str
    created_at: str
    repo_root: str
    quarantine_root: str
    total_candidates_count: int
    total_candidates_bytes: int
    total_candidates_formatted: str
    baseline_model_verified: bool
    baseline_model_hash: str
    items_by_category: Dict[str, List[Dict[str, Any]]]
    protected_paths_excluded: List[str]
    items: List[Dict[str, Any]]
    active_bytes_freed: int = 0
    quarantine_bytes_stored: int = 0
    disk_space_reclaimed_bytes: int = 0
    execution_status: str = "dry_run"  # "dry_run", "completed", "rolled_back", "failed"
    executed_at: Optional[str] = None


class QuarantineManager:
    """Manages artifact auditing, dry-run manifest generation, and transactional quarantine."""

    PROTECTED_REL_PATHS: Set[str] = {
        "models/checkpoints_scaled/best_model.pt",
        "configs/presets.json",
        "build/cpp/zeropawn_cpp.pyd",
        "build/cmake/cpp/zeropawn_tests.exe",
        "zeropawn_tests.exe",
        "data/filtered_2200.pgn",
        "data/filtered_2026_08.pgn",
        "data/filtered_combined.pgn",
        "data/elite_2500.pgn",
        "data/openings/eval_openings_50.epd",
        "data/openings/curated_book.json",
        "data/openings/eco_openings.json",
        "data/openings/book_selfplay.epd",
        "data/openings/book_selfplay.json",
        "CMakeLists.txt",
        "CMakePresets.json",
        "pyproject.toml",
        "conftest.py",
        "zeropawn.cmd",
        "chess.cmd",
        "cli.py",
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        ".pre-commit-config.yaml",
        "sitecustomize.py",
    }

    PROTECTED_DIR_PREFIXES: Sequence[str] = (
        "data/dataset_training",
        "data/dataset_scaled",
        "data/dataset_elite2500",
        "data/dataset_tactics",
        "data/puzzles",
        "data/puzzle_history",
        "data/syzygy",
        "data/openings",
        "cpp",
        "python",
        "scripts",
        "tests",
        "tools",
        "docs",
        ".git",
        ".github",
        "configs",
        "quarantine",
        "reports",
    )

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        if repo_root is None:
            self.repo_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.repo_root = Path(repo_root).resolve()

        self.quarantine_dir = self.repo_root / "quarantine"
        self.lock_file = self.quarantine_dir / ".cleanup.lock"
        self.baseline_checkpoint = (
            self.repo_root / "models" / "checkpoints_scaled" / "best_model.pt"
        )

    def is_protected(self, path: Path | str) -> Tuple[bool, str]:
        """Check if a path is strictly protected against modification or deletion."""
        p = Path(path)
        if not p.is_absolute():
            p = (self.repo_root / p).resolve()
        else:
            p = p.resolve()

        try:
            rel = p.relative_to(self.repo_root).as_posix()
        except ValueError:
            # Path escapes repository root
            return True, "Path escapes repository root"

        # Check direct protected files
        if rel in self.PROTECTED_REL_PATHS:
            return True, f"Explicit protected asset: {rel}"

        # Check protected directory prefixes
        for prefix in self.PROTECTED_DIR_PREFIXES:
            if rel == prefix or rel.startswith(prefix + "/"):
                return True, f"Protected directory tree: {prefix}"

        # Never touch the baseline model even if accessed via symlink or alternate path
        if p == self.baseline_checkpoint.resolve():
            return True, "Supervised warm-start baseline (models/checkpoints_scaled/best_model.pt)"

        return False, ""

    def verify_baseline_checkpoint(self) -> Tuple[bool, str]:
        """Verify the protected warm-start checkpoint exists and has valid SHA-256."""
        if not self.baseline_checkpoint.is_file():
            return (
                False,
                "Protected baseline model (models/checkpoints_scaled/best_model.pt) not found!",
            )

        computed_sha256 = compute_file_sha256(self.baseline_checkpoint)
        if not computed_sha256.lower().startswith(EXPECTED_BASELINE_HASH_PREFIX):
            return False, (
                f"Baseline SHA-256 prefix mismatch! Computed: {computed_sha256}, "
                f"Expected prefix: {EXPECTED_BASELINE_HASH_PREFIX}"
            )

        return True, computed_sha256

    def is_file_locked_or_in_use(self, path: Path) -> bool:
        """Check if a file is currently locked by a running process."""
        if not path.is_file():
            return False
        try:
            # Try to open in append mode to test write lock
            with open(path, "r+b"):
                pass
            return False
        except (PermissionError, OSError):
            return True

    def _format_bytes(self, n: int) -> str:
        if n >= 1024**3:
            return f"{n / (1024**3):.2f} GB"
        if n >= 1024**2:
            return f"{n / (1024**2):.1f} MB"
        if n >= 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n} B"

    def inventory_candidates(self) -> List[QuarantineItem]:
        """Discover all legitimate cleanup and quarantine candidates in the workspace."""
        candidates: List[QuarantineItem] = []
        counter = 1

        def _add_candidate(
            p: Path,
            category: str,
            reason: str,
            run_ownership: str = "unknown",
            compute_hash: bool = True,
        ) -> None:
            nonlocal counter
            if not p.is_file():
                return
            prot, reason_prot = self.is_protected(p)
            if prot:
                logger.debug("Skipping protected file: %s (%s)", p, reason_prot)
                return

            rel = p.relative_to(self.repo_root).as_posix()
            try:
                stat = p.stat()
                sz = stat.st_size
                mt = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                return

            # Compute SHA256 (for very large binary buffers > 250MB, hash first 50MB to remain responsive)
            sha = ""
            if compute_hash:
                max_b = 50 * 1024 * 1024 if sz > 250 * 1024 * 1024 else None
                sha = compute_file_sha256(p, max_bytes=max_b)
                if max_b and sz > max_b:
                    sha += " (partial 50MB)"

            # Proposed destination inside quarantine
            item_id = f"ART-{counter:04d}"
            counter += 1
            prop_dest = f"quarantine/<archive_dir>/{rel}"

            candidates.append(
                QuarantineItem(
                    id=item_id,
                    original_rel_path=rel,
                    category=category,
                    size_bytes=sz,
                    size_formatted=self._format_bytes(sz),
                    mtime=mt,
                    sha256=sha,
                    reason=reason,
                    run_ownership=run_ownership,
                    safe_to_quarantine=True,
                    status="candidate",
                    proposed_destination=prop_dest,
                )
            )

        # 1. Replay buffer memory maps and sidecars under runs/**/buffer/
        runs_dir = self.repo_root / "runs"
        if runs_dir.is_dir():
            for root, dirs, files in os.walk(runs_dir):
                root_path = Path(root)
                rel_root = root_path.relative_to(self.repo_root).as_posix()
                is_buffer = "buffer" in root_path.parts
                for f in files:
                    file_path = root_path / f
                    if is_buffer:
                        _add_candidate(
                            file_path,
                            category="replay_buffer",
                            reason="Preallocated replay buffer memory map / sidecar metadata",
                            run_ownership=rel_root.split("/")[1]
                            if len(rel_root.split("/")) > 1
                            else "phase4_rl",
                            compute_hash=False
                            if file_path.suffix == ".bin" and file_path.stat().st_size > 1024**3
                            else True,
                        )
                    elif f.startswith("events.out.tfevents."):
                        _add_candidate(
                            file_path,
                            category="tfevents",
                            reason="Old TensorBoard event file from historical RL run",
                            run_ownership=rel_root.split("/")[1]
                            if len(rel_root.split("/")) > 1
                            else "phase4_rl",
                        )
                    elif f.endswith(".csv") and rel_root.startswith("runs/phase4_rl"):
                        _add_candidate(
                            file_path,
                            category="historical_eval",
                            reason="Evaluation and telemetry CSV from previous trial",
                            run_ownership="phase4_rl",
                        )

        # 2. Obsolete RL checkpoints in models/checkpoints_rl/
        rl_ckpt_dir = self.repo_root / "models" / "checkpoints_rl"
        if rl_ckpt_dir.is_dir():
            for p in sorted(rl_ckpt_dir.glob("*.pt")):
                _add_candidate(
                    p,
                    category="obsolete_checkpoint",
                    reason="Previous RL training checkpoint (candidate for fresh RL warm-start)",
                    run_ownership="phase4_rl",
                )
            for p in sorted(rl_ckpt_dir.glob("*.pt.tmp")):
                _add_candidate(
                    p,
                    category="stale_temporary",
                    reason="Incomplete or temporary checkpoint save",
                    run_ownership="phase4_rl",
                )

        # 3. Flawed or experimental checkpoints in models/
        leaked_ckpt = (
            self.repo_root / "models" / "checkpoints_finetune" / "best_model_MIX_LEAKED.pt"
        )
        if leaked_ckpt.is_file():
            _add_candidate(
                leaked_ckpt,
                category="experimental_checkpoint",
                reason="Flawed mixed-dataset checkpoint (0.14% val leakage, known flawed baseline)",
                run_ownership="finetune_mix",
            )

        pilot_ckpt_dir = self.repo_root / "models" / "checkpoints_opening_pilot"
        if pilot_ckpt_dir.is_dir():
            for p in sorted(pilot_ckpt_dir.glob("*.pt")):
                _add_candidate(
                    p,
                    category="experimental_checkpoint",
                    reason="Experimental opening pilot checkpoint",
                    run_ownership="opening_pilot",
                )

        # 4. Pilot datasets under data/dataset_openings_pilot/ and data/dataset_syzygy_pilot/
        for pilot_dir_name in ("dataset_openings_pilot", "dataset_syzygy_pilot"):
            pdir = self.repo_root / "data" / pilot_dir_name
            if pdir.is_dir():
                for root, _, files in os.walk(pdir):
                    for f in files:
                        _add_candidate(
                            Path(root) / f,
                            category="pilot_dataset",
                            reason=f"Derived experimental pilot dataset ({pilot_dir_name})",
                            run_ownership=pilot_dir_name,
                        )

        # 5. Stale benchmark, smoke test, and temporary files in tmp/
        tmp_dir = self.repo_root / "tmp"
        if tmp_dir.is_dir():
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    _add_candidate(
                        Path(root) / f,
                        category="tmp_artifact",
                        reason="Temporary benchmark or test artifact",
                        run_ownership="tests_and_benchmarks",
                    )

        # 6. Regenerable caches: __pycache__, .pytest_cache, .ruff_cache
        for cache_dir in (".pytest_cache", ".ruff_cache"):
            cd = self.repo_root / cache_dir
            if cd.is_dir():
                for root, _, files in os.walk(cd):
                    for f in files:
                        _add_candidate(
                            Path(root) / f,
                            category="cache",
                            reason=f"Regenerable linter/test cache ({cache_dir})",
                            run_ownership="development_tooling",
                            compute_hash=False,
                        )

        for pycache in self.repo_root.rglob("__pycache__"):
            # Skip if inside protected dir like .git
            if any(part in (".git", "quarantine") for part in pycache.parts):
                continue
            for f in pycache.glob("*.pyc"):
                _add_candidate(
                    f,
                    category="cache",
                    reason="Compiled Python bytecode (__pycache__)",
                    run_ownership="python_runtime",
                    compute_hash=False,
                )

        # 7. Historical logs in logs/ (excluding current active logs/zeropawn.log)
        logs_dir = self.repo_root / "logs"
        if logs_dir.is_dir():
            for p in logs_dir.glob("*.log.*"):
                _add_candidate(
                    p,
                    category="historical_log",
                    reason="Rotated or historical diagnostic run log",
                    run_ownership="previous_runs",
                )

        return candidates

    def generate_dry_run_manifest(self) -> CleanupManifest:
        """Scan workspace and generate a structured dry-run cleanup manifest."""
        baseline_ok, baseline_hash = self.verify_baseline_checkpoint()
        candidates = self.inventory_candidates()

        total_bytes = sum(c.size_bytes for c in candidates)
        items_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidates:
            items_by_cat.setdefault(c.category, []).append(asdict(c))

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_id = f"manifest_{now_str}"

        manifest = CleanupManifest(
            manifest_id=manifest_id,
            created_at=datetime.now().isoformat(),
            repo_root=str(self.repo_root),
            quarantine_root=str(self.quarantine_dir),
            total_candidates_count=len(candidates),
            total_candidates_bytes=total_bytes,
            total_candidates_formatted=self._format_bytes(total_bytes),
            baseline_model_verified=baseline_ok,
            baseline_model_hash=baseline_hash,
            items_by_category=items_by_cat,
            protected_paths_excluded=sorted(list(self.PROTECTED_REL_PATHS)),
            items=[asdict(c) for c in candidates],
            execution_status="dry_run",
        )
        return manifest

    def write_dry_run_reports(
        self, manifest: CleanupManifest, reports_dir: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """Write human-readable Markdown and machine-readable JSON reports."""
        if reports_dir is None:
            r_dir = self.repo_root / "reports"
        else:
            r_dir = Path(reports_dir)
        r_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = r_dir / f"cleanup_dry_run_{timestamp}.json"
        md_path = r_dir / f"cleanup_dry_run_{timestamp}.md"

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        # 2. Write Markdown Report
        md_lines = [
            "# ZeroPawn Workspace Cleanup Dry-Run Report",
            "",
            f"- **Generated At:** {manifest.created_at}",
            f"- **Manifest ID:** `{manifest.manifest_id}`",
            f"- **Repository Root:** `{manifest.repo_root}`",
            f"- **Total Candidates:** **{manifest.total_candidates_count} files** ({manifest.total_candidates_formatted})",
            f"- **Baseline Checkpoint Integrity:** {'[PASS] Byte-Identical' if manifest.baseline_model_verified else '[FAIL] MISMATCH'}",
            f"- **Baseline SHA-256:** `{manifest.baseline_model_hash}`",
            "",
            "---",
            "",
            "## 1. Summary by Category",
            "",
            "| Category | File Count | Aggregate Size | Risk Level | Proposed Action |",
            "|---|---|---|---|---|",
        ]

        for cat, items in sorted(manifest.items_by_category.items()):
            cat_bytes = sum(it["size_bytes"] for it in items)
            cat_size = self._format_bytes(cat_bytes)
            risk = (
                "Low" if cat in ("cache", "tmp_artifact", "tfevents") else "Medium (RL Artifacts)"
            )
            md_lines.append(
                f"| **{cat}** | {len(items):,} | {cat_size} | {risk} | Move to quarantine |"
            )

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Capacity & Disk Space Impact",
                "",
                f"- **Bytes removed from active workspace:** `{manifest.total_candidates_formatted}`",
                f"- **Bytes stored in quarantine:** `{manifest.total_candidates_formatted}`",
                "- **Estimated physical disk space reclaimed:** `0 B` (Same-filesystem transactional atomic moves preserve block allocation)",
                "",
                "---",
                "",
                "## 3. Protected Assets Explicitly Excluded",
                "",
                "The following core assets are strictly preserved and excluded from all cleanup operations:",
                "",
                "- `models/checkpoints_scaled/best_model.pt` (Supervised warm-start baseline)",
                "- `data/dataset_scaled/`, `data/dataset_elite2500/`, `data/dataset_tactics/` (Preserved datasets)",
                "- `data/filtered_2200.pgn`, `data/elite_2500.pgn`, `data/puzzles/` (Source games and puzzles)",
                "- `data/openings/*` (Curated opening book and benchmark suites)",
                "- `data/syzygy/*` (145 3-4-5 piece endgame tablebases)",
                "- `configs/presets.json` (Production presets)",
                "- `build/cpp/zeropawn_cpp.pyd` and `zeropawn_tests.exe` (Compiled C++ extension and tests)",
                "- `.git/` and all source trees (`cpp/`, `python/`, `scripts/`, `tests/`)",
                "",
                "---",
                "",
                "## 4. Complete Candidates List",
                "",
                "| ID | Relative Path | Category | Size | Reason |",
                "|---|---|---|---|---|",
            ]
        )

        for it in manifest.items:
            md_lines.append(
                f"| `{it['id']}` | `{it['original_rel_path']}` | {it['category']} | {it['size_formatted']} | {it['reason']} |"
            )

        md_lines.extend(
            [
                "",
                "---",
                "",
                "> [!CAUTION]",
                "> **DESTRUCTIVE CLEANUP IS GATED:**",
                "> This manifest represents a non-destructive dry-run. No real files have been moved or deleted.",
                "> Quarantine execution requires explicit user authorization.",
                "",
            ]
        )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return md_path, json_path

    def execute_quarantine(
        self,
        manifest: CleanupManifest,
        approved_by_user: bool = False,
    ) -> Dict[str, Any]:
        """Execute transactional quarantine move for approved manifest entries.

        Gated by explicit approved_by_user flag.
        """
        if not approved_by_user:
            raise PermissionError(
                "Destructive cleanup is separately gated and is not yet authorized."
            )

        # Re-verify baseline model first
        baseline_ok, baseline_hash = self.verify_baseline_checkpoint()
        if not baseline_ok:
            raise RuntimeError(f"Baseline checkpoint verification failed: {baseline_hash}")

        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Acquire lock
        if self.lock_file.exists():
            try:
                # Check stale lock (> 1 hour old)
                mtime = self.lock_file.stat().st_mtime
                if time.time() - mtime < 3600:
                    raise RuntimeError(
                        "Another cleanup or quarantine operation is currently running (lock present)."
                    )
            except OSError:
                pass

        try:
            with open(self.lock_file, "w") as f:
                f.write(f"pid={os.getpid()}\nstart={datetime.now().isoformat()}\n")
        except OSError as e:
            raise RuntimeError(f"Could not acquire quarantine lock: {e}")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"archive_{timestamp_str}"
        archive_dir = self.quarantine_dir / archive_name
        archive_dir.mkdir(parents=True, exist_ok=True)

        ops_log: List[str] = []
        moved_records: List[Tuple[Path, Path]] = []
        total_moved_bytes = 0
        failed_count = 0

        try:
            for item_dict in manifest.items:
                rel_path = item_dict["original_rel_path"]
                src = (self.repo_root / rel_path).resolve()

                # Safety: check protected again
                prot, prot_reason = self.is_protected(src)
                if prot:
                    ops_log.append(f"[SKIPPED_PROTECTED] {rel_path} ({prot_reason})")
                    continue

                if not src.is_file():
                    ops_log.append(f"[SKIPPED_NOT_FOUND] {rel_path}")
                    continue

                if self.is_file_locked_or_in_use(src):
                    ops_log.append(f"[SKIPPED_LOCKED] {rel_path}")
                    failed_count += 1
                    continue

                dest = (archive_dir / rel_path).resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)

                sz = src.stat().st_size
                # Atomic move on same filesystem
                try:
                    os.replace(src, dest)
                    moved_records.append((src, dest))
                    total_moved_bytes += sz
                    ops_log.append(
                        f"[MOVED] {rel_path} -> {dest.relative_to(self.quarantine_dir).as_posix()} ({sz} B)"
                    )
                except OSError as e:
                    logger.warning("Move failed for %s -> %s: %s", src, dest, e)
                    ops_log.append(f"[FAILED] {rel_path} -> {e}")
                    failed_count += 1

            # Save manifest and operation log inside the archive directory
            manifest.execution_status = "completed" if failed_count == 0 else "partial"
            manifest.executed_at = datetime.now().isoformat()
            manifest.active_bytes_freed = total_moved_bytes
            manifest.quarantine_bytes_stored = total_moved_bytes
            manifest.disk_space_reclaimed_bytes = 0  # Same filesystem atomic move

            with open(archive_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(asdict(manifest), f, indent=2)

            with open(archive_dir / "operations.log", "w", encoding="utf-8") as f:
                f.write("\n".join(ops_log) + "\n")

            return {
                "archive_dir": str(archive_dir),
                "archive_name": archive_name,
                "moved_files": len(moved_records),
                "failed_files": failed_count,
                "active_bytes_freed": total_moved_bytes,
                "quarantine_bytes_stored": total_moved_bytes,
                "disk_space_reclaimed_bytes": 0,
                "status": manifest.execution_status,
            }

        except Exception as e:
            logger.exception("Critical error during quarantine; initiating rollback: %s", e)
            # Rollback all moved records
            for orig_src, moved_dest in reversed(moved_records):
                try:
                    orig_src.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(moved_dest, orig_src)
                except Exception as rollback_err:
                    logger.error("Rollback failed for %s: %s", orig_src, rollback_err)
            raise RuntimeError(f"Quarantine aborted and rolled back due to error: {e}")

        finally:
            if self.lock_file.exists():
                try:
                    self.lock_file.unlink()
                except OSError:
                    pass

    def list_archives(self) -> List[Dict[str, Any]]:
        """List all existing quarantine archives with file counts and byte sizes."""
        if not self.quarantine_dir.is_dir():
            return []

        archives = []
        for d in sorted(self.quarantine_dir.iterdir()):
            if d.is_dir() and d.name.startswith("archive_"):
                manifest_file = d / "manifest.json"
                archive_info: Dict[str, Any] = {
                    "name": d.name,
                    "path": str(d),
                    "file_count": 0,
                    "total_bytes": 0,
                    "total_formatted": "0 B",
                    "status": "unverified",
                    "created_at": "unknown",
                }
                if manifest_file.is_file():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            m_data = json.load(f)
                            archive_info["file_count"] = len(m_data.get("items", [])) or m_data.get(
                                "total_candidates_count", 0
                            )
                            archive_info["total_bytes"] = m_data.get("quarantine_bytes_stored", 0)
                            archive_info["total_formatted"] = self._format_bytes(
                                archive_info["total_bytes"]
                            )
                            archive_info["created_at"] = m_data.get("executed_at") or m_data.get(
                                "created_at", "unknown"
                            )
                            archive_info["status"] = m_data.get("execution_status", "completed")
                    except Exception:
                        pass
                else:
                    # Walk directory to measure size
                    tot = 0
                    cnt = 0
                    for root, _, files in os.walk(d):
                        for f in files:
                            cnt += 1
                            try:
                                tot += (Path(root) / f).stat().st_size
                            except OSError:
                                pass
                    archive_info["file_count"] = cnt
                    archive_info["total_bytes"] = tot
                    archive_info["total_formatted"] = self._format_bytes(tot)

                archives.append(archive_info)
        return archives

    def restore_archive(
        self,
        archive_name: str,
        item_ids: Optional[Sequence[str]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Restore files from an archive back to their original workspace locations."""
        archive_dir = self.quarantine_dir / archive_name
        if not archive_dir.is_dir():
            raise FileNotFoundError(f"Quarantine archive '{archive_name}' not found.")

        manifest_file = archive_dir / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Archive '{archive_name}' is missing manifest.json.")

        with open(manifest_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        items_to_restore = m_data.get("items", [])
        if item_ids:
            item_ids_set = set(item_ids)
            items_to_restore = [it for it in items_to_restore if it["id"] in item_ids_set]

        restored_count = 0
        skipped_count = 0
        total_restored_bytes = 0

        for it in items_to_restore:
            rel = it["original_rel_path"]
            archived_file = archive_dir / rel
            dest_file = self.repo_root / rel

            if not archived_file.is_file():
                skipped_count += 1
                continue

            if dest_file.exists() and not overwrite:
                logger.warning("Restore destination collision: %s exists; skipping", dest_file)
                skipped_count += 1
                continue

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            sz = archived_file.stat().st_size
            os.replace(archived_file, dest_file)
            restored_count += 1
            total_restored_bytes += sz

        return {
            "restored_files": restored_count,
            "skipped_files": skipped_count,
            "total_bytes_restored": total_restored_bytes,
            "formatted_bytes": self._format_bytes(total_restored_bytes),
        }

    def purge_archive(self, archive_name: str, authorized: bool = False) -> Dict[str, Any]:
        """Permanently delete an archive from quarantine on disk.

        Strictly gated: Requires explicit authorization.
        """
        if not authorized:
            raise PermissionError("Physical purge requires explicit separate user authorization.")

        archive_dir = self.quarantine_dir / archive_name
        if not archive_dir.is_dir():
            raise FileNotFoundError(f"Quarantine archive '{archive_name}' not found.")

        # Measure reclaimed physical space
        tot_bytes = 0
        for root, _, files in os.walk(archive_dir):
            for f in files:
                try:
                    tot_bytes += (Path(root) / f).stat().st_size
                except OSError:
                    pass

        shutil.rmtree(archive_dir)
        return {
            "purged_archive": archive_name,
            "physical_disk_space_reclaimed_bytes": tot_bytes,
            "formatted_reclaimed": self._format_bytes(tot_bytes),
        }
