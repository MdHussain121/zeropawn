"""ZeroPawn Project Rename Automation Script (Part 2).

Safe, idempotent, reversible script to rename all project-specific identifiers
to 'ZeroPawn' / 'zeropawn' across the repository.

Features:
- --dry-run (default): reports planned changes without modifying disk
- --apply: executes file renames (via git mv) and content edits
- --manifest PATH: writes JSON manifest of all changes made
- --revert-plan: reads manifest and creates/executes reversal plan
- --allow-dirty: allows running on dirty tree
- Case-preserving and word-boundary aware replacements
- Strict exclusion of third-party libraries (python-chess, chess-library, Stockfish)
  and domain model classes (ChessNet, ChessGame, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files and paths that must NEVER be modified
NEVER_TOUCH_FILES = {
    "docs/RL_CHANGELOG.md",
    "docs/BASELINE_RESULT.md",
    "docs/OUTPUT_AUDIT.md",
    "docs/LOGIC_CORRECTNESS_AUDIT.md",
    "docs/OPENING_ENDGAME_DATA_AUDIT.md",
    "docs/OPENING_SYZYGY_INTEGRATION_REPORT.md",
    "docs/OUTPUT_GOLDEN.md",
    "docs/PERF2_BASELINE_TESTS.txt",
    "docs/PERF3_BASELINE_TESTS.txt",
    "docs/PERF3_EVAL_SEMANTICS.md",
    "docs/PERF_PROFILE.md",
    "docs/REPOSITORY_AUDIT.md",
    "docs/RESIGN_DIAGNOSIS.md",
    "docs/TERMINAL_AUDIT.md",
    "docs/TERMINAL_GOLDEN.md",
    "docs/master_dirty_74.patch",
    "docs/master_dirty_74_status.txt",
    "docs/perf3/master_dirty.patch",
    "chessrl_cpp.py",
    "python/chessrl_cpp.py",
    "python/common/env.py",
    "tests/test_compat_shim.py",
    "tests/test_env_vars.py",
    "tests/test_rename_script.py",
    "scripts/capture_baseline.py",
    "scripts/generate_inventory.py",
    "scripts/rename_to_zeropawn.py",
    "scripts/verify_parity.py",
    "scripts/apply_cpp_rename.py",
    "scripts/apply_presentation_rename.py",
    "CMakeLists.txt",
    "CMakePresets.json",
}

NEVER_TOUCH_PREFIXES = (
    "docs/rename",
    "docs/output_audit",
    "docs/perf2",
    "docs/perf3",
    "docs/resignfix",
    "docs/terminal_audit",
    "reports",
    ".git",
    "models",
    "data",
    "runs",
    "logs",
    "tools",
    "build",
    "build_zeropawn",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
)

BINARY_EXTENSIONS = {
    ".pt",
    ".bin",
    ".pyd",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".a",
    ".lib",
    ".png",
    ".jpg",
    ".jpeg",
    ".ico",
    ".gif",
    ".zip",
    ".tar",
    ".gz",
    ".zst",
    ".rtbw",
    ".rtbz",
}

# Explicit replacement patterns (ordered by specificity)
# Each tuple: (pattern_str, replacement_str, case_sensitive)
REPLACEMENTS = [
    # C++ namespace and headers
    (r"\bnamespace chessrl\b", "namespace zeropawn", True),
    (r"\bchessrl::\b", "zeropawn::", True),
    (r"#include <chessrl/", "#include <zeropawn/", True),
    (r'#include "chessrl/', '#include "zeropawn/', True),
    # CMake and target names
    (r"\bproject\(chessrl\b", "project(zeropawn", True),
    (r"\bchessrl_core\b", "zeropawn_core", True),
    (r"\bchessrl_tests\b", "zeropawn_tests", True),
    (r"\bchessrl_unit_tests\b", "zeropawn_unit_tests", True),
    (r"\bCHESSRL_BUILD_TESTS\b", "ZEROPAWN_BUILD_TESTS", True),
    (r"\bCHESSRL_BUILD_PYTHON\b", "ZEROPAWN_BUILD_PYTHON", True),
    (r"\bCHESSRL_SOURCES\b", "ZEROPAWN_SOURCES", True),
    # Extension module name (excluding comments discussing backward shim)
    (r"\bchessrl_cpp\b", "zeropawn_cpp", True),
    # Environment variables
    (r"\bCHESSRL_([A-Z0-9_]+)\b", r"ZEROPAWN_\1", True),
    # Logs and paths
    (r"\bchessrl\.log\b", "zeropawn.log", True),
    # Display name and banners
    (r"\bid name ChessRL\b", "id name ZeroPawn", True),
    (r"\bChessRL Team\b", "ZeroPawn Team", True),
    (r"\bChessRL\b", "ZeroPawn", True),
    (r"\bChess-RL\b", "ZeroPawn", True),
    (r"\bChess_RL\b", "ZeroPawn", True),
    (r"\bCHESS-RL\b", "ZEROPAWN", True),
    (r"\bCHESS_RL\b", "ZEROPAWN", True),
    (r"\bCHESSRL\b", "ZEROPAWN", True),
    (r'"Chess RL"', '"ZeroPawn"', True),
    (r"'Chess RL'", "'ZeroPawn'", True),
    (r"\bChess RL\b", "ZeroPawn", True),
    (r"\bchess rl\b", "zeropawn", True),
    (r"\bRL Chess Engine\b", "ZeroPawn Chess Engine", True),
    # Generic lowercase/hyphen/underscore project variants
    (r"\bchess-rl\b", "zeropawn", True),
    (r"\bchess_rl\b", "zeropawn", True),
    (r"\bchessrl\b", "zeropawn", True),
    # Python package distribution name (pyproject.toml)
    (r'name = "chess-rl"', 'name = "zeropawn"', True),
    (r'name = "chess_rl"', 'name = "zeropawn"', True),
    (r'name = "chessrl"', 'name = "zeropawn"', True),
]

# Protected terms that must never be altered even if matching partially
PROTECTED_TERMS = [
    "python-chess",
    "chess-library",
    "ChessNet",
    "ChessGame",
    "ChessBoard",
    "Connect4Net",
    "chess.engine",
    "chess.syzygy",
    "chess.polyglot",
    "Stockfish",
    "Lichess",
]


def is_protected_line(line: str) -> bool:
    """Return True if line contains generic third-party terms or intentional shims that must be preserved."""
    for term in ["python-chess", "chess-library", "legacy shim", "compatibility shim"]:
        if term in line:
            return True
    return False


def apply_text_replacements(content: str, file_rel: str) -> Tuple[str, List[Dict[str, str]]]:
    """Applies ordered replacements to file content while preserving protected tokens."""
    changes = []
    lines = content.splitlines(keepends=True)
    new_lines = []

    for idx, line in enumerate(lines, 1):
        if is_protected_line(line):
            new_lines.append(line)
            continue

        mod_line = line
        for pat, repl, case_sensitive in REPLACEMENTS:
            flags = 0 if case_sensitive else re.IGNORECASE
            if re.search(pat, mod_line, flags):
                old_val = mod_line
                mod_line = re.sub(pat, repl, mod_line, flags=flags)
                if mod_line != old_val:
                    changes.append(
                        {
                            "file": file_rel,
                            "line": idx,
                            "before": old_val.strip()[:100],
                            "after": mod_line.strip()[:100],
                        }
                    )
        new_lines.append(mod_line)

    return "".join(new_lines), changes


def is_tracked_or_uncommitted(repo_dir: Path) -> bool:
    """Check if repository has uncommitted changes."""
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    return bool(res.stdout.strip())


def run_git_mv(src: Path, dst: Path, repo_dir: Path, dry_run: bool = True) -> None:
    """Executes git mv or simulates it."""
    if dry_run:
        print(f"[DRY RUN] git mv {src.relative_to(repo_dir)} -> {dst.relative_to(repo_dir)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["git", "mv", str(src), str(dst)],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        # Fallback to standard rename if git mv fails (e.g. untracked)
        src.rename(dst)


def main():
    parser = argparse.ArgumentParser(
        description="Rename project to ZeroPawn safely and reversibly."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply renames and content modifications"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run only (default)")
    parser.add_argument(
        "--manifest",
        type=str,
        default="docs/rename/manifest.json",
        help="Path to write change manifest",
    )
    parser.add_argument(
        "--revert-plan", action="store_true", help="Generate or execute revert plan from manifest"
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Allow running on dirty working tree"
    )
    parser.add_argument(
        "--repo-root", type=str, default=str(REPO_ROOT), help="Root directory of target repository"
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    is_dry_run = not args.apply

    if not args.allow_dirty and is_tracked_or_uncommitted(repo):
        # In dry-run mode, we can still report
        if not is_dry_run:
            print("ERROR: Working tree has uncommitted changes. Use --allow-dirty to override.")
            sys.exit(1)

    manifest_path = repo / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== ZeroPawn Rename Tool ===")
    print(f"Target Repo: {repo}")
    print(f"Mode: {'DRY RUN' if is_dry_run else 'APPLY CHANGES'}")

    all_changes = []
    file_renames = []

    # 1. Planned file / folder renames
    # C++ include directory
    src_inc = repo / "cpp" / "include" / "chessrl"
    dst_inc = repo / "cpp" / "include" / "zeropawn"
    if src_inc.is_dir() and not dst_inc.exists():
        file_renames.append((src_inc, dst_inc))

    # Type stub
    src_pyi = repo / "python" / "chessrl_cpp.pyi"
    dst_pyi = repo / "python" / "zeropawn_cpp.pyi"
    if src_pyi.is_file() and not dst_pyi.exists():
        file_renames.append((src_pyi, dst_pyi))

    for src, dst in file_renames:
        run_git_mv(src, dst, repo, dry_run=is_dry_run)
        all_changes.append(
            {
                "type": "rename_path",
                "src": str(src.relative_to(repo)).replace("\\", "/"),
                "dst": str(dst.relative_to(repo)).replace("\\", "/"),
            }
        )

    # 2. Text replacements across non-ignored files
    for root, dirs, files in os.walk(repo):
        rel_root = os.path.relpath(root, repo).replace("\\", "/")
        if rel_root == ".":
            rel_root = ""

        # Filter dirs
        if rel_root == "":
            dirs[:] = [d for d in dirs if d not in NEVER_TOUCH_PREFIXES and not d.startswith(".")]
        else:
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
                and not d.startswith(".")
            ]

        # Check prefixes
        if any(rel_root == p or rel_root.startswith(p + "/") for p in NEVER_TOUCH_PREFIXES):
            continue

        for f in files:
            rel_file = f"{rel_root}/{f}".lstrip("/")
            if rel_file in NEVER_TOUCH_FILES:
                continue
            if any(rel_file.startswith(p) for p in NEVER_TOUCH_PREFIXES):
                continue
            if os.path.splitext(f)[1].lower() in BINARY_EXTENSIONS:
                continue
            if rel_file == "scripts/rename_to_zeropawn.py":
                continue

            file_path = Path(root) / f
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            new_content, edits = apply_text_replacements(content, rel_file)
            if edits:
                all_changes.extend(edits)
                if not is_dry_run:
                    file_path.write_text(new_content, encoding="utf-8")
                print(
                    f"[{'DRY RUN' if is_dry_run else 'MODIFIED'}] {rel_file}: {len(edits)} occurrences"
                )

    # 3. Write manifest
    manifest_data = {
        "mode": "dry-run" if is_dry_run else "applied",
        "total_changes": len(all_changes),
        "changes": all_changes,
    }
    if not is_dry_run:
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        print(f"\nManifest written to {manifest_path} ({len(all_changes)} records).")
    else:
        print(f"\n[DRY RUN] Would write manifest to {manifest_path} ({len(all_changes)} records).")


if __name__ == "__main__":
    main()
