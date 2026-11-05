"""Record native extension provenance (SA5-01).

Generates a JSON manifest tracing the deployed native binary
(``build/cpp/zeropawn_cpp*.pyd``) to the source tree and toolchain that
produced it.  The manifest records, where derivable:

- the artifact (repo-relative path, SHA-256, size, mtime, Python ABI tag)
- the source revision (git HEAD + tracked-file dirtiness) and SHA-256 of
  every C++ source/header and CMake listfile in ``cpp/``
- the compiler identity/version, build configuration, generator, CMake
  version and actual compile flags — derived from the *persisted CMake
  configure metadata* of the matching Ninja build tree under ``build/``
- the pybind11 version fetched by that build tree
- a byte-identity check between the deployed artifact and the surviving
  build-tree output

Truthfulness contract (SA5-01): the manifest NEVER claims the deployed
binary was reproducibly rebuilt from source.  ``provenance_level`` is
computed from what could actually be established:

- ``recorded-from-build-metadata`` — toolchain identity derived from the
  build tree's persisted configure metadata; the artifact itself was NOT
  rebuilt by this script.
- ``artifact-only`` — no build metadata found; nothing beyond the binary
  itself can be asserted.

The generated manifest is written to ``build/native_provenance.json``
(gitignored, like the build tree it describes).  ``--check`` re-derives
the artifact and source hashes and exits non-zero on drift.

Usage::

    python scripts/record_native_provenance.py            # write manifest
    python scripts/record_native_provenance.py --check    # validate + drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "zeropawn-native-provenance/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("build/native_provenance.json")

_PYD_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)[-.](?P<abi>cp\d+)(?:[-.](?P<plat>[\w]+))?\.pyd$")
_SET_LINE_RE = re.compile(r'^set\(\s*(\w+)\s+"([^"]*)"\s*\)$')
_CXX_STD_RE = re.compile(r"set\(CMAKE_CXX_STANDARD\s+(\d+)\)")
_PKGVERSION_RE = re.compile(r"--with-pkgversion='([^']+)'")
_CACHE_TUPLE_RE = re.compile(r"^(\w+):(STRING|FILEPATH|PATH|INTERNAL|BOOL|UNINITIALIZED)=(.*)$")
_VERSION_H_RE = re.compile(r"#define\s+PYBIND11_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

#: C++ source extensions hashed into the manifest's ``source.files`` map.
SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")


def _now_utc() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _generic_exe(exe_value: str) -> str:
    """Strip a toolchain path down to its last two directories + binary name.

    The generated manifest avoids machine-specific absolute paths; the
    tail (e.g. ``mingw64/bin/g++.exe``) is enough to identify the toolchain.
    """
    parts = Path(exe_value.replace("\\", "/")).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else exe_value


def _parse_sets(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = _SET_LINE_RE.match(line.strip())
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _parse_cache(cache_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _CACHE_TUPLE_RE.match(line)
        if match:
            out[match.group(1)] = match.group(3)
    return out


def _find_compiler_metadata(build_tree: Path) -> tuple[Path, dict[str, str]] | None:
    """Locate CMakeFiles/*/CMakeCXXCompiler.cmake in a build tree."""
    cmake_files = build_tree / "CMakeFiles"
    if not cmake_files.is_dir():
        return None
    for meta in sorted(cmake_files.glob("*/CMakeCXXCompiler.cmake")):
        return meta, _parse_sets(meta.read_text(encoding="utf-8", errors="replace"))
    return None


def _pybind11_version(build_tree: Path) -> str | None:
    header = build_tree / "_deps" / "pybind11-src" / "include" / "pybind11" / "detail" / "common.h"
    if not header.is_file():
        return None
    found = dict(_VERSION_H_RE.findall(header.read_text(encoding="utf-8", errors="replace")))
    if not found:
        return None
    return ".".join(found.get(part, "?") for part in ("MAJOR", "MINOR", "PATCH"))


def _cxx_standard(repo_root: Path) -> str | None:
    text = (repo_root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    match = _CXX_STD_RE.search(text)
    return match.group(1) if match else None


def _cxx_flags_sample(build_tree: Path) -> list[str] | None:
    """Actual compile flags for the pyd target, minus include/output paths.

    Read from the build tree's ``compile_commands.json`` (the entry whose
    source is ``src/bindings.cpp``).  Include-path and output-path flags are
    dropped so the manifest stays free of machine paths.
    """
    cc_path = build_tree / "compile_commands.json"
    if not cc_path.is_file():
        return None
    try:
        entries = json.loads(cc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = next(
        (e for e in entries if Path(e.get("file", "")).name == "bindings.cpp"),
        None,
    )
    if entry is None:
        return None
    tokens = shlex.split(entry.get("command", ""))
    flags: list[str] = []
    skip_next = False
    drop_prefixes = ("-I", "-F")
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in ("-o", "-MF", "-MT", "-MJ", "-isystem", "-I", "-include"):
            skip_next = True
            continue
        if token.startswith(drop_prefixes) or _DRIVE_RE.match(token):
            continue
        if token.endswith(tuple(SOURCE_SUFFIXES)):
            continue
        flags.append(token)
    return flags or None


def _git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    head = run(["rev-parse", "HEAD"])
    if head is None:
        return None, None
    dirty_out = run(["status", "--porcelain", "--untracked-files=no"])
    dirty = None if dirty_out is None else bool(dirty_out)
    return head, dirty


def _artifact_abi(name: str) -> dict[str, str | None]:
    match = _PYD_RE.match(name)
    if not match:
        return {"python_abi_tag": None, "platform_tag": None, "module_name": None}
    return {
        "python_abi_tag": match.group("abi"),
        "platform_tag": match.group("plat"),
        "module_name": match.group("name"),
    }


def _select_build_tree(repo_root: Path, artifact_name: str) -> Path | None:
    """Pick the Ninja build tree under ``build/`` matching the deployed artifact.

    Prefers trees whose generator is Ninja (single-config output layout, same
    as the deployed ``build/cpp/`` staging) and that hold an output pyd with
    the deployed name; ties break on the most recent output mtime.
    """
    build_root = repo_root / "build"
    if not build_root.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for tree in sorted(build_root.glob("cmake-*")):
        cache = tree / "CMakeCache.txt"
        if not cache.is_file():
            continue
        if _parse_cache(cache).get("CMAKE_GENERATOR") != "Ninja":
            continue
        output = tree / "cpp" / artifact_name
        if output.is_file():
            candidates.append((output.stat().st_mtime, tree))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def _source_hashes(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(repo_root.rglob("*")):
        rel = path.relative_to(repo_root)
        parts = rel.parts
        if not parts or parts[0] != "cpp":
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES or not path.is_file():
            continue
        hashes[rel.as_posix()] = _sha256(path)
    for extra in ("CMakeLists.txt", "cpp/CMakeLists.txt"):
        if (repo_root / extra).is_file():
            hashes[extra] = _sha256(repo_root / extra)
    return hashes


def _toolchain_info(repo_root: Path, build_tree: Path | None) -> tuple[dict, str]:
    info: dict = {"derivation": None, "build_tree": None}
    if build_tree is None:
        return info, "artifact-only"
    compiler_meta = _find_compiler_metadata(build_tree)
    cache = (
        _parse_cache(build_tree / "CMakeCache.txt")
        if (build_tree / "CMakeCache.txt").is_file()
        else {}
    )
    if compiler_meta is None and not cache:
        return info, "artifact-only"
    compiler = compiler_meta[1] if compiler_meta else {}
    linker_value = compiler.get("CMAKE_CXX_COMPILER_LINKER", "")
    pkgversion = _PKGVERSION_RE.search(linker_value)
    cmake_version = ".".join(
        cache.get(f"CMAKE_CACHE_{part}_VERSION", "?") for part in ("MAJOR", "MINOR", "PATCH")
    )
    exe = compiler.get("CMAKE_CXX_COMPILER", "")
    debug_raw = cache.get("PYTHON_IS_DEBUG")
    info.update(
        {
            "derivation": "persisted CMake configure metadata of the build tree "
            "(CMakeFiles/CMakeCXXCompiler.cmake + CMakeCache.txt) and the "
            "2026-11-05 pre-push audit build logs (mingw_build.log)",
            "build_tree": build_tree.name,
            "compiler_id": compiler.get("CMAKE_CXX_COMPILER_ID") or None,
            "compiler_version": compiler.get("CMAKE_CXX_COMPILER_VERSION") or None,
            "compiler_distribution": pkgversion.group(1) if pkgversion else None,
            "compiler_exe": _generic_exe(exe) if exe else None,
            "platform": compiler.get("CMAKE_CXX_PLATFORM_ID") or None,
            "architecture": compiler.get("CMAKE_CXX_COMPILER_ARCHITECTURE_ID") or None,
            "generator": cache.get("CMAKE_GENERATOR") or None,
            "cmake_version": cmake_version,
            "build_type": cache.get("CMAKE_BUILD_TYPE") or None,
            "cxx_flags_release": cache.get("CMAKE_CXX_FLAGS_RELEASE") or None,
            "python_is_debug": None
            if not debug_raw
            else debug_raw.strip().upper() in ("1", "ON", "TRUE", "YES"),
        }
    )
    flags_sample = _cxx_flags_sample(build_tree)
    if flags_sample:
        info["cxx_compile_flags_sample"] = flags_sample
    pybind11_version = _pybind11_version(build_tree)
    if pybind11_version:
        info["pybind11_version"] = pybind11_version
    return info, "recorded-from-build-metadata"


def build_manifest(
    repo_root: Path,
    artifact: Path,
    build_tree: Path | None = None,
    generated_utc: str | None = None,
) -> dict:
    """Assemble the provenance manifest dict for ``artifact`` under ``repo_root``."""
    artifact_name = artifact.name
    abi = _artifact_abi(artifact_name)
    stat = artifact.stat()
    head, dirty = _git_state(repo_root)
    toolchain, level = _toolchain_info(repo_root, build_tree)
    deployed_sha = _sha256(artifact)

    build_tree_check: dict = {"available": False}
    if build_tree is not None:
        output = build_tree / "cpp" / artifact_name
        if output.is_file():
            output_sha = _sha256(output)
            build_tree_check = {
                "available": True,
                "output_artifact_sha256": output_sha,
                "byte_identity_with_deployed": output_sha == deployed_sha,
            }

    notes = [
        "The deployed binary was NOT rebuilt by this script; recording it here "
        "does not establish a reproducible source-to-binary binding.",
    ]
    if build_tree_check.get("available"):
        notes.append(
            "Byte identity between the deployed artifact and the surviving "
            "build-tree output was OBSERVED as "
            f"{build_tree_check['byte_identity_with_deployed']}."
        )

    manifest = {
        "schema": SCHEMA,
        "generated_utc": generated_utc or _now_utc(),
        "generator": "scripts/record_native_provenance.py",
        "provenance_level": level,
        "artifact": {
            "path": artifact.relative_to(repo_root).as_posix(),
            "sha256": deployed_sha,
            "size_bytes": stat.st_size,
            "build_timestamp_utc": _utc_iso(stat.st_mtime),
            "timestamp_note": "artifact mtime; best available build-timestamp proxy",
            **abi,
        },
        "source": {
            "git_head": head,
            "git_tracked_dirty": dirty,
            "cxx_standard": _cxx_standard(repo_root),
            "cxx_standard_note": "as set in CMakeLists.txt (CMAKE_CXX_STANDARD)",
            "files": _source_hashes(repo_root),
            "head_note": (
                "HEAD of the recording checkout; the artifact may predate it. "
                "Exact source revision at build time is NOT pinned by this manifest."
            ),
        },
        "toolchain": toolchain,
        "build_tree_check": build_tree_check,
        "notes": notes,
    }
    return manifest


def _resolve_artifact(repo_root: Path, artifact: Path | None) -> Path | None:
    if artifact is not None:
        return artifact
    candidates = sorted((repo_root / "build" / "cpp").glob("*.pyd"))
    return candidates[-1] if candidates else None


def check_manifest(
    repo_root: Path,
    manifest_path: Path,
    artifact: Path | None = None,
) -> tuple[bool, list[str]]:
    """Re-derive the manifest's hashes against the current tree.

    Returns ``(ok, problems)``; ``ok`` is False when the artifact or any
    recorded source file no longer matches the recorded SHA-256.
    """
    problems: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"manifest unreadable: {exc}"]
    if manifest.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {manifest.get('schema')!r}")

    art = manifest.get("artifact", {})
    rel = art.get("path")
    if not rel:
        return False, ["manifest has no artifact.path"]
    art_path = Path(artifact) if artifact is not None else repo_root / rel
    if not art_path.is_file():
        problems.append(f"artifact missing: {rel}")
    elif _sha256(art_path) != art.get("sha256"):
        problems.append("artifact sha256 drift")
    if not problems:
        source_files = manifest.get("source", {}).get("files", {})
        for rel_path, recorded in sorted(source_files.items()):
            src = repo_root / rel_path
            if not src.is_file():
                problems.append(f"source missing: {rel_path}")
            elif _sha256(src) != recorded:
                problems.append(f"source drift: {rel_path}")
        if not problems:
            print("provenance check OK: artifact and source hashes match")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--artifact", type=Path, default=None, help="deployed .pyd to record")
    parser.add_argument(
        "--build-tree",
        type=Path,
        default=None,
        help="CMake build tree whose metadata supplies toolchain identity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest output path (repo-relative unless absolute)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing manifest instead of writing one; exit 0 = ok/absent, 1 = drift",
    )
    args = parser.parse_args(argv)
    repo_root: Path = args.repo.resolve()

    if args.check:
        manifest_path = args.output
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path
        if not manifest_path.is_file():
            print(f"no manifest at {manifest_path.name}; nothing to check")
            return 0
        artifact = _resolve_artifact(repo_root, args.artifact)
        ok, problems = check_manifest(repo_root, manifest_path, artifact)
        for problem in problems:
            print(f"DRIFT: {problem}")
        return 0 if ok else 1

    artifact = _resolve_artifact(repo_root, args.artifact)
    if artifact is None:
        print("no deployed artifact found under build/cpp/ (pass --artifact)", file=sys.stderr)
        return 2
    build_tree = args.build_tree
    if build_tree is None:
        build_tree = _select_build_tree(repo_root, artifact.name)
    elif not build_tree.is_absolute():
        build_tree = repo_root / build_tree

    manifest = build_manifest(repo_root, artifact, build_tree)
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output.relative_to(repo_root).as_posix()} "
        f"(artifact {manifest['artifact']['sha256'][:12]}..., level={manifest['provenance_level']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
