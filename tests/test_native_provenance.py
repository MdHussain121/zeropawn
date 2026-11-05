"""SA5-01: native provenance manifest / generator contract.

Pins three properties:

1. The generator (``scripts/record_native_provenance.py``) runs hermetically
   on a synthetic repo+build-tree in ``tmp_path`` and produces a manifest
   whose recorded facts match reality (hashes, ABI tag, toolchain fields)
   and which stays free of machine-specific paths.
2. ``--check``/``check_manifest`` passes on an unchanged tree and reports
   drift (non-zero / problems) when the artifact or a source file changes.
3. If a generated manifest exists in *this* repository
   (``build/native_provenance.json``), it parses, follows the schema, and
   still matches the deployed artifact and source hashes (skip-if-absent).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.record_native_provenance import (
    REPO_ROOT,
    SCHEMA,
    build_manifest,
    check_manifest,
    main,
)

#: Same shape as the repository-wide privacy pin (tests/test_privacy_portable_paths.py).
DRIVE_LETTER_USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users")

ARTIFACT_NAME = "zeropawn_fake.cp312-win_amd64.pyd"
ARTIFACT_BYTES = b"fake pyd payload v1"
ARTIFACT_SHA256 = hashlib.sha256(ARTIFACT_BYTES).hexdigest()


def _make_fake_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Synthetic repo: cpp sources, deployed artifact, and a fake Ninja build tree."""
    repo = tmp_path / "repo"
    (repo / "cpp" / "src").mkdir(parents=True)
    (repo / "cpp" / "include" / "zeropawn").mkdir(parents=True)
    (repo / "cpp" / "vendor" / "chess-library").mkdir(parents=True)

    (repo / "CMakeLists.txt").write_text("set(CMAKE_CXX_STANDARD 17)\n", encoding="utf-8")
    (repo / "cpp" / "CMakeLists.txt").write_text("# cpp module\n", encoding="utf-8")
    (repo / "cpp" / "src" / "bindings.cpp").write_text("// bindings\n", encoding="utf-8")
    (repo / "cpp" / "src" / "chess_game.cpp").write_text("// game\n", encoding="utf-8")
    (repo / "cpp" / "include" / "zeropawn" / "game.hpp").write_text("// header\n", encoding="utf-8")
    (repo / "cpp" / "vendor" / "chess-library" / "chess.hpp").write_text(
        "// vendor\n", encoding="utf-8"
    )

    # Deployed artifact (build/cpp/, as staged for deployment).
    artifact = repo / "build" / "cpp" / ARTIFACT_NAME
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(ARTIFACT_BYTES)

    # Fake Ninja build tree with persisted CMake configure metadata.
    build_tree = repo / "build" / "cmake-fake"
    meta_dir = build_tree / "CMakeFiles" / "x64"
    meta_dir.mkdir(parents=True)
    (meta_dir / "CMakeCXXCompiler.cmake").write_text(
        'set(CMAKE_CXX_COMPILER "C:/toolchain/mingw64/bin/g++.exe")\n'
        'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
        'set(CMAKE_CXX_COMPILER_VERSION "99.9.0")\n'
        'set(CMAKE_CXX_PLATFORM_ID "MinGW")\n'
        'set(CMAKE_CXX_COMPILER_ARCHITECTURE_ID "x86_64")\n'
        "set(CMAKE_CXX_COMPILER_LINKER \"Configured with: --with-pkgversion='FakeW64 r9'\")\n",
        encoding="utf-8",
    )
    (build_tree / "CMakeCache.txt").write_text(
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "CMAKE_CACHE_MAJOR_VERSION:INTERNAL=4\n"
        "CMAKE_CACHE_MINOR_VERSION:INTERNAL=3\n"
        "CMAKE_CACHE_PATCH_VERSION:INTERNAL=2\n"
        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
        "PYTHON_IS_DEBUG:INTERNAL=0\n",
        encoding="utf-8",
    )
    version_header = (
        build_tree / "_deps" / "pybind11-src" / "include" / "pybind11" / "detail" / "common.h"
    )
    version_header.parent.mkdir(parents=True)
    version_header.write_text(
        "#define PYBIND11_VERSION_MAJOR 9\n"
        "#define PYBIND11_VERSION_MINOR 9\n"
        "#define PYBIND11_VERSION_PATCH 9\n",
        encoding="utf-8",
    )
    (build_tree / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "file": "C:/toolchain-repo/cpp/src/bindings.cpp",
                    "command": "C:/toolchain/mingw64/bin/g++.exe -Dzeropawn_fake_EXPORTS "
                    "-I/toolchain-repo/cpp/include "
                    "-isystem C:/toolchain/pybind11/include "
                    "-static-libgcc -O3 -DNDEBUG -std=c++17 -fvisibility=hidden "
                    "-o fake.obj -c C:/toolchain-repo/cpp/src/bindings.cpp",
                }
            ]
        ),
        encoding="utf-8",
    )
    # Build-tree output pyd (same bytes as deployed by default).
    (build_tree / "cpp").mkdir()
    (build_tree / "cpp" / ARTIFACT_NAME).write_bytes(ARTIFACT_BYTES)
    return repo, artifact, build_tree


def test_generator_hermetic_manifest_contract(tmp_path):
    repo, artifact, build_tree = _make_fake_repo(tmp_path)

    manifest = build_manifest(repo, artifact, build_tree)

    assert manifest["schema"] == SCHEMA
    # Truthfulness contract: recording is NOT a reproducible rebuild.
    assert manifest["provenance_level"] == "recorded-from-build-metadata"
    assert any("NOT rebuilt" in note for note in manifest["notes"])

    # Artifact facts.
    assert manifest["artifact"]["path"] == f"build/cpp/{ARTIFACT_NAME}"
    assert manifest["artifact"]["sha256"] == ARTIFACT_SHA256
    assert manifest["artifact"]["python_abi_tag"] == "cp312"
    assert manifest["artifact"]["platform_tag"] == "win_amd64"
    assert manifest["artifact"]["module_name"] == "zeropawn_fake"

    # Source facts.
    assert manifest["source"]["cxx_standard"] == "17"
    assert "cpp/src/bindings.cpp" in manifest["source"]["files"]
    assert "cpp/vendor/chess-library/chess.hpp" in manifest["source"]["files"]
    assert "CMakeLists.txt" in manifest["source"]["files"]

    # Toolchain facts (all derived from the fake persisted metadata).
    toolchain = manifest["toolchain"]
    assert toolchain["compiler_id"] == "GNU"
    assert toolchain["compiler_version"] == "99.9.0"
    assert toolchain["compiler_distribution"] == "FakeW64 r9"
    assert toolchain["compiler_exe"] == "mingw64/bin/g++.exe"
    assert toolchain["generator"] == "Ninja"
    assert toolchain["cmake_version"] == "4.3.2"
    assert toolchain["build_type"] == "Release"
    assert toolchain["python_is_debug"] is False
    assert toolchain["pybind11_version"] == "9.9.9"
    # Flags sample keeps real flags but strips include/output paths.
    assert "-std=c++17" in toolchain["cxx_compile_flags_sample"]
    assert "-static-libgcc" in toolchain["cxx_compile_flags_sample"]
    assert not any(
        tok.startswith(("-I", "-isystem")) for tok in toolchain["cxx_compile_flags_sample"]
    )

    # Byte identity with the surviving build-tree output is OBSERVED.
    assert manifest["build_tree_check"]["byte_identity_with_deployed"] is True

    # Privacy contract: no machine-specific user-profile paths anywhere in
    # the manifest (resolved paths on the author's machine are not embedded —
    # only portable placeholders and repo-relative paths are recorded).
    text = json.dumps(manifest)
    assert not DRIVE_LETTER_USER_PATH.search(text)


def test_generator_reports_byte_identity_mismatch(tmp_path):
    repo, artifact, build_tree = _make_fake_repo(tmp_path)
    # The real deployed pyd vs the surviving build-tree output differ; the
    # manifest must record that as observed false, never paper over it.
    (build_tree / "cpp" / ARTIFACT_NAME).write_bytes(b"relinked variant, different bytes")

    manifest = build_manifest(repo, artifact, build_tree)
    assert manifest["build_tree_check"]["available"] is True
    assert manifest["build_tree_check"]["byte_identity_with_deployed"] is False
    assert manifest["build_tree_check"]["output_artifact_sha256"] != ARTIFACT_SHA256


def test_check_passes_then_detects_drift(tmp_path):
    repo, _artifact, _build_tree = _make_fake_repo(tmp_path)
    output = repo / "build" / "native_provenance.json"

    assert main(["--repo", str(repo), "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["artifact"]["sha256"] == ARTIFACT_SHA256

    ok, problems = check_manifest(repo, output)
    assert ok, problems
    assert problems == []

    # Artifact drift.
    artifact_path = repo / "build" / "cpp" / ARTIFACT_NAME
    artifact_path.write_bytes(b"tampered")
    ok, problems = check_manifest(repo, output)
    assert not ok
    assert any("artifact sha256 drift" in p for p in problems)

    # Source drift (artifact restored).
    artifact_path.write_bytes(ARTIFACT_BYTES)
    (repo / "cpp" / "src" / "chess_game.cpp").write_text("// game v2\n", encoding="utf-8")
    ok, problems = check_manifest(repo, output)
    assert not ok
    assert any("source drift: cpp/src/chess_game.cpp" in p for p in problems)


def test_check_is_skip_when_manifest_absent(tmp_path):
    repo, _artifact, _build_tree = _make_fake_repo(tmp_path)
    # No manifest written yet: --check is a no-op skip, not an error.
    assert main(["--repo", str(repo), "--check"]) == 0


def test_repo_manifest_if_present_is_consistent():
    """If the generated manifest exists in this repo it must still hold true."""
    manifest_path = REPO_ROOT / "build" / "native_provenance.json"
    if not manifest_path.is_file():
        pytest.skip("no generated provenance manifest in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == SCHEMA
    assert manifest["provenance_level"] in {"recorded-from-build-metadata", "artifact-only"}
    # The recorded binary is NOT claimed as reproducibly rebuilt.
    assert any("NOT rebuilt" in note for note in manifest["notes"])
    ok, problems = check_manifest(REPO_ROOT, manifest_path)
    assert ok, problems
