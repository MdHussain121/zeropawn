# Native Binary Provenance (SA5-01)

This document establishes how ZeroPawn records the provenance of its deployed
native extension (`build/cpp/zeropawn_cpp*.pyd`) — that is, how a deployed
binary can be traced back to the source tree and toolchain that produced it.

## Mechanism

- **Generator (committed):** `scripts/record_native_provenance.py`
- **Generated manifest:** `build/native_provenance.json` — deliberately
  **gitignored**. It binds to the local deployed artifact (an untracked binary)
  and goes stale the moment the artifact or a C++ source changes; committing it
  would create a record Git cannot verify. Regenerate it locally as needed.

```bash
python scripts/record_native_provenance.py          # (re)generate the manifest
python scripts/record_native_provenance.py --check  # verify recorded hashes, exit 1 on drift
```

`--check` exits `0` when no manifest exists (skip-if-absent contract) and on a
fully matching tree; it exits `1` if the artifact or any recorded source file
no longer matches its recorded SHA-256. The contract is pinned by
`tests/test_native_provenance.py`, which also runs the generator hermetically
against a synthetic repo/build tree and asserts no machine-specific paths leak
into the manifest.

## What the manifest records

| Field group | Contents |
| --- | --- |
| `artifact` | repo-relative path, SHA-256, size, build timestamp (mtime, UTC — best available proxy), Python ABI/platform tags parsed from the filename |
| `source` | git HEAD of the *recording* checkout, tracked-file dirtiness, `CMAKE_CXX_STANDARD` from the root `CMakeLists.txt`, SHA-256 of every C++ source/header under `cpp/` (including vendored `chess-library`) plus both `CMakeLists.txt` files |
| `toolchain` | compiler ID/version/distribution/exe tail, platform/architecture, generator, CMake version, build type, release flags, actual compile-flag sample (include/output paths stripped), pybind11 version — all derived from the build tree's **persisted CMake configure metadata** (`CMakeFiles/*/CMakeCXXCompiler.cmake`, `CMakeCache.txt`, `compile_commands.json`) |
| `build_tree_check` | SHA-256 of the surviving build-tree output pyd and whether it is byte-identical to the deployed artifact |
| `notes` | explicit truthfulness caveats (see below) |

The manifest never contains absolute machine paths (compiler paths are reduced
to their `mingw64/bin/...`-style tail; include/output paths are stripped from
the flag sample).

## Truthfulness levels

`provenance_level` is computed, never asserted:

- **`recorded-from-build-metadata`** — toolchain identity was re-derived from
  the build tree's persisted CMake metadata and the 2026-11-05 pre-push audit
  build logs. The artifact itself was **not** rebuilt by the generator.
- **`artifact-only`** — no build metadata was found; nothing beyond the binary
  itself can be asserted.

Recording a manifest never establishes a *reproducible* source-to-binary
binding; that would require rebuilding the artifact from a pinned revision in
the same environment and comparing hashes.

## Current recorded provenance (2026-11-05/06, SA5-01 remediation)

- Artifact: `build/cpp/zeropawn_cpp.cp312-win_amd64.pyd`,
  SHA-256 `f1788450d4eca15577fae7e2e1a217577783ae6b3906f17ca9b28f3d4ffb3ae7`,
  1,457,678 bytes, built (mtime) 2026-11-05 23:55 IST.
- Attribution: **recorded, not reproducibly rebuilt.** The binary was built
  from this tree's C++ sources by a prior session using MinGW-w64 (WinLibs
  UCRT) GCC 16.2.0 + Ninja + CMake 4.3.2 + pybind11 v2.13.6, C++17, Release
  (static libgcc/libstdc++ runtime), Python 3.12 ABI. Toolchain identity is
  derived from the persisted CMake configure metadata of the `build/cmake-mingw`
  tree and the `mingw_build.log` of the 2026-11-05 pre-push audit.
- Byte identity: the surviving `build/cmake-mingw` output pyd
  (SHA-256 `9fbd8a9a…`) is **not** byte-identical to the deployed artifact —
  recorded as observed `false`, never papered over.
- Source revision at build time: **not pinned.** The artifact's mtime falls
  between commits `76ca91c` (23:33 IST) and `3742977` (23:59 IST) on
  2026-11-05; the working tree was dirty at recording time. `source.git_head`
  records the HEAD of the *recording* checkout only.

## Future improvement

To upgrade provenance to "reproducibly rebuilt", pin the toolchain (compiler
version, pybind11 release), rebuild from a tagged revision, and record a
manifest whose `build_tree_check.byte_identity_with_deployed` is `true`. A CI
step running `record_native_provenance.py --check` would then enforce drift
detection between published artifacts and source.
