"""ZeroPawn Python package.
Ensures C++ extension (zeropawn_cpp) and required DLLs are loaded.
"""

import os
import sys
from pathlib import Path

# S6-01: in an installed wheel the native extension is staged as package data
# under python/_native/ (scripts/build_wheel.py copies it from build/cpp/ at
# build time; the directory does not exist in a repo checkout, so this is a
# no-op for developers). Make it importable as zeropawn_cpp and register the
# directory for Windows dependent-DLL resolution (libwinpthread-1.dll).
# ORDERING (2026-11-03): this block runs BEFORE the build-dir loop below so
# the build dirs are inserted ABOVE it on sys.path. The wheel staging copy
# must never outrank a freshly built source-tree pyd inside a repo checkout —
# the staging snapshot is only refreshed at wheel-build time, so giving it
# priority would silently shadow newer builds (the S3-04 stale-binary hazard
# class). In an installed wheel no build/ dirs exist and _native remains the
# sole provider.
_native_candidate = Path(__file__).resolve().parent / "_native"
if _native_candidate.is_dir():
    if str(_native_candidate) not in sys.path:
        sys.path.insert(0, str(_native_candidate))
    if sys.platform == "win32":
        try:
            os.add_dll_directory(str(_native_candidate))
        except OSError:
            pass

# Add potential CMake build directories to sys.path
_repo_root = Path(__file__).resolve().parent.parent
_build_candidates = [
    _repo_root / "build_zeropawn_dev" / "cpp",
    _repo_root / "build" / "cpp",
    _repo_root / "build_zeropawn" / "cpp",
    _repo_root / "build_zeropawn",
    _repo_root / "build",
    _repo_root / "build" / "Release",
    _repo_root / "build" / "Debug",
    _repo_root / "build" / "branch" / "cpp",
    _repo_root / "build" / "cmake" / "cpp",
]
for _b in reversed(_build_candidates):
    if _b.is_dir():
        if str(_b) in sys.path:
            sys.path.remove(str(_b))
        sys.path.insert(0, str(_b))

# On Windows with MinGW, ensure libwinpthread-1.dll and runtime can be found
if sys.platform == "win32":
    _local_app = os.environ.get("LOCALAPPDATA", "")
    _compiler_candidates = [
        (
            Path(_local_app)
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "mingw64"
            / "bin"
        )
        if _local_app
        else None,
        Path("C:/mingw64/bin"),
        Path("C:/msys64/mingw64/bin"),
    ]
    for _candidate in _compiler_candidates:
        if _candidate and _candidate.is_dir():
            try:
                os.add_dll_directory(str(_candidate))
            except Exception:
                pass
            _curr_path = os.environ.get("PATH", "")
            if not _curr_path.startswith(f"{_candidate};"):
                os.environ["PATH"] = f"{_candidate};{_curr_path}"
            break
