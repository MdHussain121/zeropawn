"""pytest conftest — add C++ build directories to sys.path and DLL
search path so zeropawn_cpp can be imported without manual installation."""

import os
import sys
from pathlib import Path
import pytest

_repo_root = Path(__file__).resolve().parent

# Purge foreign editable install finders from sys.meta_path so worktree imports local modules
sys.meta_path = [f for f in sys.meta_path if "Editable" not in getattr(f, "__name__", "")]
if str(_repo_root) in sys.path:
    sys.path.remove(str(_repo_root))
sys.path.insert(0, str(_repo_root))

# Check multiple standard build directories
_build_candidates = [
    _repo_root / "build" / "cpp",
    _repo_root / "build_zeropawn" / "cpp",
    _repo_root / "build_zeropawn",
    _repo_root / "build",
    _repo_root / "build" / "Release",
    _repo_root / "build" / "Debug",
    _repo_root / "build" / "branch" / "cpp",
    _repo_root / "build" / "cmake" / "cpp",
]
for _candidate in reversed(_build_candidates):
    if _candidate.is_dir():
        if str(_candidate) in sys.path:
            sys.path.remove(str(_candidate))
        sys.path.insert(0, str(_candidate))

# On Windows with MinGW, add the compiler bin dir so libwinpthread-1.dll can be found
if sys.platform == "win32":
    _local_app = os.environ.get("LOCALAPPDATA", "")
    _compiler_dirs = [
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
    for _cdir in _compiler_dirs:
        if _cdir and _cdir.is_dir():
            try:
                os.add_dll_directory(str(_cdir))
            except (AttributeError, OSError):
                pass
            if str(_cdir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{_cdir};{os.environ.get('PATH', '')}"
            break
# Sibling repository build directories (for worktrees)
_main_candidates = [
    _repo_root.parent / "chess-rl" / "build" / "cpp",
    _repo_root.parent / "chess-rl" / "build_zeropawn" / "cpp",
]
for _candidate in reversed(_main_candidates):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.append(str(_candidate))


# Lazy fixture for zeropawn_cpp extension
@pytest.fixture
def zeropawn_cpp():
    """Lazily import and return the zeropawn_cpp extension, skipping the test if unavailable."""
    return pytest.importorskip("zeropawn_cpp")


@pytest.fixture
def repo_root():
    """Absolute path to the repository root."""
    return str(_repo_root)
