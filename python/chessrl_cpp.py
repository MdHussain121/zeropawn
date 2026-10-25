"""Compatibility wrapper for legacy chessrl_cpp imports.

Emits a one-time DeprecationWarning and transparently delegates all attribute
lookups to zeropawn_cpp.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "'chessrl_cpp' is deprecated and will be removed in a future release. "
    "Please import 'zeropawn_cpp' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import zeropawn_cpp (or load fallback during transition)
try:
    import zeropawn_cpp as _impl
except ImportError:
    import importlib.util
    from pathlib import Path

    _repo = Path(__file__).resolve().parent.parent
    _search_dirs = [
        _repo / "build_zeropawn" / "cpp",
        _repo / "build" / "cpp",
        _repo / "python",
    ]
    _candidates = []
    for _d in _search_dirs:
        if _d.is_dir():
            _candidates.extend(sorted(_d.glob("*zeropawn_cpp*.pyd")))
            _candidates.extend(sorted(_d.glob("*chessrl_cpp*.pyd")))
    _impl = None
    for cand in _candidates:
        if cand.is_file():
            mod_name = "zeropawn_cpp" if "zeropawn" in cand.name else "chessrl_cpp"
            spec = importlib.util.spec_from_file_location(mod_name, str(cand))
            _impl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_impl)
            break
    if _impl is None:
        raise ImportError(
            "Neither zeropawn_cpp nor legacy chessrl_cpp binary extension could be loaded."
        )

for _k, _v in _impl.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v


# Delegate attribute access
def __getattr__(name: str):
    return getattr(_impl, name)
