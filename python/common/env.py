"""Environment variable management for ZeroPawn.

Supports primary ZEROPAWN_* variables with fallback to legacy CHESSRL_*
and emits a one-time DeprecationWarning on legacy access.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Set

_WARNED_VARS: Set[str] = set()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve environment variable, checking ZEROPAWN_* first, then CHESSRL_*.

    Args:
        key: Variable name or suffix (e.g. 'OUTPUT_MODE', 'ZEROPAWN_OUTPUT_MODE', 'CHESSRL_OUTPUT_MODE')
        default: Fallback value if neither variable is set

    Returns:
        The environment variable value or default.
    """
    suffix = key
    if suffix.startswith("ZEROPAWN_"):
        suffix = suffix[len("ZEROPAWN_") :]
    elif suffix.startswith("CHESSRL_"):
        suffix = suffix[len("CHESSRL_") :]

    primary = f"ZEROPAWN_{suffix}"
    fallback = f"CHESSRL_{suffix}"

    if primary in os.environ:
        return os.environ[primary]

    if fallback in os.environ:
        if fallback not in _WARNED_VARS:
            _WARNED_VARS.add(fallback)
            warnings.warn(
                f"Environment variable '{fallback}' is deprecated; please use '{primary}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return os.environ[fallback]

    return default


def set_env(key: str, value: str) -> None:
    """Set both ZEROPAWN_* and legacy CHESSRL_* variables for full backward compatibility."""
    suffix = key
    if suffix.startswith("ZEROPAWN_"):
        suffix = suffix[len("ZEROPAWN_") :]
    elif suffix.startswith("CHESSRL_"):
        suffix = suffix[len("CHESSRL_") :]

    os.environ[f"ZEROPAWN_{suffix}"] = value
    os.environ[f"CHESSRL_{suffix}"] = value
