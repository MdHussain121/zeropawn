"""Root compatibility shim for chessrl_cpp -> python.chessrl_cpp."""

from python.chessrl_cpp import *  # noqa: F401,F403
import python.chessrl_cpp as _shim


def __getattr__(name: str):
    return getattr(_shim, name)
