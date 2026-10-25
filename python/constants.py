"""Shared constants for ZeroPawn (mirrors cpp/include/zeropawn/encoder.hpp)."""

NUM_PLANES: int = 19
BOARD_SIZE: int = 8
NUM_SQUARES: int = 64
STATE_SIZE: int = 1216  # NUM_PLANES * NUM_SQUARES
NUM_ACTIONS: int = 4672  # 64 * 73

FULL_SIMS: int = 800
FAST_SIMS: int = 150
FULL_P: float = 0.35

RESIGN_THRESHOLD: float = -0.95

# WDL (win/draw/loss) labels in value-head convention.
WDL_WIN: float = 1.0
WDL_DRAW: float = 0.0
WDL_LOSS: float = -1.0
WDL_LABELS: tuple = ("win", "draw", "loss")
