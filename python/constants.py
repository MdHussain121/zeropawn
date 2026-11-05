"""Shared constants for ZeroPawn (mirrors cpp/include/zeropawn/encoder.hpp)."""

NUM_PLANES: int = 19
BOARD_SIZE: int = 8
NUM_SQUARES: int = 64
STATE_SIZE: int = 1216  # NUM_PLANES * NUM_SQUARES
NUM_ACTIONS: int = 4672  # 64 * 73
PLANES_SHAPE: tuple = (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)  # encode_state output shape

FULL_SIMS: int = 800
FAST_SIMS: int = 150
# Probability that a self-play move uses the full-search simulation budget
# (the fast-search budget is used otherwise). 0.25 is the live production
# default (SelfPlayWorker / SelfPlayGame in python/selfplay/worker.py);
# benchmark/measurement scripts deliberately override full_search_prob=0.35
# for their runs. SA1-05: this replaces the dead, divergent FULL_P=0.35
# constant (referenced nowhere; production default was 0.25).
FULL_SEARCH_PROB: float = 0.25

RESIGN_THRESHOLD: float = -0.95

# WDL (win/draw/loss) labels in value-head convention.
WDL_WIN: float = 1.0
WDL_DRAW: float = 0.0
WDL_LOSS: float = -1.0
WDL_LABELS: tuple = ("win", "draw", "loss")
