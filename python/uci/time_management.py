"""Mode-aware UCI time management (audit H-3).

The UCI ``go`` path previously assumed a hardcoded 400 simulations/second when
translating ``go movetime`` / ``go wtime`` budgets into a simulation count.
Measured throughput differs from that assumption by 2-6x depending on device
and search mode (serial B=1: ~120-250 sims/s on CUDA; internal-batched B=16:
~1.5k-3.5k sims/s), so clocked moves were systematically mis-allocated.

This module replaces the constant with an explicit, deterministic estimator:

- Throughput observations (one per completed search) are EWMA'd per
  ``(device, search_mode, batch_size)`` key. A changed configuration gets a
  fresh key with no samples, so stale measurements from another mode can
  never be reused.
- Before any observation exists, a deliberately CONSERVATIVE per-configuration
  default is used (below typical measured rates) so early clocked moves
  under-use rather than over-commit the clock.
- Budgets are converted to simulations with an explicit purpose margin
  (movetime: hard deadline, 10% headroom; real clocks: 20% headroom).

Mid-search interruption is NOT supported by the native MCTS (single C++ call
with no stop callback), so ``stop`` remains a no-op while a search is running
(unchanged UCI contract); this estimator keeps the predicted search time
within the allocated budget instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThroughputKey:
    """Configuration a throughput observation belongs to."""

    device: str  # "cuda" | "cpu"
    search_mode: str  # "serial" | "batched"
    batch_size: int  # effective internal batch size (1 for serial)

    @staticmethod
    def from_engine(device: str, search_mode: str, batch_size: int) -> "ThroughputKey":
        return ThroughputKey(str(device), str(search_mode), max(1, int(batch_size)))


# Conservative cold-start estimates (sims/s), deliberately BELOW typical
# measured rates on this project's reference hardware (RTX 4060 Laptop).
_CONSERVATIVE_DEFAULTS: dict[tuple[str, str], float] = {
    ("cuda", "serial"): 100.0,
    ("cuda", "batched"): 800.0,
    ("cpu", "serial"): 40.0,
    ("cpu", "batched"): 200.0,
}

# Fraction of the estimated capacity actually allocated: the rest covers I/O,
# bestmove emission, and estimate error.
_ALLOCATION_MARGINS: dict[str, float] = {
    "movetime": 0.9,  # hard deadline: leave 10% headroom
    "clock": 0.8,  # soft clock: leave 20% headroom
}

# EWMA smoothing factor for observed throughput.
_EWMA_ALPHA = 0.3

# Observation plausibility guards. The native search floor is ~117k sims/s with
# a trivial evaluator; a search that actually runs neural inference stays in
# the low thousands. Observations outside these bounds are measurement noise
# (sub-millisecond timing) or cache-served searches with no real inference, and
# must not inflate the estimate into an overcommitted clock.
_MIN_ELAPSED_SEC = 0.001
_MAX_PLAUSIBLE_RATE = 20_000.0


def allocate_clock_ms(total_ms: float, inc_ms: float, moves_to_go: Optional[int] = None) -> float:
    """Wall-clock budget (ms) for one move from the remaining clock.

    Keeps the repo's established allocation shape (about 1/30 of the remaining
    time plus the increment, bounded to [100, 10000] ms); when the GUI sends
    ``movestogo``, spend 1/moves-to-go of the remaining clock instead.
    """
    if total_ms <= 0:
        return 0.0
    if moves_to_go is not None and moves_to_go > 0:
        share = total_ms / max(1, min(moves_to_go, 40))
    else:
        share = total_ms / 30.0
    allocated = share + inc_ms
    return max(100.0, min(allocated, 10_000.0))


class SearchTimeManager:
    """Estimates simulation budgets from observed search throughput.

    All budget inputs are explicit and the state is a plain per-key EWMA, so
    the estimator is deterministic and unit-testable.
    """

    def __init__(self) -> None:
        self._ewma: dict[ThroughputKey, float] = {}
        self._samples: dict[ThroughputKey, int] = {}

    def record(self, key: ThroughputKey, sims: int, elapsed_sec: float) -> None:
        """Record one completed search (sims simulated in elapsed_sec seconds).

        Implausible observations are ignored: sub-millisecond timings are
        measurement noise, and rates beyond ``_MAX_PLAUSIBLE_RATE`` indicate a
        search served entirely from the NN cache (no real inference) — either
        would inflate the estimate and overcommit later clocked moves.
        """
        if sims <= 0 or elapsed_sec <= 0:
            return
        if elapsed_sec < _MIN_ELAPSED_SEC:
            return
        rate = sims / elapsed_sec
        if rate > _MAX_PLAUSIBLE_RATE:
            return
        prev = self._ewma.get(key)
        if prev is None:
            self._ewma[key] = rate
        else:
            self._ewma[key] = prev + _EWMA_ALPHA * (rate - prev)
        self._samples[key] = self._samples.get(key, 0) + 1

    def observed_rate(self, key: ThroughputKey) -> Optional[float]:
        return self._ewma.get(key)

    def sample_count(self, key: ThroughputKey) -> int:
        return self._samples.get(key, 0)

    def estimate_rate(self, key: ThroughputKey) -> float:
        """Conservative sims/s estimate for the configuration.

        The observed EWMA is the best estimator of the current hardware and
        configuration; the conservative default only covers the cold start.
        Configurations are keyed separately, so a measurement from another
        mode/device/batch-size is never reused.
        """
        rate = self._ewma.get(key)
        if rate is not None and rate > 0:
            return rate
        return _CONSERVATIVE_DEFAULTS.get((key.device, key.search_mode), 50.0)

    def sims_for_budget(self, key: ThroughputKey, allocated_ms: float, purpose: str) -> int:
        """Simulation count expected to complete within ``allocated_ms``."""
        if allocated_ms <= 0:
            return 0
        rate = self.estimate_rate(key)
        margin = _ALLOCATION_MARGINS.get(purpose, 0.8)
        budget_sec = (allocated_ms / 1000.0) * margin
        return max(1, int(rate * budget_sec))
