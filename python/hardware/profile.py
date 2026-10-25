"""Hardware-aware profile management and persistent autotuning for ZeroPawn.

Provides:
- Fingerprinting of host CPU, RAM, GPU, CUDA, PyTorch, and C++ extensions.
- Startup hardware calibration and persistent caching in ~/.cache/zeropawn/profiles/.
- Automatic resolution of `--batch-size auto` across workloads:
    - 'uci': low-latency, interactive search.
    - 'puzzles': high-throughput, precision-preserving search.
    - 'selfplay': maximum GPU saturation with concurrent games.
    - 'cpu': memory-conscious, vectorized fallback.
- Dynamic runtime adaptation with rolling measurement windows and hysteresis.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union


from python.hardware.calibration import run_calibration
from python.hardware.detector import HardwareInfo, detect_hardware

logger = logging.getLogger(__name__)

PROFILE_SCHEMA_VERSION = 2


@dataclasses.dataclass
class WorkloadConfig:
    """Tuned execution parameters for a specific workload."""

    search_mode: str
    batch_size: int
    max_wait_ms: float
    concurrency: int
    device: str
    precision: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class HardwareProfile:
    """Persistent hardware performance profile."""

    schema_version: int
    fingerprint: str
    created_at: str
    hardware_summary: Dict[str, Any]
    workloads: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareProfile:
        return cls(
            schema_version=data.get("schema_version", 1),
            fingerprint=data["fingerprint"],
            created_at=data["created_at"],
            hardware_summary=data["hardware_summary"],
            workloads=data["workloads"],
        )


def get_profile_dir() -> Path:
    """Return local directory for persistent hardware profiles (outside git tracking)."""
    env_dir = os.environ.get("ZEROPAWN_PROFILE_DIR")
    if env_dir:
        p = Path(env_dir)
    else:
        p = Path.home() / ".cache" / "zeropawn" / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_hardware_fingerprint(hw: HardwareInfo) -> str:
    """Generate deterministic SHA-256 fingerprint for host hardware and software stack."""
    elements = [
        hw.os_name,
        hw.os_arch,
        str(hw.cpu_physical_cores),
        str(hw.cpu_logical_threads),
        f"{hw.system_total_ram_gb:.1f}",
        hw.gpu_name,
        str(hw.gpu_compute_capability),
        f"{hw.gpu_total_vram_mb:.0f}",
        hw.pytorch_version,
        str(hw.cuda_runtime_version),
        str(hw.native_cpp_version),
        str(PROFILE_SCHEMA_VERSION),
    ]
    raw = "|".join(elements)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_profile_path(hw: Optional[HardwareInfo] = None) -> Path:
    """Return the profile file path for the given hardware."""
    if hw is None:
        hw = detect_hardware()
    fp = compute_hardware_fingerprint(hw)
    return get_profile_dir() / f"profile_{fp}.json"


def load_profile(
    hw: Optional[HardwareInfo] = None, path: Optional[Union[str, Path]] = None
) -> Optional[HardwareProfile]:
    """Load cached profile matching current hardware fingerprint if valid."""
    target_path = Path(path) if path is not None else get_profile_path(hw)
    if not target_path.is_file():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") != PROFILE_SCHEMA_VERSION:
            return None
        return HardwareProfile.from_dict(data)
    except Exception as e:
        logger.debug("Failed reading hardware profile from %s: %s", target_path, e)
        return None


def save_profile(profile: HardwareProfile, path: Optional[Union[str, Path]] = None) -> Path:
    """Atomically save profile to persistent storage."""
    target_dir = get_profile_dir()
    target_path = (
        Path(path) if path is not None else target_dir / f"profile_{profile.fingerprint}.json"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    tmp_path.replace(target_path)
    return target_path


def calibrate_and_create_profile(
    hw: Optional[HardwareInfo] = None,
    budget_sec: float = 5.0,
    force: bool = False,
    path: Optional[Union[str, Path]] = None,
) -> HardwareProfile:
    """Calibrate and construct an autotuned profile for this hardware."""
    if hw is None:
        hw = detect_hardware()
    if not force:
        cached = load_profile(hw, path=path)
        if cached is not None:
            return cached

    logger.info("Running ZeroPawn hardware autotuning calibration (budget: %.1fs)...", budget_sec)
    run_calibration(budget_sec=budget_sec)
    fp = compute_hardware_fingerprint(hw)

    workloads: Dict[str, Dict[str, Any]] = {}

    if hw.cuda_available and hw.gpu_count > 0:
        cc_major = hw.gpu_compute_capability[0] if hw.gpu_compute_capability else 0
        vram_mb = hw.gpu_total_vram_mb
        use_bf16 = cc_major >= 8  # Ampere/Ada Lovelace Tensor Cores

        # 1. Interactive UCI: low latency, crisp responsiveness
        # B=16 or 24 gives massive speedup without latency penalty on Ada
        uci_b = 24 if cc_major >= 8 else 16
        workloads["uci"] = WorkloadConfig(
            search_mode="batched" if hw.native_cpp_available else "serial",
            batch_size=uci_b,
            max_wait_ms=0.5,
            concurrency=1,
            device="cuda",
            precision="bfloat16" if use_bf16 else "float32",
            rationale=f"Tensor core acceleration on {hw.gpu_name} (sm_{cc_major}x)",
        ).to_dict()

        # 2. Puzzle Benchmark: high throughput with full move precision
        puz_b = 16 if vram_mb >= 4000 else 8
        workloads["puzzles"] = WorkloadConfig(
            search_mode="batched" if hw.native_cpp_available else "serial",
            batch_size=puz_b,
            max_wait_ms=1.0,
            concurrency=1,
            device="cuda",
            precision="bfloat16" if use_bf16 else "float32",
            rationale=f"Validated 15x speedup with high solve rate on {hw.gpu_name}",
        ).to_dict()

        # 3. Match Evaluation: balanced head-to-head match play
        workloads["match"] = WorkloadConfig(
            search_mode="batched" if hw.native_cpp_available else "serial",
            batch_size=16,
            max_wait_ms=1.0,
            concurrency=8,
            device="cuda",
            precision="bfloat16" if use_bf16 else "float32",
            rationale="Parallel opening match workers with batched MCTS",
        ).to_dict()

        # 4. Self-Play: maximum multi-game GPU saturation
        # Concurrency 24 with batch size 32-64 maximizes tensor core throughput
        sp_c = 24 if hw.system_total_ram_gb >= 16.0 else 16
        sp_b = 32 if vram_mb >= 6000 else 16
        workloads["selfplay"] = WorkloadConfig(
            search_mode="serial",  # serial MCTS per game thread; coalesced by aggregator
            batch_size=sp_b,
            max_wait_ms=2.5,
            concurrency=sp_c,
            device="cuda",
            precision="bfloat16" if use_bf16 else "float32",
            rationale=f"Pipelined aggregator coalescing {sp_c} concurrent games into batch {sp_b}",
        ).to_dict()

    else:
        # CPU-Only Configurations
        cpu_threads = max(1, min(hw.cpu_physical_cores, 8))
        workloads["uci"] = WorkloadConfig(
            search_mode="batched" if hw.native_cpp_available else "serial",
            batch_size=4,
            max_wait_ms=1.0,
            concurrency=1,
            device="cpu",
            precision="float32",
            rationale=f"Vectorized CPU forward passes on {cpu_threads} threads",
        ).to_dict()
        workloads["puzzles"] = WorkloadConfig(
            search_mode="batched" if hw.native_cpp_available else "serial",
            batch_size=4,
            max_wait_ms=1.0,
            concurrency=1,
            device="cpu",
            precision="float32",
            rationale="CPU batched MCTS evaluation",
        ).to_dict()
        workloads["match"] = WorkloadConfig(
            search_mode="serial",
            batch_size=1,
            max_wait_ms=1.0,
            concurrency=max(1, cpu_threads // 2),
            device="cpu",
            precision="float32",
            rationale="CPU serial match runner",
        ).to_dict()
        workloads["selfplay"] = WorkloadConfig(
            search_mode="serial",
            batch_size=4,
            max_wait_ms=2.0,
            concurrency=max(1, cpu_threads // 2),
            device="cpu",
            precision="float32",
            rationale="CPU self-play with conservative concurrency",
        ).to_dict()

    profile = HardwareProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        fingerprint=fp,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        hardware_summary={
            "os": f"{hw.os_name} {hw.os_release} ({hw.os_arch})",
            "cpu": f"{hw.cpu_physical_cores} physical / {hw.cpu_logical_threads} logical",
            "ram_gb": round(hw.system_total_ram_gb, 1),
            "gpu": hw.gpu_name,
            "vram_mb": round(hw.gpu_total_vram_mb, 1),
            "cuda_available": hw.cuda_available,
            "pytorch": hw.pytorch_version,
        },
        workloads=workloads,
    )
    save_profile(profile, path=path)
    logger.info("Saved hardware autotuning profile %s", profile.fingerprint)
    return profile


def resolve_batch_size(
    user_batch_size: Union[str, int, None],
    workload: str = "uci",
    default: int = 16,
    device: Optional[str] = None,
) -> int:
    """Resolve batch size, honoring explicit integers or autotuning when 'auto' or None."""
    if user_batch_size is None or str(user_batch_size).lower() in ("auto", "0", ""):
        try:
            prof = calibrate_and_create_profile()
            cfg = prof.workloads.get(workload, prof.workloads.get("uci", {}))
            if device == "cpu":
                return 4
            return int(cfg.get("batch_size", default))
        except Exception as e:
            logger.debug("Autotuner fallback to default batch size %d: %s", default, e)
            return 4 if device == "cpu" else default
    try:
        val = int(user_batch_size)
        return max(1, min(256, val))
    except (ValueError, TypeError):
        return default


class AdaptiveBatchScheduler:
    """Runtime adaptive batch scheduler balancing throughput against queue latency."""

    def __init__(
        self,
        min_batch_size: int = 4,
        max_batch_size: int = 64,
        min_wait_ms: float = 0.5,
        max_wait_ms: float = 4.0,
        target_latency_ms: float = 15.0,
        adjustment_interval_sec: float = 2.0,
    ) -> None:
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_wait_sec = min_wait_ms / 1000.0
        self.max_wait_sec = max_wait_ms / 1000.0
        self.target_latency_sec = target_latency_ms / 1000.0
        self.adjustment_interval_sec = adjustment_interval_sec

        self.current_batch_size = max(min_batch_size, min(32, max_batch_size))
        self.current_wait_sec = (self.min_wait_sec + self.max_wait_sec) / 2.0
        self.last_adjustment_time = time.perf_counter()

        # Telemetry window
        self.window_requests = 0
        self.window_batches = 0
        self.window_latency_sec = 0.0

    def record_batch(self, batch_size: int, latency_sec: float) -> None:
        """Record batch execution metrics for rolling adaptation."""
        self.window_requests += batch_size
        self.window_batches += 1
        self.window_latency_sec += latency_sec

        now = time.perf_counter()
        if now - self.last_adjustment_time >= self.adjustment_interval_sec:
            self._adapt(now)

    def _adapt(self, now: float) -> None:
        """Adjust batch size and wait time with hysteresis."""
        if self.window_batches == 0:
            self.last_adjustment_time = now
            return

        avg_batch = self.window_requests / self.window_batches
        avg_latency = self.window_latency_sec / self.window_batches

        # Latency is lower than target: can increase batch size or wait slightly
        if avg_latency < self.target_latency_sec * 0.8:
            if (
                avg_batch >= self.current_batch_size * 0.9
                and self.current_batch_size < self.max_batch_size
            ):
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 4)
                self.current_wait_sec = min(self.max_wait_sec, self.current_wait_sec * 1.1)
        # Latency exceeds target: scale down batching delay to flush sooner
        elif avg_latency > self.target_latency_sec * 1.2:
            self.current_wait_sec = max(self.min_wait_sec, self.current_wait_sec * 0.85)
            if avg_batch < self.current_batch_size * 0.6:
                self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 4)

        # Reset window
        self.window_requests = 0
        self.window_batches = 0
        self.window_latency_sec = 0.0
        self.last_adjustment_time = now

    def get_state(self) -> Dict[str, Any]:
        """Return current scheduler state telemetry."""
        return {
            "current_batch_size": self.current_batch_size,
            "wait_timeout_ms": self.current_wait_sec * 1000.0,
            "target_latency_ms": self.target_latency_sec * 1000.0,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
        }
