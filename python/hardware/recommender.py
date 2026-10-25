"""Hardware-aware pre-training configuration recommender for ZeroPawn."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from python.hardware.calibration import CalibrationReport
from python.hardware.detector import HardwareInfo


@dataclasses.dataclass
class RecommendedConfig:
    """Actionable hardware-aware execution recommendations."""

    train_device: str
    infer_device: str
    search_device: str
    concurrency: int
    mcts_workers: int
    torch_num_threads: int
    inference_max_batch_size: int
    inference_max_wait_ms: float
    watchdog_rss_gb: float
    watchdog_free_ram_gb: float
    use_separate_cuda_streams: bool
    rationale: Dict[str, str] = dataclasses.field(default_factory=dict)
    confidence: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary_table(self) -> List[Tuple[str, str, str]]:
        """Return list of (parameter, recommended_value, rationale)."""
        return [
            ("Training Device", self.train_device, self.rationale.get("train_device", "")),
            ("Inference Device", self.infer_device, self.rationale.get("infer_device", "")),
            ("Search/Board Device", self.search_device, self.rationale.get("search_device", "")),
            (
                "Concurrent Self-Play Games",
                str(self.concurrency),
                self.rationale.get("concurrency", ""),
            ),
            (
                "PyTorch CPU Threads",
                str(self.torch_num_threads),
                self.rationale.get("torch_num_threads", ""),
            ),
            (
                "Inference Max Batch Size",
                str(self.inference_max_batch_size),
                self.rationale.get("inference_max_batch_size", ""),
            ),
            (
                "Inference Max Wait",
                f"{self.inference_max_wait_ms:.1f} ms",
                self.rationale.get("inference_max_wait_ms", ""),
            ),
            (
                "Watchdog RSS Ceiling",
                f"{self.watchdog_rss_gb:.1f} GB",
                self.rationale.get("watchdog_rss_gb", ""),
            ),
            (
                "Free System RAM Floor",
                f"{self.watchdog_free_ram_gb:.1f} GB",
                self.rationale.get("watchdog_free_ram_gb", ""),
            ),
            (
                "CUDA Stream Isolation",
                "Enabled" if self.use_separate_cuda_streams else "Disabled",
                self.rationale.get("streams", ""),
            ),
        ]


def recommend_configuration(
    hw: HardwareInfo,
    calib: Optional[CalibrationReport] = None,
    workload: str = "rl",
    force_device: Optional[str] = None,
) -> RecommendedConfig:
    """Formulate safe, device-specific execution recommendations.

    Respects:
    - 8 GB VRAM envelope (prevents OOM on consumer laptop GPUs)
    - 16 GB system RAM ceiling (preserves 2.0 GB free floor)
    - Intel P-core / E-core hybrid topology (avoids CPU thread thrashing)
    - Complete CPU-only fallback when CUDA is absent or requested
    """
    rationale: Dict[str, str] = {}

    # 1. Device Selection
    if force_device:
        fd = force_device.lower().strip()
        if fd in ("cuda", "gpu"):
            train_dev = "cuda" if hw.cuda_available else "cpu"
            infer_dev = "cuda" if hw.cuda_available else "cpu"
            rationale["train_device"] = "Explicit user request (--device cuda)"
            rationale["infer_device"] = "Explicit user request (--device cuda)"
        elif fd == "cpu":
            train_dev = "cpu"
            infer_dev = "cpu"
            rationale["train_device"] = "Explicit user request (--device cpu)"
            rationale["infer_device"] = "Explicit user request (--device cpu)"
        else:
            train_dev = "cuda" if hw.cuda_available else "cpu"
            infer_dev = "cuda" if hw.cuda_available else "cpu"
            rationale["train_device"] = "Auto-selected based on hardware capability"
            rationale["infer_device"] = "Auto-selected based on hardware capability"
    elif hw.cuda_available and hw.gpu_count > 0:
        train_dev = "cuda"
        infer_dev = "cuda"
        rationale["train_device"] = f"Accelerated via {hw.gpu_name}"
        rationale["infer_device"] = f"Accelerated via {hw.gpu_name}"
    else:
        train_dev = "cpu"
        infer_dev = "cpu"
        rationale["train_device"] = "CPU-only execution (CUDA unavailable)"
        rationale["infer_device"] = "CPU-only execution (CUDA unavailable)"

    search_dev = "cpu"
    rationale["search_device"] = "High-speed bitboard & MCTS tree search in native C++ on CPU"

    # 2. Worker Concurrency & CPU Threads
    # On hybrid Intel systems (e.g. 6 P-cores + 4 E-cores = 16 threads):
    # Concurrency 24 with batching achieves maximum throughput while preserving RAM.
    # On CPU-only systems: concurrency should be smaller to prevent CPU starvation.
    if train_dev == "cuda":
        if hw.cpu_is_hybrid:
            concurrency = 24
            rationale["concurrency"] = (
                f"Optimized for Intel Hybrid {hw.cpu_p_cores}P+{hw.cpu_e_cores}E topology (verified +186% throughput)"
            )
        elif hw.cpu_logical_threads >= 16:
            concurrency = 24
            rationale["concurrency"] = f"Scaled for {hw.cpu_logical_threads} logical processors"
        elif hw.cpu_logical_threads >= 8:
            concurrency = 16
            rationale["concurrency"] = f"Scaled for {hw.cpu_logical_threads} logical processors"
        else:
            concurrency = max(4, hw.cpu_logical_threads)
            rationale["concurrency"] = "Bounded for low core-count system"

        torch_threads = max(2, min(4, hw.cpu_physical_cores // 2))
        rationale["torch_num_threads"] = "Conservative intra-op threading to prevent GIL contention"
    else:
        # CPU-only workload
        concurrency = max(2, min(8, hw.cpu_physical_cores - 1))
        rationale["concurrency"] = (
            f"CPU-only mode: bounded to {concurrency} workers to prevent compute starvation"
        )
        torch_threads = max(1, min(4, hw.cpu_physical_cores))
        rationale["torch_num_threads"] = "Dedicated CPU inference threads"

    # 3. Batch Size and Aggregation Timeout
    if infer_dev == "cuda":
        # 8GB VRAM: optimal batch 128-256 leaves ample headroom
        if hw.gpu_total_vram_mb >= 11000:
            batch_size = 256
            max_wait_ms = 1.5
            rationale["inference_max_batch_size"] = "Large VRAM headroom (>= 12GB)"
        elif hw.gpu_total_vram_mb >= 7000:
            batch_size = 128
            max_wait_ms = 1.5
            rationale["inference_max_batch_size"] = (
                "Safe 8GB GDDR6 laptop VRAM profile (<1.5 GB NN allocation)"
            )
        else:
            batch_size = 64
            max_wait_ms = 1.0
            rationale["inference_max_batch_size"] = "Constrained VRAM (<6GB)"
        rationale["inference_max_wait_ms"] = (
            "Full arrival-window aggregation (averaging 15-17 items/batch)"
        )
    else:
        batch_size = 16
        max_wait_ms = 5.0
        rationale["inference_max_batch_size"] = "CPU inference batch bounded for low latency"
        rationale["inference_max_wait_ms"] = "Slightly longer window for CPU batch formation"

    # 4. Memory Watchdog Ceilings
    # Respect 16 GB system RAM:
    total_ram = hw.system_total_ram_gb
    if total_ram >= 28.0:
        watchdog_rss = 14.0
        free_ram_floor = 4.0
    elif total_ram >= 14.0:
        watchdog_rss = 7.5
        free_ram_floor = 2.0
        rationale["watchdog_rss_gb"] = (
            "Strict 7.5 GB RSS ceiling preserving system responsiveness on 16GB RAM"
        )
        rationale["watchdog_free_ram_gb"] = "Mandatory 2.0 GB free physical RAM safety floor"
    else:
        watchdog_rss = max(2.0, total_ram * 0.5)
        free_ram_floor = 1.0
        rationale["watchdog_rss_gb"] = f"Scaled for {total_ram:.1f} GB RAM machine"
        rationale["watchdog_free_ram_gb"] = "1.0 GB free floor"

    # 5. CUDA Streams
    use_streams = train_dev == "cuda" and infer_dev == "cuda"
    if use_streams:
        rationale["streams"] = (
            "High-priority stream for inference, dedicated stream for training backprop"
        )
    else:
        rationale["streams"] = "Not applicable (CPU or mixed-device)"

    return RecommendedConfig(
        train_device=train_dev,
        infer_device=infer_dev,
        search_device=search_dev,
        concurrency=concurrency,
        mcts_workers=concurrency,
        torch_num_threads=torch_threads,
        inference_max_batch_size=batch_size,
        inference_max_wait_ms=max_wait_ms,
        watchdog_rss_gb=watchdog_rss,
        watchdog_free_ram_gb=free_ram_floor,
        use_separate_cuda_streams=use_streams,
        rationale=rationale,
        confidence="high" if (calib and not calib.skipped) else "medium",
    )
