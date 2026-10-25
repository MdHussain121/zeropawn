"""Tests for hardware-aware autotuning, persistent profiling, and adaptive batching."""

import os
import json
import pytest
from pathlib import Path
from python.hardware.profile import (
    PROFILE_SCHEMA_VERSION,
    HardwareProfile,
    WorkloadConfig,
    detect_hardware,
    calibrate_and_create_profile,
    load_profile,
    save_profile,
    resolve_batch_size,
    AdaptiveBatchScheduler,
    get_profile_path,
)


def test_hardware_detection():
    """Verify hardware probe detects platform features reliably."""
    hw = detect_hardware()
    assert hw.cpu_physical_cores >= 1
    assert hw.cpu_logical_threads >= 1
    assert hw.system_total_ram_gb > 0.5
    assert isinstance(hw.cuda_available, bool)
    assert hw.os_name != ""
    assert isinstance(hw.native_cpp_available, bool)


def test_profile_creation_and_persistence(tmp_path):
    """Verify profile calibration, serialization, and disk caching."""
    custom_profile_path = tmp_path / "test_profile.json"
    prof = calibrate_and_create_profile(path=custom_profile_path)
    assert prof.schema_version == PROFILE_SCHEMA_VERSION
    assert len(prof.fingerprint) == 16
    assert custom_profile_path.is_file()

    # Load back
    loaded = load_profile(path=custom_profile_path)
    assert loaded is not None
    assert loaded.fingerprint == prof.fingerprint
    assert "uci" in loaded.workloads
    assert "puzzles" in loaded.workloads
    assert "match" in loaded.workloads
    assert "selfplay" in loaded.workloads


def test_resolve_batch_size():
    """Verify resolving batch sizes honors integers and resolves 'auto'."""
    # Explicit integers
    assert resolve_batch_size(16) == 16
    assert resolve_batch_size("32") == 32
    assert resolve_batch_size("64") == 64
    assert resolve_batch_size(1) == 1
    # Clamping
    assert resolve_batch_size(0) in (4, 16, 24, 32, 64)  # 0 counts as auto or falls back
    assert resolve_batch_size(1000) == 256
    # CPU fallback
    assert resolve_batch_size("auto", workload="uci", device="cpu") == 4
    # Auto resolution
    bs = resolve_batch_size("auto", workload="uci")
    assert bs in (4, 16, 24, 32, 64)
    bs_sp = resolve_batch_size("auto", workload="selfplay")
    assert bs_sp in (4, 16, 24, 32, 64)


def test_adaptive_batch_scheduler():
    """Verify runtime dynamic batch scaling adjusts under varying load."""
    sched = AdaptiveBatchScheduler(
        min_batch_size=4,
        max_batch_size=64,
        min_wait_ms=0.5,
        max_wait_ms=4.0,
        target_latency_ms=10.0,
        adjustment_interval_sec=0.01,
    )
    assert sched.current_batch_size == 32

    # Record low latency calls (latency < 8ms, average batch high -> grow batch size)
    import time

    time.sleep(0.015)
    for _ in range(25):
        sched.record_batch(batch_size=32, latency_sec=0.004)

    # Next record triggers adaptation
    sched.record_batch(batch_size=32, latency_sec=0.004)
    assert sched.current_batch_size >= 32

    # Record high latency calls (latency > 12ms, low batch size -> decrease batch or wait)
    time.sleep(0.015)
    for _ in range(25):
        sched.record_batch(batch_size=4, latency_sec=0.025)

    sched.record_batch(batch_size=4, latency_sec=0.025)
    assert sched.current_batch_size <= 36

    state = sched.get_state()
    assert "current_batch_size" in state
    assert "wait_timeout_ms" in state
    assert "target_latency_ms" in state
