"""Tests for Hardware Detection, Bounded Calibration, and Hardware-Aware Recommendations."""

import time
from unittest import mock

import pytest
from python.hardware.calibration import CalibrationReport, run_bounded_calibration
from python.hardware.detector import HardwareInfo, detect_hardware
from python.hardware.recommender import RecommendedConfig, recommend_configuration


def test_detect_hardware_structure():
    """Verify detect_hardware returns structured information without exceptions."""
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.os_name != "unknown"
    assert hw.cpu_logical_threads > 0
    assert hw.system_total_ram_gb > 0.0
    assert isinstance(hw.cuda_available, bool)

    d = hw.to_dict()
    assert "cpu_physical_cores" in d
    assert "system_total_ram_gb" in d
    assert "cuda_available" in d


def test_bounded_calibration_budget_enforcement():
    """Verify calibration finishes strictly within its time budget."""
    budget_sec = 3.0
    t0 = time.time()
    report = run_bounded_calibration(max_duration_sec=budget_sec)
    duration = time.time() - t0

    assert isinstance(report, CalibrationReport)
    # Total runtime should not significantly exceed the requested budget
    assert duration < budget_sec + 1.5
    assert report.duration_sec <= duration


def test_recommend_configuration_cuda_system():
    """Verify recommendations for modern GPU system with hybrid Intel CPU."""
    synthetic_hw = HardwareInfo(
        os_name="Windows",
        os_release="11",
        os_arch="AMD64",
        python_version="3.12.10",
        pytorch_version="2.13.0+cu126",
        cuda_available=True,
        cuda_runtime_version="12.6",
        gpu_count=1,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        gpu_compute_capability=(8, 9),
        gpu_total_vram_mb=8188.0,
        gpu_free_vram_mb=7500.0,
        cpu_physical_cores=10,
        cpu_logical_threads=16,
        cpu_is_hybrid=True,
        cpu_p_cores=6,
        cpu_e_cores=4,
        system_total_ram_gb=15.7,
        system_available_ram_gb=8.5,
        native_cpp_available=True,
    )

    rec = recommend_configuration(synthetic_hw, workload="rl")
    assert isinstance(rec, RecommendedConfig)
    assert rec.train_device == "cuda"
    assert rec.infer_device == "cuda"
    assert rec.search_device == "cpu"
    assert rec.concurrency == 24
    assert rec.inference_max_batch_size == 128
    assert rec.inference_max_wait_ms == 1.5
    assert rec.watchdog_rss_gb == 7.5
    assert rec.watchdog_free_ram_gb == 2.0
    assert rec.use_separate_cuda_streams is True


def test_recommend_configuration_cpu_only_system():
    """Verify recommendations when CUDA is absent or CPU forced."""
    synthetic_hw = HardwareInfo(
        os_name="Linux",
        os_release="6.8.0",
        os_arch="x86_64",
        python_version="3.12.0",
        pytorch_version="2.13.0+cpu",
        cuda_available=False,
        cuda_runtime_version=None,
        gpu_count=0,
        gpu_name="None",
        gpu_compute_capability=None,
        gpu_total_vram_mb=0.0,
        gpu_free_vram_mb=0.0,
        cpu_physical_cores=8,
        cpu_logical_threads=16,
        cpu_is_hybrid=False,
        cpu_p_cores=8,
        cpu_e_cores=0,
        system_total_ram_gb=16.0,
        system_available_ram_gb=10.0,
        native_cpp_available=True,
    )

    rec = recommend_configuration(synthetic_hw, workload="rl")
    assert rec.train_device == "cpu"
    assert rec.infer_device == "cpu"
    assert rec.search_device == "cpu"
    assert rec.concurrency <= 12
    assert rec.use_separate_cuda_streams is False


def test_recommend_configuration_force_device_cpu():
    """Verify forced CPU override on a CUDA capable machine."""
    synthetic_hw = HardwareInfo(
        os_name="Windows",
        os_release="11",
        os_arch="AMD64",
        python_version="3.12.10",
        pytorch_version="2.13.0+cu126",
        cuda_available=True,
        cuda_runtime_version="12.6",
        gpu_count=1,
        gpu_name="RTX 4060",
        gpu_compute_capability=(8, 9),
        gpu_total_vram_mb=8192.0,
        gpu_free_vram_mb=6000.0,
        cpu_physical_cores=8,
        cpu_logical_threads=16,
        cpu_is_hybrid=False,
        cpu_p_cores=8,
        cpu_e_cores=0,
        system_total_ram_gb=16.0,
        system_available_ram_gb=8.0,
        native_cpp_available=True,
    )

    rec = recommend_configuration(synthetic_hw, force_device="cpu")
    assert rec.train_device == "cpu"
    assert rec.infer_device == "cpu"
    assert "Explicit user request" in rec.rationale["train_device"]
