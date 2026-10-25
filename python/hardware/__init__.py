"""Hardware detection, bounded calibration, and runtime recommendation package for ZeroPawn."""

from python.hardware.detector import HardwareInfo, detect_hardware
from python.hardware.calibration import CalibrationReport, run_calibration
from python.hardware.recommender import RecommendedConfig, recommend_configuration

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "CalibrationReport",
    "run_calibration",
    "RecommendedConfig",
    "recommend_configuration",
]
