"""Hardware detection and system capability discovery for ZeroPawn."""

from __future__ import annotations

import dataclasses
import platform
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import psutil


@dataclasses.dataclass
class HardwareInfo:
    """Structured hardware and environment information."""

    os_name: str
    os_release: str
    os_arch: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_runtime_version: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    gpu_count: int = 0
    gpu_name: str = "None"
    gpu_compute_capability: Optional[Tuple[int, int]] = None
    gpu_total_vram_mb: float = 0.0
    gpu_free_vram_mb: float = 0.0
    gpu_power_draw_w: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    gpu_throttle_reasons: Optional[str] = None
    cpu_physical_cores: int = 1
    cpu_logical_threads: int = 1
    cpu_is_hybrid: bool = False
    cpu_p_cores: Optional[int] = None
    cpu_e_cores: Optional[int] = None
    system_total_ram_gb: float = 0.0
    system_available_ram_gb: float = 0.0
    native_cpp_available: bool = False
    native_cpp_version: str = "N/A"
    native_cpp_path: str = "N/A"

    @property
    def bf16_supported(self) -> bool:
        if not self.cuda_available or not self.gpu_compute_capability:
            return False
        return self.gpu_compute_capability[0] >= 8

    @property
    def fp16_supported(self) -> bool:
        if not self.cuda_available or not self.gpu_compute_capability:
            return False
        return (self.gpu_compute_capability[0] > 5) or (
            self.gpu_compute_capability[0] == 5 and self.gpu_compute_capability[1] >= 3
        )

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["bf16_supported"] = self.bf16_supported
        d["fp16_supported"] = self.fp16_supported
        return d

    def summary_table(self) -> List[Tuple[str, str]]:
        """Return list of (key, value) pairs for display."""
        rows = [
            ("Operating System", f"{self.os_name} {self.os_release} ({self.os_arch})"),
            ("Python Runtime", self.python_version),
            (
                "PyTorch Version",
                f"{self.pytorch_version} (CUDA runtime: {self.cuda_runtime_version or 'N/A'})",
            ),
            ("CUDA Status", "Available" if self.cuda_available else "Not Available"),
        ]
        if self.cuda_available and self.gpu_count > 0:
            cc = (
                f"sm_{self.gpu_compute_capability[0]}{self.gpu_compute_capability[1]}"
                if self.gpu_compute_capability
                else "N/A"
            )
            rows.append(("GPU Device", f"{self.gpu_name} ({cc})"))
            rows.append(
                (
                    "VRAM Total / Free",
                    f"{self.gpu_total_vram_mb:.0f} MB / {self.gpu_free_vram_mb:.0f} MB",
                )
            )
            if self.gpu_temp_c is not None:
                rows.append(
                    (
                        "GPU Telemetry",
                        f"Temp: {self.gpu_temp_c:.1f}°C | Power: {self.gpu_power_draw_w:.1f}W",
                    )
                )
            if self.gpu_throttle_reasons and self.gpu_throttle_reasons != "None":
                rows.append(("GPU Throttle Flags", self.gpu_throttle_reasons))
        else:
            rows.append(("GPU Device", "CPU-only execution"))

        cpu_desc = (
            f"{self.cpu_physical_cores} Physical Cores, {self.cpu_logical_threads} Logical Threads"
        )
        if self.cpu_is_hybrid and self.cpu_p_cores and self.cpu_e_cores:
            cpu_desc += f" (Hybrid Intel: {self.cpu_p_cores} P-cores + {self.cpu_e_cores} E-cores)"
        rows.append(("CPU Topology", cpu_desc))
        rows.append(
            (
                "System RAM",
                f"{self.system_available_ram_gb:.1f} GB available / {self.system_total_ram_gb:.1f} GB total",
            )
        )
        rows.append(
            (
                "Native C++ (zeropawn_cpp)",
                f"Active v{self.native_cpp_version}"
                if self.native_cpp_available
                else "Unavailable",
            )
        )
        return rows


def _detect_cpu_topology() -> Tuple[int, int, bool, Optional[int], Optional[int]]:
    """Detect CPU physical cores, logical threads, and hybrid P/E topology."""
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or 1
    is_hybrid = False
    p_cores = None
    e_cores = None

    # Check for Intel Hybrid Architecture (Alder Lake / Raptor Lake: e.g. 10 cores / 16 threads = 6 P-cores with HT (12) + 4 E-cores (4))
    # Heuristic: if logical != physical and logical != 2 * physical, it's a hybrid architecture:
    # 2 * P + E = logical, and P + E = physical => P = logical - physical, E = 2 * physical - logical
    if physical < logical < 2 * physical:
        calc_p = logical - physical
        calc_e = 2 * physical - logical
        if calc_p > 0 and calc_e > 0:
            is_hybrid = True
            p_cores = calc_p
            e_cores = calc_e

    return physical, logical, is_hybrid, p_cores, e_cores


def _query_nvidia_smi() -> Dict[str, Any]:
    """Query nvidia-smi for power, temperature, clocks, and throttle flags."""
    res = {
        "power_w": None,
        "temp_c": None,
        "throttle": "None",
    }
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, encoding="utf-8", timeout=2
        ).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 2:
                try:
                    p_val = float(parts[0])
                    # On laptop GPUs, nvidia-smi power.draw can report milliwatts when idle (e.g. 590.0 mW = 0.59W)
                    if p_val > 150.0:
                        p_val = p_val / 1000.0
                    res["power_w"] = p_val
                    res["temp_c"] = float(parts[1])
                except ValueError:
                    pass
    except Exception:
        pass

    try:
        cmd = ["nvidia-smi", "-q", "-d", "PERFORMANCE"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, encoding="utf-8", timeout=2)
        reasons = []
        in_reasons = False
        for line in out.splitlines():
            line_str = line.strip()
            if "Clocks Event Reasons" in line_str and "Counters" not in line_str:
                in_reasons = True
                continue
            if in_reasons:
                if "Counters" in line_str or ":" not in line_str:
                    break
                k, v = [x.strip() for x in line_str.split(":", 1)]
                if v == "Active" and k != "Idle":
                    reasons.append(k)
        if reasons:
            res["throttle"] = ", ".join(reasons)
    except Exception:
        pass

    return res


def detect_hardware() -> HardwareInfo:
    """Discover real system hardware and runtime capabilities."""
    os_name = platform.system()
    os_release = platform.release()
    os_arch = platform.machine()
    python_ver = platform.python_version()

    import torch

    pytorch_ver = torch.__version__
    cuda_runtime = torch.version.cuda if hasattr(torch.version, "cuda") else None
    cuda_avail = bool(torch.cuda.is_available())

    gpu_count = 0
    gpu_name = "None"
    gpu_capability = None
    total_vram_mb = 0.0
    free_vram_mb = 0.0

    if cuda_avail:
        try:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_capability = torch.cuda.get_device_capability(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                total_vram_mb = total_bytes / (1024.0 * 1024.0)
                free_vram_mb = free_bytes / (1024.0 * 1024.0)
        except Exception:
            cuda_avail = False

    smi_info = (
        _query_nvidia_smi() if cuda_avail else {"power_w": None, "temp_c": None, "throttle": "None"}
    )
    physical_cores, logical_threads, is_hybrid, p_cores, e_cores = _detect_cpu_topology()

    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024.0**3)
    avail_ram_gb = vm.available / (1024.0**3)

    # Native C++ module check
    native_cpp_avail = False
    native_cpp_ver = "N/A"
    native_cpp_path = "N/A"
    try:
        import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
        import zeropawn_cpp

        native_cpp_avail = True
        native_cpp_ver = getattr(zeropawn_cpp, "version", lambda: "0.1.0-phase1")()
        native_cpp_path = getattr(zeropawn_cpp, "__file__", "loaded")
    except Exception:
        pass

    return HardwareInfo(
        os_name=os_name,
        os_release=os_release,
        os_arch=os_arch,
        python_version=python_ver,
        pytorch_version=pytorch_ver,
        cuda_available=cuda_avail,
        cuda_runtime_version=cuda_runtime,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_compute_capability=gpu_capability,
        gpu_total_vram_mb=total_vram_mb,
        gpu_free_vram_mb=free_vram_mb,
        gpu_power_draw_w=smi_info["power_w"],
        gpu_temp_c=smi_info["temp_c"],
        gpu_throttle_reasons=smi_info["throttle"],
        cpu_physical_cores=physical_cores,
        cpu_logical_threads=logical_threads,
        cpu_is_hybrid=is_hybrid,
        cpu_p_cores=p_cores,
        cpu_e_cores=e_cores,
        system_total_ram_gb=total_ram_gb,
        system_available_ram_gb=avail_ram_gb,
        native_cpp_available=native_cpp_avail,
        native_cpp_version=native_cpp_ver,
        native_cpp_path=native_cpp_path,
    )
