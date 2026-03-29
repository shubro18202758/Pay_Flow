"""
PayFlow — GPU Health Check via Native NVML System Calls
========================================================
Queries the NVIDIA Management Library (NVML) through ctypes to read GPU memory
state without any third-party dependency. This is a zero-install, zero-pip
health gate that runs BEFORE any heavy library (torch, xgboost) is imported.

NVML is a C API exposed by nvml.dll (Windows) / libnvidia-ml.so (Linux).
Every NVIDIA driver installation ships it. We call it directly via ctypes
to avoid pulling in pynvml/torch just for a memory check.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import platform
import struct
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── NVML Constants ────────────────────────────────────────────────────────────

NVML_SUCCESS = 0
NVML_DEVICE_NAME_BUFFER_SIZE = 96


# ── NVML Structures ──────────────────────────────────────────────────────────

class NvmlMemory(ctypes.Structure):
    """Maps to nvmlMemory_t: { total, free, used } in bytes."""
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlUtilization(ctypes.Structure):
    """Maps to nvmlUtilization_t: { gpu%, memory% }."""
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


# ── NVML Loader ──────────────────────────────────────────────────────────────

def _load_nvml() -> Optional[ctypes.CDLL]:
    """
    Load the NVML shared library using native system paths.
    Returns None if NVIDIA driver is not installed.
    """
    system = platform.system()

    if system == "Windows":
        # nvml.dll lives next to the driver in System32 or the NVIDIA folder
        search_paths = [
            "nvml.dll",
            r"C:\Windows\System32\nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
        ]
        for path in search_paths:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
        # Last resort: let ctypes search PATH
        try:
            return ctypes.WinDLL("nvml")  # type: ignore[attr-defined]
        except OSError:
            return None

    elif system == "Linux":
        lib_path = ctypes.util.find_library("nvidia-ml")
        if lib_path:
            try:
                return ctypes.CDLL(lib_path)
            except OSError:
                pass
        # Fallback hardcoded paths
        for path in ["/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "libnvidia-ml.so.1"]:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
        return None

    return None


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GPUDeviceInfo:
    index: int
    name: str
    total_mb: float
    used_mb: float
    free_mb: float
    gpu_utilization_pct: int
    mem_utilization_pct: int
    temperature_c: int

    @property
    def used_pct(self) -> float:
        return round((self.used_mb / self.total_mb) * 100, 1) if self.total_mb > 0 else 0.0


@dataclass
class HealthCheckResult:
    passed: bool
    gpu: Optional[GPUDeviceInfo]
    required_mb: float
    available_mb: float
    message: str


# ── Core Query Function ──────────────────────────────────────────────────────

def query_gpu(device_index: int = 0) -> Optional[GPUDeviceInfo]:
    """
    Query GPU memory and utilization via raw NVML C calls.
    Returns None if NVML is unavailable or the device doesn't exist.
    """
    nvml = _load_nvml()
    if nvml is None:
        logger.warning("NVML library not found. Is the NVIDIA driver installed?")
        return None

    # nvmlInit_v2()
    rc = nvml.nvmlInit_v2()
    if rc != NVML_SUCCESS:
        logger.error("nvmlInit_v2 failed with code %d", rc)
        return None

    try:
        # Get device handle
        handle = ctypes.c_void_p()
        rc = nvml.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(device_index), ctypes.byref(handle)
        )
        if rc != NVML_SUCCESS:
            logger.error("nvmlDeviceGetHandleByIndex_v2(%d) failed: %d", device_index, rc)
            return None

        # Device name
        name_buf = ctypes.create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
        nvml.nvmlDeviceGetName(handle, name_buf, ctypes.c_uint(NVML_DEVICE_NAME_BUFFER_SIZE))
        device_name = name_buf.value.decode("utf-8", errors="replace")

        # Memory info
        mem_info = NvmlMemory()
        rc = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem_info))
        if rc != NVML_SUCCESS:
            logger.error("nvmlDeviceGetMemoryInfo failed: %d", rc)
            return None

        # Utilization rates
        util = NvmlUtilization()
        rc = nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
        gpu_util = util.gpu if rc == NVML_SUCCESS else -1
        mem_util = util.memory if rc == NVML_SUCCESS else -1

        # Temperature
        temp = ctypes.c_uint()
        # NVML_TEMPERATURE_GPU = 0
        rc = nvml.nvmlDeviceGetTemperature(handle, ctypes.c_uint(0), ctypes.byref(temp))
        temperature = temp.value if rc == NVML_SUCCESS else -1

        return GPUDeviceInfo(
            index=device_index,
            name=device_name,
            total_mb=round(mem_info.total / (1024 * 1024), 1),
            used_mb=round(mem_info.used / (1024 * 1024), 1),
            free_mb=round(mem_info.free / (1024 * 1024), 1),
            gpu_utilization_pct=gpu_util,
            mem_utilization_pct=mem_util,
            temperature_c=temperature,
        )

    finally:
        nvml.nvmlShutdown()


# ── Health Gate ──────────────────────────────────────────────────────────────

# VRAM requirements for Qwen-3.5-9B at Q4_K_M + q8_0 KV cache + 16K context
LLM_MODEL_WEIGHT_MB = 5200.0
LLM_KV_CACHE_16K_Q8_MB = 1475.0
CUDA_OVERHEAD_MB = 400.0
SAFETY_MARGIN_MB = 200.0

LLM_TOTAL_REQUIRED_MB = (
    LLM_MODEL_WEIGHT_MB + LLM_KV_CACHE_16K_Q8_MB + CUDA_OVERHEAD_MB + SAFETY_MARGIN_MB
)  # ~7,275 MB


def check_vram_for_llm(
    required_mb: float = LLM_TOTAL_REQUIRED_MB,
    device_index: int = 0,
) -> HealthCheckResult:
    """
    Pre-flight health check: query GPU free VRAM via NVML and determine
    if the LLM can safely initialize without risking OOM.

    This MUST be called before `ollama.generate()` or `ollama.chat()`
    to prevent the Ollama daemon from loading the model into an already-
    saturated GPU (which causes hard driver crashes on Windows, not just
    Python exceptions).
    """
    gpu = query_gpu(device_index)

    if gpu is None:
        return HealthCheckResult(
            passed=False,
            gpu=None,
            required_mb=required_mb,
            available_mb=0.0,
            message=(
                "NVML query failed. Cannot verify GPU memory state. "
                "Ensure NVIDIA driver is installed and GPU is accessible."
            ),
        )

    if gpu.free_mb >= required_mb:
        return HealthCheckResult(
            passed=True,
            gpu=gpu,
            required_mb=required_mb,
            available_mb=gpu.free_mb,
            message=(
                f"GPU health OK. {gpu.name}: {gpu.free_mb:.0f} MB free "
                f"(need {required_mb:.0f} MB). Headroom: {gpu.free_mb - required_mb:.0f} MB."
            ),
        )

    # Insufficient VRAM — identify what's consuming it
    deficit = required_mb - gpu.free_mb
    return HealthCheckResult(
        passed=False,
        gpu=gpu,
        required_mb=required_mb,
        available_mb=gpu.free_mb,
        message=(
            f"INSUFFICIENT VRAM. {gpu.name}: {gpu.free_mb:.0f} MB free, "
            f"need {required_mb:.0f} MB (deficit: {deficit:.0f} MB). "
            f"Currently {gpu.used_mb:.0f} MB in use ({gpu.used_pct}%). "
            f"Actions: (1) flush PyTorch cache, (2) unload other models, "
            f"(3) close GPU-using applications."
        ),
    )


def check_vram_for_analysis(
    required_mb: float = 3584.0,  # XGBoost 1GB + GNN 2GB + overhead 0.5GB
    device_index: int = 0,
) -> HealthCheckResult:
    """Pre-flight check before ML/GNN workloads (Analysis mode)."""
    gpu = query_gpu(device_index)

    if gpu is None:
        return HealthCheckResult(
            passed=False, gpu=None, required_mb=required_mb,
            available_mb=0.0,
            message="NVML query failed. Cannot verify GPU state.",
        )

    if gpu.free_mb >= required_mb:
        return HealthCheckResult(
            passed=True, gpu=gpu, required_mb=required_mb,
            available_mb=gpu.free_mb,
            message=f"GPU health OK for analysis. {gpu.free_mb:.0f} MB free.",
        )

    deficit = required_mb - gpu.free_mb
    return HealthCheckResult(
        passed=False, gpu=gpu, required_mb=required_mb,
        available_mb=gpu.free_mb,
        message=(
            f"INSUFFICIENT VRAM for analysis. Need {required_mb:.0f} MB, "
            f"have {gpu.free_mb:.0f} MB (deficit: {deficit:.0f} MB). "
            f"Ensure LLM is unloaded (keep_alive=0)."
        ),
    )


# ── Pretty Print (for CLI diagnostics) ──────────────────────────────────────

def print_gpu_diagnostic(device_index: int = 0) -> None:
    """Print a full GPU diagnostic panel to stdout."""
    gpu = query_gpu(device_index)
    if gpu is None:
        print("[PayFlow] No GPU detected via NVML.")
        return

    bar_len = 40
    used_bars = int((gpu.used_mb / gpu.total_mb) * bar_len)
    free_bars = bar_len - used_bars

    print(f"""
┌─────────────────────────────────────────────────────────┐
│  PayFlow GPU Diagnostic (NVML native)                   │
├─────────────────────────────────────────────────────────┤
│  Device:       {gpu.name:<41s} │
│  VRAM Total:   {gpu.total_mb:>8.0f} MB                              │
│  VRAM Used:    {gpu.used_mb:>8.0f} MB ({gpu.used_pct:>5.1f}%)                      │
│  VRAM Free:    {gpu.free_mb:>8.0f} MB                              │
│  GPU Util:     {gpu.gpu_utilization_pct:>7d} %                               │
│  Mem Util:     {gpu.mem_utilization_pct:>7d} %                               │
│  Temperature:  {gpu.temperature_c:>7d} °C                              │
├─────────────────────────────────────────────────────────┤
│  [{"█" * used_bars}{"░" * free_bars}]  │
│   {"^used":>{used_bars + 5}s}  {"^free":>{free_bars}s}                  │
├─────────────────────────────────────────────────────────┤
│  LLM budget:   {LLM_TOTAL_REQUIRED_MB:>8.0f} MB  {"✓ FITS" if gpu.free_mb >= LLM_TOTAL_REQUIRED_MB else "✗ OOM":>27s}  │
│  Analysis:     {3584:>8.0f} MB  {"✓ FITS" if gpu.free_mb >= 3584 else "✗ OOM":>27s}  │
└─────────────────────────────────────────────────────────┘""")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_gpu_diagnostic()

    result = check_vram_for_llm()
    print(f"\nLLM Health Check: {'PASS' if result.passed else 'FAIL'}")
    print(f"  {result.message}")

    result_analysis = check_vram_for_analysis()
    print(f"\nAnalysis Health Check: {'PASS' if result_analysis.passed else 'FAIL'}")
    print(f"  {result_analysis.message}")
