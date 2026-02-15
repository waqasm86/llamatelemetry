"""
llamatelemetry.gpu.nvml - nvidia-smi based GPU queries.

Refactored from api/multigpu.py (detect_gpus) and telemetry/metrics.py (_query_nvidia_smi).
"""

import subprocess
import time
from typing import List, Optional

from .schemas import GPUDevice, GPUSnapshot, GPUSamplerHandle


def _query_nvidia_smi_full() -> List[dict]:
    """Query nvidia-smi for full device info."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            devices = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    devices.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "compute_capability": parts[3],
                            "driver_version": parts[4],
                        }
                    )
            return devices
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def _query_nvidia_smi_utilization() -> List[GPUSnapshot]:
    """Query nvidia-smi for current utilisation readings."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            now = time.time()
            snapshots = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    snapshots.append(
                        GPUSnapshot(
                            gpu_id=int(parts[0]),
                            timestamp=now,
                            mem_used_mb=int(parts[1]),
                            mem_total_mb=int(parts[2]),
                            utilization_pct=int(parts[3]),
                            power_w=float(parts[4]) if parts[4] not in ("[N/A]", "") else 0.0,
                            temp_c=int(parts[5]) if parts[5] not in ("[N/A]", "") else 0,
                        )
                    )
            return snapshots
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def list_devices() -> List[GPUDevice]:
    """Return a list of all detected NVIDIA GPUs."""
    raw = _query_nvidia_smi_full()
    return [
        GPUDevice(
            id=d["index"],
            name=d["name"],
            memory_total_mb=d["memory_total_mb"],
            compute_capability=d["compute_capability"],
            driver_version=d["driver_version"],
        )
        for d in raw
    ]


def snapshot(gpu_id: Optional[int] = None) -> List[GPUSnapshot]:
    """
    Take a point-in-time GPU utilisation snapshot.

    Args:
        gpu_id: If given, filter to a single GPU.

    Returns:
        List of GPUSnapshot (one per GPU, or one if *gpu_id* specified).
    """
    snaps = _query_nvidia_smi_utilization()
    if gpu_id is not None:
        snaps = [s for s in snaps if s.gpu_id == gpu_id]
    return snaps


def start_sampler(interval_ms: int = 200) -> GPUSamplerHandle:
    """
    Start a background thread that polls GPU utilisation.

    Args:
        interval_ms: Milliseconds between polls (default 200).

    Returns:
        GPUSamplerHandle - call ``.stop()`` to terminate.
    """
    handle = GPUSamplerHandle(interval_ms=interval_ms)
    handle.start(_query_nvidia_smi_utilization)
    return handle
