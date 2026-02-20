"""
llamatelemetry.distributed.topology - GPU topology detection.

Detects GPU count, capabilities, interconnect hints, and selects
appropriate multi-GPU mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MultiGPUMode(Enum):
    """Multi-GPU execution mode."""

    SINGLE = "single"
    SPLIT = "split"  # GPU 0: inference, GPU 1: analytics
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    DATA_PARALLEL = "data_parallel"


@dataclass
class GPUInfo:
    """Information about a single GPU.

    Attributes:
        index: GPU device index.
        name: GPU name (e.g. "Tesla T4").
        memory_total_mb: Total memory in MB.
        compute_capability: Compute capability (e.g. "7.5").
        driver_version: NVIDIA driver version.
    """

    index: int = 0
    name: str = ""
    memory_total_mb: int = 0
    compute_capability: str = ""
    driver_version: str = ""

    @property
    def sm_version(self) -> float:
        """Parse compute capability as float."""
        try:
            return float(self.compute_capability)
        except (ValueError, TypeError):
            return 0.0

    @property
    def supports_tf32(self) -> bool:
        """Whether this GPU supports TF32 (Ampere+, SM 8.0+)."""
        return self.sm_version >= 8.0

    @property
    def supports_bf16(self) -> bool:
        """Whether this GPU supports BF16 (Ampere+, SM 8.0+)."""
        return self.sm_version >= 8.0

    @property
    def supports_flash_attn(self) -> bool:
        """Whether this GPU supports FlashAttention (SM 7.5+)."""
        return self.sm_version >= 7.5


@dataclass
class GPUTopology:
    """System GPU topology.

    Attributes:
        gpus: List of detected GPUs.
        gpu_count: Number of GPUs.
        total_vram_mb: Total VRAM across all GPUs.
        nvlink_detected: Whether NVLink is detected (best effort).
        recommended_mode: Recommended multi-GPU mode.
    """

    gpus: List[GPUInfo] = field(default_factory=list)
    gpu_count: int = 0
    total_vram_mb: int = 0
    nvlink_detected: bool = False
    recommended_mode: MultiGPUMode = MultiGPUMode.SINGLE

    @property
    def is_multi_gpu(self) -> bool:
        return self.gpu_count > 1

    @property
    def is_kaggle_t4_dual(self) -> bool:
        """Check if this is a Kaggle dual T4 environment."""
        if self.gpu_count != 2:
            return False
        return all("T4" in g.name for g in self.gpus)


def detect_topology() -> GPUTopology:
    """Detect the current GPU topology.

    Returns:
        GPUTopology with detected hardware information.

    Example:
        >>> topo = detect_topology()
        >>> print(f"GPUs: {topo.gpu_count}, VRAM: {topo.total_vram_mb}MB")
        >>> print(f"Mode: {topo.recommended_mode.value}")
    """
    topology = GPUTopology()

    try:
        from ..gpu.nvml import list_devices
        devices = list_devices()

        for dev in devices:
            topology.gpus.append(GPUInfo(
                index=dev.id,
                name=dev.name,
                memory_total_mb=dev.memory_total_mb,
                compute_capability=dev.compute_capability,
                driver_version=dev.driver_version,
            ))

        topology.gpu_count = len(topology.gpus)
        topology.total_vram_mb = sum(g.memory_total_mb for g in topology.gpus)

    except Exception:
        # Try torch fallback
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    topology.gpus.append(GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total_mb=int(props.total_mem / (1024 * 1024)),
                        compute_capability=f"{props.major}.{props.minor}",
                    ))
                topology.gpu_count = count
                topology.total_vram_mb = sum(g.memory_total_mb for g in topology.gpus)
        except ImportError:
            pass

    # Determine recommended mode
    topology.recommended_mode = _recommend_mode(topology)

    # NVLink detection (best effort)
    topology.nvlink_detected = _detect_nvlink(topology)

    return topology


def _recommend_mode(topology: GPUTopology) -> MultiGPUMode:
    """Recommend a multi-GPU mode based on topology."""
    if topology.gpu_count <= 1:
        return MultiGPUMode.SINGLE

    if topology.is_kaggle_t4_dual:
        return MultiGPUMode.SPLIT  # GPU 0: inference, GPU 1: analytics

    if topology.gpu_count == 2:
        return MultiGPUMode.TENSOR_PARALLEL

    if topology.gpu_count >= 4:
        return MultiGPUMode.DATA_PARALLEL

    return MultiGPUMode.SPLIT


def _detect_nvlink(topology: GPUTopology) -> bool:
    """Best-effort NVLink detection."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "NV" in result.stdout:
            return True
    except Exception:
        pass
    return False
