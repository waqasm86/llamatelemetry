"""
llamatelemetry.llama.autotune - Adaptive tuning for llama.cpp server settings.

Probes GPU capabilities and model characteristics to choose optimal settings
for n_ctx, n_batch, n_ubatch, n_gpu_layers, and serving profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class TuningProfile(Enum):
    """Tuning profile for llama.cpp server optimization."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    LOW_VRAM = "low_vram"
    BALANCED = "balanced"


@dataclass
class AutotuneResult:
    """Result of auto-tuning analysis.

    Attributes:
        n_ctx: Recommended context length.
        n_batch: Recommended batch size.
        n_ubatch: Recommended micro-batch size.
        n_gpu_layers: Recommended GPU layers (-1 = all).
        mmap: Use memory-mapped file.
        mlock: Lock model in RAM.
        n_threads: CPU threads for non-GPU ops.
        profile: Selected tuning profile.
        estimated_vram_mb: Estimated VRAM usage.
        notes: Explanation of choices.
    """

    n_ctx: int = 2048
    n_batch: int = 512
    n_ubatch: int = 512
    n_gpu_layers: int = -1
    mmap: bool = True
    mlock: bool = False
    n_threads: int = 4
    profile: str = "balanced"
    estimated_vram_mb: float = 0.0
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def to_server_args(self) -> Dict[str, Any]:
        """Convert to llama-server command-line arguments."""
        return {
            "-c": self.n_ctx,
            "-b": self.n_batch,
            "-ub": self.n_ubatch,
            "-ngl": self.n_gpu_layers,
            "--mmap" if self.mmap else "--no-mmap": None,
            "-t": self.n_threads,
        }


def autotune(
    model_size_mb: Optional[float] = None,
    vram_available_mb: Optional[float] = None,
    gpu_count: int = 1,
    profile: TuningProfile = TuningProfile.BALANCED,
    quant_type: str = "Q4_K_M",
) -> AutotuneResult:
    """Auto-tune llama.cpp server settings based on hardware and model.

    Args:
        model_size_mb: Model file size in MB. Auto-detected if possible.
        vram_available_mb: Available VRAM in MB. Auto-detected if possible.
        gpu_count: Number of GPUs.
        profile: Tuning profile.
        quant_type: GGUF quantization type.

    Returns:
        AutotuneResult with recommended settings.

    Example:
        >>> result = autotune(model_size_mb=900, vram_available_mb=15000)
        >>> print(f"Recommended ctx: {result.n_ctx}, batch: {result.n_batch}")
        >>> print(f"GPU layers: {result.n_gpu_layers}")
    """
    # Auto-detect VRAM if not provided
    if vram_available_mb is None:
        vram_available_mb = _detect_vram_mb()

    result = AutotuneResult(profile=profile.value)

    # Determine GPU layers
    if model_size_mb and vram_available_mb:
        # Rough estimate: model needs ~1.2x its file size in VRAM
        model_vram_estimate = model_size_mb * 1.2
        if model_vram_estimate < vram_available_mb * 0.8:
            result.n_gpu_layers = -1  # Fully offload
            result.notes.append(f"Full GPU offload: model ~{model_vram_estimate:.0f}MB, VRAM ~{vram_available_mb:.0f}MB")
        else:
            # Partial offload
            ratio = (vram_available_mb * 0.7) / model_vram_estimate
            result.n_gpu_layers = max(1, int(ratio * 40))  # Rough layer estimate
            result.notes.append(f"Partial GPU offload: {result.n_gpu_layers} layers")
    else:
        result.n_gpu_layers = -1
        result.notes.append("VRAM not detected, defaulting to full GPU offload")

    # Profile-specific settings
    if profile == TuningProfile.LATENCY:
        result.n_ctx = 2048
        result.n_batch = 512
        result.n_ubatch = 256
        result.mlock = True
        result.notes.append("Latency profile: smaller context, locked memory")

    elif profile == TuningProfile.THROUGHPUT:
        result.n_ctx = 4096
        result.n_batch = 2048
        result.n_ubatch = 512
        result.notes.append("Throughput profile: larger batch, more context")

    elif profile == TuningProfile.LOW_VRAM:
        result.n_ctx = 1024
        result.n_batch = 256
        result.n_ubatch = 128
        result.n_gpu_layers = max(1, result.n_gpu_layers // 2) if result.n_gpu_layers > 0 else 10
        result.notes.append("Low-VRAM profile: reduced context and batch")

    else:  # BALANCED
        result.n_ctx = 2048
        result.n_batch = 512
        result.n_ubatch = 512
        result.notes.append("Balanced profile: default settings")

    # Multi-GPU adjustments
    if gpu_count > 1:
        result.n_ctx = min(result.n_ctx * 2, 8192)
        result.notes.append(f"Multi-GPU ({gpu_count}x): doubled context to {result.n_ctx}")

    # Estimate VRAM usage
    if model_size_mb:
        ctx_vram = result.n_ctx * 2 * 128 / (1024 * 1024)  # Rough KV cache estimate
        result.estimated_vram_mb = model_size_mb * 1.2 + ctx_vram

    return result


def _detect_vram_mb() -> Optional[float]:
    """Try to detect available VRAM in MB."""
    try:
        from ..gpu.nvml import list_devices
        devices = list_devices()
        if devices:
            return float(devices[0].memory_total_mb)
    except Exception:
        pass
    return None
