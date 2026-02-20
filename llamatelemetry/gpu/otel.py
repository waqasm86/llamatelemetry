"""
llamatelemetry.gpu.otel - GPU span enricher for OpenTelemetry.

Provides GPU utilization snapshots and attaches delta attributes to spans.
Shared by both llama.cpp and Transformers backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from ..semconv import keys


@dataclass
class GPUSnapshot:
    """Point-in-time GPU utilization snapshot.

    Attributes:
        gpu_id: GPU device index.
        util: GPU utilization percentage (0-100).
        mem_used_mb: GPU memory used in MB.
        mem_total_mb: Total GPU memory in MB.
        power_w: Power draw in watts.
        temp_c: Temperature in Celsius.
    """

    gpu_id: int = 0
    util: Optional[float] = None
    mem_used_mb: Optional[int] = None
    mem_total_mb: Optional[int] = None
    power_w: Optional[float] = None
    temp_c: Optional[int] = None


class GPUSpanEnricher:
    """Attaches GPU utilization deltas to OTel spans.

    Takes before/after snapshots around inference and computes deltas for:
    - GPU utilization change
    - Memory usage change
    - Power draw change

    Example:
        >>> enricher = GPUSpanEnricher(device_index=0)
        >>> before = enricher.snapshot()
        >>> # ... run inference ...
        >>> after = enricher.snapshot()
        >>> enricher.attach_deltas(span, before, after)
    """

    def __init__(self, device_index: Optional[int] = None):
        """Initialize GPU span enricher.

        Args:
            device_index: GPU device index to monitor. None = first GPU (0).
        """
        self._device_index = device_index if device_index is not None else 0

    def snapshot(self) -> GPUSnapshot:
        """Take a point-in-time GPU utilization snapshot.

        Uses nvidia-smi via the existing gpu/nvml module.

        Returns:
            GPUSnapshot with current GPU state.
        """
        try:
            from .nvml import snapshot as nvml_snapshot

            snaps = nvml_snapshot(gpu_id=self._device_index)
            if snaps:
                s = snaps[0]
                return GPUSnapshot(
                    gpu_id=s.gpu_id,
                    util=s.utilization_pct,
                    mem_used_mb=s.mem_used_mb,
                    mem_total_mb=s.mem_total_mb,
                    power_w=s.power_w,
                    temp_c=s.temp_c,
                )
        except Exception:
            pass

        return GPUSnapshot(gpu_id=self._device_index)

    def attach_deltas(
        self,
        span: Any,
        before: GPUSnapshot,
        after: GPUSnapshot,
    ) -> None:
        """Attach GPU utilization deltas to a span.

        Sets attributes:
            - gpu.id
            - gpu.utilization.delta
            - gpu.memory.used_mb.delta
            - gpu.power.w.delta
            - gpu.memory.used_mb (current)
            - gpu.memory.total_mb (current)

        Args:
            span: An OTel span (or noop span).
            before: Snapshot taken before inference.
            after: Snapshot taken after inference.
        """
        span.set_attribute(keys.GPU_ID, str(after.gpu_id))

        if after.util is not None:
            span.set_attribute(keys.GPU_UTILIZATION_PCT, after.util)
        if after.mem_used_mb is not None:
            span.set_attribute(keys.GPU_MEM_USED_MB, after.mem_used_mb)
        if after.mem_total_mb is not None:
            span.set_attribute(keys.GPU_MEM_TOTAL_MB, after.mem_total_mb)
        if after.power_w is not None:
            span.set_attribute(keys.GPU_POWER_W, after.power_w)
        if after.temp_c is not None:
            span.set_attribute(keys.GPU_TEMP_C, after.temp_c)

        # Deltas
        if before.util is not None and after.util is not None:
            span.set_attribute("gpu.utilization.delta", after.util - before.util)
        if before.mem_used_mb is not None and after.mem_used_mb is not None:
            span.set_attribute("gpu.memory.used_mb.delta", after.mem_used_mb - before.mem_used_mb)
        if before.power_w is not None and after.power_w is not None:
            span.set_attribute("gpu.power.w.delta", after.power_w - before.power_w)

    def attach_static(self, span: Any) -> None:
        """Attach current GPU state to a span (no delta computation).

        Args:
            span: An OTel span.
        """
        snap = self.snapshot()
        span.set_attribute(keys.GPU_ID, str(snap.gpu_id))
        if snap.util is not None:
            span.set_attribute(keys.GPU_UTILIZATION_PCT, snap.util)
        if snap.mem_used_mb is not None:
            span.set_attribute(keys.GPU_MEM_USED_MB, snap.mem_used_mb)
        if snap.mem_total_mb is not None:
            span.set_attribute(keys.GPU_MEM_TOTAL_MB, snap.mem_total_mb)
        if snap.power_w is not None:
            span.set_attribute(keys.GPU_POWER_W, snap.power_w)
        if snap.temp_c is not None:
            span.set_attribute(keys.GPU_TEMP_C, snap.temp_c)
