"""
llamatelemetry.semconv.attrs - Attribute builder helpers.

Convenience functions that return ``Dict[str, Any]`` suitable for
``span.set_attributes()`` or OTel Resource creation.
"""

import uuid
from typing import Any, Dict, Optional

from . import keys


def run_id(rid: Optional[str] = None) -> Dict[str, Any]:
    """Return a run-id attribute dict (auto-generates UUID4 if *rid* is None)."""
    return {keys.RUN_ID: rid or str(uuid.uuid4())}


def gpu_attrs(
    gpu_id: int,
    utilization_pct: Optional[float] = None,
    mem_used_mb: Optional[float] = None,
    mem_total_mb: Optional[float] = None,
    power_w: Optional[float] = None,
    temp_c: Optional[float] = None,
) -> Dict[str, Any]:
    """Build GPU snapshot attribute dict."""
    attrs: Dict[str, Any] = {keys.GPU_ID: str(gpu_id)}
    if utilization_pct is not None:
        attrs[keys.GPU_UTILIZATION_PCT] = utilization_pct
    if mem_used_mb is not None:
        attrs[keys.GPU_MEM_USED_MB] = mem_used_mb
    if mem_total_mb is not None:
        attrs[keys.GPU_MEM_TOTAL_MB] = mem_total_mb
    if power_w is not None:
        attrs[keys.GPU_POWER_W] = power_w
    if temp_c is not None:
        attrs[keys.GPU_TEMP_C] = temp_c
    return attrs


def nccl_attrs(
    collective: str,
    nbytes: Optional[int] = None,
    wait_ms: Optional[float] = None,
    split_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Build NCCL collective attribute dict."""
    attrs: Dict[str, Any] = {keys.NCCL_COLLECTIVE: collective}
    if nbytes is not None:
        attrs[keys.NCCL_BYTES] = nbytes
    if wait_ms is not None:
        attrs[keys.NCCL_WAIT_MS] = wait_ms
    if split_mode is not None:
        attrs[keys.NCCL_SPLIT_MODE] = split_mode
    return attrs


# -- span helpers ----------------------------------------------------------

def set_gpu_attrs(span: Any, gpu_id: int, **extra: Any) -> None:
    """Attach GPU attributes to an active span."""
    span.set_attribute(keys.GPU_ID, str(gpu_id))
    for k, v in extra.items():
        span.set_attribute(k, v)


def set_nccl_attrs(
    span: Any,
    collective: str,
    nbytes: Optional[int] = None,
    wait_ms: Optional[float] = None,
    split_mode: Optional[str] = None,
) -> None:
    """Attach NCCL attributes to an active span."""
    span.set_attribute(keys.NCCL_COLLECTIVE, collective)
    if nbytes is not None:
        span.set_attribute(keys.NCCL_BYTES, nbytes)
    if wait_ms is not None:
        span.set_attribute(keys.NCCL_WAIT_MS, wait_ms)
    if split_mode is not None:
        span.set_attribute(keys.NCCL_SPLIT_MODE, split_mode)
