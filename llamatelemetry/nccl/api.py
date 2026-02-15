"""
llamatelemetry.nccl.api - Simplified NCCL tracing API.

Simplified from api/nccl.py (552 lines -> ~60 lines).
"""

from typing import Any, Optional

_enabled = False


def enable(enabled: bool = True) -> None:
    """Toggle NCCL tracing."""
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    """Return whether NCCL tracing is active."""
    return _enabled


def annotate_collective(
    name: str,
    nbytes: Optional[int] = None,
    wait_ms: Optional[float] = None,
    **attrs: Any,
) -> None:
    """
    Create a span for an NCCL collective operation.

    Args:
        name: Collective name (e.g. "allreduce", "allgather").
        nbytes: Bytes transferred.
        wait_ms: Wall-clock wait time in milliseconds.
        **attrs: Additional span attributes.
    """
    if not _enabled:
        return

    from ..otel.provider import get_tracer
    from ..semconv import keys

    tracer = get_tracer("llamatelemetry.nccl")
    with tracer.start_as_current_span(f"nccl.{name}") as span:
        span.set_attribute(keys.NCCL_COLLECTIVE, name)
        if nbytes is not None:
            span.set_attribute(keys.NCCL_BYTES, nbytes)
        if wait_ms is not None:
            span.set_attribute(keys.NCCL_WAIT_MS, wait_ms)
        for k, v in attrs.items():
            span.set_attribute(k, v)
