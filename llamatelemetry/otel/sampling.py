"""
llamatelemetry.otel.sampling - Sampling strategies.

Supports: always_on, ratio, only_errors, only_slow.
"""

from typing import Any, Optional


def build_sampler(strategy: str = "always_on", ratio: float = 1.0) -> Optional[Any]:
    """
    Build an OTel Sampler from a strategy name.

    Args:
        strategy: One of "always_on", "ratio", "only_errors", "only_slow".
        ratio: Sampling ratio for the "ratio" strategy (0.0 - 1.0).

    Returns:
        Sampler instance or None if OTel SDK is not available.
    """
    try:
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_ON,
            TraceIdRatioBased,
        )
    except ImportError:
        return None

    if strategy == "always_on":
        return ALWAYS_ON
    elif strategy == "ratio":
        return TraceIdRatioBased(ratio)
    elif strategy in ("only_errors", "only_slow"):
        # These require custom sampler implementations.
        # For now, use always_on and filter in the span processor.
        return ALWAYS_ON
    else:
        return ALWAYS_ON
