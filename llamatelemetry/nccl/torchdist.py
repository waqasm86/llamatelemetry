"""
llamatelemetry.nccl.torchdist - Auto-instrumentation for torch.distributed collectives.

Monkeypatches torch.distributed collective operations (all_reduce, all_gather, etc.)
to automatically create NCCL spans with tensor sizes, dtype, world_size, and rank.

All torch imports are lazy - this module is safe to import without torch installed.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional

from ..otel.provider import get_tracer
from ..semconv import keys


# Collectives to instrument
_COLLECTIVES = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "barrier",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "all_to_all",
    "gather",
    "scatter",
    "reduce",
]


class TorchDistributedInstrumentor:
    """Auto-instruments torch.distributed collective operations.

    Monkeypatches torch.distributed.* collectives to create NCCL spans
    automatically during DDP/FSDP training or multi-GPU inference.

    Example:
        >>> instrumentor = TorchDistributedInstrumentor()
        >>> instrumentor.instrument()
        >>> # Now all torch.distributed collectives are traced
        >>> # ...
        >>> instrumentor.uninstrument()
    """

    def __init__(self, enabled: bool = True):
        """Initialize the torch.distributed instrumentor.

        Args:
            enabled: Whether instrumentation is active.
        """
        self.enabled = enabled
        self._patched = False
        self._originals: Dict[str, Callable] = {}

    def instrument(self) -> None:
        """Monkeypatch torch.distributed collectives to create spans.

        Each collective call will:
        - Create a span named nccl.<operation>
        - Attach tensor sizes, dtype, world_size, rank
        - Record timing
        """
        if self._patched or not self.enabled:
            return

        try:
            import torch.distributed as dist
        except ImportError:
            return

        if not dist.is_available():
            return

        tracer = get_tracer("llamatelemetry.nccl")

        for op_name in _COLLECTIVES:
            if not hasattr(dist, op_name):
                continue

            original = getattr(dist, op_name)
            self._originals[op_name] = original

            wrapped = self._make_wrapper(original, op_name, tracer)
            setattr(dist, op_name, wrapped)

        self._patched = True

    def uninstrument(self) -> None:
        """Restore original torch.distributed functions."""
        if not self._patched:
            return

        try:
            import torch.distributed as dist
        except ImportError:
            return

        for op_name, original in self._originals.items():
            if hasattr(dist, op_name):
                setattr(dist, op_name, original)

        self._originals.clear()
        self._patched = False

    @staticmethod
    def _make_wrapper(
        original: Callable, op_name: str, tracer: Any
    ) -> Callable:
        """Create a wrapper function for a torch.distributed collective."""

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = f"nccl.{op_name}"

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute(keys.NCCL_COLLECTIVE, op_name)

                # Try to extract tensor info from first argument
                _annotate_tensor_info(span, args, kwargs)

                # Try to get world size and rank
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        span.set_attribute("nccl.world_size", dist.get_world_size())
                        span.set_attribute("nccl.rank", dist.get_rank())
                except Exception:
                    pass

                start = time.perf_counter()
                try:
                    result = original(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    span.set_attribute(keys.NCCL_WAIT_MS, elapsed_ms)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise

        return wrapper


def _annotate_tensor_info(span: Any, args: tuple, kwargs: dict) -> None:
    """Try to extract tensor metadata from collective arguments."""
    try:
        import torch

        tensor = None
        if args and isinstance(args[0], torch.Tensor):
            tensor = args[0]
        elif "tensor" in kwargs and isinstance(kwargs["tensor"], torch.Tensor):
            tensor = kwargs["tensor"]

        if tensor is not None:
            nbytes = tensor.nelement() * tensor.element_size()
            span.set_attribute(keys.NCCL_BYTES, nbytes)
            span.set_attribute("nccl.tensor.shape", str(list(tensor.shape)))
            span.set_attribute("nccl.tensor.dtype", str(tensor.dtype))
    except Exception:
        pass
