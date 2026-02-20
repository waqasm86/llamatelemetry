"""
llamatelemetry.cuda.optim.cudagraphs - CUDA Graph capture and replay.

Captures stable decode steps as CUDA graphs for reduced kernel launch overhead.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple


class CudaGraphManager:
    """Manages CUDA graph capture and replay for decode steps.

    CUDA graphs capture a sequence of kernel launches and replay them
    with minimal overhead, ideal for repetitive decode steps.

    Example:
        >>> manager = CudaGraphManager()
        >>> # Warmup
        >>> manager.warmup(decode_fn, sample_input)
        >>> # Capture
        >>> manager.capture(decode_fn, sample_input)
        >>> # Replay
        >>> output = manager.replay(new_input)
    """

    def __init__(self, pool_size: int = 1):
        """Initialize CUDA graph manager.

        Args:
            pool_size: Number of graph pools to maintain.
        """
        self._pool_size = pool_size
        self._graphs: Dict[str, Any] = {}
        self._static_inputs: Dict[str, Any] = {}
        self._static_outputs: Dict[str, Any] = {}

    def warmup(self, fn: Callable, *args: Any, num_warmup: int = 3) -> None:
        """Warmup the function before capture.

        Args:
            fn: Function to warmup.
            *args: Sample arguments.
            num_warmup: Number of warmup iterations.
        """
        try:
            import torch
            for _ in range(num_warmup):
                fn(*args)
            torch.cuda.synchronize()
        except ImportError:
            pass

    def capture(self, key: str, fn: Callable, *args: Any) -> bool:
        """Capture a CUDA graph.

        Args:
            key: Graph identifier.
            fn: Function to capture.
            *args: Arguments for the function.

        Returns:
            True if capture succeeded.
        """
        try:
            import torch

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn(*args)

            self._graphs[key] = graph
            self._static_outputs[key] = output
            return True
        except (ImportError, Exception):
            return False

    def replay(self, key: str) -> Optional[Any]:
        """Replay a captured CUDA graph.

        Args:
            key: Graph identifier.

        Returns:
            Static output from the graph, or None if not captured.
        """
        if key not in self._graphs:
            return None

        try:
            self._graphs[key].replay()
            return self._static_outputs.get(key)
        except Exception:
            return None

    def has_graph(self, key: str) -> bool:
        """Check if a graph is captured."""
        return key in self._graphs

    def clear(self) -> None:
        """Clear all captured graphs."""
        self._graphs.clear()
        self._static_inputs.clear()
        self._static_outputs.clear()

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA graphs are available."""
        try:
            import torch
            return torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")
        except ImportError:
            return False
