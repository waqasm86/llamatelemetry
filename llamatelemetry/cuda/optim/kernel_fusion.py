"""
llamatelemetry.cuda.optim.kernel_fusion - Kernel fusion management.

Registry for fused CUDA kernels that combine multiple operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class FusedKernel:
    """A registered fused kernel.

    Attributes:
        name: Kernel name.
        fn: Kernel function.
        ops_fused: List of operation names that are fused.
        requires_sm: Minimum SM version required.
    """

    name: str
    fn: Callable
    ops_fused: list
    requires_sm: int = 70


class KernelFusionManager:
    """Registry and manager for fused CUDA kernels.

    Example:
        >>> manager = KernelFusionManager()
        >>> manager.register("fused_gelu_dropout", fn, ops_fused=["gelu", "dropout"])
        >>> if manager.has("fused_gelu_dropout"):
        ...     output = manager.execute("fused_gelu_dropout", input_tensor)
    """

    def __init__(self):
        self._kernels: Dict[str, FusedKernel] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        ops_fused: Optional[list] = None,
        requires_sm: int = 70,
    ) -> None:
        """Register a fused kernel.

        Args:
            name: Kernel name.
            fn: Kernel function.
            ops_fused: Operations fused by this kernel.
            requires_sm: Minimum SM version.
        """
        self._kernels[name] = FusedKernel(
            name=name,
            fn=fn,
            ops_fused=ops_fused or [],
            requires_sm=requires_sm,
        )

    def has(self, name: str) -> bool:
        """Check if a fused kernel is registered."""
        return name in self._kernels

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered fused kernel.

        Args:
            name: Kernel name.
            *args, **kwargs: Arguments for the kernel.

        Returns:
            Kernel output.
        """
        if name not in self._kernels:
            raise KeyError(f"Fused kernel '{name}' not registered")
        return self._kernels[name].fn(*args, **kwargs)

    def list_kernels(self) -> list:
        """List all registered fused kernels."""
        return list(self._kernels.keys())
