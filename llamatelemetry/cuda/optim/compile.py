"""
llamatelemetry.cuda.optim.compile - torch.compile management.

Controls torch.compile optimization with mode selection and guard management.
"""

from __future__ import annotations

from typing import Any, Optional


class CompileManager:
    """Manages torch.compile for model optimization.

    Example:
        >>> manager = CompileManager(mode="reduce-overhead")
        >>> compiled_model = manager.compile(model)
    """

    def __init__(
        self,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: Optional[bool] = None,
        backend: str = "inductor",
    ):
        """Initialize compile manager.

        Args:
            mode: Compile mode ("default", "reduce-overhead", "max-autotune").
            fullgraph: Require full graph compilation.
            dynamic: Enable dynamic shapes.
            backend: Compilation backend.
        """
        self._mode = mode
        self._fullgraph = fullgraph
        self._dynamic = dynamic
        self._backend = backend

    def compile(self, model: Any) -> Any:
        """Apply torch.compile to a model.

        Args:
            model: PyTorch model.

        Returns:
            Compiled model (or original if torch.compile unavailable).
        """
        try:
            import torch
            return torch.compile(
                model,
                mode=self._mode,
                fullgraph=self._fullgraph,
                dynamic=self._dynamic,
                backend=self._backend,
            )
        except (ImportError, Exception):
            return model

    @staticmethod
    def is_available() -> bool:
        """Check if torch.compile is available."""
        try:
            import torch
            return hasattr(torch, "compile")
        except ImportError:
            return False
