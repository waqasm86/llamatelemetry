"""
llamatelemetry.cuda.optim.autocast - Autocast dtype management.

Controls mixed-precision inference (FP16/BF16) with device-aware defaults.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional


class AutocastManager:
    """Manages autocast context for mixed-precision inference.

    Example:
        >>> manager = AutocastManager(dtype="fp16")
        >>> with manager.context():
        ...     output = model(input_ids)
    """

    def __init__(self, dtype: Optional[str] = "fp16", device_type: str = "cuda"):
        """Initialize autocast manager.

        Args:
            dtype: Target dtype ("fp16", "bf16", None for disabled).
            device_type: Device type for autocast.
        """
        self._dtype_str = dtype
        self._device_type = device_type

    def context(self) -> Any:
        """Return an autocast context manager."""
        if self._dtype_str is None:
            return contextlib.nullcontext()

        try:
            import torch

            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map.get(self._dtype_str)
            if dtype is None:
                return contextlib.nullcontext()

            return torch.autocast(device_type=self._device_type, dtype=dtype)
        except ImportError:
            return contextlib.nullcontext()

    @property
    def is_enabled(self) -> bool:
        return self._dtype_str is not None

    @staticmethod
    def detect_best_dtype() -> str:
        """Detect the best autocast dtype for the current GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 8:
                    return "bf16"
            return "fp16"
        except ImportError:
            return "fp16"
