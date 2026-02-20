"""
llamatelemetry.cuda.optim.flash_attn - FlashAttention policy-driven management.

Wraps the existing flash attention helper into a policy-driven interface.
"""

from __future__ import annotations

from typing import Any, Optional


class FlashAttnManager:
    """Manages FlashAttention availability and configuration.

    Example:
        >>> manager = FlashAttnManager()
        >>> if manager.is_available():
        ...     output = manager.forward(q, k, v)
    """

    def __init__(self, enabled: bool = True):
        """Initialize FlashAttention manager.

        Args:
            enabled: Whether FlashAttention is enabled.
        """
        self._enabled = enabled
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if FlashAttention is available and enabled."""
        if not self._enabled:
            return False

        if self._available is None:
            self._available = self._check_available()
        return self._available

    def forward(self, q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
        """Run flash attention forward pass.

        Falls back to standard attention if FlashAttention is unavailable.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.

        Returns:
            Attention output tensor.
        """
        if self.is_available():
            try:
                from flash_attn import flash_attn_func
                return flash_attn_func(q, k, v, **kwargs)
            except Exception:
                pass

        # Fallback to SDPA
        return self._sdpa_fallback(q, k, v)

    @staticmethod
    def _check_available() -> bool:
        """Check if flash_attn package is importable."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    @staticmethod
    def _sdpa_fallback(q: Any, k: Any, v: Any) -> Any:
        """Fallback to PyTorch scaled dot product attention."""
        try:
            import torch
            import torch.nn.functional as F
            return F.scaled_dot_product_attention(q, k, v)
        except (ImportError, Exception):
            raise RuntimeError("Neither FlashAttention nor PyTorch SDPA available")
