"""
llamatelemetry.cuda.policy - CUDA optimization policy.

Single configuration object for controlling CUDA-specific optimizations:
autocast, torch.compile, CUDA graphs, flash attention, prefill chunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CudaPolicy:
    """CUDA optimization policy for Transformers inference.

    Controls which GPU optimizations are enabled during inference.
    This is the inference equivalent of the OTel provider config.

    Attributes:
        autocast_dtype: Autocast dtype ("fp16", "bf16", None for disabled).
        use_torch_compile: Enable torch.compile for model optimization.
        use_cuda_graphs: Enable CUDA graph capture for decode steps.
        use_flash_attn: Enable FlashAttention if available.
        use_sdpa: Enable Scaled Dot Product Attention (PyTorch native).
        prefill_chunk_size: Chunk size for prefill splitting (0 = no chunking).
        max_batch_tokens: Maximum tokens per batch.
        enable_tf32: Enable TF32 for matmul (Ampere+).
        memory_efficient: Enable memory-efficient attention variants.

    Example:
        >>> policy = CudaPolicy(
        ...     autocast_dtype="fp16",
        ...     use_flash_attn=True,
        ...     use_cuda_graphs=True,
        ... )
    """

    autocast_dtype: Optional[str] = "fp16"
    use_torch_compile: bool = False
    use_cuda_graphs: bool = False
    use_flash_attn: bool = True
    use_sdpa: bool = True
    prefill_chunk_size: int = 512
    max_batch_tokens: int = 4096
    enable_tf32: bool = True
    memory_efficient: bool = False

    def apply(self) -> None:
        """Apply this policy's global settings (e.g. TF32, backends)."""
        try:
            import torch
            if self.enable_tf32 and torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            if self.use_sdpa:
                # PyTorch 2.0+ SDPA is default
                pass

        except ImportError:
            pass

    @property
    def attention_backend(self) -> str:
        """Determine the attention backend to use."""
        if self.use_flash_attn:
            return "flash_attn"
        if self.use_sdpa:
            return "sdpa"
        return "eager"

    @classmethod
    def for_t4(cls) -> "CudaPolicy":
        """Optimized policy for Tesla T4 (SM 7.5)."""
        return cls(
            autocast_dtype="fp16",
            use_torch_compile=False,  # Limited benefit on T4
            use_cuda_graphs=True,
            use_flash_attn=True,
            use_sdpa=True,
            enable_tf32=False,  # T4 doesn't support TF32
            prefill_chunk_size=256,
        )

    @classmethod
    def for_a100(cls) -> "CudaPolicy":
        """Optimized policy for A100 (SM 8.0)."""
        return cls(
            autocast_dtype="bf16",
            use_torch_compile=True,
            use_cuda_graphs=True,
            use_flash_attn=True,
            use_sdpa=True,
            enable_tf32=True,
            prefill_chunk_size=1024,
            max_batch_tokens=8192,
        )
