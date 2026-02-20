"""
llamatelemetry.cuda.optim - CUDA optimization modules.

Policy-driven optimization utilities for Transformers inference:
autocast, torch.compile, CUDA graphs, flash attention, paged attention,
and kernel fusion.
"""

from .autocast import AutocastManager
from .compile import CompileManager
from .cudagraphs import CudaGraphManager
from .flash_attn import FlashAttnManager
from .paged_attention import PagedAttentionManager
from .kernel_fusion import KernelFusionManager
