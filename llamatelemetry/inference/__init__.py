"""
llamatelemetry.inference - CUDA inference subsystem.

Enhanced inference capabilities including:
- Unified InferenceEngine protocol (llama.cpp + Transformers)
- Performance primitives (TTFT, TPOT, TPS, VRAM peak)
- Request scheduling and batching
- KV cache management
- FlashAttention integration
- Benchmark harness

Example:
    >>> from llamatelemetry.inference.api import create_engine
    >>> engine = create_engine(backend="llama.cpp", llama_server_url="http://127.0.0.1:8090")
    >>> engine.start()
    >>> result = engine.generate(InferenceRequest(messages=[{"role": "user", "content": "Hi"}]))
"""

# Legacy imports (compatibility layer) - torch optional
try:
    from .flash_attn import (
        FlashAttentionConfig,
        enable_flash_attention,
        flash_attention_forward,
    )

    from .kv_cache import (
        KVCache,
        KVCacheConfig,
        PagedKVCache,
        optimize_kv_cache,
    )

    from .batch import (
        BatchInferenceOptimizer,
        ContinuousBatching,
        batch_inference_optimized,
    )
except ImportError:
    # torch not installed; legacy inference APIs unavailable
    FlashAttentionConfig = None  # type: ignore[assignment,misc]
    enable_flash_attention = None  # type: ignore[assignment]
    flash_attention_forward = None  # type: ignore[assignment]
    KVCache = None  # type: ignore[assignment,misc]
    KVCacheConfig = None  # type: ignore[assignment,misc]
    PagedKVCache = None  # type: ignore[assignment,misc]
    optimize_kv_cache = None  # type: ignore[assignment]
    BatchInferenceOptimizer = None  # type: ignore[assignment,misc]
    ContinuousBatching = None  # type: ignore[assignment,misc]
    batch_inference_optimized = None  # type: ignore[assignment]

# Inference contract (current)
from .base import InferenceRequest, InferenceResult, InferenceEngine
from .types import SamplingParams, BatchConstraints, DeviceConfig
from .config import CudaInferenceConfig
from .events import InferenceEvents, EventRecorder
from .api import create_engine

__all__ = [
    # Inference contract
    'InferenceRequest',
    'InferenceResult',
    'InferenceEngine',
    'SamplingParams',
    'BatchConstraints',
    'DeviceConfig',
    'CudaInferenceConfig',
    'InferenceEvents',
    'EventRecorder',
    'create_engine',

    # FlashAttention
    'FlashAttentionConfig',
    'enable_flash_attention',
    'flash_attention_forward',

    # KV-cache
    'KVCache',
    'KVCacheConfig',
    'PagedKVCache',
    'optimize_kv_cache',

    # Batch optimization
    'BatchInferenceOptimizer',
    'ContinuousBatching',
    'batch_inference_optimized',
]
