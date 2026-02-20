"""
llamatelemetry.inference.api - Public inference API.

Provides the single high-level entrypoint: create_engine().
One API, two worlds (llama.cpp + Transformers).
"""

from __future__ import annotations

from typing import Any, Optional

from .base import InferenceEngine
from .config import CudaInferenceConfig
from .runtime import InferenceRuntime


def create_engine(
    backend: str = "llama.cpp",
    config: Optional[CudaInferenceConfig] = None,
    telemetry: bool = True,
    scheduler: bool = False,
    multi_gpu: str = "auto",
    **kwargs: Any,
) -> InferenceRuntime:
    """Create a configured inference engine.

    This is the primary user-facing API for llamatelemetry inference.

    Args:
        backend: Backend type ("llama.cpp" or "transformers").
        config: Full inference configuration. Created from kwargs if None.
        telemetry: Enable OTel telemetry.
        scheduler: Enable request scheduler/batching.
        multi_gpu: Multi-GPU mode ("auto", "single", "tensor_parallel", "split").
        **kwargs: Additional configuration (model_path, server_url, etc.).

    Returns:
        InferenceRuntime ready for .generate() calls.

    Example:
        >>> engine = create_engine(
        ...     backend="llama.cpp",
        ...     llama_server_url="http://127.0.0.1:8090",
        ...     telemetry=True,
        ... )
        >>> engine.start()
        >>> from llamatelemetry.inference.base import InferenceRequest
        >>> result = engine.generate(InferenceRequest(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... ))
        >>> print(f"TPS: {result.tps:.1f}, TTFT: {result.ttft_ms:.1f}ms")
    """
    if config is None:
        config = CudaInferenceConfig(
            backend=backend,
            telemetry=telemetry,
            scheduler=scheduler,
            multi_gpu=multi_gpu,
            **{k: v for k, v in kwargs.items() if hasattr(CudaInferenceConfig, k)},
        )

    return InferenceRuntime.from_config(config)
