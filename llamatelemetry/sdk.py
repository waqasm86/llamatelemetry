"""
llamatelemetry.sdk - User-facing convenience API for instrumented inference.

Provides simple factory functions to create instrumented backends for both
llama.cpp (GGUF) and Transformers (original) models.

Example:
    >>> import llamatelemetry
    >>> llamatelemetry.init(service_name="my-app")
    >>>
    >>> # llama.cpp backend
    >>> client = llamatelemetry.sdk.instrument_llamacpp("http://127.0.0.1:8090")
    >>> resp = client.invoke(LLMRequest(operation="chat", messages=[...]))
    >>>
    >>> # Transformers backend
    >>> client = llamatelemetry.sdk.instrument_transformers(model, tokenizer)
    >>> resp = client.invoke(LLMRequest(operation="chat", messages=[...]))
"""

from __future__ import annotations

from typing import Any, Optional

from .otel.provider import get_tracer
from .backends.base import LLMRequest, LLMResponse
from .backends.llamacpp import LlamaCppBackend
from .transformers.instrumentation import InstrumentedBackend, TransformersInstrumentorConfig
from .gpu.otel import GPUSpanEnricher


def instrument_llamacpp(
    base_url: str = "http://127.0.0.1:8090",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    gpu_device: Optional[int] = None,
    config: Optional[TransformersInstrumentorConfig] = None,
    **kwargs: Any,
) -> InstrumentedBackend:
    """Create an instrumented llama.cpp backend.

    Args:
        base_url: llama.cpp server URL.
        model: Model name for span attributes.
        api_key: Optional API key.
        gpu_device: GPU device index for enrichment.
        config: Instrumentation configuration.

    Returns:
        InstrumentedBackend wrapping llama.cpp with OTel tracing.

    Example:
        >>> client = instrument_llamacpp("http://127.0.0.1:8090")
        >>> from llamatelemetry.backends.base import LLMRequest
        >>> resp = client.invoke(LLMRequest(
        ...     operation="chat",
        ...     model="gemma-3-1b",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... ))
    """
    tracer = get_tracer("llamatelemetry.sdk")
    backend = LlamaCppBackend(base_url=base_url, api_key=api_key, **kwargs)
    gpu = GPUSpanEnricher(device_index=gpu_device)
    return InstrumentedBackend(
        backend=backend,
        tracer=tracer,
        gpu=gpu,
        config=config,
    )


def instrument_transformers(
    model: Any,
    tokenizer: Any,
    device: Optional[str] = None,
    autocast_dtype: Optional[str] = None,
    gpu_device: Optional[int] = None,
    config: Optional[TransformersInstrumentorConfig] = None,
    **kwargs: Any,
) -> InstrumentedBackend:
    """Create an instrumented Transformers backend.

    Args:
        model: HuggingFace model (AutoModelForCausalLM, etc.).
        tokenizer: Associated tokenizer.
        device: Device string ("cuda", "cuda:0", "cpu").
        autocast_dtype: Autocast dtype ("fp16", "bf16").
        gpu_device: GPU device index for enrichment.
        config: Instrumentation configuration.

    Returns:
        InstrumentedBackend wrapping Transformers with OTel tracing.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
        >>> client = instrument_transformers(model, tokenizer)
    """
    from .transformers.backend import TransformersBackend

    tracer = get_tracer("llamatelemetry.sdk")
    backend = TransformersBackend(
        model=model,
        tokenizer=tokenizer,
        device=device,
        autocast_dtype=autocast_dtype,
        **kwargs,
    )
    gpu = GPUSpanEnricher(device_index=gpu_device)
    return InstrumentedBackend(
        backend=backend,
        tracer=tracer,
        gpu=gpu,
        config=config,
    )
