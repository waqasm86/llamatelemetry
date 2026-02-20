"""
llamatelemetry.telemetry.auto_instrument - Auto-instrumentation utilities.

Provides decorators and context managers for automatic span creation
in GenAI inference operations.

Example:
    >>> from llamatelemetry.telemetry.auto_instrument import inference_span
    >>>
    >>> with inference_span(tracer, model="gemma-3-4b") as span:
    ...     result = engine.infer(prompt)
    ...     span.set_attribute("gen_ai.usage.output_tokens", result.tokens_generated)
"""

from functools import wraps
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
import time

from ..semconv import gen_ai as gen_ai_keys
from ..otel.gen_ai_utils import build_gen_ai_span_attrs, build_span_name


def instrument_inference(
    tracer: Any,
    model_name: str = "",
    gpu_id: int = 0,
    split_mode: str = "none",
    operation: str = gen_ai_keys.OP_GENERATE_CONTENT
) -> Callable:
    """
    Decorator to auto-instrument inference methods.

    Wraps a function to automatically create OpenTelemetry GenAI spans
    with GenAI semantic attributes.

    Args:
        tracer: OpenTelemetry tracer instance
        model_name: Model name for span attributes
        gpu_id: GPU device ID
        split_mode: Multi-GPU split mode ("none", "layer", "row")
        operation: Span operation name

    Returns:
        Decorated function with automatic span creation

    Example:
        >>> @instrument_inference(tracer, model_name="gemma-3-4b")
        ... def generate(prompt: str) -> str:
        ...     return engine.infer(prompt).text
        >>>
        >>> text = generate("Hello!")  # Automatically traced
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if tracer is None:
                return func(*args, **kwargs)

            try:
                from opentelemetry.trace import SpanKind
                span_kind = SpanKind.CLIENT
            except Exception:
                span_kind = None

            span_name = build_span_name(operation, model_name or None)
            span_attrs = build_gen_ai_span_attrs(
                operation=operation,
                provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
                model=model_name or None,
            )

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                attributes=span_attrs if span_kind is not None else None,
            ) as span:
                span.set_attribute("gpu.id", str(gpu_id))
                span.set_attribute("nccl.split_mode", split_mode)

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error.type", e.__class__.__name__)
                    raise

                latency_ms = (time.time() - start_time) * 1000

                # Try to extract metrics from result
                _annotate_from_result(span, result, latency_ms)

                return result

        return wrapper
    return decorator


def _annotate_from_result(span: Any, result: Any, latency_ms: float) -> None:
    """Extract metrics from result and annotate span."""
    try:
        # Handle InferResult objects
        if hasattr(result, 'tokens_generated'):
            tokens = result.tokens_generated
            span.set_attribute(gen_ai_keys.GEN_AI_USAGE_OUTPUT_TOKENS, tokens)

        if hasattr(result, 'success'):
            span.set_attribute("result.success", result.success)

        if hasattr(result, 'error_message') and result.error_message:
            span.set_attribute("error.type", "InferenceError")

        # Handle dict results
        if isinstance(result, dict):
            if "tokens_predicted" in result:
                tokens = result["tokens_predicted"]
                span.set_attribute(gen_ai_keys.GEN_AI_USAGE_OUTPUT_TOKENS, tokens)

            if "timings" in result:
                t = result["timings"]
                if "predicted_per_second" in t:
                    span.set_attribute("llamatelemetry.predicted_per_second", t["predicted_per_second"])

    except Exception:
        pass  # Don't fail if annotation fails


@contextmanager
def inference_span(
    tracer: Any,
    operation: str = gen_ai_keys.OP_GENERATE_CONTENT,
    model: str = "",
    **attributes
):
    """
    Context manager for inference spans.

    Creates an OpenTelemetry GenAI span with GenAI attributes.
    Automatically records timing and allows adding custom attributes.

    Args:
        tracer: OpenTelemetry tracer instance
        operation: Span operation name
        model: Model name for span attributes
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span instance (or None if tracer is None)

    Example:
        >>> with inference_span(tracer, model="gemma-3-4b") as span:
        ...     result = engine.infer(prompt)
        ...     if span:
        ...         span.set_attribute("gen_ai.usage.output_tokens", result.tokens_generated)
    """
    if tracer is None:
        yield None
        return

    span_name = build_span_name(operation, model or None)
    span_attrs = build_gen_ai_span_attrs(
        operation=operation,
        provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
        model=model or None,
    )

    with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:

        for key, value in attributes.items():
            span.set_attribute(key, value)

        start_time = time.time()
        try:
            yield span
        finally:
            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute("llamatelemetry.latency_ms", latency_ms)


@contextmanager
def batch_inference_span(
    tracer: Any,
    batch_size: int,
    model: str = "",
    **attributes
):
    """
    Context manager for batch inference spans.

    Creates a parent span for batch operations with child spans
    for individual requests.

    Args:
        tracer: OpenTelemetry tracer instance
        batch_size: Number of items in batch
        model: Model name
        **attributes: Additional span attributes

    Yields:
        Tuple of (parent_span, create_child_span function)

    Example:
        >>> with batch_inference_span(tracer, batch_size=10, model="gemma") as (parent, child_fn):
        ...     for i, prompt in enumerate(prompts):
        ...         with child_fn(i):
        ...             result = engine.infer(prompt)
    """
    if tracer is None:
        yield None, lambda idx: _null_context()
        return

    span_name = build_span_name(gen_ai_keys.OP_GENERATE_CONTENT, model or None)
    span_attrs = build_gen_ai_span_attrs(
        operation=gen_ai_keys.OP_GENERATE_CONTENT,
        provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
        model=model or None,
    )

    with tracer.start_as_current_span(span_name, attributes=span_attrs) as parent:
        parent.set_attribute("batch.size", batch_size)

        for key, value in attributes.items():
            parent.set_attribute(key, value)

        def create_child(index: int):
            return tracer.start_as_current_span(f"llamatelemetry.inference.{index}")

        yield parent, create_child


@contextmanager
def _null_context():
    """Null context manager for when tracer is None."""
    yield None


def create_gen_ai_attributes(
    model: str = "",
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    gpu_id: int = 0,
    split_mode: str = "none"
) -> Dict[str, Any]:
    """
    Create standard GenAI span attributes.

    Args:
        model: Model name
        prompt_tokens: Number of prompt tokens
        output_tokens: Number of generated tokens
        latency_ms: Inference latency in milliseconds
        gpu_id: GPU device ID
        split_mode: Multi-GPU split mode

    Returns:
        Dictionary of span attributes
    """
    attrs = {
        gen_ai_keys.GEN_AI_PROVIDER_NAME: gen_ai_keys.PROVIDER_LLAMA_CPP,
        gen_ai_keys.GEN_AI_REQUEST_MODEL: model,
        gen_ai_keys.GEN_AI_USAGE_INPUT_TOKENS: prompt_tokens,
        gen_ai_keys.GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        "llamatelemetry.latency_ms": latency_ms,
        "gpu.id": str(gpu_id),
        "nccl.split_mode": split_mode,
    }

    return attrs


def annotate_span_from_result(
    span: Any,
    result: Any,
    model: str = "",
    gpu_id: int = 0,
    split_mode: str = "none"
) -> None:
    """
    Annotate span with attributes from inference result.

    Args:
        span: OpenTelemetry span
        result: InferResult or dict with inference results
        model: Model name
        gpu_id: GPU device ID
        split_mode: Multi-GPU split mode
    """
    if span is None:
        return

    try:
        span.set_attribute(gen_ai_keys.GEN_AI_PROVIDER_NAME, gen_ai_keys.PROVIDER_LLAMA_CPP)
        span.set_attribute(gen_ai_keys.GEN_AI_REQUEST_MODEL, model)
        span.set_attribute("gpu.id", str(gpu_id))
        span.set_attribute("nccl.split_mode", split_mode)

        if hasattr(result, 'tokens_generated'):
            span.set_attribute(gen_ai_keys.GEN_AI_USAGE_OUTPUT_TOKENS, result.tokens_generated)

        if hasattr(result, 'latency_ms'):
            span.set_attribute("llamatelemetry.latency_ms", result.latency_ms)

        if hasattr(result, 'success'):
            span.set_attribute("result.success", result.success)

        if hasattr(result, 'error_message') and result.error_message:
            span.set_attribute("error.type", "InferenceError")

    except Exception:
        pass


__all__ = [
    "instrument_inference",
    "inference_span",
    "batch_inference_span",
    "create_gen_ai_attributes",
    "annotate_span_from_result",
]
