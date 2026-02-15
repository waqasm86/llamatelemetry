"""
llamatelemetry.llama.phases - Prefill/decode span hierarchy + trace_request().

Creates the span tree:
    llm.request
        llm.phase.prefill
        llm.phase.decode
And emits associated metrics.
"""

import time
from contextlib import contextmanager
from typing import Any, Optional

from ..otel.provider import get_tracer, get_meter
from ..semconv import keys


@contextmanager
def trace_request(
    request_id: str = "",
    model: str = "",
    stream: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
):
    """
    Context manager that creates the ``llm.request`` -> ``llm.phase.*`` span hierarchy.

    Usage::

        with trace_request(request_id="r1", model="gemma-3-1b") as req:
            # ... perform inference ...
            req.set_completion_tokens(42)

    Args:
        request_id: Unique request ID.
        model: Model name.
        stream: Whether the request is streaming.
        prompt_tokens: Number of prompt tokens (set upfront if known).
        completion_tokens: Number of completion tokens (set upfront if known).
    """
    tracer = get_tracer("llamatelemetry.llama")
    meter = get_meter("llamatelemetry.llama")

    _duration_hist = meter.create_histogram(
        name="llm.request.duration_ms",
        description="LLM request latency",
        unit="ms",
    )
    _tokens_counter = meter.create_counter(
        name="llm.tokens.total",
        description="Total tokens generated",
        unit="tokens",
    )

    handle = _RequestHandle(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    start = time.perf_counter()

    with tracer.start_as_current_span("llm.request") as root_span:
        root_span.set_attribute(keys.LLM_SYSTEM, "llamatelemetry")
        root_span.set_attribute(keys.LLM_MODEL, model)
        root_span.set_attribute(keys.LLM_STREAM, stream)
        if request_id:
            root_span.set_attribute(keys.REQUEST_ID, request_id)

        # Prefill child span
        with tracer.start_as_current_span("llm.phase.prefill") as pfill:
            pfill.set_attribute(keys.LLM_PHASE, "prefill")
            pfill.set_attribute(keys.LLM_INPUT_TOKENS, prompt_tokens)

        try:
            yield handle
        except Exception as exc:
            root_span.set_attribute(keys.LLM_ERROR, str(exc))
            root_span.record_exception(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_out = handle.completion_tokens

            # Decode child span
            with tracer.start_as_current_span("llm.phase.decode") as dec:
                dec.set_attribute(keys.LLM_PHASE, "decode")
                dec.set_attribute(keys.LLM_OUTPUT_TOKENS, total_out)
                tps = (total_out / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
                dec.set_attribute(keys.LLM_TOKENS_PER_SECOND, tps)

            root_span.set_attribute(keys.LLM_REQUEST_DURATION_MS, elapsed_ms)
            root_span.set_attribute(keys.LLM_OUTPUT_TOKENS, total_out)
            root_span.set_attribute(keys.LLM_INPUT_TOKENS, handle.prompt_tokens)
            root_span.set_attribute(keys.LLM_TOKENS_TOTAL, handle.prompt_tokens + total_out)

            attrs = {keys.LLM_MODEL: model}
            _duration_hist.record(elapsed_ms, attrs)
            _tokens_counter.add(total_out, attrs)


class _RequestHandle:
    """Mutable container so callers can set token counts inside the CM."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def set_completion_tokens(self, n: int) -> None:
        self.completion_tokens = n

    def set_prompt_tokens(self, n: int) -> None:
        self.prompt_tokens = n
