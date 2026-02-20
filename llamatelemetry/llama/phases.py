"""
llamatelemetry.llama.phases - Prefill/decode span hierarchy + trace_request().

Creates the span tree:
    {gen_ai.operation.name} {gen_ai.request.model}
        llamatelemetry.phase.prefill
        llamatelemetry.phase.decode
And emits associated GenAI metrics.
"""

import time
from contextlib import contextmanager
from typing import Any, Optional

from ..otel.provider import get_tracer
from ..otel.gen_ai_metrics import get_gen_ai_metrics
from ..otel.gen_ai_utils import build_gen_ai_span_attrs, build_span_name
from ..semconv import gen_ai as gen_ai_keys
from ..semconv.gen_ai_builder import build_gen_ai_attrs_from_request, build_gen_ai_attrs_from_response


@contextmanager
def trace_request(
    request_id: str = "",
    model: str = "",
    stream: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
):
    """
    Context manager that creates the GenAI root span with prefill/decode children.

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

    handle = _RequestHandle(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    start = time.perf_counter()

    span_name = build_span_name(gen_ai_keys.OP_CHAT, model or None)
    span_attrs = build_gen_ai_span_attrs(
        operation=gen_ai_keys.OP_CHAT,
        provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
        model=model or None,
    )

    try:
        from opentelemetry.trace import SpanKind
        span_kind = SpanKind.CLIENT
    except Exception:
        span_kind = None

    with tracer.start_as_current_span(
        span_name,
        kind=span_kind,
        attributes=span_attrs if span_kind is not None else None,
    ) as root_span:
        if request_id:
            root_span.set_attribute("request.id", request_id)

        # gen_ai.* request attributes
        gen_ai_req = build_gen_ai_attrs_from_request(
            model=model,
            operation=gen_ai_keys.OP_CHAT,
            provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
            stream=stream,
        )
        for k, v in gen_ai_req.items():
            root_span.set_attribute(k, v)

        # Prefill child span
        with tracer.start_as_current_span("llamatelemetry.phase.prefill") as pfill:
            pass

        try:
            yield handle
        except Exception as exc:
            root_span.set_attribute("error.type", exc.__class__.__name__)
            root_span.record_exception(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_out = handle.completion_tokens

            # Decode child span
            with tracer.start_as_current_span("llamatelemetry.phase.decode") as dec:
                pass

            # gen_ai.* response attributes
            gen_ai_resp = build_gen_ai_attrs_from_response(
                input_tokens=handle.prompt_tokens,
                output_tokens=total_out,
            )
            for k, v in gen_ai_resp.items():
                root_span.set_attribute(k, v)

            metrics = get_gen_ai_metrics()
            base_attrs = build_gen_ai_span_attrs(
                operation=gen_ai_keys.OP_CHAT,
                provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
                model=model or None,
            )
            metrics.record_client_operation_duration(elapsed_ms / 1000.0, base_attrs)
            if handle.prompt_tokens:
                metrics.record_client_token_usage(
                    handle.prompt_tokens, gen_ai_keys.TOKEN_INPUT, base_attrs
                )
            if total_out:
                metrics.record_client_token_usage(
                    total_out, gen_ai_keys.TOKEN_OUTPUT, base_attrs
                )


class _RequestHandle:
    """Mutable container so callers can set token counts inside the CM."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def set_completion_tokens(self, n: int) -> None:
        self.completion_tokens = n

    def set_prompt_tokens(self, n: int) -> None:
        self.prompt_tokens = n
