"""
llamatelemetry.llama.client - LlamaCppClient + wrap_openai_client.

Re-exports the full LlamaCppClient from extras/api and adds OTel instrumentation.
"""

from typing import Any
import time

# Re-export all dataclasses and the client from the original api module
from ..api.client import (
    StopType,
    Message,
    Choice,
    Usage,
    Timings,
    CompletionResponse,
    EmbeddingData,
    EmbeddingsResponse,
    RerankResult,
    RerankResponse,
    TokenizeResponse,
    ModelInfo,
    SlotInfo,
    HealthStatus,
    LoraAdapter,
    LlamaCppClient,
    ChatCompletionsAPI,
    EmbeddingsClientAPI,
    ModelsClientAPI,
    SlotsClientAPI,
    LoraClientAPI,
)


def wrap_openai_client(client: Any, strict_operation_names: bool = True) -> Any:
    """
    Instrument an OpenAI-compatible client to emit OTel spans.

    Monkey-patches ``client.chat.completions.create`` so every call
    produces a GenAI root span with child spans for prefill/decode.

    Args:
        client: An OpenAI-compatible client (e.g. ``LlamaCppClient`` or
                ``openai.OpenAI``).

    Returns:
        The same client, now instrumented.
    """
    from ..otel.provider import get_tracer
    from ..otel.gen_ai_metrics import get_gen_ai_metrics
    from ..otel.gen_ai_utils import build_gen_ai_span_attrs, build_span_name, parse_server_address

    tracer = get_tracer("llamatelemetry.llama")

    # Detect the chat.completions.create path
    _has_chat = hasattr(client, "chat") and hasattr(getattr(client, "chat"), "completions")
    if not _has_chat:
        return client

    original_create = client.chat.completions.create

    def _instrumented_create(*args: Any, **kwargs: Any) -> Any:
        from ..semconv import gen_ai as gen_ai_keys
        from ..semconv.gen_ai_builder import (
            build_gen_ai_attrs_from_request,
            build_gen_ai_attrs_from_response,
        )

        model = kwargs.get("model", "")
        stream = kwargs.get("stream", False)
        server_address, server_port = parse_server_address(getattr(client, "base_url", None))
        start_time = time.perf_counter()

        try:
            from opentelemetry.trace import SpanKind
            span_kind = SpanKind.CLIENT
        except Exception:
            span_kind = None

        operation = gen_ai_keys.normalize_operation(
            gen_ai_keys.OP_CHAT,
            strict=strict_operation_names,
        )
        span_name = build_span_name(operation, model)
        span_attrs = build_gen_ai_span_attrs(
            operation=operation,
            provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
            model=str(model) if model else None,
            server_address=server_address,
            server_port=server_port,
        )

        with tracer.start_as_current_span(
            span_name,
            kind=span_kind,
            attributes=span_attrs if span_kind is not None else None,
        ) as span:

            # gen_ai.* request attributes
            gen_ai_req = build_gen_ai_attrs_from_request(
                model=str(model),
                operation=operation,
                provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                max_tokens=kwargs.get("max_tokens"),
                stream=stream,
            )
            for k, v in gen_ai_req.items():
                span.set_attribute(k, v)

            try:
                result = original_create(*args, **kwargs)

                # Annotate response
                input_tokens = 0
                output_tokens = 0
                if hasattr(result, "usage") and result.usage is not None:
                    input_tokens = getattr(result.usage, "prompt_tokens", 0)
                    output_tokens = getattr(result.usage, "completion_tokens", 0)

                finish_reason = None
                if hasattr(result, "choices") and result.choices:
                    finish_reason = getattr(result.choices[0], "finish_reason", None)

                # gen_ai.* response attributes
                gen_ai_resp = build_gen_ai_attrs_from_response(
                    response_id=getattr(result, "id", None),
                    response_model=getattr(result, "model", None),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reasons=[finish_reason] if finish_reason else None,
                )
                for k, v in gen_ai_resp.items():
                    span.set_attribute(k, v)

                # Metrics
                metrics = get_gen_ai_metrics()
                base_attrs = build_gen_ai_span_attrs(
                    operation=operation,
                    provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
                    model=str(model) if model else None,
                    response_model=getattr(result, "model", None),
                    server_address=server_address,
                    server_port=server_port,
                )
                metrics.record_client_operation_duration(
                    time.perf_counter() - start_time,
                    base_attrs,
                )
                if input_tokens:
                    metrics.record_client_token_usage(input_tokens, gen_ai_keys.TOKEN_INPUT, base_attrs)
                if output_tokens:
                    metrics.record_client_token_usage(output_tokens, gen_ai_keys.TOKEN_OUTPUT, base_attrs)

                return result
            except Exception as exc:
                span.set_attribute("error.type", exc.__class__.__name__)
                span.record_exception(exc)
                raise

    client.chat.completions.create = _instrumented_create
    return client


__all__ = [
    "StopType",
    "Message",
    "Choice",
    "Usage",
    "Timings",
    "CompletionResponse",
    "EmbeddingData",
    "EmbeddingsResponse",
    "RerankResult",
    "RerankResponse",
    "TokenizeResponse",
    "ModelInfo",
    "SlotInfo",
    "HealthStatus",
    "LoraAdapter",
    "LlamaCppClient",
    "ChatCompletionsAPI",
    "EmbeddingsClientAPI",
    "ModelsClientAPI",
    "SlotsClientAPI",
    "LoraClientAPI",
    "wrap_openai_client",
]
