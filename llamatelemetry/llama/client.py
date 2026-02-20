"""
llamatelemetry.llama.client - LlamaCppClient + wrap_openai_client.

Re-exports the full LlamaCppClient from extras/api and adds OTel instrumentation.
"""

from typing import Any

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


def wrap_openai_client(client: Any) -> Any:
    """
    Instrument an OpenAI-compatible client to emit OTel spans.

    Monkey-patches ``client.chat.completions.create`` so every call
    produces an ``llm.request`` root span with child spans for prefill/decode.

    Args:
        client: An OpenAI-compatible client (e.g. ``LlamaCppClient`` or
                ``openai.OpenAI``).

    Returns:
        The same client, now instrumented.
    """
    from ..otel.provider import get_tracer
    from ..semconv import keys

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

        with tracer.start_as_current_span("llm.request") as span:
            # Legacy llm.* attributes
            span.set_attribute(keys.LLM_SYSTEM, "llamatelemetry")
            span.set_attribute(keys.LLM_MODEL, str(model))
            span.set_attribute(keys.LLM_STREAM, stream)

            # gen_ai.* request attributes
            gen_ai_req = build_gen_ai_attrs_from_request(
                model=str(model),
                operation=gen_ai_keys.OP_CHAT,
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

                # Annotate response - legacy llm.*
                input_tokens = 0
                output_tokens = 0
                if hasattr(result, "usage") and result.usage is not None:
                    input_tokens = getattr(result.usage, "prompt_tokens", 0)
                    output_tokens = getattr(result.usage, "completion_tokens", 0)
                    span.set_attribute(keys.LLM_INPUT_TOKENS, input_tokens)
                    span.set_attribute(keys.LLM_OUTPUT_TOKENS, output_tokens)
                    span.set_attribute(
                        keys.LLM_TOKENS_TOTAL,
                        getattr(result.usage, "total_tokens", 0),
                    )

                finish_reason = None
                if hasattr(result, "choices") and result.choices:
                    finish_reason = getattr(result.choices[0], "finish_reason", None)
                    if finish_reason:
                        span.set_attribute(keys.LLM_FINISH_REASON, finish_reason)

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

                return result
            except Exception as exc:
                span.set_attribute(keys.LLM_ERROR, str(exc))
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
