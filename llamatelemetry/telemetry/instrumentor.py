"""
llamatelemetry.telemetry.instrumentor - OpenTelemetry instrumentor for LlamaCppClient.

Follows opentelemetry-python-contrib patterns for auto-instrumentation.

Example:
    >>> from llamatelemetry.telemetry import LlamaCppClientInstrumentor
    >>>
    >>> # Auto-instrument all LlamaCppClient instances
    >>> LlamaCppClientInstrumentor().instrument()
    >>>
    >>> # Now all client calls are traced
    >>> client = LlamaCppClient()
    >>> response = client.complete("Hello")  # Automatically traced
"""

from dataclasses import dataclass
from typing import Collection, Optional, Any, Dict, List, Tuple
import time


@dataclass
class LlamaCppClientInstrumentorConfig:
    """Configuration for LlamaCppClient auto-instrumentation."""

    record_content: bool = False
    record_tools: bool = False
    record_events: bool = False
    emit_metrics: bool = True
    record_content_max_chars: int = 2000


class LlamaCppClientInstrumentor:
    """
    Auto-instrumentor for LlamaCppClient.

    Wraps LlamaCppClient methods to automatically create OpenTelemetry
    spans with GenAI semantic attributes.

    Example:
        >>> # Instrument at application startup
        >>> instrumentor = LlamaCppClientInstrumentor()
        >>> instrumentor.instrument()
        >>>
        >>> # All client calls are now traced
        >>> client = LlamaCppClient(base_url="http://localhost:8080")
        >>> response = client.chat.create(messages=[...])
        >>>
        >>> # Uninstrument when done
        >>> instrumentor.uninstrument()
    """

    _original_request = None
    _original_chat_create = None
    _original_complete = None
    _instrumented = False
    _tracer = None
    _config = None

    def __init__(self, config: Optional[LlamaCppClientInstrumentorConfig] = None) -> None:
        self._config = config or LlamaCppClientInstrumentorConfig()

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return required packages for instrumentation."""
        return ["llamatelemetry >= 1.2.0"]

    def instrument(
        self,
        tracer_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
    ) -> None:
        """
        Instrument LlamaCppClient.

        Args:
            tracer_provider: OpenTelemetry TracerProvider (uses global if None)
            meter_provider: OpenTelemetry MeterProvider (uses global if None)
        """
        if self._instrumented:
            return

        try:
            from ..api.client import LlamaCppClient
        except ImportError:
            return

        # Get tracer
        if tracer_provider is None:
            try:
                from opentelemetry import trace
                tracer_provider = trace.get_tracer_provider()
            except ImportError:
                return

        self._tracer = tracer_provider.get_tracer(
            "llamatelemetry.instrumentor",
            "1.2.0"
        )

        if meter_provider is not None:
            try:
                from opentelemetry import metrics as otel_metrics
                otel_metrics.set_meter_provider(meter_provider)
            except Exception:
                pass

        # Store original methods
        self._original_request = LlamaCppClient._request

        # Wrap _request method (lowest level, catches all API calls)
        config = self._config
        def instrumented_request(
            self_client,
            method: str,
            endpoint: str,
            json_data=None,
            params=None,
            stream=False
        ):
            tracer = LlamaCppClientInstrumentor._tracer
            if tracer is None:
                return LlamaCppClientInstrumentor._original_request(
                    self_client, method, endpoint, json_data, params, stream
                )

            # Determine operation name from endpoint
            operation, is_gen_ai = _endpoint_to_operation(endpoint, method)

            server_address, server_port = _parse_server(self_client)
            model = _extract_model(json_data, params)

            try:
                from opentelemetry.trace import SpanKind
                span_kind = SpanKind.CLIENT
            except Exception:
                span_kind = None

            if is_gen_ai:
                from ..otel.gen_ai_utils import build_gen_ai_span_attrs, build_span_name
                span_name = build_span_name(operation, model)
                span_attrs = build_gen_ai_span_attrs(
                    operation=operation,
                    provider=_provider_name(),
                    model=model,
                    server_address=server_address,
                    server_port=server_port,
                )
            else:
                span_name = operation
                span_attrs = {}
                if server_address:
                    span_attrs["server.address"] = server_address
                if server_port is not None:
                    span_attrs["server.port"] = server_port

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                attributes=span_attrs if span_kind is not None else None,
            ) as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", f"{self_client.base_url}{endpoint}")
                span.set_attribute("llamatelemetry.endpoint", endpoint)

                gen_ai_req_attrs: Dict[str, Any] = {}
                if is_gen_ai and json_data:
                    gen_ai_req_attrs = _build_request_attrs(operation, json_data, stream, model)
                    for k, v in gen_ai_req_attrs.items():
                        span.set_attribute(k, v)

                    if config.record_tools:
                        tool_attrs = _build_tool_attrs(json_data)
                        for k, v in tool_attrs.items():
                            span.set_attribute(k, v)

                    if config.record_content:
                        content_attrs = _build_content_attrs(
                            json_data,
                            record_content_max_chars=config.record_content_max_chars,
                        )
                        for k, v in content_attrs.items():
                            span.set_attribute(k, v)

                start_time = time.perf_counter()

                try:
                    result = LlamaCppClientInstrumentor._original_request(
                        self_client, method, endpoint, json_data, params, stream
                    )

                    latency_s = time.perf_counter() - start_time

                    if is_gen_ai and isinstance(result, dict):
                        gen_ai_resp_attrs = _build_response_attrs(result)
                        for k, v in gen_ai_resp_attrs.items():
                            span.set_attribute(k, v)

                        if config.record_events:
                            event_attrs = {}
                            event_attrs.update(gen_ai_req_attrs)
                            event_attrs.update(gen_ai_resp_attrs)
                            if config.record_content:
                                output_attrs = _build_output_content_attrs(
                                    result,
                                    record_content_max_chars=config.record_content_max_chars,
                                )
                                event_attrs.update(output_attrs)
                            if config.record_tools:
                                event_attrs.update(_build_tool_attrs(json_data))
                            span.add_event(
                                "gen_ai.client.inference.operation.details",
                                attributes=event_attrs,
                            )

                        if config.emit_metrics:
                            _emit_metrics(
                                operation=operation,
                                model=model,
                                result=result,
                                latency_s=latency_s,
                                server_address=server_address,
                                server_port=server_port,
                            )

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error.type", e.__class__.__name__)
                    raise

        LlamaCppClient._request = instrumented_request
        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove instrumentation."""
        if not self._instrumented:
            return

        try:
            from ..api.client import LlamaCppClient

            if self._original_request:
                LlamaCppClient._request = self._original_request

        except ImportError:
            pass

        self._instrumented = False
        self._tracer = None

    def is_instrumented(self) -> bool:
        """Check if instrumentation is active."""
        return self._instrumented


def _endpoint_to_operation(endpoint: str, method: str) -> Tuple[str, bool]:
    """Convert API endpoint to span operation name."""
    endpoint = endpoint.strip("/")

    # Map common endpoints to semantic operation names
    endpoint_map = {
        "completion": "text_completion",
        "v1/completions": "text_completion",
        "chat/completions": "chat",
        "v1/chat/completions": "chat",
        "embeddings": "embeddings",
        "v1/embeddings": "embeddings",
        "health": "llamacpp.health",
        "metrics": "llamacpp.metrics",
        "slots": "llamacpp.slots",
        "props": "llamacpp.props",
        "models": "llamacpp.models",
    }

    if endpoint_map.get(endpoint) in ("chat", "text_completion", "embeddings"):
        return endpoint_map[endpoint], True
    return endpoint_map.get(endpoint, f"llamacpp.{method.lower()}.{endpoint}"), False


def _provider_name() -> str:
    from ..semconv import gen_ai
    return gen_ai.PROVIDER_LLAMA_CPP


def _parse_server(self_client: Any) -> Tuple[Optional[str], Optional[int]]:
    from ..otel.gen_ai_utils import parse_server_address
    return parse_server_address(getattr(self_client, "base_url", None))


def _extract_model(json_data: Optional[Dict[str, Any]], params: Optional[Dict[str, Any]]) -> Optional[str]:
    if json_data and isinstance(json_data, dict):
        model = json_data.get("model") or json_data.get("model_name")
        if model:
            return str(model)
    if params and isinstance(params, dict):
        model = params.get("model") or params.get("model_name")
        if model:
            return str(model)
    return None


def _build_request_attrs(
    operation: str,
    json_data: Dict[str, Any],
    stream: bool,
    model: Optional[str],
) -> Dict[str, Any]:
    from ..semconv.gen_ai_builder import build_gen_ai_attrs_from_request
    from ..semconv import gen_ai as gen_ai_keys

    model_name = model or json_data.get("model") or json_data.get("model_name") or "unknown"

    stop = json_data.get("stop")
    stop_sequences = None
    if isinstance(stop, list):
        stop_sequences = stop
    elif isinstance(stop, str):
        stop_sequences = [stop]

    max_tokens = json_data.get("max_tokens")
    if max_tokens is None:
        max_tokens = json_data.get("n_predict")

    choice_count = json_data.get("n")
    encoding_formats = None
    enc = json_data.get("encoding_format")
    if enc is not None:
        encoding_formats = [enc] if isinstance(enc, str) else enc

    return build_gen_ai_attrs_from_request(
        model=str(model_name),
        operation=operation,
        provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
        temperature=json_data.get("temperature"),
        top_p=json_data.get("top_p"),
        top_k=json_data.get("top_k"),
        max_tokens=max_tokens,
        frequency_penalty=json_data.get("frequency_penalty"),
        presence_penalty=json_data.get("presence_penalty"),
        seed=json_data.get("seed"),
        stop_sequences=stop_sequences,
        choice_count=choice_count,
        stream=json_data.get("stream", stream),
        conversation_id=json_data.get("conversation_id"),
        encoding_formats=encoding_formats,
    )


def _build_tool_attrs(json_data: Dict[str, Any]) -> Dict[str, Any]:
    from ..semconv.gen_ai_builder import build_gen_ai_attrs_from_tools

    return build_gen_ai_attrs_from_tools(
        tool_definitions=json_data.get("tools"),
        tool_calls=json_data.get("tool_calls"),
        record_content=True,
    )


def _build_content_attrs(json_data: Dict[str, Any], record_content_max_chars: int) -> Dict[str, Any]:
    from ..semconv.gen_ai_builder import build_content_attrs

    input_messages = json_data.get("messages")
    if not input_messages and json_data.get("prompt") is not None:
        input_messages = [{"role": "user", "content": json_data.get("prompt")}]

    return build_content_attrs(
        input_messages=input_messages,
        record_content=True,
        record_content_max_chars=record_content_max_chars,
    )


def _build_output_content_attrs(result: Dict[str, Any], record_content_max_chars: int) -> Dict[str, Any]:
    from ..semconv.gen_ai_builder import build_content_attrs

    output_messages: Optional[List[Dict[str, Any]]] = None
    if "choices" in result and result.get("choices"):
        choice0 = result["choices"][0] or {}
        message = choice0.get("message") or {}
        if "content" in message:
            output_messages = [{"role": "assistant", "content": message.get("content")}]
        elif "text" in choice0:
            output_messages = [{"role": "assistant", "content": choice0.get("text")}]
    elif "content" in result:
        output_messages = [{"role": "assistant", "content": result.get("content")}]

    return build_content_attrs(
        output_messages=output_messages,
        record_content=True,
        record_content_max_chars=record_content_max_chars,
    )


def _build_response_attrs(result: Dict[str, Any]) -> Dict[str, Any]:
    from ..semconv.gen_ai_builder import build_gen_ai_attrs_from_response

    input_tokens = None
    output_tokens = None
    finish_reason = None

    if "usage" in result and isinstance(result["usage"], dict):
        usage = result["usage"]
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
    else:
        if "tokens_evaluated" in result:
            input_tokens = result.get("tokens_evaluated")
        if "tokens_predicted" in result:
            output_tokens = result.get("tokens_predicted")

    if "choices" in result and result.get("choices"):
        finish_reason = result["choices"][0].get("finish_reason")
    if finish_reason is None:
        finish_reason = result.get("finish_reason") or result.get("stop_reason")

    finish_reasons = [finish_reason] if finish_reason else None

    return build_gen_ai_attrs_from_response(
        response_id=result.get("id"),
        response_model=result.get("model"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reasons=finish_reasons,
    )


def _emit_metrics(
    *,
    operation: str,
    model: Optional[str],
    result: Dict[str, Any],
    latency_s: float,
    server_address: Optional[str],
    server_port: Optional[int],
) -> None:
    from ..otel.gen_ai_metrics import get_gen_ai_metrics
    from ..otel.gen_ai_utils import build_gen_ai_span_attrs
    from ..semconv import gen_ai as gen_ai_keys

    metrics = get_gen_ai_metrics()
    response_model = result.get("model")
    base_attrs = build_gen_ai_span_attrs(
        operation=operation,
        provider=gen_ai_keys.PROVIDER_LLAMA_CPP,
        model=model,
        response_model=response_model,
        server_address=server_address,
        server_port=server_port,
    )

    metrics.record_client_operation_duration(latency_s, base_attrs)

    usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
    input_tokens = usage.get("prompt_tokens") or result.get("tokens_evaluated")
    output_tokens = usage.get("completion_tokens") or result.get("tokens_predicted")

    if input_tokens is not None:
        metrics.record_client_token_usage(input_tokens, gen_ai_keys.TOKEN_INPUT, base_attrs)
    if output_tokens is not None:
        metrics.record_client_token_usage(output_tokens, gen_ai_keys.TOKEN_OUTPUT, base_attrs)

    timings = result.get("timings")
    if isinstance(timings, dict):
        prompt_ms = timings.get("prompt_ms", 0.0) or 0.0
        predicted_ms = timings.get("predicted_ms", 0.0) or 0.0
        if prompt_ms > 0 or predicted_ms > 0:
            metrics.record_server_request_duration((prompt_ms + predicted_ms) / 1000.0, base_attrs)
        if prompt_ms > 0:
            metrics.record_server_time_to_first_token(prompt_ms / 1000.0, base_attrs)
        if predicted_ms > 0 and output_tokens:
            denom = max(output_tokens - 1, 1)
            metrics.record_server_time_per_output_token(
                (predicted_ms / 1000.0) / denom,
                base_attrs,
            )


# Global instrumentor instance for convenience
_global_instrumentor: Optional[LlamaCppClientInstrumentor] = None


def instrument_llamacpp_client(
    tracer_provider: Optional[Any] = None,
    meter_provider: Optional[Any] = None,
) -> LlamaCppClientInstrumentor:
    """
    Convenience function to instrument LlamaCppClient.

    Args:
        tracer_provider: OpenTelemetry TracerProvider
        meter_provider: OpenTelemetry MeterProvider

    Returns:
        Instrumentor instance

    Example:
        >>> from llamatelemetry.telemetry import instrument_llamacpp_client
        >>> instrumentor = instrument_llamacpp_client()
    """
    global _global_instrumentor

    if _global_instrumentor is None:
        _global_instrumentor = LlamaCppClientInstrumentor()

    _global_instrumentor.instrument(tracer_provider, meter_provider)
    return _global_instrumentor


def uninstrument_llamacpp_client() -> None:
    """
    Convenience function to remove LlamaCppClient instrumentation.
    """
    global _global_instrumentor

    if _global_instrumentor is not None:
        _global_instrumentor.uninstrument()


__all__ = [
    "LlamaCppClientInstrumentorConfig",
    "LlamaCppClientInstrumentor",
    "instrument_llamacpp_client",
    "uninstrument_llamacpp_client",
]
