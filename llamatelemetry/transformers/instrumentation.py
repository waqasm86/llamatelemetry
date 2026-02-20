"""
llamatelemetry.transformers.instrumentation - Instrumented backend wrapper.

Wraps any LLMBackend to automatically create OTel GenAI spans, events, and metrics.
Works for both llama.cpp and Transformers backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from ..backends.base import LLMBackend, LLMRequest, LLMResponse
from ..otel.provider import get_tracer
from ..semconv.gen_ai_builder import (
    build_gen_ai_attrs_from_request,
    build_gen_ai_attrs_from_response,
    build_gen_ai_attrs_from_tools,
    build_content_attrs,
)
from ..semconv import gen_ai
from ..otel.gen_ai_metrics import get_gen_ai_metrics
from ..otel.gen_ai_utils import build_gen_ai_span_attrs, build_span_name, parse_server_address


@dataclass
class TransformersInstrumentorConfig:
    """Configuration for instrumented backend behavior.

    Attributes:
        record_content: Record raw prompt/response content (OFF by default for privacy).
        enable_gpu_enrichment: Attach GPU utilization deltas to spans.
        record_content_max_chars: Max characters to record when content recording is on.
        record_tools: Record tool definitions and call arguments.
        record_events: Emit GenAI operation detail events.
        emit_metrics: Emit GenAI metrics.
    """

    record_content: bool = False
    enable_gpu_enrichment: bool = True
    record_content_max_chars: int = 2000
    record_tools: bool = False
    record_events: bool = False
    emit_metrics: bool = True


class InstrumentedBackend:
    """Wrapper that adds OTel tracing to any LLMBackend.

    Creates spans:
        - {gen_ai.operation.name} {gen_ai.request.model} (root, CLIENT)
        - llamatelemetry.phase.prefill (child)
        - llamatelemetry.phase.decode (child)

    Sets attributes:
        - gen_ai.* (official OTel GenAI semconv)
        - gpu.* deltas (if GPU enrichment is enabled)

    Example:
        >>> from llamatelemetry.backends.llamacpp import LlamaCppBackend
        >>> backend = LlamaCppBackend("http://127.0.0.1:8090")
        >>> instrumented = InstrumentedBackend(backend=backend)
        >>> resp = instrumented.invoke(LLMRequest(
        ...     operation="chat",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... ))
    """

    def __init__(
        self,
        backend: LLMBackend,
        tracer: Optional[Any] = None,
        gpu: Optional[Any] = None,
        config: Optional[TransformersInstrumentorConfig] = None,
    ):
        """Initialize instrumented backend.

        Args:
            backend: The underlying LLM backend to instrument.
            tracer: OTel tracer instance. Auto-obtained if None.
            gpu: GPUSpanEnricher instance for GPU metric enrichment.
            config: Instrumentation configuration.
        """
        self._backend = backend
        self._tracer = tracer or get_tracer("llamatelemetry.instrumented")
        self._gpu = gpu
        self._config = config or TransformersInstrumentorConfig()

    @property
    def name(self) -> str:
        return self._backend.name

    def invoke(self, req: LLMRequest) -> LLMResponse:
        """Execute a traced LLM request.

        Creates the full GenAI span hierarchy with gen_ai.* attributes.
        """
        provider = req.provider or self._backend.name
        model = req.model or None
        server_address, server_port = parse_server_address(
            getattr(self._backend, "_base_url", None)
            or getattr(self._backend, "base_url", None)
        )

        # Take GPU snapshot before
        gpu_before = None
        if self._gpu and self._config.enable_gpu_enrichment:
            gpu_before = self._gpu.snapshot()

        start = time.perf_counter()
        span_name = build_span_name(req.operation, model)

        try:
            from opentelemetry.trace import SpanKind
            span_kind = (
                SpanKind.INTERNAL if self._backend.name == "transformers" else SpanKind.CLIENT
            )
        except Exception:
            span_kind = None

        span_attrs = build_gen_ai_span_attrs(
            operation=req.operation,
            provider=provider,
            model=model,
            server_address=server_address,
            server_port=server_port,
        )

        with self._tracer.start_as_current_span(
            span_name,
            kind=span_kind,
            attributes=span_attrs if span_kind is not None else None,
        ) as root_span:
            # Set gen_ai.* request attributes
            gen_ai_req_attrs = build_gen_ai_attrs_from_request(
                model=model or "",
                operation=req.operation,
                provider=provider,
                temperature=req.parameters.get("temperature") if req.parameters else None,
                top_p=req.parameters.get("top_p") if req.parameters else None,
                top_k=req.parameters.get("top_k") if req.parameters else None,
                max_tokens=req.parameters.get("max_tokens") if req.parameters else None,
                stream=req.stream,
                conversation_id=req.conversation_id,
            )
            for k, v in gen_ai_req_attrs.items():
                root_span.set_attribute(k, v)

            if req.request_id:
                root_span.set_attribute("request.id", req.request_id)

            if self._config.record_tools:
                tool_attrs = build_gen_ai_attrs_from_tools(
                    tool_definitions=req.parameters.get("tools") if req.parameters else None,
                    tool_calls=req.parameters.get("tool_calls") if req.parameters else None,
                    record_content=True,
                )
                for k, v in tool_attrs.items():
                    root_span.set_attribute(k, v)

            if self._config.record_content:
                input_messages = req.messages
                if not input_messages and req.prompt:
                    input_messages = [{"role": "user", "content": req.prompt}]
                content_attrs = build_content_attrs(
                    input_messages=input_messages,
                    record_content=True,
                    record_content_max_chars=self._config.record_content_max_chars,
                )
                for k, v in content_attrs.items():
                    root_span.set_attribute(k, v)

            # Prefill span
            with self._tracer.start_as_current_span("llamatelemetry.phase.prefill") as pfill:
                pass

            try:
                # Execute the actual backend call
                resp = self._backend.invoke(req)

                elapsed_ms = (time.perf_counter() - start) * 1000.0

                # Decode span
                with self._tracer.start_as_current_span("llamatelemetry.phase.decode") as dec:
                    pass

                # Set gen_ai.* response attributes
                finish_reasons = [resp.finish_reason] if resp.finish_reason else None
                gen_ai_resp_attrs = build_gen_ai_attrs_from_response(
                    response_id=resp.response_id,
                    response_model=resp.response_model,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    finish_reasons=finish_reasons,
                )
                for k, v in gen_ai_resp_attrs.items():
                    root_span.set_attribute(k, v)

                if self._config.record_events:
                    event_attrs = {}
                    event_attrs.update(gen_ai_req_attrs)
                    event_attrs.update(gen_ai_resp_attrs)
                    if self._config.record_content:
                        output_messages = None
                        if resp.output_text:
                            output_messages = [{"role": "assistant", "content": resp.output_text}]
                        content_attrs = build_content_attrs(
                            input_messages=req.messages,
                            output_messages=output_messages,
                            record_content=True,
                            record_content_max_chars=self._config.record_content_max_chars,
                        )
                        event_attrs.update(content_attrs)
                    if self._config.record_tools:
                        tool_attrs = build_gen_ai_attrs_from_tools(
                            tool_definitions=req.parameters.get("tools") if req.parameters else None,
                            tool_calls=req.parameters.get("tool_calls") if req.parameters else None,
                            record_content=True,
                        )
                        event_attrs.update(tool_attrs)
                    root_span.add_event(
                        "gen_ai.client.inference.operation.details",
                        attributes=event_attrs,
                    )

                if self._config.emit_metrics:
                    metrics = get_gen_ai_metrics()
                    base_attrs = build_gen_ai_span_attrs(
                        operation=req.operation,
                        provider=provider,
                        model=model,
                        response_model=resp.response_model,
                        server_address=server_address,
                        server_port=server_port,
                    )
                    metrics.record_client_operation_duration(
                        elapsed_ms / 1000.0, base_attrs
                    )
                    if resp.input_tokens is not None:
                        metrics.record_client_token_usage(
                            resp.input_tokens, gen_ai.TOKEN_INPUT, base_attrs
                        )
                    if resp.output_tokens is not None:
                        metrics.record_client_token_usage(
                            resp.output_tokens, gen_ai.TOKEN_OUTPUT, base_attrs
                        )

                    # Server-side metrics if timings are available
                    timings = getattr(resp.raw, "timings", None) if resp.raw is not None else None
                    if timings:
                        prompt_ms = getattr(timings, "prompt_ms", 0.0) or 0.0
                        predicted_ms = getattr(timings, "predicted_ms", 0.0) or 0.0
                        if prompt_ms > 0 or predicted_ms > 0:
                            metrics.record_server_request_duration(
                                (prompt_ms + predicted_ms) / 1000.0, base_attrs
                            )
                        if prompt_ms > 0:
                            metrics.record_server_time_to_first_token(
                                prompt_ms / 1000.0, base_attrs
                            )
                        if predicted_ms > 0 and resp.output_tokens:
                            denom = max(resp.output_tokens - 1, 1)
                            metrics.record_server_time_per_output_token(
                                (predicted_ms / 1000.0) / denom, base_attrs
                            )

                # Attach GPU deltas
                if self._gpu and self._config.enable_gpu_enrichment and gpu_before:
                    gpu_after = self._gpu.snapshot()
                    self._gpu.attach_deltas(root_span, gpu_before, gpu_after)

                return resp

            except Exception as exc:
                root_span.set_attribute("error.type", exc.__class__.__name__)
                root_span.record_exception(exc)
                raise
