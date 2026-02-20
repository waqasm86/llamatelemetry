"""
llamatelemetry.transformers.instrumentation - Instrumented backend wrapper.

Wraps any LLMBackend to automatically create OTel spans with gen_ai.* + legacy llm.*
attributes. Works for both llama.cpp and Transformers backends.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..backends.base import LLMBackend, LLMRequest, LLMResponse
from ..otel.provider import get_tracer
from ..semconv import keys
from ..semconv.mapping import set_dual_attrs
from ..semconv.gen_ai_builder import (
    build_gen_ai_attrs_from_request,
    build_gen_ai_attrs_from_response,
)
from ..semconv import gen_ai


@dataclass
class TransformersInstrumentorConfig:
    """Configuration for instrumented backend behavior.

    Attributes:
        record_content: Record raw prompt/response content (OFF by default for privacy).
        record_prompt_hash: Record SHA-256 hash of prompts.
        record_output_hash: Record SHA-256 hash of outputs.
        enable_gpu_enrichment: Attach GPU utilization deltas to spans.
        record_content_max_chars: Max characters to record when content recording is on.
        record_tools: Record tool definitions and call arguments.
    """

    record_content: bool = False
    record_prompt_hash: bool = True
    record_output_hash: bool = True
    enable_gpu_enrichment: bool = True
    record_content_max_chars: int = 2000
    record_tools: bool = False


class InstrumentedBackend:
    """Wrapper that adds OTel tracing to any LLMBackend.

    Creates spans:
        - llm.request (root)
        - llm.phase.prefill (child)
        - llm.phase.decode (child)

    Sets attributes:
        - gen_ai.* (official OTel GenAI semconv)
        - legacy llm.* (backward compatibility)
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

        Creates the full span hierarchy with gen_ai.* and llm.* attributes.
        """
        provider = req.provider or self._backend.name
        model = req.model or ""

        # Take GPU snapshot before
        gpu_before = None
        if self._gpu and self._config.enable_gpu_enrichment:
            gpu_before = self._gpu.snapshot()

        start = time.perf_counter()

        with self._tracer.start_as_current_span("llm.request") as root_span:
            # Set gen_ai.* request attributes
            gen_ai_req_attrs = build_gen_ai_attrs_from_request(
                model=model,
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

            # Set legacy llm.* attributes
            root_span.set_attribute(keys.LLM_SYSTEM, "llamatelemetry")
            root_span.set_attribute(keys.LLM_MODEL, model)
            root_span.set_attribute(keys.LLM_STREAM, req.stream)
            if req.request_id:
                root_span.set_attribute(keys.REQUEST_ID, req.request_id)

            # Record content hash if configured
            if self._config.record_prompt_hash and req.messages:
                import json
                prompt_str = json.dumps(req.messages)
                root_span.set_attribute(
                    "llamatelemetry.prompt.sha256",
                    hashlib.sha256(prompt_str.encode()).hexdigest(),
                )

            # Prefill span
            with self._tracer.start_as_current_span("llm.phase.prefill") as pfill:
                pfill.set_attribute(keys.LLM_PHASE, "prefill")

            try:
                # Execute the actual backend call
                resp = self._backend.invoke(req)

                elapsed_ms = (time.perf_counter() - start) * 1000.0

                # Decode span
                with self._tracer.start_as_current_span("llm.phase.decode") as dec:
                    dec.set_attribute(keys.LLM_PHASE, "decode")
                    if resp.output_tokens:
                        dec.set_attribute(keys.LLM_OUTPUT_TOKENS, resp.output_tokens)
                        tps = (resp.output_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
                        dec.set_attribute(keys.LLM_TOKENS_PER_SECOND, tps)

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

                # Set legacy llm.* response attributes
                root_span.set_attribute(keys.LLM_REQUEST_DURATION_MS, elapsed_ms)
                if resp.input_tokens is not None:
                    root_span.set_attribute(keys.LLM_INPUT_TOKENS, resp.input_tokens)
                if resp.output_tokens is not None:
                    root_span.set_attribute(keys.LLM_OUTPUT_TOKENS, resp.output_tokens)
                    total = (resp.input_tokens or 0) + resp.output_tokens
                    root_span.set_attribute(keys.LLM_TOKENS_TOTAL, total)
                if resp.finish_reason:
                    root_span.set_attribute(keys.LLM_FINISH_REASON, resp.finish_reason)

                # Record output hash
                if self._config.record_output_hash and resp.output_text:
                    root_span.set_attribute(
                        "llamatelemetry.output.sha256",
                        hashlib.sha256(resp.output_text.encode()).hexdigest(),
                    )

                # Attach GPU deltas
                if self._gpu and self._config.enable_gpu_enrichment and gpu_before:
                    gpu_after = self._gpu.snapshot()
                    self._gpu.attach_deltas(root_span, gpu_before, gpu_after)

                return resp

            except Exception as exc:
                root_span.set_attribute(keys.LLM_ERROR, str(exc))
                root_span.record_exception(exc)
                raise
