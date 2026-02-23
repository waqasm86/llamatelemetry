"""
llamatelemetry.otel_gen_ai.context - Span context managers with gen_ai attributes
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai

logger = logging.getLogger(__name__)


class InferenceSpanContext:
    """
    Context manager for inference spans.

    Automatically manages gen_ai.* attributes and metrics recording.
    """

    def __init__(
        self,
        tracer,
        model_name: str,
        operation: str = "chat",
        conversation_id: Optional[str] = None,
    ):
        """
        Initialize span context.

        Args:
            tracer: GenAITracer instance
            model_name: Model identifier
            operation: Operation type
            conversation_id: Session ID (optional)
        """
        self.tracer = tracer
        self.model_name = model_name
        self.operation = operation
        self.conversation_id = conversation_id
        self.span = None
        self.start_time = None

    def __enter__(self):
        """Enter span context."""
        # Start span
        self.span = self.tracer.tracer.start_as_current_span(
            f"llama.{self.operation}"
        )

        # Set core gen_ai attributes
        self.span.set_attribute(gen_ai.GEN_AI_PROVIDER_NAME, self.tracer.provider_name)
        self.span.set_attribute(gen_ai.GEN_AI_REQUEST_MODEL, self.model_name)
        self.span.set_attribute(gen_ai.GEN_AI_OPERATION_NAME, self.operation)

        if self.conversation_id:
            self.span.set_attribute(gen_ai.GEN_AI_CONVERSATION_ID, self.conversation_id)

        self.start_time = time.perf_counter()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit span context and record metrics."""
        if self.start_time and self.span:
            elapsed = time.perf_counter() - self.start_time

            # Record operation duration metric
            attributes = {}
            self.tracer.record_operation_duration(
                elapsed,
                self.model_name,
                self.operation,
                attributes,
            )

        if self.span:
            self.span.end()


class InferenceContext:
    """
    High-level context for complete inference tracking.

    Manages all gen_ai attributes throughout inference lifecycle.
    """

    def __init__(self, tracer, model_name: str, operation: str = "chat"):
        """
        Initialize inference context.

        Args:
            tracer: GenAITracer instance
            model_name: Model identifier
            operation: Operation type
        """
        self.tracer = tracer
        self.model_name = model_name
        self.operation = operation
        self.span = None
        self.start_time = None

        # Request parameters
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None
        self.top_k: Optional[int] = None
        self.max_tokens: Optional[int] = None
        self.seed: Optional[int] = None
        self.conversation_id: Optional[str] = None

        # Response metrics
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.ttft_ms: Optional[float] = None
        self.tpot_ms: Optional[float] = None
        self.finish_reason: Optional[str] = None

    def start(self) -> 'InferenceContext':
        """Start inference context."""
        self.span = self.tracer.tracer.start_as_current_span(
            f"llama.{self.operation}"
        )

        # Set core attributes
        self.span.set_attribute(gen_ai.GEN_AI_PROVIDER_NAME, self.tracer.provider_name)
        self.span.set_attribute(gen_ai.GEN_AI_REQUEST_MODEL, self.model_name)
        self.span.set_attribute(gen_ai.GEN_AI_OPERATION_NAME, self.operation)

        self.start_time = time.perf_counter()
        return self

    def set_request_parameters(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> 'InferenceContext':
        """Set request parameters in span."""
        if self.span is None:
            raise RuntimeError("Call start() before setting parameters")

        # Store parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.seed = seed
        self.conversation_id = conversation_id

        # Set in span
        if temperature is not None:
            self.span.set_attribute(gen_ai.GEN_AI_REQUEST_TEMPERATURE, temperature)
        if top_p is not None:
            self.span.set_attribute(gen_ai.GEN_AI_REQUEST_TOP_P, top_p)
        if top_k is not None:
            self.span.set_attribute(gen_ai.GEN_AI_REQUEST_TOP_K, top_k)
        if max_tokens is not None:
            self.span.set_attribute(gen_ai.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
        if seed is not None:
            self.span.set_attribute(gen_ai.GEN_AI_REQUEST_SEED, seed)
        if conversation_id is not None:
            self.span.set_attribute(gen_ai.GEN_AI_CONVERSATION_ID, conversation_id)

        return self

    def set_response(
        self,
        input_tokens: int,
        output_tokens: int,
        ttft_ms: float,
        tpot_ms: float,
        finish_reason: str = "stop",
    ) -> 'InferenceContext':
        """Set response metrics in span."""
        if self.span is None:
            raise RuntimeError("Call start() before setting response")

        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.finish_reason = finish_reason

        # Set token usage
        self.span.set_attribute(gen_ai.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        self.span.set_attribute(gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

        # Set response metadata
        self.span.set_attribute(gen_ai.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        self.span.set_attribute(gen_ai.GEN_AI_OUTPUT_TYPE, "text")

        # Record metrics
        self.tracer.record_token_usage(input_tokens, output_tokens, self.model_name)
        self.tracer.record_ttft(ttft_ms / 1000.0, self.model_name)
        self.tracer.record_tpot(tpot_ms / 1000.0, self.model_name)

        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> 'InferenceContext':
        """Add event to span."""
        if self.span:
            self.span.add_event(name, attributes or {})
        return self

    def end(self) -> None:
        """End inference context."""
        if self.span:
            self.span.end()
            self.span = None

    def __enter__(self) -> 'InferenceContext':
        """Enter context manager."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.end()

    def __repr__(self) -> str:
        return (
            f"InferenceContext("
            f"model={self.model_name}, "
            f"op={self.operation}, "
            f"tokens={self.output_tokens})"
        )
