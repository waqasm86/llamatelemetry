"""
llamatelemetry.otel_gen_ai.tracer - OpenTelemetry gen_ai semantic convention tracer

Provides high-level tracing API with automatic gen_ai.* attribute management.
"""

from typing import Optional, Dict, Any, List
import time
import logging

from opentelemetry import trace, metrics
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai

logger = logging.getLogger(__name__)


class GenAITracer:
    """
    Wrapper for OpenTelemetry with gen_ai.* semantic conventions.

    Automatically manages:
      - Span creation with gen_ai attributes
      - Metric recording
      - Context propagation
      - Event tracking
    """

    def __init__(
        self,
        tracer: trace.Tracer,
        meter: metrics.Meter,
        provider_name: str = "llamatelemetry",
    ):
        """
        Initialize tracer with OpenTelemetry providers.

        Args:
            tracer: OpenTelemetry Tracer instance
            meter: OpenTelemetry Meter instance
            provider_name: Provider identifier (for gen_ai.provider.name)
        """
        self.tracer = tracer
        self.meter = meter
        self.provider_name = provider_name

        logger.info(f"Initializing GenAITracer (provider: {provider_name})")

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize all gen_ai metrics."""
        # Operation duration histogram (seconds)
        self.operation_duration = self.meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI client-side operation duration",
        )

        # Token usage histogram (count)
        self.token_usage = self.meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
        )

        # Server request duration (seconds) - TTLB
        self.server_request_duration = self.meter.create_histogram(
            "gen_ai.server.request.duration",
            unit="s",
            description="Time-to-last-byte / server-side operation duration",
        )

        # Time-to-first-token (seconds)
        self.ttft = self.meter.create_histogram(
            "gen_ai.server.time_to_first_token",
            unit="s",
            description="Time to generate first output token",
        )

        # Time-per-output-token (seconds)
        self.tpot = self.meter.create_histogram(
            "gen_ai.server.time_per_output_token",
            unit="s",
            description="Time per output token after first token",
        )

        logger.info("Metrics initialized successfully")

    def trace_inference(
        self,
        model_name: str,
        operation: str = "chat",
        conversation_id: Optional[str] = None,
    ):
        """
        Create span for inference operation.

        Returns context manager for automatic span lifecycle.

        Args:
            model_name: Model identifier (e.g., "llama-2-13b-Q4_K_M.gguf")
            operation: Operation type ("chat", "embeddings", "text_completion")
            conversation_id: Session/conversation identifier

        Returns:
            InferenceSpanContext for use with 'with' statement
        """
        from .context import InferenceSpanContext

        return InferenceSpanContext(
            tracer=self,
            model_name=model_name,
            operation=operation,
            conversation_id=conversation_id,
        )

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> trace.Span:
        """
        Start a span with optional attributes.

        Args:
            name: Span name
            attributes: Initial attributes dict

        Returns:
            Active OpenTelemetry Span
        """
        span = self.tracer.start_as_current_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return span

    def record_operation_duration(
        self,
        duration_seconds: float,
        model_name: str,
        operation: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record operation duration metric.

        Args:
            duration_seconds: Operation duration in seconds
            model_name: Model identifier
            operation: Operation type
            attributes: Additional attributes
        """
        base_attrs = {
            gen_ai.GEN_AI_REQUEST_MODEL: model_name,
            gen_ai.GEN_AI_OPERATION_NAME: operation,
        }

        if attributes:
            base_attrs.update(attributes)

        self.operation_duration.record(duration_seconds, base_attrs)

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record token usage metrics.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model identifier
            attributes: Additional attributes
        """
        base_attrs = {
            gen_ai.GEN_AI_REQUEST_MODEL: model_name,
        }

        if attributes:
            base_attrs.update(attributes)

        # Record input tokens
        self.token_usage.record(
            input_tokens,
            {**base_attrs, gen_ai.GEN_AI_TOKEN_TYPE: "input"},
        )

        # Record output tokens
        self.token_usage.record(
            output_tokens,
            {**base_attrs, gen_ai.GEN_AI_TOKEN_TYPE: "output"},
        )

    def record_ttft(
        self,
        ttft_seconds: float,
        model_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record time-to-first-token metric.

        Args:
            ttft_seconds: TTFT in seconds
            model_name: Model identifier
            attributes: Additional attributes
        """
        base_attrs = {
            gen_ai.GEN_AI_REQUEST_MODEL: model_name,
        }

        if attributes:
            base_attrs.update(attributes)

        self.ttft.record(ttft_seconds, base_attrs)

    def record_tpot(
        self,
        tpot_seconds: float,
        model_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record time-per-output-token metric.

        Args:
            tpot_seconds: TPOT in seconds
            model_name: Model identifier
            attributes: Additional attributes
        """
        base_attrs = {
            gen_ai.GEN_AI_REQUEST_MODEL: model_name,
        }

        if attributes:
            base_attrs.update(attributes)

        self.tpot.record(tpot_seconds, base_attrs)

    def __repr__(self) -> str:
        return f"GenAITracer(provider={self.provider_name})"
