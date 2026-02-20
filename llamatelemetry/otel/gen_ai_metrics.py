"""
llamatelemetry.otel.gen_ai_metrics - GenAI metrics helpers.

Implements GenAI client/server metrics from the OpenTelemetry GenAI spec.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..semconv import gen_ai
from .provider import get_meter


class GenAIMetrics:
    """Singleton-like helper for GenAI metric instruments."""

    def __init__(self, meter: Optional[Any] = None):
        self._meter = meter or get_meter("llamatelemetry.gen_ai")

        self._client_operation_duration = self._meter.create_histogram(
            name="gen_ai.client.operation.duration",
            description="GenAI client operation duration.",
            unit="s",
        )
        self._client_token_usage = self._meter.create_histogram(
            name="gen_ai.client.token.usage",
            description="GenAI client token usage.",
            unit="{token}",
        )
        self._server_request_duration = self._meter.create_histogram(
            name="gen_ai.server.request.duration",
            description="GenAI server request duration.",
            unit="s",
        )
        self._server_time_to_first_token = self._meter.create_histogram(
            name="gen_ai.server.time_to_first_token",
            description="GenAI server time to first token.",
            unit="s",
        )
        self._server_time_per_output_token = self._meter.create_histogram(
            name="gen_ai.server.time_per_output_token",
            description="GenAI server time per output token.",
            unit="s",
        )

    def record_client_operation_duration(self, duration_s: float, attrs: Dict[str, Any]) -> None:
        self._client_operation_duration.record(duration_s, attrs)

    def record_client_token_usage(self, token_count: int, token_type: str, attrs: Dict[str, Any]) -> None:
        usage_attrs = dict(attrs)
        usage_attrs[gen_ai.GEN_AI_TOKEN_TYPE] = token_type
        self._client_token_usage.record(token_count, usage_attrs)

    def record_server_request_duration(self, duration_s: float, attrs: Dict[str, Any]) -> None:
        self._server_request_duration.record(duration_s, attrs)

    def record_server_time_to_first_token(self, ttft_s: float, attrs: Dict[str, Any]) -> None:
        self._server_time_to_first_token.record(ttft_s, attrs)

    def record_server_time_per_output_token(self, tpot_s: float, attrs: Dict[str, Any]) -> None:
        self._server_time_per_output_token.record(tpot_s, attrs)


_metrics: Optional[GenAIMetrics] = None


def get_gen_ai_metrics() -> GenAIMetrics:
    global _metrics
    if _metrics is None:
        _metrics = GenAIMetrics()
    return _metrics
