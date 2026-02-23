"""
llamatelemetry.otel_gen_ai.metrics - Gen AI metrics helpers
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GenAIMetrics:
    """
    Helper class for recording gen_ai metrics.

    Provides structured access to all gen_ai metric names.
    """

    # Operation duration (client-side, end-to-end)
    OPERATION_DURATION = "gen_ai.client.operation.duration"

    # Token usage (input and output)
    TOKEN_USAGE = "gen_ai.client.token.usage"

    # Server request duration (time-to-last-byte)
    SERVER_REQUEST_DURATION = "gen_ai.server.request.duration"

    # Time-to-first-token (prefill latency)
    TIME_TO_FIRST_TOKEN = "gen_ai.server.time_to_first_token"

    # Time-per-output-token (decode throughput reciprocal)
    TIME_PER_OUTPUT_TOKEN = "gen_ai.server.time_per_output_token"

    # Units
    UNIT_SECONDS = "s"
    UNIT_TOKENS = "{token}"

    @staticmethod
    def get_metric_description(metric_name: str) -> str:
        """Get human-readable description for metric."""
        descriptions = {
            GenAIMetrics.OPERATION_DURATION: "Client-side GenAI operation duration",
            GenAIMetrics.TOKEN_USAGE: "Number of tokens used (input/output)",
            GenAIMetrics.SERVER_REQUEST_DURATION: "Server-side operation duration (TTLB)",
            GenAIMetrics.TIME_TO_FIRST_TOKEN: "Time to generate first output token",
            GenAIMetrics.TIME_PER_OUTPUT_TOKEN: "Time per output token after first",
        }
        return descriptions.get(metric_name, "Unknown metric")

    @staticmethod
    def get_metric_unit(metric_name: str) -> str:
        """Get unit for metric."""
        units = {
            GenAIMetrics.OPERATION_DURATION: GenAIMetrics.UNIT_SECONDS,
            GenAIMetrics.TOKEN_USAGE: GenAIMetrics.UNIT_TOKENS,
            GenAIMetrics.SERVER_REQUEST_DURATION: GenAIMetrics.UNIT_SECONDS,
            GenAIMetrics.TIME_TO_FIRST_TOKEN: GenAIMetrics.UNIT_SECONDS,
            GenAIMetrics.TIME_PER_OUTPUT_TOKEN: GenAIMetrics.UNIT_SECONDS,
        }
        return units.get(metric_name, "")


class MetricRecorder:
    """
    Structured metric recording with validation.
    """

    def __init__(self, meter):
        """Initialize with OpenTelemetry Meter."""
        self.meter = meter
        self._histograms = {}

    def get_histogram(self, metric_name: str):
        """Get or create histogram."""
        if metric_name not in self._histograms:
            description = GenAIMetrics.get_metric_description(metric_name)
            unit = GenAIMetrics.get_metric_unit(metric_name)

            self._histograms[metric_name] = self.meter.create_histogram(
                metric_name,
                unit=unit,
                description=description,
            )

        return self._histograms[metric_name]

    def record(
        self,
        metric_name: str,
        value: float,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record metric value."""
        histogram = self.get_histogram(metric_name)
        histogram.record(value, attributes or {})
