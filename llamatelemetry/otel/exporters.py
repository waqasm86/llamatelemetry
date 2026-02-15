"""
llamatelemetry.otel.exporters - OTLP HTTP/gRPC + console exporters.

Refactored from telemetry/exporter.py.
"""

import warnings
from typing import Any, Dict, List, Optional


def build_span_exporters(
    endpoint: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """
    Build SpanExporter instances.

    Args:
        endpoint: OTLP endpoint URL (http/https -> HTTP, else gRPC).
                  None -> console-only debug mode.
        headers: Extra HTTP headers (e.g. Authorization for Grafana Cloud).

    Returns:
        List of configured SpanExporter instances.
    """
    exporters: List[Any] = []

    if endpoint is None:
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            exporters.append(ConsoleSpanExporter())
        except ImportError:
            pass
        return exporters

    if endpoint.startswith(("http://", "https://")):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            kwargs: Dict[str, Any] = {"endpoint": endpoint}
            if headers:
                kwargs["headers"] = headers
            exporters.append(OTLPSpanExporter(**kwargs))
        except ImportError:
            warnings.warn(
                "OTLP HTTP exporter not available. Install: "
                "pip install opentelemetry-exporter-otlp-proto-http",
                ImportWarning,
                stacklevel=2,
            )
    else:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            kwargs = {"endpoint": endpoint}
            if headers:
                kwargs["headers"] = headers
            exporters.append(OTLPSpanExporter(**kwargs))
        except ImportError:
            warnings.warn(
                "OTLP gRPC exporter not available. Install: "
                "pip install opentelemetry-exporter-otlp-proto-grpc",
                ImportWarning,
                stacklevel=2,
            )

    return exporters


def build_metric_exporters(
    endpoint: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """
    Build MetricReader instances for MeterProvider.

    Returns a list of PeriodicExportingMetricReader or empty list.
    """
    readers: List[Any] = []

    if endpoint is None:
        return readers

    if endpoint.startswith(("http://", "https://")):
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            kwargs: Dict[str, Any] = {"endpoint": endpoint}
            if headers:
                kwargs["headers"] = headers
            readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(**kwargs))
            )
        except ImportError:
            pass
    else:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            kwargs = {"endpoint": endpoint}
            if headers:
                kwargs["headers"] = headers
            readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(**kwargs))
            )
        except ImportError:
            pass

    return readers
