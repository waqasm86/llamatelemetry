"""
llamatelemetry.otel.provider - TracerProvider + MeterProvider management.

Refactored from telemetry/tracer.py, telemetry/resource.py, telemetry/metrics.py.
"""

import os
import subprocess
from typing import Any, Dict, Optional

from ..config import LlamaTelemetryConfig
from ..semconv import keys

_tracer_provider = None
_meter_provider = None
_otel_available = False

try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    _otel_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Noop fallbacks
# ---------------------------------------------------------------------------

class _NoopTracer:
    """No-op tracer when OTel is unavailable."""

    def start_as_current_span(self, name: str, **kwargs):
        return _NoopSpanContext()

    def start_span(self, name: str, **kwargs):
        return _NoopSpan()


class _NoopSpanContext:
    """No-op context manager for spans."""

    def __enter__(self):
        return _NoopSpan()

    def __exit__(self, *args):
        pass


class _NoopSpan:
    """No-op span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoopMeter:
    """No-op meter."""

    def create_counter(self, *a, **kw):
        return _NoopInstrument()

    def create_histogram(self, *a, **kw):
        return _NoopInstrument()

    def create_observable_gauge(self, *a, **kw):
        return _NoopInstrument()


class _NoopInstrument:
    def add(self, *a, **kw):
        pass

    def record(self, *a, **kw):
        pass


# ---------------------------------------------------------------------------
# GPU Resource builder (from telemetry/resource.py)
# ---------------------------------------------------------------------------

def _nvidia_smi_query() -> Dict[str, Any]:
    """Run nvidia-smi and return parsed GPU attributes."""
    attrs: Dict[str, Any] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        {
                            "name": parts[0],
                            "memory_total": parts[1],
                            "driver_version": parts[2],
                            "compute_capability": parts[3],
                        }
                    )
            if gpus:
                attrs["gpu.count"] = len(gpus)
                attrs[keys.GPU_NAME] = gpus[0]["name"]
                attrs[keys.GPU_MEM_TOTAL_MB] = gpus[0]["memory_total"]
                attrs[keys.GPU_DRIVER_VERSION] = gpus[0]["driver_version"]
                attrs[keys.GPU_COMPUTE_CAP] = gpus[0]["compute_capability"]
                if len(gpus) > 1:
                    attrs["gpu.names"] = ",".join(g["name"] for g in gpus)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return attrs


def _detect_platform() -> str:
    if os.path.exists("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def _build_resource(config: LlamaTelemetryConfig) -> Any:
    """Build an OTel Resource with GPU and service attributes."""
    from .._version import __version__, __binary_version__

    attributes: Dict[str, Any] = {
        "service.name": config.service_name,
        "service.version": __version__,
        "llamatelemetry.version": __version__,
        "llamatelemetry.binary_version": __binary_version__,
        "platform": _detect_platform(),
    }

    if config.enable_gpu:
        gpu_attrs = _nvidia_smi_query()
        attributes.update(gpu_attrs)

    if _otel_available:
        return Resource.create(attributes)
    return attributes


# ---------------------------------------------------------------------------
# Provider lifecycle
# ---------------------------------------------------------------------------

def init_providers(config: LlamaTelemetryConfig) -> None:
    """Create and register TracerProvider + MeterProvider."""
    global _tracer_provider, _meter_provider

    if not _otel_available:
        return

    from .exporters import build_span_exporters, build_metric_exporters
    from .sampling import build_sampler

    resource = _build_resource(config)

    # Sampler
    sampler = build_sampler(config.sampling_strategy, config.sampling_ratio)

    # TracerProvider
    tp_kwargs: Dict[str, Any] = {"resource": resource}
    if sampler is not None:
        tp_kwargs["sampler"] = sampler
    _tracer_provider = TracerProvider(**tp_kwargs)

    # Span exporters
    span_exporters = build_span_exporters(config.otlp_endpoint, config.otlp_headers)
    for exp in span_exporters:
        _tracer_provider.add_span_processor(BatchSpanProcessor(exp))

    # Redaction processor (optional)
    if config.redact_prompts or config.redact_keys:
        from .redaction import RedactionSpanProcessor

        _tracer_provider.add_span_processor(
            RedactionSpanProcessor(
                redact_prompts=config.redact_prompts,
                redact_keys=config.redact_keys or [],
            )
        )

    trace.set_tracer_provider(_tracer_provider)

    # MeterProvider
    metric_readers = build_metric_exporters(config.otlp_endpoint, config.otlp_headers)
    mp_kwargs: Dict[str, Any] = {"resource": resource}
    if metric_readers:
        mp_kwargs["metric_readers"] = metric_readers
    _meter_provider = MeterProvider(**mp_kwargs)
    otel_metrics.set_meter_provider(_meter_provider)


def get_tracer(name: str = "llamatelemetry") -> Any:
    """Return a Tracer (or NoopTracer if not initialised)."""
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(name)
    if _otel_available:
        return trace.get_tracer(name)
    return _NoopTracer()


def get_meter(name: str = "llamatelemetry") -> Any:
    """Return a Meter (or NoopMeter if not initialised)."""
    if _meter_provider is not None:
        return _meter_provider.get_meter(name)
    if _otel_available:
        return otel_metrics.get_meter(name)
    return _NoopMeter()


def flush_providers(timeout_s: float = 5.0) -> None:
    """Flush all pending spans and metrics."""
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_millis=int(timeout_s * 1000))
        except Exception:
            pass
    if _meter_provider is not None:
        try:
            _meter_provider.force_flush(timeout_millis=int(timeout_s * 1000))
        except Exception:
            pass


def shutdown_providers(timeout_s: float = 5.0) -> None:
    """Shutdown providers and release resources."""
    global _tracer_provider, _meter_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception:
            pass
        _tracer_provider = None
    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
        except Exception:
            pass
        _meter_provider = None
