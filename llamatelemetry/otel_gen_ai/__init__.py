"""
llamatelemetry.otel_gen_ai - OpenTelemetry gen_ai.* semantic conventions integration

Complete integration of OpenTelemetry with gen_ai semantic conventions for GGUF LLM observability.

Features:
  - 45+ gen_ai.* semantic convention attributes
  - 5 histogram metrics (TTFT, TPOT, operation duration, token usage)
  - Automatic span context and attribute management
  - GPU monitoring integration
  - OTLP HTTP exporter configuration
  - All observability runs on GPU (no CPU overhead)
"""

from .tracer import GenAITracer
from .metrics import GenAIMetrics
from .context import InferenceSpanContext, InferenceContext
from .gpu_monitor import GPUMonitor

__all__ = [
    'GenAITracer',
    'GenAIMetrics',
    'InferenceSpanContext',
    'InferenceContext',
    'GPUMonitor',
]

__version__ = '2.0.0'
