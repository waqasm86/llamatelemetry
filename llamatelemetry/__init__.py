"""
llamatelemetry - CUDA-first OpenTelemetry Python SDK for LLM inference.

v1.0.0 - Production-grade GPU-native LLM observability.

Quick start::

    import llamatelemetry

    llamatelemetry.init(
        service_name="my-llm-app",
        otlp_endpoint="https://otlp.example.com/v1/traces",
    )

    @llamatelemetry.trace()
    def generate(prompt):
        ...

    llamatelemetry.shutdown()
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: LD_LIBRARY_PATH + binary discovery (moved from old __init__)
# ---------------------------------------------------------------------------
from ._internal import _setup_paths  # noqa: F401  (side-effect import)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
from ._version import __version__, __binary_version__

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from .config import LlamaTelemetryConfig, get_config, set_config, is_initialized

# ---------------------------------------------------------------------------
# Submodule re-exports (lazy-friendly)
# ---------------------------------------------------------------------------
from . import semconv  # noqa: F401
from . import otel  # noqa: F401
from . import llama  # noqa: F401
from . import gpu  # noqa: F401
from . import nccl  # noqa: F401
from . import artifacts  # noqa: F401
from . import kaggle  # noqa: F401


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init(
    service_name: str = "llamatelemetry",
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
    environment: str = "development",
    sampling: str = "always_on",
    sampling_ratio: float = 1.0,
    redact: bool = False,
    redact_keys: Optional[List[str]] = None,
    enable_gpu: bool = True,
    enable_llama_cpp: bool = True,
    enable_nccl: bool = False,
) -> None:
    """
    Initialise the llamatelemetry SDK.

    Call this once at application startup before creating spans.
    """
    cfg = LlamaTelemetryConfig(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        otlp_headers=otlp_headers,
        environment=environment,
        sampling_strategy=sampling,
        sampling_ratio=sampling_ratio,
        redact_prompts=redact,
        redact_keys=redact_keys,
        enable_gpu=enable_gpu,
        enable_llama_cpp=enable_llama_cpp,
        enable_nccl=enable_nccl,
        _initialized=True,
    )
    set_config(cfg)
    otel.init_providers(cfg)

    if enable_nccl:
        nccl.enable(True)


def configure(**kwargs: Any) -> None:
    """
    Update configuration after ``init()``.

    Accepts the same keyword arguments as ``init()``.
    """
    cfg = get_config()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def flush(timeout_s: float = 5.0) -> None:
    """Flush all pending telemetry data."""
    otel.flush_providers(timeout_s)


def shutdown(timeout_s: float = 5.0) -> None:
    """Shutdown providers and release resources."""
    otel.shutdown_providers(timeout_s)


def version() -> str:
    """Return the SDK version string."""
    return __version__


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
from ._internal.decorators import (
    trace_decorator as trace,
    workflow_decorator as workflow,
    task_decorator as task,
    tool_decorator as tool,
)

# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------
from ._internal.decorators import (
    span_context as span,
    session_context as session,
    suppress_tracing,
)

# ---------------------------------------------------------------------------
# Backward compatibility (v0.1.0)
# ---------------------------------------------------------------------------
from .compat import InferenceEngine, InferResult  # noqa: F401

# Legacy utility re-exports so ``from llamatelemetry import ServerManager`` still works
from .server import ServerManager  # noqa: F401

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = [
    # Lifecycle
    "init",
    "flush",
    "shutdown",
    "configure",
    "version",
    # Decorators
    "trace",
    "workflow",
    "task",
    "tool",
    # Context managers
    "span",
    "session",
    "suppress_tracing",
    # Submodules
    "llama",
    "gpu",
    "nccl",
    "semconv",
    "artifacts",
    "kaggle",
    "otel",
    # Backward compat
    "InferenceEngine",
    "InferResult",
    "ServerManager",
]
