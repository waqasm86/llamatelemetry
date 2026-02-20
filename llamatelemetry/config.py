"""
llamatelemetry.config - Thread-safe singleton configuration.

Provides a single source of truth for all SDK settings.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LlamaTelemetryConfig:
    """Configuration for the llamatelemetry SDK."""

    # Service identity
    service_name: str = "llamatelemetry"
    environment: str = "development"

    # OTLP export
    otlp_endpoint: Optional[str] = None
    otlp_headers: Optional[Dict[str, str]] = None

    # Sampling
    sampling_strategy: str = "always_on"
    sampling_ratio: float = 1.0

    # Privacy
    redact_prompts: bool = False
    redact_keys: Optional[list] = None

    # Feature flags
    enable_gpu: bool = True
    enable_llama_cpp: bool = True
    enable_nccl: bool = False
    enable_trace_graphs: bool = False

    # Internal
    _initialized: bool = field(default=False, repr=False)


_lock = threading.Lock()
_config: Optional[LlamaTelemetryConfig] = None


def get_config() -> LlamaTelemetryConfig:
    """Return the current config, creating a default if needed."""
    global _config
    with _lock:
        if _config is None:
            _config = LlamaTelemetryConfig()
        return _config


def set_config(config: LlamaTelemetryConfig) -> None:
    """Replace the global config."""
    global _config
    with _lock:
        _config = config


def is_initialized() -> bool:
    """Check whether ``init()`` has been called."""
    with _lock:
        return _config is not None and _config._initialized
