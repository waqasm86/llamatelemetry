"""
llamatelemetry.otel.gen_ai_utils - GenAI span helpers.

Utilities for building GenAI-compliant span names and attributes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from ..semconv import gen_ai


def build_span_name(operation: str, model: Optional[str]) -> str:
    """Build span name as '{operation} {model}' when model is available."""
    if model:
        return f"{operation} {model}"
    return operation


def parse_server_address(base_url: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Extract server.address and server.port from a base URL."""
    if not base_url:
        return None, None
    try:
        parsed = urlparse(base_url)
        return parsed.hostname, parsed.port
    except Exception:
        return None, None


def build_gen_ai_span_attrs(
    *,
    operation: str,
    provider: str,
    model: Optional[str] = None,
    response_model: Optional[str] = None,
    server_address: Optional[str] = None,
    server_port: Optional[int] = None,
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build base GenAI span attributes."""
    attrs: Dict[str, Any] = {
        gen_ai.GEN_AI_OPERATION_NAME: operation,
        gen_ai.GEN_AI_PROVIDER_NAME: provider,
    }
    if model:
        attrs[gen_ai.GEN_AI_REQUEST_MODEL] = model
    if response_model:
        attrs[gen_ai.GEN_AI_RESPONSE_MODEL] = response_model
    if server_address:
        attrs["server.address"] = server_address
    if server_port is not None:
        attrs["server.port"] = server_port
    if error_type:
        attrs["error.type"] = error_type
    return attrs
