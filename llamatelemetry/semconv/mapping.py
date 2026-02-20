"""
llamatelemetry.semconv.mapping - Mapper for gen_ai.* attributes.

Converts backend payloads into OpenTelemetry GenAI semantic convention attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import gen_ai


@dataclass(frozen=True)
class GenAIAttrs:
    """Structured container for GenAI semantic convention attributes."""

    provider: Optional[str] = None
    model: Optional[str] = None
    operation: Optional[str] = None
    response_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reasons: Optional[list] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop_sequences: Optional[list] = None
    stream: Optional[bool] = None
    response_model: Optional[str] = None
    conversation_id: Optional[str] = None


def to_gen_ai_attrs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a backend payload dict to official gen_ai.* attributes.

    Args:
        payload: Dictionary with keys like 'model', 'operation', 'input_tokens', etc.

    Returns:
        Dictionary of gen_ai.* attributes (only non-None values).
    """
    mapping = {
        "provider": gen_ai.GEN_AI_PROVIDER_NAME,
        "model": gen_ai.GEN_AI_REQUEST_MODEL,
        "operation": gen_ai.GEN_AI_OPERATION_NAME,
        "response_id": gen_ai.GEN_AI_RESPONSE_ID,
        "input_tokens": gen_ai.GEN_AI_USAGE_INPUT_TOKENS,
        "output_tokens": gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
        "finish_reasons": gen_ai.GEN_AI_RESPONSE_FINISH_REASONS,
        "temperature": gen_ai.GEN_AI_REQUEST_TEMPERATURE,
        "top_p": gen_ai.GEN_AI_REQUEST_TOP_P,
        "top_k": gen_ai.GEN_AI_REQUEST_TOP_K,
        "max_tokens": gen_ai.GEN_AI_REQUEST_MAX_TOKENS,
        "frequency_penalty": gen_ai.GEN_AI_REQUEST_FREQUENCY_PENALTY,
        "presence_penalty": gen_ai.GEN_AI_REQUEST_PRESENCE_PENALTY,
        "seed": gen_ai.GEN_AI_REQUEST_SEED,
        "stop_sequences": gen_ai.GEN_AI_REQUEST_STOP_SEQUENCES,
        "response_model": gen_ai.GEN_AI_RESPONSE_MODEL,
        "conversation_id": gen_ai.GEN_AI_CONVERSATION_ID,
    }

    result: Dict[str, Any] = {}
    for key, attr_name in mapping.items():
        value = payload.get(key)
        if value is not None:
            result[attr_name] = value
    return result


def set_gen_ai_attrs(span: Any, payload: Dict[str, Any]) -> None:
    """
    Set gen_ai.* attributes on a span.

    Args:
        span: An OpenTelemetry span (or noop span).
        payload: Dictionary with attribute values.
    """
    attrs = to_gen_ai_attrs(payload)
    for k, v in attrs.items():
        span.set_attribute(k, v)
