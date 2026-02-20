"""
llamatelemetry.semconv.gen_ai_builder - Builder functions for GenAI span attributes.

Extracts gen_ai.* attributes from request/response objects in a structured way.
Supports both llama.cpp (OpenAI-compatible) and Transformers backends.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from . import gen_ai


def build_gen_ai_attrs_from_request(
    model: str,
    operation: str = gen_ai.OP_CHAT,
    provider: str = gen_ai.PROVIDER_LLAMA_CPP,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[float] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    choice_count: Optional[int] = None,
    stream: Optional[bool] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build gen_ai.* request attributes.

    Args:
        model: Model name (e.g. 'gemma-3-1b-it-Q4_K_M').
        operation: Operation name (chat, text_completion, embeddings).
        provider: Provider name (llama_cpp, transformers, openai).
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        top_k: Top-k sampling.
        max_tokens: Maximum tokens to generate.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        seed: Random seed.
        stop_sequences: Stop sequences.
        choice_count: Number of choices requested.
        stream: Whether streaming is enabled.
        conversation_id: Conversation/session ID.

    Returns:
        Dictionary of gen_ai.* request attributes.
    """
    attrs: Dict[str, Any] = {
        gen_ai.GEN_AI_OPERATION_NAME: operation,
        gen_ai.GEN_AI_PROVIDER_NAME: provider,
        gen_ai.GEN_AI_REQUEST_MODEL: model,
    }

    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_TEMPERATURE, temperature)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_TOP_P, top_p)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_TOP_K, top_k)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_FREQUENCY_PENALTY, frequency_penalty)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_SEED, seed)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences)
    _set_if_not_none(attrs, gen_ai.GEN_AI_REQUEST_CHOICE_COUNT, choice_count)
    _set_if_not_none(attrs, gen_ai.GEN_AI_CONVERSATION_ID, conversation_id)

    return attrs


def build_gen_ai_attrs_from_response(
    response_id: Optional[str] = None,
    response_model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    finish_reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build gen_ai.* response attributes.

    Args:
        response_id: Response ID from the backend.
        response_model: Actual model used for generation.
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        finish_reasons: List of finish reasons.

    Returns:
        Dictionary of gen_ai.* response attributes.
    """
    attrs: Dict[str, Any] = {}

    _set_if_not_none(attrs, gen_ai.GEN_AI_RESPONSE_ID, response_id)
    _set_if_not_none(attrs, gen_ai.GEN_AI_RESPONSE_MODEL, response_model)
    _set_if_not_none(attrs, gen_ai.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    _set_if_not_none(attrs, gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    _set_if_not_none(attrs, gen_ai.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    return attrs


def build_gen_ai_attrs_from_tools(
    tool_definitions: Optional[List[Dict[str, Any]]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    record_content: bool = False,
) -> Dict[str, Any]:
    """
    Build gen_ai.* tool attributes.

    Args:
        tool_definitions: List of tool definition objects.
        tool_calls: List of tool call objects.
        record_content: Whether to record tool content (OFF by default for privacy).

    Returns:
        Dictionary of gen_ai.* tool attributes.
    """
    attrs: Dict[str, Any] = {}

    if record_content and tool_definitions:
        attrs[gen_ai.GEN_AI_TOOL_DEFINITIONS] = json.dumps(tool_definitions)

    if tool_calls and record_content:
        for call in tool_calls:
            if "id" in call:
                attrs[gen_ai.GEN_AI_TOOL_CALL_ID] = call["id"]
            if "name" in call:
                attrs[gen_ai.GEN_AI_TOOL_NAME] = call["name"]
            if "type" in call:
                attrs[gen_ai.GEN_AI_TOOL_TYPE] = call["type"]
            if "arguments" in call:
                attrs[gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS] = json.dumps(call["arguments"])
            if "result" in call:
                attrs[gen_ai.GEN_AI_TOOL_CALL_RESULT] = json.dumps(call["result"])

    return attrs


def build_content_attrs(
    input_messages: Optional[List[Dict[str, Any]]] = None,
    output_messages: Optional[List[Dict[str, Any]]] = None,
    system_instructions: Optional[List[Dict[str, Any]]] = None,
    record_content: bool = False,
    record_content_max_chars: int = 2000,
) -> Dict[str, Any]:
    """
    Build gen_ai.* content attributes (PII-sensitive, OFF by default).

    Args:
        input_messages: Input chat messages.
        output_messages: Output messages.
        system_instructions: System instructions.
        record_content: Whether to record content (OFF by default).
        record_content_max_chars: Maximum characters to record.

    Returns:
        Dictionary of gen_ai.* content attributes, or hashes if content recording is off.
    """
    attrs: Dict[str, Any] = {}

    if record_content:
        if input_messages:
            content = json.dumps(input_messages)[:record_content_max_chars]
            attrs[gen_ai.GEN_AI_INPUT_MESSAGES] = content
        if output_messages:
            content = json.dumps(output_messages)[:record_content_max_chars]
            attrs[gen_ai.GEN_AI_OUTPUT_MESSAGES] = content
        if system_instructions:
            content = json.dumps(system_instructions)[:record_content_max_chars]
            attrs[gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS] = content
    else:
        # Record SHA-256 hashes instead of content for debuggability without PII
        if input_messages:
            attrs["llamatelemetry.prompt.sha256"] = _sha256(json.dumps(input_messages))
        if output_messages:
            attrs["llamatelemetry.output.sha256"] = _sha256(json.dumps(output_messages))

    return attrs


def _set_if_not_none(attrs: Dict[str, Any], key: str, value: Any) -> None:
    """Set a key in attrs dict only if value is not None."""
    if value is not None:
        attrs[key] = value


def _sha256(text: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
