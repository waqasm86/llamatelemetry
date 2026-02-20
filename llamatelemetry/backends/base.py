"""
llamatelemetry.backends.base - Unified backend interface for GGUF + original models.

Defines the internal contract that both llama.cpp and Transformers backends implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMRequest:
    """Standardized LLM request object for any backend.

    Attributes:
        operation: Operation type - "chat", "text_completion", "embeddings".
        model: Model name or identifier.
        provider: Provider name (e.g. "llama_cpp", "transformers").
        messages: Chat messages (for chat operations).
        prompt: Raw prompt string (for completions).
        input_texts: Input texts (for embeddings).
        parameters: Additional request parameters (temperature, top_p, etc.).
        request_id: Optional request identifier.
        conversation_id: Optional conversation/session identifier.
        stream: Whether to stream the response.
    """

    operation: str = "chat"
    model: Optional[str] = None
    provider: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None
    input_texts: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory=dict)
    request_id: Optional[str] = None
    conversation_id: Optional[str] = None
    stream: bool = False


@dataclass
class LLMResponse:
    """Standardized LLM response object from any backend.

    Attributes:
        output_text: Generated text (single response).
        output_texts: Generated texts (multiple responses or embeddings).
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        finish_reason: Why generation stopped.
        response_id: Backend-provided response identifier.
        response_model: Actual model used for generation.
        latency_ms: Request latency in milliseconds.
        raw: Raw backend response for advanced use.
    """

    output_text: Optional[str] = None
    output_texts: Optional[List[str]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    latency_ms: Optional[float] = None
    raw: Optional[Any] = None


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends (llama.cpp, Transformers, etc.).

    Any backend must implement:
        - name: A string identifier for the backend.
        - invoke(): Takes an LLMRequest, returns an LLMResponse.
    """

    name: str

    def invoke(self, req: LLMRequest) -> LLMResponse:
        """Execute an LLM request and return a standardized response."""
        ...
