"""
llamatelemetry.backends.llamacpp - LlamaCpp backend adapter.

Wraps the existing LlamaCppClient into the unified LLMBackend interface.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .base import LLMBackend, LLMRequest, LLMResponse


class LlamaCppBackend:
    """LLM backend for llama.cpp server (OpenAI-compatible).

    Wraps llamatelemetry's existing LlamaCppClient to conform to the
    unified LLMBackend interface.

    Example:
        >>> backend = LlamaCppBackend("http://127.0.0.1:8090")
        >>> req = LLMRequest(
        ...     operation="chat",
        ...     model="gemma-3-1b",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> resp = backend.invoke(req)
        >>> print(resp.output_text)
    """

    name = "llama.cpp"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize the llama.cpp backend.

        Args:
            base_url: llama.cpp server URL.
            api_key: Optional API key.
            timeout_s: Request timeout in seconds.
        """
        from ..api.client import LlamaCppClient
        from ..utils import require_cuda

        require_cuda()

        self._client = LlamaCppClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
        )
        self._base_url = base_url

    def invoke(self, req: LLMRequest) -> LLMResponse:
        """Execute an LLM request via the llama.cpp server.

        Supports chat, completions, and embeddings operations.
        """
        start = time.perf_counter()

        if req.operation == "chat":
            return self._invoke_chat(req, start)
        elif req.operation == "completions" or req.operation == "text_completion":
            return self._invoke_completions(req, start)
        elif req.operation == "embeddings":
            return self._invoke_embeddings(req, start)
        else:
            raise ValueError(f"Unsupported operation: {req.operation}")

    def _invoke_chat(self, req: LLMRequest, start: float) -> LLMResponse:
        """Handle chat completion requests."""
        kwargs: dict = {}
        if req.model:
            kwargs["model"] = req.model
        if req.parameters:
            kwargs.update(req.parameters)
        if req.stream:
            kwargs["stream"] = True

        messages = req.messages or []
        result = self._client.chat.create(messages=messages, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0

        output_text = None
        finish_reason = None
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            output_text = getattr(choice, "message", {}).get("content", "")
            finish_reason = getattr(choice, "finish_reason", None)

        input_tokens = None
        output_tokens = None
        if hasattr(result, "usage") and result.usage is not None:
            input_tokens = getattr(result.usage, "prompt_tokens", None)
            output_tokens = getattr(result.usage, "completion_tokens", None)

        response_id = getattr(result, "id", None)
        response_model = getattr(result, "model", None)

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            response_id=response_id,
            response_model=response_model,
            latency_ms=elapsed,
            raw=result,
        )

    def _invoke_completions(self, req: LLMRequest, start: float) -> LLMResponse:
        """Handle text completion requests."""
        kwargs: dict = {}
        if req.model:
            kwargs["model"] = req.model
        if req.parameters:
            kwargs.update(req.parameters)

        prompt = req.prompt or ""
        result = self._client.completions(prompt=prompt, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0

        output_text = getattr(result, "content", None)

        return LLMResponse(
            output_text=output_text,
            latency_ms=elapsed,
            raw=result,
        )

    def _invoke_embeddings(self, req: LLMRequest, start: float) -> LLMResponse:
        """Handle embedding requests."""
        texts = req.input_texts or []
        result = self._client.embeddings.create(input=texts)
        elapsed = (time.perf_counter() - start) * 1000.0

        return LLMResponse(
            output_texts=[],
            latency_ms=elapsed,
            raw=result,
        )
