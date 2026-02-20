"""
llamatelemetry.inference.engines.llamacpp_engine - llama.cpp inference engine.

Wraps the llama.cpp server into a production engine with consistent metrics
(TTFT, TPOT, TPS, VRAM peak).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional

from ..base import InferenceEngine, InferenceRequest, InferenceResult
from ..events import InferenceEvents
from ..metrics import compute_all_metrics
from ..config import CudaInferenceConfig


class LlamaCppEngine:
    """Inference engine for llama.cpp server (GGUF models).

    Wraps the existing llama.cpp OpenAI-compatible client path into the
    unified InferenceEngine interface with production-grade metrics.

    Example:
        >>> engine = LlamaCppEngine.from_config(CudaInferenceConfig(
        ...     backend="llama.cpp",
        ...     llama_server_url="http://127.0.0.1:8090",
        ... ))
        >>> engine.warmup()
        >>> result = engine.generate(InferenceRequest(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=128,
        ... ))
        >>> print(f"TTFT: {result.ttft_ms:.1f}ms, TPS: {result.tps:.1f}")
    """

    name = "llama.cpp"

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8090",
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        config: Optional[CudaInferenceConfig] = None,
    ):
        """Initialize llama.cpp engine.

        Args:
            server_url: llama.cpp server URL.
            api_key: Optional API key.
            timeout_s: Request timeout.
            config: Full inference configuration.
        """
        self._server_url = server_url
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._config = config
        self._client = None

    @classmethod
    def from_config(cls, config: CudaInferenceConfig) -> "LlamaCppEngine":
        """Create engine from configuration."""
        return cls(
            server_url=config.llama_server_url or "http://127.0.0.1:8090",
            config=config,
        )

    def warmup(self) -> None:
        """Initialize the llama.cpp client and verify server connectivity."""
        from ...api.client import LlamaCppClient
        from ...utils import require_cuda

        require_cuda()

        self._client = LlamaCppClient(
            base_url=self._server_url,
            api_key=self._api_key,
            timeout=self._timeout_s,
        )

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Execute synchronous inference via llama.cpp server."""
        if self._client is None:
            self.warmup()

        events = InferenceEvents()
        events.mark_start()

        # Build request kwargs
        kwargs: Dict[str, Any] = {}
        if request.sampling:
            if request.sampling.temperature != 0.7:
                kwargs["temperature"] = request.sampling.temperature
            if request.sampling.top_p != 1.0:
                kwargs["top_p"] = request.sampling.top_p
            if request.sampling.top_k > 0:
                kwargs["top_k"] = request.sampling.top_k
            if request.sampling.frequency_penalty != 0.0:
                kwargs["frequency_penalty"] = request.sampling.frequency_penalty
            if request.sampling.presence_penalty != 0.0:
                kwargs["presence_penalty"] = request.sampling.presence_penalty
            if request.sampling.repetition_penalty != 1.0:
                kwargs["repeat_penalty"] = request.sampling.repetition_penalty
            if request.sampling.seed is not None:
                kwargs["seed"] = request.sampling.seed
            if request.sampling.stop_sequences:
                kwargs["stop"] = request.sampling.stop_sequences
        kwargs["max_tokens"] = request.max_tokens

        # Execute
        messages = request.messages or []
        if not messages and request.prompt:
            messages = [{"role": "user", "content": request.prompt}]

        result = self._client.chat.create(messages=messages, **kwargs)

        events.mark_first_token()

        # Extract response data
        output_text = ""
        finish_reason = "stop"
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            msg = getattr(choice, "message", None)
            if msg:
                output_text = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        input_tokens = 0
        output_tokens = 0
        if hasattr(result, "usage") and result.usage:
            input_tokens = getattr(result.usage, "prompt_tokens", 0)
            output_tokens = getattr(result.usage, "completion_tokens", 0)
        if hasattr(result, "timings") and result.timings:
            input_tokens = input_tokens or getattr(result.timings, "prompt_n", 0)
            output_tokens = output_tokens or getattr(result.timings, "predicted_n", 0)

        events.mark_last_token()
        events.mark_complete()
        events.set_token_counts(input_tokens, output_tokens)

        # Compute metrics
        all_metrics = compute_all_metrics(events)

        # Prefer server-reported timings if available
        if hasattr(result, "timings") and result.timings:
            prompt_ms = getattr(result.timings, "prompt_ms", 0.0) or 0.0
            predicted_ms = getattr(result.timings, "predicted_ms", 0.0) or 0.0
            total_ms = prompt_ms + predicted_ms if (prompt_ms or predicted_ms) else all_metrics["total_latency_ms"]

            if prompt_ms > 0:
                all_metrics["ttft_ms"] = prompt_ms
                if input_tokens > 0:
                    all_metrics["prefill_tps"] = input_tokens / (prompt_ms / 1000.0)

            if predicted_ms > 0 and output_tokens > 0:
                all_metrics["tps"] = output_tokens / (predicted_ms / 1000.0)
                denom = max(output_tokens - 1, 1)
                all_metrics["tpot_ms"] = (predicted_ms / denom)

            all_metrics["total_latency_ms"] = total_ms

        return InferenceResult(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=all_metrics["ttft_ms"],
            tpot_ms=all_metrics["tpot_ms"],
            tps=all_metrics["tps"],
            prefill_tps=all_metrics["prefill_tps"],
            total_latency_ms=all_metrics["total_latency_ms"],
            queue_delay_ms=all_metrics["queue_delay_ms"],
            finish_reason=finish_reason,
            request_id=request.request_id,
            raw=result,
        )

    def stream_generate(self, request: InferenceRequest) -> Iterator[str]:
        """Execute streaming inference (yields token chunks)."""
        if self._client is None:
            self.warmup()

        messages = request.messages or []
        if not messages and request.prompt:
            messages = [{"role": "user", "content": request.prompt}]

        kwargs: Dict[str, Any] = {"stream": True, "max_tokens": request.max_tokens}
        if request.sampling:
            if request.sampling.temperature != 0.7:
                kwargs["temperature"] = request.sampling.temperature
            if request.sampling.top_p != 1.0:
                kwargs["top_p"] = request.sampling.top_p
            if request.sampling.top_k > 0:
                kwargs["top_k"] = request.sampling.top_k
            if request.sampling.frequency_penalty != 0.0:
                kwargs["frequency_penalty"] = request.sampling.frequency_penalty
            if request.sampling.presence_penalty != 0.0:
                kwargs["presence_penalty"] = request.sampling.presence_penalty
            if request.sampling.repetition_penalty != 1.0:
                kwargs["repeat_penalty"] = request.sampling.repetition_penalty
            if request.sampling.seed is not None:
                kwargs["seed"] = request.sampling.seed
            if request.sampling.stop_sequences:
                kwargs["stop"] = request.sampling.stop_sequences

        result = self._client.chat.create(messages=messages, **kwargs)

        # If result is iterable (streaming), yield chunks
        if hasattr(result, "__iter__"):
            for chunk in result:
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta") or choices[0].get("message")
                        if isinstance(delta, dict):
                            content = delta.get("content")
                            if content:
                                yield content
                        else:
                            content = getattr(delta, "content", None)
                            if content:
                                yield content
                        text = choices[0].get("text")
                        if text:
                            yield text
                elif hasattr(chunk, "choices") and chunk.choices:
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta and hasattr(delta, "content") and delta.content:
                        yield delta.content
        else:
            # Fallback: non-streaming response
            if hasattr(result, "choices") and result.choices:
                msg = getattr(result.choices[0], "message", None)
                if msg:
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    yield content

    def shutdown(self) -> None:
        """Release engine resources."""
        self._client = None
