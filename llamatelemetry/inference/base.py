"""
llamatelemetry.inference.base - CUDA inference contract.

Defines the unified interface that both llama.cpp and Transformers engines implement.
One contract, two runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

from .types import SamplingParams, DeviceConfig


@dataclass
class InferenceRequest:
    """Unified inference request for any engine.

    Attributes:
        prompt: Raw prompt string.
        messages: Chat messages (for chat operations).
        sampling: Sampling parameters.
        device: Device configuration.
        max_tokens: Maximum tokens to generate.
        stream: Whether to stream output.
        request_id: Optional request identifier.
        conversation_id: Optional conversation ID.
        metadata: Additional request metadata.
    """

    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    sampling: Optional[SamplingParams] = None
    device: Optional[DeviceConfig] = None
    max_tokens: int = 256
    stream: bool = False
    request_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Unified inference result from any engine.

    Contains both the output text and comprehensive performance metrics.

    Attributes:
        output_text: Generated text.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        ttft_ms: Time to first token in milliseconds.
        tpot_ms: Time per output token in milliseconds.
        tps: Tokens per second (steady state).
        prefill_tps: Prefill tokens per second.
        total_latency_ms: Total request latency in milliseconds.
        vram_peak_mb: Peak VRAM usage during request.
        vram_delta_mb: VRAM change during request.
        queue_delay_ms: Time spent waiting in scheduler queue.
        kv_cache_bytes: KV cache bytes used for this request.
        finish_reason: Why generation stopped.
        request_id: Request identifier.
        raw: Raw engine-specific result.
    """

    output_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    tps: float = 0.0
    prefill_tps: float = 0.0
    total_latency_ms: float = 0.0
    vram_peak_mb: float = 0.0
    vram_delta_mb: float = 0.0
    queue_delay_ms: float = 0.0
    kv_cache_bytes: int = 0
    finish_reason: str = "stop"
    request_id: Optional[str] = None
    raw: Optional[Any] = None


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol for CUDA inference engines.

    Any engine (llama.cpp, Transformers) must implement:
        - name: Engine identifier.
        - warmup(): Prepare the engine (load model, allocate memory).
        - generate(): Synchronous generation.
        - stream_generate(): Streaming generation (yields token chunks).
        - shutdown(): Release resources.
    """

    name: str

    def warmup(self) -> None:
        """Prepare the engine for inference (load model, allocate buffers)."""
        ...

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Execute synchronous inference."""
        ...

    def stream_generate(self, request: InferenceRequest) -> Iterator[str]:
        """Execute streaming inference, yielding token chunks."""
        ...

    def shutdown(self) -> None:
        """Release engine resources."""
        ...
