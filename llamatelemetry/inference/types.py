"""
llamatelemetry.inference.types - Dataclasses for inference configuration.

Provides SamplingParams, BatchConstraints, and DeviceConfig used by engines
and the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.

    Attributes:
        temperature: Sampling temperature (0.0 = greedy).
        top_p: Top-p (nucleus) sampling.
        top_k: Top-k sampling.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        seed: Random seed for reproducibility.
        stop_sequences: Sequences that stop generation.
        repetition_penalty: Repetition penalty factor.
    """

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    repetition_penalty: float = 1.0


@dataclass
class BatchConstraints:
    """Constraints for batching and scheduling.

    Attributes:
        max_batch_size: Maximum number of requests in a batch.
        max_batch_tokens: Maximum total tokens in a batch.
        max_wait_ms: Maximum time to wait for batch formation.
        max_concurrent_sessions: Maximum concurrent active sessions.
    """

    max_batch_size: int = 8
    max_batch_tokens: int = 4096
    max_wait_ms: float = 50.0
    max_concurrent_sessions: int = 32


@dataclass
class DeviceConfig:
    """GPU device configuration.

    Attributes:
        device_ids: List of GPU device indices to use.
        primary_device: Primary device for single-GPU operations.
        dtype: Data type ("fp16", "bf16", "fp32").
        attention_backend: Attention implementation ("flash_attn", "sdpa", "eager").
        use_torch_compile: Enable torch.compile for decode.
        use_cuda_graphs: Enable CUDA graph capture for decode.
    """

    device_ids: List[int] = field(default_factory=lambda: [0])
    primary_device: int = 0
    dtype: str = "fp16"
    attention_backend: str = "sdpa"
    use_torch_compile: bool = False
    use_cuda_graphs: bool = False


@dataclass
class EngineStats:
    """Runtime statistics from an inference engine.

    Attributes:
        total_requests: Total requests processed.
        total_tokens_generated: Total tokens generated.
        avg_ttft_ms: Average time to first token.
        avg_tpot_ms: Average time per output token.
        avg_tps: Average tokens per second.
        peak_vram_mb: Peak VRAM usage observed.
        active_sessions: Currently active sessions.
    """

    total_requests: int = 0
    total_tokens_generated: int = 0
    avg_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_tps: float = 0.0
    peak_vram_mb: float = 0.0
    active_sessions: int = 0
