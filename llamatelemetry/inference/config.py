"""
llamatelemetry.inference.config - CUDA inference configuration.

Provides CudaInferenceConfig as the single configuration object for
controlling inference behavior across both GGUF and Transformers engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import SamplingParams, BatchConstraints, DeviceConfig


@dataclass
class CudaInferenceConfig:
    """Configuration for CUDA inference engines.

    This is the "OTEL provider config equivalent" for GPU inference.

    Attributes:
        backend: Backend type ("llama.cpp" or "transformers").
        model_path: Path to model file (GGUF) or HuggingFace model ID.
        device: Device configuration.
        sampling: Default sampling parameters.
        batch: Batching constraints.
        dtype: Data type for computation ("fp16", "bf16", "fp32").
        attention: Attention implementation ("flash_attn", "sdpa", "eager").
        use_torch_compile: Enable torch.compile optimization.
        use_cuda_graphs: Enable CUDA graph capture for decode steps.
        max_batch_tokens: Maximum tokens per batch.
        max_concurrent_sessions: Maximum concurrent sessions.
        kv_cache_policy: KV cache eviction policy ("lru", "fifo", "session_pin").
        prefill_chunk_size: Chunk size for prefill splitting.
        streaming: Enable streaming by default.
        telemetry: Enable OTel telemetry.
        scheduler: Enable request scheduler.
        multi_gpu: Multi-GPU mode ("auto", "single", "tensor_parallel", "split").
        extra: Additional engine-specific configuration.

        transformers_device_map: Optional device_map ("auto" or dict).
        transformers_max_memory: Optional max_memory dict per GPU.
        transformers_load_in_4bit: Enable 4-bit loading (bitsandbytes).
        transformers_load_in_8bit: Enable 8-bit loading (bitsandbytes).
        transformers_bnb_4bit_compute_dtype: Compute dtype for 4-bit.
        transformers_bnb_4bit_quant_type: Quant type for 4-bit (e.g. "nf4").
        transformers_bnb_4bit_use_double_quant: Enable double quantization.
        transformers_trust_remote_code: Enable trust_remote_code in HF loader.
    """

    backend: str = "llama.cpp"
    model_path: Optional[str] = None
    device: DeviceConfig = field(default_factory=DeviceConfig)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    batch: BatchConstraints = field(default_factory=BatchConstraints)
    dtype: str = "fp16"
    attention: str = "sdpa"
    use_torch_compile: bool = False
    use_cuda_graphs: bool = False
    max_batch_tokens: int = 4096
    max_concurrent_sessions: int = 32
    kv_cache_policy: str = "lru"
    prefill_chunk_size: int = 512
    streaming: bool = False
    telemetry: bool = True
    scheduler: bool = False
    multi_gpu: str = "auto"
    extra: Dict[str, Any] = field(default_factory=dict)

    # llama.cpp specific
    llama_server_url: Optional[str] = None
    llama_n_ctx: int = 2048
    llama_n_batch: int = 512
    llama_n_ubatch: int = 512
    llama_n_gpu_layers: int = -1
    llama_mmap: bool = True
    llama_mlock: bool = False
    # transformers specific
    transformers_device_map: Optional[Any] = None
    transformers_max_memory: Optional[Dict[str, str]] = None
    transformers_load_in_4bit: bool = False
    transformers_load_in_8bit: bool = False
    transformers_bnb_4bit_compute_dtype: Optional[str] = None
    transformers_bnb_4bit_quant_type: Optional[str] = None
    transformers_bnb_4bit_use_double_quant: bool = False
    transformers_trust_remote_code: bool = False

    def to_device_config(self) -> DeviceConfig:
        """Convert to DeviceConfig."""
        return DeviceConfig(
            device_ids=self.device.device_ids,
            primary_device=self.device.primary_device,
            dtype=self.dtype,
            attention_backend=self.attention,
            use_torch_compile=self.use_torch_compile,
            use_cuda_graphs=self.use_cuda_graphs,
        )
