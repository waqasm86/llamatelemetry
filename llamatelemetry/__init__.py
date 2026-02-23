"""
llamatelemetry v2.0 - CUDA-first OpenTelemetry Python SDK for GGUF LLM inference.

Complete built-in integration of:
- llama.cpp (native pybind11 bindings)
- NCCL (native pybind11 bindings)
- OpenTelemetry gen_ai.* semantic conventions (45 attributes, 5 metrics)
- Kaggle dual T4 GPU optimization

Quick start::

    import llamatelemetry
    from llamatelemetry.kaggle_integration import ModelDownloader, KaggleEnvironment

    # Setup Kaggle environment
    env = KaggleEnvironment()
    config = env.setup_llamatelemetry(
        service_name="my-llm-app",
        otlp_endpoint="https://otlp.example.com/v1/traces",
    )

    # Download GGUF model
    downloader = ModelDownloader(cache_dir=env.model_cache_dir)
    model_path = downloader.get_model_by_shortname("llama-2-13b", "Q4_K_M")

    # Create inference engine (GPU-only, multi-GPU support)
    engine = llamatelemetry.create_engine(
        model_path=str(model_path),
        service_name="my-llm-app",
        otlp_endpoint=config['otlp_endpoint'],
        n_gpu_layers=40,  # All layers to GPU
        multi_gpu=True,   # Dual T4 support
    )

    # Generate with automatic observability
    response = engine.generate(
        prompt="Explain GGUF quantization.",
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        conversation_id="session_123",
    )

    print(f"Output: {response.text}")
    print(f"Metrics: TTFT={response.ttft_ms:.1f}ms, TPOT={response.tpot_ms:.2f}ms")

    engine.shutdown()
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# v2.0 - Native bindings for 100% built-in integration
# ---------------------------------------------------------------------------

# Submodules
from . import llama_cpp_native  # noqa: F401
from . import nccl_native  # noqa: F401
from . import otel_gen_ai  # noqa: F401
from . import kaggle_integration  # noqa: F401

# High-level inference engine with full observability
from .inference_engine import InferenceEngine, create_engine, GenerateResponse  # noqa: F401

# Core native APIs - llama.cpp
from .llama_cpp_native import (  # noqa: F401
    LlamaModel,
    LlamaContext,
    LlamaBatch,
    SamplerChain,
    SamplerType,
    Tokenizer,
    InferenceLoop,
)

# Core native APIs - NCCL
from .nccl_native import (  # noqa: F401
    NCCLCommunicator,
    DataType,
    ReductionOp,
    ResultCode,
)

# Core native APIs - OpenTelemetry Gen AI
from .otel_gen_ai import (  # noqa: F401
    GenAITracer,
    InferenceContext,
    GPUMonitor,
)

# Core native APIs - Kaggle Integration
from .kaggle_integration import (  # noqa: F401
    ModelDownloader,
    KaggleGPUConfig,
    KaggleEnvironment,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "2.0.0"
__binary_version__ = "2.0.0"

# ---------------------------------------------------------------------------
# __all__ - Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Version
    "__version__",
    "__binary_version__",

    # High-level API
    "InferenceEngine",
    "create_engine",
    "GenerateResponse",

    # llama.cpp native APIs
    "LlamaModel",
    "LlamaContext",
    "LlamaBatch",
    "SamplerChain",
    "SamplerType",
    "Tokenizer",
    "InferenceLoop",

    # NCCL native APIs
    "NCCLCommunicator",
    "DataType",
    "ReductionOp",
    "ResultCode",

    # OpenTelemetry Gen AI APIs
    "GenAITracer",
    "InferenceContext",
    "GPUMonitor",

    # Kaggle integration APIs
    "ModelDownloader",
    "KaggleGPUConfig",
    "KaggleEnvironment",

    # Submodules
    "llama_cpp_native",
    "nccl_native",
    "otel_gen_ai",
    "kaggle_integration",
]
