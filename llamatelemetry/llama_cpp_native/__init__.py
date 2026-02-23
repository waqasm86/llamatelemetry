"""
llamatelemetry.llama_cpp_native - Direct native C++ bindings to llama.cpp

This module provides complete native integration with llama.cpp via pybind11,
eliminating the need for HTTP servers and enabling GPU-only inference.

Features:
  - Direct model loading and inference
  - Native batch processing
  - Composable sampling pipelines
  - KV cache management
  - Multi-GPU support
  - Built-in GGUF quantization awareness
"""

from .model import LlamaModel
from .context import LlamaContext
from .batch import LlamaBatch
from .sampler import SamplerChain, SamplerType
from .tokenizer import Tokenizer
from .inference import InferenceLoop

__all__ = [
    'LlamaModel',
    'LlamaContext',
    'LlamaBatch',
    'SamplerChain',
    'SamplerType',
    'Tokenizer',
    'InferenceLoop',
]

__version__ = '2.0.0'
