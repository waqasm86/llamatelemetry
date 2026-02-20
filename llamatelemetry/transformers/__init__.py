"""
llamatelemetry.transformers - Transformers/original model backend and instrumentation.

Provides runtime inference instrumentation for HuggingFace Transformers models,
with the same span hierarchy and gen_ai.* attributes as the llama.cpp backend.

All dependencies (torch, transformers) are lazy-imported and optional.
"""

from .backend import TransformersBackend
from .instrumentation import InstrumentedBackend, TransformersInstrumentorConfig
