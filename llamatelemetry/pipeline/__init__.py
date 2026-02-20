"""
llamatelemetry.pipeline - Pipeline observability for the LLM lifecycle.

Provides span creation for the full pipeline:
fine-tune -> merge LoRA -> export GGUF -> quantize -> benchmark -> deploy -> serve
"""

from .spans import PipelineContext, PipelineTracer
