"""
llamatelemetry.pipeline.spans - Pipeline span creation for LLM lifecycle operations.

Creates OTel spans for pipeline operations:
    - merge_lora
    - export_gguf
    - quantize
    - benchmark
    - finetune

Each operation carries a PipelineContext with run_id, base_model, adapter, etc.
so the full lifecycle can be correlated in dashboards.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..otel.provider import get_tracer
from ..semconv import keys


@dataclass
class PipelineContext:
    """Context for pipeline operations, enabling lifecycle correlation.

    Attributes:
        run_id: Unique pipeline run identifier (auto-generated if not provided).
        base_model: Base model name or path.
        adapter: Adapter name or path (e.g. LoRA adapter).
        output_artifact: Output artifact path (e.g. GGUF file).
        quantization: Quantization type (e.g. Q4_K_M).
        model_sha256: SHA-256 of the output model artifact.
        tokenizer_sha256: SHA-256 of the tokenizer.
        metadata: Additional pipeline metadata.
    """

    run_id: str = ""
    base_model: Optional[str] = None
    adapter: Optional[str] = None
    output_artifact: Optional[str] = None
    quantization: Optional[str] = None
    model_sha256: Optional[str] = None
    tokenizer_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid.uuid4())


class PipelineTracer:
    """Creates OTel spans for pipeline lifecycle operations.

    Example:
        >>> tracer = PipelineTracer()
        >>> ctx = PipelineContext(base_model="llama-3-8b", adapter="my-lora")
        >>> with tracer.span_merge_lora(ctx):
        ...     merged = merge_adapters(model)
        >>> with tracer.span_export_gguf(ctx):
        ...     export_to_gguf(merged, "output.gguf")
    """

    def __init__(self, tracer: Optional[Any] = None):
        """Initialize pipeline tracer.

        Args:
            tracer: OTel tracer instance. Auto-obtained if None.
        """
        self._tracer = tracer or get_tracer("llamatelemetry.pipeline")

    @contextmanager
    def span_finetune(self, ctx: PipelineContext, **extra: Any):
        """Create a span for fine-tuning operations."""
        with self._tracer.start_as_current_span("gen_ai.pipeline.finetune") as span:
            self._set_context_attrs(span, ctx)
            for k, v in extra.items():
                span.set_attribute(k, v)
            yield span

    @contextmanager
    def span_merge_lora(self, ctx: PipelineContext, **extra: Any):
        """Create a span for LoRA adapter merging."""
        with self._tracer.start_as_current_span("gen_ai.pipeline.merge_lora") as span:
            self._set_context_attrs(span, ctx)
            if ctx.adapter:
                span.set_attribute("pipeline.adapter", ctx.adapter)
            for k, v in extra.items():
                span.set_attribute(k, v)
            yield span

    @contextmanager
    def span_export_gguf(self, ctx: PipelineContext, **extra: Any):
        """Create a span for GGUF export."""
        with self._tracer.start_as_current_span("gen_ai.pipeline.export_gguf") as span:
            self._set_context_attrs(span, ctx)
            if ctx.output_artifact:
                span.set_attribute("pipeline.output_artifact", ctx.output_artifact)
            for k, v in extra.items():
                span.set_attribute(k, v)
            yield span

    @contextmanager
    def span_quantize(self, ctx: PipelineContext, **extra: Any):
        """Create a span for quantization."""
        with self._tracer.start_as_current_span("gen_ai.pipeline.quantize") as span:
            self._set_context_attrs(span, ctx)
            if ctx.quantization:
                span.set_attribute(keys.LLM_QUANT, ctx.quantization)
            for k, v in extra.items():
                span.set_attribute(k, v)
            yield span

    @contextmanager
    def span_benchmark(self, ctx: PipelineContext, **extra: Any):
        """Create a span for benchmark operations."""
        with self._tracer.start_as_current_span("gen_ai.pipeline.benchmark") as span:
            self._set_context_attrs(span, ctx)
            for k, v in extra.items():
                span.set_attribute(k, v)
            yield span

    @contextmanager
    def span_deploy(self, ctx: PipelineContext, **extra: Any):
        """Create a span for deployment operations."""
        with self._tracer.start_as_current_span("gen_ai.pipeline.deploy") as span:
            self._set_context_attrs(span, ctx)
            for k, v in extra.items():
                span.set_attribute(k, v)
            yield span

    def _set_context_attrs(self, span: Any, ctx: PipelineContext) -> None:
        """Set common pipeline context attributes on a span."""
        span.set_attribute(keys.RUN_ID, ctx.run_id)
        span.set_attribute(keys.LLM_SYSTEM, "llamatelemetry")
        if ctx.base_model:
            span.set_attribute(keys.LLM_MODEL, ctx.base_model)
        if ctx.model_sha256:
            span.set_attribute(keys.LLM_GGUF_SHA256, ctx.model_sha256)
        if ctx.quantization:
            span.set_attribute(keys.LLM_QUANT, ctx.quantization)
        for k, v in ctx.metadata.items():
            span.set_attribute(f"pipeline.{k}", v)
