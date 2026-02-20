"""Tests for llamatelemetry.pipeline.spans (PipelineContext, PipelineTracer)."""

import pytest


class TestPipelineContext:
    """Test PipelineContext dataclass."""

    def test_auto_generates_run_id(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx = PipelineContext()
        assert ctx.run_id != ""
        assert len(ctx.run_id) == 36  # UUID4 string

    def test_custom_run_id(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx = PipelineContext(run_id="my-run-001")
        assert ctx.run_id == "my-run-001"

    def test_default_none_fields(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx = PipelineContext()
        assert ctx.base_model is None
        assert ctx.adapter is None
        assert ctx.output_artifact is None
        assert ctx.quantization is None
        assert ctx.model_sha256 is None
        assert ctx.tokenizer_sha256 is None

    def test_with_full_context(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx = PipelineContext(
            run_id="run-001",
            base_model="llama-3-8b",
            adapter="my-lora",
            output_artifact="/models/llama-3-8b-Q4.gguf",
            quantization="Q4_K_M",
            model_sha256="abc123",
        )
        assert ctx.base_model == "llama-3-8b"
        assert ctx.adapter == "my-lora"
        assert ctx.output_artifact == "/models/llama-3-8b-Q4.gguf"
        assert ctx.quantization == "Q4_K_M"
        assert ctx.model_sha256 == "abc123"

    def test_metadata_default_empty(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx = PipelineContext()
        assert ctx.metadata == {}

    def test_metadata_custom(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx = PipelineContext(metadata={"gpu": "T4", "rank": 0})
        assert ctx.metadata["gpu"] == "T4"
        assert ctx.metadata["rank"] == 0

    def test_run_ids_are_unique(self):
        from llamatelemetry.pipeline.spans import PipelineContext
        ctx1 = PipelineContext()
        ctx2 = PipelineContext()
        assert ctx1.run_id != ctx2.run_id


class TestPipelineTracer:
    """Test PipelineTracer span context managers."""

    def test_init(self):
        from llamatelemetry.pipeline.spans import PipelineTracer
        tracer = PipelineTracer()
        assert tracer is not None

    def test_span_merge_lora(self):
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(base_model="llama-3-8b", adapter="my-lora")
        tracer = PipelineTracer()
        with tracer.span_merge_lora(ctx) as span:
            assert span is not None

    def test_span_export_gguf(self):
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(base_model="llama-3-8b", output_artifact="/out/model.gguf")
        tracer = PipelineTracer()
        with tracer.span_export_gguf(ctx) as span:
            assert span is not None

    def test_span_quantize(self):
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(quantization="Q4_K_M")
        tracer = PipelineTracer()
        with tracer.span_quantize(ctx) as span:
            assert span is not None

    def test_span_benchmark(self):
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(base_model="gemma-3-1b")
        tracer = PipelineTracer()
        with tracer.span_benchmark(ctx) as span:
            assert span is not None

    def test_span_finetune(self):
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(base_model="llama-3-8b", adapter="my-lora")
        tracer = PipelineTracer()
        with tracer.span_finetune(ctx) as span:
            assert span is not None

    def test_span_deploy(self):
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(base_model="gemma-3-1b")
        tracer = PipelineTracer()
        with tracer.span_deploy(ctx) as span:
            assert span is not None

    def test_span_with_extra_attributes(self):
        """Extra kwargs should be set on the span without error."""
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(base_model="llama-3-8b")
        tracer = PipelineTracer()
        with tracer.span_merge_lora(ctx, extra_key="extra_value") as span:
            assert span is not None

    def test_full_pipeline_sequence(self):
        """Run all pipeline stages sequentially without errors."""
        from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
        ctx = PipelineContext(
            base_model="llama-3-8b",
            adapter="my-lora",
            quantization="Q4_K_M",
        )
        tracer = PipelineTracer()
        with tracer.span_merge_lora(ctx):
            pass
        with tracer.span_export_gguf(ctx):
            pass
        with tracer.span_quantize(ctx):
            pass
        with tracer.span_benchmark(ctx):
            pass
        with tracer.span_deploy(ctx):
            pass


class TestPipelinePackageImport:
    """Test pipeline package imports."""

    def test_package_importable(self):
        import llamatelemetry.pipeline
        assert llamatelemetry.pipeline is not None

    def test_pipeline_context_importable(self):
        from llamatelemetry.pipeline import PipelineContext
        assert PipelineContext is not None

    def test_pipeline_tracer_importable(self):
        from llamatelemetry.pipeline import PipelineTracer
        assert PipelineTracer is not None
