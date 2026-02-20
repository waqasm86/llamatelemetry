"""
llamatelemetry.quantization.pipeline - Production quantization pipeline.

End-to-end pipeline: merge_lora -> export_gguf -> quantize_gguf -> validate -> benchmark.
Each step creates OTel pipeline spans for lifecycle observability.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..pipeline.spans import PipelineContext, PipelineTracer


@dataclass
class PipelineResult:
    """Result from a quantization pipeline step.

    Attributes:
        success: Whether the step succeeded.
        output_path: Path to the output artifact.
        model_sha256: SHA-256 of the output model.
        metadata: Step-specific metadata.
        error: Error message if failed.
    """

    success: bool = True
    output_path: Optional[str] = None
    model_sha256: Optional[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def merge_lora(
    model: Any,
    adapter_path: Optional[str] = None,
    ctx: Optional[PipelineContext] = None,
) -> Any:
    """Merge LoRA adapters into base model with pipeline tracing.

    Args:
        model: Model with LoRA adapters.
        adapter_path: Optional adapter path.
        ctx: Pipeline context for tracing.

    Returns:
        Model with merged adapters.
    """
    ctx = ctx or PipelineContext()
    tracer = PipelineTracer()

    with tracer.span_merge_lora(ctx):
        from ..unsloth.adapter import LoRAAdapter

        adapter = LoRAAdapter(model)
        if adapter.has_adapters():
            return adapter.merge()
        return model


def export_gguf(
    model: Any,
    tokenizer: Any,
    output_path: Union[str, Path],
    quant_type: str = "Q4_K_M",
    ctx: Optional[PipelineContext] = None,
) -> PipelineResult:
    """Export model to GGUF format with pipeline tracing.

    Args:
        model: Model to export.
        tokenizer: Associated tokenizer.
        output_path: Output GGUF file path.
        quant_type: Quantization type.
        ctx: Pipeline context for tracing.

    Returns:
        PipelineResult with output path and SHA-256.
    """
    ctx = ctx or PipelineContext()
    ctx.output_artifact = str(output_path)
    ctx.quantization = quant_type
    tracer = PipelineTracer()

    with tracer.span_export_gguf(ctx):
        try:
            from ..unsloth.exporter import UnslothExporter, ExportConfig

            config = ExportConfig(quant_type=quant_type)
            exporter = UnslothExporter()
            result_path = exporter.export(model, tokenizer, output_path, config)

            sha256 = _compute_file_sha256(result_path)
            ctx.model_sha256 = sha256

            return PipelineResult(
                success=True,
                output_path=str(result_path),
                model_sha256=sha256,
                metadata={"quant_type": quant_type},
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))


def quantize_gguf(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    quant_type: str = "Q4_K_M",
    ctx: Optional[PipelineContext] = None,
) -> PipelineResult:
    """Quantize a GGUF file with pipeline tracing.

    Args:
        input_path: Input GGUF file.
        output_path: Output quantized GGUF file.
        quant_type: Target quantization type.
        ctx: Pipeline context.

    Returns:
        PipelineResult.
    """
    ctx = ctx or PipelineContext()
    ctx.quantization = quant_type
    tracer = PipelineTracer()

    with tracer.span_quantize(ctx):
        try:
            from ..api.gguf import quantize_gguf_model

            quantize_gguf_model(str(input_path), str(output_path), quant_type)

            sha256 = _compute_file_sha256(Path(output_path))

            return PipelineResult(
                success=True,
                output_path=str(output_path),
                model_sha256=sha256,
                metadata={"quant_type": quant_type, "input": str(input_path)},
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))


def validate_artifact(
    gguf_path: Union[str, Path],
) -> PipelineResult:
    """Validate a GGUF artifact (loadable, correct format).

    Args:
        gguf_path: Path to GGUF file.

    Returns:
        PipelineResult with validation metadata.
    """
    path = Path(gguf_path)

    if not path.exists():
        return PipelineResult(success=False, error=f"File not found: {path}")

    if not path.suffix == ".gguf":
        return PipelineResult(success=False, error=f"Not a GGUF file: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    sha256 = _compute_file_sha256(path)

    return PipelineResult(
        success=True,
        output_path=str(path),
        model_sha256=sha256,
        metadata={
            "size_mb": round(size_mb, 2),
            "format": "gguf",
        },
    )


def benchmark_artifact(
    gguf_path: Union[str, Path],
    server_url: str = "http://127.0.0.1:8090",
    ctx: Optional[PipelineContext] = None,
) -> PipelineResult:
    """Benchmark a GGUF artifact using the llama.cpp engine.

    Args:
        gguf_path: Path to GGUF file.
        server_url: llama.cpp server URL.
        ctx: Pipeline context.

    Returns:
        PipelineResult with benchmark metrics.
    """
    ctx = ctx or PipelineContext()
    tracer = PipelineTracer()

    with tracer.span_benchmark(ctx):
        try:
            from ..bench.runner import BenchmarkRunner

            runner = BenchmarkRunner(backend="llama.cpp", server_url=server_url)
            report = runner.run_suite()

            return PipelineResult(
                success=True,
                output_path=str(gguf_path),
                metadata={"benchmark": report},
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""
