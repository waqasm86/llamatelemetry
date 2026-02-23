"""
llamatelemetry.inference_engine - High-level inference engine with complete observability

Integrates:
  - Native llama.cpp inference (GPU-only)
  - Native NCCL distributed support (multi-GPU)
  - Complete OpenTelemetry gen_ai.* semantic conventions
  - GPU monitoring and metrics
  - Kaggle dual T4 optimization
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import logging

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai

from .llama_cpp_native import LlamaModel, LlamaContext, InferenceLoop
from .otel_gen_ai import GenAITracer, InferenceContext, GPUMonitor
from .nccl_native import NCCLCommunicator
from .kaggle_integration import ModelDownloader, KaggleGPUConfig, KaggleEnvironment

logger = logging.getLogger(__name__)


@dataclass
class GenerateResponse:
    """Response from text generation."""
    text: str
    tokens: List[int]
    token_count: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    finish_reason: str
    input_tokens: int
    output_tokens: int


class InferenceEngine:
    """
    Complete GPU-only inference engine with observability.

    Features:
      - Native llama.cpp inference (no HTTP)
      - Full OpenTelemetry tracing (gen_ai.* attributes)
      - Dual T4 GPU optimization
      - Automatic model downloading from HuggingFace
      - Distributed multi-GPU support via NCCL
    """

    def __init__(
        self,
        model_path: str,
        service_name: str = "llamatelemetry-inference",
        otlp_endpoint: Optional[str] = None,
        n_gpu_layers: int = 40,
        multi_gpu: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to GGUF model
            service_name: Service name for tracing
            otlp_endpoint: OTLP HTTP endpoint
            n_gpu_layers: Number of layers to offload to GPU
            multi_gpu: Enable multi-GPU (dual T4)
            verbose: Enable debug logging
        """
        self.model_path = model_path
        self.service_name = service_name
        self.verbose = verbose

        logger.info(f"Initializing InferenceEngine")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Service: {service_name}")
        logger.info(f"  Multi-GPU: {multi_gpu}")

        # Setup OpenTelemetry
        self._setup_telemetry(otlp_endpoint)

        # Setup GPU monitoring
        self.gpu_monitor = GPUMonitor(meter=self.meter_provider.get_meter(__name__))

        # Get GPU config
        if KaggleEnvironment.is_kaggle():
            self.gpu_config = KaggleGPUConfig()
        else:
            self.gpu_config = KaggleGPUConfig()

        # Load model
        self.model = LlamaModel(
            model_path,
            n_gpu_layers=n_gpu_layers,
            split_mode="layer" if multi_gpu else "none",
            main_gpu=0,
        )

        # Create context
        self.context = LlamaContext(self.model)

        # Create inference loop
        self.inference_loop = InferenceLoop(
            self.model,
            self.context,
            verbose=verbose,
        )

        # Setup multi-GPU if needed
        self.multi_gpu = multi_gpu
        self.nccl_comm = None
        if multi_gpu and self.gpu_config.is_dual_gpu():
            try:
                self.nccl_comm = NCCLCommunicator(
                    nranks=2,
                    rank=0,
                    device=0,
                )
                logger.info("Multi-GPU (NCCL) initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NCCL: {e}")
                self.multi_gpu = False

        logger.info("InferenceEngine initialized successfully")

    def _setup_telemetry(self, otlp_endpoint: Optional[str] = None) -> None:
        """Setup OpenTelemetry with OTLP HTTP exporter."""
        logger.info("Setting up OpenTelemetry...")

        # Create TracerProvider
        self.tracer_provider = TracerProvider()

        # OTLP exporter
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                timeout=10,
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            logger.info(f"OTLP exporter configured: {otlp_endpoint}")
        else:
            logger.info("OTLP exporter not configured (dev mode)")

        # Create MeterProvider
        self.meter_provider = MeterProvider()

        # Get tracer and create GenAITracer
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        self.gen_ai_tracer = GenAITracer(
            tracer=tracer,
            meter=meter,
            provider_name="llamatelemetry",
        )

        logger.info("OpenTelemetry initialized")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> GenerateResponse:
        """
        Generate text with full observability.

        All generation happens on GPU(s). CPU is only used for tokenization
        (which is fast and minimal).

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-P (nucleus) threshold
            top_k: Top-K limit
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop_sequences: Sequences that stop generation
            seed: Random seed for reproducibility
            conversation_id: Session/conversation ID

        Returns:
            GenerateResponse with text and metrics
        """
        # Create inference context with all gen_ai attributes
        with InferenceContext(
            self.gen_ai_tracer,
            model_name=self.model.metadata.get('ftype', 'unknown'),
            operation="chat",
        ) as ctx:
            # Set request parameters
            ctx.set_request_parameters(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=seed,
                conversation_id=conversation_id,
            )

            # GPU-only inference
            start_time = time.perf_counter()

            response = self.inference_loop.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                seed=seed,
            )

            total_ms = (time.perf_counter() - start_time) * 1000

            # Record response metrics in tracing
            ctx.set_response(
                input_tokens=len(self.inference_loop.tokenizer.encode(prompt)),
                output_tokens=response.token_count,
                ttft_ms=response.ttft_ms,
                tpot_ms=response.tpot_ms,
                finish_reason=response.finish_reason,
            )

            # Record GPU metrics
            self.gpu_monitor.record_metrics_to_otel()

            # Log if verbose
            if self.verbose:
                logger.info(f"Generation complete:")
                logger.info(f"  Tokens: {response.token_count}")
                logger.info(f"  TTFT: {response.ttft_ms:.1f}ms")
                logger.info(f"  TPOT: {response.tpot_ms:.2f}ms")
                logger.info(f"  Total: {total_ms:.1f}ms")

            return GenerateResponse(
                text=response.text,
                tokens=response.tokens,
                token_count=response.token_count,
                ttft_ms=response.ttft_ms,
                tpot_ms=response.tpot_ms,
                total_ms=total_ms,
                finish_reason=response.finish_reason,
                input_tokens=len(self.inference_loop.tokenizer.encode(prompt)),
                output_tokens=response.token_count,
            )

    def shutdown(self) -> None:
        """Cleanup resources."""
        logger.info("Shutting down InferenceEngine...")

        # Force flush spans
        self.tracer_provider.force_flush()

        # Destroy NCCL communicator if present
        if self.nccl_comm:
            self.nccl_comm.destroy()

        # Free context and model
        self.context.free()
        self.model.free()

        logger.info("Shutdown complete")

    def __repr__(self) -> str:
        return (
            f"InferenceEngine("
            f"model={self.model.metadata.get('ftype')}, "
            f"service={self.service_name})"
        )

    def __del__(self) -> None:
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except Exception:
            pass


# Convenience factory function
def create_engine(
    model_path: str,
    service_name: str = "llamatelemetry-inference",
    otlp_endpoint: Optional[str] = None,
    n_gpu_layers: int = 40,
    multi_gpu: bool = False,
) -> InferenceEngine:
    """
    Create inference engine.

    Args:
        model_path: Path to GGUF model
        service_name: Service name
        otlp_endpoint: OTLP HTTP endpoint
        n_gpu_layers: GPU layers
        multi_gpu: Enable dual T4

    Returns:
        InferenceEngine instance
    """
    return InferenceEngine(
        model_path=model_path,
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        n_gpu_layers=n_gpu_layers,
        multi_gpu=multi_gpu,
    )
