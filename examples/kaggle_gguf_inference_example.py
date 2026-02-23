"""
llamatelemetry v2.0.0 - Kaggle GGUF Inference with Complete Observability

Example: Complete workflow for GGUF LLM inference on dual T4 GPUs
with full OpenTelemetry gen_ai.* semantic conventions.

This example demonstrates:
  1. Model downloading from HuggingFace
  2. GPU-only inference (no CPU)
  3. Automatic observability with gen_ai.* attributes
  4. Multi-GPU (dual T4) support
  5. Performance metrics (TTFT, TPOT)
"""

import logging
from pathlib import Path

# llamatelemetry imports
import llamatelemetry
from llamatelemetry import InferenceEngine, GenerateResponse
from llamatelemetry.kaggle_integration import (
    ModelDownloader,
    KaggleGPUConfig,
    KaggleEnvironment,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def example_1_download_and_infer():
    """
    Example 1: Download model from HuggingFace and run inference.

    Steps:
      1. Setup Kaggle environment
      2. Download GGUF model from HuggingFace
      3. Initialize inference engine with full observability
      4. Generate text with automatic tracing
    """
    logger.info("=" * 80)
    logger.info("Example 1: Download and Infer")
    logger.info("=" * 80)

    # Step 1: Setup Kaggle environment
    logger.info("\n[Step 1] Setting up Kaggle environment...")
    kaggle_env = KaggleEnvironment()
    config = kaggle_env.setup_llamatelemetry(
        service_name="kaggle-llm-inference",
        otlp_endpoint="https://otlp.example.com/v1/traces",  # Configure your backend
    )
    logger.info(f"Configuration: {config}")

    # Step 2: Download model from HuggingFace
    logger.info("\n[Step 2] Downloading GGUF model from HuggingFace...")
    downloader = ModelDownloader(cache_dir=kaggle_env.model_cache_dir)

    # Download Llama 2 13B Q4_K_M quantized version
    model_path = downloader.get_model_by_shortname(
        shortname="llama-2-13b",
        quantization="Q4_K_M",
    )
    logger.info(f"Model downloaded to: {model_path}")

    # Verify model integrity
    if not downloader.verify_model(model_path):
        logger.error("Model verification failed!")
        return

    # Step 3: Initialize inference engine
    logger.info("\n[Step 3] Initializing inference engine...")
    engine = InferenceEngine(
        model_path=str(model_path),
        service_name="kaggle-llm-inference",
        otlp_endpoint=config['otlp_endpoint'],
        n_gpu_layers=40,  # Offload all layers to GPU for 13B
        multi_gpu=True,   # Enable dual T4 support
        verbose=True,
    )

    logger.info(f"Engine: {engine}")
    logger.info(f"GPU Config: {engine.gpu_config}")

    # Step 4: Generate with automatic observability
    logger.info("\n[Step 4] Generating text with full observability...")

    prompts = [
        "What is GGUF quantization and why is it useful?",
        "Explain the benefits of tensor parallelism in LLMs.",
        "How does OpenTelemetry improve LLM observability?",
    ]

    for prompt in prompts:
        logger.info(f"\nPrompt: {prompt}")

        response = engine.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            conversation_id="kaggle-session-123",
        )

        logger.info(f"Response: {response.text}")
        logger.info(f"Metrics:")
        logger.info(f"  - Input tokens: {response.input_tokens}")
        logger.info(f"  - Output tokens: {response.output_tokens}")
        logger.info(f"  - TTFT: {response.ttft_ms:.1f}ms")
        logger.info(f"  - TPOT: {response.tpot_ms:.2f}ms")
        logger.info(f"  - Total: {response.total_ms:.1f}ms")
        logger.info(f"  - Finish reason: {response.finish_reason}")

    # Cleanup
    logger.info("\n[Cleanup] Shutting down...")
    engine.shutdown()
    logger.info("Done!")


def example_2_multi_gpu_inference():
    """
    Example 2: Multi-GPU (dual T4) inference with NCCL.

    Demonstrates:
      - Dual GPU detection
      - Layer splitting across GPUs
      - NCCL communicator initialization
      - Distributed inference
    """
    logger.info("=" * 80)
    logger.info("Example 2: Multi-GPU Inference")
    logger.info("=" * 80)

    # Detect dual T4 configuration
    logger.info("\n[Step 1] Detecting Kaggle GPU configuration...")
    gpu_config = KaggleGPUConfig()

    if not gpu_config.is_dual_gpu():
        logger.warning("Dual GPU not available. Skipping multi-GPU example.")
        return

    logger.info(f"GPU Config: {gpu_config}")
    logger.info(f"Device count: {gpu_config.device_count}")
    logger.info(f"Total VRAM: {gpu_config.total_vram_gb}GB")

    # Get model-specific configuration
    logger.info("\n[Step 2] Getting optimal model configuration...")
    model_config = gpu_config.get_model_config("llama-2-13b")
    logger.info(f"Model config: {model_config}")

    # Initialize engine with multi-GPU
    logger.info("\n[Step 3] Initializing multi-GPU engine...")
    downloader = ModelDownloader()
    model_path = downloader.get_model_by_shortname(
        shortname="llama-2-13b",
        quantization="Q4_K_M",
    )

    engine = InferenceEngine(
        model_path=str(model_path),
        service_name="kaggle-multi-gpu",
        n_gpu_layers=model_config['n_gpu_layers'],
        multi_gpu=True,
        verbose=True,
    )

    logger.info(f"Multi-GPU engine initialized")
    logger.info(f"NCCL communicator: {engine.nccl_comm}")

    # Generate with multi-GPU
    logger.info("\n[Step 4] Generating with multi-GPU support...")
    response = engine.generate(
        prompt="Explain distributed inference with tensor parallelism.",
        max_tokens=256,
    )

    logger.info(f"Generated: {response.text[:100]}...")
    logger.info(f"Metrics: TTFT={response.ttft_ms:.1f}ms, TPOT={response.tpot_ms:.2f}ms")

    # Cleanup
    engine.shutdown()


def example_3_custom_observability():
    """
    Example 3: Custom observability setup with gen_ai.* attributes.

    Shows:
      - Manual span creation
      - Setting gen_ai.* attributes
      - Custom metric recording
      - Batch inference tracking
    """
    logger.info("=" * 80)
    logger.info("Example 3: Custom Observability")
    logger.info("=" * 80)

    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai

    # Initialize engine (without separate model download for brevity)
    logger.info("\n[Step 1] Initializing engine...")
    downloader = ModelDownloader()
    model_path = downloader.get_model_by_shortname(
        shortname="llama-2-7b",  # Smaller model for demo
        quantization="Q4_K_M",
    )

    engine = InferenceEngine(
        model_path=str(model_path),
        service_name="kaggle-custom-otel",
    )

    # Create custom span context
    logger.info("\n[Step 2] Creating custom observability context...")

    with engine.gen_ai_tracer.trace_inference(
        model_name="llama-2-7b-Q4_K_M.gguf",
        operation="chat",
        conversation_id="custom-session-456",
    ) as span:
        # Manually set attributes
        span.set_attribute(gen_ai.GEN_AI_REQUEST_TEMPERATURE, 0.8)
        span.set_attribute(gen_ai.GEN_AI_REQUEST_TOP_P, 0.95)
        span.set_attribute(gen_ai.GEN_AI_REQUEST_MAX_TOKENS, 512)

        # Add custom events
        span.add_event("prefill_started")

        logger.info("Generating text...")
        response = engine.generate(
            prompt="Tell me about GGUF quantization.",
            max_tokens=512,
            temperature=0.8,
            top_p=0.95,
        )

        span.add_event("prefill_completed", {"ttft_ms": response.ttft_ms})
        span.add_event("decode_completed", {"tokens": response.token_count})

        # Record custom metrics
        engine.gen_ai_tracer.record_token_usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model_name="llama-2-7b-Q4_K_M.gguf",
            attributes={"batch_id": "batch_001"},
        )

    logger.info(f"Generated: {response.text[:100]}...")

    # Cleanup
    engine.shutdown()


def example_4_batch_inference():
    """
    Example 4: Batch inference with conversation tracking.

    Demonstrates:
      - Session/conversation management
      - Multi-turn conversation
      - Conversation-level metrics
    """
    logger.info("=" * 80)
    logger.info("Example 4: Batch Inference with Conversations")
    logger.info("=" * 80)

    # Setup
    logger.info("\n[Step 1] Setting up batch inference...")
    downloader = ModelDownloader()
    model_path = downloader.get_model_by_shortname(
        shortname="llama-2-7b",
        quantization="Q4_K_M",
    )

    engine = InferenceEngine(
        model_path=str(model_path),
        service_name="kaggle-batch",
    )

    # Batch conversation
    conversation_id = "batch-conv-789"
    prompts = [
        "What is CUDA?",
        "Explain GPU memory hierarchy.",
        "How does tensor parallelism work?",
    ]

    logger.info(f"\n[Step 2] Running batch inference for conversation: {conversation_id}")

    for i, prompt in enumerate(prompts):
        logger.info(f"\n[Batch {i+1}/{len(prompts)}] Generating...")

        response = engine.generate(
            prompt=prompt,
            max_tokens=256,
            conversation_id=conversation_id,
        )

        logger.info(f"Q: {prompt}")
        logger.info(f"A: {response.text[:100]}...")
        logger.info(f"Metrics: {response.token_count} tokens, TTFT={response.ttft_ms:.1f}ms")

    # Cleanup
    engine.shutdown()
    logger.info("\nBatch inference complete!")


if __name__ == "__main__":
    """
    Run examples.

    Examples:
      python kaggle_gguf_inference_example.py 1  # Download and infer
      python kaggle_gguf_inference_example.py 2  # Multi-GPU
      python kaggle_gguf_inference_example.py 3  # Custom observability
      python kaggle_gguf_inference_example.py 4  # Batch inference
    """
    import sys

    example_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    try:
        if example_num == 1:
            example_1_download_and_infer()
        elif example_num == 2:
            example_2_multi_gpu_inference()
        elif example_num == 3:
            example_3_custom_observability()
        elif example_num == 4:
            example_4_batch_inference()
        else:
            logger.error(f"Unknown example: {example_num}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)
