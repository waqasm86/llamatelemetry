# Kaggle Dual T4 GPU Guide (v2.0.0)

**Complete guide for running LlamaTelemetry v2.0.0 on Kaggle Dual Nvidia T4 GPUs (SM 7.5) with CUDA 12.5.**

This guide covers:
- Environment setup and GPU detection
- Installation and CUDA binary loading
- Creating and running inference engines
- Dual GPU coordination with NCCL
- OpenTelemetry observability
- Performance optimization
- Production deployment

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dual GPU Inference](#dual-gpu-inference)
- [OpenTelemetry Integration](#opentelemetry-integration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Verify Kaggle Environment

```python
import subprocess
import sys

# Check Python version
print(f"Python: {sys.version}")

# Check CUDA availability
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout)

# Expected output: Tesla T4 x2
```

### Enable GPU on Kaggle

1. Open notebook settings (top-right corner)
2. Select **Accelerator**: `GPU: Tesla T4 x2` (DUAL GPU required)
3. Enable **Internet** (for model downloads)
4. Save settings and restart

### Verify CUDA Setup

```python
import torch

# Check PyTorch CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
print(f"GPU 1: {torch.cuda.get_device_name(1)}")

# Check CUDA version
print(f"CUDA version: {torch.version.cuda}")
```

**Expected output:**
```
CUDA available: True
CUDA devices: 2
GPU 0: Tesla T4
GPU 1: Tesla T4
CUDA version: 12.5
```

---

## Installation

### Step 1: Install from GitHub

```python
# In Kaggle notebook cell:
!pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v2.0.0[gpu,kaggle]
```

### Step 2: Import and Verify

```python
from llamatelemetry import create_engine
from llamatelemetry.kaggle_integration import get_dual_gpu_config

# Verify installation
print("✅ LlamaTelemetry v2.0.0 installed")

# Verify CUDA binary loading
engine = create_engine(
    model="mistral-7b-instruct-v0.2",
    service_name="kaggle-test"
)
print("✅ CUDA binary loaded successfully")
```

### Step 3: Verify GPU Availability

```python
# Get dual GPU configuration
gpu_config = get_dual_gpu_config()
print(f"GPU 0: {gpu_config.gpu_0_name}")
print(f"GPU 1: {gpu_config.gpu_1_name}")
print(f"Total VRAM: {gpu_config.total_vram_gb} GB")
```

---

## Quick Start

### Basic Inference (5 minutes)

```python
from llamatelemetry import create_engine

# Create inference engine
engine = create_engine(
    model="mistral-7b-instruct-v0.2",
    service_name="my-kaggle-app",
    n_gpu_layers=30  # Offload to both T4 GPUs
)

# Generate response
response = engine.generate(
    prompt="What is machine learning?",
    max_tokens=100
)

# Print output
print("Generated text:")
print(response.text)
print(f"\nPerformance:")
print(f"  TTFT: {response.ttft_ms:.1f}ms (Time-to-First-Token)")
print(f"  TPOT: {response.tpot_ms:.3f}ms (Time-Per-Output-Token)")
print(f"  Throughput: {response.throughput:.0f} tokens/sec")
```

**Expected performance (Dual T4):**
```
TTFT: 3-5 ms
TPOT: 0.5-1 ms
Throughput: 1000-2000 tokens/sec
```

---

## Dual GPU Inference

### Auto-Configuration

```python
from llamatelemetry import create_engine
from llamatelemetry.kaggle_integration import get_dual_gpu_config

# Auto-detect dual T4 configuration
gpu_config = get_dual_gpu_config()

# Create engine with dual GPU setup
engine = create_engine(
    model="mistral-7b-instruct-v0.2",
    service_name="dual-gpu-inference",
    gpu_config=gpu_config,
    n_gpu_layers=30  # Split model across both GPUs
)

# Inference automatically uses both GPUs
response = engine.generate("Hello world", max_tokens=50)
```

### Manual GPU Configuration

```python
from llamatelemetry import create_engine

# Fine-tune GPU allocation
engine = create_engine(
    model="mistral-7b-instruct-v0.2",
    service_name="custom-config",
    n_gpu_layers=30,     # Layers to offload to GPU
    n_batch=128,         # Batch size
    n_threads=4,         # CPU threads
    context_size=4096,   # Context length
)

# Check GPU memory usage
response = engine.generate("Test prompt", max_tokens=10)
print(f"GPU Memory Used: {response.gpu_memory_used_mb} MB")
```

### Layer Splitting for Large Models

```python
from llamatelemetry import create_engine

# For 13B+ models, split across dual T4s
engine = create_engine(
    model="llama-2-13b-chat",
    service_name="large-model",
    n_gpu_layers=40,  # More layers for dual GPU
    n_gpu_split=0.5,  # Even split between GPUs
)

response = engine.generate(
    prompt="Explain quantum computing",
    max_tokens=256
)
```

---

## OpenTelemetry Integration

### Basic Setup

```python
from llamatelemetry import create_engine
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Create OTLP exporter
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317")

# Create engine with tracing
engine = create_engine(
    model="mistral-7b",
    service_name="kaggle-inference",
    otlp_exporter=otlp_exporter
)

# Traces are automatically exported
response = engine.generate("Test prompt")
```

### Local Jaeger Tracing (For Testing)

```python
# Start Jaeger collector in Kaggle
import subprocess

# This requires running Jaeger in background
# For quick testing, use simple HTTP export:
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
processor = BatchSpanProcessor(exporter)

# Create engine with custom processor
engine = create_engine(model="mistral-7b")
engine.tracer.add_span_processor(processor)
```

### Observability Metrics

LlamaTelemetry v2.0.0 automatically tracks:

**45 GenAI Semantic Attributes:**
- `gen_ai.operation.name` (e.g., "inference")
- `gen_ai.request.model` (e.g., "mistral-7b")
- `gen_ai.request.max_tokens`
- `gen_ai.response.finish_reason`
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.usage.total_tokens`
- ... and 38 more attributes

**5 Histogram Metrics:**
- `gen_ai.client.token.count` (tokens per request)
- `gen_ai.client.operation.duration` (latency)
- `gen_ai.server.request.duration` (server latency)
- `gen_ai.server.operation.duration` (operation time)
- GPU memory usage (custom)

---

## Performance Tuning

### Batch Processing

```python
from llamatelemetry import create_engine

engine = create_engine(
    model="mistral-7b",
    service_name="batch-inference",
    n_batch=256  # Larger batch size for throughput
)

prompts = [
    "What is AI?",
    "Explain machine learning",
    "Define deep learning",
]

for prompt in prompts:
    response = engine.generate(prompt, max_tokens=100)
    print(f"TTFT: {response.ttft_ms}ms | TPOT: {response.tpot_ms}ms")
```

### Context Window Optimization

```python
# For short contexts (faster)
engine_short = create_engine(
    model="mistral-7b",
    context_size=1024,  # Smaller = faster
    n_gpu_layers=30
)

# For long contexts (more memory)
engine_long = create_engine(
    model="mistral-7b",
    context_size=8192,  # Larger = slower but more capacity
    n_gpu_layers=20    # Fewer layers to fit in VRAM
)
```

### Memory-Optimized Models

```python
# For Q4 quantization (4-bit)
engine_q4 = create_engine(
    model="mistral-7b-q4",  # Q4 variant
    n_gpu_layers=30,
    context_size=4096
)

# For Q5 quantization (5-bit, larger)
engine_q5 = create_engine(
    model="mistral-7b-q5",  # Q5 variant
    n_gpu_layers=25,
    context_size=4096
)

# For F16 (full precision, very large)
engine_f16 = create_engine(
    model="mistral-7b-f16",  # F16 variant
    n_gpu_layers=15,
    context_size=2048
)
```

---

## Model Selection

### Recommended Models for Kaggle Dual T4

| Model | Size | Q4 Size | Recommended |
|-------|------|---------|-------------|
| **Mistral 7B** | 7B | 3.5 GB | ✅ Best |
| **Llama 2 7B** | 7B | 3.5 GB | ✅ Best |
| **Zephyr 7B** | 7B | 3.5 GB | ✅ Best |
| **Qwen 7B** | 7B | 3.5 GB | ✅ Best |
| **Llama 2 13B** | 13B | 7 GB | ⚠️ Works (slower) |
| **Mistral 8x7B** | 56B | 28 GB | ❌ Too large |

### Loading Custom Models

```python
from llamatelemetry import create_engine

# From HuggingFace Hub
engine = create_engine(
    model="meta-llama/Llama-2-7b-chat",
    service_name="hf-model",
    hf_token="your-token"  # Optional
)

# From local GGUF file
engine = create_engine(
    model="/kaggle/input/models/custom-7b-q4.gguf",
    service_name="custom-model"
)
```

---

## Troubleshooting

### GPU Not Detected

```python
# Check GPU settings
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# If 0 devices:
# 1. Go to notebook settings
# 2. Change Accelerator to "GPU: Tesla T4 x2"
# 3. Restart notebook
# 4. Re-run cell
```

### CUDA Binary Load Error

```python
# This error means the .so file wasn't found:
# ImportError: llamatelemetry_cpp.so not found

# Solution:
!pip install --upgrade llamatelemetry[gpu,kaggle]

# Force reinstall:
!pip install --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.0.0[gpu,kaggle]
```

### Out of Memory (OOM)

```python
# Reduce GPU layers
engine = create_engine(
    model="mistral-7b",
    n_gpu_layers=20  # Instead of 30
)

# Or reduce batch size
engine = create_engine(
    model="mistral-7b",
    n_batch=64  # Instead of 128
)

# Or reduce context window
engine = create_engine(
    model="mistral-7b",
    context_size=2048  # Instead of 4096
)
```

### Slow Inference

```python
# Check TTFT/TPOT metrics
response = engine.generate("test", max_tokens=10)
print(f"TTFT: {response.ttft_ms}ms (should be 2-5ms)")
print(f"TPOT: {response.tpot_ms}ms (should be 0.5-1ms)")

# If too slow:
# 1. Increase n_gpu_layers (offload more to GPU)
# 2. Reduce context_size (smaller KV cache)
# 3. Check if NCCL is working (dual GPU coordination)
```

### NCCL Issues

```python
# NCCL errors are non-critical
# Single-GPU mode will still work

from llamatelemetry.nccl_native import get_nccl_status

try:
    status = get_nccl_status()
    print(f"NCCL: {status}")
except Exception as e:
    print(f"NCCL not available (optional): {e}")
```

---

## Advanced: Custom Inference Loop

```python
from llamatelemetry import create_engine

engine = create_engine(
    model="mistral-7b",
    service_name="custom-loop"
)

# Streaming inference (if supported)
prompt = "Write a short story about AI"
tokens_generated = 0

for token in engine.generate_streaming(prompt, max_tokens=100):
    print(token, end="", flush=True)
    tokens_generated += 1

print(f"\n\nGenerated {tokens_generated} tokens")
```

---

## Production Deployment

### Running the Validation Notebook

Before deploying to production, validate the setup:

```bash
# In notebook:
# Run: kaggle-llamatelemetry-v2-0-0-production-validation.ipynb
# Expected: ~17 minutes
# Expected result: ✅ 100% PRODUCTION READY
```

### Exporting Telemetry

```python
# Setup OTLP export to production backend
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="https://your-observability-backend.com:4317"
)

engine = create_engine(
    model="mistral-7b",
    otlp_exporter=exporter
)

# All traces are now exported to your backend
```

---

## Resources

- **Quick Start:** [QUICK_START_KAGGLE_VALIDATION.txt](../QUICK_START_KAGGLE_VALIDATION.txt)
- **Validation Notebook:** `kaggle-llamatelemetry-v2-0-0-production-validation.ipynb`
- **Deployment Guide:** [PRODUCTION_DEPLOYMENT_GUIDE_V2_0_0.md](../PRODUCTION_DEPLOYMENT_GUIDE_V2_0_0.md)
- **API Reference:** [../README.md](../README.md)

---

**Last Updated:** Feb 24, 2026
**Version:** 2.0.0
**Target:** Kaggle Dual Nvidia T4 GPUs (SM 7.5, CUDA 12.5)
