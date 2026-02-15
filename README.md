# llamatelemetry v1.0.0

**CUDA-first OpenTelemetry Python SDK for LLM inference observability**

llamatelemetry combines:
- **llama.cpp GGUF inference** - High-performance quantized model inference
- **NCCL-aware multi-GPU execution** - Dual T4 tensor parallelism and split-GPU workflows
- **OpenTelemetry traces + metrics** - Production-grade observability with OTLP export
- **GPU analytics and visualization** - RAPIDS cuGraph + Graphistry interactive dashboards

This repository is optimized for **Kaggle dual Tesla T4 notebooks** and small GGUF models (1B-5B parameters, Q4_K_M quantization). It ships lightweight Python code and downloads large CUDA binaries on first import.

## What You Get
- **Decorator-based tracing** (`@trace`, `@workflow`, `@task`, `@tool`) with OpenTelemetry spans
- **LLM request tracing** with prefill/decode span hierarchy and semantic attributes
- **GPU-aware metrics** (latency, tokens/sec, VRAM usage, temperature, power draw)
- **Prompt redaction** for privacy-sensitive deployments
- **Split-GPU workflow** (GPU 0: inference, GPU 1: analytics/visualization)
- **Graph-based trace visualization** with Graphistry interactive dashboards
- **Kaggle auto-configuration** for Grafana Cloud and Graphistry
- **16 comprehensive tutorials** covering foundation to production workflows

## Quick Start

```python
import llamatelemetry

# Initialize the SDK
llamatelemetry.init(
    service_name="my-llm-app",
    otlp_endpoint="https://otlp.example.com/v1/traces",
)

# Trace functions with decorators
@llamatelemetry.trace()
def generate(prompt: str) -> str:
    client = llamatelemetry.llama.LlamaCppClient("http://127.0.0.1:8090")
    resp = client.chat.create(messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message["content"]

result = generate("Hello, world!")

# Clean shutdown
llamatelemetry.shutdown()
```

### Kaggle Quick Start (Dual T4)

```python
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v1.0.0

import llamatelemetry

# Auto-configure from Kaggle secrets
llamatelemetry.kaggle.auto_configure_grafana_cloud()

# Initialize with tracing
llamatelemetry.init(service_name="kaggle-llm")

# Start server with one-liner
llamatelemetry.llama.quick_start(
    model_path="/kaggle/working/models/gemma-3-1b-it-Q4_K_M.gguf",
    preset="kaggle_t4_dual",
)

# Trace inference requests
with llamatelemetry.llama.trace_request(model="gemma-3-1b", request_id="r1") as req:
    client = llamatelemetry.llama.LlamaCppClient("http://127.0.0.1:8090")
    resp = client.chat.create(messages=[{"role": "user", "content": "Hello!"}])
    req.set_completion_tokens(resp.usage.completion_tokens)

llamatelemetry.shutdown()
```

## v1.0.0 API Overview

| Module | Purpose |
|--------|---------|
| `llamatelemetry.init()` / `.shutdown()` | SDK lifecycle |
| `llamatelemetry.trace()` / `.workflow()` / `.task()` / `.tool()` | Decorators |
| `llamatelemetry.span()` / `.session()` | Context managers |
| `llamatelemetry.llama` | LlamaCppClient, ServerManager, trace_request, GGUF |
| `llamatelemetry.gpu` | GPU device listing, snapshots, background sampler |
| `llamatelemetry.nccl` | NCCL collective tracing |
| `llamatelemetry.otel` | Provider, exporters, sampling, redaction |
| `llamatelemetry.semconv` | Attribute key constants |
| `llamatelemetry.artifacts` | Trace graph export |
| `llamatelemetry.kaggle` | Secret loading, Grafana/Graphistry auto-config |

## Documentation Map
Start here:
- `docs/INDEX.md`

Common entry points:
- `docs/INSTALLATION.md`
- `docs/QUICK_START_GUIDE.md`
- `docs/ARCHITECTURE.md`
- `docs/INTEGRATION_GUIDE.md`
- `docs/NOTEBOOKS_GUIDE.md`
- `docs/TROUBLESHOOTING.md`

Notebooks (16 comprehensive tutorials):
- `notebooks/README.md` - Notebooks overview
- Foundation: Notebooks 01-04 (Quick start, server setup, multi-GPU, quantization)
- Integration: Notebooks 05-06 (Unsloth, Graphistry split-GPU)
- Advanced: Notebooks 07-10 (Knowledge graphs, workflows)
- Deep Dive: Notebooks 11-13 (Neural network viz, attention, embeddings)
- **Observability Trilogy**: Notebooks 14-16 (OpenTelemetry, real-time monitoring, production stack) ⭐ **NEW**

## Project Layout (High Level)
- `llamatelemetry/` Python SDK
- `csrc/` CUDA/C++ kernels and bindings
- `docs/` Guides and references
- `notebooks/` Authoritative notebook specs
- `examples/` Minimal runnable examples
- `scripts/` Build and release utilities

## Support and Contribution
- `CONTRIBUTING.md`
- `CHANGELOG.md`
