# Kaggle Guide (v1.2.0)

This guide covers the end-to-end llamatelemetry workflow on Kaggle T4 GPU notebooks using the v1.2.0
`gen_ai.*` semantic conventions.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Quick Start: Auto-Configure](#quick-start-auto-configure)
- [CUDA Binary Bootstrap](#cuda-binary-bootstrap)
- [Server Management](#server-management)
- [Inference with LlamaCppClient](#inference-with-llamacppclient)
- [Benchmarking with BenchmarkRunner](#benchmarking-with-benchmarkrunner)
- [GPU Telemetry](#gpu-telemetry)
- [GenAI Semantic Conventions](#genai-semantic-conventions)
- [GenAI Metrics](#genai-metrics)
- [Dual-GPU Layout](#dual-gpu-layout)
- [Exporting Telemetry](#exporting-telemetry)
- [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Install llamatelemetry

```python
# In a Kaggle notebook cell:
!pip install llamatelemetry==1.2.0
```

### Verify CUDA is Available

```python
import llamatelemetry

# Enforce CUDA requirement (raises RuntimeError if no GPU)
llamatelemetry.require_cuda()

# Or check availability without raising:
available = llamatelemetry.detect_cuda()
print(f"CUDA available: {available}")

# Check GPU compatibility (SM 7.5+ required for T4)
compat = llamatelemetry.check_gpu_compatibility()
print(compat)
```

---

## Quick Start: Auto-Configure

The `kaggle` subpackage provides one-call setup for Kaggle environments:

```python
from llamatelemetry.kaggle import auto_configure

# Detects T4 GPU count, sets up OTLP endpoints, configures secrets
config = auto_configure(
    model_name="gemma-3-4b-Q4_K_M",
    otlp_endpoint="http://localhost:4317",
    gpu_split=[0.5, 0.5],   # Dual T4: 50/50
)
print(config)
```

`auto_configure()` performs:
1. GPU detection via NVML/nvidia-smi
2. CUDA binary download (if not already cached)
3. OTLP exporter initialization
4. ServerConfig creation with tensor split for dual-GPU

---

## CUDA Binary Bootstrap

llamatelemetry ships Python source only. CUDA binaries are downloaded on first use from GitHub
Releases:

```python
from llamatelemetry.artifacts import download_binaries

# Downloads llamatelemetry-v1.2.0-cuda12-kaggle-t4x2.tar.gz
# and unpacks to llamatelemetry/lib/ and llamatelemetry/binaries/
download_binaries(version="1.2.0", target="kaggle-t4x2")
```

The binary package includes:
- `llama-server` — llama.cpp server (b7760, CUDA 12.5)
- `llama-cli` — CLI for quick inference tests
- NCCL libraries for multi-GPU communication

---

## Server Management

Use `ServerManager` to start and manage the llama.cpp backend:

```python
from llamatelemetry import ServerManager, ServerConfig

config = ServerConfig(
    model_path="/kaggle/input/gemma-3-4b/gemma-3-4b-Q4_K_M.gguf",
    host="127.0.0.1",
    port=8080,
    n_gpu_layers=-1,      # All layers on GPU
    tensor_split=[0.5, 0.5],  # Dual T4 (GPU0 + GPU1)
    n_ctx=4096,
    flash_attn=True,      # Enabled on SM 7.5+
)

server = ServerManager(config)
server.start()
print(f"Server running at {server.url}")
```

### Server Health Check

```python
import time

# Wait for server to become ready
for _ in range(30):
    if server.is_healthy():
        break
    time.sleep(1)
else:
    raise RuntimeError("Server did not start in 30s")
```

---

## Inference with LlamaCppClient

```python
from llamatelemetry.llama import LlamaCppClient

client = LlamaCppClient(
    base_url="http://127.0.0.1:8080",
    strict_operation_names=True,   # Validate gen_ai operation names (v1.2.0)
)

response = client.chat(
    messages=[{"role": "user", "content": "Explain CUDA tensor cores in 3 sentences."}],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### OpenAI-Compatible Wrapper

```python
from llamatelemetry.llama import wrap_openai_client
from openai import OpenAI

openai_client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
instrumented = wrap_openai_client(openai_client, strict_operation_names=True)

response = instrumented.chat.completions.create(
    model="gemma-3-4b-Q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Benchmarking with BenchmarkRunner

`BenchmarkRunner` captures inference latency phases aligned with GenAI semconv:

```python
from llamatelemetry.bench import BenchmarkRunner, BenchmarkProfile

runner = BenchmarkRunner(
    client=client,
    profile=BenchmarkProfile.STANDARD,   # 10 warmup + 50 benchmark prompts
)

results = runner.run(
    model_name="gemma-3-4b-Q4_K_M",
    prompts=["Write a haiku about GPUs.", "Explain attention mechanisms."],
)

print(results.summary())
# Outputs: TTFT p50/p95, tokens/sec, prefill_ms, decode_ms per request
```

### Compare Configurations

```python
from llamatelemetry.bench import compare_configs

compare_configs(
    configs=[
        {"tensor_split": [1.0, 0.0], "label": "Single GPU"},
        {"tensor_split": [0.5, 0.5], "label": "Dual GPU split"},
    ],
    model_path="/kaggle/input/gemma-3-4b/gemma-3-4b-Q4_K_M.gguf",
)
```

---

## GPU Telemetry

### GPU Span Enricher

Automatically attaches GPU metrics to every inference span:

```python
import llamatelemetry

llamatelemetry.init(
    service_name="kaggle-inference",
    otlp_endpoint="http://localhost:4317",
    gpu_enrichment=True,   # Enables GPUSpanEnricher
)
```

### Manual GPU Metrics

```python
from llamatelemetry.gpu import GPUMonitor

monitor = GPUMonitor()
snapshot = monitor.snapshot()

for gpu in snapshot.gpus:
    print(f"GPU {gpu.index}: {gpu.utilization_pct}% util, {gpu.memory_used_mb}MB used")
```

---

## GenAI Semantic Conventions

v1.2.0 uses `gen_ai.*` OTel attributes exclusively (legacy `llm.*` attributes removed):

| Attribute | Example Value | Description |
|-----------|--------------|-------------|
| `gen_ai.system` | `"llamacpp"` | Inference backend |
| `gen_ai.operation.name` | `"chat"` | Operation type |
| `gen_ai.request.model` | `"gemma-3-4b-Q4_K_M"` | Model identifier |
| `gen_ai.request.max_tokens` | `256` | Max tokens requested |
| `gen_ai.response.id` | `"cmpl-abc123"` | Response ID |
| `gen_ai.usage.input_tokens` | `42` | Prompt token count |
| `gen_ai.usage.output_tokens` | `128` | Completion token count |
| `server.address` | `"127.0.0.1"` | Server host |
| `server.port` | `8080` | Server port |

### normalize_operation()

```python
from llamatelemetry.semconv import normalize_operation

op = normalize_operation("CHAT")    # → "chat"
op = normalize_operation("embed")   # → "embeddings"
op = normalize_operation("unknown") # → "text_completion"
```

---

## GenAI Metrics

v1.2.0 records five GenAI histogram instruments on every request:

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.client.token.usage` | `{token}` | Token count (input + output) |
| `gen_ai.client.operation.duration` | `s` | Total request duration |
| `gen_ai.server.time_to_first_token` | `s` | TTFT latency |
| `gen_ai.server.time_per_output_token` | `s` | Per-token decode time |
| `gen_ai.server.request.active` | `{request}` | In-flight request count |

These are automatically emitted when `llamatelemetry.init()` is called with an OTLP metrics
endpoint.

---

## Dual-GPU Layout

Recommended Kaggle T4x2 configuration (16 GB × 2):

```
GPU 0 (T4, 16 GB)           GPU 1 (T4, 16 GB)
────────────────────         ────────────────────
llama.cpp server             Graphistry analytics
tensor_split=0.5             cuDF / cuGraph
Prefill + Decode             OTLP Collector
NCCL peer-to-peer            Telemetry storage
```

```python
# Tensor split: 50% on each GPU
config = ServerConfig(
    model_path="...",
    tensor_split=[0.5, 0.5],
    n_gpu_layers=-1,
)
```

---

## Exporting Telemetry

### OTLP (Recommended)

```python
import llamatelemetry

llamatelemetry.init(
    service_name="my-kaggle-notebook",
    otlp_endpoint="http://localhost:4317",   # gRPC
    # otlp_endpoint="http://localhost:4318", # HTTP
    enable_metrics=True,
)
```

### Console Export (Debugging)

```python
llamatelemetry.init(
    service_name="debug",
    exporter="console",
)
```

---

## Troubleshooting

### CUDA Not Available

```
RuntimeError: CUDA is required but not available.
```

Ensure the Kaggle notebook has GPU accelerator enabled:
**Settings → Accelerator → GPU T4 x2**

### PermissionError on Binary Paths

In restricted containers, some PATH entries may be inaccessible. llamatelemetry v1.2.0 handles this
gracefully — `PermissionError` and `OSError` on PATH scanning are silently skipped.

### Server Won't Start

```bash
# Check if llama-server binary exists:
ls -la $(python -c "import llamatelemetry; print(llamatelemetry.__path__[0])")/binaries/

# Run health check:
curl http://127.0.0.1:8080/health
```

### OTLP Connection Refused

Start a local collector in another cell:

```bash
!docker run -d -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector:latest
```

---

## See Also

- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) — Getting started in 5 minutes
- [GOLDEN_PATH.md](GOLDEN_PATH.md) — Recommended end-to-end workflow
- [CONFIGURATION.md](CONFIGURATION.md) — All configuration options
- [API_REFERENCE.md](API_REFERENCE.md) — Full API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and data flow
