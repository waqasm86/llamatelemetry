# Integration Guide (v1.2.0)

This guide explains how to integrate llamatelemetry into your LLM inference workflow using the
v1.2.0 `gen_ai.*` semantic conventions, `InferenceEngine` protocol, and GenAI metrics.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Initialization](#initialization)
- [InferenceEngine Protocol](#inferenceengine-protocol)
- [llama.cpp Integration](#llamacpp-integration)
- [Transformers Integration](#transformers-integration)
- [BenchmarkRunner](#benchmarkrunner)
- [PipelineTracer](#pipelinetracer)
- [GenAI Semantic Conventions](#genai-semantic-conventions)
- [GenAI Metrics](#genai-metrics)
- [Span Redaction](#span-redaction)
- [Multi-GPU Integration](#multi-gpu-integration)
- [Graphistry Visualization](#graphistry-visualization)
- [Full Example](#full-example)

---

## Overview

llamatelemetry v1.2.0 provides CUDA-first LLM observability built on OpenTelemetry:

```
Your Application
      │
      ▼
llamatelemetry.init()          ← OTLP tracer + metrics + GPU enricher
      │
      ├─ llama.LlamaCppClient  ← llama.cpp server instrumentation
      ├─ transformers           ← HuggingFace Transformers instrumentation
      └─ bench.BenchmarkRunner ← Latency phase capture (TTFT, TPOT)
            │
            ▼
      OTLP Collector → Grafana / Graphistry / Jaeger
```

All spans use `gen_ai.*` attributes per OpenTelemetry GenAI semantic conventions.

---

## Installation

```bash
pip install llamatelemetry==1.2.0
```

For GPU features, CUDA 12.x is required:

```python
import llamatelemetry
llamatelemetry.require_cuda()  # Raises RuntimeError if no CUDA GPU found
```

---

## Initialization

```python
import llamatelemetry

llamatelemetry.init(
    service_name="my-inference-service",
    service_version="1.0.0",
    otlp_endpoint="http://localhost:4317",
    enable_metrics=True,
    gpu_enrichment=True,         # Attach GPU metrics to every span
    enable_trace_graphs=True,    # Build Graphistry pipeline graph (v1.2.0)
)
```

### Configuration Dataclass

```python
from llamatelemetry import LlamaTelemetryConfig

config = LlamaTelemetryConfig(
    service_name="inference-prod",
    otlp_endpoint="http://otel-collector:4317",
    enable_metrics=True,
    enable_trace_graphs=True,
    gpu_enrichment=True,
)
llamatelemetry.init(config=config)
```

---

## InferenceEngine Protocol

v1.2.0 defines the `InferenceEngine` protocol that all backends implement:

```python
from llamatelemetry.inference import InferenceEngine

class MyCustomBackend:
    """Implements InferenceEngine protocol."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        ...

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict:
        ...

    @property
    def model_name(self) -> str:
        ...
```

Use `InferenceEngine` as the type annotation when writing backend-agnostic code:

```python
def run_inference(engine: InferenceEngine, prompt: str) -> str:
    return engine.generate(prompt, max_tokens=512)
```

---

## llama.cpp Integration

### ServerManager + LlamaCppClient

```python
from llamatelemetry import ServerManager, ServerConfig
from llamatelemetry.llama import LlamaCppClient

# Start the server
config = ServerConfig(
    model_path="/path/to/model.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    flash_attn=True,
)
server = ServerManager(config)
server.start()

# Create instrumented client
client = LlamaCppClient(
    base_url=server.url,
    strict_operation_names=True,  # Validate gen_ai.operation.name (v1.2.0)
)

# Chat completion — automatically emits gen_ai.* spans + metrics
response = client.chat(
    messages=[{"role": "user", "content": "What is flash attention?"}],
    max_tokens=512,
)
print(response.choices[0].message.content)

server.stop()
```

### wrap_openai_client()

Drop-in instrumentation for any OpenAI-compatible client:

```python
from llamatelemetry.llama import wrap_openai_client
from openai import OpenAI

raw_client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
client = wrap_openai_client(raw_client, strict_operation_names=True)

# All calls are now traced with gen_ai.* attributes
response = client.chat.completions.create(
    model="gemma-3-4b-Q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Inference Phases

llamatelemetry v1.2.0 captures detailed latency phases on every llama.cpp request:

| Phase | Attribute | Description |
|-------|-----------|-------------|
| Prefill | `gen_ai.prefill_ms` | Prompt processing time (ms) |
| First token | `gen_ai.ttft_ms` / `gen_ai.server.time_to_first_token` | Time to first token |
| Decode | `gen_ai.decode_ms` | Token generation time (ms) |
| Per-token | `gen_ai.tpot_ms` / `gen_ai.server.time_per_output_token` | Time per output token |

---

## Transformers Integration

Instrument HuggingFace Transformers pipelines:

```python
from llamatelemetry.transformers import TransformersInstrumentor

instrumentor = TransformersInstrumentor(
    record_events=True,       # Emit span events for each generation step
    emit_metrics=True,        # Record GenAI histogram metrics (v1.2.0)
    strict_operation_names=True,  # Validate operation names
)
instrumentor.instrument()

# Now all transformers pipeline calls are traced
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2", device=0)
result = pipe("Once upon a time", max_new_tokens=50)
```

### TransformersBackend (lower-level)

```python
from llamatelemetry.transformers import TransformersBackend

backend = TransformersBackend(
    model_name_or_path="gpt2",
    device="cuda:0",
    stop_sequences=[".", "\n"],
    repetition_penalty=1.1,
)
output = backend.generate("Write a poem about CUDA:")
```

---

## BenchmarkRunner

`BenchmarkRunner` runs structured benchmarks and records latency phases as GenAI spans:

```python
from llamatelemetry.bench import BenchmarkRunner, BenchmarkProfile

runner = BenchmarkRunner(
    client=client,
    profile=BenchmarkProfile.STANDARD,   # warmup=10, runs=50
)

results = runner.run(
    model_name="gemma-3-4b-Q4_K_M",
    prompts=[
        "Explain attention mechanisms.",
        "Write a Python function to sort a list.",
        "What is CUDA Unified Memory?",
    ],
)

# Print summary table
print(results.summary())

# Export to DataFrame
df = results.to_dataframe()
```

### BenchmarkProfile Options

| Profile | Warmup | Runs | Description |
|---------|--------|------|-------------|
| `QUICK` | 2 | 10 | Fast smoke test |
| `STANDARD` | 10 | 50 | Default benchmark |
| `THOROUGH` | 20 | 200 | Statistical accuracy |

---

## PipelineTracer

`PipelineTracer` creates hierarchical spans for multi-stage inference pipelines:

```python
from llamatelemetry.pipeline import PipelineTracer

tracer = PipelineTracer(service_name="rag-pipeline")

with tracer.pipeline("rag-query") as pipeline_span:
    with tracer.stage("retrieval") as stage:
        docs = retrieve_documents(query)
        stage.set_attribute("retrieval.doc_count", len(docs))

    with tracer.stage("generation") as stage:
        response = client.chat(messages=[...])
        stage.set_attribute("gen_ai.usage.output_tokens",
                            response.usage.completion_tokens)
```

---

## GenAI Semantic Conventions

v1.2.0 uses `gen_ai.*` attributes exclusively. Legacy `llm.*` attributes are removed.

### Standard Span Attributes

```python
from llamatelemetry.semconv import set_gen_ai_attrs
from llamatelemetry.semconv import normalize_operation

# Build span attributes from inference result
attrs = set_gen_ai_attrs(
    system="llamacpp",
    operation=normalize_operation("chat"),   # → "chat"
    model="gemma-3-4b-Q4_K_M",
    input_tokens=42,
    output_tokens=128,
    server_address="127.0.0.1",
    server_port=8080,
)
span.set_attributes(attrs)
```

### normalize_operation()

Maps raw operation strings to canonical `gen_ai.operation.name` values:

```python
from llamatelemetry.semconv import normalize_operation

normalize_operation("CHAT")         # → "chat"
normalize_operation("completion")   # → "text_completion"
normalize_operation("embed")        # → "embeddings"
normalize_operation("embedding")    # → "embeddings"
normalize_operation("unknown")      # → "text_completion" (default)
```

---

## GenAI Metrics

v1.2.0 records five histogram metrics per request via `otel/gen_ai_metrics.py`:

```python
from llamatelemetry.otel.gen_ai_metrics import GenAIMetrics

metrics = GenAIMetrics(meter=meter)

# Record after each request:
metrics.record_token_usage(input_tokens=42, output_tokens=128,
                            gen_ai_system="llamacpp", operation="chat",
                            model="gemma-3-4b-Q4_K_M")
metrics.record_operation_duration(duration_s=1.23, ...)
metrics.record_ttft(ttft_s=0.18, ...)
metrics.record_tpot(tpot_s=0.032, ...)
```

When `llamatelemetry.init(enable_metrics=True)` is set, all of these are recorded automatically.

---

## Span Redaction

Redact sensitive prompt/completion content from spans:

```python
from llamatelemetry.otel import RedactionSpanProcessor

processor = RedactionSpanProcessor(
    redact_prompts=True,       # Redact gen_ai.prompt values
    redact_keys=[              # Additional keys to redact
        "gen_ai.request.messages",
        "gen_ai.response.text",
    ],
)

llamatelemetry.init(
    ...,
    span_processors=[processor],
)
```

---

## Multi-GPU Integration

### Tensor Split (llama.cpp)

```python
config = ServerConfig(
    model_path="/path/to/model.gguf",
    tensor_split=[0.5, 0.5],    # 50% on GPU 0, 50% on GPU 1
    n_gpu_layers=-1,
)
```

### NCCL Distributed

```python
from llamatelemetry.nccl import NCCLGroup

group = NCCLGroup(world_size=2, backend="nccl")
group.init_process_group()
```

### Distributed Topology

```python
from llamatelemetry.distributed import TopologyMapper

mapper = TopologyMapper()
topology = mapper.detect()
print(topology)  # Prints NVLink / PCIe interconnect map
```

---

## Graphistry Visualization

Visualize the inference pipeline graph on Graphistry:

```python
from llamatelemetry.graphistry import GraphistryViz

viz = GraphistryViz(auto_register=True)
viz.plot_pipeline_graph(
    model_name="gemma-3-4b-Q4_K_M",
    backend="llama.cpp",
    gpu_names=["T4", "T4"],
    tensor_split=[0.5, 0.5],
    server_url="http://127.0.0.1:8080",
)
```

---

## Full Example

End-to-end llamatelemetry v1.2.0 integration:

```python
import llamatelemetry
from llamatelemetry import ServerManager, ServerConfig
from llamatelemetry.llama import LlamaCppClient
from llamatelemetry.bench import BenchmarkRunner, BenchmarkProfile
from llamatelemetry.otel import RedactionSpanProcessor

# 1. Verify CUDA
llamatelemetry.require_cuda()

# 2. Initialize telemetry
llamatelemetry.init(
    service_name="llm-service",
    otlp_endpoint="http://localhost:4317",
    enable_metrics=True,
    gpu_enrichment=True,
    enable_trace_graphs=True,
    span_processors=[
        RedactionSpanProcessor(redact_prompts=True),
    ],
)

# 3. Start server
config = ServerConfig(
    model_path="/models/gemma-3-4b-Q4_K_M.gguf",
    tensor_split=[0.5, 0.5],
    n_gpu_layers=-1,
    flash_attn=True,
)
server = ServerManager(config)
server.start()

# 4. Create instrumented client
client = LlamaCppClient(
    base_url=server.url,
    strict_operation_names=True,
)

# 5. Run inference
response = client.chat(
    messages=[{"role": "user", "content": "Explain CUDA tensor cores."}],
    max_tokens=512,
)
print(response.choices[0].message.content)

# 6. Benchmark
runner = BenchmarkRunner(client=client, profile=BenchmarkProfile.STANDARD)
results = runner.run(model_name="gemma-3-4b-Q4_K_M",
                     prompts=["What is flash attention?"] * 10)
print(results.summary())

# 7. Cleanup
server.stop()
llamatelemetry.shutdown()
```

---

## See Also

- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) — 5-minute quickstart
- [GOLDEN_PATH.md](GOLDEN_PATH.md) — Recommended end-to-end workflow
- [API_REFERENCE.md](API_REFERENCE.md) — Full API documentation
- [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) — Kaggle-specific setup
- [CONFIGURATION.md](CONFIGURATION.md) — All configuration options
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design
