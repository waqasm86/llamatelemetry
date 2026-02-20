# llamatelemetry v1.1.0

**CUDA-first OpenTelemetry Python SDK for LLM inference observability.**

GPU-native telemetry for quantized LLM inference pipelines — dual Tesla T4 on Kaggle, llama.cpp GGUF backends, PyTorch/Transformers, OpenTelemetry OTLP export, and full ML pipeline tracing from fine-tune to deploy.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](CHANGELOG.md)

---

## What's in v1.1.0

| Subsystem | Description |
|---|---|
| **GenAI Semantic Conventions** | Full `gen_ai.*` OTel attribute registry, dual-emit with legacy `llm.*` |
| **Unified Inference Contract** | `InferenceEngine` protocol for llama.cpp + Transformers backends |
| **Performance Primitives** | TTFT, TPOT, TPS, prefill TPS, VRAM peak, queue delay |
| **Benchmark Harness** | `BenchmarkRunner`, `BenchmarkReport`, `compare_reports()` with regression detection |
| **Pipeline Observability** | OTel spans for full ML lifecycle: finetune → merge_lora → export_gguf → quantize → benchmark → deploy |
| **GPU Span Enrichment** | `GPUSpanEnricher` before/after snapshots with delta computation |
| **Scheduler + KV Cache** | Request scheduling, paged KV cache allocator, LRU/FIFO/session-pinning eviction |
| **Torch-optional design** | All core modules work without PyTorch; GPU/unsloth/cuda modules raise clear errors when torch is missing |

---

## Quick Start

```python
import llamatelemetry

# Initialise once at startup
llamatelemetry.init(
    service_name="my-llm-app",
    otlp_endpoint="https://otlp.example.com/v1/traces",
)

# Trace any function
@llamatelemetry.trace()
def generate(prompt: str) -> str:
    client = llamatelemetry.llama.LlamaCppClient("http://127.0.0.1:8090")
    resp = client.chat.create(messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message["content"]

result = generate("Hello, world!")
llamatelemetry.shutdown()
```

### Kaggle Dual T4 (recommended)

```python
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v1.1.0

import llamatelemetry
from llamatelemetry.kaggle import auto_configure

# Auto-detects dual T4, sets LD_LIBRARY_PATH, loads Kaggle secrets
cfg = auto_configure()

llamatelemetry.init(
    service_name="kaggle-inference",
    enable_gpu=True,
    enable_nccl=True,
)

# Start llama.cpp server
llamatelemetry.llama.quick_start(
    model_path="/kaggle/working/models/llama-3-8b-Q4_K_M.gguf",
    preset="kaggle_t4_dual",
)
```

### Production inference with full telemetry

```python
from llamatelemetry.inference.api import create_engine
from llamatelemetry.inference.base import InferenceRequest
from llamatelemetry.inference.types import SamplingParams

engine = create_engine(backend="llama.cpp", llama_server_url="http://127.0.0.1:8090")
engine.start()

result = engine.generate(InferenceRequest(
    messages=[{"role": "user", "content": "Explain GGUF quantization."}],
    max_tokens=256,
    sampling=SamplingParams(temperature=0.7, top_p=0.9),
))
print(f"TPS: {result.tps:.1f}  TTFT: {result.ttft_ms:.0f}ms  VRAM: {result.vram_peak_mb:.0f}MB")
engine.shutdown()
```

### GenAI semantic conventions (OTel-standard)

```python
from llamatelemetry.semconv.gen_ai import GenAI
from llamatelemetry.semconv.mapping import set_dual_attrs

with tracer.start_as_current_span("llm.inference") as span:
    set_dual_attrs(span, request, response)  # emits gen_ai.* + legacy llm.*
```

### Pipeline observability (finetune → deploy)

```python
from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext

ctx = PipelineContext(
    base_model="llama-3-8b",
    adapter="my-lora",
    quantization="Q4_K_M",
)
tracer = PipelineTracer()

with tracer.span_merge_lora(ctx):
    merged = merge_lora_adapters(model)

with tracer.span_export_gguf(ctx):
    export_to_gguf(merged, "model-q4.gguf")

with tracer.span_quantize(ctx):
    quantize("model-q4.gguf", quant_type="Q4_K_M")

with tracer.span_benchmark(ctx):
    runner.run()

with tracer.span_deploy(ctx):
    deploy("model-q4.gguf")
```

### Benchmarking + regression detection

```python
from llamatelemetry.bench import BenchmarkRunner, BenchmarkReport, compare_reports

runner = BenchmarkRunner(backend="llama.cpp")
report = runner.run(llama_server_url="http://127.0.0.1:8090")
report.save("baseline.json")

# Later — detect regressions (>10% drop in TPS = regression)
baseline = BenchmarkReport.load("baseline.json")
comparisons = compare_reports(baseline.to_dict(), current.to_dict(), regression_threshold_pct=10.0)
for c in comparisons:
    if c.regression:
        print(f"REGRESSION: {c.test_name} {c.metric} dropped {c.delta_pct:.1f}%")
```

---

## Installation

```bash
pip install llamatelemetry
```

**With optional dependencies:**

```bash
pip install "llamatelemetry[gpu]"          # pynvml GPU metrics
pip install "llamatelemetry[kaggle]"       # HuggingFace Hub download
pip install "llamatelemetry[graphistry]"   # Graphistry trace visualization
pip install "llamatelemetry[all]"          # everything
```

---

## Module Map

| Module | Purpose |
|---|---|
| `llamatelemetry` | `init()`, `shutdown()`, `trace()`, `span()`, `workflow()` |
| `llamatelemetry.inference` | `create_engine()`, `InferenceRequest`, `InferenceResult`, `CudaInferenceConfig` |
| `llamatelemetry.inference.events` | `InferenceEvents`, `EventRecorder` — TTFT/TPOT timestamp lifecycle |
| `llamatelemetry.inference.metrics` | `compute_ttft()`, `compute_tps()`, `compute_all_metrics()` |
| `llamatelemetry.bench` | `BenchmarkRunner`, `BenchmarkReport`, `compare_reports()` |
| `llamatelemetry.pipeline` | `PipelineTracer`, `PipelineContext` — ML lifecycle OTel spans |
| `llamatelemetry.semconv` | `keys`, `gen_ai`, `GenAIAttrs`, `set_dual_attrs()` |
| `llamatelemetry.semconv.gen_ai` | Full OTel `gen_ai.*` attribute registry |
| `llamatelemetry.backends` | `LLMBackend` protocol, `LlamaCppBackend`, `TransformersBackend` |
| `llamatelemetry.gpu` | GPU device listing, snapshots, `GPUSpanEnricher` |
| `llamatelemetry.llama` | `LlamaCppClient`, `LlamaCppServer`, GGUF phases, autotune |
| `llamatelemetry.nccl` | NCCL collective tracing, `TorchDistributedInstrumentor` |
| `llamatelemetry.quantization` | GGUF conversion, NF4, dynamic quantization, `QuantizationPipeline` |
| `llamatelemetry.unsloth` | Unsloth loader, LoRA adapter merge, GGUF export |
| `llamatelemetry.otel` | OTel provider, exporters, sampling, span redaction |
| `llamatelemetry.artifacts` | `ArtifactManifest` — pipeline artifact tracking + SHA256 |
| `llamatelemetry.distributed` | `ClusterTopology`, `NodeInfo` — multi-GPU topology detection |
| `llamatelemetry.cuda` | CUDA graphs, tensor core utils, Triton kernel registry |
| `llamatelemetry.kaggle` | Auto-configuration for Kaggle dual T4 |
| `llamatelemetry.sdk` | `instrument_llamacpp()`, `instrument_transformers()` factory |

---

## Hardware Target

**Kaggle Dual Tesla T4** (SM 7.5, 2 × 15 GB VRAM):

- GGUF models up to 30 B parameters at Q4_K_M
- Split-GPU inference via llama.cpp tensor parallelism
- NCCL all-reduce for multi-GPU coordination
- FlashAttention-2 for long-context (4K–8K tokens)

---

## Documentation

| Guide | Path |
|---|---|
| Documentation Index | [docs/INDEX.md](docs/INDEX.md) |
| Quick Start | [docs/QUICK_START_GUIDE.md](docs/QUICK_START_GUIDE.md) |
| Installation | [docs/INSTALLATION.md](docs/INSTALLATION.md) |
| API Reference | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Configuration | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| GGUF Guide | [docs/GGUF_GUIDE.md](docs/GGUF_GUIDE.md) |
| Kaggle Guide | [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) |
| Notebooks | [docs/NOTEBOOKS_GUIDE.md](docs/NOTEBOOKS_GUIDE.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Project Layout

```
llamatelemetry/          Python SDK package (v1.1.0)
  inference/             Production inference subsystem (engine, scheduler, KV cache)
  bench/                 Benchmark harness (runner, report, regression detection)
  pipeline/              ML pipeline OTel spans (finetune → merge → export → quantize → deploy)
  semconv/               Semantic conventions (gen_ai.* OTel standard + legacy llm.*)
  backends/              Unified LLM backend protocol (LlamaCppBackend, TransformersBackend)
  llama/                 llama.cpp integration (client, server, phases, autotune)
  gpu/                   GPU metrics, snapshots, GPUSpanEnricher
  nccl/                  NCCL collective tracing, TorchDistributedInstrumentor
  quantization/          GGUF conversion, NF4, dynamic quant, QuantizationPipeline
  unsloth/               Unsloth loader, LoRA merge (PipelineTracer wired in)
  otel/                  OpenTelemetry providers, exporters, sampling
  cuda/                  CUDA graphs, tensor core utils, Triton kernels
  distributed/           Multi-GPU topology detection
  kaggle/                Kaggle auto-configuration + CUDA binary bootstrap
  sdk.py                 High-level factory (instrument_llamacpp, instrument_transformers)
csrc/                    C++/CUDA kernels (pybind11 bindings)
docs/                    Documentation (guides, reference, architecture)
notebooks/               16 Kaggle Jupyter notebooks (foundation → production)
tests/                   Test suite (244 pass, 24 skip)
releases/
  v1.1.0/                Current release (source + CUDA binary)
  v1.0.0/                Previous release
scripts/                 Build and release utilities
```

---

## Releases

| Version | Source | CUDA Binary | SHA256 |
|---|---|---|---|
| **v1.1.0** (current) | [source.tar.gz](releases/v1.1.0/llamatelemetry-v1.1.0-source.tar.gz) | [cuda12-kaggle-t4x2.tar.gz](releases/v1.1.0/llamatelemetry-v1.1.0-cuda12-kaggle-t4x2.tar.gz) (1.4 GB) | `89b5b7db...` |
| v1.0.0 | [source.tar.gz](releases/v1.0.0/llamatelemetry-v1.0.0-source.tar.gz) | [cuda12-kaggle-t4x2.tar.gz](releases/v1.0.0/llamatelemetry-v1.0.0-cuda12-kaggle-t4x2.tar.gz) (1.4 GB) | `89b5b7db...` |

The CUDA binary ships pre-built llama.cpp + NCCL for **CUDA 12 / Tesla T4 (SM 7.5)**. Downloaded automatically on first import in Kaggle.

---

## License

MIT © 2026 Waqas Muhammad — [waqasm86@gmail.com](mailto:waqasm86@gmail.com)

GitHub: [llamatelemetry/llamatelemetry](https://github.com/llamatelemetry/llamatelemetry)
