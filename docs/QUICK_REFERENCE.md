# Quick Reference (v1.2.0)

## Module Cheat-Sheet

| Module | Key exports |
|---|---|
| `llamatelemetry` | `init()`, `shutdown()`, `trace()`, `span()`, `workflow()` |
| `llamatelemetry.inference` | `create_engine()`, `InferenceRequest`, `InferenceResult`, `CudaInferenceConfig` |
| `llamatelemetry.inference.events` | `InferenceEvents`, `EventRecorder` |
| `llamatelemetry.inference.metrics` | `compute_ttft()`, `compute_tps()`, `compute_all_metrics()` |
| `llamatelemetry.bench` | `BenchmarkRunner`, `BenchmarkReport`, `compare_reports()` |
| `llamatelemetry.pipeline` | `PipelineTracer`, `PipelineContext` |
| `llamatelemetry.semconv` | `keys`, `gen_ai`, `GenAIAttrs`, `set_gen_ai_attrs()` |
| `llamatelemetry.semconv.gen_ai` | Full `gen_ai.*` OTel attribute registry |
| `llamatelemetry.backends` | `LLMBackend`, `LlamaCppBackend`, `TransformersBackend` |
| `llamatelemetry.gpu` | `list_devices()`, `GPUSnapshot`, `GPUSpanEnricher` |
| `llamatelemetry.llama` | `LlamaCppClient`, `ServerManager`, `quick_start()` |
| `llamatelemetry.nccl` | `NCCLTracer`, `TorchDistributedInstrumentor` |
| `llamatelemetry.quantization` | `QuantizationPipeline`, `convert_to_gguf()`, `quantize_nf4()` |
| `llamatelemetry.unsloth` | `load_model()`, `merge_lora_adapters()`, `export_to_llamatelemetry()` |
| `llamatelemetry.otel` | `setup_provider()`, `get_tracer()`, `get_meter()` |
| `llamatelemetry.artifacts` | `ArtifactManifest` |
| `llamatelemetry.distributed` | `ClusterTopology`, `NodeInfo` |
| `llamatelemetry.cuda` | CUDA graphs, tensor core utils, Triton kernel registry |
| `llamatelemetry.kaggle` | `auto_configure()` |
| `llamatelemetry.sdk` | `instrument_llamacpp()`, `instrument_transformers()` |

---

## Common Snippets

### Initialize SDK

```python
import llamatelemetry
llamatelemetry.init(service_name="my-app", otlp_endpoint="https://otlp.example.com/v1/traces")
```

### Trace a function

```python
@llamatelemetry.trace()
def generate(prompt: str) -> str:
    ...
```

### Run inference with engine

```python
from llamatelemetry.inference.api import create_engine
from llamatelemetry.inference.base import InferenceRequest

engine = create_engine(backend="llama.cpp", llama_server_url="http://127.0.0.1:8090")
engine.start()
result = engine.generate(InferenceRequest(messages=[{"role": "user", "content": "Hello"}]))
print(result.tps, result.ttft_ms, result.vram_peak_mb)
engine.shutdown()
```

### Run benchmark + detect regression

```python
from llamatelemetry.bench import BenchmarkRunner, BenchmarkReport, compare_reports
runner = BenchmarkRunner(backend="llama.cpp")
report = runner.run(llama_server_url="http://127.0.0.1:8090")
report.save("baseline.json")
```

### Emit GenAI OTel attributes

```python
from llamatelemetry.semconv import gen_ai
from llamatelemetry.semconv.mapping import set_gen_ai_attrs
with tracer.start_as_current_span("chat gemma-3-1b") as span:
    set_gen_ai_attrs(span, {
        "provider": gen_ai.PROVIDER_LLAMA_CPP,
        "operation": gen_ai.OP_CHAT,
        "model": "gemma-3-1b",
        "input_tokens": 42,
        "output_tokens": 128,
    })
```

### Pipeline observability

```python
from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext
ctx = PipelineContext(base_model="llama-3-8b", quantization="Q4_K_M")
tracer = PipelineTracer()
with tracer.span_merge_lora(ctx): ...
with tracer.span_export_gguf(ctx): ...
with tracer.span_quantize(ctx): ...
with tracer.span_benchmark(ctx): ...
with tracer.span_deploy(ctx): ...
```

### Kaggle auto-configure

```python
from llamatelemetry.kaggle import auto_configure
cfg = auto_configure()
llamatelemetry.init(service_name="kaggle-inference", enable_gpu=True, enable_nccl=True)
```

---

## Key Docs

| Guide | Path |
|---|---|
| Documentation Index | `docs/INDEX.md` |
| Installation | `docs/INSTALLATION.md` |
| Quick Start | `docs/QUICK_START_GUIDE.md` |
| API Reference | `docs/API_REFERENCE.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Configuration | `docs/CONFIGURATION.md` |
| GGUF Guide | `docs/GGUF_GUIDE.md` |
| Kaggle Guide | `docs/KAGGLE_GUIDE.md` |
| Notebooks | `docs/NOTEBOOKS_GUIDE.md` |
| Changelog | `CHANGELOG.md` |
