# API Reference (v1.2.0)

Full reference for all public llamatelemetry modules and classes.

---

## Top-Level (`llamatelemetry`)

```python
import llamatelemetry

llamatelemetry.init(
    service_name: str,
    otlp_endpoint: str = None,
    enable_gpu: bool = False,
    enable_nccl: bool = False,
    service_version: str = "1.2.0",
)

llamatelemetry.shutdown()

@llamatelemetry.trace(name: str = None, attributes: dict = None)
@llamatelemetry.span(name: str)
@llamatelemetry.workflow(name: str)
```

---

## Inference Engine (`llamatelemetry.inference`)

```python
from llamatelemetry.inference.api import create_engine
from llamatelemetry.inference.base import InferenceRequest, InferenceResult
from llamatelemetry.inference.types import SamplingParams, CudaInferenceConfig

engine = create_engine(
    backend: str,                   # "llama.cpp" or "transformers"
    llama_server_url: str = None,   # for llama.cpp backend
    config: CudaInferenceConfig = None,
)
engine.start()
engine.shutdown()

result: InferenceResult = engine.generate(request: InferenceRequest)
```

### `InferenceRequest`

| Field | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | Chat messages `[{"role": ..., "content": ...}]` |
| `max_tokens` | `int` | Max tokens to generate |
| `sampling` | `SamplingParams` | Sampling configuration |
| `request_id` | `str` | Optional request identifier |

### `InferenceResult`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Generated text |
| `ttft_ms` | `float` | Time-to-first-token (ms) |
| `tps` | `float` | Tokens per second |
| `vram_peak_mb` | `float` | Peak VRAM usage (MB) |
| `prefill_tps` | `float` | Prefill throughput (tok/s) |
| `queue_delay_ms` | `float` | Time waiting in request queue (ms) |

### `SamplingParams`

```python
SamplingParams(
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
)
```

### `CudaInferenceConfig`

```python
CudaInferenceConfig(
    gpu_layers: int = 99,
    tensor_split: str = "1.0,0.0",
    ctx_size: int = 4096,
    batch_size: int = 512,
    flash_attn: bool = True,
)
```

---

## Inference Events (`llamatelemetry.inference.events`)

```python
from llamatelemetry.inference.events import InferenceEvents, EventRecorder

events = InferenceEvents()
events.record_request_start()
events.record_first_token()
events.record_completion(n_tokens: int)

recorder = EventRecorder()
recorder.start()
recorder.first_token()
recorder.done(n_tokens: int)
```

---

## Inference Metrics (`llamatelemetry.inference.metrics`)

```python
from llamatelemetry.inference.metrics import compute_ttft, compute_tps, compute_all_metrics

ttft_ms = compute_ttft(events: InferenceEvents)
tps = compute_tps(events: InferenceEvents, n_tokens: int)
metrics = compute_all_metrics(events: InferenceEvents, n_tokens: int)
```

---

## Benchmark Harness (`llamatelemetry.bench`)

```python
from llamatelemetry.bench import BenchmarkRunner, BenchmarkReport, compare_reports

runner = BenchmarkRunner(backend: str = "llama.cpp")
report: BenchmarkReport = runner.run(
    llama_server_url: str = "http://127.0.0.1:8090",
    prompts: list[str] = None,
    n_tokens: int = 128,
)

report.save(path: str)
report2 = BenchmarkReport.load(path: str)

comparisons = compare_reports(
    baseline: dict,
    current: dict,
    regression_threshold_pct: float = 10.0,
)
for c in comparisons:
    print(c.test_name, c.metric, c.delta_pct, c.regression)
```

---

## Pipeline Tracer (`llamatelemetry.pipeline`)

```python
from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext

ctx = PipelineContext(
    base_model: str = None,
    adapter: str = None,
    quantization: str = None,
    output_artifact: str = None,
)

tracer = PipelineTracer()

with tracer.span_finetune(ctx): ...
with tracer.span_merge_lora(ctx): ...
with tracer.span_export_gguf(ctx): ...
with tracer.span_quantize(ctx): ...
with tracer.span_benchmark(ctx): ...
with tracer.span_deploy(ctx): ...
```

---

## Semantic Conventions (`llamatelemetry.semconv`)

```python
from llamatelemetry.semconv import gen_ai
from llamatelemetry.semconv.mapping import set_gen_ai_attrs

# Emit gen_ai.* on a span
set_gen_ai_attrs(span, request)

# Direct attribute access
span.set_attribute(gen_ai.GEN_AI_PROVIDER_NAME, "llama_cpp")
span.set_attribute(gen_ai.GEN_AI_OPERATION_NAME, "chat")
span.set_attribute(gen_ai.GEN_AI_REQUEST_MODEL, "llama-3-8b")
span.set_attribute(gen_ai.GEN_AI_USAGE_INPUT_TOKENS, 42)
span.set_attribute(gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS, 128)
```

---

## GPU (`llamatelemetry.gpu`)

```python
from llamatelemetry.gpu import list_devices, GPUSnapshot, GPUSpanEnricher

devices = list_devices()
snapshot = GPUSnapshot.capture(device_id=0)

enricher = GPUSpanEnricher(device_id=0)
with enricher.enrich(span):
    # span gets before/after VRAM snapshots and delta
    ...
```

---

## llama.cpp (`llamatelemetry.llama`)

```python
from llamatelemetry.llama import LlamaCppClient, ServerManager, quick_start

# One-liner server start
quick_start(model_path: str, preset: str = "kaggle_t4_dual")

# Manual server control
server = ServerManager()
server.start_server(model_path, gpu_layers=99, tensor_split="1.0,0.0", flash_attn=1)
server.stop_server()

# Inference client
client = LlamaCppClient(base_url: str = "http://127.0.0.1:8090")
resp = client.chat.create(messages=[...], max_tokens=256)
print(resp.choices[0].message["content"])
```

---

## NCCL (`llamatelemetry.nccl`)

```python
from llamatelemetry.nccl import NCCLTracer, TorchDistributedInstrumentor

tracer = NCCLTracer()
tracer.trace_all_reduce(tensor, op="sum")

instrumentor = TorchDistributedInstrumentor()
instrumentor.instrument()   # patches torch.distributed ops with OTel spans
instrumentor.uninstrument()
```

---

## Quantization (`llamatelemetry.quantization`)

```python
from llamatelemetry.quantization import QuantizationPipeline
from llamatelemetry.quantization.gguf import convert_to_gguf
from llamatelemetry.quantization.nf4 import quantize_nf4
from llamatelemetry.quantization.dynamic import quantize_dynamic

pipeline = QuantizationPipeline(pipeline_ctx=ctx)
pipeline.run(model, output_path="model-q4.gguf", quant_type="Q4_K_M")
```

---

## Unsloth (`llamatelemetry.unsloth`)

```python
from llamatelemetry.unsloth import load_model, merge_lora_adapters, export_to_llamatelemetry

model, tokenizer = load_model("unsloth/llama-3-8b", load_in_4bit=True)

merged = merge_lora_adapters(model, ctx=pipeline_ctx)

output_path = export_to_llamatelemetry(
    model=merged,
    tokenizer=tokenizer,
    output_path="/kaggle/working/model-q4.gguf",
    pipeline_ctx=pipeline_ctx,
)
```

---

## OTel Provider (`llamatelemetry.otel`)

```python
from llamatelemetry.otel import setup_provider, get_tracer, get_meter
from llamatelemetry.otel.exporters import build_otlp_exporter
from llamatelemetry.otel.sampling import configure_sampler

provider = setup_provider(service_name="my-app", otlp_endpoint="...")
tracer = get_tracer("llamatelemetry")
meter = get_meter("llamatelemetry")
```

---

## Artifacts (`llamatelemetry.artifacts`)

```python
from llamatelemetry.artifacts import ArtifactManifest

manifest = ArtifactManifest(pipeline_id="run-001")
manifest.add(path="model-q4.gguf", kind="gguf_model")
manifest.save("manifest.json")
manifest2 = ArtifactManifest.load("manifest.json")
```

---

## Distributed (`llamatelemetry.distributed`)

```python
from llamatelemetry.distributed import ClusterTopology, NodeInfo

topology = ClusterTopology.detect()
for node in topology.nodes:
    print(node.rank, node.gpu_id, node.vram_mb)
```

---

## Backends (`llamatelemetry.backends`)

```python
from llamatelemetry.backends import LLMBackend, LlamaCppBackend, TransformersBackend

# LLMBackend protocol — implement for custom backends
class MyBackend(LLMBackend):
    def generate(self, request: InferenceRequest) -> InferenceResult: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...
```

---

## Kaggle (`llamatelemetry.kaggle`)

```python
from llamatelemetry.kaggle import auto_configure

cfg = auto_configure()
# Auto-detects dual T4, sets LD_LIBRARY_PATH, loads Kaggle secrets
```

---

## SDK Factory (`llamatelemetry.sdk`)

```python
from llamatelemetry.sdk import instrument_llamacpp, instrument_transformers

# High-level factory — patches client/model with OTel instrumentation
instrument_llamacpp(client)
instrument_transformers(model, tokenizer)
```
