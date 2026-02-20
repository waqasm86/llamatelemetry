# Quick Start Guide (v1.2.0)

Get llamatelemetry running on Kaggle dual T4 in minutes.

## Requirements

- Kaggle notebook with **GPU T4 × 2** accelerator
- Internet enabled (for pip install + CUDA binary download)

---

## 1. Install

```python
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v1.2.0
```

---

## 2. Auto-Configure (Kaggle Dual T4)

```python
import llamatelemetry
from llamatelemetry.kaggle import auto_configure

# Detects dual T4, sets LD_LIBRARY_PATH, downloads CUDA binary on first run
cfg = auto_configure()

llamatelemetry.init(
    service_name="kaggle-llm",
    enable_gpu=True,
    enable_nccl=True,
    otlp_endpoint="https://otlp.example.com/v1/traces",  # optional
)
```

---

## 3. Download a GGUF Model

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)
```

---

## 4. Start llama.cpp Server (GPU 0)

```python
# One-liner with dual-T4 preset
llamatelemetry.llama.quick_start(model_path=model_path, preset="kaggle_t4_dual")

# Or manual control
from llamatelemetry.llama import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
    ctx_size=4096,
)
```

---

## 5. Run Inference with Full Telemetry

### Using the Inference Engine (recommended)

```python
from llamatelemetry.inference.api import create_engine
from llamatelemetry.inference.base import InferenceRequest
from llamatelemetry.inference.types import SamplingParams

engine = create_engine(backend="llama.cpp", llama_server_url="http://127.0.0.1:8090")
engine.start()

result = engine.generate(InferenceRequest(
    messages=[{"role": "user", "content": "What is CUDA?"}],
    max_tokens=128,
    sampling=SamplingParams(temperature=0.7, top_p=0.9),
))

print(result.text)
print(f"TPS: {result.tps:.1f}  TTFT: {result.ttft_ms:.0f}ms  VRAM: {result.vram_peak_mb:.0f}MB")
```

### Using the llama.cpp Client directly

```python
from llamatelemetry.llama import LlamaCppClient, trace_request

client = LlamaCppClient("http://127.0.0.1:8090")

with trace_request(model="gemma-3-1b", request_id="r1") as req:
    resp = client.chat.create(
        messages=[{"role": "user", "content": "Explain GPU tensor cores."}],
        max_tokens=80,
    )
    req.set_completion_tokens(resp.usage.completion_tokens)

print(resp.choices[0].message["content"])
```

---

## 6. Use Trace Decorators

```python
@llamatelemetry.trace()
def ask(question: str) -> str:
    resp = client.chat.create(
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    return resp.choices[0].message["content"]

answer = ask("Explain GGUF quantization.")
```

---

## 7. Benchmark + Regression Detection

```python
from llamatelemetry.bench import BenchmarkRunner, BenchmarkReport, compare_reports

runner = BenchmarkRunner(backend="llama.cpp")
report = runner.run(llama_server_url="http://127.0.0.1:8090")
report.save("baseline.json")
print(f"Avg TPS: {report.avg_tps:.1f}  Avg TTFT: {report.avg_ttft_ms:.0f}ms")
```

---

## 8. Pipeline Observability (finetune → deploy)

```python
from llamatelemetry.pipeline.spans import PipelineTracer, PipelineContext

ctx = PipelineContext(base_model="llama-3-8b", adapter="my-lora", quantization="Q4_K_M")
tracer = PipelineTracer()

with tracer.span_merge_lora(ctx):
    merged = merge_lora_adapters(model, ctx=ctx)

with tracer.span_export_gguf(ctx):
    export_to_llamatelemetry(merged, tokenizer, "/kaggle/working/model.gguf", pipeline_ctx=ctx)

with tracer.span_benchmark(ctx):
    runner.run()

with tracer.span_deploy(ctx):
    llamatelemetry.llama.quick_start("/kaggle/working/model.gguf")
```

---

## 9. Shutdown

```python
engine.shutdown()
llamatelemetry.shutdown()
```

---

## Next Steps

| Guide | Path |
|---|---|
| Notebooks (16 tutorials) | `docs/NOTEBOOKS_GUIDE.md` |
| Full API reference | `docs/API_REFERENCE.md` |
| Configuration reference | `docs/CONFIGURATION.md` |
| GGUF model selection | `docs/GGUF_GUIDE.md` |
| Kaggle optimization tips | `docs/KAGGLE_GUIDE.md` |
