# Golden Path (v1.2.0)

This page shows the simplest, end-to-end path to get value from llamatelemetry:
init → create_engine → generate/chat/embeddings → dashboards.

---

## 1) Install

```bash
pip install llamatelemetry
```

Optional extras:

```bash
pip install "llamatelemetry[gpu]"
pip install "llamatelemetry[all]"
```

---

## 2) Initialize telemetry once

```python
import llamatelemetry

llamatelemetry.init(
    service_name="my-llm-app",
    otlp_endpoint="https://otlp.example.com/v1/traces",
    enable_gpu=True,
    redact_prompts=True,  # recommended for production
)
```

---

## 3) Create an engine (GGUF or Transformers)

### GGUF (llama.cpp server)

```python
from llamatelemetry.inference.api import create_engine

engine = create_engine(
    backend="llama.cpp",
    llama_server_url="http://127.0.0.1:8090",
    telemetry=True,
)
engine.start()
```

### Transformers (original models)

```python
from llamatelemetry.inference.api import create_engine

engine = create_engine(
    backend="transformers",
    model_id="mistral-7b-instruct",
    telemetry=True,
)
engine.start()
```

---

## 4) Run inference

```python
from llamatelemetry.inference.base import InferenceRequest
from llamatelemetry.inference.types import SamplingParams

req = InferenceRequest(
    messages=[{"role": "user", "content": "Explain GGUF quantization."}],
    max_tokens=256,
    sampling=SamplingParams(temperature=0.7, top_p=0.9),
)

result = engine.generate(req)
print(f"TPS: {result.tps:.1f}  TTFT: {result.ttft_ms:.0f}ms  VRAM: {result.vram_peak_mb:.0f}MB")
```

---

## 5) Events, metrics, and redaction

Enable GenAI operation detail events (opt-in):

```python
from llamatelemetry.transformers.instrumentation import TransformersInstrumentorConfig

cfg = TransformersInstrumentorConfig(
    record_events=True,
    record_content=False,  # keep prompts out of telemetry by default
)
```

Redaction is enforced by the RedactionSpanProcessor when `redact_prompts=True`.

---

## 6) Shutdown cleanly

```python
engine.shutdown()
llamatelemetry.shutdown()
```

---

## Notes

- GenAI attributes follow the OpenTelemetry `gen_ai.*` registry by default.
- Phase spans (`llamatelemetry.phase.prefill` / `llamatelemetry.phase.decode`) carry timing attributes when available.
