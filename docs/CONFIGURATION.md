# Configuration Reference (v1.1.0)

All runtime configuration options for llamatelemetry.

---

## SDK Initialization (`llamatelemetry.init`)

```python
llamatelemetry.init(
    service_name: str,              # Required. OTel service name.
    otlp_endpoint: str = None,      # OTLP HTTP endpoint (e.g. "https://host/v1/traces")
    service_version: str = "1.1.0", # OTel service version attribute.
    enable_gpu: bool = False,        # Enable pynvml GPU metrics collection.
    enable_nccl: bool = False,       # Enable NCCL collective span instrumentation.
    sample_rate: float = 1.0,        # Trace sampling rate (0.0 – 1.0).
    export_interval_ms: int = 5000,  # Metric export interval (ms).
    trace_export_timeout_ms: int = 30000,  # Span export timeout (ms).
)
```

---

## Inference Engine (`CudaInferenceConfig`)

```python
from llamatelemetry.inference.types import CudaInferenceConfig

config = CudaInferenceConfig(
    gpu_layers: int = 99,           # Layers offloaded to GPU (-1 = all).
    tensor_split: str = "1.0,0.0", # GPU memory split (GPU0, GPU1).
    ctx_size: int = 4096,           # Context window size (tokens).
    batch_size: int = 512,          # Batch/chunk size for prompt processing.
    flash_attn: bool = True,        # Enable FlashAttention-2 (SM 7.5+).
    n_parallel: int = 1,            # Parallel inference slots.
    rope_freq_base: float = 0.0,    # RoPE base frequency (0 = model default).
    cache_type_k: str = "f16",      # KV cache key type.
    cache_type_v: str = "f16",      # KV cache value type.
)
```

---

## Sampling Parameters (`SamplingParams`)

```python
from llamatelemetry.inference.types import SamplingParams

params = SamplingParams(
    temperature: float = 0.8,       # Sampling temperature.
    top_p: float = 0.95,            # Nucleus sampling probability.
    top_k: int = 40,                # Top-k sampling cutoff.
    repeat_penalty: float = 1.1,    # Repetition penalty (1.0 = off).
    min_p: float = 0.05,            # Min-p sampling threshold.
    seed: int = -1,                 # Random seed (-1 = random).
)
```

---

## Benchmark Runner (`BenchmarkRunner`)

```python
from llamatelemetry.bench import BenchmarkRunner

runner = BenchmarkRunner(
    backend: str = "llama.cpp",     # "llama.cpp" or "transformers"
    n_warmup: int = 2,              # Warmup requests before timing.
    n_runs: int = 5,                # Timed inference runs per test.
    concurrency: int = 1,           # Concurrent requests.
)

report = runner.run(
    llama_server_url: str = "http://127.0.0.1:8090",
    prompts: list[str] = None,      # Custom prompts (uses defaults if None).
    n_tokens: int = 128,            # Tokens to generate per run.
)
```

### Regression Detection

```python
from llamatelemetry.bench import compare_reports

comparisons = compare_reports(
    baseline: dict,                 # BenchmarkReport.to_dict() from saved baseline.
    current: dict,                  # BenchmarkReport.to_dict() from current run.
    regression_threshold_pct: float = 10.0,  # Delta % that triggers regression flag.
)
```

---

## llama.cpp Server (`ServerManager`)

```python
from llamatelemetry.llama import ServerManager

server = ServerManager()
server.start_server(
    model_path: str,                # Path to GGUF model file.
    host: str = "127.0.0.1",
    port: int = 8090,
    gpu_layers: int = 99,
    tensor_split: str = "1.0,0.0",
    split_mode: str = "row",        # "row" or "layer"
    ctx_size: int = 4096,
    batch_size: int = 512,
    flash_attn: int = 1,            # 0 or 1
    n_parallel: int = 1,
)
server.stop_server()
```

### Presets

```python
llamatelemetry.llama.quick_start(
    model_path: str,
    preset: str = "kaggle_t4_dual",  # "kaggle_t4_dual" | "single_gpu" | "cpu"
)
```

| Preset | gpu_layers | tensor_split | ctx_size |
|---|---|---|---|
| `kaggle_t4_dual` | 99 | `"1.0,0.0"` | 4096 |
| `single_gpu` | 99 | `"1.0"` | 4096 |
| `cpu` | 0 | `""` | 2048 |

---

## OTel Provider (`llamatelemetry.otel`)

```python
from llamatelemetry.otel import setup_provider

provider = setup_provider(
    service_name: str,
    service_version: str = "1.1.0",
    otlp_endpoint: str = None,      # None = stdout exporter.
    sample_rate: float = 1.0,
    export_interval_ms: int = 5000,
)
```

---

## Kaggle Auto-Configure (`llamatelemetry.kaggle`)

```python
from llamatelemetry.kaggle import auto_configure

cfg = auto_configure(
    binary_version: str = "1.0.0",  # CUDA binary version to bootstrap.
    cache_dir: str = "/kaggle/working/.llamatelemetry",
    load_secrets: bool = True,       # Load Kaggle secrets into env.
)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMATELEMETRY_SERVICE_NAME` | `"llamatelemetry"` | OTel service name |
| `LLAMATELEMETRY_OTLP_ENDPOINT` | `""` | OTLP HTTP endpoint |
| `LLAMATELEMETRY_SAMPLE_RATE` | `"1.0"` | Trace sample rate |
| `LLAMATELEMETRY_BINARY_VERSION` | `"1.0.0"` | CUDA binary version |
| `LLAMATELEMETRY_CACHE_DIR` | `"/kaggle/working/.llamatelemetry"` | Binary cache directory |
| `LLAMATELEMETRY_LOG_LEVEL` | `"INFO"` | SDK log level |
