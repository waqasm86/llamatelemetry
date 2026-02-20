# Architecture Overview (v1.1.0)

llamatelemetry is a CUDA-first OpenTelemetry Python SDK for LLM inference observability.

---

## Core Design Principles

- **CUDA-first**: Built for GPU-native inference; GPU metrics are first-class citizens.
- **Torch-optional**: Core modules (inference, bench, pipeline, semconv, otel) work without PyTorch. GPU-heavy modules (cuda, unsloth, nccl, quantization) raise clear errors when torch is absent.
- **OTel-standard**: All telemetry emits OpenTelemetry spans and metrics. Compatible with any OTLP backend.
- **Dual-emit**: Semantic conventions emit both `gen_ai.*` (OTel standard) and legacy `llm.*` attributes.

---

## Subsystem Map

```
llamatelemetry/
├── inference/          Production inference engine (engine, scheduler, KV cache, events, metrics)
├── bench/              Benchmark harness (runner, report, regression detection)
├── pipeline/           ML pipeline OTel spans (finetune → merge → export → quantize → deploy)
├── semconv/            Semantic conventions (gen_ai.* OTel standard + legacy llm.*)
├── backends/           Unified LLM backend protocol (LlamaCppBackend, TransformersBackend)
├── llama/              llama.cpp integration (client, server, phases, autotune)
├── gpu/                GPU metrics, snapshots, GPUSpanEnricher
├── nccl/               NCCL collective tracing, TorchDistributedInstrumentor
├── quantization/       GGUF conversion, NF4, dynamic quant, QuantizationPipeline
├── unsloth/            Unsloth loader, LoRA merge, GGUF export (PipelineTracer wired in)
├── otel/               OTel providers, exporters, sampling, span redaction
├── cuda/               CUDA graphs, tensor core utils, Triton kernel registry
├── distributed/        Multi-GPU topology detection
├── kaggle/             Kaggle auto-configuration + CUDA binary bootstrap
├── artifacts/          ArtifactManifest — pipeline artifact tracking + SHA256
└── sdk.py              High-level factory (instrument_llamacpp, instrument_transformers)
```

---

## Split-GPU Pattern (Kaggle Dual T4)

```
GPU 0 (15 GB)                     GPU 1 (15 GB)
─────────────────────              ─────────────────────
llama.cpp server                   RAPIDS + cuML
GGUF model (Q4_K_M)                Graphistry visualization
Inference engine                   NCCL coordination
OTel span emission
```

- GPU 0 handles all inference traffic via llama.cpp tensor parallelism.
- GPU 1 is available for analytics, visualization, and NCCL all-reduce.
- NCCL provides inter-GPU coordination for large model splits.

---

## Inference Data Flow

```
User request
    │
    ▼
InferenceEngine.generate(InferenceRequest)
    │
    ├─ RequestScheduler (queue, paged KV cache allocation)
    │
    ├─ LlamaCppBackend / TransformersBackend
    │       │
    │       ├─ llama.cpp HTTP API (GPU 0)
    │       └─ OTel span: gen_ai.* + llm.* attributes
    │
    ├─ EventRecorder (TTFT, TPOT timestamps)
    │
    ├─ GPUSpanEnricher (before/after VRAM snapshots + delta)
    │
    └─ InferenceResult (text, ttft_ms, tps, vram_peak_mb, ...)
```

---

## ML Pipeline Observability

```
finetune ──► merge_lora ──► export_gguf ──► quantize ──► benchmark ──► deploy
    │             │               │              │             │            │
    └─────────────┴───────────────┴──────────────┴─────────────┴────────────┘
                        PipelineTracer OTel spans
                        PipelineContext (base_model, adapter, quantization, artifact)
```

Each stage emits a nested OTel span with pipeline context attributes. Artifacts are tracked via `ArtifactManifest` with SHA256 checksums.

---

## OpenTelemetry Integration

```
llamatelemetry.init(service_name, otlp_endpoint)
    │
    ├─ TracerProvider (OTLP exporter or stdout)
    ├─ MeterProvider (OTLP or Prometheus)
    └─ Sampling config (head-based, always-on, or ratio)

span attributes:
    gen_ai.system           = "llamacpp"
    gen_ai.request.model    = "llama-3-8b"
    gen_ai.usage.input_tokens  = 42
    gen_ai.usage.output_tokens = 128
    llm.model               = "llama-3-8b"     (legacy dual-emit)
    llm.request.type        = "chat"            (legacy dual-emit)
```

---

## CUDA Binary Bootstrap

On first import in Kaggle:
1. `llamatelemetry.kaggle.bootstrap` checks for cached binary.
2. If absent, downloads `llamatelemetry-v1.0.0-cuda12-kaggle-t4x2.tar.gz` (~1.4 GB) from GitHub Releases.
3. SHA256 is verified before extraction.
4. `LD_LIBRARY_PATH` is set to include extracted llama.cpp and NCCL libraries.
5. Subsequent imports skip download (cached in `/kaggle/working/.llamatelemetry/`).

---

## Hardware Target

**Kaggle Dual Tesla T4** (SM 7.5, 2 × 15 GB VRAM):

| Capability | Detail |
|---|---|
| Max model size | ~30B parameters at Q4_K_M |
| Inference throughput | 30–50 tok/s (single T4) |
| Context length | 4K–8K tokens (FlashAttention-2) |
| Multi-GPU | NCCL all-reduce + llama.cpp tensor parallelism |
| VRAM efficiency | Q4_K_M achieves ~7.8× compression vs FP32 |
