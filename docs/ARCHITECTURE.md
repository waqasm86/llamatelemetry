# llamatelemetry Architecture

## Overview
llamatelemetry is a Python SDK that orchestrates GPU-accelerated LLM inference
around llama.cpp binaries, with optional OpenTelemetry-based observability and
GPU-native graph visualization workflows. The package is intentionally light on
Python code and pulls large CUDA binaries on first import.

High-level goals:
- Simple, PyTorch-style Python API for inference.
- Hybrid bootstrap that auto-downloads llama.cpp CUDA binaries.
- First-class telemetry for tracing + metrics (OpenTelemetry + OTLP).
- Split-GPU workflows for inference + graph analytics on Kaggle T4s.

## Core Layers

1) Python SDK (user-facing)
- `llamatelemetry/__init__.py`: `InferenceEngine`, bootstrap, quick helpers.
- `llamatelemetry/server.py`: llama-server lifecycle management.
- `llamatelemetry/models.py`: GGUF model loading + registry + HF downloads.

2) API Surface (llama.cpp server client)
- `llamatelemetry/api/*`: OpenAI-compatible, native completion, embeddings,
  slots, LoRA, GGUF utilities, multi-GPU configuration.

3) Observability (optional)
- `llamatelemetry/telemetry/*`: OTel tracer + meter + exporters + GPU metrics.
- `llamatelemetry/telemetry/graphistry_export.py`: span-to-graph conversion.

4) Performance Utilities (optional)
- `llamatelemetry/inference/*`: FlashAttention hooks, KV cache, batching.
- `llamatelemetry/cuda/*`: CUDA graphs, Triton kernel registry, tensor cores.

5) Graph + Knowledge Workflows (optional)
- `llamatelemetry/graphistry/*`: GPU graph visualization + RAPIDS helpers.
- `llamatelemetry/louie/*`: Knowledge extraction and graph analysis.

6) Native Extension
- `csrc/*`: minimal CUDA ops, device + tensor primitives exposed via pybind11.
- `core/__init__.py`: Python wrapper types around the native extension.

## Data Flow (Typical Inference)

1. Import `llamatelemetry`.
2. Bootstrap downloads CUDA binaries on first import if missing.
3. `InferenceEngine.load_model()` resolves GGUF model path (local or HF).
4. `ServerManager.start_server()` launches `llama-server` with CUDA args.
5. `InferenceEngine.infer()` sends POST to `/completion` endpoint.
6. Result is returned as `InferResult`, with local latency/token metrics.
7. Optional: OTel spans + metrics are emitted when telemetry is enabled.

## Binary Bootstrap Strategy

- Python package is small (code only).
- CUDA binaries live in release tarballs (llama.cpp build artifacts).
- On import, if binaries are missing, `llamatelemetry/_internal/bootstrap.py`
  downloads the tarball and configures `LD_LIBRARY_PATH` and
  `LLAMA_SERVER_PATH`.

## Multi-GPU Support

- Multi-GPU logic is mostly exposed via `llamatelemetry/api/multigpu.py`.
- Kaggle dual Tesla T4 is the primary target.
- Split-GPU workflows allow inference on GPU 0 and graph analytics on GPU 1.

## Observability Model

- `setup_telemetry()` builds OTel Resource with GPU metadata.
- Tracing: spans with LLM semantic conventions.
- Metrics: GPU utilization, memory, inference latency, token counters.
- Export: OTLP gRPC/HTTP or console; optional Graphistry visualization.

## Packaging Notes

- `pyproject.toml` excludes large binaries from wheels.
- SDK version (0.1.0) is separate from binary artifact version (0.1.0).
- Primary distribution is GitHub + HF for binaries.

