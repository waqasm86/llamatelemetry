# Architecture Overview

llamatelemetry is a CUDA-first observability layer for GGUF inference.

## Core Components
- **llama.cpp** server for GGUF inference
- **NCCL** for multi-GPU split and transport
- **OpenTelemetry** for traces and metrics
- **Graphistry + RAPIDS** for graph-based analytics

## Split-GPU Pattern
- GPU0: llama.cpp server and GGUF model
- GPU1: RAPIDS + Graphistry visualization

## Data Flow
1. Request hits llama-server
2. OTel spans/metrics recorded in Python SDK
3. Optional OTLP export to backend
4. Visualization from captured spans
