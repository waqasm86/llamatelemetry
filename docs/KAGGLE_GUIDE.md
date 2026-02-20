# Kaggle Guide

This guide summarizes the Kaggle workflow for llamatelemetry.

## Recommended Flow
1. Install the package (see `docs/INSTALLATION.md`)
2. Download a small GGUF model (1B-5B)
3. Start the server on GPU0/1 (tensor split) or load Transformers with device_map="auto"
4. Run inference via `InferenceRuntime`
5. Capture telemetry and visualize with Graphistry on GPU1

## Notes
- Prefer split-GPU: GPU0 inference, GPU1 analytics
- Use explicit OTLP endpoints with `/v1/traces` and `/v1/metrics`
