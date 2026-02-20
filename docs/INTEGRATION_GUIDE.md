# Integration Guide

This guide explains how to integrate llamatelemetry with your inference workflow.

## Typical Flow
1. Start llama.cpp server with `ServerManager`
2. Use `LlamaCppClient` for inference
3. Enable telemetry and export via OTLP
4. Visualize spans using Graphistry
5. Build a pipeline graph (requests -> model -> GPUs -> phases -> exporter)

## Files to Reference
- `docs/QUICK_START_GUIDE.md`
- `docs/CONFIGURATION.md`
- `docs/TROUBLESHOOTING.md`

## Graphistry pipeline graph

```python
from llamatelemetry.graphistry import GraphistryViz

viz = GraphistryViz(auto_register=True)
viz.plot_pipeline_graph(
    model_name="gemma-3-4b-Q4_K_M",
    backend="llama.cpp",
    gpu_names=["T4", "T4"],
    tensor_split=[0.5, 0.5],
    server_url="http://127.0.0.1:8080",
)
```
