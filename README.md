# llamatelemetry v0.1.0

CUDA-first OpenTelemetry Python SDK for LLM inference observability and explainability.

llamatelemetry combines:
- llama.cpp GGUF inference
- NCCL-aware multi-GPU execution
- OpenTelemetry traces + metrics
- GPU analytics and visualization with RAPIDS + Graphistry

This repository is optimized for Kaggle dual T4 notebooks and small GGUF models (1B-5B, Q4_K_M). It ships lightweight Python code and downloads large CUDA binaries on first import.

## What You Get
- LLM request tracing with semantic attributes
- GPU-aware metrics (latency, tokens, utilization)
- Split-GPU workflow (GPU0 inference, GPU1 analytics)
- Graph-based trace visualization

## Quick Start (Kaggle Dual T4)
```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

from huggingface_hub import hf_hub_download
from llamatelemetry.server import ServerManager

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
)
```

## Documentation Map
Start here:
- `docs/INDEX.md`

Common entry points:
- `docs/INSTALLATION.md`
- `docs/QUICK_START_GUIDE.md`
- `docs/ARCHITECTURE.md`
- `docs/INTEGRATION_GUIDE.md`
- `docs/NOTEBOOKS_GUIDE.md`
- `docs/TROUBLESHOOTING.md`

Notebooks:
- `notebooks/README.md`
- `notebooks/14-15-16-INDEX.md`

## Project Layout (High Level)
- `llamatelemetry/` Python SDK
- `csrc/` CUDA/C++ kernels and bindings
- `docs/` Guides and references
- `notebooks/` Authoritative notebook specs
- `examples/` Minimal runnable examples
- `scripts/` Build and release utilities

## Support and Contribution
- `CONTRIBUTING.md`
- `CHANGELOG.md`
