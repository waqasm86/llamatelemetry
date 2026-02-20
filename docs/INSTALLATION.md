# Installation (v1.1.0)

llamatelemetry v1.1.0 targets Kaggle dual Tesla T4 notebooks (CUDA 12, SM 7.5).

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| CUDA runtime | 12.x (pre-installed on Kaggle) |
| GPU | Tesla T4 × 2 (recommended) or any CUDA-capable GPU |
| Internet | Required on first import (CUDA binary auto-download) |

## Install (Kaggle / pip)

```python
# Latest stable
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v1.1.0

# Or from default branch
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git
```

## Install with optional extras

```bash
pip install "llamatelemetry[gpu]"          # pynvml GPU metrics
pip install "llamatelemetry[kaggle]"       # HuggingFace Hub download
pip install "llamatelemetry[graphistry]"   # Graphistry trace visualization
pip install "llamatelemetry[all]"          # all optional extras
```

## Verify

```python
import llamatelemetry
print(llamatelemetry.__version__)          # 1.1.0
```

## CUDA Binary Auto-Download

On first import in Kaggle, the SDK auto-downloads the pre-built CUDA binary (~1.4 GB):
- llama.cpp server for GGUF inference (CUDA 12, SM 7.5)
- NCCL libraries for multi-GPU coordination

The binary is cached in `/kaggle/working/.llamatelemetry/` and reused on subsequent runs.

## Torch-Optional Design

Core modules work without PyTorch. Only GPU-specific modules (cuda, unsloth, nccl, quantization) require torch. Install PyTorch separately if needed:

```bash
pip install torch
```

## Notes

- PyPI distribution is not used; install directly from GitHub.
- The CUDA binary is identical to the v1.0.0 binary (no C++/CUDA changes in v1.1.0).
- See `docs/QUICK_START_GUIDE.md` to run your first inference.
