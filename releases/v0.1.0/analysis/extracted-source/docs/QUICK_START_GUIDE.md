# Quick Start (Kaggle Dual T4)

This guide gets you running in minutes on Kaggle dual T4.

## Requirements
- Kaggle notebook
- GPU T4 x2
- Internet enabled

## Install
```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Download GGUF Model
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)
```

## Start Server (GPU0)
```python
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
)
```

## Run Inference
```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient("http://127.0.0.1:8090")
resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is CUDA?"}],
    max_tokens=80,
)
print(resp.choices[0].message.content)
```

## Next
- `docs/NOTEBOOKS_GUIDE.md`
- `docs/INTEGRATION_GUIDE.md`
