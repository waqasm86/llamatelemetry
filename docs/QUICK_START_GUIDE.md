# Quick Start (Kaggle Dual T4)

This guide gets you running with llamatelemetry v1.0.0 in minutes on Kaggle dual T4.

## Requirements
- Kaggle notebook
- GPU T4 x2
- Internet enabled

## Install
```python
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v1.0.0
```

## Initialize the SDK
```python
import llamatelemetry

llamatelemetry.init(
    service_name="kaggle-llm",
    otlp_endpoint="https://otlp.example.com/v1/traces",  # optional
)
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
# One-liner with preset
llamatelemetry.llama.quick_start(model_path=model_path, preset="kaggle_t4_dual")

# Or manual control
from llamatelemetry.llama import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
)
```

## Run Traced Inference
```python
from llamatelemetry.llama import LlamaCppClient, trace_request

client = LlamaCppClient("http://127.0.0.1:8090")

with trace_request(model="gemma-3-1b", request_id="r1") as req:
    resp = client.chat.create(
        messages=[{"role": "user", "content": "What is CUDA?"}],
        max_tokens=80,
    )
    req.set_completion_tokens(resp.usage.completion_tokens)

print(resp.choices[0].message["content"])
```

## Use Decorators
```python
@llamatelemetry.trace()
def ask(question: str) -> str:
    resp = client.chat.create(
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    return resp.choices[0].message["content"]

answer = ask("Explain GPU tensor cores.")
```

## Shutdown
```python
llamatelemetry.shutdown()
```

## Next
- `docs/NOTEBOOKS_GUIDE.md`
- `docs/INTEGRATION_GUIDE.md`
