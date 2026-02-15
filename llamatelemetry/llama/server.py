"""
llamatelemetry.llama.server - ServerManager + instrument_server() + quick_start().

Re-exports ServerManager from the root server.py module and adds
OTel-aware helpers that absorb common notebook boilerplate.
"""

from typing import Any, Dict, Optional

# Re-export the original ServerManager
from ..server import ServerManager


# ---------------------------------------------------------------------------
# OTel instrumentation for a running server
# ---------------------------------------------------------------------------

_instrumented_servers: Dict[str, Dict[str, Any]] = {}


def instrument_server(
    base_url: str = "http://127.0.0.1:8090",
    model_name: str = "",
    gguf_sha256: str = "",
    tensor_split: Optional[str] = None,
) -> None:
    """
    Register a running llama-server for monitoring.

    This stores server metadata so that spans and metrics created
    by ``trace_request()`` carry the correct attributes.

    Args:
        base_url: Server URL.
        model_name: Model name for span attributes.
        gguf_sha256: GGUF file SHA-256.
        tensor_split: Tensor split configuration string.
    """
    _instrumented_servers[base_url] = {
        "model_name": model_name,
        "gguf_sha256": gguf_sha256,
        "tensor_split": tensor_split,
    }


def get_server_meta(base_url: str = "http://127.0.0.1:8090") -> Dict[str, Any]:
    """Return metadata for an instrumented server."""
    return _instrumented_servers.get(base_url, {})


def quick_start(
    model_path: str,
    preset: str = "kaggle_t4_dual",
    port: int = 8090,
    **kwargs: Any,
) -> ServerManager:
    """
    One-liner server startup absorbing common notebook boilerplate.

    Args:
        model_path: Path to GGUF model.
        preset: Preset name ("kaggle_t4_dual", "kaggle_t4_single", "local").
        port: Server port.
        **kwargs: Forwarded to ``ServerManager.start_server()``.

    Returns:
        Running ServerManager instance.
    """
    presets = {
        "kaggle_t4_dual": {
            "gpu_layers": 99,
            "ctx_size": 8192,
            "batch_size": 2048,
            "ubatch_size": 512,
            "flash_attn": True,
            "n_parallel": 1,
        },
        "kaggle_t4_single": {
            "gpu_layers": 99,
            "ctx_size": 4096,
            "batch_size": 512,
            "ubatch_size": 128,
            "n_parallel": 1,
        },
        "local": {
            "gpu_layers": 99,
            "ctx_size": 2048,
            "batch_size": 512,
            "ubatch_size": 128,
            "n_parallel": 1,
        },
    }

    cfg = presets.get(preset, presets["local"])
    cfg.update(kwargs)

    mgr = ServerManager(server_url=f"http://127.0.0.1:{port}")
    mgr.start_server(model_path=model_path, port=port, **cfg)

    return mgr
