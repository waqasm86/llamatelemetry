"""
llamatelemetry.extras - Non-core modules preserved for compatibility.

These modules are kept for backward compatibility and advanced use cases
but are not part of the core API surface.

Submodules:
    - chat: ChatEngine
    - embeddings: EmbeddingEngine
    - jupyter: Jupyter widgets
    - models: ModelManager, SmartModelDownloader
    - gguf_parser: GGUFReader with mmap
    - quantization/: NF4, GGUF conversion, dynamic quant
    - unsloth/: Unsloth loader/exporter/adapter
    - cuda/: CUDA graphs, triton, tensor cores
    - inference/: Flash attn, KV cache, batch
    - louie/: Louie client
    - graphistry/: GraphWorkload, RAPIDSBackend, viz
    - api/: Original API module (LlamaCppClient, gguf, multigpu, nccl)
"""
