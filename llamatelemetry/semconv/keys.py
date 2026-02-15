"""
llamatelemetry.semconv.keys - Attribute key constants.

Consolidates all scattered attribute string literals into a single module.
"""

# ---------------------------------------------------------------------------
# LLM attributes
# ---------------------------------------------------------------------------
LLM_SYSTEM = "llm.system"
LLM_MODEL = "llm.model"
LLM_REQUEST_DURATION_MS = "llm.request.duration_ms"
LLM_TOKENS_TOTAL = "llm.tokens.total"
LLM_TOKENS_PER_SECOND = "llm.tokens_per_sec"
LLM_INPUT_TOKENS = "llm.input.tokens"
LLM_OUTPUT_TOKENS = "llm.output.tokens"
LLM_PHASE = "llm.phase"
LLM_QUANT = "llm.quantization"
LLM_GGUF_SHA256 = "llm.gguf.sha256"
LLM_STREAM = "llm.stream"
LLM_FINISH_REASON = "llm.finish_reason"
LLM_ERROR = "llm.error"

# ---------------------------------------------------------------------------
# GPU attributes
# ---------------------------------------------------------------------------
GPU_ID = "gpu.id"
GPU_UTILIZATION_PCT = "gpu.utilization_pct"
GPU_MEM_USED_MB = "gpu.memory.used_mb"
GPU_MEM_TOTAL_MB = "gpu.memory.total_mb"
GPU_POWER_W = "gpu.power_w"
GPU_TEMP_C = "gpu.temperature_c"
GPU_NAME = "gpu.name"
GPU_COMPUTE_CAP = "gpu.compute_capability"
GPU_DRIVER_VERSION = "gpu.driver_version"

# ---------------------------------------------------------------------------
# NCCL attributes
# ---------------------------------------------------------------------------
NCCL_COLLECTIVE = "nccl.collective"
NCCL_BYTES = "nccl.bytes"
NCCL_WAIT_MS = "nccl.wait_ms"
NCCL_SPLIT_MODE = "nccl.split_mode"

# ---------------------------------------------------------------------------
# Service / request attributes
# ---------------------------------------------------------------------------
RUN_ID = "run.id"
REQUEST_ID = "request.id"
SESSION_ID = "session.id"
USER_ID = "user.id"
