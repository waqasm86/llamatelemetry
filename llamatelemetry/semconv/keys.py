"""
llamatelemetry.semconv.keys - Attribute key constants.

Consolidates all scattered attribute string literals into a single module.
"""

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
