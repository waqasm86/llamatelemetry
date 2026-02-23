"""
llamatelemetry.nccl_native.types - NCCL type definitions and enums
"""

from enum import IntEnum, Enum


class DataType(IntEnum):
    """NCCL data types for collective operations."""
    INT8 = 0
    UINT8 = 1
    INT32 = 2
    UINT32 = 3
    INT64 = 4
    UINT64 = 5
    FLOAT16 = 6  # half / fp16
    FLOAT32 = 7  # float
    FLOAT64 = 8  # double
    BFLOAT16 = 9  # bfloat16
    FLOAT8_E4M3 = 10  # E4M3 format (8-bit FP)
    FLOAT8_E5M2 = 11  # E5M2 format (8-bit FP)


class ReductionOp(IntEnum):
    """NCCL reduction operations."""
    SUM = 0
    PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4
    MAXN = 5
    MINMULT = 6
    MAXMULT = 7


class ResultCode(IntEnum):
    """NCCL result codes (error codes)."""
    SUCCESS = 0
    UNHANDLED_CUDA_ERROR = 1
    SYSTEM_ERROR = 2
    INTERNAL_ERROR = 3
    INVALID_ARGUMENT = 4
    INVALID_USAGE = 5
    REMOTE_ERROR = 6
    IN_PROGRESS = 7


class SplitMode(Enum):
    """Communicator split modes."""
    NOCOLOR = -1


class Algorithm(IntEnum):
    """Collective algorithms."""
    RING = 0  # 2N-2 steps
    TREE = 1  # log N steps
    DIRECT_WRITE = 2
    NVLS = 3  # NVLink Switching (H100+ only)
    COLLNET = 4


class Protocol(IntEnum):
    """Reduction protocols."""
    LL = 0  # Low-Latency (small messages)
    LL128 = 1  # Balanced (default)
    SIMPLE = 2  # High-throughput (large messages)


def result_code_to_string(code: ResultCode) -> str:
    """Convert result code to error message."""
    messages = {
        ResultCode.SUCCESS: "Success",
        ResultCode.UNHANDLED_CUDA_ERROR: "Unhandled CUDA error",
        ResultCode.SYSTEM_ERROR: "System error",
        ResultCode.INTERNAL_ERROR: "Internal error",
        ResultCode.INVALID_ARGUMENT: "Invalid argument",
        ResultCode.INVALID_USAGE: "Invalid usage",
        ResultCode.REMOTE_ERROR: "Remote error",
        ResultCode.IN_PROGRESS: "Operation in progress",
    }
    return messages.get(code, f"Unknown error ({code})")


class NCCLException(Exception):
    """Base exception for NCCL errors."""
    pass


class NCCLError(NCCLException):
    """NCCL operation failed."""

    def __init__(self, code: ResultCode, message: str = None):
        self.code = code
        self.message = message or result_code_to_string(code)
        super().__init__(self.message)
