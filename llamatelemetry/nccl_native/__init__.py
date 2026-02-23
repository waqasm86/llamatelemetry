"""
llamatelemetry.nccl_native - Direct native C++ bindings to NCCL

This module provides complete native integration with NCCL via pybind11,
for multi-GPU distributed inference without torch.distributed dependency.

Features:
  - Direct communicator initialization and management
  - All collective operations (AllReduce, AllGather, ReduceScatter, etc.)
  - Point-to-point (Send/Recv) operations
  - Custom reduction operators
  - RMA (Remote Memory Access) operations
  - Graph building with grouped operations
  - Full support for dual T4 GPUs
"""

from .communicator import NCCLCommunicator
from .collectives import (
    allreduce,
    broadcast,
    reduce,
    allgather,
    reduce_scatter,
    alltoall,
    send,
    recv,
)
from .types import DataType, ReductionOp, ResultCode

__all__ = [
    'NCCLCommunicator',
    'allreduce',
    'broadcast',
    'reduce',
    'allgather',
    'reduce_scatter',
    'alltoall',
    'send',
    'recv',
    'DataType',
    'ReductionOp',
    'ResultCode',
]

__version__ = '2.0.0'
