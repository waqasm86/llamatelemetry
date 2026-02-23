"""
llamatelemetry.nccl_native.collectives - High-level collective operation APIs

Wrapper functions for common NCCL collective operations.
Works with NCCLCommunicator instances.
"""

from typing import Optional
from .types import DataType, ReductionOp


def allreduce(
    comm,
    sendbuff,
    recvbuff,
    count: int,
    datatype: DataType,
    op: ReductionOp,
    stream=None,
) -> None:
    """
    All-to-all reduction operation.

    All ranks send data, all ranks receive the reduction result.

    Args:
        comm: NCCLCommunicator instance
        sendbuff: Send buffer
        recvbuff: Receive buffer
        count: Element count
        datatype: DataType enum
        op: ReductionOp enum
        stream: CUDA stream
    """
    comm.allreduce(sendbuff, recvbuff, count, datatype, op, stream)


def broadcast(
    comm,
    buff,
    count: int,
    datatype: DataType,
    root: int,
    stream=None,
) -> None:
    """
    One-to-all broadcast.

    Root rank sends to all others.

    Args:
        comm: NCCLCommunicator instance
        buff: Buffer (root: send data, others: receive location)
        count: Element count
        datatype: DataType enum
        root: Root rank
        stream: CUDA stream
    """
    comm.broadcast(buff, count, datatype, root, stream)


def reduce(
    comm,
    sendbuff,
    recvbuff,
    count: int,
    datatype: DataType,
    op: ReductionOp,
    root: int,
    stream=None,
) -> None:
    """
    All-to-one reduction.

    All ranks send, root rank receives reduction result.

    Args:
        comm: NCCLCommunicator instance
        sendbuff: Send buffer (all ranks)
        recvbuff: Receive buffer (only used by root)
        count: Element count
        datatype: DataType enum
        op: ReductionOp enum
        root: Root rank
        stream: CUDA stream
    """
    comm.reduce(sendbuff, recvbuff, count, datatype, op, root, stream)


def allgather(
    comm,
    sendbuff,
    recvbuff,
    sendcount: int,
    datatype: DataType,
    stream=None,
) -> None:
    """
    All-to-all gather.

    Each rank receives data from all other ranks.

    Each rank sends sendcount elements, receives sendcount*nranks elements.

    Args:
        comm: NCCLCommunicator instance
        sendbuff: Send buffer (sendcount elements)
        recvbuff: Receive buffer (sendcount*nranks elements)
        sendcount: Elements to send from this rank
        datatype: DataType enum
        stream: CUDA stream
    """
    comm.allgather(sendbuff, recvbuff, sendcount, datatype, stream)


def reduce_scatter(
    comm,
    sendbuff,
    recvbuff,
    recvcount: int,
    datatype: DataType,
    op: ReductionOp,
    stream=None,
) -> None:
    """
    Reduce then scatter.

    First reduce all data, then scatter the result.

    Each rank sends recvcount*nranks elements, receives recvcount elements.

    Args:
        comm: NCCLCommunicator instance
        sendbuff: Send buffer (recvcount*nranks elements)
        recvbuff: Receive buffer (recvcount elements)
        recvcount: Elements to receive
        datatype: DataType enum
        op: ReductionOp enum
        stream: CUDA stream
    """
    comm.reduce_scatter(sendbuff, recvbuff, recvcount, datatype, op, stream)


def alltoall(
    comm,
    sendbuff,
    recvbuff,
    count: int,
    datatype: DataType,
    stream=None,
) -> None:
    """
    All-to-all permutation.

    Each rank sends to and receives from all other ranks.

    Args:
        comm: NCCLCommunicator instance
        sendbuff: Send buffer
        recvbuff: Receive buffer
        count: Elements per rank
        datatype: DataType enum
        stream: CUDA stream
    """
    comm.alltoall(sendbuff, recvbuff, count, datatype, stream)


def send(
    comm,
    sendbuff,
    count: int,
    datatype: DataType,
    peer: int,
    stream=None,
) -> None:
    """
    Send to peer rank (point-to-point).

    Args:
        comm: NCCLCommunicator instance
        sendbuff: Send buffer
        count: Element count
        datatype: DataType enum
        peer: Destination rank
        stream: CUDA stream
    """
    comm.send(sendbuff, count, datatype, peer, stream)


def recv(
    comm,
    recvbuff,
    count: int,
    datatype: DataType,
    peer: int,
    stream=None,
) -> None:
    """
    Receive from peer rank (point-to-point).

    Args:
        comm: NCCLCommunicator instance
        recvbuff: Receive buffer
        count: Element count
        datatype: DataType enum
        peer: Source rank
        stream: CUDA stream
    """
    comm.recv(recvbuff, count, datatype, peer, stream)
