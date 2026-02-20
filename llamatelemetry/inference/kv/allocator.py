"""
llamatelemetry.inference.kv.allocator - KV cache block allocator.

Manages GPU memory blocks for KV cache with fragmentation tracking.
Designed for Transformers engine; llama.cpp manages its own KV cache internally.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CacheBlock:
    """A single KV cache block.

    Attributes:
        block_id: Unique block identifier.
        size_bytes: Block size in bytes.
        session_id: Session that owns this block (None if free).
        seq_start: Start sequence position covered by this block.
        seq_end: End sequence position covered by this block.
        last_accessed: Timestamp of last access.
    """

    block_id: int
    size_bytes: int
    session_id: Optional[str] = None
    seq_start: int = 0
    seq_end: int = 0
    last_accessed: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.session_id is None


@dataclass
class AllocatorStats:
    """Block allocator statistics.

    Attributes:
        total_blocks: Total number of blocks.
        free_blocks: Number of free blocks.
        used_blocks: Number of allocated blocks.
        total_bytes: Total memory managed.
        used_bytes: Memory in use.
        fragmentation_ratio: Fragmentation metric (0.0 = none, 1.0 = fully fragmented).
    """

    total_blocks: int = 0
    free_blocks: int = 0
    used_blocks: int = 0
    total_bytes: int = 0
    used_bytes: int = 0
    fragmentation_ratio: float = 0.0


class BlockAllocator:
    """KV cache block allocator with fragmentation tracking.

    Pre-allocates a pool of fixed-size blocks and manages allocation/deallocation.

    Example:
        >>> allocator = BlockAllocator(
        ...     num_blocks=256,
        ...     block_size_bytes=4096,
        ... )
        >>> blocks = allocator.allocate(session_id="s1", num_blocks=4)
        >>> print(f"Allocated {len(blocks)} blocks")
        >>> allocator.free(session_id="s1")
    """

    def __init__(
        self,
        num_blocks: int = 256,
        block_size_bytes: int = 4096,
    ):
        """Initialize block allocator.

        Args:
            num_blocks: Total number of blocks to manage.
            block_size_bytes: Size of each block in bytes.
        """
        self._block_size = block_size_bytes
        self._blocks: List[CacheBlock] = [
            CacheBlock(block_id=i, size_bytes=block_size_bytes)
            for i in range(num_blocks)
        ]
        self._session_blocks: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

    def allocate(
        self,
        session_id: str,
        num_blocks: int,
    ) -> List[CacheBlock]:
        """Allocate blocks for a session.

        Args:
            session_id: Session identifier.
            num_blocks: Number of blocks to allocate.

        Returns:
            List of allocated CacheBlocks.

        Raises:
            MemoryError: If insufficient free blocks.
        """
        import time

        with self._lock:
            free = [b for b in self._blocks if b.is_free]
            if len(free) < num_blocks:
                raise MemoryError(
                    f"Insufficient KV cache blocks: need {num_blocks}, have {len(free)} free"
                )

            allocated = free[:num_blocks]
            now = time.perf_counter()
            block_ids = []

            for block in allocated:
                block.session_id = session_id
                block.last_accessed = now
                block_ids.append(block.block_id)

            self._session_blocks.setdefault(session_id, []).extend(block_ids)
            return allocated

    def free(self, session_id: str) -> int:
        """Free all blocks owned by a session.

        Args:
            session_id: Session to free.

        Returns:
            Number of blocks freed.
        """
        with self._lock:
            block_ids = self._session_blocks.pop(session_id, [])
            for bid in block_ids:
                self._blocks[bid].session_id = None
                self._blocks[bid].seq_start = 0
                self._blocks[bid].seq_end = 0
            return len(block_ids)

    def stats(self) -> AllocatorStats:
        """Get current allocator statistics."""
        with self._lock:
            free = sum(1 for b in self._blocks if b.is_free)
            used = len(self._blocks) - free

            # Simple fragmentation: ratio of non-contiguous free blocks
            frag = 0.0
            if free > 0 and used > 0:
                transitions = 0
                for i in range(1, len(self._blocks)):
                    if self._blocks[i].is_free != self._blocks[i - 1].is_free:
                        transitions += 1
                frag = min(1.0, transitions / max(1, len(self._blocks) // 2))

            return AllocatorStats(
                total_blocks=len(self._blocks),
                free_blocks=free,
                used_blocks=used,
                total_bytes=len(self._blocks) * self._block_size,
                used_bytes=used * self._block_size,
                fragmentation_ratio=frag,
            )
