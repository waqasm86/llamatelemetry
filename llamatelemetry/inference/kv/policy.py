"""
llamatelemetry.inference.kv.policy - KV cache eviction and reuse policies.

Provides LRU, FIFO, and session-pinning eviction strategies for the KV cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Set


class CachePolicy(ABC):
    """Abstract base class for KV cache eviction policies."""

    @abstractmethod
    def on_access(self, session_id: str) -> None:
        """Record a cache access for a session."""
        ...

    @abstractmethod
    def on_allocate(self, session_id: str, num_blocks: int) -> None:
        """Record block allocation for a session."""
        ...

    @abstractmethod
    def on_free(self, session_id: str) -> None:
        """Record block deallocation for a session."""
        ...

    @abstractmethod
    def select_victim(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """Select a session to evict.

        Args:
            exclude: Sessions that should not be evicted.

        Returns:
            Session ID to evict, or None if no victim available.
        """
        ...


class LRUPolicy(CachePolicy):
    """Least Recently Used eviction policy.

    Evicts the session that was accessed least recently.

    Example:
        >>> policy = LRUPolicy()
        >>> policy.on_allocate("s1", 4)
        >>> policy.on_allocate("s2", 4)
        >>> policy.on_access("s1")
        >>> victim = policy.select_victim()  # Returns "s2"
    """

    def __init__(self):
        self._order: OrderedDict[str, int] = OrderedDict()

    def on_access(self, session_id: str) -> None:
        if session_id in self._order:
            self._order.move_to_end(session_id)

    def on_allocate(self, session_id: str, num_blocks: int) -> None:
        self._order[session_id] = num_blocks
        self._order.move_to_end(session_id)

    def on_free(self, session_id: str) -> None:
        self._order.pop(session_id, None)

    def select_victim(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        exclude = exclude or set()
        for session_id in self._order:
            if session_id not in exclude:
                return session_id
        return None


class FIFOPolicy(CachePolicy):
    """First In, First Out eviction policy.

    Evicts the oldest session regardless of access pattern.
    """

    def __init__(self):
        self._order: List[str] = []
        self._sessions: Set[str] = set()

    def on_access(self, session_id: str) -> None:
        pass  # FIFO ignores access pattern

    def on_allocate(self, session_id: str, num_blocks: int) -> None:
        if session_id not in self._sessions:
            self._order.append(session_id)
            self._sessions.add(session_id)

    def on_free(self, session_id: str) -> None:
        self._sessions.discard(session_id)
        if session_id in self._order:
            self._order.remove(session_id)

    def select_victim(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        exclude = exclude or set()
        for session_id in self._order:
            if session_id not in exclude:
                return session_id
        return None


class SessionPinPolicy(CachePolicy):
    """Session pinning policy with LRU fallback.

    Pinned sessions are never evicted. Unpinned sessions use LRU eviction.
    """

    def __init__(self):
        self._pinned: Set[str] = set()
        self._lru = LRUPolicy()

    def pin(self, session_id: str) -> None:
        """Pin a session (prevent eviction)."""
        self._pinned.add(session_id)

    def unpin(self, session_id: str) -> None:
        """Unpin a session (allow eviction)."""
        self._pinned.discard(session_id)

    def on_access(self, session_id: str) -> None:
        self._lru.on_access(session_id)

    def on_allocate(self, session_id: str, num_blocks: int) -> None:
        self._lru.on_allocate(session_id, num_blocks)

    def on_free(self, session_id: str) -> None:
        self._lru.on_free(session_id)
        self._pinned.discard(session_id)

    def select_victim(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        exclude = (exclude or set()) | self._pinned
        return self._lru.select_victim(exclude=exclude)
