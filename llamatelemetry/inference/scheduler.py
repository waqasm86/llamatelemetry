"""
llamatelemetry.inference.scheduler - Request scheduler for batching and queuing.

Implements micro-batching and continuous batching for production inference.
Supports FIFO, priority, and fair-share scheduling policies.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from .base import InferenceRequest, InferenceResult
from .events import InferenceEvents
from .types import BatchConstraints


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"


@dataclass
class ScheduledRequest:
    """A request wrapped with scheduling metadata.

    Attributes:
        request: The original inference request.
        events: Event recorder for timing.
        priority: Request priority (lower = higher priority).
        tenant_id: Tenant identifier for fair-share scheduling.
        callback: Optional completion callback.
    """

    request: InferenceRequest
    events: InferenceEvents = field(default_factory=InferenceEvents)
    priority: int = 0
    tenant_id: Optional[str] = None
    callback: Optional[Callable[[InferenceResult], None]] = None

    def __post_init__(self):
        self.events.mark_enqueued()


@dataclass
class Batch:
    """A batch of requests ready for execution.

    Attributes:
        requests: List of scheduled requests.
        total_tokens: Estimated total tokens in batch.
        created_at: Batch creation timestamp.
    """

    requests: List[ScheduledRequest] = field(default_factory=list)
    total_tokens: int = 0
    created_at: float = field(default_factory=time.perf_counter)


class Scheduler:
    """Request scheduler for inference batching.

    Manages a queue of incoming requests and produces batches according
    to configured constraints and scheduling policy.

    Example:
        >>> scheduler = Scheduler(constraints=BatchConstraints(
        ...     max_batch_size=4,
        ...     max_batch_tokens=2048,
        ...     max_wait_ms=50.0,
        ... ))
        >>> scheduler.submit(InferenceRequest(messages=[...]))
        >>> batch = scheduler.poll()
        >>> if batch:
        ...     for req in batch.requests:
        ...         result = engine.generate(req.request)
    """

    def __init__(
        self,
        constraints: Optional[BatchConstraints] = None,
        policy: SchedulingPolicy = SchedulingPolicy.FIFO,
    ):
        """Initialize scheduler.

        Args:
            constraints: Batching constraints.
            policy: Scheduling policy.
        """
        self._constraints = constraints or BatchConstraints()
        self._policy = policy
        self._queue: Deque[ScheduledRequest] = deque()
        self._lock = threading.Lock()
        self._stats = SchedulerStats()

    def submit(
        self,
        request: InferenceRequest,
        priority: int = 0,
        tenant_id: Optional[str] = None,
        callback: Optional[Callable[[InferenceResult], None]] = None,
    ) -> None:
        """Submit a request to the scheduler queue.

        Args:
            request: Inference request.
            priority: Request priority (lower = higher).
            tenant_id: Tenant ID for fair-share.
            callback: Optional result callback.
        """
        scheduled = ScheduledRequest(
            request=request,
            priority=priority,
            tenant_id=tenant_id,
            callback=callback,
        )

        with self._lock:
            if self._policy == SchedulingPolicy.PRIORITY:
                # Insert in priority order
                inserted = False
                for i, existing in enumerate(self._queue):
                    if priority < existing.priority:
                        self._queue.insert(i, scheduled)
                        inserted = True
                        break
                if not inserted:
                    self._queue.append(scheduled)
            else:
                self._queue.append(scheduled)
            self._stats.total_submitted += 1

    def poll(self, force: bool = False) -> Optional[Batch]:
        """Poll for a ready batch.

        Returns a batch if enough requests are queued or max_wait_ms has elapsed.

        Returns:
            Batch of requests, or None if no batch is ready.
        """
        with self._lock:
            if not self._queue:
                return None

            if force:
                return self._form_batch()

            # Check if we have enough requests or waited long enough
            oldest = self._queue[0]
            wait_ms = (time.perf_counter() - (oldest.events.enqueued_ts or 0)) * 1000.0

            if (
                len(self._queue) >= self._constraints.max_batch_size
                or wait_ms >= self._constraints.max_wait_ms
            ):
                return self._form_batch()

            return None

    def _form_batch(self) -> Batch:
        """Form a batch from queued requests (must hold lock)."""
        batch = Batch()
        token_estimate = 0

        while self._queue and len(batch.requests) < self._constraints.max_batch_size:
            req = self._queue[0]
            # Estimate tokens (rough: 4 chars per token)
            est_tokens = req.request.max_tokens + (
                len(req.request.prompt or "") // 4 if req.request.prompt else 100
            )

            if token_estimate + est_tokens > self._constraints.max_batch_tokens:
                # Avoid deadlock: if the oldest request alone exceeds the limit,
                # emit it as a single-item batch.
                if not batch.requests:
                    self._queue.popleft()
                    req.events.mark_start()
                    batch.requests.append(req)
                    token_estimate += est_tokens
                break

            self._queue.popleft()
            req.events.mark_start()
            batch.requests.append(req)
            token_estimate += est_tokens

        batch.total_tokens = token_estimate
        self._stats.total_batches += 1
        return batch

    def stats(self) -> "SchedulerStats":
        """Return current scheduler statistics."""
        with self._lock:
            self._stats.queue_depth = len(self._queue)
        return self._stats

    @property
    def queue_depth(self) -> int:
        """Current number of requests in queue."""
        with self._lock:
            return len(self._queue)


@dataclass
class SchedulerStats:
    """Scheduler runtime statistics."""

    total_submitted: int = 0
    total_batches: int = 0
    queue_depth: int = 0
