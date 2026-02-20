"""
llamatelemetry.inference.events - Lightweight event recorder for inference timing.

Records timestamps for: enqueue, start, first_token, last_token, complete.
These events feed into metrics.py for TTFT/TPOT/TPS computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InferenceEvents:
    """Timestamp recorder for inference request lifecycle.

    Attributes:
        enqueued_ts: When the request entered the scheduler queue.
        start_ts: When processing actually started.
        first_token_ts: When the first output token was produced.
        last_token_ts: When the last output token was produced.
        complete_ts: When the request was fully completed.
        input_tokens: Number of input tokens (set after tokenization).
        output_tokens: Number of output tokens (set after generation).
        token_timestamps: Per-token timestamps for detailed analysis.
        vram_before_mb: VRAM usage before request.
        vram_after_mb: VRAM usage after request.
        vram_peak_mb: Peak VRAM during request.
    """

    enqueued_ts: Optional[float] = None
    start_ts: Optional[float] = None
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    complete_ts: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    token_timestamps: List[float] = field(default_factory=list)
    vram_before_mb: Optional[float] = None
    vram_after_mb: Optional[float] = None
    vram_peak_mb: Optional[float] = None

    def mark_enqueued(self) -> None:
        """Record the enqueue timestamp."""
        self.enqueued_ts = time.perf_counter()

    def mark_start(self) -> None:
        """Record the processing start timestamp."""
        self.start_ts = time.perf_counter()

    def mark_first_token(self) -> None:
        """Record the first token timestamp."""
        self.first_token_ts = time.perf_counter()
        self.token_timestamps.append(self.first_token_ts)

    def mark_token(self) -> None:
        """Record a token timestamp during decode."""
        ts = time.perf_counter()
        self.token_timestamps.append(ts)
        self.last_token_ts = ts

    def mark_last_token(self) -> None:
        """Record the last token timestamp."""
        self.last_token_ts = time.perf_counter()

    def mark_complete(self) -> None:
        """Record the completion timestamp."""
        self.complete_ts = time.perf_counter()
        if self.last_token_ts is None:
            self.last_token_ts = self.complete_ts

    def set_token_counts(self, input_tokens: int, output_tokens: int) -> None:
        """Set token counts after generation."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def set_vram(
        self,
        before_mb: Optional[float] = None,
        after_mb: Optional[float] = None,
        peak_mb: Optional[float] = None,
    ) -> None:
        """Set VRAM measurements."""
        if before_mb is not None:
            self.vram_before_mb = before_mb
        if after_mb is not None:
            self.vram_after_mb = after_mb
        if peak_mb is not None:
            self.vram_peak_mb = peak_mb

    @property
    def total_duration_s(self) -> float:
        """Total request duration in seconds."""
        if self.start_ts and self.complete_ts:
            return self.complete_ts - self.start_ts
        if self.start_ts and self.last_token_ts:
            return self.last_token_ts - self.start_ts
        return 0.0


class EventRecorder:
    """Factory for creating and managing InferenceEvents."""

    @staticmethod
    def new() -> InferenceEvents:
        """Create a new event recorder."""
        return InferenceEvents()

    @staticmethod
    def from_timestamps(
        start: float,
        first_token: float,
        last_token: float,
        input_tokens: int,
        output_tokens: int,
    ) -> InferenceEvents:
        """Create events from pre-recorded timestamps.

        Useful for backends that provide timing info in their responses.
        """
        return InferenceEvents(
            start_ts=start,
            first_token_ts=first_token,
            last_token_ts=last_token,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
