"""
llamatelemetry.inference.metrics - Performance primitives for CUDA inference.

Computes TTFT, TPOT, TPS, queue delay, and VRAM peaks from event timestamps.
These metrics exist independently of OTel — the inference system stands alone.
"""

from __future__ import annotations

from typing import List, Optional

from .events import InferenceEvents


def compute_ttft(events: InferenceEvents) -> float:
    """Compute Time to First Token (TTFT) in milliseconds.

    TTFT = time from request start to first token emission = prefill latency.

    Args:
        events: Inference event timestamps.

    Returns:
        TTFT in milliseconds, or 0.0 if timestamps are missing.
    """
    if events.start_ts and events.first_token_ts:
        return (events.first_token_ts - events.start_ts) * 1000.0
    return 0.0


def compute_tpot(events: InferenceEvents) -> float:
    """Compute Time Per Output Token (TPOT) in milliseconds.

    TPOT = decode duration / number of output tokens.

    Args:
        events: Inference event timestamps.

    Returns:
        TPOT in milliseconds, or 0.0 if data is insufficient.
    """
    if (
        events.first_token_ts
        and events.last_token_ts
        and events.output_tokens
        and events.output_tokens > 1
    ):
        decode_duration = events.last_token_ts - events.first_token_ts
        return (decode_duration / (events.output_tokens - 1)) * 1000.0
    return 0.0


def compute_tps(tokens: int, duration_s: float) -> float:
    """Compute tokens per second.

    Args:
        tokens: Number of tokens generated.
        duration_s: Duration in seconds.

    Returns:
        Tokens per second, or 0.0 if duration is zero.
    """
    if duration_s > 0 and tokens > 0:
        return tokens / duration_s
    return 0.0


def compute_queue_delay(enqueued_ts: Optional[float], start_ts: Optional[float]) -> float:
    """Compute queue delay in milliseconds.

    Time spent waiting in the scheduler queue before processing started.

    Args:
        enqueued_ts: Timestamp when request was enqueued.
        start_ts: Timestamp when processing started.

    Returns:
        Queue delay in milliseconds, or 0.0 if timestamps missing.
    """
    if enqueued_ts and start_ts:
        return (start_ts - enqueued_ts) * 1000.0
    return 0.0


def compute_prefill_tps(events: InferenceEvents) -> float:
    """Compute prefill tokens per second.

    Args:
        events: Inference event timestamps.

    Returns:
        Prefill tokens per second, or 0.0 if data insufficient.
    """
    ttft_s = compute_ttft(events) / 1000.0
    if ttft_s > 0 and events.input_tokens and events.input_tokens > 0:
        return events.input_tokens / ttft_s
    return 0.0


def compute_all_metrics(events: InferenceEvents) -> dict:
    """Compute all inference performance metrics.

    Args:
        events: Inference event timestamps.

    Returns:
        Dictionary with all computed metrics.
    """
    ttft = compute_ttft(events)
    tpot = compute_tpot(events)
    total_duration_s = 0.0
    if events.start_ts and events.last_token_ts:
        total_duration_s = events.last_token_ts - events.start_ts

    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "tps": compute_tps(events.output_tokens or 0, total_duration_s),
        "prefill_tps": compute_prefill_tps(events),
        "queue_delay_ms": compute_queue_delay(events.enqueued_ts, events.start_ts),
        "total_latency_ms": total_duration_s * 1000.0,
        "input_tokens": events.input_tokens or 0,
        "output_tokens": events.output_tokens or 0,
    }
