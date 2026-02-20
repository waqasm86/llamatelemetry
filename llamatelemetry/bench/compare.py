"""
llamatelemetry.bench.compare - Benchmark comparison and regression detection.

Diffs two benchmark JSON reports and flags performance regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark reports.

    Attributes:
        test_name: Name of the test.
        metric: Metric being compared.
        baseline: Baseline value.
        current: Current value.
        delta: Absolute change.
        delta_pct: Percentage change.
        regression: Whether this is a regression.
    """

    test_name: str = ""
    metric: str = ""
    baseline: float = 0.0
    current: float = 0.0
    delta: float = 0.0
    delta_pct: float = 0.0
    regression: bool = False


def compare_reports(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    regression_threshold_pct: float = 10.0,
) -> List[ComparisonResult]:
    """Compare two benchmark reports and detect regressions.

    Args:
        baseline: Baseline benchmark report (dict).
        current: Current benchmark report (dict).
        regression_threshold_pct: Percentage change threshold for regression alerts.

    Returns:
        List of ComparisonResult with comparisons and regression flags.

    Example:
        >>> results = compare_reports(old_report, new_report, regression_threshold_pct=5.0)
        >>> for r in results:
        ...     if r.regression:
        ...         print(f"REGRESSION: {r.test_name}/{r.metric}: {r.delta_pct:+.1f}%")
    """
    comparisons: List[ComparisonResult] = []

    baseline_results = {r["name"]: r for r in baseline.get("results", [])}
    current_results = {r["name"]: r for r in current.get("results", [])}

    # Metrics where higher is worse (latency-like)
    higher_is_worse = {"ttft_ms", "tpot_ms", "total_latency_ms", "vram_peak_mb"}
    # Metrics where lower is worse (throughput-like)
    lower_is_worse = {"tps"}

    for name in baseline_results:
        if name not in current_results:
            continue

        base = baseline_results[name]
        curr = current_results[name]

        for metric in ["ttft_ms", "tpot_ms", "tps", "total_latency_ms", "vram_peak_mb"]:
            base_val = base.get(metric, 0.0)
            curr_val = curr.get(metric, 0.0)

            if base_val == 0:
                continue

            delta = curr_val - base_val
            delta_pct = (delta / base_val) * 100.0

            # Determine regression
            regression = False
            if metric in higher_is_worse and delta_pct > regression_threshold_pct:
                regression = True
            elif metric in lower_is_worse and delta_pct < -regression_threshold_pct:
                regression = True

            comparisons.append(ComparisonResult(
                test_name=name,
                metric=metric,
                baseline=base_val,
                current=curr_val,
                delta=round(delta, 2),
                delta_pct=round(delta_pct, 2),
                regression=regression,
            ))

    return comparisons


def format_comparison(comparisons: List[ComparisonResult]) -> str:
    """Format comparison results as a readable string.

    Args:
        comparisons: List of comparison results.

    Returns:
        Formatted string.
    """
    lines = ["Benchmark Comparison", "=" * 60]

    regressions = [c for c in comparisons if c.regression]
    if regressions:
        lines.append(f"\nREGRESSIONS DETECTED: {len(regressions)}")
        for r in regressions:
            lines.append(f"  {r.test_name}/{r.metric}: {r.baseline:.2f} -> {r.current:.2f} ({r.delta_pct:+.1f}%)")

    lines.append(f"\nAll comparisons ({len(comparisons)}):")
    for c in comparisons:
        flag = " REGRESSION" if c.regression else ""
        lines.append(f"  {c.test_name}/{c.metric}: {c.baseline:.2f} -> {c.current:.2f} ({c.delta_pct:+.1f}%){flag}")

    return "\n".join(lines)
