"""
llamatelemetry.bench.report - Benchmark report generation and storage.

Produces JSON reports with hardware info, engine config, per-test metrics,
and aggregates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TestResult:
    """Result from a single benchmark test.

    Attributes:
        name: Test name (e.g. "short_prompt", "long_prompt").
        prompt_length: Input prompt character length.
        max_tokens: Maximum tokens generated.
        ttft_ms: Average time to first token.
        tpot_ms: Average time per output token.
        tps: Average tokens per second.
        total_latency_ms: Average total latency.
        input_tokens: Input token count.
        output_tokens: Output token count.
        vram_peak_mb: Peak VRAM usage.
        iterations: Number of iterations run.
    """

    name: str = ""
    prompt_length: int = 0
    max_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    tps: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    vram_peak_mb: float = 0.0
    iterations: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prompt_length": self.prompt_length,
            "max_tokens": self.max_tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "tpot_ms": round(self.tpot_ms, 2),
            "tps": round(self.tps, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "vram_peak_mb": round(self.vram_peak_mb, 2),
            "iterations": self.iterations,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report.

    Attributes:
        backend: Backend used.
        hardware: Hardware information.
        results: List of test results.
        timestamp: Report generation timestamp.
        llamatelemetry_version: SDK version.
    """

    backend: str = ""
    hardware: Dict[str, Any] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    timestamp: str = ""
    llamatelemetry_version: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.llamatelemetry_version:
            try:
                from .._version import __version__
                self.llamatelemetry_version = __version__
            except ImportError:
                self.llamatelemetry_version = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "backend": self.backend,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "llamatelemetry_version": self.llamatelemetry_version,
            "results": [r.to_dict() for r in self.results],
            "aggregates": self._compute_aggregates(),
        }

    def _compute_aggregates(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all tests."""
        if not self.results:
            return {}

        return {
            "avg_ttft_ms": round(sum(r.ttft_ms for r in self.results) / len(self.results), 2),
            "avg_tpot_ms": round(sum(r.tpot_ms for r in self.results) / len(self.results), 2),
            "avg_tps": round(sum(r.tps for r in self.results) / len(self.results), 2),
            "max_vram_peak_mb": round(max(r.vram_peak_mb for r in self.results), 2),
            "total_tests": len(self.results),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save report as JSON.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BenchmarkReport":
        """Load report from JSON.

        Args:
            path: Input file path.

        Returns:
            BenchmarkReport instance.
        """
        with open(path) as f:
            data = json.load(f)

        results = [TestResult(**r) for r in data.get("results", [])]
        return cls(
            backend=data.get("backend", ""),
            hardware=data.get("hardware", {}),
            results=results,
            timestamp=data.get("timestamp", ""),
            llamatelemetry_version=data.get("llamatelemetry_version", ""),
        )
