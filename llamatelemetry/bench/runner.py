"""
llamatelemetry.bench.runner - Benchmark runner for inference engines.

Runs a suite of prompts against any engine and collects performance metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .profiles import BenchmarkProfile, get_default_profiles
from .report import BenchmarkReport, TestResult


class BenchmarkRunner:
    """Runs benchmark suites against inference engines.

    Example:
        >>> runner = BenchmarkRunner(backend="llama.cpp", server_url="http://127.0.0.1:8090")
        >>> report = runner.run_suite()
        >>> report.save("benchmark_results.json")
    """

    def __init__(
        self,
        backend: str = "llama.cpp",
        server_url: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        profiles: Optional[List[BenchmarkProfile]] = None,
        num_iterations: int = 3,
        warmup_iterations: int = 1,
    ):
        """Initialize benchmark runner.

        Args:
            backend: Backend type ("llama.cpp" or "transformers").
            server_url: llama.cpp server URL (for llama.cpp backend).
            model: Pre-loaded model (for transformers backend).
            tokenizer: Pre-loaded tokenizer (for transformers backend).
            profiles: Benchmark profiles to run. Uses defaults if None.
            num_iterations: Number of iterations per test.
            warmup_iterations: Number of warmup iterations.
        """
        self._backend = backend
        self._server_url = server_url
        self._model = model
        self._tokenizer = tokenizer
        self._profiles = profiles or get_default_profiles()
        self._num_iterations = num_iterations
        self._warmup_iterations = warmup_iterations

    def run_suite(self) -> Dict[str, Any]:
        """Run the full benchmark suite.

        Returns:
            Dictionary with benchmark results.
        """
        engine = self._create_engine()
        engine.warmup()

        results = []
        hardware_info = self._collect_hardware_info()

        for profile in self._profiles:
            result = self._run_profile(engine, profile)
            results.append(result)

        engine.shutdown()

        report = BenchmarkReport(
            backend=self._backend,
            hardware=hardware_info,
            results=results,
        )
        return report.to_dict()

    def run_single(self, prompt: str, max_tokens: int = 128) -> TestResult:
        """Run a single benchmark prompt.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            TestResult with metrics.
        """
        engine = self._create_engine()
        engine.warmup()

        from ..inference.base import InferenceRequest

        request = InferenceRequest(prompt=prompt, max_tokens=max_tokens)
        result = engine.generate(request)

        engine.shutdown()

        return TestResult(
            name="single",
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            tps=result.tps,
            total_latency_ms=result.total_latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            vram_peak_mb=result.vram_peak_mb,
        )

    def _create_engine(self) -> Any:
        """Create the inference engine."""
        from ..inference.config import CudaInferenceConfig

        if self._backend == "llama.cpp":
            from ..inference.engines.llamacpp_engine import LlamaCppEngine

            return LlamaCppEngine(
                server_url=self._server_url or "http://127.0.0.1:8090"
            )
        elif self._backend == "transformers":
            from ..inference.engines.torch_engine import TorchEngine

            return TorchEngine(model=self._model, tokenizer=self._tokenizer)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _run_profile(self, engine: Any, profile: BenchmarkProfile) -> TestResult:
        """Run a single benchmark profile."""
        from ..inference.base import InferenceRequest

        # Warmup
        for _ in range(self._warmup_iterations):
            req = InferenceRequest(prompt=profile.prompt, max_tokens=profile.max_tokens)
            engine.generate(req)

        # Benchmark iterations
        ttft_values = []
        tpot_values = []
        tps_values = []
        latency_values = []

        for _ in range(self._num_iterations):
            req = InferenceRequest(prompt=profile.prompt, max_tokens=profile.max_tokens)
            result = engine.generate(req)
            ttft_values.append(result.ttft_ms)
            tpot_values.append(result.tpot_ms)
            tps_values.append(result.tps)
            latency_values.append(result.total_latency_ms)

        # Compute averages
        n = len(ttft_values)
        return TestResult(
            name=profile.name,
            prompt_length=len(profile.prompt),
            max_tokens=profile.max_tokens,
            ttft_ms=sum(ttft_values) / n if n else 0,
            tpot_ms=sum(tpot_values) / n if n else 0,
            tps=sum(tps_values) / n if n else 0,
            total_latency_ms=sum(latency_values) / n if n else 0,
            input_tokens=0,
            output_tokens=profile.max_tokens,
            vram_peak_mb=0.0,
            iterations=n,
        )

    @staticmethod
    def _collect_hardware_info() -> Dict[str, Any]:
        """Collect hardware information for the report."""
        info: Dict[str, Any] = {}

        try:
            from ..gpu.nvml import list_devices
            devices = list_devices()
            if devices:
                info["gpu_count"] = len(devices)
                info["gpus"] = [
                    {
                        "name": d.name,
                        "memory_total_mb": d.memory_total_mb,
                        "compute_capability": d.compute_capability,
                    }
                    for d in devices
                ]
        except Exception:
            pass

        return info
