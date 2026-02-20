"""
llamatelemetry.inference.runtime - Inference runtime orchestrator.

Orchestrates the flow: submit requests -> scheduler -> engine -> results.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

from .base import InferenceEngine, InferenceRequest, InferenceResult
from .scheduler import Scheduler, SchedulingPolicy
from .types import BatchConstraints
from .config import CudaInferenceConfig


class InferenceRuntime:
    """Orchestrates inference with scheduling and batching.

    Connects the scheduler to the engine and manages the request lifecycle.

    Example:
        >>> from llamatelemetry.inference.engines.llamacpp_engine import LlamaCppEngine
        >>> engine = LlamaCppEngine(server_url="http://127.0.0.1:8090")
        >>> runtime = InferenceRuntime(engine=engine, enable_scheduler=True)
        >>> runtime.start()
        >>> result = runtime.generate(InferenceRequest(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... ))
        >>> runtime.stop()
    """

    def __init__(
        self,
        engine: InferenceEngine,
        enable_scheduler: bool = False,
        batch_constraints: Optional[BatchConstraints] = None,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.FIFO,
    ):
        """Initialize inference runtime.

        Args:
            engine: The inference engine to use.
            enable_scheduler: Whether to enable request batching/scheduling.
            batch_constraints: Batching constraints for the scheduler.
            scheduling_policy: Scheduling policy.
        """
        self._engine = engine
        self._enable_scheduler = enable_scheduler
        self._scheduler = None
        self._running = False

        if enable_scheduler:
            self._scheduler = Scheduler(
                constraints=batch_constraints,
                policy=scheduling_policy,
            )

    @classmethod
    def from_config(cls, config: CudaInferenceConfig) -> "InferenceRuntime":
        """Create runtime from configuration.

        Args:
            config: CUDA inference configuration.

        Returns:
            Configured InferenceRuntime.
        """
        if config.backend == "llama.cpp":
            from .engines.llamacpp_engine import LlamaCppEngine
            engine = LlamaCppEngine.from_config(config)
        elif config.backend == "transformers":
            from .engines.torch_engine import TorchEngine
            engine = TorchEngine.from_config(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

        return cls(
            engine=engine,
            enable_scheduler=config.scheduler,
            batch_constraints=config.batch,
        )

    def start(self) -> None:
        """Start the runtime (warm up engine)."""
        self._engine.warmup()
        self._running = True

    def stop(self) -> None:
        """Stop the runtime and release resources."""
        self._running = False
        self._engine.shutdown()

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Execute an inference request.

        If scheduler is enabled, the request goes through the batch queue.
        Otherwise, it's executed directly.

        Args:
            request: Inference request.

        Returns:
            Inference result with performance metrics.
        """
        if not self._running:
            self.start()

        if self._enable_scheduler and self._scheduler:
            return self._generate_scheduled(request)
        else:
            return self._engine.generate(request)

    def generate_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Execute a batch of inference requests.

        Args:
            requests: List of inference requests.

        Returns:
            List of inference results.
        """
        if not self._running:
            self.start()

        results = []
        for req in requests:
            results.append(self.generate(req))
        return results

    def _generate_scheduled(self, request: InferenceRequest) -> InferenceResult:
        """Execute a request through the scheduler.

        For simplicity, this is synchronous: submit, poll, execute.
        A production implementation would use async/threading.
        """
        result_holder: List[InferenceResult] = []

        def on_complete(result: InferenceResult) -> None:
            result_holder.append(result)

        self._scheduler.submit(request, callback=on_complete)

        # Poll and execute
        batch = self._scheduler.poll()
        if batch:
            for scheduled_req in batch.requests:
                res = self._engine.generate(scheduled_req.request)
                if scheduled_req.callback:
                    scheduled_req.callback(res)

        return result_holder[0] if result_holder else self._engine.generate(request)

    @property
    def engine(self) -> InferenceEngine:
        """Access the underlying engine."""
        return self._engine

    @property
    def scheduler(self) -> Optional[Scheduler]:
        """Access the scheduler (if enabled)."""
        return self._scheduler
