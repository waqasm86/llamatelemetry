"""
llamatelemetry.compat - Backward compatibility shims for legacy APIs.

Provides InferenceEngine and InferResult with deprecation warnings
pointing to the current v1.2.0 APIs.
"""

import warnings
from typing import Any, Dict, List, Optional


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated as of v1.2.0. Use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class InferResult:
    """Backward-compatible InferResult (deprecated)."""

    def __init__(self) -> None:
        _deprecated("InferResult", "llamatelemetry.llama.CompletionResponse")
        self.success: bool = False
        self.text: str = ""
        self.tokens_generated: int = 0
        self.latency_ms: float = 0.0
        self.tokens_per_sec: float = 0.0
        self.error_message: str = ""

    def __repr__(self) -> str:
        if self.success:
            return (
                f"InferResult(tokens={self.tokens_generated}, "
                f"latency={self.latency_ms:.2f}ms, "
                f"throughput={self.tokens_per_sec:.2f} tok/s)"
            )
        return f"InferResult(Error: {self.error_message})"

    def __str__(self) -> str:
        return self.text


class InferenceEngine:
    """
    Backward-compatible InferenceEngine (deprecated).

    Use ``llamatelemetry.init()`` + ``llamatelemetry.llama.LlamaCppClient``
    for the v1.2.0 API.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8090",
        enable_telemetry: bool = False,
        telemetry_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        _deprecated(
            "InferenceEngine",
            "llamatelemetry.init() + llamatelemetry.llama.LlamaCppClient",
        )
        from .utils import require_cuda

        require_cuda()
        # Delegate to the original implementation that still lives in __init__.py
        # during the transition period.
        from . import _legacy_init  # type: ignore[attr-defined]

        self._impl = _legacy_init.InferenceEngine(
            server_url=server_url,
            enable_telemetry=enable_telemetry,
            telemetry_config=telemetry_config,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    def __enter__(self) -> "InferenceEngine":
        self._impl.__enter__()
        return self

    def __exit__(self, *args: Any) -> bool:
        return self._impl.__exit__(*args)
