"""
llamatelemetry.otel.redaction - SpanProcessor for prompt/key redaction.

Strips or hashes sensitive span attributes before export.
"""

import hashlib
from typing import Any, List, Optional, Sequence

_PROMPT_KEYS = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.tool.definitions",
    }
)

try:
    from opentelemetry.sdk.trace import ReadableSpan, Span
    from opentelemetry.sdk.trace.export import SpanProcessor
    from opentelemetry.context import Context

    class RedactionSpanProcessor(SpanProcessor):
        """Redact sensitive attributes from spans before export."""

        def __init__(
            self,
            redact_prompts: bool = False,
            redact_keys: Optional[List[str]] = None,
        ):
            self._redact_prompts = redact_prompts
            self._redact_keys = set(redact_keys or [])

        def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
            pass

        def on_end(self, span: ReadableSpan) -> None:
            if not span.attributes:
                return

            mutable = dict(span.attributes)
            changed = False

            if self._redact_prompts:
                for key in _PROMPT_KEYS:
                    if key in mutable:
                        mutable[key] = "[REDACTED]"
                        changed = True

            for key in self._redact_keys:
                if key in mutable:
                    val = mutable[key]
                    if isinstance(val, str):
                        mutable[key] = hashlib.sha256(val.encode()).hexdigest()[:16]
                    else:
                        mutable[key] = "[REDACTED]"
                    changed = True

            if changed:
                # ReadableSpan.attributes is a MappingProxy; we update the
                # underlying _attributes dict when possible.
                if hasattr(span, "_attributes"):
                    object.__setattr__(span, "_attributes", mutable)

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 0) -> bool:
            return True

except ImportError:

    class RedactionSpanProcessor:  # type: ignore[no-redef]
        """Stub when OTel SDK is not installed."""

        def __init__(self, **kwargs: Any):
            pass
