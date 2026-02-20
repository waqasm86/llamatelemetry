"""
llamatelemetry._internal.decorators - Decorator implementations.

Provides @trace, @workflow, @task, @tool decorators plus span/session
context managers.
"""

import asyncio
import functools
from contextlib import contextmanager
from typing import Any, Callable, Optional

from ..otel.provider import get_tracer
from ..semconv import keys


# ---------------------------------------------------------------------------
# Decorator factory
# ---------------------------------------------------------------------------

def _make_decorator(span_kind: str, name: Optional[str] = None, **extra_attrs: Any):
    """
    Factory that creates a sync+async-aware decorator.

    Args:
        span_kind: Semantic label (e.g. "workflow", "task", "tool").
        name: Explicit span name; defaults to the function's qualified name.
        **extra_attrs: Additional attributes attached to every span.
    """

    def decorator(fn: Callable) -> Callable:
        span_name = name or f"{span_kind}.{fn.__qualname__}"

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer("llamatelemetry")
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("llamatelemetry.span_kind", span_kind)
                    for k, v in extra_attrs.items():
                        span.set_attribute(k, v)
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_attribute("error.type", exc.__class__.__name__)
                        raise

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer("llamatelemetry")
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("llamatelemetry.span_kind", span_kind)
                    for k, v in extra_attrs.items():
                        span.set_attribute(k, v)
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_attribute("error.type", exc.__class__.__name__)
                        raise

            return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Public decorators
# ---------------------------------------------------------------------------

def trace_decorator(name: Optional[str] = None, **attrs: Any):
    """General-purpose tracing decorator."""
    return _make_decorator("trace", name=name, **attrs)


def workflow_decorator(name: Optional[str] = None, **attrs: Any):
    """Marks a function as a workflow."""
    return _make_decorator("workflow", name=name, **attrs)


def task_decorator(name: Optional[str] = None, **attrs: Any):
    """Marks a function as a task."""
    return _make_decorator("task", name=name, **attrs)


def tool_decorator(name: Optional[str] = None, **attrs: Any):
    """Marks a function as a tool call."""
    return _make_decorator("tool", name=name, **attrs)


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def span_context(name: str, **attrs: Any):
    """
    Manual span context manager.

    Usage::

        with span_context("my.operation", foo="bar") as span:
            ...
    """
    tracer = get_tracer("llamatelemetry")
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, v)
        yield span


@contextmanager
def session_context(session_id: str, user_id: Optional[str] = None):
    """
    Sets session and user attributes on all child spans.

    Usage::

        with session_context("s-123", user_id="u-456"):
            ...
    """
    tracer = get_tracer("llamatelemetry")
    with tracer.start_as_current_span("session") as span:
        span.set_attribute(keys.SESSION_ID, session_id)
        if user_id:
            span.set_attribute(keys.USER_ID, user_id)
        yield span


@contextmanager
def suppress_tracing():
    """
    Context manager to suppress tracing for the enclosed block.

    OTel's ``suppress_instrumentation`` context is used when available,
    otherwise this is a no-op.
    """
    try:
        from opentelemetry.context import attach, detach
        from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY  # type: ignore[attr-defined]
        from opentelemetry import context as ctx

        token = attach(ctx.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            yield
        finally:
            detach(token)
    except (ImportError, AttributeError):
        yield
