"""Tests for llamatelemetry._internal.decorators."""
import pytest
from llamatelemetry._internal.decorators import (
    trace_decorator,
    workflow_decorator,
    task_decorator,
    tool_decorator,
    span_context,
    session_context,
    suppress_tracing,
)


def test_trace_decorator_sync():
    @trace_decorator(name="test.op")
    def my_func(x):
        return x + 1
    assert my_func(1) == 2
    assert my_func.__name__ == "my_func"


def test_workflow_decorator_sync():
    @workflow_decorator(name="test.workflow")
    def my_workflow():
        return "done"
    assert my_workflow() == "done"


def test_task_decorator_sync():
    @task_decorator()
    def my_task():
        return 42
    assert my_task() == 42


def test_tool_decorator_sync():
    @tool_decorator(name="my.tool")
    def my_tool(a, b):
        return a + b
    assert my_tool(3, 4) == 7


def test_trace_decorator_async():
    """Test async decorator by running coroutine manually."""
    import asyncio

    @trace_decorator(name="test.async_op")
    async def async_func(x):
        return x * 2

    result = asyncio.get_event_loop().run_until_complete(async_func(5))
    assert result == 10


def test_decorator_preserves_exception():
    @trace_decorator()
    def bad_func():
        raise ValueError("test error")
    with pytest.raises(ValueError, match="test error"):
        bad_func()


def test_span_context():
    with span_context("test.span", foo="bar") as s:
        pass  # NoopSpan - just verify it doesn't crash


def test_session_context():
    with session_context("s-123", user_id="u-456") as s:
        pass


def test_suppress_tracing():
    with suppress_tracing():
        pass
