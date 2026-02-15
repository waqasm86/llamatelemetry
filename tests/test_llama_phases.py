"""Tests for llamatelemetry.llama.phases."""
import pytest
from llamatelemetry.llama.phases import trace_request, _RequestHandle


def test_request_handle():
    h = _RequestHandle(prompt_tokens=10, completion_tokens=20)
    assert h.prompt_tokens == 10
    assert h.completion_tokens == 20

    h.set_completion_tokens(50)
    assert h.completion_tokens == 50

    h.set_prompt_tokens(15)
    assert h.prompt_tokens == 15


def test_trace_request_basic():
    with trace_request(request_id="r1", model="test-model") as req:
        req.set_completion_tokens(42)
    assert req.completion_tokens == 42


def test_trace_request_defaults():
    with trace_request() as req:
        pass
    assert req.prompt_tokens == 0
    assert req.completion_tokens == 0


def test_trace_request_with_tokens():
    with trace_request(prompt_tokens=100, completion_tokens=200, stream=True) as req:
        pass
    assert req.prompt_tokens == 100
    assert req.completion_tokens == 200


def test_trace_request_exception_propagation():
    with pytest.raises(RuntimeError, match="inference failed"):
        with trace_request(model="test") as req:
            raise RuntimeError("inference failed")
