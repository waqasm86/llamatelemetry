"""Tests for llamatelemetry top-level API."""
import importlib
import sys
import pytest


def test_version():
    import llamatelemetry
    assert llamatelemetry.version() == "1.0.0"
    assert llamatelemetry.__version__ == "1.0.0"


def test_public_api_exists():
    import llamatelemetry
    # Lifecycle
    assert callable(llamatelemetry.init)
    assert callable(llamatelemetry.flush)
    assert callable(llamatelemetry.shutdown)
    assert callable(llamatelemetry.configure)
    assert callable(llamatelemetry.version)
    # Decorators
    assert callable(llamatelemetry.trace)
    assert callable(llamatelemetry.workflow)
    assert callable(llamatelemetry.task)
    assert callable(llamatelemetry.tool)
    # Context managers
    assert callable(llamatelemetry.span)
    assert callable(llamatelemetry.session)
    assert callable(llamatelemetry.suppress_tracing)


def test_submodules_accessible():
    import llamatelemetry
    assert hasattr(llamatelemetry, "llama")
    assert hasattr(llamatelemetry, "gpu")
    assert hasattr(llamatelemetry, "nccl")
    assert hasattr(llamatelemetry, "semconv")
    assert hasattr(llamatelemetry, "artifacts")
    assert hasattr(llamatelemetry, "kaggle")
    assert hasattr(llamatelemetry, "otel")


def test_backward_compat_exports():
    import llamatelemetry
    assert hasattr(llamatelemetry, "InferenceEngine")
    assert hasattr(llamatelemetry, "InferResult")
    assert hasattr(llamatelemetry, "ServerManager")


def test_all_list():
    import llamatelemetry
    expected = {
        "init", "flush", "shutdown", "configure", "version",
        "trace", "workflow", "task", "tool",
        "span", "session", "suppress_tracing",
        "llama", "gpu", "nccl", "semconv", "artifacts", "kaggle", "otel",
        "InferenceEngine", "InferResult", "ServerManager",
    }
    assert expected.issubset(set(llamatelemetry.__all__))
