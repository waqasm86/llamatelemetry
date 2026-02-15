"""Tests for llamatelemetry.nccl.api."""
import pytest
from llamatelemetry.nccl import api


@pytest.fixture(autouse=True)
def reset_nccl():
    original = api._enabled
    api._enabled = False
    yield
    api._enabled = original


def test_enable_toggle():
    assert api.is_enabled() is False
    api.enable(True)
    assert api.is_enabled() is True
    api.enable(False)
    assert api.is_enabled() is False


def test_enable_default_true():
    api.enable()
    assert api.is_enabled() is True


def test_annotate_when_disabled():
    # Should be a no-op when disabled
    api.annotate_collective("allreduce", nbytes=1024, wait_ms=1.5)


def test_annotate_when_enabled():
    api.enable(True)
    # Should not raise (uses NoopTracer)
    api.annotate_collective("allreduce", nbytes=1024, wait_ms=1.5, custom="val")
    api.annotate_collective("allgather")
