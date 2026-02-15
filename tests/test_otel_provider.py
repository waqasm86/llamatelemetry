"""Tests for llamatelemetry.otel.provider."""
import pytest
from llamatelemetry.otel.provider import (
    _NoopTracer, _NoopSpan, _NoopSpanContext, _NoopMeter, _NoopInstrument,
    get_tracer, get_meter, _detect_platform,
)


def test_noop_tracer_start_as_current_span():
    tracer = _NoopTracer()
    ctx = tracer.start_as_current_span("test")
    with ctx as span:
        assert isinstance(span, _NoopSpan)


def test_noop_tracer_start_span():
    tracer = _NoopTracer()
    span = tracer.start_span("test")
    assert isinstance(span, _NoopSpan)


def test_noop_span_operations():
    span = _NoopSpan()
    span.set_attribute("key", "value")
    span.set_status("OK")
    span.record_exception(Exception("test"))
    span.end()
    # All should be no-ops without error


def test_noop_span_context_manager():
    span = _NoopSpan()
    with span as s:
        assert isinstance(s, _NoopSpan)


def test_noop_meter_instruments():
    meter = _NoopMeter()
    counter = meter.create_counter("test.counter")
    hist = meter.create_histogram("test.hist")
    gauge = meter.create_observable_gauge("test.gauge")
    assert isinstance(counter, _NoopInstrument)
    assert isinstance(hist, _NoopInstrument)
    assert isinstance(gauge, _NoopInstrument)


def test_noop_instrument_operations():
    inst = _NoopInstrument()
    inst.add(1)
    inst.record(2.5)
    # Should not raise


def test_get_tracer_returns_something():
    tracer = get_tracer("test")
    assert tracer is not None
    assert hasattr(tracer, "start_as_current_span")


def test_get_meter_returns_something():
    meter = get_meter("test")
    assert meter is not None


def test_detect_platform():
    platform = _detect_platform()
    assert platform in ("kaggle", "colab", "local")
