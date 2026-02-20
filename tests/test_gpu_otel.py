"""Tests for llamatelemetry.gpu.otel (GPUSnapshot, GPUSpanEnricher)."""

import pytest


class FakeSpan:
    """Minimal OTel-compatible span for testing."""
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value


class TestGPUSnapshot:
    """Test GPUSnapshot dataclass."""

    def test_default_values(self):
        from llamatelemetry.gpu.otel import GPUSnapshot
        snap = GPUSnapshot()
        assert snap.gpu_id == 0
        assert snap.util is None
        assert snap.mem_used_mb is None
        assert snap.mem_total_mb is None
        assert snap.power_w is None
        assert snap.temp_c is None

    def test_with_values(self):
        from llamatelemetry.gpu.otel import GPUSnapshot
        snap = GPUSnapshot(
            gpu_id=1,
            util=75.5,
            mem_used_mb=8192,
            mem_total_mb=16384,
            power_w=120.0,
            temp_c=68,
        )
        assert snap.gpu_id == 1
        assert snap.util == 75.5
        assert snap.mem_used_mb == 8192
        assert snap.mem_total_mb == 16384
        assert snap.power_w == 120.0
        assert snap.temp_c == 68

    def test_multiple_devices(self):
        from llamatelemetry.gpu.otel import GPUSnapshot
        snap0 = GPUSnapshot(gpu_id=0, util=30.0)
        snap1 = GPUSnapshot(gpu_id=1, util=60.0)
        assert snap0.gpu_id == 0
        assert snap1.gpu_id == 1


class TestGPUSpanEnricher:
    """Test GPUSpanEnricher."""

    def test_init_default_device(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher
        enricher = GPUSpanEnricher()
        assert enricher._device_index == 0

    def test_init_explicit_device(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher
        enricher = GPUSpanEnricher(device_index=1)
        assert enricher._device_index == 1

    def test_init_none_device_defaults_to_zero(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher
        enricher = GPUSpanEnricher(device_index=None)
        assert enricher._device_index == 0

    def test_snapshot_returns_gpu_snapshot(self):
        """snapshot() should always return GPUSnapshot (even without nvidia-smi)."""
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        enricher = GPUSpanEnricher(device_index=0)
        snap = enricher.snapshot()
        assert isinstance(snap, GPUSnapshot)
        assert snap.gpu_id == 0

    def test_attach_deltas_sets_gpu_id(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        from llamatelemetry.semconv import keys
        enricher = GPUSpanEnricher(device_index=0)
        before = GPUSnapshot(gpu_id=0, util=20.0, mem_used_mb=2000, power_w=80.0)
        after = GPUSnapshot(gpu_id=0, util=80.0, mem_used_mb=4000, power_w=120.0)
        span = FakeSpan()
        enricher.attach_deltas(span, before, after)
        assert span.attrs[keys.GPU_ID] == "0"

    def test_attach_deltas_computes_util_delta(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        enricher = GPUSpanEnricher(device_index=0)
        before = GPUSnapshot(gpu_id=0, util=20.0)
        after = GPUSnapshot(gpu_id=0, util=80.0)
        span = FakeSpan()
        enricher.attach_deltas(span, before, after)
        assert "gpu.utilization.delta" in span.attrs
        assert abs(span.attrs["gpu.utilization.delta"] - 60.0) < 1e-6

    def test_attach_deltas_computes_mem_delta(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        enricher = GPUSpanEnricher()
        before = GPUSnapshot(gpu_id=0, mem_used_mb=2000)
        after = GPUSnapshot(gpu_id=0, mem_used_mb=4500)
        span = FakeSpan()
        enricher.attach_deltas(span, before, after)
        assert "gpu.memory.used_mb.delta" in span.attrs
        assert span.attrs["gpu.memory.used_mb.delta"] == 2500

    def test_attach_deltas_computes_power_delta(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        enricher = GPUSpanEnricher()
        before = GPUSnapshot(gpu_id=0, power_w=80.0)
        after = GPUSnapshot(gpu_id=0, power_w=120.0)
        span = FakeSpan()
        enricher.attach_deltas(span, before, after)
        assert "gpu.power.w.delta" in span.attrs
        assert abs(span.attrs["gpu.power.w.delta"] - 40.0) < 1e-6

    def test_attach_deltas_with_none_values(self):
        """None util/mem/power should not produce delta attributes."""
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        enricher = GPUSpanEnricher()
        before = GPUSnapshot(gpu_id=0)   # all None
        after = GPUSnapshot(gpu_id=0)    # all None
        span = FakeSpan()
        enricher.attach_deltas(span, before, after)
        assert "gpu.utilization.delta" not in span.attrs
        assert "gpu.memory.used_mb.delta" not in span.attrs
        assert "gpu.power.w.delta" not in span.attrs

    def test_attach_static_sets_gpu_id(self):
        from llamatelemetry.gpu.otel import GPUSpanEnricher
        from llamatelemetry.semconv import keys
        enricher = GPUSpanEnricher(device_index=0)
        span = FakeSpan()
        enricher.attach_static(span)
        assert keys.GPU_ID in span.attrs

    def test_attach_static_with_full_snapshot(self):
        """attach_static should set all available GPU attributes."""
        from llamatelemetry.gpu.otel import GPUSpanEnricher, GPUSnapshot
        from llamatelemetry.semconv import keys

        class FullEnricher(GPUSpanEnricher):
            def snapshot(self):
                return GPUSnapshot(
                    gpu_id=0,
                    util=55.0,
                    mem_used_mb=3000,
                    mem_total_mb=16384,
                    power_w=95.0,
                    temp_c=72,
                )

        enricher = FullEnricher(device_index=0)
        span = FakeSpan()
        enricher.attach_static(span)
        assert span.attrs[keys.GPU_UTILIZATION_PCT] == 55.0
        assert span.attrs[keys.GPU_MEM_USED_MB] == 3000
        assert span.attrs[keys.GPU_MEM_TOTAL_MB] == 16384
        assert span.attrs[keys.GPU_POWER_W] == 95.0
        assert span.attrs[keys.GPU_TEMP_C] == 72
