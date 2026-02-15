"""Tests for llamatelemetry.gpu.schemas."""
import time
from llamatelemetry.gpu.schemas import GPUDevice, GPUSnapshot, GPUSamplerHandle


def test_gpu_device():
    dev = GPUDevice(
        id=0, name="Tesla T4", memory_total_mb=15360,
        compute_capability="7.5", driver_version="535.129.03",
    )
    assert dev.id == 0
    assert dev.name == "Tesla T4"
    assert dev.memory_total_mb == 15360
    assert dev.compute_capability == "7.5"
    assert dev.driver_version == "535.129.03"


def test_gpu_snapshot():
    snap = GPUSnapshot(
        gpu_id=0, timestamp=time.time(),
        utilization_pct=75, mem_used_mb=8192, mem_total_mb=15360,
        power_w=65.5, temp_c=72,
    )
    assert snap.gpu_id == 0
    assert snap.utilization_pct == 75
    assert snap.mem_used_mb == 8192
    assert snap.power_w == 65.5
    assert snap.temp_c == 72


def test_gpu_snapshot_defaults():
    snap = GPUSnapshot(gpu_id=1, timestamp=0.0)
    assert snap.utilization_pct == 0
    assert snap.mem_used_mb == 0
    assert snap.power_w == 0.0
    assert snap.temp_c == 0


def test_sampler_handle_lifecycle():
    handle = GPUSamplerHandle(interval_ms=100)
    assert handle.get_latest() is None
    assert handle.get_snapshots() == []

    # Test with a mock query function
    call_count = 0
    def mock_query():
        nonlocal call_count
        call_count += 1
        return [GPUSnapshot(gpu_id=0, timestamp=time.time(), utilization_pct=50)]

    handle.start(mock_query)
    time.sleep(0.35)
    handle.stop()

    assert call_count >= 2
    snaps = handle.get_snapshots()
    assert len(snaps) >= 2
    latest = handle.get_latest()
    assert latest is not None
    assert latest.utilization_pct == 50
