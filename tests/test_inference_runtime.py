"""Tests for the new llamatelemetry.inference subsystem (no torch required)."""

import pytest


class TestInferenceTypes:
    """Test inference type dataclasses."""

    def test_sampling_params_defaults(self):
        from llamatelemetry.inference.types import SamplingParams
        p = SamplingParams()
        assert p.temperature == 0.7
        assert p.top_p == 1.0
        assert p.top_k == 0
        assert p.frequency_penalty == 0.0
        assert p.presence_penalty == 0.0
        assert p.seed is None
        assert p.stop_sequences is None
        assert p.repetition_penalty == 1.0

    def test_sampling_params_custom(self):
        from llamatelemetry.inference.types import SamplingParams
        p = SamplingParams(temperature=0.5, top_p=0.9, top_k=50, seed=42)
        assert p.temperature == 0.5
        assert p.top_p == 0.9
        assert p.top_k == 50
        assert p.seed == 42

    def test_batch_constraints_defaults(self):
        from llamatelemetry.inference.types import BatchConstraints
        bc = BatchConstraints()
        assert bc.max_batch_size == 8
        assert bc.max_batch_tokens == 4096
        assert bc.max_wait_ms == 50.0
        assert bc.max_concurrent_sessions == 32

    def test_batch_constraints_custom(self):
        from llamatelemetry.inference.types import BatchConstraints
        bc = BatchConstraints(max_batch_size=4, max_batch_tokens=2048)
        assert bc.max_batch_size == 4
        assert bc.max_batch_tokens == 2048

    def test_device_config_defaults(self):
        from llamatelemetry.inference.types import DeviceConfig
        dc = DeviceConfig()
        assert dc.device_ids == [0]
        assert dc.primary_device == 0
        assert dc.dtype == "fp16"
        assert dc.attention_backend == "sdpa"
        assert dc.use_torch_compile is False
        assert dc.use_cuda_graphs is False

    def test_device_config_multi_gpu(self):
        from llamatelemetry.inference.types import DeviceConfig
        dc = DeviceConfig(device_ids=[0, 1], primary_device=0, dtype="bf16")
        assert dc.device_ids == [0, 1]
        assert dc.dtype == "bf16"

    def test_engine_stats_defaults(self):
        from llamatelemetry.inference.types import EngineStats
        stats = EngineStats()
        assert stats.total_requests == 0
        assert stats.total_tokens_generated == 0
        assert stats.avg_ttft_ms == 0.0
        assert stats.avg_tps == 0.0
        assert stats.peak_vram_mb == 0.0
        assert stats.active_sessions == 0


class TestInferenceBase:
    """Test InferenceRequest and InferenceResult dataclasses."""

    def test_inference_request_defaults(self):
        from llamatelemetry.inference.base import InferenceRequest
        req = InferenceRequest()
        assert req.prompt is None
        assert req.messages is None
        assert req.max_tokens == 256
        assert req.stream is False
        assert req.request_id is None
        assert req.conversation_id is None
        assert req.metadata == {}

    def test_inference_request_chat(self):
        from llamatelemetry.inference.base import InferenceRequest
        msgs = [{"role": "user", "content": "Hi"}]
        req = InferenceRequest(messages=msgs, max_tokens=128, stream=True)
        assert req.messages == msgs
        assert req.max_tokens == 128
        assert req.stream is True

    def test_inference_request_prompt(self):
        from llamatelemetry.inference.base import InferenceRequest
        req = InferenceRequest(prompt="Once upon a time", max_tokens=64)
        assert req.prompt == "Once upon a time"

    def test_inference_request_with_sampling(self):
        from llamatelemetry.inference.base import InferenceRequest
        from llamatelemetry.inference.types import SamplingParams
        sp = SamplingParams(temperature=0.5)
        req = InferenceRequest(sampling=sp)
        assert req.sampling.temperature == 0.5

    def test_inference_result_defaults(self):
        from llamatelemetry.inference.base import InferenceResult
        r = InferenceResult()
        assert r.output_text == ""
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.ttft_ms == 0.0
        assert r.tpot_ms == 0.0
        assert r.tps == 0.0
        assert r.prefill_tps == 0.0
        assert r.total_latency_ms == 0.0
        assert r.vram_peak_mb == 0.0
        assert r.vram_delta_mb == 0.0
        assert r.queue_delay_ms == 0.0
        assert r.kv_cache_bytes == 0
        assert r.finish_reason == "stop"
        assert r.request_id is None
        assert r.raw is None

    def test_inference_result_custom(self):
        from llamatelemetry.inference.base import InferenceResult
        r = InferenceResult(
            output_text="Hello!",
            input_tokens=5,
            output_tokens=2,
            ttft_ms=50.0,
            tps=120.0,
            total_latency_ms=200.0,
            vram_peak_mb=8192.0,
        )
        assert r.output_text == "Hello!"
        assert r.input_tokens == 5
        assert r.output_tokens == 2
        assert r.ttft_ms == 50.0
        assert r.tps == 120.0
        assert r.total_latency_ms == 200.0
        assert r.vram_peak_mb == 8192.0

    def test_inference_engine_protocol_importable(self):
        from llamatelemetry.inference.base import InferenceEngine
        assert InferenceEngine is not None


class TestCudaInferenceConfig:
    """Test CudaInferenceConfig."""

    def test_defaults(self):
        from llamatelemetry.inference.config import CudaInferenceConfig
        cfg = CudaInferenceConfig()
        assert cfg.backend == "llama.cpp"
        assert cfg.telemetry is True
        assert cfg.scheduler is False
        assert cfg.multi_gpu == "auto"
        assert cfg.dtype == "fp16"
        assert cfg.kv_cache_policy == "lru"

    def test_llamacpp_backend(self):
        from llamatelemetry.inference.config import CudaInferenceConfig
        cfg = CudaInferenceConfig(backend="llama.cpp", llama_server_url="http://127.0.0.1:8090")
        assert cfg.backend == "llama.cpp"
        assert cfg.llama_server_url == "http://127.0.0.1:8090"

    def test_transformers_backend(self):
        from llamatelemetry.inference.config import CudaInferenceConfig
        cfg = CudaInferenceConfig(backend="transformers")
        assert cfg.backend == "transformers"

    def test_with_sampling_params(self):
        from llamatelemetry.inference.config import CudaInferenceConfig
        from llamatelemetry.inference.types import SamplingParams
        sp = SamplingParams(temperature=0.3)
        cfg = CudaInferenceConfig(sampling=sp)
        assert cfg.sampling.temperature == 0.3

    def test_with_device_config(self):
        from llamatelemetry.inference.config import CudaInferenceConfig
        from llamatelemetry.inference.types import DeviceConfig
        dc = DeviceConfig(device_ids=[0, 1])
        cfg = CudaInferenceConfig(device=dc)
        assert cfg.device.device_ids == [0, 1]

    def test_to_device_config(self):
        from llamatelemetry.inference.config import CudaInferenceConfig
        from llamatelemetry.inference.types import DeviceConfig
        cfg = CudaInferenceConfig(dtype="bf16")
        dc = cfg.to_device_config()
        assert isinstance(dc, DeviceConfig)
        assert dc.dtype == "bf16"


class TestInferenceEvents:
    """Test InferenceEvents lifecycle."""

    def test_events_default_all_none(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        assert ev.enqueued_ts is None
        assert ev.start_ts is None
        assert ev.first_token_ts is None
        assert ev.last_token_ts is None
        assert ev.complete_ts is None
        assert ev.input_tokens is None
        assert ev.output_tokens is None
        assert ev.token_timestamps == []
        assert ev.vram_before_mb is None
        assert ev.vram_after_mb is None
        assert ev.vram_peak_mb is None

    def test_mark_enqueued(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.mark_enqueued()
        assert ev.enqueued_ts is not None

    def test_mark_start(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.mark_start()
        assert ev.start_ts is not None

    def test_mark_first_token(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.mark_first_token()
        assert ev.first_token_ts is not None
        assert len(ev.token_timestamps) == 1

    def test_mark_complete(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.mark_start()
        ev.mark_complete()
        assert ev.complete_ts is not None
        assert ev.last_token_ts is not None  # auto-filled

    def test_set_token_counts(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.set_token_counts(input_tokens=50, output_tokens=100)
        assert ev.input_tokens == 50
        assert ev.output_tokens == 100

    def test_set_vram(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.set_vram(before_mb=2000.0, after_mb=4000.0, peak_mb=4200.0)
        assert ev.vram_before_mb == 2000.0
        assert ev.vram_after_mb == 4000.0
        assert ev.vram_peak_mb == 4200.0

    def test_total_duration_s_with_start_and_complete(self):
        import time
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        ev.mark_start()
        time.sleep(0.02)
        ev.mark_complete()
        assert ev.total_duration_s >= 0.01

    def test_total_duration_s_missing_timestamps(self):
        from llamatelemetry.inference.events import InferenceEvents
        ev = InferenceEvents()
        assert ev.total_duration_s == 0.0

    def test_event_recorder_new(self):
        from llamatelemetry.inference.events import EventRecorder, InferenceEvents
        ev = EventRecorder.new()
        assert isinstance(ev, InferenceEvents)

    def test_event_recorder_from_timestamps(self):
        import time
        from llamatelemetry.inference.events import EventRecorder
        now = time.perf_counter()
        ev = EventRecorder.from_timestamps(
            start=now,
            first_token=now + 0.05,
            last_token=now + 0.5,
            input_tokens=100,
            output_tokens=50,
        )
        assert ev.start_ts == now
        assert ev.first_token_ts == now + 0.05
        assert ev.input_tokens == 100
        assert ev.output_tokens == 50


class TestInferenceMetrics:
    """Test standalone inference metric computation functions."""

    def test_compute_ttft(self):
        import time
        from llamatelemetry.inference.events import InferenceEvents
        from llamatelemetry.inference.metrics import compute_ttft
        now = time.perf_counter()
        ev = InferenceEvents(start_ts=now, first_token_ts=now + 0.1)  # 100ms
        ttft = compute_ttft(ev)
        assert abs(ttft - 100.0) < 5.0  # allow ±5ms tolerance

    def test_compute_ttft_missing_timestamps(self):
        from llamatelemetry.inference.events import InferenceEvents
        from llamatelemetry.inference.metrics import compute_ttft
        ev = InferenceEvents()
        assert compute_ttft(ev) == 0.0

    def test_compute_tps(self):
        from llamatelemetry.inference.metrics import compute_tps
        # 50 tokens in 2s = 25 tps
        assert abs(compute_tps(50, 2.0) - 25.0) < 1e-6

    def test_compute_tps_zero_duration(self):
        from llamatelemetry.inference.metrics import compute_tps
        assert compute_tps(50, 0.0) == 0.0

    def test_compute_tps_zero_tokens(self):
        from llamatelemetry.inference.metrics import compute_tps
        assert compute_tps(0, 2.0) == 0.0

    def test_compute_queue_delay(self):
        import time
        from llamatelemetry.inference.metrics import compute_queue_delay
        now = time.perf_counter()
        delay = compute_queue_delay(enqueued_ts=now, start_ts=now + 0.05)
        assert delay >= 0.0

    def test_compute_queue_delay_missing(self):
        from llamatelemetry.inference.metrics import compute_queue_delay
        assert compute_queue_delay(None, None) == 0.0

    def test_compute_all_metrics_from_events(self):
        import time
        from llamatelemetry.inference.events import EventRecorder
        from llamatelemetry.inference.metrics import compute_all_metrics
        now = time.perf_counter()
        ev = EventRecorder.from_timestamps(
            start=now,
            first_token=now + 0.05,
            last_token=now + 0.5,
            input_tokens=100,
            output_tokens=50,
        )
        metrics = compute_all_metrics(ev)
        assert "ttft_ms" in metrics
        assert "tpot_ms" in metrics
        assert "tps" in metrics
        assert "queue_delay_ms" in metrics
        assert "total_latency_ms" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics
        assert metrics["input_tokens"] == 100
        assert metrics["output_tokens"] == 50


class TestInferenceAPIImport:
    """Test inference.api create_engine factory."""

    def test_create_engine_importable(self):
        from llamatelemetry.inference.api import create_engine
        assert callable(create_engine)

    def test_inference_package_exports(self):
        from llamatelemetry.inference import (
            InferenceRequest,
            InferenceResult,
            InferenceEngine,
            SamplingParams,
            BatchConstraints,
            DeviceConfig,
            CudaInferenceConfig,
            InferenceEvents,
            EventRecorder,
            create_engine,
        )
        assert InferenceRequest is not None
        assert InferenceResult is not None
        assert create_engine is not None


class TestInferenceSchedulerRuntime:
    """Test runtime scheduler edge cases."""

    def test_scheduler_does_not_leave_queued_request(self):
        from llamatelemetry.inference.base import InferenceRequest, InferenceResult
        from llamatelemetry.inference.runtime import InferenceRuntime
        from llamatelemetry.inference.types import BatchConstraints

        class DummyEngine:
            name = "dummy"

            def __init__(self):
                self.calls = 0

            def warmup(self):
                pass

            def generate(self, request):
                self.calls += 1
                return InferenceResult(output_text="ok")

            def stream_generate(self, request):
                yield "ok"

            def shutdown(self):
                pass

        engine = DummyEngine()
        runtime = InferenceRuntime(
            engine=engine,
            enable_scheduler=True,
            batch_constraints=BatchConstraints(max_wait_ms=10.0),
        )
        runtime.start()
        runtime.generate(InferenceRequest(prompt="hello", max_tokens=8))
        assert runtime.scheduler is not None
        assert runtime.scheduler.queue_depth == 0
        assert engine.calls == 1

    def test_scheduler_handles_oversize_request(self):
        from llamatelemetry.inference.base import InferenceRequest, InferenceResult
        from llamatelemetry.inference.runtime import InferenceRuntime
        from llamatelemetry.inference.types import BatchConstraints

        class DummyEngine:
            name = "dummy"

            def __init__(self):
                self.calls = 0

            def warmup(self):
                pass

            def generate(self, request):
                self.calls += 1
                return InferenceResult(output_text="ok")

            def stream_generate(self, request):
                yield "ok"

            def shutdown(self):
                pass

        engine = DummyEngine()
        runtime = InferenceRuntime(
            engine=engine,
            enable_scheduler=True,
            batch_constraints=BatchConstraints(max_batch_tokens=4),
        )
        runtime.start()
        runtime.generate(InferenceRequest(prompt="this is a long prompt", max_tokens=10))
        assert runtime.scheduler is not None
        assert runtime.scheduler.queue_depth == 0
        assert engine.calls == 1
