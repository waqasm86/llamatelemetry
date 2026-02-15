"""Tests for llamatelemetry.config module."""
import threading
import pytest
from llamatelemetry.config import (
    LlamaTelemetryConfig,
    get_config,
    set_config,
    is_initialized,
    _lock,
)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config between tests."""
    import llamatelemetry.config as mod
    original = mod._config
    mod._config = None
    yield
    mod._config = original


def test_default_config():
    cfg = LlamaTelemetryConfig()
    assert cfg.service_name == "llamatelemetry"
    assert cfg.environment == "development"
    assert cfg.otlp_endpoint is None
    assert cfg.sampling_strategy == "always_on"
    assert cfg.sampling_ratio == 1.0
    assert cfg.redact_prompts is False
    assert cfg.enable_gpu is True
    assert cfg.enable_llama_cpp is True
    assert cfg.enable_nccl is False
    assert cfg._initialized is False


def test_get_config_creates_default():
    cfg = get_config()
    assert isinstance(cfg, LlamaTelemetryConfig)
    assert cfg.service_name == "llamatelemetry"


def test_set_config():
    custom = LlamaTelemetryConfig(service_name="test-svc", environment="production")
    set_config(custom)
    assert get_config().service_name == "test-svc"
    assert get_config().environment == "production"


def test_is_initialized_false_by_default():
    assert is_initialized() is False


def test_is_initialized_true_after_set():
    cfg = LlamaTelemetryConfig(_initialized=True)
    set_config(cfg)
    assert is_initialized() is True


def test_thread_safety():
    results = []

    def worker(name):
        cfg = LlamaTelemetryConfig(service_name=name, _initialized=True)
        set_config(cfg)
        results.append(get_config().service_name)

    threads = [threading.Thread(target=worker, args=(f"svc-{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 10
