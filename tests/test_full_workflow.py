"""Integration test for the full llamatelemetry v1.0.0 workflow."""

import os
import pytest
import llamatelemetry


def test_sdk_version():
    """Verify SDK reports v1.0.0."""
    assert llamatelemetry.version() == "1.0.0"
    assert llamatelemetry.__version__ == "1.0.0"


def test_engine_creation():
    """InferenceEngine can be instantiated (backward compat)."""
    engine = llamatelemetry.InferenceEngine()
    assert engine is not None


def test_gpu_detection():
    """GPU detection via the gpu submodule works."""
    devices = llamatelemetry.gpu.list_devices()
    assert isinstance(devices, list)
    if devices:
        assert devices[0].name  # GPUDevice has a name attribute


def test_init_and_shutdown():
    """Full init -> shutdown lifecycle completes without error."""
    llamatelemetry.init(service_name="test-workflow")
    assert llamatelemetry.is_initialized()
    llamatelemetry.shutdown()


@pytest.mark.skipif(
    not os.path.exists(
        "/media/waqasm86/External1/Project-Nvidia/"
        "Ubuntu-Cuda-Llama.cpp-Executable/bin/"
        "gemma-3-1b-it-Q4_K_M.gguf"
    ),
    reason="Test model not found",
)
def test_model_inference():
    """End-to-end model loading and inference (requires local model)."""
    model_path = (
        "/media/waqasm86/External1/Project-Nvidia/"
        "Ubuntu-Cuda-Llama.cpp-Executable/bin/"
        "gemma-3-1b-it-Q4_K_M.gguf"
    )
    engine = llamatelemetry.InferenceEngine()
    engine.load_model(model_path, gpu_layers=8, verbose=True)
    result = engine.infer("What is AI?", max_tokens=50)
    assert result.text
    assert result.tokens_per_sec > 0
