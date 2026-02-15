#!/usr/bin/env python3
"""
End-to-end test for llamatelemetry with actual model inference.
"""
import os
import time
from pathlib import Path

import pytest
import llamatelemetry
from llamatelemetry.utils import check_gpu_compatibility


def _find_test_model() -> str | None:
    model_paths = [
        "/media/waqasm86/External1/Project-Nvidia/Ubuntu-Cuda-Llama.cpp-Executable/bin/gemma-3-1b-it-Q4_K_M.gguf",
        str(Path.home() / "gemma-3-1b-it-Q4_K_M.gguf"),
        str(Path.cwd() / "test_model.gguf"),
        str(Path.cwd() / "gemma-3-1b-it-Q4_K_M.gguf"),
    ]

    for path in model_paths:
        if os.path.exists(path):
            return path
    return None


def test_end_to_end_inference():
    print("=" * 60)
    print("llamatelemetry END-TO-END TEST")
    print("=" * 60)

    print("\n1. GPU Compatibility Check:")
    compat = check_gpu_compatibility()
    print(f"   GPU: {compat['gpu_name']}")
    print(f"   Compute Capability: {compat['compute_capability']}")
    print(f"   Compatible: {compat['compatible']}")

    if not compat["compatible"]:
        pytest.skip("Compatible NVIDIA GPU not detected")

    print("\n2. Creating Inference Engine...")
    engine = llamatelemetry.InferenceEngine()
    print("   Engine created")

    print("\n3. Looking for test model...")
    model_path = _find_test_model()

    if not model_path:
        pytest.skip(
            "No GGUF model found. "
            "Download one to run the end-to-end inference test."
        )

    print(f"   Found model: {model_path}")

    print("\n4. Loading model...")
    # For low VRAM, use conservative settings
    engine.load_model(
        model_path,
        gpu_layers=8,
        ctx_size=512,
        batch_size=256,
        ubatch_size=64,
        verbose=True,
        auto_start=True,
    )
    print("   Model loaded successfully!")

    print("\n5. Running inference test...")
    test_prompts = [
        "Explain artificial intelligence in one sentence.",
        "What is 2+2?",
        "Say hello in Spanish.",
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n   Prompt {i+1}: '{prompt}'")
        start_time = time.time()
        result = engine.infer(prompt, max_tokens=50, temperature=0.7)
        elapsed = time.time() - start_time

        if result.success:
            print(f"   Response: {result.text.strip()}")
            print(f"   Tokens: {result.tokens_generated}")
            print(f"   Speed: {result.tokens_per_sec:.1f} tok/s")
            print(f"   Time: {elapsed:.2f}s")
        else:
            pytest.fail(f"Inference failed: {result.error_message}")

    print("\n6. Checking metrics...")
    metrics = engine.get_metrics()
    print(f"   Total requests: {metrics['throughput']['total_requests']}")
    print(f"   Total tokens: {metrics['throughput']['total_tokens']}")
    if metrics["throughput"]["total_requests"] > 0:
        print(f"   Avg tokens/sec: {metrics['throughput']['tokens_per_sec']:.1f}")

    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETE!")
    print("llamatelemetry is fully functional!")
    print("=" * 60)
