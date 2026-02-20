"""
llamatelemetry.bench.profiles - Benchmark profiles (prompt sets).

Defines standard benchmark profiles for reproducible testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkProfile:
    """A benchmark profile defining a test scenario.

    Attributes:
        name: Profile name (e.g. "short_prompt", "long_prompt").
        prompt: Input prompt text.
        max_tokens: Maximum tokens to generate.
        category: Profile category (e.g. "latency", "throughput").
        description: Human-readable description.
    """

    name: str
    prompt: str
    max_tokens: int = 128
    category: str = "general"
    description: str = ""


def get_default_profiles() -> List[BenchmarkProfile]:
    """Return the default set of benchmark profiles.

    Returns:
        List of BenchmarkProfile instances covering common scenarios.
    """
    return [
        BenchmarkProfile(
            name="short_prompt",
            prompt="What is the capital of France?",
            max_tokens=64,
            category="latency",
            description="Short prompt, tests TTFT and baseline latency",
        ),
        BenchmarkProfile(
            name="medium_prompt",
            prompt=(
                "Explain the concept of attention mechanisms in transformer neural networks. "
                "Include details about self-attention, multi-head attention, and how they "
                "enable the model to process sequences efficiently."
            ),
            max_tokens=256,
            category="general",
            description="Medium prompt, tests balanced performance",
        ),
        BenchmarkProfile(
            name="long_prompt",
            prompt=(
                "Write a detailed technical analysis of the following topics:\n"
                "1. How CUDA kernels are scheduled on NVIDIA GPUs\n"
                "2. The memory hierarchy in modern GPU architectures\n"
                "3. How tensor cores accelerate matrix multiplication\n"
                "4. The role of shared memory in kernel optimization\n"
                "5. How warp scheduling affects performance\n\n"
                "For each topic, provide specific examples and performance "
                "considerations for the Tesla T4 GPU architecture (SM 7.5). "
                "Include comparisons with newer architectures where relevant."
            ),
            max_tokens=512,
            category="throughput",
            description="Long prompt, tests throughput and sustained generation",
        ),
        BenchmarkProfile(
            name="code_prompt",
            prompt=(
                "Write a Python function that implements a binary search tree with "
                "insert, search, and delete operations. Include type hints, docstrings, "
                "and handle edge cases."
            ),
            max_tokens=512,
            category="code",
            description="Code generation prompt",
        ),
        BenchmarkProfile(
            name="chat_multi_turn",
            prompt=(
                "You are a helpful AI assistant. The user asks:\n"
                "User: Hi, can you help me understand CUDA programming?\n"
                "Assistant: Of course! CUDA is NVIDIA's parallel computing platform.\n"
                "User: How do I optimize memory access patterns?\n"
                "Assistant:"
            ),
            max_tokens=256,
            category="chat",
            description="Multi-turn chat simulation",
        ),
    ]


# Pre-defined profile sets
LATENCY_PROFILES = [p for p in get_default_profiles() if p.category == "latency"]
THROUGHPUT_PROFILES = [p for p in get_default_profiles() if p.category == "throughput"]
CODE_PROFILES = [p for p in get_default_profiles() if p.category == "code"]
