"""
llamatelemetry.bench.datasets - Benchmark prompt datasets.

Tiny prompt sets organized by category for reproducible benchmarking.
"""

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name three colors.",
    "What is the capital of France?",
    "Say hello in Japanese.",
    "What is Python?",
]

LONG_PROMPTS = [
    (
        "Write a comprehensive guide to understanding transformer neural networks. "
        "Cover the architecture, self-attention mechanism, positional encoding, "
        "feed-forward layers, and how modern LLMs like GPT and LLaMA build on "
        "the original transformer design. Include mathematical formulations "
        "where appropriate."
    ),
    (
        "Explain the complete lifecycle of a machine learning model in production, "
        "from data collection and preprocessing, through training and evaluation, "
        "to deployment and monitoring. Discuss best practices for each stage "
        "and common pitfalls to avoid."
    ),
]

CODE_PROMPTS = [
    "Write a Python class that implements a thread-safe LRU cache with configurable max size.",
    "Implement a CUDA kernel in C++ that performs matrix multiplication using shared memory.",
    "Write a Python async HTTP server that handles concurrent requests with rate limiting.",
]

CHAT_PROMPTS = [
    "User: What's the weather like today?\nAssistant:",
    "User: Can you help me debug my Python code?\nAssistant: Sure! What's the issue?\nUser: I'm getting a KeyError.\nAssistant:",
]
