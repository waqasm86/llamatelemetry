"""Tests for llamatelemetry.llama.gguf."""
import hashlib
import tempfile
import os
import pytest

from llamatelemetry.llama.gguf import compute_sha256


def test_compute_sha256():
    # Create a temp file with known content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as f:
        f.write(b"test content for sha256")
        tmp_path = f.name

    try:
        result = compute_sha256(tmp_path)
        expected = hashlib.sha256(b"test content for sha256").hexdigest()
        assert result == expected
    finally:
        os.unlink(tmp_path)


def test_compute_sha256_empty_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_path = f.name

    try:
        result = compute_sha256(tmp_path)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
    finally:
        os.unlink(tmp_path)


def test_gguf_module_exports():
    from llamatelemetry.llama import gguf
    assert hasattr(gguf, "compute_sha256")
    assert callable(gguf.compute_sha256)
