"""Tests for CUDA-only enforcement."""

import pytest


def test_require_cuda_raises_when_missing(monkeypatch):
    from llamatelemetry import utils

    monkeypatch.setattr(utils, "detect_cuda", lambda: {"available": False, "gpus": []})

    with pytest.raises(RuntimeError):
        utils.require_cuda()
