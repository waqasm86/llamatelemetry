"""Tests for llamatelemetry.kaggle.secrets."""
import os
import pytest
from unittest.mock import patch, MagicMock
from llamatelemetry.kaggle.secrets import KaggleSecrets, load_secrets


def test_kaggle_secrets_class_exists():
    assert KaggleSecrets is not None


def test_kaggle_secrets_known_secrets():
    assert "HF_TOKEN" in KaggleSecrets.KNOWN_SECRETS
    assert "OTLP_ENDPOINT" in KaggleSecrets.KNOWN_SECRETS


def test_load_secrets_from_env():
    """Test load_secrets falls back to env vars when not on Kaggle."""
    with patch.dict(os.environ, {"MY_SECRET": "test_value"}, clear=False):
        result = load_secrets({"MY_SECRET": "MY_SECRET"})
        assert result["MY_SECRET"] == "test_value"


def test_load_secrets_missing():
    """Test load_secrets returns None for missing secrets."""
    env_key = "LLAMATELEMETRY_TEST_NONEXISTENT_SECRET_XYZ123"
    if env_key in os.environ:
        del os.environ[env_key]
    result = load_secrets({env_key: env_key})
    assert result[env_key] is None
