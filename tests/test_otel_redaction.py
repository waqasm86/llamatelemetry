"""Tests for llamatelemetry.otel.redaction."""
from llamatelemetry.otel.redaction import RedactionSpanProcessor, _PROMPT_KEYS


def test_prompt_keys_defined():
    assert isinstance(_PROMPT_KEYS, frozenset)
    assert "gen_ai.input.messages" in _PROMPT_KEYS
    assert "gen_ai.output.messages" in _PROMPT_KEYS
    assert "gen_ai.system_instructions" in _PROMPT_KEYS


def test_redaction_processor_creation():
    proc = RedactionSpanProcessor(redact_prompts=True, redact_keys=["api_key"])
    assert proc is not None


def test_redaction_processor_no_args():
    proc = RedactionSpanProcessor()
    assert proc is not None


def test_redaction_processor_prompts_flag():
    proc = RedactionSpanProcessor(redact_prompts=True)
    assert proc._redact_prompts is True


def test_redaction_processor_custom_keys():
    proc = RedactionSpanProcessor(redact_keys=["secret", "password"])
    assert "secret" in proc._redact_keys
    assert "password" in proc._redact_keys
