"""Tests for llamatelemetry.semconv.gen_ai, gen_ai_builder, and mapping modules."""

import pytest


# ---------------------------------------------------------------------------
# gen_ai constants
# ---------------------------------------------------------------------------

class TestGenAIConstants:
    """Test gen_ai.* semantic convention constants."""

    def test_core_attributes_are_strings(self):
        from llamatelemetry.semconv import gen_ai
        attrs = [
            gen_ai.GEN_AI_SYSTEM,
            gen_ai.GEN_AI_PROVIDER_NAME,
            gen_ai.GEN_AI_OPERATION_NAME,
        ]
        for a in attrs:
            assert isinstance(a, str), f"{a} is not a string"
            assert a.startswith("gen_ai."), f"{a} doesn't start with gen_ai."

    def test_request_attributes_are_strings(self):
        from llamatelemetry.semconv import gen_ai
        req_attrs = [
            gen_ai.GEN_AI_REQUEST_MODEL,
            gen_ai.GEN_AI_REQUEST_TEMPERATURE,
            gen_ai.GEN_AI_REQUEST_TOP_P,
            gen_ai.GEN_AI_REQUEST_TOP_K,
            gen_ai.GEN_AI_REQUEST_MAX_TOKENS,
            gen_ai.GEN_AI_REQUEST_FREQUENCY_PENALTY,
            gen_ai.GEN_AI_REQUEST_PRESENCE_PENALTY,
            gen_ai.GEN_AI_REQUEST_SEED,
            gen_ai.GEN_AI_REQUEST_STOP_SEQUENCES,
            gen_ai.GEN_AI_REQUEST_CHOICE_COUNT,
        ]
        for a in req_attrs:
            assert isinstance(a, str), f"{a} is not a string"
            assert a.startswith("gen_ai.request."), f"{a} doesn't start with gen_ai.request."

    def test_response_attributes_are_strings(self):
        from llamatelemetry.semconv import gen_ai
        resp_attrs = [
            gen_ai.GEN_AI_RESPONSE_ID,
            gen_ai.GEN_AI_RESPONSE_MODEL,
            gen_ai.GEN_AI_RESPONSE_FINISH_REASONS,
        ]
        for a in resp_attrs:
            assert isinstance(a, str), f"{a} is not a string"
            assert a.startswith("gen_ai.response."), f"{a} doesn't start with gen_ai.response."

    def test_usage_attributes_are_strings(self):
        from llamatelemetry.semconv import gen_ai
        usage_attrs = [
            gen_ai.GEN_AI_USAGE_INPUT_TOKENS,
            gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
            gen_ai.GEN_AI_TOKEN_TYPE,
        ]
        for a in usage_attrs:
            assert isinstance(a, str), f"{a} is not a string"

    def test_operation_name_values(self):
        from llamatelemetry.semconv import gen_ai
        ops = [
            gen_ai.OP_CHAT,
            gen_ai.OP_CREATE_AGENT,
            gen_ai.OP_EMBEDDINGS,
            gen_ai.OP_EXECUTE_TOOL,
            gen_ai.OP_GENERATE_CONTENT,
            gen_ai.OP_INVOKE_AGENT,
            gen_ai.OP_TEXT_COMPLETION,
        ]
        for op in ops:
            assert isinstance(op, str), f"Operation {op} is not a string"

    def test_provider_name_values(self):
        from llamatelemetry.semconv import gen_ai
        providers = [
            gen_ai.PROVIDER_ANTHROPIC,
            gen_ai.PROVIDER_OPENAI,
            gen_ai.PROVIDER_LLAMA_CPP,
            gen_ai.PROVIDER_TRANSFORMERS,
            gen_ai.PROVIDER_UNSLOTH,
            gen_ai.PROVIDER_MISTRAL,
            gen_ai.PROVIDER_GEMINI,
        ]
        for p in providers:
            assert isinstance(p, str), f"Provider {p} is not a string"

    def test_llama_cpp_provider_value(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.PROVIDER_LLAMA_CPP == "llama_cpp"

    def test_chat_operation_value(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.OP_CHAT == "chat"

    def test_embeddings_operation_value(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.OP_EMBEDDINGS == "embeddings"

    def test_output_type_values(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.OUTPUT_TEXT == "text"
        assert gen_ai.OUTPUT_JSON == "json"
        assert gen_ai.OUTPUT_IMAGE == "image"
        assert gen_ai.OUTPUT_SPEECH == "speech"

    def test_token_type_values(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.TOKEN_INPUT == "input"
        assert gen_ai.TOKEN_OUTPUT == "output"

    def test_tool_type_values(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.TOOL_FUNCTION == "function"
        assert gen_ai.TOOL_EXTENSION == "extension"
        assert gen_ai.TOOL_DATASTORE == "datastore"

    def test_no_duplicate_attribute_names(self):
        """All gen_ai.* attribute strings must be unique."""
        from llamatelemetry.semconv import gen_ai
        all_vals = [v for k, v in vars(gen_ai).items()
                    if not k.startswith("_") and isinstance(v, str)
                    and v.startswith("gen_ai.")]
        assert len(all_vals) == len(set(all_vals)), "Duplicate gen_ai.* attribute strings found"

    def test_gen_ai_importable_from_semconv(self):
        from llamatelemetry.semconv import gen_ai
        assert gen_ai is not None


# ---------------------------------------------------------------------------
# gen_ai_builder
# ---------------------------------------------------------------------------

class TestGenAIBuilder:
    """Test gen_ai_builder functions."""

    def test_build_request_attrs_minimal(self):
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_request
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_request(model="gemma-3-1b")
        assert attrs[gen_ai.GEN_AI_REQUEST_MODEL] == "gemma-3-1b"
        assert attrs[gen_ai.GEN_AI_OPERATION_NAME] == gen_ai.OP_CHAT
        assert attrs[gen_ai.GEN_AI_PROVIDER_NAME] == gen_ai.PROVIDER_LLAMA_CPP

    def test_build_request_attrs_full(self):
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_request
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_request(
            model="llama-3-8b",
            operation=gen_ai.OP_TEXT_COMPLETION,
            provider=gen_ai.PROVIDER_TRANSFORMERS,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=256,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            seed=42,
            stop_sequences=["</s>"],
            choice_count=1,
            stream=True,
            conversation_id="conv-123",
        )
        assert attrs[gen_ai.GEN_AI_REQUEST_MODEL] == "llama-3-8b"
        assert attrs[gen_ai.GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert attrs[gen_ai.GEN_AI_REQUEST_TOP_P] == 0.9
        assert attrs[gen_ai.GEN_AI_REQUEST_TOP_K] == 50
        assert attrs[gen_ai.GEN_AI_REQUEST_MAX_TOKENS] == 256
        assert attrs[gen_ai.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.1
        assert attrs[gen_ai.GEN_AI_REQUEST_PRESENCE_PENALTY] == 0.2
        assert attrs[gen_ai.GEN_AI_REQUEST_SEED] == 42
        assert attrs[gen_ai.GEN_AI_REQUEST_STOP_SEQUENCES] == ["</s>"]
        assert attrs[gen_ai.GEN_AI_CONVERSATION_ID] == "conv-123"

    def test_build_request_attrs_none_excluded(self):
        """None values should not appear in output."""
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_request
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_request(model="test", temperature=None)
        assert gen_ai.GEN_AI_REQUEST_TEMPERATURE not in attrs
        assert gen_ai.GEN_AI_REQUEST_TOP_P not in attrs

    def test_build_response_attrs_minimal(self):
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_response
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_response(input_tokens=10, output_tokens=50)
        assert attrs[gen_ai.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert attrs[gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS] == 50

    def test_build_response_attrs_full(self):
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_response
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_response(
            response_id="chatcmpl-abc123",
            response_model="gemma-3-1b",
            input_tokens=100,
            output_tokens=200,
            finish_reasons=["stop"],
        )
        assert attrs[gen_ai.GEN_AI_RESPONSE_ID] == "chatcmpl-abc123"
        assert attrs[gen_ai.GEN_AI_RESPONSE_MODEL] == "gemma-3-1b"
        assert attrs[gen_ai.GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS] == 200
        assert attrs[gen_ai.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]

    def test_build_response_attrs_none_excluded(self):
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_response
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_response()
        assert len(attrs) == 0

    def test_build_tool_attrs_no_content(self):
        """Tool attrs with record_content=False should return empty dict."""
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_tools
        attrs = build_gen_ai_attrs_from_tools(
            tool_definitions=[{"name": "my_tool"}],
            tool_calls=[{"name": "my_tool", "id": "call-1"}],
            record_content=False,
        )
        assert len(attrs) == 0

    def test_build_tool_attrs_with_content(self):
        from llamatelemetry.semconv.gen_ai_builder import build_gen_ai_attrs_from_tools
        from llamatelemetry.semconv import gen_ai
        attrs = build_gen_ai_attrs_from_tools(
            tool_definitions=[{"name": "my_tool", "type": "function"}],
            tool_calls=[{"id": "call-1", "name": "my_tool", "type": "function"}],
            record_content=True,
        )
        assert gen_ai.GEN_AI_TOOL_DEFINITIONS in attrs
        assert gen_ai.GEN_AI_TOOL_CALL_ID in attrs
        assert attrs[gen_ai.GEN_AI_TOOL_CALL_ID] == "call-1"

    def test_build_content_attrs_no_content(self):
        """Content not recorded when record_content=False - hashes used instead."""
        from llamatelemetry.semconv.gen_ai_builder import build_content_attrs
        from llamatelemetry.semconv import gen_ai
        msgs = [{"role": "user", "content": "Hello"}]
        attrs = build_content_attrs(
            input_messages=msgs,
            record_content=False,
        )
        assert gen_ai.GEN_AI_INPUT_MESSAGES not in attrs
        assert "llamatelemetry.prompt.sha256" in attrs
        sha = attrs["llamatelemetry.prompt.sha256"]
        assert isinstance(sha, str) and len(sha) == 64  # SHA-256 hex

    def test_build_content_attrs_with_content(self):
        from llamatelemetry.semconv.gen_ai_builder import build_content_attrs
        from llamatelemetry.semconv import gen_ai
        msgs = [{"role": "user", "content": "Hello!"}]
        attrs = build_content_attrs(
            input_messages=msgs,
            record_content=True,
        )
        assert gen_ai.GEN_AI_INPUT_MESSAGES in attrs

    def test_build_content_attrs_truncation(self):
        """Content is truncated to record_content_max_chars."""
        from llamatelemetry.semconv.gen_ai_builder import build_content_attrs
        from llamatelemetry.semconv import gen_ai
        long_content = "x" * 10000
        msgs = [{"role": "user", "content": long_content}]
        attrs = build_content_attrs(
            input_messages=msgs,
            record_content=True,
            record_content_max_chars=100,
        )
        assert len(attrs[gen_ai.GEN_AI_INPUT_MESSAGES]) <= 100


# ---------------------------------------------------------------------------
# semconv.mapping (dual-emit)
# ---------------------------------------------------------------------------

class TestSemconvMapping:
    """Test semconv.mapping dual-emit helpers."""

    def test_to_gen_ai_attrs_basic(self):
        from llamatelemetry.semconv.mapping import to_gen_ai_attrs
        from llamatelemetry.semconv import gen_ai
        payload = {"model": "gemma-3-1b", "input_tokens": 10, "output_tokens": 50}
        attrs = to_gen_ai_attrs(payload)
        assert attrs[gen_ai.GEN_AI_REQUEST_MODEL] == "gemma-3-1b"
        assert attrs[gen_ai.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert attrs[gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS] == 50

    def test_to_legacy_llm_attrs_basic(self):
        from llamatelemetry.semconv.mapping import to_legacy_llm_attrs
        from llamatelemetry.semconv import keys
        payload = {"model": "gemma-3-1b", "input_tokens": 10, "output_tokens": 50, "stream": False}
        attrs = to_legacy_llm_attrs(payload)
        assert attrs[keys.LLM_MODEL] == "gemma-3-1b"
        assert attrs[keys.LLM_INPUT_TOKENS] == 10
        assert attrs[keys.LLM_OUTPUT_TOKENS] == 50

    def test_to_legacy_llm_attrs_finish_reasons_list(self):
        """finish_reasons list is reduced to first element for legacy attribute."""
        from llamatelemetry.semconv.mapping import to_legacy_llm_attrs
        from llamatelemetry.semconv import keys
        payload = {"finish_reasons": ["stop", "length"]}
        attrs = to_legacy_llm_attrs(payload)
        assert attrs[keys.LLM_FINISH_REASON] == "stop"

    def test_dual_emit_attrs_merges_both(self):
        from llamatelemetry.semconv.mapping import dual_emit_attrs
        from llamatelemetry.semconv import gen_ai, keys
        payload = {"model": "llama-3-8b", "input_tokens": 100, "output_tokens": 200}
        attrs = dual_emit_attrs(payload)
        # Should contain both gen_ai.* and llm.*
        assert gen_ai.GEN_AI_REQUEST_MODEL in attrs
        assert keys.LLM_MODEL in attrs

    def test_gen_ai_attrs_dataclass(self):
        from llamatelemetry.semconv.mapping import GenAIAttrs
        attrs = GenAIAttrs(model="gemma-3-1b", input_tokens=10, output_tokens=50)
        assert attrs.model == "gemma-3-1b"
        assert attrs.input_tokens == 10
        assert attrs.output_tokens == 50

    def test_gen_ai_attrs_defaults_to_none(self):
        from llamatelemetry.semconv.mapping import GenAIAttrs
        attrs = GenAIAttrs()
        assert attrs.model is None
        assert attrs.provider is None
        assert attrs.input_tokens is None

    def test_set_dual_attrs_on_noop_span(self):
        """set_dual_attrs should not raise even with a minimal span-like object."""
        from llamatelemetry.semconv.mapping import set_dual_attrs

        class FakeSpan:
            def __init__(self):
                self.attrs = {}
            def set_attribute(self, k, v):
                self.attrs[k] = v

        span = FakeSpan()
        payload = {"model": "test-model", "input_tokens": 5, "output_tokens": 10}
        set_dual_attrs(span, payload)
        assert len(span.attrs) > 0

    def test_none_values_excluded(self):
        from llamatelemetry.semconv.mapping import to_gen_ai_attrs
        payload = {"model": "test", "input_tokens": None, "output_tokens": None}
        attrs = to_gen_ai_attrs(payload)
        from llamatelemetry.semconv import gen_ai
        assert gen_ai.GEN_AI_USAGE_INPUT_TOKENS not in attrs
        assert gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
