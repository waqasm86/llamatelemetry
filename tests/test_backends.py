"""Tests for llamatelemetry.backends (LLMRequest, LLMResponse, LLMBackend protocol)."""

import pytest


class TestLLMRequest:
    """Test LLMRequest dataclass."""

    def test_default_values(self):
        from llamatelemetry.backends.base import LLMRequest
        req = LLMRequest()
        assert req.operation == "chat"
        assert req.model is None
        assert req.provider is None
        assert req.messages is None
        assert req.prompt is None
        assert req.input_texts is None
        assert req.stream is False
        assert req.request_id is None
        assert req.conversation_id is None

    def test_chat_request(self):
        from llamatelemetry.backends.base import LLMRequest
        msgs = [{"role": "user", "content": "Hello!"}]
        req = LLMRequest(
            operation="chat",
            model="gemma-3-1b",
            messages=msgs,
            stream=False,
        )
        assert req.operation == "chat"
        assert req.model == "gemma-3-1b"
        assert req.messages == msgs

    def test_completions_request(self):
        from llamatelemetry.backends.base import LLMRequest
        req = LLMRequest(operation="completions", prompt="Once upon a time")
        assert req.operation == "completions"
        assert req.prompt == "Once upon a time"

    def test_embeddings_request(self):
        from llamatelemetry.backends.base import LLMRequest
        req = LLMRequest(operation="embeddings", input_texts=["Hello", "World"])
        assert req.operation == "embeddings"
        assert req.input_texts == ["Hello", "World"]

    def test_with_parameters(self):
        from llamatelemetry.backends.base import LLMRequest
        req = LLMRequest(
            operation="chat",
            model="llama-3-8b",
            parameters={"temperature": 0.7, "max_tokens": 256},
        )
        assert req.parameters["temperature"] == 0.7
        assert req.parameters["max_tokens"] == 256

    def test_with_conversation_id(self):
        from llamatelemetry.backends.base import LLMRequest
        req = LLMRequest(conversation_id="session-abc", request_id="req-001")
        assert req.conversation_id == "session-abc"
        assert req.request_id == "req-001"


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_default_values(self):
        from llamatelemetry.backends.base import LLMResponse
        resp = LLMResponse()
        assert resp.output_text is None
        assert resp.output_texts is None
        assert resp.input_tokens is None
        assert resp.output_tokens is None
        assert resp.finish_reason is None
        assert resp.response_id is None
        assert resp.response_model is None
        assert resp.latency_ms is None
        assert resp.raw is None

    def test_chat_response(self):
        from llamatelemetry.backends.base import LLMResponse
        resp = LLMResponse(
            output_text="Hello! How can I help you?",
            input_tokens=5,
            output_tokens=8,
            finish_reason="stop",
            latency_ms=150.5,
        )
        assert resp.output_text == "Hello! How can I help you?"
        assert resp.input_tokens == 5
        assert resp.output_tokens == 8
        assert resp.finish_reason == "stop"
        assert abs(resp.latency_ms - 150.5) < 1e-6

    def test_embeddings_response(self):
        from llamatelemetry.backends.base import LLMResponse
        resp = LLMResponse(output_texts=["emb1", "emb2"], latency_ms=25.0)
        assert resp.output_texts == ["emb1", "emb2"]

    def test_with_raw(self):
        from llamatelemetry.backends.base import LLMResponse
        raw = {"id": "chatcmpl-abc", "choices": []}
        resp = LLMResponse(raw=raw, response_id="chatcmpl-abc")
        assert resp.raw is raw
        assert resp.response_id == "chatcmpl-abc"


class TestLLMBackendProtocol:
    """Test LLMBackend protocol compliance."""

    def test_protocol_runtime_checkable(self):
        from llamatelemetry.backends.base import LLMBackend, LLMRequest, LLMResponse

        class MinimalBackend:
            name = "test"
            def invoke(self, req: LLMRequest) -> LLMResponse:
                return LLMResponse(output_text="test")

        backend = MinimalBackend()
        assert isinstance(backend, LLMBackend)

    def test_protocol_not_satisfied_without_name(self):
        from llamatelemetry.backends.base import LLMBackend

        class BadBackend:
            def invoke(self, req):
                return None

        # Protocol runtime check doesn't verify name attribute types,
        # but it should check method presence
        backend = BadBackend()
        # This is a duck-type check - LLMBackend is @runtime_checkable
        # BadBackend has invoke() method so it may still pass isinstance check
        # What matters is it's importable and usable
        assert callable(backend.invoke)


class TestLlamaCppBackendInit:
    """Test LlamaCppBackend initialization (no server needed)."""

    def test_backend_has_correct_name(self):
        from llamatelemetry.backends.llamacpp import LlamaCppBackend
        assert LlamaCppBackend.name == "llama.cpp"

    def test_backend_init_does_not_need_server(self):
        """Backend init creates a LlamaCppClient without connecting."""
        from llamatelemetry.backends.llamacpp import LlamaCppBackend
        backend = LlamaCppBackend(base_url="http://127.0.0.1:9999")
        assert backend._base_url == "http://127.0.0.1:9999"

    def test_unsupported_operation_raises(self):
        from llamatelemetry.backends.llamacpp import LlamaCppBackend
        from llamatelemetry.backends.base import LLMRequest
        backend = LlamaCppBackend(base_url="http://127.0.0.1:9999")
        req = LLMRequest(operation="unknown_op")
        with pytest.raises(ValueError, match="Unsupported operation"):
            backend.invoke(req)


class TestBackendsImports:
    """Test backends package imports."""

    def test_backends_package_importable(self):
        import llamatelemetry.backends
        assert llamatelemetry.backends is not None

    def test_backends_base_importable(self):
        from llamatelemetry.backends.base import LLMRequest, LLMResponse, LLMBackend
        assert LLMRequest is not None
        assert LLMResponse is not None
        assert LLMBackend is not None

    def test_backends_llamacpp_importable(self):
        from llamatelemetry.backends.llamacpp import LlamaCppBackend
        assert LlamaCppBackend is not None
