"""
llamatelemetry.transformers.backend - Transformers (original model) backend.

Implements the LLMBackend protocol for HuggingFace Transformers models.
All torch/transformers imports are lazy to keep the core SDK lightweight.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..backends.base import LLMBackend, LLMRequest, LLMResponse


class TransformersBackend:
    """LLM backend for HuggingFace Transformers models (original/non-GGUF).

    Wraps model.generate() into the unified LLMBackend interface.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
        >>> backend = TransformersBackend(model=model, tokenizer=tokenizer)
        >>> req = LLMRequest(operation="chat", messages=[{"role": "user", "content": "Hi"}])
        >>> resp = backend.invoke(req)
    """

    name = "transformers"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        autocast_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """Initialize Transformers backend.

        Args:
            model: A HuggingFace model (AutoModelForCausalLM, etc.).
            tokenizer: Associated tokenizer.
            device: Device string ("cuda", "cuda:0", "cpu"). Auto-detected if None.
            autocast_dtype: Autocast dtype ("fp16", "bf16", None).
            trust_remote_code: Trust remote code in tokenizer.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._trust_remote_code = trust_remote_code
        self._autocast_dtype = autocast_dtype

        if device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    def invoke(self, req: LLMRequest) -> LLMResponse:
        """Execute an LLM request using the Transformers model.

        Supports chat and completions operations.
        """
        start = time.perf_counter()

        if req.operation == "chat":
            return self._invoke_chat(req, start)
        elif req.operation in ("completions", "text_completion"):
            return self._invoke_completions(req, start)
        elif req.operation == "embeddings":
            return self._invoke_embeddings(req, start)
        else:
            raise ValueError(f"Unsupported operation: {req.operation}")

    def _invoke_chat(self, req: LLMRequest, start: float) -> LLMResponse:
        """Handle chat completion using chat template if available."""
        import torch

        messages = req.messages or []
        params = req.parameters or {}

        # Format messages using tokenizer chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt_text = self._format_messages_simple(messages)

        return self._generate(prompt_text, params, start)

    def _invoke_completions(self, req: LLMRequest, start: float) -> LLMResponse:
        """Handle raw text completion."""
        prompt_text = req.prompt or ""
        params = req.parameters or {}
        return self._generate(prompt_text, params, start)

    def _invoke_embeddings(self, req: LLMRequest, start: float) -> LLMResponse:
        """Handle embedding requests (stub - not all models support this)."""
        elapsed = (time.perf_counter() - start) * 1000.0
        return LLMResponse(
            output_texts=[],
            latency_ms=elapsed,
        )

    def _generate(
        self, prompt_text: str, params: Dict[str, Any], start: float
    ) -> LLMResponse:
        """Core generation logic."""
        import torch

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        input_token_count = inputs["input_ids"].shape[-1]

        # Build generate kwargs
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": params.get("max_tokens", 256),
        }
        if "temperature" in params:
            gen_kwargs["temperature"] = params["temperature"]
            gen_kwargs["do_sample"] = True
        if "top_p" in params:
            gen_kwargs["top_p"] = params["top_p"]
            gen_kwargs["do_sample"] = True
        if "top_k" in params:
            gen_kwargs["top_k"] = params["top_k"]
            gen_kwargs["do_sample"] = True
        if "seed" in params:
            torch.manual_seed(params["seed"])

        # Run generation with optional autocast
        with self._maybe_autocast():
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )

        # Decode output
        new_tokens = output_ids[0][input_token_count:]
        output_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_token_count = len(new_tokens)

        elapsed = (time.perf_counter() - start) * 1000.0

        return LLMResponse(
            output_text=output_text,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            finish_reason="stop",
            latency_ms=elapsed,
        )

    def _maybe_autocast(self):
        """Return autocast context if configured, otherwise nullcontext."""
        import contextlib

        if self._autocast_dtype is None:
            return contextlib.nullcontext()

        import torch

        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self._autocast_dtype)
        if dtype is None:
            return contextlib.nullcontext()

        return torch.autocast(device_type=self._device.split(":")[0], dtype=dtype)

    @staticmethod
    def _format_messages_simple(messages: List[Dict[str, Any]]) -> str:
        """Fallback message formatting when chat template is unavailable."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)
