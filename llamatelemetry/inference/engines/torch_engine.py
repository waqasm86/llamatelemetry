"""
llamatelemetry.inference.engines.torch_engine - Transformers inference engine.

Implements the InferenceEngine protocol for HuggingFace Transformers models.
All torch/transformers imports are lazy to keep core SDK lightweight.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional

from ..base import InferenceEngine, InferenceRequest, InferenceResult
from ..events import InferenceEvents
from ..metrics import compute_all_metrics
from ..config import CudaInferenceConfig


class TorchEngine:
    """Inference engine for HuggingFace Transformers models.

    Provides the same InferenceResult (TTFT, TPOT, TPS, VRAM) as LlamaCppEngine
    but using PyTorch model.generate().

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
        >>> engine = TorchEngine(model=model, tokenizer=tokenizer)
        >>> engine.warmup()
        >>> result = engine.generate(InferenceRequest(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=128,
        ... ))
        >>> print(f"TTFT: {result.ttft_ms:.1f}ms, TPS: {result.tps:.1f}")
    """

    name = "transformers"

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[CudaInferenceConfig] = None,
    ):
        """Initialize Transformers engine.

        Args:
            model: Pre-loaded model. If None, loads from model_path.
            tokenizer: Pre-loaded tokenizer. If None, loads from model_path.
            model_path: HuggingFace model ID or local path.
            device: Device string ("cuda", "cuda:0", "cpu").
            config: Full inference configuration.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._config = config

        if device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    @classmethod
    def from_config(cls, config: CudaInferenceConfig) -> "TorchEngine":
        """Create engine from configuration."""
        return cls(
            model_path=config.model_path,
            config=config,
        )

    def warmup(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        if self._model_path is None:
            raise ValueError("Either model or model_path must be provided")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(
            self._config.dtype if self._config else "fp16",
            torch.float16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=dtype,
            device_map=self._device,
        )

        # Apply torch.compile if configured
        if self._config and self._config.use_torch_compile:
            self._model = torch.compile(self._model)

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Execute synchronous inference using model.generate()."""
        import torch

        if self._model is None:
            self.warmup()

        events = InferenceEvents()

        # Get VRAM before
        vram_before = self._get_vram_mb()
        events.set_vram(before_mb=vram_before)

        events.mark_start()

        # Prepare input
        if request.messages:
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt_text = self._tokenizer.apply_chat_template(
                    request.messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in request.messages
                ) + "\nassistant:"
        else:
            prompt_text = request.prompt or ""

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        input_token_count = inputs["input_ids"].shape[-1]

        # Build generate kwargs
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": request.max_tokens}
        if request.sampling:
            if request.sampling.temperature > 0:
                gen_kwargs["temperature"] = request.sampling.temperature
                gen_kwargs["do_sample"] = True
            if request.sampling.top_p < 1.0:
                gen_kwargs["top_p"] = request.sampling.top_p
            if request.sampling.top_k > 0:
                gen_kwargs["top_k"] = request.sampling.top_k
            if request.sampling.seed is not None:
                torch.manual_seed(request.sampling.seed)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        events.mark_first_token()

        # Decode
        new_tokens = output_ids[0][input_token_count:]
        output_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_token_count = len(new_tokens)

        events.mark_last_token()
        events.mark_complete()
        events.set_token_counts(input_token_count, output_token_count)

        # Get VRAM after
        vram_after = self._get_vram_mb()
        events.set_vram(after_mb=vram_after)

        # Compute metrics
        all_metrics = compute_all_metrics(events)

        return InferenceResult(
            output_text=output_text,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            ttft_ms=all_metrics["ttft_ms"],
            tpot_ms=all_metrics["tpot_ms"],
            tps=all_metrics["tps"],
            prefill_tps=all_metrics["prefill_tps"],
            total_latency_ms=all_metrics["total_latency_ms"],
            vram_peak_mb=max(vram_before or 0, vram_after or 0),
            vram_delta_mb=(vram_after or 0) - (vram_before or 0),
            finish_reason="stop",
            request_id=request.request_id,
        )

    def stream_generate(self, request: InferenceRequest) -> Iterator[str]:
        """Execute streaming inference (basic implementation)."""
        # For simplicity, generate fully then yield
        result = self.generate(request)
        yield result.output_text

    def shutdown(self) -> None:
        """Release model and free GPU memory."""
        if self._model is not None:
            try:
                import torch
                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                self._model = None

    def _get_vram_mb(self) -> Optional[float]:
        """Get current VRAM usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return None
