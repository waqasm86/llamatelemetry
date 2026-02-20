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
            device: Device string ("cuda", "cuda:0").
            config: Full inference configuration.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._config = config
        self._input_device = None

        if device is None:
            self._device = "cuda"
        else:
            if device.startswith("cpu"):
                raise ValueError("CPU devices are not supported. CUDA is required.")
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

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Transformers inference.")

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
        model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if self._config:
            if self._config.transformers_device_map is not None:
                model_kwargs["device_map"] = self._config.transformers_device_map
            if self._config.transformers_max_memory is not None:
                model_kwargs["max_memory"] = self._config.transformers_max_memory
            if self._config.transformers_trust_remote_code:
                model_kwargs["trust_remote_code"] = True

            if self._config.transformers_load_in_4bit or self._config.transformers_load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_kwargs: Dict[str, Any] = {
                        "load_in_4bit": self._config.transformers_load_in_4bit,
                        "load_in_8bit": self._config.transformers_load_in_8bit,
                    }
                    if self._config.transformers_bnb_4bit_compute_dtype:
                        bnb_dtype = dtype_map.get(
                            self._config.transformers_bnb_4bit_compute_dtype,
                            torch.float16,
                        )
                        bnb_kwargs["bnb_4bit_compute_dtype"] = bnb_dtype
                    if self._config.transformers_bnb_4bit_quant_type:
                        bnb_kwargs["bnb_4bit_quant_type"] = self._config.transformers_bnb_4bit_quant_type
                    if self._config.transformers_bnb_4bit_use_double_quant:
                        bnb_kwargs["bnb_4bit_use_double_quant"] = True
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        "BitsAndBytesConfig required for 4-bit/8-bit loading. "
                        "Install bitsandbytes and transformers with quantization support."
                    ) from exc

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **model_kwargs,
        )

        if not model_kwargs.get("device_map"):
            self._model.to(self._device)
            self._input_device = self._device
        else:
            self._input_device = self._infer_input_device()

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

        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        input_device = self._input_device or self._device
        inputs = inputs.to(input_device)
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
            if request.sampling.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = request.sampling.repetition_penalty
            if request.sampling.seed is not None:
                torch.manual_seed(request.sampling.seed)
            if request.sampling.stop_sequences:
                try:
                    from transformers import StoppingCriteria, StoppingCriteriaList

                    stop_ids = [
                        self._tokenizer.encode(s, add_special_tokens=False)
                        for s in request.sampling.stop_sequences
                    ]
                    stop_ids = [s for s in stop_ids if s]

                    if stop_ids:
                        class _StopOnSequences(StoppingCriteria):
                            def __init__(self, sequences):
                                self._sequences = sequences

                            def __call__(self, input_ids, scores, **kwargs):
                                for seq in self._sequences:
                                    if input_ids.shape[-1] >= len(seq):
                                        if input_ids[0, -len(seq):].tolist() == seq:
                                            return True
                                return False

                        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                            [_StopOnSequences(stop_ids)]
                        )
                except Exception:
                    pass

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode
        new_tokens = output_ids[0][input_token_count:]
        output_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_token_count = len(new_tokens)

        if events.start_ts is not None and output_token_count > 0:
            # Without streaming, TTFT is unknown; use start_ts to avoid overstating it.
            events.first_token_ts = events.start_ts

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

    def _infer_input_device(self) -> str:
        """Infer the best input device for sharded models."""
        if self._model is None:
            return self._device

        try:
            if hasattr(self._model, "hf_device_map") and isinstance(self._model.hf_device_map, dict):
                for device in self._model.hf_device_map.values():
                    if isinstance(device, str) and device.startswith("cuda"):
                        return device
        except Exception:
            pass

        try:
            first_param = next(self._model.parameters())
            return str(first_param.device)
        except Exception:
            return self._device
