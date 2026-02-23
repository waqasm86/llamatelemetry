"""
llamatelemetry.llama_cpp_native.inference - High-level inference loop

Combines model, context, batch, sampling, and tokenization.
Implements prefill + decode phases with optional multi-GPU distribution.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import logging

from .model import LlamaModel
from .context import LlamaContext
from .batch import LlamaBatch
from .sampler import SamplerChain, create_sampler
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerateResponse:
    """Response from text generation."""
    text: str
    tokens: List[int]
    token_count: int
    ttft_ms: float  # Time-to-first-token
    tpot_ms: float  # Time-per-output-token
    total_ms: float
    finish_reason: str


class InferenceLoop:
    """
    High-level inference interface.

    Handles:
      - Prefill + decode phases
      - Token generation with sampling
      - Performance measurement (TTFT, TPOT)
      - Optional multi-GPU distribution
    """

    def __init__(
        self,
        model: LlamaModel,
        context: LlamaContext,
        sampler: Optional[SamplerChain] = None,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        """
        Create inference loop.

        Args:
            model: Loaded LlamaModel
            context: LlamaContext for inference state
            sampler: SamplerChain for token generation
            n_batch: Batch size for processing
            verbose: Enable debug logging
        """
        self.model = model
        self.context = context
        self.tokenizer = Tokenizer(model)
        self.sampler = sampler or create_sampler()
        self.n_batch = n_batch
        self.verbose = verbose

        logger.info(f"Inference loop initialized")
        logger.info(f"  Model: {model.metadata.get('ftype')}")
        logger.info(f"  Batch size: {n_batch}")
        logger.info(f"  Sampler: {sampler}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> GenerateResponse:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-P (nucleus) threshold
            top_k: Top-K limit
            frequency_penalty: Frequency penalty coefficient
            presence_penalty: Presence penalty coefficient
            stop_sequences: Sequences that stop generation
            seed: Random seed (for reproducibility)

        Returns:
            GenerateResponse with generated text and metrics
        """
        start_time = time.perf_counter()

        # Create sampler with parameters
        sampler = create_sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
        )

        # Tokenize prompt
        input_tokens = self.tokenizer.encode(prompt, add_special=True)
        if not input_tokens:
            input_tokens = [1]  # Default to BOS

        input_token_count = len(input_tokens)
        if self.verbose:
            logger.info(f"Prompt tokens: {input_token_count}")

        # Prefill phase
        batch = LlamaBatch(size=self.n_batch, n_seq_max=1)

        # Add input tokens to batch
        prefill_start = time.perf_counter()
        for i, token in enumerate(input_tokens):
            batch.add(token, i, [0], logits=False)

        # Last token needs logits
        batch.add(input_tokens[-1], len(input_tokens) - 1, [0], logits=True)

        # Prefill
        self.context.encode(batch)
        prefill_elapsed = time.perf_counter() - prefill_start

        # Decode phase
        output_tokens = []
        decode_start = time.perf_counter()
        ttft = None

        batch.clear()

        for pos in range(len(input_tokens), len(input_tokens) + max_tokens):
            # Get logits and sample
            logits = self.context.get_logits_ith(0)

            # Sample token
            token = sampler.sample(self.context, idx=0)

            # Record TTFT
            if not output_tokens:
                ttft = time.perf_counter() - decode_start

            output_tokens.append(token)

            if self.verbose and len(output_tokens) % 10 == 0:
                logger.debug(f"Generated {len(output_tokens)} tokens")

            # Check for end-of-sequence
            if self.tokenizer.is_eog(token):
                finish_reason = "stop"
                break

            # Prepare next token for decode
            batch.add(token, pos, [0], logits=True)
            self.context.decode(batch)
            batch.clear()
        else:
            finish_reason = "length"

        decode_elapsed = time.perf_counter() - decode_start
        total_elapsed = time.perf_counter() - start_time

        # Calculate metrics
        ttft_ms = ttft * 1000 if ttft else 0
        tpot_ms = (decode_elapsed / len(output_tokens) * 1000) if output_tokens else 0
        total_ms = total_elapsed * 1000

        # Detokenize output
        output_text = self.tokenizer.decode(output_tokens, remove_special=True)

        if self.verbose:
            logger.info(f"Generation complete:")
            logger.info(f"  Input tokens: {input_token_count}")
            logger.info(f"  Output tokens: {len(output_tokens)}")
            logger.info(f"  TTFT: {ttft_ms:.1f}ms")
            logger.info(f"  TPOT: {tpot_ms:.2f}ms")
            logger.info(f"  Total: {total_ms:.1f}ms")

        return GenerateResponse(
            text=output_text,
            tokens=output_tokens,
            token_count=len(output_tokens),
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            finish_reason=finish_reason,
        )

    def __repr__(self) -> str:
        return f"InferenceLoop(model={self.model.metadata.get('ftype')})"
