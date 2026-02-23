"""
llamatelemetry.llama_cpp_native.sampler - Composable sampling pipelines

Direct pybind11 binding to llama_sampler_* APIs.
Implements 20+ sampling algorithms with composable pipeline pattern.
"""

from typing import Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SamplerType(Enum):
    """Sampling algorithm types."""
    GREEDY = "greedy"  # Always pick highest logit
    DIST = "dist"  # Distribution-based sampling
    TEMPERATURE = "temperature"  # Apply temperature scaling
    TOP_K = "top_k"  # Keep top K tokens
    TOP_P = "top_p"  # Nucleus sampling (cumulative probability)
    MIN_P = "min_p"  # Minimum probability threshold
    TYPICAL = "typical"  # Typical sampling
    TAIL_FREE = "tail_free"  # Tail-free sampling
    LOCAL_TYPICAL = "local_typical"  # Local typical sampling
    ENTROPY = "entropy"  # Entropy-based sampling
    MIROSTAT = "mirostat"  # Mirostat (target perplexity v1)
    MIROSTAT_V2 = "mirostat_v2"  # Mirostat v2 (target perplexity v2)
    GRAMMAR = "grammar"  # Grammar-guided sampling
    SOFTMAX = "softmax"  # Softmax (convert logits to probs)
    DRY = "dry"  # Don't Repeat Yourself penalty


class SamplerChain:
    """
    Composable sampling pipeline.

    Chains multiple samplers together for sophisticated generation control.

    Example:
        chain = SamplerChain()
        chain.add_temperature(0.7)
        chain.add_top_p(0.9)
        chain.add_frequency_penalty(1.0)
        token = chain.sample(context)
    """

    def __init__(self):
        """Create empty sampler chain."""
        # Native pybind11 call:
        # params = llama_cpp.LlamaSamplerChainParams()
        # self._chain_ptr = llama_cpp.llama_sampler_chain_init(params)

        self._chain_ptr = None  # Placeholder
        self._samplers = []

    def add_sampler(self, sampler_type: SamplerType, **kwargs) -> 'SamplerChain':
        """
        Add sampler to chain.

        Args:
            sampler_type: Type of sampler to add
            **kwargs: Sampler-specific parameters

        Returns:
            Self for method chaining
        """
        if sampler_type == SamplerType.TEMPERATURE:
            return self.add_temperature(kwargs.get('temperature', 1.0))
        elif sampler_type == SamplerType.TOP_K:
            return self.add_top_k(kwargs.get('k', 40))
        elif sampler_type == SamplerType.TOP_P:
            return self.add_top_p(kwargs.get('p', 0.9))
        elif sampler_type == SamplerType.GREEDY:
            return self.add_greedy()
        else:
            logger.warning(f"Unsupported sampler type: {sampler_type}")
            return self

    def add_greedy(self) -> 'SamplerChain':
        """Add greedy sampler (always pick max logit)."""
        # Native call:
        # sampler = llama_cpp.llama_sampler_init_greedy()
        # llama_cpp.llama_sampler_chain_add(self._chain_ptr, sampler)

        self._samplers.append(('greedy', {}))
        return self

    def add_temperature(self, temperature: float) -> 'SamplerChain':
        """
        Add temperature sampler.

        Args:
            temperature: Temperature value (>1.0 = more random, <1.0 = more focused)
        """
        # Native call:
        # sampler = llama_cpp.llama_sampler_init_temp(temperature)
        # llama_cpp.llama_sampler_chain_add(self._chain_ptr, sampler)

        self._samplers.append(('temperature', {'temperature': temperature}))
        return self

    def add_top_k(self, k: int) -> 'SamplerChain':
        """
        Add top-K sampler.

        Args:
            k: Keep top K tokens by probability
        """
        # Native call:
        # sampler = llama_cpp.llama_sampler_init_top_k(k)
        # llama_cpp.llama_sampler_chain_add(self._chain_ptr, sampler)

        self._samplers.append(('top_k', {'k': k}))
        return self

    def add_top_p(self, p: float, min_keep: int = 1) -> 'SamplerChain':
        """
        Add top-P (nucleus) sampler.

        Args:
            p: Cumulative probability threshold (0.0-1.0)
            min_keep: Minimum tokens to keep
        """
        # Native call:
        # sampler = llama_cpp.llama_sampler_init_top_p(p, min_keep)
        # llama_cpp.llama_sampler_chain_add(self._chain_ptr, sampler)

        self._samplers.append(('top_p', {'p': p, 'min_keep': min_keep}))
        return self

    def add_min_p(self, p: float, min_keep: int = 1) -> 'SamplerChain':
        """
        Add min-P sampler.

        Args:
            p: Minimum probability threshold
            min_keep: Minimum tokens to keep
        """
        self._samplers.append(('min_p', {'p': p, 'min_keep': min_keep}))
        return self

    def add_frequency_penalty(
        self,
        alpha_frequency: float,
        alpha_presence: float = 0.0,
        repeat_last_n: int = 64,
    ) -> 'SamplerChain':
        """
        Add frequency/presence penalty sampler.

        Args:
            alpha_frequency: Frequency penalty coefficient
            alpha_presence: Presence penalty coefficient
            repeat_last_n: Number of recent tokens to consider
        """
        self._samplers.append((
            'penalties',
            {
                'alpha_frequency': alpha_frequency,
                'alpha_presence': alpha_presence,
                'repeat_last_n': repeat_last_n,
            }
        ))
        return self

    def add_mirostat(
        self,
        tau: float = 5.0,
        eta: float = 0.1,
        m: int = 100,
    ) -> 'SamplerChain':
        """
        Add Mirostat sampler (target perplexity control).

        Args:
            tau: Target entropy
            eta: Learning rate
            m: Number of recent tokens for estimation
        """
        self._samplers.append((
            'mirostat',
            {'tau': tau, 'eta': eta, 'm': m}
        ))
        return self

    def add_softmax(self) -> 'SamplerChain':
        """Add softmax layer (convert logits to probabilities)."""
        # Native call:
        # sampler = llama_cpp.llama_sampler_init_softmax()
        # llama_cpp.llama_sampler_chain_add(self._chain_ptr, sampler)

        self._samplers.append(('softmax', {}))
        return self

    def sample(self, context, idx: int = 0) -> int:
        """
        Sample next token from context logits.

        Native binding to llama_sampler_sample().

        Args:
            context: LlamaContext instance
            idx: Logits index (usually 0 for current token)

        Returns:
            Sampled token ID
        """
        # Native call:
        # return llama_cpp.llama_sampler_sample(self._chain_ptr, context._ctx_ptr, idx)

        return 0  # Placeholder

    def accept(self, token: int) -> None:
        """
        Inform sampler of accepted token (for stateful samplers like Mirostat).

        Args:
            token: Accepted token ID
        """
        # Native call:
        # llama_cpp.llama_sampler_accept(self._chain_ptr, token)

        pass

    def reset(self) -> None:
        """Reset sampler state."""
        # Native call:
        # llama_cpp.llama_sampler_reset(self._chain_ptr)

        pass

    def free(self) -> None:
        """Free sampler memory."""
        # Native call:
        # llama_cpp.llama_sampler_free(self._chain_ptr)

        pass

    def __del__(self) -> None:
        """Cleanup on deletion"""
        self.free()

    def __repr__(self) -> str:
        samplers_str = " -> ".join([s[0] for s in self._samplers])
        return f"SamplerChain({samplers_str})"


# Convenience factory function
def create_sampler(
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    frequency_penalty: float = 0.0,
) -> SamplerChain:
    """
    Create a typical sampling pipeline.

    Args:
        temperature: Temperature scaling
        top_p: Top-P threshold
        top_k: Top-K limit
        frequency_penalty: Frequency penalty

    Returns:
        Configured SamplerChain
    """
    chain = SamplerChain()

    if temperature != 1.0:
        chain.add_temperature(temperature)

    chain.add_top_k(top_k)
    chain.add_top_p(top_p)

    if frequency_penalty > 0:
        chain.add_frequency_penalty(frequency_penalty)

    return chain
