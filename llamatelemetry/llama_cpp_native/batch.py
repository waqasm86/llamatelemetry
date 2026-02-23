"""
llamatelemetry.llama_cpp_native.batch - Token batch management

Direct pybind11 binding to llama_batch operations.
Handles token batching for prefill/decode phases.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class LlamaBatch:
    """
    Native token batch wrapper.

    Directly binds to:
      - llama_batch_init()
      - llama_batch_add()
      - llama_batch_clear()
      - llama_batch_free()
    """

    def __init__(self, size: int, embd_size: int = 0, n_seq_max: int = 1):
        """
        Create batch for token processing.

        Args:
            size: Maximum number of tokens in batch
            embd_size: Embedding size (0 for token-based)
            n_seq_max: Maximum simultaneous sequences
        """
        self.size = size
        self.embd_size = embd_size
        self.n_seq_max = n_seq_max
        self.n_tokens = 0

        logger.debug(
            f"Creating batch: size={size}, embd={embd_size}, seq_max={n_seq_max}"
        )

        # Native pybind11 call:
        # self._batch_ptr = llama_cpp.llama_batch_init(size, embd_size, n_seq_max)

        self._batch_ptr = None  # Placeholder

    def add(
        self,
        token: int,
        pos: int,
        seq_ids: List[int],
        logits: bool = True,
    ) -> None:
        """
        Add token to batch.

        Native binding to llama_batch_add().

        Args:
            token: Token ID
            pos: Position in sequence
            seq_ids: Sequence IDs for this token
            logits: Whether to compute logits for this token
        """
        if self.n_tokens >= self.size:
            raise RuntimeError(f"Batch full (capacity={self.size})")

        # Native call:
        # llama_cpp.llama_batch_add(
        #     self._batch_ptr, token, pos, seq_ids, len(seq_ids), logits
        # )

        self.n_tokens += 1

    def add_embedding(
        self,
        embd: List[float],
        pos: int,
        seq_ids: List[int],
        logits: bool = True,
    ) -> None:
        """
        Add embedding to batch (instead of token).

        Args:
            embd: Embedding vector
            pos: Position in sequence
            seq_ids: Sequence IDs
            logits: Whether to compute logits
        """
        if self.n_tokens >= self.size:
            raise RuntimeError(f"Batch full (capacity={self.size})")

        if not self.embd_size:
            raise RuntimeError("Batch created without embedding support")

        # Native call would handle embedding

        self.n_tokens += 1

    def clear(self) -> None:
        """Clear batch for reuse."""
        # Native call:
        # llama_cpp.llama_batch_clear(self._batch_ptr)

        self.n_tokens = 0

    def is_full(self) -> bool:
        """Check if batch is full"""
        return self.n_tokens >= self.size

    def free(self) -> None:
        """Free batch memory."""
        # Native call:
        # llama_cpp.llama_batch_free(self._batch_ptr)
        pass

    def __del__(self) -> None:
        """Cleanup on deletion"""
        self.free()

    def __repr__(self) -> str:
        return f"LlamaBatch(size={self.size}, tokens={self.n_tokens})"
