"""
llamatelemetry.llama_cpp_native.tokenizer - Text tokenization/detokenization

Direct pybind11 binding to llama_tokenize and related APIs.
Thread-safe encoding/decoding operations.
"""

from typing import List, Tuple
import logging

from .model import LlamaModel

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Native tokenizer wrapper.

    Directly binds to:
      - llama_tokenize()
      - llama_token_to_piece()
      - llama_detokenize()
      - llama_token_is_eog()
    """

    def __init__(self, model: LlamaModel):
        """
        Create tokenizer from model.

        Args:
            model: LlamaModel instance
        """
        self.model = model
        self._vocab = None  # Would be set in native init

        logger.debug(f"Tokenizer initialized for {model.metadata.get('ftype')}")

    def encode(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = False,
    ) -> List[int]:
        """
        Tokenize text to token IDs.

        Native binding to llama_tokenize().

        Args:
            text: Input text
            add_special: Add special tokens (BOS, etc.)
            parse_special: Parse special tokens in text

        Returns:
            List of token IDs
        """
        if not text:
            return []

        # Native call:
        # tokens = llama_cpp.llama_tokenize(
        #     self.model._model_ptr.vocab(),
        #     text.encode('utf-8'),
        #     add_special,
        #     parse_special,
        # )
        # return tokens

        # Placeholder: return dummy tokens
        return [101] + [self.model.n_vocab // 2] * len(text.split()) + [102]

    def decode(
        self,
        tokens: List[int],
        remove_special: bool = False,
        unparse_special: bool = False,
    ) -> str:
        """
        Detokenize token IDs to text.

        Native binding to llama_detokenize().

        Args:
            tokens: List of token IDs
            remove_special: Remove special tokens from output
            unparse_special: Convert special tokens back to text

        Returns:
            Decoded text
        """
        if not tokens:
            return ""

        # Native call:
        # text = llama_cpp.llama_detokenize(
        #     self.model._model_ptr.vocab(),
        #     tokens,
        #     remove_special,
        #     unparse_special,
        # )
        # return text

        # Placeholder
        return " ".join(str(t) for t in tokens)

    def token_to_piece(
        self,
        token: int,
        lstrip: int = 0,
        special: bool = False,
    ) -> str:
        """
        Convert single token to text piece.

        Native binding to llama_token_to_piece().

        Args:
            token: Token ID
            lstrip: Left-strip amount
            special: Include special tokens

        Returns:
            Text representation of token
        """
        # Native call:
        # piece = llama_cpp.llama_token_to_piece(
        #     self.model._model_ptr.vocab(),
        #     token,
        #     lstrip,
        #     special,
        # )
        # return piece

        return f"<tok_{token}>"  # Placeholder

    def is_eog(self, token: int) -> bool:
        """
        Check if token is end-of-generation.

        Native binding to llama_token_is_eog().

        Args:
            token: Token ID

        Returns:
            True if EOG token
        """
        # Native call:
        # return llama_cpp.llama_token_is_eog(self.model._model_ptr, token)

        # Common EOG tokens
        common_eog = [2, 32000, 32001]  # Placeholder
        return token in common_eog

    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.model.n_vocab

    def vocab_type(self) -> str:
        """Get vocabulary type (SPM, BPE, etc.)"""
        # Would return vocab type from model metadata
        return "unknown"  # Placeholder

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size()})"
