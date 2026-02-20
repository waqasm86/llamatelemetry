"""
llamatelemetry.artifacts.manifest - Model artifact manifest.

Stores metadata about built artifacts: model SHA-256, tokenizer SHA-256,
quantization type, context length, build parameters.
Enables closed-loop correlation between build pipeline and runtime.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class ArtifactManifest:
    """Metadata for a built model artifact.

    Attributes:
        model_sha256: SHA-256 hash of the model file.
        tokenizer_sha256: SHA-256 hash of the tokenizer.
        model_name: Human-readable model name.
        quantization: Quantization type (e.g. Q4_K_M).
        context_length: Maximum context length.
        vocab_size: Vocabulary size.
        model_size_mb: Model file size in MB.
        build_timestamp: When the artifact was built.
        build_params: Build parameters used.
        rope_scaling: RoPE scaling configuration.
        source_model: Source model identifier.
        adapter_name: LoRA adapter name (if applicable).
        llamatelemetry_version: SDK version used for build.
    """

    model_sha256: str = ""
    tokenizer_sha256: str = ""
    model_name: str = ""
    quantization: str = ""
    context_length: int = 0
    vocab_size: int = 0
    model_size_mb: float = 0.0
    build_timestamp: str = ""
    build_params: Dict[str, Any] = field(default_factory=dict)
    rope_scaling: Optional[Dict[str, Any]] = None
    source_model: str = ""
    adapter_name: str = ""
    llamatelemetry_version: str = ""

    def __post_init__(self):
        if not self.build_timestamp:
            self.build_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.llamatelemetry_version:
            try:
                from .._version import __version__
                self.llamatelemetry_version = __version__
            except ImportError:
                self.llamatelemetry_version = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_sha256": self.model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_name": self.model_name,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "model_size_mb": self.model_size_mb,
            "build_timestamp": self.build_timestamp,
            "build_params": self.build_params,
            "rope_scaling": self.rope_scaling,
            "source_model": self.source_model,
            "adapter_name": self.adapter_name,
            "llamatelemetry_version": self.llamatelemetry_version,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save manifest to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ArtifactManifest":
        """Load manifest from JSON file.

        Args:
            path: Input file path.

        Returns:
            ArtifactManifest instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def from_gguf(
        cls,
        gguf_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "ArtifactManifest":
        """Create manifest from a GGUF file.

        Args:
            gguf_path: Path to GGUF model file.
            tokenizer_path: Optional path to tokenizer directory.

        Returns:
            ArtifactManifest with computed hashes and metadata.
        """
        path = Path(gguf_path)
        manifest = cls(**kwargs)

        if path.exists():
            manifest.model_sha256 = _file_sha256(path)
            manifest.model_size_mb = round(path.stat().st_size / (1024 * 1024), 2)

        if tokenizer_path:
            tok_path = Path(tokenizer_path)
            if tok_path.is_dir():
                # Hash the tokenizer.json or tokenizer_config.json
                for name in ["tokenizer.json", "tokenizer_config.json"]:
                    tok_file = tok_path / name
                    if tok_file.exists():
                        manifest.tokenizer_sha256 = _file_sha256(tok_file)
                        break
            elif tok_path.is_file():
                manifest.tokenizer_sha256 = _file_sha256(tok_path)

        return manifest


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
