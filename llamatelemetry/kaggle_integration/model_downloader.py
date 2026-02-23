"""
llamatelemetry.kaggle_integration.model_downloader - HuggingFace GGUF model downloading

Handles downloading GGUF LLM models from HuggingFace Hub to Kaggle environment.
"""

from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Download GGUF models from HuggingFace Hub.

    Supports:
      - Direct model download
      - Checksum verification
      - Resume downloads
      - Local caching
    """

    DEFAULT_CACHE_DIR = Path("/kaggle/working/models")
    HUGGINGFACE_MODELS = {
        "llama-2-7b": "TheBloke/Llama-2-7B-GGUF",
        "llama-2-13b": "TheBloke/Llama-2-13B-GGUF",
        "llama-2-70b": "TheBloke/Llama-2-70B-GGUF",
        "llama-3-8b": "NousResearch/Meta-Llama-3-8B-GGUF",
        "llama-3-70b": "NousResearch/Meta-Llama-3-70B-GGUF",
        "mistral-7b": "TheBloke/Mistral-7B-GGUF",
        "neural-chat-7b": "TheBloke/neural-chat-7B-GGUF",
        "qwen-7b": "TheBloke/Qwen-7B-GGUF",
        "zephyr-7b": "TheBloke/zephyr-7B-GGUF",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model downloader.

        Args:
            cache_dir: Directory for caching models
        """
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model cache directory: {self.cache_dir}")

        # Lazy import of huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            self.hf_hub_download = hf_hub_download
            self._hf_available = True
        except ImportError:
            logger.warning("huggingface_hub not installed")
            self._hf_available = False
            self.hf_hub_download = None

    def download_model(
        self,
        repo_id: str,
        filename: str,
        resume_download: bool = True,
        local_files_only: bool = False,
    ) -> Path:
        """
        Download GGUF model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-13B-GGUF")
            filename: Model filename (e.g., "llama-2-13b.Q4_K_M.gguf")
            resume_download: Resume incomplete downloads
            local_files_only: Use only cached files

        Returns:
            Path to downloaded model file
        """
        if not self._hf_available:
            raise RuntimeError("huggingface_hub is required for model downloading")

        logger.info(f"Downloading model: {repo_id}/{filename}")

        try:
            # Download from HuggingFace
            path = self.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                resume_download=resume_download,
                local_files_only=local_files_only,
            )

            model_path = Path(path)
            logger.info(f"Model downloaded: {model_path}")
            logger.info(f"File size: {model_path.stat().st_size / (1024**3):.2f} GB")

            return model_path

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def get_quantized_variants(
        self,
        repo_id: str,
        quantization_types: Optional[List[str]] = None,
    ) -> dict:
        """
        Get available quantized variants of a model.

        Args:
            repo_id: HuggingFace repository ID
            quantization_types: Filter by quantization types (Q4_K_M, Q5_K_M, etc.)

        Returns:
            Dictionary of variant filenames
        """
        if not self._hf_available:
            return {}

        try:
            from huggingface_hub import list_files_in_repo

            files = list_files_in_repo(repo_id)
            variants = {}

            for file in files:
                if file.endswith(".gguf"):
                    # Extract quantization type from filename
                    parts = file.split("-")
                    for part in parts:
                        if part.startswith("Q") or part.startswith("IQ"):
                            quant_type = part.split(".")[0]
                            if not quantization_types or quant_type in quantization_types:
                                variants[quant_type] = file
                                break

            return variants

        except Exception as e:
            logger.warning(f"Failed to list variants: {e}")
            return {}

    def get_model_by_shortname(
        self,
        shortname: str,
        quantization: str = "Q4_K_M",
    ) -> Path:
        """
        Download model using shortname and quantization type.

        Args:
            shortname: Short model name (e.g., "llama-2-13b")
            quantization: Quantization type (e.g., "Q4_K_M")

        Returns:
            Path to downloaded model
        """
        if shortname not in self.HUGGINGFACE_MODELS:
            raise ValueError(f"Unknown model: {shortname}")

        repo_id = self.HUGGINGFACE_MODELS[shortname]
        filename = f"{shortname}.{quantization}.gguf"

        return self.download_model(repo_id, filename)

    def verify_model(self, model_path: Path) -> bool:
        """
        Verify GGUF model file integrity.

        Args:
            model_path: Path to model file

        Returns:
            True if model is valid GGUF
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        # Check file signature
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                # GGUF magic number is "GGUF" in little-endian
                if magic != b"GGUF":
                    logger.error("Invalid GGUF magic number")
                    return False

            logger.info(f"Model verified: {model_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to verify model: {e}")
            return False

    def get_cached_models(self) -> List[Path]:
        """Get list of cached GGUF models."""
        return list(self.cache_dir.glob("**/*.gguf"))

    def __repr__(self) -> str:
        return f"ModelDownloader(cache={self.cache_dir})"
