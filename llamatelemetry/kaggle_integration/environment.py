"""
llamatelemetry.kaggle_integration.environment - Kaggle environment setup

Configures paths, secrets, and environment variables for Kaggle execution.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class KaggleEnvironment:
    """
    Kaggle environment configuration.

    Handles:
      - Path setup (input, output, working)
      - Secret loading (API keys, tokens)
      - CUDA environment variables
      - OTLP configuration
    """

    KAGGLE_INPUT_DIR = Path("/kaggle/input")
    KAGGLE_WORKING_DIR = Path("/kaggle/working")
    KAGGLE_OUTPUT_DIR = Path("/kaggle/output")
    MODEL_CACHE_DIR = KAGGLE_WORKING_DIR / "models"

    def __init__(self):
        """Initialize Kaggle environment."""
        logger.info("Initializing Kaggle environment...")

        # Create directories
        self.model_cache_dir = self.MODEL_CACHE_DIR
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load environment
        self._setup_cuda_env()
        self._load_secrets()

        logger.info("Kaggle environment ready")

    @staticmethod
    def is_kaggle() -> bool:
        """Check if running in Kaggle environment."""
        return Path("/kaggle").exists()

    def _setup_cuda_env(self) -> None:
        """Setup CUDA environment variables."""
        # Ensure CUDA 12.x is used
        os.environ['CUDA_VERSION'] = '12.2'

        # Disable unnecessary logging
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['NCCL_DEBUG'] = 'INFO'

        logger.info("CUDA environment configured")

    def _load_secrets(self) -> None:
        """Load Kaggle secrets if available."""
        try:
            # Load from Kaggle secrets (if in kernel)
            secrets_dir = Path("/kaggle/input/your-secret-path")  # User configurable

            if secrets_dir.exists():
                logger.info("Loading Kaggle secrets...")
                # User would configure secret loading here
            else:
                logger.debug("No secrets directory found")

        except Exception as e:
            logger.debug(f"Failed to load secrets: {e}")

    def get_otlp_config(self) -> Dict[str, str]:
        """
        Get OTLP exporter configuration.

        Returns:
            {
                'endpoint': str,
                'headers': Dict[str, str],
                'timeout': int,
            }
        """
        # Get from environment or defaults
        endpoint = os.environ.get(
            'OTLP_ENDPOINT',
            'https://otlp.example.com/v1/traces',
        )

        headers = {}
        api_key = os.environ.get('OTLP_API_KEY')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        return {
            'endpoint': endpoint,
            'headers': headers,
            'timeout': int(os.environ.get('OTLP_TIMEOUT', '10')),
        }

    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """
        Get path for model (in working directory).

        Args:
            model_name: Model filename (if None, returns cache directory)

        Returns:
            Path to model file or directory
        """
        if model_name:
            return self.model_cache_dir / model_name

        return self.model_cache_dir

    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """
        Get output file path.

        Args:
            filename: Output filename (if None, returns output directory)

        Returns:
            Path to output file or directory
        """
        if filename:
            return self.KAGGLE_OUTPUT_DIR / filename

        return self.KAGGLE_OUTPUT_DIR

    def setup_llamatelemetry(
        self,
        service_name: str = "kaggle-llm-inference",
        otlp_endpoint: Optional[str] = None,
    ) -> Dict:
        """
        Setup llamatelemetry for Kaggle execution.

        Args:
            service_name: Service name for tracing
            otlp_endpoint: OTLP endpoint URL

        Returns:
            Configuration dictionary
        """
        logger.info(f"Setting up llamatelemetry (service: {service_name})")

        otlp_config = self.get_otlp_config()
        if otlp_endpoint:
            otlp_config['endpoint'] = otlp_endpoint

        config = {
            'service_name': service_name,
            'otlp_endpoint': otlp_config['endpoint'],
            'otlp_headers': otlp_config['headers'],
            'model_cache_dir': str(self.model_cache_dir),
            'output_dir': str(self.KAGGLE_OUTPUT_DIR),
            'is_kaggle': self.is_kaggle(),
        }

        logger.info(f"Configuration: {config}")
        return config

    def __repr__(self) -> str:
        return (
            f"KaggleEnvironment("
            f"models={self.model_cache_dir}, "
            f"output={self.KAGGLE_OUTPUT_DIR})"
        )


# Convenience function for quick setup
def auto_configure() -> KaggleEnvironment:
    """
    Auto-configure Kaggle environment for llamatelemetry.

    Returns:
        Configured KaggleEnvironment instance
    """
    env = KaggleEnvironment()
    logger.info("Auto-configuration complete")
    return env
