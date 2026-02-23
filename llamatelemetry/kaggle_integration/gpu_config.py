"""
llamatelemetry.kaggle_integration.gpu_config - Kaggle GPU configuration

Automatic detection and configuration of dual Tesla T4 GPUs in Kaggle environment.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KaggleGPUConfig:
    """
    Kaggle dual Tesla T4 GPU configuration.

    Handles:
      - GPU detection (dual T4s)
      - Memory allocation (15 GB per T4)
      - Layer splitting across GPUs
      - Optimal batch sizes
    """

    # Tesla T4 specifications
    T4_VRAM_GB = 15
    T4_COMPUTE_CAPABILITY = 75  # SM 7.5
    T4_BANDWIDTH_GBS = 300  # ~300 GB/s

    # Model-specific optimizations
    MODEL_CONFIGS = {
        # Model name: (n_layers, optimal_batch_size, split_mode)
        "llama-2-7b": (32, 512, "layer"),
        "llama-2-13b": (40, 512, "layer"),
        "llama-2-70b": (80, 256, "row"),  # Tensor parallelism
        "llama-3-8b": (32, 512, "layer"),
        "llama-3-70b": (80, 256, "row"),
        "mistral-7b": (32, 512, "layer"),
    }

    def __init__(self):
        """Initialize Kaggle GPU configuration."""
        logger.info("Detecting Kaggle GPU configuration...")

        self.device_count = self._detect_gpus()
        self.gpu_properties = self._get_gpu_properties()
        self.total_vram_gb = self._get_total_vram()

        logger.info(f"Detected {self.device_count} T4 GPUs")
        logger.info(f"Total VRAM: {self.total_vram_gb} GB")

    def _detect_gpus(self) -> int:
        """Detect number of GPUs."""
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return count
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return 0

    def _get_gpu_properties(self) -> List[Dict]:
        """Get properties of each GPU."""
        properties = []

        try:
            import pynvml
            pynvml.nvmlInit()

            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                properties.append({
                    'device_id': i,
                    'name': name,
                    'vram_mb': memory.total // (1024 * 1024),
                    'compute_capability': self.T4_COMPUTE_CAPABILITY,
                })

            pynvml.nvmlShutdown()

        except Exception as e:
            logger.warning(f"Failed to get GPU properties: {e}")

        return properties

    def _get_total_vram(self) -> int:
        """Get total VRAM across all GPUs."""
        return sum(p['vram_mb'] for p in self.gpu_properties) // 1024

    def get_layer_split(
        self,
        model_name: str,
        n_layers: int,
    ) -> Dict[int, List[int]]:
        """
        Calculate optimal layer split for dual GPUs.

        Args:
            model_name: Model identifier
            n_layers: Total number of transformer layers

        Returns:
            {gpu_id: [layer_ids]}
        """
        if self.device_count < 2:
            # Single GPU: all layers
            return {0: list(range(n_layers))}

        # Dual GPU: split layers evenly
        layers_per_gpu = n_layers // 2

        return {
            0: list(range(0, layers_per_gpu)),
            1: list(range(layers_per_gpu, n_layers)),
        }

    def get_model_config(self, model_name: str) -> Dict:
        """
        Get optimal configuration for model.

        Args:
            model_name: Model identifier

        Returns:
            {
                'n_gpu_layers': int,
                'n_batch': int,
                'n_ubatch': int,
                'split_mode': str,
                'tensor_split': [float, float],
            }
        """
        if model_name not in self.MODEL_CONFIGS:
            logger.warning(f"No config for {model_name}, using defaults")
            n_layers, batch_size, split_mode = 40, 512, "layer"
        else:
            n_layers, batch_size, split_mode = self.MODEL_CONFIGS[model_name]

        # Determine n_gpu_layers based on available VRAM
        # Rough estimate: ~1-2 GB per layer for 13B Q4_K_M model
        bytes_per_layer = 2 * (1024 ** 3)  # 2 GB per layer estimate
        available_bytes = self.total_vram_gb * (1024 ** 3)
        n_gpu_layers = min(
            n_layers,
            int(available_bytes / bytes_per_layer),
        )

        # Tensor split for dual GPU (equal distribution)
        tensor_split = [0.5, 0.5] if self.device_count >= 2 else [1.0]

        return {
            'n_gpu_layers': n_gpu_layers,
            'n_batch': batch_size,
            'n_ubatch': batch_size,
            'split_mode': split_mode,
            'tensor_split': tensor_split,
            'main_gpu': 0,
        }

    def get_inference_params(self) -> Dict:
        """
        Get optimal inference parameters for Kaggle dual T4s.

        Returns:
            {
                'n_threads': int,
                'n_threads_batch': int,
                'n_ctx': int,
                'n_batch': int,
                'offload_kqv': bool,
            }
        """
        return {
            'n_threads': 4,  # Conservative for shared Kaggle kernel
            'n_threads_batch': 4,
            'n_ctx': 4096,
            'n_batch': 512,
            'offload_kqv': True,  # Offload K/V to GPU for better performance
        }

    def is_dual_gpu(self) -> bool:
        """Check if dual GPU system."""
        return self.device_count >= 2

    def __repr__(self) -> str:
        return (
            f"KaggleGPUConfig("
            f"gpus={self.device_count}, "
            f"vram={self.total_vram_gb}GB)"
        )
