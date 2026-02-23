"""
llamatelemetry.kaggle_integration - Kaggle-specific utilities

Features:
  - Automatic dual T4 GPU detection
  - HuggingFace model downloading
  - Model splitting and memory management
  - Kaggle secrets integration (API keys, tokens)
  - OTLP endpoint configuration
"""

from .model_downloader import ModelDownloader
from .gpu_config import KaggleGPUConfig
from .environment import KaggleEnvironment

__all__ = [
    'ModelDownloader',
    'KaggleGPUConfig',
    'KaggleEnvironment',
]

__version__ = '2.0.0'
