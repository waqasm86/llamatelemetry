"""
llamatelemetry.distributed - Multi-GPU awareness and topology detection.

Provides hardware topology detection, GPU capability assessment, and
multi-GPU mode selection for inference and training.
"""

from .topology import GPUTopology, detect_topology, MultiGPUMode
