"""
llamatelemetry.llama.gguf - GGUF parsing, metadata, SHA256.

Refactored from api/gguf.py - keeps core parsing, drops CLI wrappers (-> extras).
"""

import hashlib
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

# Re-export the enums and core constants from the original module
from ..api.gguf import (
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType,
    GGMLType,
    QUANT_TYPE_NAMES,
    QUANT_TYPE_INFO,
    QuantTypeInfo,
    GGUFMetadata,
    GGUFTensorInfo,
    GGUFModelInfo,
    read_string,
    read_value,
    parse_gguf_header,
    validate_gguf,
    get_model_summary,
)


def compute_sha256(path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()
