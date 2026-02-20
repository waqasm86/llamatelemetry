"""
llamatelemetry._internal._setup_paths - LD_LIBRARY_PATH + binary discovery.

This module is imported for its side-effects at package load time.
Extracted from the legacy __init__.py bootstrap block.
"""

import logging
import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # llamatelemetry/
_BIN_DIR = _PACKAGE_DIR / "binaries" / "cuda12"
_LIB_DIR = _PACKAGE_DIR / "lib"
_MODEL_CACHE = _PACKAGE_DIR / "models"

# Ensure model cache exists
_MODEL_CACHE.mkdir(parents=True, exist_ok=True)


def _add_to_ld_path(lib_dir: Path) -> None:
    lib_str = str(lib_dir.absolute())
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_str not in current:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_str}:{current}" if current else lib_str
        )


# 1. Set LD_LIBRARY_PATH for bundled shared libs
if _LIB_DIR.exists():
    _add_to_ld_path(_LIB_DIR)
    logging.debug("llamatelemetry: LD_LIBRARY_PATH includes %s", _LIB_DIR)

def _should_bootstrap() -> bool:
    if os.getenv("LLAMATELEMETRY_DISABLE_BOOTSTRAP", "").lower() in {"1", "true", "yes"}:
        return False
    if os.getenv("LLAMATELEMETRY_BOOTSTRAP", "").lower() in {"1", "true", "yes"}:
        return True
    # Auto-bootstrap only in notebook environments
    if os.path.exists("/kaggle") or "COLAB_GPU" in os.environ:
        return True
    return False


# 2. Set LLAMA_SERVER_PATH
_LLAMA_SERVER = _BIN_DIR / "llama-server"
if _LLAMA_SERVER.exists():
    os.environ["LLAMA_SERVER_PATH"] = str(_LLAMA_SERVER.absolute())
    if not os.access(_LLAMA_SERVER, os.X_OK):
        try:
            os.chmod(_LLAMA_SERVER, 0o755)
        except Exception:
            pass
else:
    # Hybrid bootstrap: download on first import
    if _should_bootstrap():
        try:
            from .bootstrap import bootstrap

            bootstrap()

            # Re-apply after bootstrap
            if _LIB_DIR.exists():
                _add_to_ld_path(_LIB_DIR)
            if _LLAMA_SERVER.exists():
                os.environ["LLAMA_SERVER_PATH"] = str(_LLAMA_SERVER.absolute())
                if not os.access(_LLAMA_SERVER, os.X_OK):
                    os.chmod(_LLAMA_SERVER, 0o755)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"llamatelemetry bootstrap failed: {exc}\n"
                "Some features may not work. Set LLAMATELEMETRY_DISABLE_BOOTSTRAP=1 to suppress.",
                RuntimeWarning,
                stacklevel=2,
            )

# 3. Fallback search for lib/ and llama-server
if not _LIB_DIR.exists():
    for candidate in [
        _PACKAGE_DIR / "lib",
        _PACKAGE_DIR.parent / "lib",
    ]:
        if candidate.exists():
            _add_to_ld_path(candidate)
            break

if not _LLAMA_SERVER.exists():
    for candidate in [
        Path("/usr/local/bin/llama-server"),
        Path("/usr/bin/llama-server"),
        Path.home() / ".cache/llamatelemetry/llama-server",
    ]:
        if candidate.exists():
            os.environ["LLAMA_SERVER_PATH"] = str(candidate.absolute())
            if not os.access(candidate, os.X_OK):
                try:
                    os.chmod(candidate, 0o755)
                except Exception:
                    pass
            break
