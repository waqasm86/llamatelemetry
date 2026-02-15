"""
llamatelemetry.gpu.schemas - GPU dataclasses.

Refactored from api/multigpu.py (GPUInfo, detect_gpus).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GPUDevice:
    """Static information about a single GPU."""

    id: int
    name: str
    memory_total_mb: int
    compute_capability: str
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class GPUSnapshot:
    """Point-in-time GPU utilisation reading."""

    gpu_id: int
    timestamp: float
    utilization_pct: int = 0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    power_w: float = 0.0
    temp_c: int = 0


class GPUSamplerHandle:
    """Handle returned by ``start_sampler()``."""

    def __init__(self, interval_ms: int = 200):
        self._interval = interval_ms / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._snapshots: List[GPUSnapshot] = []
        self._latest: Optional[GPUSnapshot] = None

    def _run(self, query_fn) -> None:
        while self._running:
            try:
                snaps = query_fn()
                with self._lock:
                    self._snapshots.extend(snaps)
                    if snaps:
                        self._latest = snaps[-1]
            except Exception:
                pass
            time.sleep(self._interval)

    def start(self, query_fn) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(query_fn,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_snapshots(self) -> List[GPUSnapshot]:
        with self._lock:
            return list(self._snapshots)

    def get_latest(self) -> Optional[GPUSnapshot]:
        with self._lock:
            return self._latest
