"""
llamatelemetry.otel_gen_ai.gpu_monitor - GPU metrics monitoring

Provides GPU-side metrics collection using pynvml.
All monitoring happens on GPU to avoid CPU overhead.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    GPU metrics collection via pynvml.

    Monitors:
      - Memory utilization (used/total)
      - GPU utilization (SM activity)
      - Temperature
      - Power consumption (if available)
      - Compute capability and device properties
    """

    def __init__(self, meter=None, device_ids: Optional[List[int]] = None):
        """
        Initialize GPU monitor.

        Args:
            meter: OpenTelemetry Meter (optional)
            device_ids: List of GPU device IDs to monitor (None = all)
        """
        self.meter = meter
        self.device_ids = device_ids
        self._initialized = False
        self._pynvml = None

        self._init_pynvml()

    def _init_pynvml(self) -> None:
        """Initialize pynvml library."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml

            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"GPU Monitor initialized: {device_count} devices")

            if self.device_ids is None:
                self.device_ids = list(range(device_count))

            self._initialized = True

        except Exception as e:
            logger.warning(f"Failed to initialize pynvml: {e}")
            logger.warning("GPU monitoring disabled")
            self._initialized = False

    def get_device_count(self) -> int:
        """Get number of CUDA devices."""
        if not self._initialized:
            return 0
        return self._pynvml.nvmlDeviceGetCount()

    def get_device_properties(self, device_id: int = 0) -> Dict:
        """
        Get GPU device properties.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with device properties
        """
        if not self._initialized:
            return {}

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = self._pynvml.nvmlDeviceGetName(handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

            return {
                'device_id': device_id,
                'name': name,
                'memory_total_mb': memory.total // (1024 * 1024),
                'memory_free_mb': memory.free // (1024 * 1024),
                'memory_used_mb': memory.used // (1024 * 1024),
            }
        except Exception as e:
            logger.warning(f"Failed to get device properties: {e}")
            return {}

    def get_memory_info(self, device_id: int = 0) -> Dict:
        """
        Get GPU memory information.

        Args:
            device_id: GPU device ID

        Returns:
            {'used_mb': int, 'total_mb': int, 'free_mb': int, 'utilization': float}
        """
        if not self._initialized:
            return {}

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_mb = memory.total // (1024 * 1024)
            used_mb = memory.used // (1024 * 1024)
            free_mb = memory.free // (1024 * 1024)
            utilization = (used_mb / total_mb) if total_mb > 0 else 0.0

            return {
                'used_mb': used_mb,
                'total_mb': total_mb,
                'free_mb': free_mb,
                'utilization': utilization,
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {}

    def get_utilization(self, device_id: int = 0) -> Dict:
        """
        Get GPU utilization.

        Args:
            device_id: GPU device ID

        Returns:
            {'gpu_util': float, 'memory_util': float}
        """
        if not self._initialized:
            return {}

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)

            return {
                'gpu_util': util.gpu / 100.0,  # 0.0-1.0
                'memory_util': util.memory / 100.0,
            }
        except Exception as e:
            logger.warning(f"Failed to get utilization: {e}")
            return {}

    def get_temperature(self, device_id: int = 0) -> Optional[float]:
        """
        Get GPU temperature in Celsius.

        Args:
            device_id: GPU device ID

        Returns:
            Temperature in Celsius, or None
        """
        if not self._initialized:
            return None

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            temp = self._pynvml.nvmlDeviceGetTemperature(handle, 0)
            return float(temp)
        except Exception as e:
            logger.debug(f"Failed to get temperature: {e}")
            return None

    def get_power_usage(self, device_id: int = 0) -> Optional[float]:
        """
        Get GPU power usage in Watts (if available).

        Args:
            device_id: GPU device ID

        Returns:
            Power in Watts, or None
        """
        if not self._initialized:
            return None

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_id)
            power_mw = self._pynvml.nvmlDeviceGetPowerUsage(handle)
            return float(power_mw) / 1000.0
        except Exception as e:
            logger.debug(f"Failed to get power usage: {e}")
            return None

    def get_all_metrics(self, device_ids: Optional[List[int]] = None) -> Dict:
        """
        Get all metrics for specified devices.

        Args:
            device_ids: List of device IDs (None = all)

        Returns:
            Dictionary with metrics per device
        """
        device_ids = device_ids or self.device_ids
        metrics = {}

        for dev_id in device_ids:
            metrics[dev_id] = {
                'properties': self.get_device_properties(dev_id),
                'memory': self.get_memory_info(dev_id),
                'utilization': self.get_utilization(dev_id),
                'temperature': self.get_temperature(dev_id),
                'power': self.get_power_usage(dev_id),
            }

        return metrics

    def record_metrics_to_otel(self) -> None:
        """Record GPU metrics to OpenTelemetry meters."""
        if not self.meter or not self._initialized:
            return

        try:
            # Create gauges if needed
            gpu_memory_gauge = self.meter.create_gauge(
                "gpu.memory.used",
                unit="By",
                description="GPU memory used in bytes",
            )

            gpu_utilization_gauge = self.meter.create_gauge(
                "gpu.utilization",
                unit="1",
                description="GPU utilization (0-1)",
            )

            # Record per device
            for device_id in self.device_ids:
                memory_info = self.get_memory_info(device_id)
                util_info = self.get_utilization(device_id)

                if memory_info:
                    gpu_memory_gauge.record(
                        memory_info['used_mb'] * (1024 * 1024),
                        {'device_id': str(device_id)},
                    )

                if util_info:
                    gpu_utilization_gauge.record(
                        util_info['gpu_util'],
                        {'device_id': str(device_id)},
                    )

        except Exception as e:
            logger.warning(f"Failed to record OTel metrics: {e}")

    def __repr__(self) -> str:
        if self._initialized:
            return f"GPUMonitor(devices={self.device_ids})"
        else:
            return "GPUMonitor(disabled)"
