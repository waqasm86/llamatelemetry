# Notebook 15: Real-Time Inference Monitoring & Performance Analysis

**Live Performance Dashboards with llama.cpp Metrics + Plotly**

---

## Objectives Demonstrated

✅ **CUDA Inference** (GPU 0) - Continuous inference workload
✅ **LLM Observability** (GPU 0) - llama.cpp /metrics endpoint + CUDA monitoring
✅ **Visualizations** (GPU 1) - Real-time Plotly dashboards with live updates

---

## Overview

This notebook demonstrates **real-time performance monitoring** of LLM inference by continuously polling llama.cpp's built-in `/metrics` endpoint and NVIDIA's GPU metrics, then visualizing them as live-updating Plotly dashboards on GPU 1.

**What You'll Learn:**
- Access llama.cpp's Prometheus `/metrics` endpoint
- Monitor GPU utilization with `nvidia-smi` and `pynvml`
- Poll llama.cpp `/slots` endpoint for request queue monitoring
- Create live-updating Plotly dashboards with `plotly.graph_objects.FigureWidget`
- Identify performance bottlenecks and optimization opportunities
- Benchmark different configurations (batch size, context length, etc.)

**Time:** 30 minutes
**Difficulty:** Intermediate-Advanced
**VRAM:** GPU 0: 5-8 GB, GPU 1: 1-2 GB

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NOTEBOOK 15 ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0: Tesla T4 (15GB VRAM)                                    │
│  ├─ llama.cpp llama-server (--metrics enabled)                  │
│  ├─ GGUF Model: Qwen2.5-3B-Q4_K_M                               │
│  ├─ Continuous Inference Workload:                              │
│  │  ├─ Multiple concurrent requests                             │
│  │  ├─ Variable batch sizes                                     │
│  │  └─ Different context lengths                                │
│  └─ Metrics Endpoints:                                           │
│     ├─ /metrics (Prometheus format)                             │
│     ├─ /slots (request queue state)                             │
│     └─ /health (server health check)                            │
│                                                                  │
│  GPU 0 Monitoring:                                               │
│  ├─ NVIDIA-SMI (GPU utilization, memory, temperature)           │
│  └─ PyNVML (programmatic GPU metrics)                           │
│                                                                  │
│  GPU 1: Tesla T4 (15GB VRAM)                                    │
│  ├─ Live Plotly Dashboards:                                     │
│  │  ├─ Request Throughput (requests/sec)                        │
│  │  ├─ Token Generation Rate (tokens/sec)                       │
│  │  ├─ Latency Distribution (P50, P95, P99)                     │
│  │  ├─ GPU Utilization (% over time)                            │
│  │  ├─ GPU Memory Usage (MB over time)                          │
│  │  ├─ Queue Depth (pending requests)                           │
│  │  └─ Temperature & Power Draw                                 │
│  └─ Data Processing: Pandas + NumPy                             │
│                                                                  │
│  Polling Loop (Background Thread):                              │
│  ├─ Poll /metrics every 1 second                                │
│  ├─ Poll /slots every 0.5 seconds                               │
│  ├─ Query nvidia-smi every 2 seconds                            │
│  └─ Update Plotly dashboards in real-time                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notebook Structure

### **Part 1: Setup & Dependencies (3 min)**

**Cell 1-3: Installation**
```python
# Cell 1: Install core packages
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Cell 2: Install monitoring packages
!pip install -q plotly pandas numpy pynvml requests

# Cell 3: Verify GPU setup
!nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

---

### **Part 2: Start Instrumented Server (5 min)**

**Cell 4-5: Download Model and Start Server**
```python
# Cell 4: Download GGUF model
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/Qwen2.5-3B-Instruct-GGUF",
    filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)

# Cell 5: Start server with metrics enabled
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llamatelemetry.server import ServerManager

server = ServerManager(server_url="http://127.0.0.1:8090")

# Start with metrics enabled
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # GPU 0 only
    flash_attn=1,
    port=8090,
    host="127.0.0.1",
    # Enable metrics endpoint
    extra_args=["--metrics"],
)

print("✅ Server started with metrics enabled!")
```

---

### **Part 3: Metrics Collection Infrastructure (7 min)**

**Cell 6: Define Metrics Collector**
```python
import requests
import time
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import threading
import pandas as pd
import numpy as np

class LlamaMetricsCollector:
    """Collects metrics from llama.cpp server endpoints"""

    def __init__(self, base_url: str = "http://127.0.0.1:8090"):
        self.base_url = base_url
        self.metrics_history = defaultdict(list)
        self.slots_history = []
        self.gpu_metrics_history = []
        self.timestamps = []
        self.running = False
        self.lock = threading.Lock()

    def parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """Parse Prometheus-format metrics from /metrics endpoint"""
        metrics = {}

        # Parse metric lines (format: metric_name{labels} value)
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue

            # Simple parsing (handles metrics without labels)
            match = re.match(r"(\w+)\s+([\d.]+)", line)
            if match:
                metric_name, value = match.groups()
                metrics[metric_name] = float(value)

        return metrics

    def fetch_server_metrics(self) -> Dict[str, float]:
        """Fetch metrics from /metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=2)
            if response.status_code == 200:
                return self.parse_prometheus_metrics(response.text)
        except Exception as e:
            print(f"Error fetching metrics: {e}")
        return {}

    def fetch_slots_info(self) -> List[Dict]:
        """Fetch slot information from /slots endpoint"""
        try:
            response = requests.get(f"{self.base_url}/slots", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching slots: {e}")
        return []

    def fetch_gpu_metrics(self) -> Dict[str, float]:
        """Fetch GPU metrics using pynvml"""
        try:
            import pynvml

            # Initialize NVML (if not already done)
            try:
                pynvml.nvmlInit()
            except:
                pass

            # Get GPU 0 handle
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Query metrics
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W

            return {
                "gpu_utilization": utilization.gpu,  # %
                "memory_utilization": utilization.memory,  # %
                "memory_used_mb": memory_info.used / 1024**2,  # bytes to MB
                "memory_total_mb": memory_info.total / 1024**2,
                "temperature_c": temperature,
                "power_draw_w": power_draw,
            }
        except Exception as e:
            print(f"Error fetching GPU metrics: {e}")
            return {}

    def collect_once(self):
        """Collect all metrics at current timestamp"""
        timestamp = time.time()

        # Fetch from all sources
        server_metrics = self.fetch_server_metrics()
        slots_info = self.fetch_slots_info()
        gpu_metrics = self.fetch_gpu_metrics()

        # Store with lock
        with self.lock:
            self.timestamps.append(timestamp)

            # Store server metrics
            for key, value in server_metrics.items():
                self.metrics_history[key].append(value)

            # Store slots info
            self.slots_history.append({
                "timestamp": timestamp,
                "slots": slots_info,
                "num_processing": sum(1 for s in slots_info if s.get("is_processing", False)),
                "num_idle": sum(1 for s in slots_info if not s.get("is_processing", False)),
            })

            # Store GPU metrics
            gpu_record = {"timestamp": timestamp, **gpu_metrics}
            self.gpu_metrics_history.append(gpu_record)

    def start_background_collection(self, interval: float = 1.0):
        """Start background thread for continuous collection"""
        self.running = True

        def collect_loop():
            while self.running:
                self.collect_once()
                time.sleep(interval)

        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()
        print(f"📊 Started metrics collection (interval={interval}s)")

    def stop_background_collection(self):
        """Stop background collection"""
        self.running = False
        print("⏹️ Stopped metrics collection")

    def get_dataframe(self, metric_name: str) -> pd.DataFrame:
        """Get metric history as pandas DataFrame"""
        with self.lock:
            if metric_name not in self.metrics_history:
                return pd.DataFrame()

            return pd.DataFrame({
                "timestamp": pd.to_datetime(self.timestamps, unit="s"),
                "value": self.metrics_history[metric_name],
            })

    def get_gpu_dataframe(self) -> pd.DataFrame:
        """Get GPU metrics history as DataFrame"""
        with self.lock:
            if not self.gpu_metrics_history:
                return pd.DataFrame()
            return pd.DataFrame(self.gpu_metrics_history)

# Initialize collector
collector = LlamaMetricsCollector()
print("✅ Metrics collector initialized")
```

**Cell 7: Test Metrics Collection**
```python
# Test single collection
collector.collect_once()

print("\n📊 Server Metrics:")
for key in list(collector.metrics_history.keys())[:10]:
    print(f"  {key}: {collector.metrics_history[key][-1]}")

print("\n🎰 Slots Info:")
if collector.slots_history:
    latest = collector.slots_history[-1]
    print(f"  Processing: {latest['num_processing']}")
    print(f"  Idle: {latest['num_idle']}")

print("\n🖥️ GPU Metrics:")
if collector.gpu_metrics_history:
    latest = collector.gpu_metrics_history[-1]
    for key, value in latest.items():
        if key != "timestamp":
            print(f"  {key}: {value:.2f}")
```

**Cell 8: Start Background Collection**
```python
# Start collecting metrics in background
collector.start_background_collection(interval=1.0)

# Let it collect for a few seconds
time.sleep(5)

print(f"📈 Collected {len(collector.timestamps)} data points")
```

---

### **Part 4: Generate Continuous Inference Load (5 min)**

**Cell 9-10: Create Load Generator**
```python
# Cell 9: Define load generator
from llamatelemetry.api import LlamaCppClient
import random
import threading

class InferenceLoadGenerator:
    """Generates continuous inference requests"""

    def __init__(self, base_url: str, prompts: List[str]):
        self.client = LlamaCppClient(base_url)
        self.prompts = prompts
        self.running = False
        self.request_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def generate_request(self):
        """Generate single inference request"""
        try:
            prompt = random.choice(self.prompts)
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=random.randint(50, 150),
                temperature=random.uniform(0.5, 0.9),
            )

            with self.lock:
                self.request_count += 1

            return response

        except Exception as e:
            with self.lock:
                self.error_count += 1
            print(f"❌ Request error: {e}")
            return None

    def start_continuous_load(self, qps: float = 2.0):
        """Start generating continuous load at specified QPS"""
        self.running = True

        def load_loop():
            interval = 1.0 / qps
            while self.running:
                self.generate_request()
                time.sleep(interval)

        thread = threading.Thread(target=load_loop, daemon=True)
        thread.start()
        print(f"🚀 Started load generation (QPS={qps})")

    def stop_continuous_load(self):
        """Stop load generation"""
        self.running = False
        print(f"⏹️ Stopped load generation (Total: {self.request_count}, Errors: {self.error_count})")

# Define test prompts
test_prompts = [
    "Explain how CUDA kernels work",
    "What is quantization in neural networks?",
    "Describe the transformer architecture",
    "How does attention mechanism work?",
    "What are the benefits of GGUF format?",
    "Explain FlashAttention optimization",
    "What is tensor parallelism?",
    "How does KV cache improve inference?",
    "Describe NCCL in distributed training",
    "What is mixed precision training?",
]

# Initialize load generator
load_gen = InferenceLoadGenerator("http://127.0.0.1:8090", test_prompts)

# Cell 10: Start generating load
load_gen.start_continuous_load(qps=2.0)  # 2 requests per second

# Let it run for a bit
time.sleep(10)

print(f"📊 Requests sent: {load_gen.request_count}")
print(f"❌ Errors: {load_gen.error_count}")
```

---

### **Part 5: Live Plotly Dashboards (GPU 1) (10 min)**

**Cell 11-12: Switch to GPU 1 and Create Dashboard**
```python
# Cell 11: Switch to GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Cell 12: Create live dashboard with Plotly FigureWidget
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

# Create subplots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "Token Generation Rate (tokens/sec)",
        "GPU Utilization (%)",
        "Request Processing Time (ms)",
        "GPU Memory Usage (MB)",
        "Active Slots",
        "GPU Temperature (°C) & Power (W)"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.15,
)

# Initialize traces
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Tokens/sec", line=dict(color="green")), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="GPU %", line=dict(color="blue")), row=1, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Latency", line=dict(color="orange")), row=2, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Memory MB", line=dict(color="red")), row=2, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Active", line=dict(color="purple")), row=3, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Temp °C", line=dict(color="darkred")), row=3, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Power W", line=dict(color="darkorange")), row=3, col=2)

# Configure layout
fig.update_layout(
    title_text="🔴 LIVE LLM Performance Dashboard",
    showlegend=True,
    height=900,
)

# Create FigureWidget for live updates
fig_widget = go.FigureWidget(fig)
display(fig_widget)
```

**Cell 13: Dashboard Update Loop**
```python
import time
from datetime import datetime

def update_dashboard():
    """Update dashboard with latest metrics"""

    # Get GPU metrics
    df_gpu = collector.get_gpu_dataframe()
    if not df_gpu.empty:
        timestamps = pd.to_datetime(df_gpu["timestamp"], unit="s")

        # Update GPU utilization
        with fig_widget.batch_update():
            fig_widget.data[1].x = timestamps
            fig_widget.data[1].y = df_gpu["gpu_utilization"]

            # Update GPU memory
            fig_widget.data[3].x = timestamps
            fig_widget.data[3].y = df_gpu["memory_used_mb"]

            # Update temperature and power
            fig_widget.data[5].x = timestamps
            fig_widget.data[5].y = df_gpu["temperature_c"]
            fig_widget.data[6].x = timestamps
            fig_widget.data[6].y = df_gpu["power_draw_w"]

    # Get server metrics
    if "llamacpp:predicted_tokens_seconds" in collector.metrics_history:
        df_tokens = collector.get_dataframe("llamacpp:predicted_tokens_seconds")
        if not df_tokens.empty:
            with fig_widget.batch_update():
                fig_widget.data[0].x = df_tokens["timestamp"]
                fig_widget.data[0].y = df_tokens["value"]

    # Get slots info
    if collector.slots_history:
        slots_times = [pd.Timestamp(s["timestamp"], unit="s") for s in collector.slots_history]
        slots_active = [s["num_processing"] for s in collector.slots_history]

        with fig_widget.batch_update():
            fig_widget.data[4].x = slots_times
            fig_widget.data[4].y = slots_active

# Update every 2 seconds
print("🔄 Starting live dashboard updates...")
for i in range(30):  # Update 30 times
    update_dashboard()
    time.sleep(2)

print("✅ Dashboard updates complete")
```

---

### **Part 6: Performance Analysis (Optional, 5 min)**

**Cell 14-15: Statistical Analysis**
```python
# Cell 14: Calculate performance statistics
df_gpu = collector.get_gpu_dataframe()

if not df_gpu.empty:
    print("📊 Performance Statistics\n")

    print("GPU Utilization:")
    print(f"  Mean: {df_gpu['gpu_utilization'].mean():.2f}%")
    print(f"  P50:  {df_gpu['gpu_utilization'].quantile(0.50):.2f}%")
    print(f"  P95:  {df_gpu['gpu_utilization'].quantile(0.95):.2f}%")
    print(f"  Max:  {df_gpu['gpu_utilization'].max():.2f}%")

    print("\nGPU Memory:")
    print(f"  Mean: {df_gpu['memory_used_mb'].mean():.2f} MB")
    print(f"  Max:  {df_gpu['memory_used_mb'].max():.2f} MB")

    print("\nTemperature:")
    print(f"  Mean: {df_gpu['temperature_c'].mean():.2f}°C")
    print(f"  Max:  {df_gpu['temperature_c'].max():.2f}°C")

# Cell 15: Request statistics
print(f"\n🚀 Load Generator Statistics:")
print(f"  Total Requests: {load_gen.request_count}")
print(f"  Errors: {load_gen.error_count}")
print(f"  Success Rate: {(1 - load_gen.error_count / max(load_gen.request_count, 1)) * 100:.2f}%")
```

---

### **Part 7: Cleanup**

**Cell 16: Stop Everything**
```python
# Stop load generation
load_gen.stop_continuous_load()

# Stop metrics collection
collector.stop_background_collection()

# Stop server
server.stop_server()

print("✅ Cleanup complete!")
```

---

## Key Learnings

### **1. llama.cpp Metrics**
- ✅ `/metrics` endpoint provides Prometheus-format metrics
- ✅ Token generation throughput (tokens/second)
- ✅ Request processing statistics
- ✅ Cache hit rates

### **2. GPU Monitoring**
- ✅ PyNVML for programmatic GPU metrics access
- ✅ Utilization, memory, temperature, power draw
- ✅ Real-time monitoring at 1-second intervals

### **3. Request Queue Monitoring**
- ✅ `/slots` endpoint shows request queue state
- ✅ Number of processing vs idle slots
- ✅ Per-slot token generation progress

### **4. Live Visualization**
- ✅ Plotly FigureWidget for real-time updates
- ✅ Multi-panel dashboards with synchronized timelines
- ✅ Efficient batch updates for smooth rendering

### **5. Performance Analysis**
- ✅ Identify bottlenecks (GPU, memory, queue depth)
- ✅ Optimize batch size and concurrency
- ✅ Monitor thermal throttling and power limits

---

## Next Steps

- **Notebook 16:** End-to-end production observability stack
- Export metrics to Prometheus/Grafana
- Set up alerting for performance degradation
- Implement auto-scaling based on queue depth
- A/B test different model configurations

---

**🎯 Objectives Achieved:**
✅ CUDA Inference (GPU 0) - Continuous workload
✅ LLM Observability (GPU 0) - Full metrics collection
✅ Plotly Visualizations (GPU 1) - Live dashboards
