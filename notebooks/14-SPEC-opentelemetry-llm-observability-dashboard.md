# Notebook 14: OpenTelemetry LLM Observability Dashboard

**Comprehensive LLM Request Tracing & Metrics Visualization**

---

## Objectives Demonstrated

✅ **CUDA Inference** (GPU 0) - llama.cpp inference with instrumentation
✅ **LLM Observability** (GPU 0) - Full OpenTelemetry tracing and metrics
✅ **Visualizations** (GPU 1) - Graphistry trace graphs + Plotly metrics dashboards

---

## Overview

This notebook demonstrates **production-grade LLM observability** by instrumenting llamatelemetry inference with OpenTelemetry and visualizing the telemetry data in real-time using GPU-accelerated Graphistry dashboards and Plotly charts.

**What You'll Learn:**
- Instrument LLM inference with OpenTelemetry traces and metrics
- Collect request-level statistics (latency, tokens, errors)
- Export telemetry to OTLP collectors
- Visualize distributed traces as interactive graphs
- Create real-time metrics dashboards with Plotly

**Time:** 35 minutes
**Difficulty:** Advanced
**VRAM:** GPU 0: 5-8 GB, GPU 1: 2-3 GB

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NOTEBOOK 14 ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0: Tesla T4 (15GB VRAM)                                    │
│  ├─ llama.cpp llama-server                                      │
│  ├─ GGUF Model: Qwen2.5-3B-Q4_K_M                               │
│  ├─ OpenTelemetry Instrumentation:                              │
│  │  ├─ TracerProvider (span generation)                         │
│  │  ├─ MeterProvider (metrics collection)                       │
│  │  └─ OTLP Exporter (gRPC/HTTP)                                │
│  └─ LlamaTelemetry Client (instrumented)                        │
│                                                                  │
│  GPU 1: Tesla T4 (15GB VRAM)                                    │
│  ├─ Graphistry Cloud (trace visualization)                      │
│  │  ├─ Trace Span Graph (parent-child relationships)            │
│  │  ├─ Request Flow Diagram (timing waterfall)                  │
│  │  └─ Error Propagation Graph                                  │
│  ├─ Plotly Dashboards (metrics visualization)                   │
│  │  ├─ Latency Histogram                                        │
│  │  ├─ Token Usage Over Time                                    │
│  │  └─ Request Rate Time Series                                 │
│  └─ RAPIDS cuDF (data processing)                               │
│                                                                  │
│  External (Optional):                                            │
│  └─ OTLP Collector (Jaeger, Tempo, DataDog, etc.)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notebook Structure

### **Part 1: Setup & Environment (5 min)**

**Cell 1-3: Install and Import**
```python
# Cell 1: Install llamatelemetry
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Cell 2: Install OpenTelemetry packages
!pip install -q opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation

# Cell 3: Install visualization packages
!pip install -q plotly pandas pygraphistry
```

**Cell 4-5: GPU Configuration**
```python
# Cell 4: Verify dual GPU setup
import os
os.system("nvidia-smi --query-gpu=name,memory.total --format=csv")

# Cell 5: Configure GPU allocation
# GPU 0 for LLM inference
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

### **Part 2: OpenTelemetry Setup (5 min)**

**Cell 6: Configure Resource Attributes**
```python
from opentelemetry.sdk.resources import Resource

# Define service resource with GPU context
resource = Resource.create({
    "service.name": "llamatelemetry-inference",
    "service.version": "0.1.0",
    "service.instance.id": "kaggle-t4-worker-1",
    "deployment.environment": "kaggle-notebook",
    "host.name": "kaggle-gpu-0",
    "gpu.model": "Tesla T4",
    "gpu.memory.total": 15360,  # MB
    "gpu.compute_capability": "7.5",
})
```

**Cell 7: Setup TracerProvider**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Create tracer provider with resource
tracer_provider = TracerProvider(resource=resource)

# Add console exporter for debugging
tracer_provider.add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Optional: Add OTLP exporter for external collector
# tracer_provider.add_span_processor(
#     BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
# )

trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
```

**Cell 8: Setup MeterProvider**
```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)

# Create meter provider
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[
        PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=5000,  # Export every 5 seconds
        )
    ],
)

metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

# Create custom instruments
request_counter = meter.create_counter(
    name="llm.requests.total",
    description="Total number of LLM requests",
    unit="1",
)

latency_histogram = meter.create_histogram(
    name="llm.request.duration",
    description="LLM request latency",
    unit="ms",
)

token_histogram = meter.create_histogram(
    name="llm.tokens.total",
    description="Token usage per request",
    unit="{token}",
)
```

---

### **Part 3: Model Loading & Server Start (5 min)**

**Cell 9-10: Download Model**
```python
# Cell 9: Download GGUF model
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/Qwen2.5-3B-Instruct-GGUF",
    filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)
print(f"Model downloaded: {model_path}")

# Cell 10: Start llama-server on GPU 0
from llamatelemetry.server import ServerManager

server = ServerManager(server_url="http://127.0.0.1:8090")
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # 100% GPU 0
    flash_attn=1,
    port=8090,
    host="127.0.0.1",
)

print("Server ready!")
```

---

### **Part 4: Instrumented Inference (10 min)**

**Cell 11: Create Instrumented LLM Client**
```python
from llamatelemetry.api import LlamaCppClient
from opentelemetry.trace import Status, StatusCode
import time
import requests

class InstrumentedLLMClient:
    """LLM client with OpenTelemetry instrumentation"""

    def __init__(self, base_url: str, tracer, meter):
        self.client = LlamaCppClient(base_url)
        self.tracer = tracer
        self.request_counter = request_counter
        self.latency_histogram = latency_histogram
        self.token_histogram = token_histogram

    def chat_completion(self, messages: list, **kwargs):
        """Instrumented chat completion with full tracing"""

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        max_tokens = kwargs.get("max_tokens", 100)
        temperature = kwargs.get("temperature", 0.7)

        # Start span for this request
        with self.tracer.start_as_current_span(
            name=f"llm.chat.{model}",
            kind=trace.SpanKind.CLIENT,
        ) as span:
            try:
                # Set span attributes
                span.set_attribute("llm.system", "llama.cpp")
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.request.max_tokens", max_tokens)
                span.set_attribute("llm.request.temperature", temperature)
                span.set_attribute("llm.request.messages", len(messages))

                # Record start time
                start_time = time.time()

                # Make request
                response = self.client.chat.completions.create(
                    messages=messages,
                    **kwargs
                )

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Extract response data
                finish_reason = response.choices[0].finish_reason
                content = response.choices[0].message.content

                # Record response attributes
                span.set_attribute("llm.response.finish_reason", finish_reason)
                span.set_attribute("llm.response.length", len(content))

                # Record metrics
                self.request_counter.add(
                    1,
                    attributes={
                        "model": model,
                        "finish_reason": finish_reason,
                        "status": "success",
                    }
                )

                self.latency_histogram.record(
                    latency_ms,
                    attributes={"model": model, "status": "success"}
                )

                # Record token usage if available
                if hasattr(response, 'usage'):
                    input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)

                    span.set_attribute("llm.usage.input_tokens", input_tokens)
                    span.set_attribute("llm.usage.output_tokens", output_tokens)

                    self.token_histogram.record(
                        input_tokens,
                        attributes={"model": model, "token_type": "input"}
                    )
                    self.token_histogram.record(
                        output_tokens,
                        attributes={"model": model, "token_type": "output"}
                    )

                # Set success status
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                # Record error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)

                self.request_counter.add(
                    1,
                    attributes={
                        "model": model,
                        "status": "error",
                        "error_type": type(e).__name__,
                    }
                )

                raise

# Initialize instrumented client
llm = InstrumentedLLMClient("http://127.0.0.1:8090", tracer, meter)
```

**Cell 12-14: Generate Sample Requests**
```python
# Cell 12: Single request example
response = llm.chat_completion(
    messages=[{"role": "user", "content": "What is CUDA?"}],
    max_tokens=100,
    temperature=0.7,
)
print(f"Response: {response.choices[0].message.content}")

# Cell 13: Multiple requests to generate trace data
import random

prompts = [
    "Explain transformer architecture",
    "What is quantization in LLMs?",
    "How does FlashAttention work?",
    "Describe the attention mechanism",
    "What is GGUF format?",
]

responses = []
for i, prompt in enumerate(prompts):
    print(f"Request {i+1}/{len(prompts)}: {prompt[:50]}...")
    resp = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=random.randint(50, 150),
        temperature=random.uniform(0.5, 0.9),
    )
    responses.append(resp)
    time.sleep(0.5)  # Small delay between requests

print(f"\nCompleted {len(responses)} requests")

# Cell 14: Collect telemetry data for visualization
# Export spans to in-memory list for processing
from opentelemetry.sdk.trace.export import InMemorySpanExporter

# Add in-memory exporter to capture spans
memory_exporter = InMemorySpanExporter()
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

# Generate more requests
for i in range(10):
    llm.chat_completion(
        messages=[{"role": "user", "content": f"Test request {i}"}],
        max_tokens=50,
    )

# Retrieve finished spans
finished_spans = memory_exporter.get_finished_spans()
print(f"Captured {len(finished_spans)} spans")
```

---

### **Part 5: Trace Visualization with Graphistry (GPU 1) (5 min)**

**Cell 15-16: Switch to GPU 1 and Setup Graphistry**
```python
# Cell 15: Switch to GPU 1 for visualization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Cell 16: Configure Graphistry
import graphistry
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
graphistry.register(
    api=3,
    username=secrets.get_secret("Graphistry_Username"),
    personal_key_id=secrets.get_secret("Graphistry_Personal_Key_ID"),
    personal_key_secret=secrets.get_secret("Graphistry_Personal_Key_Secret"),
)
print("Graphistry configured!")
```

**Cell 17-18: Transform Spans to Graph Data**
```python
# Cell 17: Extract span data
import cudf

span_data = []
for span in finished_spans:
    span_data.append({
        "span_id": format(span.context.span_id, "016x"),
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "trace_id": format(span.context.trace_id, "032x"),
        "name": span.name,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": (span.end_time - span.start_time) / 1_000_000,  # nanoseconds to ms
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes) if span.attributes else {},
    })

df_spans = cudf.DataFrame(span_data)
print(f"Span DataFrame shape: {df_spans.shape}")
print(df_spans.head())

# Cell 18: Create edges (parent-child relationships)
if len(df_spans) > 0:
    df_edges = df_spans[df_spans["parent_span_id"].notnull()][
        ["parent_span_id", "span_id", "trace_id"]
    ].rename(columns={
        "parent_span_id": "source",
        "span_id": "destination",
    })
else:
    df_edges = cudf.DataFrame(columns=["source", "destination", "trace_id"])
print(f"Edges DataFrame shape: {df_edges.shape}")
```

**Cell 19-20: Create Trace Graph Visualization**
```python
# Cell 19: Build Graphistry visualization
g = graphistry.edges(df_edges, "source", "destination")
g = g.nodes(df_spans, "span_id")
g = g.bind(
    point_title="name",
    point_size="duration_ms",
    point_color="status",
    edge_title="trace_id",
)

# Apply layout
g = g.layout_settings(play=0)
g = g.encode_point_color("status", categorical_mapping={
    "OK": "#4CAF50",      # Green
    "ERROR": "#F44336",   # Red
    "UNSET": "#9E9E9E",   # Gray
}, as_categorical=True)

# Cell 20: Plot trace graph
url = g.plot(render=False)
print(f"🔗 Trace Graph Dashboard: {url}")
```

---

### **Part 6: Metrics Dashboards with Plotly (GPU 1) (5 min)**

**Cell 21-22: Extract Metrics Data**
```python
# Cell 21: Create metrics dataframe from spans
metrics_data = []
for span in finished_spans:
    attrs = span.attributes or {}
    metrics_data.append({
        "timestamp": pd.to_datetime(span.start_time, unit="ns"),
        "duration_ms": (span.end_time - span.start_time) / 1_000_000,
        "model": attrs.get("llm.model", "unknown"),
        "input_tokens": attrs.get("llm.usage.input_tokens", 0),
        "output_tokens": attrs.get("llm.usage.output_tokens", 0),
        "total_tokens": attrs.get("llm.usage.input_tokens", 0) + attrs.get("llm.usage.output_tokens", 0),
        "status": span.status.status_code.name,
    })

df_metrics = pd.DataFrame(metrics_data)
df_metrics = df_metrics.sort_values("timestamp")
print(f"Metrics DataFrame shape: {df_metrics.shape}")
```

**Cell 23-24: Latency Histogram**
```python
# Cell 23-24: Create Plotly visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Request Latency Distribution",
        "Token Usage Over Time",
        "Tokens per Request (Input vs Output)",
        "Request Rate Over Time"
    ),
    specs=[
        [{"type": "histogram"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}]
    ]
)

# 1. Latency histogram
fig.add_trace(
    go.Histogram(
        x=df_metrics["duration_ms"],
        name="Latency (ms)",
        marker_color="blue",
        opacity=0.7,
    ),
    row=1, col=1
)

# 2. Token usage over time
fig.add_trace(
    go.Scatter(
        x=df_metrics["timestamp"],
        y=df_metrics["total_tokens"],
        mode="lines+markers",
        name="Total Tokens",
        line=dict(color="green"),
    ),
    row=1, col=2
)

# 3. Input vs Output tokens
fig.add_trace(
    go.Scatter(
        x=df_metrics["input_tokens"],
        y=df_metrics["output_tokens"],
        mode="markers",
        name="Tokens",
        marker=dict(
            size=df_metrics["duration_ms"] / 10,  # Size by latency
            color=df_metrics["duration_ms"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Latency (ms)", x=1.15),
        ),
        text=df_metrics.apply(
            lambda r: f"Latency: {r['duration_ms']:.1f}ms<br>In: {r['input_tokens']}<br>Out: {r['output_tokens']}",
            axis=1
        ),
        hovertemplate="%{text}<extra></extra>",
    ),
    row=2, col=1
)

# 4. Request rate (cumulative)
df_metrics["request_count"] = range(1, len(df_metrics) + 1)
fig.add_trace(
    go.Scatter(
        x=df_metrics["timestamp"],
        y=df_metrics["request_count"],
        mode="lines",
        name="Cumulative Requests",
        line=dict(color="red"),
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title_text="LLM Observability Metrics Dashboard",
    showlegend=True,
    height=800,
)

fig.show()
```

---

## Key Learnings

### **1. OpenTelemetry Integration**
- ✅ Full instrumentation with traces, metrics, and logs
- ✅ Semantic conventions for GenAI workloads
- ✅ Custom resource attributes for GPU context
- ✅ Flexible export to multiple backends

### **2. Trace Visualization**
- ✅ Parent-child span relationships as interactive graphs
- ✅ Request flow waterfall diagrams
- ✅ Error propagation visualization
- ✅ GPU-accelerated graph analytics with Graphistry

### **3. Metrics Monitoring**
- ✅ Request latency tracking
- ✅ Token usage analysis
- ✅ Throughput monitoring
- ✅ Real-time dashboards with Plotly

### **4. Production Patterns**
- ✅ Context propagation for distributed tracing
- ✅ Batch export for performance
- ✅ Error handling and exception recording
- ✅ Resource attribution for multi-GPU environments

---

## Next Steps

- **Notebook 15:** Real-time performance monitoring with live metrics
- **Notebook 16:** End-to-end production observability stack
- Integrate with external collectors (Jaeger, Tempo, DataDog)
- Add custom span processors for filtering/enrichment
- Implement sampling strategies for high-volume workloads

---

**🎯 Objectives Achieved:**
✅ CUDA Inference (GPU 0)
✅ LLM Observability (GPU 0)
✅ Graphistry + Plotly Visualizations (GPU 1)
