# Notebooks Guide (v2.0.0)

The `notebooks/` directory contains 16 comprehensive Jupyter notebooks covering foundation-to-production observability workflows for Kaggle dual Tesla T4.

---

## Complete Notebook Catalog

### Phase 1: Foundation (Beginner) — 65 minutes

| # | Notebook | Time | Description |
|---|---|---|---|
| 01 | `01-quickstart` | 10 min | Basic inference setup — install, init, first chat |
| 02 | `02-llama-server-setup` | 15 min | Server configuration and optimization |
| 03 | `03-multi-gpu-inference` | 20 min | Dual GPU tensor parallelism |
| 04 | `04-gguf-quantization` | 20 min | 29 quantization types, VRAM estimation |

### Phase 2: Integration (Intermediate) — 60 minutes

| # | Notebook | Time | Description |
|---|---|---|---|
| 05 | `05-unsloth-integration` | 30 min | Fine-tuning → LoRA merge → GGUF export → deploy |
| 06 | `06-split-gpu-graphistry` | 30 min | Concurrent LLM (GPU0) + RAPIDS analytics (GPU1) |

### Phase 3: Advanced Applications — 65 minutes

| # | Notebook | Time | Description |
|---|---|---|---|
| 07 | `07-knowledge-graph-extraction` | 35 min | LLM-powered knowledge graphs with Graphistry |
| 08 | `08-document-network-analysis` | 30 min | Document similarity networks |

### Phase 4: Optimization & Production — 120 minutes

| # | Notebook | Time | Description |
|---|---|---|---|
| 09 | `09-large-models-kaggle` | 35 min | 70B models on dual T4 |
| 10 | `10-complete-workflow` | 45 min | End-to-end production pipeline |
| 11 | `11-gguf-neural-network-visualization` | 40 min | Architecture visualization (929 nodes, 981 edges) |

### Phase 5: Observability Trilogy ⭐ — 120 minutes

| # | Notebook | Time | Description |
|---|---|---|---|
| 12 | `12-gguf-attention-mechanism-explorer` | 25 min | Q-K-V decomposition, 896 attention heads |
| 13 | `13-gguf-token-embedding-visualizer` | 30 min | 3D UMAP embedding space (cuML on GPU1) |
| 14 | `14-opentelemetry-llm-observability` | 45 min | Full OpenTelemetry integration with llamatelemetry |
| 15 | `15-real-time-performance-monitoring` | 30 min | Live Plotly dashboards with llama.cpp metrics |
| 16 | `16-production-observability-stack` | 45 min | Complete stack: Graphistry + Plotly 2D/3D |

---

## Learning Paths

### Path 1: Quick Start (1 hour)
```
01 → 02 → 03
```
**Outcome:** Deploy and run LLM inference on Kaggle T4.

### Path 2: Full Foundation (3 hours)
```
01 → 02 → 03 → 04 → 05 → 06 → 10
```
**Outcome:** Deploy production-ready LLM systems.

### Path 3: Observability Focus ⭐ RECOMMENDED (2.5 hours)
```
01 → 03 → 14 → 15 → 16
```
**Outcome:** Build a complete production observability stack with OTel, live dashboards, and Graphistry.

### Path 4: Graph Analytics (2.5 hours)
```
01 → 03 → 06 → 07 → 08 → 11
```
**Outcome:** Build LLM-powered graph analytics applications.

### Path 5: Large Model Specialist (2 hours)
```
01 → 03 → 04 → 09
```
**Outcome:** Run 70B models on Kaggle dual T4.

### Path 6: Visualization Track (3.5 hours)
```
01 → 03 → 04 → 06 → 11 → 12 → 13
```
**Outcome:** Explore a quantized transformer's internals visually — architecture, attention heads, and embedding space.

---

## Observability Notebooks Comparison (14–16)

| Feature | Notebook 14 | Notebook 15 | Notebook 16 |
|---|---|---|---|
| **Focus** | OTel basics | Real-time monitoring | Full production stack |
| **Complexity** | Intermediate | Intermediate–Advanced | Expert |
| **Time** | 45 min | 30 min | 45 min |
| **OpenTelemetry** | Full | — | Full + Advanced |
| **llama.cpp Metrics** | — | Full | Full |
| **GPU Monitoring** | — | PyNVML | PyNVML |
| **Graphistry** | Basic | — | Advanced |
| **Plotly 2D** | Static | Live updates | Comprehensive |
| **Plotly 3D** | — | — | Model internals |
| **Live Dashboards** | — | FigureWidget | Multi-panel |

---

## Getting Started

1. Set Kaggle accelerator to **GPU T4 × 2**.
2. Install the SDK in notebook cell 1:
   ```python
   !pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v2.0.0
   ```
3. Start with **notebook 01** for a 10-minute quickstart.
4. Follow the **Observability Focus** path (01 → 03 → 14 → 15 → 16) for production workflows.

---

## Additional Resources

- `docs/QUICK_START_GUIDE.md` — Step-by-step first inference
- `docs/KAGGLE_GUIDE.md` — Kaggle-specific optimization tips
- `docs/GGUF_GUIDE.md` — GGUF model selection and quantization
- `docs/TROUBLESHOOTING.md` — Common issues and solutions
- `docs/API_REFERENCE.md` — Full v2.0.0 API reference
