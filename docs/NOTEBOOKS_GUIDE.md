# Notebooks Guide

The `notebooks/` directory contains 16 comprehensive Jupyter notebooks covering foundation to production-ready observability workflows.

## 📚 Complete Notebook Catalog

### **Phase 1: Foundation (Beginner)** - 65 minutes
1. **01-quickstart** (10 min) - Basic inference setup
2. **02-llama-server-setup** (15 min) - Server configuration and optimization
3. **03-multi-gpu-inference** (20 min) - Dual GPU tensor parallelism
4. **04-gguf-quantization** (20 min) - 29 quantization types, VRAM estimation

### **Phase 2: Integration (Intermediate)** - 60 minutes
5. **05-unsloth-integration** (30 min) - Fine-tuning → GGUF export → deployment
6. **06-split-gpu-graphistry** (30 min) - Concurrent LLM + RAPIDS analytics

### **Phase 3: Advanced Applications** - 65 minutes
7. **07-knowledge-graph-extraction** (35 min) - LLM-powered knowledge graphs
8. **08-document-network-analysis** (30 min) - Document similarity networks

### **Phase 4: Optimization & Production** - 120 minutes
9. **09-large-models-kaggle** (35 min) - 70B models on dual T4
10. **10-complete-workflow** (45 min) - End-to-end production pipeline
11. **11-gguf-neural-network-visualization** (40 min) - Architecture visualization (929 nodes)

### **Phase 5: Observability Trilogy** ⭐ **NEW** - 120 minutes
12. **12-gguf-attention-mechanism-explorer** (25 min) - Q-K-V decomposition, 896 attention heads
13. **13-gguf-token-embedding-visualizer** (30 min) - 3D UMAP embedding space
14. **14-opentelemetry-llm-observability** (45 min) - Full OpenTelemetry integration
15. **15-real-time-performance-monitoring** ⭐ (30 min) - Live Plotly dashboards with llama.cpp metrics
16. **16-production-observability-stack** ⭐ (45 min) - Complete observability stack (Graphistry + Plotly 2D/3D)

---

## 🎯 Learning Paths

### **Path 1: Quick Start** (1 hour)
```
01 → 02 → 03
```
**Outcome:** Deploy and run LLM inference on Kaggle T4

### **Path 2: Full Foundation** (3 hours)
```
01 → 02 → 03 → 04 → 05 → 06 → 10
```
**Outcome:** Deploy production-ready LLM systems

### **Path 3: Observability Focus** ⭐ **RECOMMENDED** (2.5 hours)
```
01 → 03 → 14 → 15 → 16
```
**Outcome:** Build complete production observability stack

### **Path 4: Graph Analytics** (2.5 hours)
```
01 → 03 → 06 → 07 → 08 → 11
```
**Outcome:** Build LLM-powered graph analytics applications

### **Path 5: Large Model Specialist** (2 hours)
```
01 → 03 → 04 → 09
```
**Outcome:** Run 70B models on Kaggle dual T4

---

## 📊 Notebooks 14-16 Comparison

| Feature | Notebook 14 | Notebook 15 | Notebook 16 |
|---------|-------------|-------------|-------------|
| **Focus** | OpenTelemetry basics | Real-time monitoring | Complete production stack |
| **Complexity** | Intermediate | Intermediate-Advanced | Expert |
| **Time** | 45 min | 30 min | 45 min |
| **OpenTelemetry** | ✅ Full | ❌ | ✅ Full + Advanced |
| **llama.cpp Metrics** | ❌ | ✅ Full | ✅ Full |
| **GPU Monitoring** | ❌ | ✅ PyNVML | ✅ PyNVML |
| **Graphistry** | ✅ Basic | ❌ | ✅ Advanced |
| **Plotly 2D** | ✅ Static | ✅ Live Updates | ✅ Comprehensive |
| **Plotly 3D** | ❌ | ❌ | ✅ Model Internals |
| **Live Dashboards** | ❌ | ✅ FigureWidget | ✅ Multi-panel |

---

## 📁 Repository Structure

All notebooks are located in the `notebooks/` directory:
- **Executed versions** (with outputs): `*-e1.ipynb`, `*-e2.ipynb`, etc.
- **Specification files**: `*-SPEC-*.md` (implementation guides)

---

## 🚀 Getting Started

1. Start with **notebooks/README.md** for an overview
2. Follow the **Quick Start** path (notebooks 01-03) for immediate results
3. Progress to the **Observability Trilogy** (notebooks 14-16) for production workflows

---

## 💡 Additional Resources

- `notebooks/14-15-16-INDEX.md` - Detailed observability notebooks guide
- `docs/KAGGLE_GUIDE.md` - Kaggle-specific optimization tips
- `docs/TROUBLESHOOTING.md` - Common issues and solutions
