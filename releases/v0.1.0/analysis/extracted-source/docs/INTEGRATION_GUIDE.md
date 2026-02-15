# Integration Guide

This guide explains how to integrate llamatelemetry with your inference workflow.

## Typical Flow
1. Start llama.cpp server with `ServerManager`
2. Use `LlamaCppClient` for inference
3. Enable telemetry and export via OTLP
4. Visualize spans using Graphistry

## Files to Reference
- `docs/QUICK_START_GUIDE.md`
- `docs/CONFIGURATION.md`
- `docs/TROUBLESHOOTING.md`
