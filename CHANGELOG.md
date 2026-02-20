# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-02-20

### Added
- GenAI span helpers and metrics instruments (`otel/gen_ai_utils.py`, `otel/gen_ai_metrics.py`).
- GenAI client/server metrics: `gen_ai.client.operation.duration`, `gen_ai.client.token.usage`,
  `gen_ai.server.request.duration`, `gen_ai.server.time_to_first_token`, `gen_ai.server.time_per_output_token`.
- GenAI operation detail events (`gen_ai.client.inference.operation.details`) as opt-in instrumentation.

### Changed
- Root span naming + kind follow the GenAI spec: `{gen_ai.operation.name} {gen_ai.request.model}`, `SpanKind.CLIENT`.
- Early span attributes for sampling: `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`.
- LlamaCpp and Transformers instrumentation emit GenAI metrics by default.
- Docs, notebooks, and build scripts updated for v1.2.0.

### Removed
- Legacy LLM attributes and dual-emit code paths.
- Deprecated GenAI attribute names.
