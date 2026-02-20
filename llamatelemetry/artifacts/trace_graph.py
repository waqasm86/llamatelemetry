"""
llamatelemetry.artifacts.trace_graph - Export trace graph -> ArtifactRef.

Refactored from telemetry/graphistry_export.py (GraphistryTraceExporter).
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd

    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False


@dataclass
class ArtifactRef:
    """Reference to an exported trace-graph artifact."""

    trace_id: str
    graphistry_url: Optional[str] = None
    manifest_path: Optional[str] = None
    data_path: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0


# ---------------------------------------------------------------------------
# In-memory span buffer (populated by RedactionSpanProcessor or manual calls)
# ---------------------------------------------------------------------------
_span_buffer: List[dict] = []
_MAX_BUFFER = 10_000


def _span_to_dict(span: Any) -> dict:
    """Convert a ReadableSpan to a flat dictionary."""
    ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
    parent = span.parent if hasattr(span, "parent") else None

    span_id = hex(ctx.span_id)[2:] if ctx else str(id(span))
    parent_id = (
        hex(parent.span_id)[2:]
        if parent and hasattr(parent, "span_id")
        else None
    )
    trace_id = hex(ctx.trace_id)[2:] if ctx else ""

    attrs = (
        dict(span.attributes)
        if hasattr(span, "attributes") and span.attributes
        else {}
    )

    start_time = span.start_time if hasattr(span, "start_time") else 0
    end_time = span.end_time if hasattr(span, "end_time") else 0
    duration_ms = (end_time - start_time) / 1e6 if end_time and start_time else 0.0

    return {
        "span_id": span_id,
        "parent_span_id": parent_id,
        "trace_id": trace_id,
        "name": span.name if hasattr(span, "name") else "unknown",
        "start_time": start_time / 1e9 if start_time else 0,
        "end_time": end_time / 1e9 if end_time else 0,
        "duration_ms": duration_ms,
        "gen_ai.request.model": attrs.get("gen_ai.request.model", ""),
        "gen_ai.usage.input_tokens": attrs.get("gen_ai.usage.input_tokens", 0),
        "gen_ai.usage.output_tokens": attrs.get("gen_ai.usage.output_tokens", 0),
        "llamatelemetry.latency_ms": attrs.get("llamatelemetry.latency_ms", duration_ms),
        "gpu.id": attrs.get("gpu.id", "0"),
        "nccl.split_mode": attrs.get("nccl.split_mode", "none"),
        "status": str(span.status) if hasattr(span, "status") else "OK",
    }


def record_spans(spans: List[Any]) -> None:
    """Add spans to the in-memory buffer (for later export)."""
    global _span_buffer
    for span in spans:
        _span_buffer.append(_span_to_dict(span))
    if len(_span_buffer) > _MAX_BUFFER:
        _span_buffer = _span_buffer[-_MAX_BUFFER:]


class TraceGraphSpanProcessor:
    """SpanProcessor that records spans for trace graph export."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: Any) -> None:
        try:
            record_spans([span])
        except Exception:
            pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True


def _build_dataframes(
    raw: List[dict],
    trace_id_filter: Optional[str] = None,
    include_gpu: bool = True,
    include_llm_phases: bool = True,
) -> Tuple[Any, Any]:
    """Build node/edge DataFrames from raw span dicts."""
    if not _PD_AVAILABLE:
        raise ImportError("pandas is required for trace graph export")

    filtered = raw
    if trace_id_filter:
        filtered = [s for s in raw if s["trace_id"] == trace_id_filter]

    nodes = []
    edges = []

    for s in filtered:
        node: Dict[str, Any] = {
            "span_id": s["span_id"],
            "name": s["name"],
            "trace_id": s["trace_id"],
            "duration_ms": s["duration_ms"],
            "status": s["status"],
        }
        if include_llm_phases:
            node["gen_ai_request_model"] = s["gen_ai.request.model"]
            node["gen_ai_latency_ms"] = s["llamatelemetry.latency_ms"]
            node["gen_ai_input_tokens"] = s["gen_ai.usage.input_tokens"]
            node["gen_ai_output_tokens"] = s["gen_ai.usage.output_tokens"]
        if include_gpu:
            node["gpu_id"] = s["gpu.id"]
            node["nccl_split_mode"] = s["nccl.split_mode"]

        nodes.append(node)

        if s["parent_span_id"]:
            edges.append(
                {
                    "src": s["parent_span_id"],
                    "dst": s["span_id"],
                    "weight": (
                        (s["gen_ai.usage.output_tokens"] / (s["llamatelemetry.latency_ms"] / 1000.0))
                        if s["llamatelemetry.latency_ms"] > 0
                        else 0.0
                    ),
                    "latency_ms": s["llamatelemetry.latency_ms"],
                }
            )

    nodes_df = pd.DataFrame(nodes)
    edges_df = (
        pd.DataFrame(edges)
        if edges
        else pd.DataFrame(columns=["src", "dst", "weight", "latency_ms"])
    )
    return nodes_df, edges_df


def export_trace_graph(
    trace_id: str = "",
    out_dir: Optional[str] = None,
    include_gpu: bool = True,
    include_llm_phases: bool = True,
) -> ArtifactRef:
    """
    Export collected spans as a trace graph artifact.

    Args:
        trace_id: Filter to a specific trace. Empty string = all traces.
        out_dir: Directory to write manifest + data files.
                 None = return ArtifactRef without writing files.
        include_gpu: Include GPU attributes in the graph.
        include_llm_phases: Include LLM phase attributes.

    Returns:
        ArtifactRef describing the exported artifact.
    """
    nodes_df, edges_df = _build_dataframes(
        _span_buffer,
        trace_id_filter=trace_id or None,
        include_gpu=include_gpu,
        include_llm_phases=include_llm_phases,
    )

    ref = ArtifactRef(
        trace_id=trace_id,
        node_count=len(nodes_df) if nodes_df is not None else 0,
        edge_count=len(edges_df) if edges_df is not None else 0,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        nodes_path = out / "nodes.csv"
        edges_path = out / "edges.csv"
        manifest_path = out / "manifest.json"

        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)

        manifest = {
            "trace_id": trace_id,
            "node_count": ref.node_count,
            "edge_count": ref.edge_count,
            "nodes_file": str(nodes_path),
            "edges_file": str(edges_path),
            "exported_at": time.time(),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        ref.manifest_path = str(manifest_path)
        ref.data_path = str(out)

    # Optionally push to graphistry
    try:
        import graphistry

        g = (
            graphistry.edges(edges_df, "src", "dst")
            .nodes(nodes_df, "span_id")
            .bind(edge_color="latency_ms")
        )
        url = g.plot(render=False)
        ref.graphistry_url = url
    except Exception:
        pass

    return ref
