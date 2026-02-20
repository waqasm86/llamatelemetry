"""Tests for llamatelemetry.artifacts.trace_graph."""
import json
import os
import tempfile
import pytest
from llamatelemetry.artifacts.trace_graph import (
    ArtifactRef,
    record_spans,
    export_trace_graph,
    _span_buffer,
)


def test_artifact_ref_defaults():
    ref = ArtifactRef(trace_id="abc123")
    assert ref.trace_id == "abc123"
    assert ref.graphistry_url is None
    assert ref.manifest_path is None
    assert ref.data_path is None
    assert ref.node_count == 0
    assert ref.edge_count == 0


def test_artifact_ref_with_values():
    ref = ArtifactRef(
        trace_id="abc",
        graphistry_url="https://example.com/graph",
        node_count=10,
        edge_count=5,
    )
    assert ref.graphistry_url == "https://example.com/graph"
    assert ref.node_count == 10
    assert ref.edge_count == 5


@pytest.fixture(autouse=True)
def clear_span_buffer():
    import llamatelemetry.artifacts.trace_graph as mod
    original = mod._span_buffer[:]
    mod._span_buffer.clear()
    yield
    mod._span_buffer.clear()
    mod._span_buffer.extend(original)


def test_record_spans_mock():
    """Test recording mock span objects."""
    class MockSpan:
        name = "test.span"
        start_time = 1000000000
        end_time = 2000000000
        parent = None
        attributes = {"gen_ai.request.model": "test"}
        status = "OK"
        def get_span_context(self):
            ctx = type("Ctx", (), {"span_id": 12345, "trace_id": 67890})()
            return ctx

    spans = [MockSpan(), MockSpan()]
    record_spans(spans)

    import llamatelemetry.artifacts.trace_graph as mod
    assert len(mod._span_buffer) == 2
    assert mod._span_buffer[0]["name"] == "test.span"


def test_export_trace_graph_no_pandas():
    """Test export when no spans and pandas may not be available."""
    try:
        import pandas
        ref = export_trace_graph(trace_id="nonexistent")
        assert isinstance(ref, ArtifactRef)
        assert ref.node_count == 0
        assert ref.edge_count == 0
    except ImportError:
        pytest.skip("pandas not available")


def test_export_to_directory():
    """Test export to a directory creates expected files."""
    try:
        import pandas
    except ImportError:
        pytest.skip("pandas not available")

    # Add some mock span data
    class MockSpan:
        name = "test.export"
        start_time = 1000000000
        end_time = 2000000000
        parent = None
        attributes = {"gen_ai.request.model": "test", "gpu.id": "0", "nccl.split_mode": "none",
                       "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 20,
                       "llamatelemetry.latency_ms": 100, "llamatelemetry.tokens_per_sec": 5.0}
        status = "OK"
        def get_span_context(self):
            return type("Ctx", (), {"span_id": 0xABCD, "trace_id": 0x12345})()

    record_spans([MockSpan()])

    with tempfile.TemporaryDirectory() as tmpdir:
        ref = export_trace_graph(out_dir=tmpdir)
        assert ref.node_count == 1
        assert ref.manifest_path is not None
        assert os.path.exists(os.path.join(tmpdir, "nodes.csv"))
        assert os.path.exists(os.path.join(tmpdir, "edges.csv"))
        assert os.path.exists(os.path.join(tmpdir, "manifest.json"))
        with open(os.path.join(tmpdir, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["node_count"] == 1
