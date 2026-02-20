"""Tests for llamatelemetry.bench (benchmark harness)."""

import json
import pytest


class TestBenchmarkProfile:
    """Test BenchmarkProfile dataclass."""

    def test_default_values(self):
        from llamatelemetry.bench.profiles import BenchmarkProfile
        p = BenchmarkProfile(name="test", prompt="Hello?")
        assert p.name == "test"
        assert p.prompt == "Hello?"
        assert p.max_tokens == 128
        assert p.category == "general"
        assert p.description == ""

    def test_custom_values(self):
        from llamatelemetry.bench.profiles import BenchmarkProfile
        p = BenchmarkProfile(
            name="long_prompt",
            prompt="Explain quantum computing.",
            max_tokens=512,
            category="throughput",
            description="Long generation test",
        )
        assert p.max_tokens == 512
        assert p.category == "throughput"

    def test_get_default_profiles(self):
        from llamatelemetry.bench.profiles import get_default_profiles
        profiles = get_default_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        names = [p.name for p in profiles]
        assert "short_prompt" in names

    def test_all_default_profiles_have_prompts(self):
        from llamatelemetry.bench.profiles import get_default_profiles
        for p in get_default_profiles():
            assert isinstance(p.prompt, str)
            assert len(p.prompt) > 0
            assert p.max_tokens > 0

    def test_profile_categories(self):
        from llamatelemetry.bench.profiles import get_default_profiles
        profiles = get_default_profiles()
        categories = {p.category for p in profiles}
        assert len(categories) > 0


class TestTestResult:
    """Test TestResult dataclass."""

    def test_defaults(self):
        from llamatelemetry.bench.report import TestResult
        r = TestResult()
        assert r.name == ""
        assert r.ttft_ms == 0.0
        assert r.tps == 0.0
        assert r.iterations == 1

    def test_custom_values(self):
        from llamatelemetry.bench.report import TestResult
        r = TestResult(
            name="short_prompt",
            ttft_ms=50.0,
            tpot_ms=10.0,
            tps=120.5,
            input_tokens=10,
            output_tokens=64,
        )
        assert r.name == "short_prompt"
        assert r.ttft_ms == 50.0
        assert r.tps == 120.5

    def test_to_dict(self):
        from llamatelemetry.bench.report import TestResult
        r = TestResult(name="test", ttft_ms=50.0, tps=100.0)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert "ttft_ms" in d
        assert "tps" in d


class TestBenchmarkReport:
    """Test BenchmarkReport."""

    def test_report_importable(self):
        from llamatelemetry.bench.report import BenchmarkReport
        assert BenchmarkReport is not None

    def test_empty_report(self):
        from llamatelemetry.bench.report import BenchmarkReport
        report = BenchmarkReport(backend="llama.cpp")
        assert report.backend == "llama.cpp"
        assert len(report.results) == 0

    def test_report_to_dict(self):
        from llamatelemetry.bench.report import BenchmarkReport
        report = BenchmarkReport(backend="llama.cpp")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "backend" in d
        assert "results" in d
        assert "timestamp" in d
        assert d["backend"] == "llama.cpp"

    def test_report_to_dict_with_results(self):
        from llamatelemetry.bench.report import BenchmarkReport, TestResult
        r = TestResult(name="short_prompt", ttft_ms=50.0, tps=120.0)
        report = BenchmarkReport(backend="llama.cpp", results=[r])
        d = report.to_dict()
        assert len(d["results"]) == 1
        assert "aggregates" in d
        assert "avg_tps" in d["aggregates"]

    def test_report_timestamp_auto_set(self):
        from llamatelemetry.bench.report import BenchmarkReport
        report = BenchmarkReport(backend="llama.cpp")
        assert report.timestamp != ""
        assert "T" in report.timestamp  # ISO format

    def test_report_version_auto_set(self):
        from llamatelemetry.bench.report import BenchmarkReport
        report = BenchmarkReport(backend="llama.cpp")
        assert report.llamatelemetry_version != ""

    def test_report_save_and_load(self, tmp_path):
        from llamatelemetry.bench.report import BenchmarkReport, TestResult
        r = TestResult(name="test", ttft_ms=50.0, tps=100.0, iterations=3)
        report = BenchmarkReport(backend="llama.cpp", results=[r])
        path = tmp_path / "report.json"
        report.save(path)
        loaded = BenchmarkReport.load(path)
        assert loaded.backend == "llama.cpp"
        assert len(loaded.results) == 1
        assert loaded.results[0].name == "test"


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_defaults(self):
        from llamatelemetry.bench.compare import ComparisonResult
        c = ComparisonResult()
        assert c.test_name == ""
        assert c.metric == ""
        assert c.baseline == 0.0
        assert c.current == 0.0
        assert c.delta == 0.0
        assert c.delta_pct == 0.0
        assert c.regression is False


class TestCompareReports:
    """Test compare_reports function."""

    def test_compare_importable(self):
        from llamatelemetry.bench.compare import compare_reports
        assert callable(compare_reports)

    def test_compare_empty_reports(self):
        from llamatelemetry.bench.compare import compare_reports
        r1 = {"results": []}
        r2 = {"results": []}
        comparisons = compare_reports(r1, r2)
        assert isinstance(comparisons, list)
        assert len(comparisons) == 0

    def test_compare_matching_tests(self):
        from llamatelemetry.bench.compare import compare_reports
        baseline = {"results": [{"name": "short_prompt", "tps": 100.0, "ttft_ms": 50.0}]}
        current = {"results": [{"name": "short_prompt", "tps": 110.0, "ttft_ms": 45.0}]}
        comparisons = compare_reports(baseline, current)
        assert isinstance(comparisons, list)

    def test_regression_detected(self):
        """A large drop in TPS should be flagged as regression."""
        from llamatelemetry.bench.compare import compare_reports
        baseline = {"results": [{"name": "short_prompt", "tps": 100.0, "ttft_ms": 50.0, "tpot_ms": 10.0}]}
        current = {"results": [{"name": "short_prompt", "tps": 50.0, "ttft_ms": 100.0, "tpot_ms": 20.0}]}
        comparisons = compare_reports(baseline, current, regression_threshold_pct=10.0)
        regressions = [c for c in comparisons if c.regression]
        assert len(regressions) > 0


class TestBenchmarkRunner:
    """Test BenchmarkRunner (no server required)."""

    def test_runner_importable(self):
        from llamatelemetry.bench.runner import BenchmarkRunner
        assert BenchmarkRunner is not None

    def test_runner_init_defaults(self):
        from llamatelemetry.bench.runner import BenchmarkRunner
        runner = BenchmarkRunner()
        assert runner._backend == "llama.cpp"
        assert runner._num_iterations == 3
        assert runner._warmup_iterations == 1
        assert runner._profiles is not None
        assert len(runner._profiles) > 0

    def test_runner_init_custom_profiles(self):
        from llamatelemetry.bench.runner import BenchmarkRunner
        from llamatelemetry.bench.profiles import BenchmarkProfile
        profiles = [BenchmarkProfile(name="custom", prompt="Test prompt")]
        runner = BenchmarkRunner(backend="transformers", profiles=profiles)
        assert runner._backend == "transformers"
        assert runner._profiles == profiles


class TestBenchPackageImports:
    """Test bench package top-level imports."""

    def test_package_imports(self):
        from llamatelemetry.bench import (
            BenchmarkRunner,
            BenchmarkReport,
            compare_reports,
            BenchmarkProfile,
        )
        assert BenchmarkRunner is not None
        assert BenchmarkReport is not None
        assert callable(compare_reports)
        assert BenchmarkProfile is not None
