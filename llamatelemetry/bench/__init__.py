"""
llamatelemetry.bench - Benchmark harness for inference engines.

Provides reproducible benchmarks for both llama.cpp and Transformers engines.
Generates JSON reports that can be diffed across runs.
"""

from .runner import BenchmarkRunner
from .report import BenchmarkReport
from .compare import compare_reports
from .profiles import BenchmarkProfile
