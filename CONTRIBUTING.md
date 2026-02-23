# Contributing to llamatelemetry

Thank you for your interest in contributing to llamatelemetry! This guide will help you get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions.

---

## Getting Started

### Types of Contributions

We welcome:

- 🐛 **Bug fixes** - Fix issues and improve stability
- ✨ **Features** - Add new functionality
- 📚 **Documentation** - Improve guides and API docs
- 🧪 **Tests** - Expand test coverage
- 📓 **Notebooks** - Tutorial and example notebooks
- 🔧 **Tooling** - Build and CI improvements

### First-Time Contributors

Good first issues are labeled with `good first issue`. Start there!

---

## Development Setup

### Prerequisites

- Python 3.11+
- CUDA 12.x (for GPU features)
- Git

### Clone Repository

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
```

### Create Virtual Environment

```bash
# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dev dependencies
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest tests/

# Check code style
ruff check llamatelemetry/
mypy llamatelemetry/
```

---

## Project Structure

```
llamatelemetry/
├── llamatelemetry/               # Main package
│   ├── __init__.py               # Public API, require_cuda(), detect_cuda()
│   ├── _version.py               # Version string (2.0.0)
│   ├── inference_engine.py       # Unified high-level API
│   ├── llama_cpp_native/         # GGUF loading, inference, quantization
│   │   ├── model.py              # Model loading
│   │   ├── inference.py          # Text generation
│   │   ├── sampler.py            # 20+ sampling methods
│   │   ├── batch.py              # Batch operations
│   │   ├── context.py            # Inference context
│   │   └── tokenizer.py          # Tokenization
│   ├── nccl_native/              # Dual GPU coordination (NCCL)
│   │   ├── communicator.py       # NCCL communicator setup
│   │   ├── collectives.py        # AllReduce, AllGather, etc.
│   │   └── types.py              # NCCL data types
│   ├── otel_gen_ai/              # OpenTelemetry integration
│   │   ├── tracer.py             # Trace provider setup
│   │   ├── metrics.py            # 5 histogram metrics
│   │   ├── gpu_monitor.py        # GPU telemetry
│   │   └── context.py            # Span context
│   ├── kaggle_integration/       # Kaggle Dual T4 setup
│   │   ├── environment.py        # Detect Kaggle environment
│   │   ├── gpu_config.py         # Dual T4 configuration
│   │   └── model_downloader.py   # HuggingFace Hub downloads
│   └── lib/                      # Compiled C++/CUDA binary
│       └── llamatelemetry_cpp*.so # 187 MB static-linked .so
├── core/                         # Core shared utilities
├── csrc/                         # C++/CUDA source (SM 7.5, llama.cpp b7760)
├── docs/                         # Documentation
│   ├── GOLDEN_PATH.md
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── KAGGLE_GUIDE.md
│   ├── INTEGRATION_GUIDE.md
│   ├── QUICK_START_GUIDE.md
│   └── QUICK_REFERENCE.md
├── notebooks/                    # Tutorial notebooks
├── scripts/                      # Build and release scripts
├── tests/                        # Test suite (246 tests)
├── releases/                     # Local release archives
│   └── v2.0.0/                   # CUDA binary + source archives
├── CHANGELOG.md
├── CONTRIBUTING.md               # This file
├── LICENSE
├── pyproject.toml
└── README.md
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `llamatelemetry/inference_engine.py` | Unified `create_engine()` API |
| `llamatelemetry/llama_cpp_native/model.py` | GGUF model loading |
| `llamatelemetry/llama_cpp_native/inference.py` | Text generation with metrics |
| `llamatelemetry/llama_cpp_native/sampler.py` | 20+ sampling algorithms |
| `llamatelemetry/nccl_native/communicator.py` | Dual GPU NCCL setup |
| `llamatelemetry/nccl_native/collectives.py` | AllReduce, AllGather operations |
| `llamatelemetry/otel_gen_ai/tracer.py` | OpenTelemetry trace provider |
| `llamatelemetry/otel_gen_ai/metrics.py` | 5 GenAI histogram instruments (v2.0.0) |
| `llamatelemetry/otel_gen_ai/gpu_monitor.py` | GPU memory/utilization tracking |
| `llamatelemetry/kaggle_integration/environment.py` | Kaggle environment detection |
| `llamatelemetry/kaggle_integration/gpu_config.py` | Dual T4 auto-configuration |
| `llamatelemetry/kaggle_integration/model_downloader.py` | HuggingFace Hub integration |

---

## Making Changes

### Branch Naming

```bash
# Feature branches
git checkout -b feature/your-feature-name

# Bug fixes
git checkout -b fix/issue-description

# Documentation
git checkout -b docs/what-you-documented
```

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance

Examples:
```
feat(server): add timeout parameter to ServerConfig
fix(client): handle connection timeout gracefully
docs(readme): update installation instructions
test(gguf): add parser edge case tests
```

### Keep Changes Focused

- One feature/fix per PR
- Small, reviewable changes
- Update tests and docs together

---

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Run specific test file
pytest tests/test_server.py

# Run specific test
pytest tests/test_server.py::test_server_config

# Run with coverage
pytest --cov=llamatelemetry tests/
```

### Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests (requires GPU)
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from llamatelemetry.server import ServerConfig

def test_server_config_defaults():
    """Test ServerConfig default values."""
    config = ServerConfig(model_path="test.gguf")
    assert config.host == "127.0.0.1"
    assert config.port == 8080
    assert config.n_gpu_layers == -1

def test_server_config_validation():
    """Test ServerConfig validation."""
    with pytest.raises(ValueError):
        ServerConfig(model_path="")  # Empty path

@pytest.mark.gpu
def test_multi_gpu_inference():
    """Test multi-GPU inference (requires 2 GPUs)."""
    # This test only runs with --gpu flag
    pass
```

---

## Code Style

### Python Style

We use:
- **Ruff** for linting and formatting
- **mypy** for type checking
- **Black** style (via Ruff)

### Ruff Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

### Run Linters

```bash
# Check style
ruff check llamatelemetry/

# Auto-fix
ruff check --fix llamatelemetry/

# Format
ruff format llamatelemetry/

# Type check
mypy llamatelemetry/
```

### Code Guidelines

```python
# Use type hints
def estimate_vram(
    model_params: int,
    bits_per_weight: float,
    context_size: int = 4096,
) -> float:
    """
    Estimate VRAM requirements.
    
    Args:
        model_params: Number of model parameters.
        bits_per_weight: Bits per weight (e.g., 4.0 for Q4).
        context_size: Context window size.
        
    Returns:
        Estimated VRAM in GB.
    """
    ...

# Use dataclasses for config
from dataclasses import dataclass

@dataclass
class ServerConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
```

---

## Documentation

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `docs/GOLDEN_PATH.md` | Recommended quickstart |
| `docs/INSTALLATION.md` | Installation guide |
| `docs/CONFIGURATION.md` | Configuration reference |
| `docs/API_REFERENCE.md` | API documentation |
| `docs/ARCHITECTURE.md` | System design |
| `docs/KAGGLE_GUIDE.md` | Kaggle T4 GPU guide |
| `docs/INTEGRATION_GUIDE.md` | Backend integration guide |
| `docs/QUICK_START_GUIDE.md` | 5-minute quickstart |
| `docs/QUICK_REFERENCE.md` | Cheat sheet |
| `docs/TROUBLESHOOTING.md` | Common issues |

### Documentation Style

- Use clear, concise language
- Include code examples
- Add tables for reference data
- Use ASCII diagrams for architecture
- Link between documents

### Docstrings

Use Google-style docstrings:

```python
def chat_completion(
    self,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> ChatCompletionResponse:
    """
    Send a chat completion request.
    
    Args:
        messages: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0-2.0).
        
    Returns:
        ChatCompletionResponse with generated content.
        
    Raises:
        ConnectionError: If server is not reachable.
        TimeoutError: If request times out.
        
    Example:
        >>> client = LlamaCppClient()
        >>> response = client.chat_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """
```

---

## Pull Request Process

### Before Submitting

1. ✅ Tests pass: `pytest tests/`
2. ✅ Linting passes: `ruff check llamatelemetry/`
3. ✅ Type checks pass: `mypy llamatelemetry/`
4. ✅ Documentation updated
5. ✅ CHANGELOG.md updated

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manually tested

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

### Review Process

1. Submit PR against `main` branch
2. CI runs tests and linting
3. Maintainer reviews code
4. Address feedback
5. Squash and merge

---

## Release Process

### Version Numbering

We use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish

```bash
# Update version
# pyproject.toml: version = "2.1.0"

# Update changelog
# Add release notes under ## [2.1.0] - YYYY-MM-DD

# Commit and tag
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v2.1.0"
git tag v2.1.0
git push origin main --tags

# Create GitHub release with assets
gh release create v2.1.0 --title "LlamaTelemetry v2.1.0" --notes "Release notes here"
```

---

## Questions?

- Open a GitHub issue for bugs or features
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🚀
