# Installation (v0.1.0)

llamatelemetry v0.1.0 is designed for Kaggle dual T4 notebooks.

## Requirements
- Kaggle notebook
- GPU T4 x2
- Python 3.11+
- CUDA 12.x runtime (pre-installed on Kaggle)

## Install
```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Verify
```python
import llamatelemetry
print(llamatelemetry.__version__)
```

## Notes
- Large CUDA binaries are downloaded on first import.
- PyPI is not used for v0.1.0 distribution.
