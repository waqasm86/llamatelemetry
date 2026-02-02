# llamatelemetry v0.1.0 Release Workflow (Kaggle Dual T4)

This guide defines the **v0.1.0** release flow for Kaggle dual Tesla T4.

**Distribution:** GitHub Releases (primary). **No PyPI** for v0.1.0.

---

## Workflow Summary

```
Build CUDA 12 binaries (SM 7.5)
    ↓
Package bundle (bin/ + lib/)
    ↓
Upload to GitHub Releases
    ↓
Verify in Kaggle notebook
```

---

## 1) Build Binaries

Build llama.cpp for SM 7.5 (Tesla T4). Use your preferred build environment (Kaggle or equivalent CUDA 12 setup).

---

## 2) Package Release Bundle

Create:
- `llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz`

---

## 3) Publish GitHub Release

See `docs/GITHUB_RELEASE_GUIDE.md` for the step-by-step process.

---

## 4) Verify in Kaggle

```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
import llamatelemetry
print(llamatelemetry.__version__)
```

---

## Notes

- v0.1.0 is **Kaggle-only** (dual T4).
- Do not publish v0.1.0 to PyPI.
