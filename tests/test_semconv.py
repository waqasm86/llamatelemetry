"""Tests for llamatelemetry.semconv module."""
from llamatelemetry.semconv import keys


def test_no_legacy_llm_keys():
    all_keys = [
        v for name, v in vars(keys).items()
        if not name.startswith("_") and isinstance(v, str)
    ]
    assert not any(k.startswith("llm.") for k in all_keys)


def test_gpu_keys_are_strings():
    gpu_keys = [
        keys.GPU_ID, keys.GPU_UTILIZATION_PCT, keys.GPU_MEM_USED_MB,
        keys.GPU_MEM_TOTAL_MB, keys.GPU_POWER_W, keys.GPU_TEMP_C,
        keys.GPU_NAME, keys.GPU_COMPUTE_CAP, keys.GPU_DRIVER_VERSION,
    ]
    for k in gpu_keys:
        assert isinstance(k, str), f"{k} is not a string"
        assert k.startswith("gpu."), f"{k} doesn't start with gpu."


def test_nccl_keys_are_strings():
    nccl_keys = [
        keys.NCCL_COLLECTIVE, keys.NCCL_BYTES,
        keys.NCCL_WAIT_MS, keys.NCCL_SPLIT_MODE,
    ]
    for k in nccl_keys:
        assert isinstance(k, str), f"{k} is not a string"
        assert k.startswith("nccl."), f"{k} doesn't start with nccl."


def test_service_keys_are_strings():
    svc_keys = [keys.RUN_ID, keys.REQUEST_ID, keys.SESSION_ID, keys.USER_ID]
    for k in svc_keys:
        assert isinstance(k, str), f"{k} is not a string"


def test_key_uniqueness():
    all_keys = [
        v for name, v in vars(keys).items()
        if not name.startswith("_") and isinstance(v, str)
    ]
    assert len(all_keys) == len(set(all_keys)), "Duplicate key constants found"


def test_attrs_module_importable():
    from llamatelemetry.semconv import attrs
    assert callable(attrs.run_id)
    assert callable(attrs.gpu_attrs)
    assert callable(attrs.nccl_attrs)
