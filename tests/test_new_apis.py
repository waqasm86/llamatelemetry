"""
Unit tests for llamatelemetry optional-dependency APIs (quantization, unsloth, cuda, inference).

These tests require torch, triton, and other heavy dependencies.
They are automatically skipped when those packages are not installed.
"""

import unittest
import sys
from pathlib import Path

# Add llamatelemetry to path (these modules are sub-packages of llamatelemetry)
sys.path.insert(0, str(Path(__file__).parent.parent / "llamatelemetry"))

_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass


@unittest.skipUnless(_torch_available, "torch not installed")
class TestQuantizationAPI(unittest.TestCase):
    """Test Quantization API (requires torch)"""

    def test_nf4_config(self):
        from quantization.nf4 import NF4Config
        config = NF4Config(blocksize=64, double_quant=True)
        self.assertEqual(config.blocksize, 64)
        self.assertTrue(config.double_quant)

    def test_gguf_quant_types(self):
        from quantization.gguf import GGUFQuantType
        self.assertIsNotNone(GGUFQuantType.Q4_K_M)
        self.assertIsNotNone(GGUFQuantType.F16)
        self.assertEqual(GGUFQuantType.Q4_K_M.value, 25)

    def test_dynamic_quantizer(self):
        from quantization.dynamic import DynamicQuantizer, QuantStrategy
        quantizer = DynamicQuantizer(target_vram_gb=12.0, strategy=QuantStrategy.BALANCED)
        config = quantizer.recommend_config(model_size_gb=3.0, verbose=False)
        self.assertIn('quant_type', config)
        self.assertIn('expected_vram_gb', config)
        self.assertEqual(config['quant_type'], 'Q4_K_M')


@unittest.skipUnless(_torch_available, "torch not installed")
class TestUnslothIntegrationAPI(unittest.TestCase):
    """Test Unsloth Integration API (requires torch)"""

    def test_unsloth_loader(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available for UnslothModelLoader")
        from unsloth.loader import UnslothModelLoader
        loader = UnslothModelLoader(max_seq_length=2048)
        self.assertEqual(loader.max_seq_length, 2048)

    def test_export_config(self):
        from unsloth.exporter import ExportConfig
        config = ExportConfig(quant_type="Q4_K_M", merge_lora=True)
        self.assertEqual(config.quant_type, "Q4_K_M")
        self.assertTrue(config.merge_lora)

    def test_adapter_config(self):
        from unsloth.adapter import AdapterConfig
        config = AdapterConfig(r=16, lora_alpha=32)
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(len(config.target_modules), 7)


@unittest.skipUnless(_torch_available, "torch not installed")
class TestCUDAOptimizationAPI(unittest.TestCase):
    """Test CUDA Optimization API (requires torch)"""

    def test_tensor_core_config(self):
        from cuda.tensor_core import TensorCoreConfig
        config = TensorCoreConfig(enabled=True, allow_tf32=True)
        self.assertTrue(config.enabled)
        self.assertTrue(config.allow_tf32)

    def test_cuda_graph(self):
        from cuda.graphs import CUDAGraph
        graph = CUDAGraph()
        self.assertFalse(graph.is_captured())

    def test_graph_pool(self):
        from cuda.graphs import GraphPool
        pool = GraphPool()
        self.assertEqual(len(pool.list_graphs()), 0)

    def test_kernel_config(self):
        from cuda.triton_kernels import KernelConfig
        config = KernelConfig(name="test", block_size=128, num_warps=4)
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.num_warps, 4)

    def test_registered_kernels(self):
        from cuda.triton_kernels import list_kernels
        kernels = list_kernels()
        self.assertGreater(len(kernels), 0)


@unittest.skipUnless(_torch_available, "torch not installed")
class TestAdvancedInferenceAPI(unittest.TestCase):
    """Test Advanced Inference API (requires torch)"""

    def test_flash_attention_config(self):
        from inference.flash_attn import FlashAttentionConfig
        config = FlashAttentionConfig(enabled=True, causal=True)
        self.assertTrue(config.enabled)
        self.assertTrue(config.causal)

    def test_optimal_context_length(self):
        from inference.flash_attn import get_optimal_context_length
        ctx_len = get_optimal_context_length(
            model_size_b=3.0,
            available_vram_gb=12.0,
            use_flash_attention=True,
        )
        self.assertGreater(ctx_len, 0)
        self.assertIn(ctx_len, [512, 1024, 2048, 4096, 8192, 16384, 32768])

    def test_kv_cache_config(self):
        from inference.kv_cache import KVCacheConfig
        config = KVCacheConfig(max_batch_size=8, max_seq_length=4096)
        self.assertEqual(config.max_batch_size, 8)
        self.assertEqual(config.max_seq_length, 4096)

    def test_kv_cache(self):
        from inference.kv_cache import KVCache, KVCacheConfig
        config = KVCacheConfig()
        cache = KVCache(config)
        self.assertEqual(len(cache.cache), 0)

    def test_batch_config(self):
        from inference.batch import BatchConfig
        config = BatchConfig(max_batch_size=8, dynamic_batching=True)
        self.assertEqual(config.max_batch_size, 8)
        self.assertTrue(config.dynamic_batching)


@unittest.skipUnless(_torch_available, "torch not installed")
class TestAPIIntegration(unittest.TestCase):
    """Test API integration and workflow (requires torch)"""

    def test_all_imports(self):
        from quantization import (
            quantize_nf4,
            NF4Quantizer,
            convert_to_gguf,
            DynamicQuantizer,
        )
        from unsloth import (
            load_unsloth_model,
            export_to_llamatelemetry,
            merge_lora_adapters,
        )
        from cuda import (
            enable_tensor_cores,
            CUDAGraph,
            list_kernels,
        )
        from inference import (
            enable_flash_attention,
            KVCache,
            batch_inference_optimized,
        )
        self.assertTrue(True)

    def test_quantization_strategies(self):
        from quantization.dynamic import QuantStrategy
        strategies = [
            QuantStrategy.AGGRESSIVE,
            QuantStrategy.BALANCED,
            QuantStrategy.QUALITY,
            QuantStrategy.MINIMAL,
        ]
        for strategy in strategies:
            self.assertIsNotNone(strategy)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestUnslothIntegrationAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestCUDAOptimizationAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedInferenceAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
