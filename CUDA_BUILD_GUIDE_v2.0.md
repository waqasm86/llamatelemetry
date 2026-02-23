# LlamaTelemetry v2.0 - CUDA C++ Bindings Build Guide

**Date:** 2026-02-23
**Status:** Ready for implementation
**Platform:** Kaggle 2× Tesla T4 GPUs (SM 7.5, 15GB VRAM each) + CUDA 12.5
**Build System:** CMake (Ninja backend) + pybind11 + GCC 11.4.0

---

## Overview

This guide integrates the proven Kaggle build strategy from the existing notebooks (v0.1.0, v2.0.0, v2.2.0) with LlamaTelemetry v2.0's Python APIs to create complete native C++ bindings.

**Output**: Native pybind11 shared library (`_llamatelemetry_cpp.so`) + LlamaTelemetry wheel package

---

## Architecture: What Gets Built

```
LlamaTelemetry v2.0 (Python - Already Complete)
    ↓
pybind11 C++ Bindings Layer (TODO - This Guide)
    ↓
llama.cpp C APIs (100+)  +  NCCL C APIs (50+)  +  CUDA Kernels
    ↓
GPU Execution: Dual T4 (Tensor-Split or Workload-Split)
```

### Files to Create

```
llamatelemetry/
├── csrc/                                    ← TODO: Create this directory
│   ├── __init__.pyi                         (Type stubs for IDE support)
│   ├── bindings.cpp                         (PYBIND11_MODULE main)
│   ├── llama_bindings.cpp                   (llama.cpp API wrapping)
│   ├── nccl_bindings.cpp                    (NCCL API wrapping)
│   ├── cuda_stubs.hpp                       (Kaggle stub library definitions)
│   └── CMakeLists.txt                       (C++ build configuration)
├── setup.py                                 (UPDATE: Add C++ extension)
├── CMakeLists.txt                           (UPDATE: Root CMake config)
└── setup.cfg                                (CREATE: Build backend config)
```

---

## Part 1: Kaggle-Specific Foundation (From Existing Notebooks)

The notebooks (p2, p3, p9, v2.2.0) established solutions for 6 Kaggle challenges:

### Challenge 1: Read-Only `/usr/local/cuda/`

**Problem**: Kaggle's `/usr/local/cuda/` is read-only. Can't write stub libraries there.

**Solution**: Create `/kaggle/working/cuda_stubs/` directory with stub library
```bash
mkdir -p /kaggle/working/cuda_stubs
gcc -shared -fPIC -o cuda_stubs/libcuda.so cuda_stubs/cuda_stub.c
```

**File: `csrc/cuda_stubs.hpp`**
```cpp
// Kaggle Stub Library Definitions
// Used at LINK TIME ONLY to satisfy linker
// At RUNTIME, real CUDA driver in /usr/local/nvidia/lib/ is used

#ifndef CUDA_STUBS_H
#define CUDA_STUBS_H

extern void* cuInit;
extern void* cuMemAlloc;
extern void* cuMemFree;
// ... 30+ more symbols

#endif
```

### Challenge 2: CMake `FindCUDAToolkit` Module Fails

**Problem**: CMake can't find `libcuda.so` → never creates `CUDA::cuda_driver` target → build fails at CMAKE GENERATE step

**Solution**: Patch `ggml/src/ggml-cuda/CMakeLists.txt` to inject target definition

**When building in csrc/CMakeLists.txt**:
```cmake
# Get path to stub library
set(CUDA_STUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cuda_stubs")

# Manual target definition (from existing notebook patch)
if(NOT TARGET CUDA::cuda_driver)
    add_library(CUDA::cuda_driver INTERFACE IMPORTED)
    set_target_properties(CUDA::cuda_driver PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CUDA_STUB_DIR}/libcuda.so"
    )
endif()
```

### Challenge 3: CUDA VMM APIs Not Available

**Problem**: Kaggle sandbox doesn't support CUDA Virtual Memory Management (cuMemCreate, cuMemMap, etc.)

**Solution**: Compile-time flag disables these APIs
```bash
-DGGML_CUDA_NO_VMM
```

**Effect**: No cuMemCreate/cuMemMap calls in compiled CUDA kernels

### Challenge 4: RAPIDS Dependency Conflicts

**Problem**: Kaggle has RAPIDS 25.6.0 pre-installed (cudf-cu12, cuda-python, numba-cuda). Upgrading breaks compatibility.

**Solution**: Install cugraph-cu12==25.6.* to match pre-installed version

**In setup.py optional dependencies**:
```python
extras_require = {
    'kaggle': [
        'cudf-cu12>=25.6',      # Pre-installed
        'cugraph-cu12==25.6.*', # Match version
        'pynvml>=11.0',
    ],
    'cuda': [
        'pycuda>=2021.1',       # For non-Kaggle CUDA
    ],
}
```

### Challenge 5: pybind11 Module Name Conflicts

**Problem**: Multiple pybind11 modules with same base name can conflict

**Solution**: Use full-qualified name pattern
```cpp
PYBIND11_MODULE(_llamatelemetry_cpp, m) {
    // ...
}
```

Load as: `from llamatelemetry import _llamatelemetry_cpp`

---

## Part 2: pybind11 C++ Bindings Structure

### File: `csrc/bindings.cpp` (Main Module Definition)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "llama_bindings.hpp"
#include "nccl_bindings.hpp"

namespace py = pybind11;

// Main module definition
PYBIND11_MODULE(_llamatelemetry_cpp, m) {
    m.doc() = "LlamaTelemetry v2.0 - Native C++ bindings for llama.cpp and NCCL";

    // Register llama.cpp bindings
    bind_llama_cpp(m);

    // Register NCCL bindings
    bind_nccl(m);

    // Module metadata
    m.attr("__version__") = "2.0.0";
    m.attr("__cuda_version__") = "12.5";
    m.attr("__sm_compute_capability__") = "7.5";  // Tesla T4
}
```

### File: `csrc/llama_bindings.cpp` (llama.cpp API Wrapping)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "llama.h"

namespace py = pybind11;

void bind_llama_cpp(py::module &m) {
    // =========================================================================
    // LLAMA MODEL
    // =========================================================================

    py::class_<llama_model>(m, "LlamaModel")
        .def(py::init([](
            const std::string& path,
            int n_gpu_layers,
            const std::string& split_mode) {
            llama_model_params params = llama_model_default_params();
            params.n_gpu_layers = n_gpu_layers;
            // Handle split_mode: "none", "layer", "row"
            if (split_mode == "layer") {
                params.split_mode = LLAMA_SPLIT_LAYER;
            } else if (split_mode == "row") {
                params.split_mode = LLAMA_SPLIT_ROW;
            }
            return std::unique_ptr<llama_model>(
                llama_load_model_from_file(path.c_str(), params)
            );
        }), py::arg("path"), py::arg("n_gpu_layers") = 99,
            py::arg("split_mode") = "layer")

        .def("n_vocab", &llama_model_n_vocab)
        .def("n_embd", &llama_model_n_embd)
        .def("n_layer", &llama_model_n_layer)
        .def("n_head", &llama_model_n_head)
        .def("n_head_kv", &llama_model_n_head_kv)
        .def("get_quantization_type", [](llama_model* model) {
            return std::string(llama_model_quantize_type(model));
        })
        // ... 30+ more methods
    ;

    // =========================================================================
    // LLAMA CONTEXT
    // =========================================================================

    py::class_<llama_context>(m, "LlamaContext")
        .def(py::init([](llama_model* model, int n_ctx) {
            llama_context_params params = llama_context_default_params();
            params.n_ctx = n_ctx;
            return std::unique_ptr<llama_context>(
                llama_new_context_with_model(model, params)
            );
        }), py::arg("model"), py::arg("n_ctx") = 2048)

        .def("encode", [](llama_context* ctx, const std::vector<int>& tokens) {
            llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
            for (size_t i = 0; i < tokens.size(); i++) {
                llama_batch_add(batch, tokens[i], i, {0}, false);
            }
            llama_decode(ctx, batch);
            llama_batch_free(batch);
        })

        .def("get_logits", [](llama_context* ctx) -> py::array_t<float> {
            const float* logits = llama_get_logits(ctx);
            int n_vocab = llama_n_vocab(llama_get_model(ctx));
            return py::array_t<float>(
                std::vector<size_t>{(size_t)n_vocab},
                logits
            );
        })

        // ... 20+ more methods
    ;

    // =========================================================================
    // LLAMA BATCH
    // =========================================================================

    py::class_<llama_batch>(m, "LlamaBatch")
        .def(py::init(&llama_batch_init),
             py::arg("size"), py::arg("embd_size") = 0, py::arg("n_seq_max") = 1)
        .def("add", [](llama_batch& batch, int token, int pos,
                       const std::vector<int>& seq_ids, bool logits) {
            llama_batch_add(batch, token, pos, seq_ids.data(), logits);
        })
        // ... 5+ more methods
    ;

    // =========================================================================
    // LLAMA SAMPLER (20+ algorithms via SamplerChain)
    // =========================================================================

    py::class_<llama_sampler_chain>(m, "SamplerChain")
        .def(py::init(&llama_sampler_chain_init))

        .def("add_temp", [](llama_sampler_chain* chain, float temp) {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
            return chain;  // For method chaining
        })

        .def("add_top_p", [](llama_sampler_chain* chain, float top_p) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
            return chain;
        })

        .def("add_top_k", [](llama_sampler_chain* chain, int top_k) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
            return chain;
        })

        // Add all 20+ sampler types (min_p, mirostat, dry, grammar, etc.)
        // ...

        .def("sample", [](llama_sampler_chain* chain,
                         llama_context* ctx, llama_model* model) -> int {
            const float* logits = llama_get_logits(ctx);
            llama_token_data_array candidates;
            // Build candidates array from logits...
            return llama_sampler_sample(chain, ctx, &candidates);
        })
    ;

    // =========================================================================
    // LLAMA TOKENIZER
    // =========================================================================

    py::class_<llama_tokenizer>(m, "Tokenizer")
        .def(py::init([](llama_model* model) {
            return std::unique_ptr<llama_tokenizer>(
                new llama_tokenizer(model)
            );
        }))

        .def("encode", [](llama_tokenizer* tok, const std::string& text)
                       -> std::vector<int> {
            // Call llama_tokenize and return token IDs
            std::vector<int> tokens;
            // ... implementation
            return tokens;
        })

        .def("decode", [](llama_tokenizer* tok, const std::vector<int>& tokens)
                       -> std::string {
            // Call llama_detokenize for each token and concatenate
            std::string result;
            // ... implementation
            return result;
        })
    ;
}
```

### File: `csrc/nccl_bindings.cpp` (NCCL API Wrapping)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nccl.h>

namespace py = pybind11;

void bind_nccl(py::module &m) {
    // =========================================================================
    // NCCL TYPES
    // =========================================================================

    py::enum_<ncclDataType_t>(m, "DataType")
        .value("INT8", ncclInt8)
        .value("INT32", ncclInt32)
        .value("INT64", ncclInt64)
        .value("FLOAT16", ncclFloat16)
        .value("FLOAT32", ncclFloat32)
        .value("FLOAT64", ncclFloat64)
        // ... 12 total types
    ;

    py::enum_<ncclRedOp_t>(m, "ReductionOp")
        .value("SUM", ncclSum)
        .value("PROD", ncclProd)
        .value("MAX", ncclMax)
        .value("MIN", ncclMin)
        .value("AVG", ncclAvg)
        // ... 8 total operations
    ;

    // =========================================================================
    // NCCL COMMUNICATOR
    // =========================================================================

    py::class_<ncclComm_t*>(m, "NCCLCommunicator")
        .def(py::init([]() -> ncclComm_t* {
            ncclUniqueId id;
            ncclGetUniqueId(&id);
            ncclComm_t* comm = (ncclComm_t*)malloc(sizeof(ncclComm_t));
            ncclCommInitRank(comm, 2, id, 0);  // rank 0 of 2
            return comm;
        }))

        .def("all_reduce", [](ncclComm_t* comm,
                             py::array_t<float> sendbuff,
                             py::array_t<float> recvbuff,
                             ncclRedOp_t op) {
            int count = sendbuff.size();
            ncclAllReduce(
                sendbuff.data(), recvbuff.data(),
                count, ncclFloat32, op, *comm, NULL
            );
        })

        .def("broadcast", [](ncclComm_t* comm,
                            py::array_t<float> buff,
                            int root) {
            ncclBroadcast(
                buff.data(), buff.data(),
                buff.size(), ncclFloat32,
                root, *comm, NULL
            );
        })

        // ... 50+ collective operations

        .def("__del__", [](ncclComm_t* comm) {
            ncclCommDestroy(*comm);
            free(comm);
        })
    ;
}
```

---

## Part 3: CMakeLists.txt Configuration

### File: `csrc/CMakeLists.txt`

```cmake
# =========================================================================
# LlamaTelemetry v2.0 C++ Bindings
# =========================================================================
# Targets: Dual T4 GPUs (SM 7.5) on Kaggle with CUDA 12.5

cmake_minimum_required(VERSION 3.20)
project(llamatelemetry_cpp CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# =========================================================================
# CUDA & TOOLKIT CONFIGURATION
# =========================================================================

# Set CUDA architecture for Tesla T4 (SM 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Disable VMM on Kaggle (not available in sandbox)
add_compile_definitions(GGML_CUDA_NO_VMM)

# Find CUDA toolkit
find_package(CUDA 12.5 REQUIRED)
find_package(CUDAToolkit 12.5 REQUIRED)

# =========================================================================
# KAGGLE-SPECIFIC: Stub Library for libcuda.so
# =========================================================================

# On Kaggle: /usr/local/cuda is read-only, libcuda.so doesn't exist
# Solution: Create minimal stub library for linking

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/kaggle")
    set(CUDA_STUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cuda_stubs")

    # Define CUDA::cuda_driver target manually (FindCUDAToolkit can't find it on Kaggle)
    if(NOT TARGET CUDA::cuda_driver)
        add_library(CUDA::cuda_driver INTERFACE IMPORTED)
        set_target_properties(CUDA::cuda_driver PROPERTIES
            INTERFACE_LINK_LIBRARIES "${CUDA_STUB_DIR}/libcuda.so"
        )
        message(STATUS "Created CUDA::cuda_driver target pointing to stub at ${CUDA_STUB_DIR}/libcuda.so")
    endif()
else()
    # Non-Kaggle: Use system CUDA
    find_package(CUDAToolkit 12.5 REQUIRED COMPONENTS cuda_driver)
endif()

# =========================================================================
# llama.cpp INTEGRATION
# =========================================================================

# Clone or use existing llama.cpp
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
    message(STATUS "Cloning llama.cpp...")
    execute_process(
        COMMAND git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp
    )
endif()

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")

# Include llama.cpp's CMakeLists (which defines llama target)
add_subdirectory("${LLAMA_CPP_DIR}" llama_cpp_build EXCLUDE_FROM_ALL)

# =========================================================================
# NCCL INTEGRATION
# =========================================================================

# Find NCCL (user must provide or build separately)
find_package(NCCL REQUIRED QUIET)
if(NOT NCCL_FOUND)
    message(WARNING "NCCL not found. Multi-GPU support will be limited.")
    # Can still build without NCCL, but collective operations unavailable
endif()

# =========================================================================
# pybind11 SETUP
# =========================================================================

find_package(pybind11 2.10 REQUIRED)

# =========================================================================
# BUILD _llamatelemetry_cpp MODULE
# =========================================================================

pybind11_add_module(_llamatelemetry_cpp
    bindings.cpp
    llama_bindings.cpp
    nccl_bindings.cpp
)

# Link to llama.cpp libraries
target_link_libraries(_llamatelemetry_cpp PRIVATE
    llama              # Main llama.cpp library
    common             # Common utilities
    ggml               # GGML core
)

# Link to NCCL if available
if(NCCL_FOUND)
    target_link_libraries(_llamatelemetry_cpp PRIVATE
        ${NCCL_LIBRARIES}
    )
    target_include_directories(_llamatelemetry_cpp PRIVATE
        ${NCCL_INCLUDE_DIRS}
    )
endif()

# Link to CUDA
target_link_libraries(_llamatelemetry_cpp PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

# Include paths
target_include_directories(_llamatelemetry_cpp PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${CUDA_INCLUDE_DIRS}"
)

# =========================================================================
# COMPILATION FLAGS
# =========================================================================

# C++ flags for all quantization types
target_compile_options(_llamatelemetry_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fPIC>
)

# CUDA flags
target_compile_options(_llamatelemetry_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -gencode=arch=compute_75,code=sm_75
        --ptxas-options=-v
        -Xcompiler -fPIC
    >
)

# =========================================================================
# OUTPUT CONFIGURATION
# =========================================================================

set_target_properties(_llamatelemetry_cpp PROPERTIES
    VERSION "2.0.0"
    SOVERSION "2"
)

# =========================================================================
# INSTALLATION
# =========================================================================

install(TARGETS _llamatelemetry_cpp
    LIBRARY DESTINATION llamatelemetry
)
```

### File: Root `CMakeLists.txt` (UPDATE)

```cmake
cmake_minimum_required(VERSION 3.20)
project(llamatelemetry CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable testing
enable_testing()

# Add C++ bindings subdirectory
add_subdirectory(csrc)

# Optional: Add examples
add_subdirectory(examples EXCLUDE_FROM_ALL)
```

---

## Part 4: Python Setup Configuration

### File: `setup.py` (UPDATE)

```python
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CMakeBuildExt(build_ext):
    """Build C++ extensions using CMake"""

    def build_extensions(self):
        # Configure CMake
        cmake_args = [
            f"-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DGGML_CUDA=ON",
            "-DLLAMA_CUDA_FA_ALL_QUANTS=ON",
        ]

        # Kaggle-specific configuration
        if os.path.exists("/kaggle"):
            cuda_stubs = os.path.join(os.path.dirname(__file__), "cuda_stubs")
            cmake_args.extend([
                f"-DCMAKE_LIBRARY_PATH={cuda_stubs}",
                "-DGGML_CUDA_NO_VMM=ON",
            ])

        # Configure
        subprocess.check_call(["cmake", "-B", "build"] + cmake_args)

        # Build
        subprocess.check_call([
            "cmake", "--build", "build",
            "--config", "Release",
            "-j", str(os.cpu_count())
        ])

        # Installation handled by cmake install

setup(
    name="llamatelemetry",
    version="2.0.0",
    packages=find_packages(exclude=["tests"]),

    python_requires=">=3.11",

    install_requires=[
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-semantic-conventions>=0.40b0",
        "opentelemetry-exporter-otlp-proto-http>=0.40b0",
        "pynvml>=11.0.0",
        "huggingface_hub>=0.20.0",
    ],

    extras_require={
        "kaggle": [
            "cudf-cu12>=25.6",       # Pre-installed on Kaggle
            "cugraph-cu12==25.6.*",  # Match Kaggle's RAPIDS version
            "graphistry>=2.0",
        ],
        "cuda": [
            "pycuda>=2021.1",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },

    ext_modules=[],  # Handled by CMake
    cmdclass={"build_ext": CMakeBuildExt},

    author="LlamaTelemetry Contributors",
    author_email="dev@llamatelemetry.org",
    description="CUDA-first OpenTelemetry Python SDK for GGUF LLM inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/llamatelemetry/llamatelemetry",
    project_urls={
        "Documentation": "https://llamatelemetry.org",
        "Source": "https://github.com/llamatelemetry/llamatelemetry",
        "Issues": "https://github.com/llamatelemetry/llamatelemetry/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
)
```

### File: `setup.cfg` (CREATE)

```ini
[metadata]
name = llamatelemetry
version = 2.0.0
author = LlamaTelemetry Contributors
author_email = dev@llamatelemetry.org
description = CUDA-first OpenTelemetry Python SDK for GGUF LLM inference
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/llamatelemetry/llamatelemetry
project_urls =
    Documentation = https://llamatelemetry.org
    Source = https://github.com/llamatelemetry/llamatelemetry
    Issues = https://github.com/llamatelemetry/llamatelemetry/issues

[options]
python_requires = >=3.11
packages = find:
include_package_data = True

[options.packages.find]
exclude =
    tests*
    docs*
    examples*

[options.entry_points]
console_scripts =
    llamatelemetry-server = llamatelemetry.cli:main_server
    llamatelemetry-quantize = llamatelemetry.cli:main_quantize

[bdist_wheel]
universal = False
```

---

## Part 5: Build Process on Kaggle

### Step-by-Step Build

#### Step 1: Prepare Environment (Kaggle Notebook Cell 1)

```python
import subprocess
import os

# Verify GPUs
print("Checking GPUs...")
subprocess.run(["nvidia-smi", "-L"])
subprocess.run(["nvcc", "--version"])

# Install build dependencies (pre-installed on Kaggle 2× T4)
subprocess.run(["cmake", "--version"])
subprocess.run(["ninja", "--version"])

# Create CUDA stub directory
os.makedirs("/kaggle/working/cuda_stubs", exist_ok=True)

# Create stub library
stub_code = """
void* cuInit = 0;
void* cuMemAlloc = 0;
// ... 30+ more symbols (see p2/p3 notebooks)
"""

with open("/kaggle/working/cuda_stubs/cuda_stub.c", "w") as f:
    f.write(stub_code)

os.system("gcc -shared -fPIC -o /kaggle/working/cuda_stubs/libcuda.so /kaggle/working/cuda_stubs/cuda_stub.c")

print("✅ CUDA stub created")
```

#### Step 2: Clone Repository (Cell 2)

```python
os.chdir("/kaggle/working")

# Get llamatelemetry v2.0 with Python APIs
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/llamatelemetry/llamatelemetry.git"])

os.chdir("llamatelemetry")
print("✅ LlamaTelemetry v2.0 cloned")
```

#### Step 3: Configure & Build (Cell 3)

```python
%time

os.chdir("/kaggle/working/llamatelemetry")

# Configure
cmake_cmd = """
cmake -B build -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="75" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_NO_VMM=ON \
  -DCMAKE_LIBRARY_PATH="/kaggle/working/cuda_stubs"
"""

os.system(cmake_cmd)

# Verify build.ninja exists
if os.path.exists("build/build.ninja"):
    print("✅ CMake configuration successful")
else:
    print("❌ CMake failed")

# Build
os.system(f"cmake --build build --config Release -j$(nproc)")

# Verify
if os.path.exists("build/lib/llamatelemetry/_llamatelemetry_cpp.so"):
    print("✅ BUILD COMPLETE")
else:
    print("❌ BUILD FAILED")
```

#### Step 4: Test & Package (Cell 4)

```python
# Test import
import sys
sys.path.insert(0, "/kaggle/working/llamatelemetry/build/lib")

try:
    from llamatelemetry import _llamatelemetry_cpp
    print(f"✅ Module imported: {_llamatelemetry_cpp}")

    # Test llama.cpp API
    model = _llamatelemetry_cpp.LlamaModel("model.gguf", n_gpu_layers=99)
    context = _llamatelemetry_cpp.LlamaContext(model, n_ctx=2048)
    print(f"✅ Llama APIs working")

    # Test NCCL if available
    try:
        comm = _llamatelemetry_cpp.NCCLCommunicator()
        print(f"✅ NCCL APIs working")
    except:
        print("⚠️  NCCL not available")

except ImportError as e:
    print(f"❌ Import failed: {e}")

# Package
os.system("pip install -e /kaggle/working/llamatelemetry")
print("✅ Package installed")
```

---

## Part 6: Expected Output

### Build Artifacts

```
llamatelemetry/
├── build/
│   ├── lib/
│   │   └── llamatelemetry/
│   │       └── _llamatelemetry_cpp.so  (3-5 MB)
│   ├── bin/
│   │   ├── llama-server
│   │   ├── llama-cli
│   │   └── ... other tools
│   └── build.ninja
├── csrc/
│   ├── bindings.cpp
│   ├── llama_bindings.cpp
│   ├── nccl_bindings.cpp
│   └── CMakeLists.txt
└── (Python v2.0 modules already exist)
```

### Verification

```bash
# Check module
file build/lib/llamatelemetry/_llamatelemetry_cpp.so
# Output: ELF 64-bit LSB shared object, x86-64, dynamically linked,
#         stripped, not SUID, BuildID[sha1]=..., for GNU/Linux 5.15

# Check CUDA support
ldd build/lib/llamatelemetry/_llamatelemetry_cpp.so | grep -i cuda
# Output: libcuda.so.1 => /usr/local/nvidia/lib/libcuda.so.1
```

---

## Part 7: Performance Expectations

| Metric | Expected Value |
|--------|---|
| **Build Time** | 8-15 minutes (Kaggle 2× T4) |
| **Module Size** | 3-5 MB (_llamatelemetry_cpp.so) |
| **Prefill Throughput** | 500-1000 tokens/sec (Llama 2 13B Q4_K_M) |
| **Decode Throughput** | 100-200 tokens/sec |
| **TTFT** | 250-500 ms (256 input tokens) |
| **TPOT** | 5-15 ms per output token |
| **GPU Memory** | ~7.4 GB per T4 (13B Q4_K_M) |

---

## Part 8: Troubleshooting

### CMake GENERATE Error: "Target `CUDA::cuda_driver` links to … but target not found"

**Cause**: CMakeLists patch not applied correctly

**Fix**: Check that `ggml/src/ggml-cuda/CMakeLists.txt` has the pybind11 patch at the top of the file

### Build Fails: "cuda_stub.c: No such file or directory"

**Cause**: CUDA stub not created

**Fix**: Run Cell 1 of Kaggle notebook to create `/kaggle/working/cuda_stubs/libcuda.so`

### Import Fails: "cannot open shared object file: No such file"

**Cause**: LD_LIBRARY_PATH not set

**Fix**:
```python
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
import llamatelemetry
```

### NCCL Collectives Fail

**Cause**: NCCL not compiled/linked

**Fix**: Ensure CMake found NCCL: `cmake -B build ... -DNCCL_DIR=/path/to/nccl`

---

## Summary

This guide integrates the proven Kaggle build strategy with LlamaTelemetry v2.0's Python APIs to create:

1. **Native pybind11 Module** (`_llamatelemetry_cpp.so`) wrapping 100+ llama.cpp + 50+ NCCL APIs
2. **Seamless Python Integration** with existing v2.0 Python wrappers
3. **Kaggle-Specific Handling** for read-only /usr/local/cuda, CUDA stub linking, and VMM disabled
4. **Dual T4 GPU Support** with tensor-split and workload-split modes
5. **Complete OpenTelemetry Observability** (45 attributes, 5 metrics)

**Status**: Ready for implementation

