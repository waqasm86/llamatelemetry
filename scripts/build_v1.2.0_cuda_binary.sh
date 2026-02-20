#!/bin/bash
################################################################################
# Build CUDA 12 Binary Bundle for llamatelemetry v1.2.0
#
# Target: Kaggle dual Tesla T4 (SM 7.5, CUDA 12)
# Output: llamatelemetry-v1.2.0-cuda12-kaggle-t4x2.tar.gz
#
# REQUIREMENTS:
#   - CUDA 12.x toolkit (nvcc)
#   - Tesla T4 or compatible GPU (SM 7.5+)
#   - cmake >= 3.24
#   - ~2 GB disk space for build
#
# This script is designed to run on a Kaggle T4 instance or any machine
# with CUDA 12 and SM 7.5+ GPU. It will NOT work on machines without
# a CUDA toolkit (e.g., GeForce 940M dev machine).
#
# Usage:
#   chmod +x scripts/build_v1.2.0_cuda_binary.sh
#   ./scripts/build_v1.2.0_cuda_binary.sh [LLAMA_CPP_DIR]
#
# Arguments:
#   LLAMA_CPP_DIR  Path to llama.cpp source (default: ../llama.cpp)
################################################################################

set -euo pipefail

# Configuration
VERSION="1.2.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="${1:-$(dirname "$PROJECT_ROOT")/llama.cpp}"
BUILD_DIR="${LLAMA_CPP_DIR}/build_cuda12_t4_v1"
OUTPUT_DIR="${PROJECT_ROOT}/releases/v${VERSION}"
ARCHIVE_NAME="llamatelemetry-v${VERSION}-cuda12-kaggle-t4x2.tar.gz"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }

################################################################################
# Pre-flight checks
################################################################################

echo "============================================================"
echo "  llamatelemetry v${VERSION} - CUDA Binary Builder"
echo "  Target: Kaggle dual Tesla T4 (SM 7.5, CUDA 12)"
echo "============================================================"
echo ""

# Check CUDA toolkit
if ! command -v nvcc &>/dev/null; then
    log_err "nvcc not found. CUDA toolkit is required."
    log_err "This script must run on a machine with CUDA 12 toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
log_info "CUDA toolkit: ${CUDA_VERSION}"

# Check nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    log_err "nvidia-smi not found. NVIDIA GPU required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
log_info "GPU: ${GPU_NAME} (SM ${GPU_CC})"

# Check cmake
if ! command -v cmake &>/dev/null; then
    log_err "cmake not found. Install with: pip install cmake"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -1 | sed 's/cmake version //')
log_info "cmake: ${CMAKE_VERSION}"

# Check llama.cpp source
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    log_err "llama.cpp not found at: $LLAMA_CPP_DIR"
    log_err "Usage: $0 [LLAMA_CPP_DIR]"
    exit 1
fi
log_ok "llama.cpp source: ${LLAMA_CPP_DIR}"

echo ""

################################################################################
# Configure
################################################################################

log_info "Configuring CMake build..."

cmake -B "$BUILD_DIR" -S "$LLAMA_CPP_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

log_ok "CMake configuration complete"

################################################################################
# Build
################################################################################

NPROC=$(nproc 2>/dev/null || echo 2)
log_info "Building with ${NPROC} cores (this may take 5-15 minutes)..."

cmake --build "$BUILD_DIR" --config Release -j"$NPROC"

log_ok "Build complete"

# Verify essential binary
if [ ! -f "$BUILD_DIR/bin/llama-server" ]; then
    log_err "llama-server not found after build"
    exit 1
fi

################################################################################
# Package
################################################################################

log_info "Creating binary package..."

TEMP_DIR=$(mktemp -d)
PKG_DIR="${TEMP_DIR}/llamatelemetry-v${VERSION}-cuda12-t4x2"
mkdir -p "${PKG_DIR}/bin" "${PKG_DIR}/lib"

# Copy binaries
BINARIES=(
    llama-server llama-cli llama-quantize llama-gguf-hash
    llama-embedding llama-bench llama-cvector-generator
    llama-lookup-create llama-lookup-merge llama-lookup-stats
    llama-perplexity llama-imatrix llama-export-lora
)

BIN_COUNT=0
for binary in "${BINARIES[@]}"; do
    if [ -f "$BUILD_DIR/bin/${binary}" ]; then
        cp "$BUILD_DIR/bin/${binary}" "${PKG_DIR}/bin/"
        chmod +x "${PKG_DIR}/bin/${binary}"
        BIN_COUNT=$((BIN_COUNT + 1))
    fi
done
log_ok "Copied ${BIN_COUNT} binaries"

# Copy shared libraries from bin/ and lib/
LIB_COUNT=0
for dir in "$BUILD_DIR/bin" "$BUILD_DIR/lib" "$BUILD_DIR/ggml/src"; do
    if [ -d "$dir" ]; then
        for lib in "$dir"/*.so*; do
            [ -f "$lib" ] || continue
            LIBNAME=$(basename "$lib")
            if [ ! -e "${PKG_DIR}/lib/${LIBNAME}" ]; then
                cp -a "$lib" "${PKG_DIR}/lib/"
                LIB_COUNT=$((LIB_COUNT + 1))
            fi
        done
    fi
done
log_ok "Copied ${LIB_COUNT} libraries"

# Copy NCCL library if available
NCCL_LIB="/usr/lib/x86_64-linux-gnu/libnccl.so.2"
if [ -f "$NCCL_LIB" ]; then
    cp -a "$NCCL_LIB" "${PKG_DIR}/lib/"
    log_ok "Included NCCL library"
fi

# Create build manifest
cat > "${PKG_DIR}/BUILD_INFO.json" <<EOF
{
    "version": "${VERSION}",
    "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "cuda_version": "${CUDA_VERSION}",
    "gpu": "${GPU_NAME}",
    "compute_capability": "${GPU_CC}",
    "cmake_version": "${CMAKE_VERSION}",
    "target": "kaggle-t4x2",
    "sm_arch": "75",
    "features": {
        "flash_attention": true,
        "cuda_graphs": true,
        "tensor_cores": true,
        "multi_gpu": true,
        "shared_libs": true
    },
    "binaries": ${BIN_COUNT},
    "libraries": ${LIB_COUNT}
}
EOF

# Create archive
mkdir -p "$OUTPUT_DIR"
ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}"

log_info "Creating archive: ${ARCHIVE_NAME}"
cd "$TEMP_DIR"
tar -czf "$ARCHIVE_PATH" "$(basename "$PKG_DIR")"

# Compute SHA256
SHA256=$(sha256sum "$ARCHIVE_PATH" | cut -d' ' -f1)
echo "$SHA256  $ARCHIVE_NAME" > "${ARCHIVE_PATH}.sha256"

# Cleanup
rm -rf "$TEMP_DIR"

################################################################################
# Summary
################################################################################

ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)

echo ""
echo "============================================================"
echo "  Build Complete!"
echo "============================================================"
echo ""
echo "  Archive:  ${ARCHIVE_NAME}"
echo "  Size:     ${ARCHIVE_SIZE}"
echo "  Location: ${ARCHIVE_PATH}"
echo "  SHA256:   ${SHA256}"
echo "  Binaries: ${BIN_COUNT}"
echo "  Libraries: ${LIB_COUNT}"
echo ""
echo "  Next steps:"
echo "    1. Update bootstrap.py BINARY_CHECKSUMS with SHA256"
echo "    2. Upload to HuggingFace: waqasm86/llamatelemetry-binaries"
echo "    3. Upload to GitHub Releases: v${VERSION}"
echo ""
log_ok "Done!"
