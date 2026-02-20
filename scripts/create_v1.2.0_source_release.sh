#!/bin/bash
################################################################################
# Create Source Release Archives for llamatelemetry v1.2.0
#
# Output:
#   releases/v1.2.0/llamatelemetry-v1.2.0-source.tar.gz
#   releases/v1.2.0/llamatelemetry-v1.2.0-source.zip
#
# Excludes:
#   - binaries/, lib/, models/ (large binary artifacts)
#   - .git/ (VCS history)
#   - __pycache__/, *.pyc (build artifacts)
#   - archive/ (old releases)
#   - *.gguf (model files)
#   - *.tar.gz, *.zip within the tree (nested archives)
#   - releases/v1.2.0/ (old release artifacts)
#
# Usage:
#   chmod +x scripts/create_v1.2.0_source_release.sh
#   ./scripts/create_v1.2.0_source_release.sh
################################################################################

set -euo pipefail

# Configuration
VERSION="1.2.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/releases/v${VERSION}"
ARCHIVE_BASE="llamatelemetry-v${VERSION}-source"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }

echo "============================================================"
echo "  llamatelemetry v${VERSION} - Source Release Builder"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Create a temporary directory for staging
TEMP_DIR=$(mktemp -d)
STAGE_DIR="${TEMP_DIR}/${ARCHIVE_BASE}"
mkdir -p "$STAGE_DIR"

log_info "Staging source files..."

# Use rsync to copy source tree with exclusions
rsync -a --quiet \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.egg-info' \
    --exclude='.eggs' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='*.so' \
    --exclude='*.so.*' \
    --exclude='*.gguf' \
    --exclude='binaries/' \
    --exclude='lib/' \
    --exclude='models/' \
    --exclude='archive/' \
    --exclude='releases/v1.2.0/' \
    --exclude='.mypy_cache' \
    --exclude='.pytest_cache' \
    --exclude='.tox' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='notebooks-local/' \
    --exclude='.env' \
    --exclude='*.log' \
    "$PROJECT_ROOT/" "$STAGE_DIR/"

# Count files
FILE_COUNT=$(find "$STAGE_DIR" -type f | wc -l)
DIR_COUNT=$(find "$STAGE_DIR" -type d | wc -l)
log_ok "Staged ${FILE_COUNT} files in ${DIR_COUNT} directories"

# Create tar.gz
TAR_PATH="${OUTPUT_DIR}/${ARCHIVE_BASE}.tar.gz"
log_info "Creating ${ARCHIVE_BASE}.tar.gz..."
cd "$TEMP_DIR"
tar -czf "$TAR_PATH" "$ARCHIVE_BASE"
TAR_SIZE=$(du -h "$TAR_PATH" | cut -f1)
TAR_SHA256=$(sha256sum "$TAR_PATH" | cut -d' ' -f1)
echo "$TAR_SHA256  ${ARCHIVE_BASE}.tar.gz" > "${TAR_PATH}.sha256"
log_ok "tar.gz: ${TAR_SIZE} (SHA256: ${TAR_SHA256})"

# Create zip
ZIP_PATH="${OUTPUT_DIR}/${ARCHIVE_BASE}.zip"
log_info "Creating ${ARCHIVE_BASE}.zip..."
cd "$TEMP_DIR"
zip -rq "$ZIP_PATH" "$ARCHIVE_BASE"
ZIP_SIZE=$(du -h "$ZIP_PATH" | cut -f1)
ZIP_SHA256=$(sha256sum "$ZIP_PATH" | cut -d' ' -f1)
echo "$ZIP_SHA256  ${ARCHIVE_BASE}.zip" > "${ZIP_PATH}.sha256"
log_ok "zip: ${ZIP_SIZE} (SHA256: ${ZIP_SHA256})"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "============================================================"
echo "  Source Release Archives Created"
echo "============================================================"
echo ""
echo "  Location: ${OUTPUT_DIR}/"
echo ""
echo "  ${ARCHIVE_BASE}.tar.gz  (${TAR_SIZE})"
echo "    SHA256: ${TAR_SHA256}"
echo ""
echo "  ${ARCHIVE_BASE}.zip     (${ZIP_SIZE})"
echo "    SHA256: ${ZIP_SHA256}"
echo ""
echo "  Files included: ${FILE_COUNT}"
echo ""
log_ok "Done!"
