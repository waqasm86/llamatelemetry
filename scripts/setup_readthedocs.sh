#!/bin/bash
# ==============================================================================
# ReadTheDocs Setup Script for llamatelemetry
# ==============================================================================
# This script helps set up ReadTheDocs for the llamatelemetry project
#
# Prerequisites:
#   - GitHub account connected to ReadTheDocs
#   - Python 3.11+ installed
#   - Repository pushed to GitHub
# ==============================================================================

set -e  # Exit on error

echo "=========================================================================="
echo "📚 ReadTheDocs Setup for llamatelemetry"
echo "=========================================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Verify local build
echo "Step 1: Verifying local documentation build..."
echo ""

if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}❌ Python 3.11 not found${NC}"
    echo "Please install Python 3.11 first"
    exit 1
fi

echo "Installing MkDocs dependencies..."
python3.11 -m pip install --quiet --user mkdocs mkdocs-material pymdown-extensions markdown mkdocs-material-extensions

echo "Building documentation locally..."
if python3.11 -m mkdocs build; then
    echo -e "${GREEN}✅ Local documentation build successful!${NC}"
    echo ""
else
    echo -e "${RED}❌ Local build failed. Please fix errors before continuing.${NC}"
    exit 1
fi

# Step 2: Verify configuration files
echo "=========================================================================="
echo "Step 2: Verifying configuration files..."
echo ""

files=(
    ".readthedocs.yaml"
    "mkdocs.yml"
    "docs/requirements.txt"
    "docs/index.md"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file"
    else
        echo -e "${RED}❌${NC} $file (missing)"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo -e "${RED}❌ Some required files are missing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All configuration files present${NC}"
echo ""

# Step 3: GitHub repository check
echo "=========================================================================="
echo "Step 3: Checking GitHub repository status..."
echo ""

if git remote -v | grep -q "llamatelemetry/llamatelemetry"; then
    echo -e "${GREEN}✅ Repository remote configured correctly${NC}"
    echo "   Remote: https://github.com/llamatelemetry/llamatelemetry"
else
    echo -e "${RED}❌ Repository remote not configured correctly${NC}"
    exit 1
fi

if git status | grep -q "nothing to commit, working tree clean"; then
    echo -e "${GREEN}✅ Working tree clean${NC}"
elif git status | grep -q "Your branch is ahead"; then
    echo -e "${YELLOW}⚠️  Unpushed commits detected${NC}"
    echo "   Please push your changes before continuing"
    exit 1
else
    echo -e "${YELLOW}⚠️  Uncommitted changes detected${NC}"
    echo "   Please commit and push your changes before continuing"
    exit 1
fi

echo ""

# Step 4: ReadTheDocs setup instructions
echo "=========================================================================="
echo "Step 4: ReadTheDocs Setup Instructions"
echo "=========================================================================="
echo ""
echo "✅ Local verification complete! Now set up ReadTheDocs:"
echo ""
echo "1. Go to ReadTheDocs:"
echo "   https://readthedocs.org/dashboard/"
echo ""
echo "2. Click 'Import a Project'"
echo ""
echo "3. Connect GitHub account (if not already connected):"
echo "   - Click 'Connect to GitHub'"
echo "   - Authorize ReadTheDocs"
echo "   - Grant access to 'llamatelemetry' organization/repository"
echo ""
echo "4. Import llamatelemetry:"
echo "   - Find 'llamatelemetry/llamatelemetry' in the list"
echo "   - Click the '+' button"
echo ""
echo "5. Configure project:"
echo "   Project name: llamatelemetry"
echo "   Project slug: llamatelemetry"
echo "   Default branch: main"
echo "   Documentation type: MkDocs"
echo ""
echo "6. Build the documentation:"
echo "   - Go to 'Builds'"
echo "   - Click 'Build Version: latest'"
echo "   - Wait for build to complete"
echo ""
echo "7. Your documentation will be available at:"
echo "   https://llamatelemetry.readthedocs.io/en/latest/"
echo ""
echo "=========================================================================="
echo "✅ Setup verification complete!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  1. Follow the instructions above to import the project on ReadTheDocs"
echo "  2. Once the build succeeds, add the badge to README.md:"
echo ""
echo "     [![Documentation](https://readthedocs.org/projects/llamatelemetry/badge/?version=latest)]"
echo "     (https://llamatelemetry.readthedocs.io/en/latest/)"
echo ""
echo "For troubleshooting, see: docs/TROUBLESHOOTING.md"
echo ""
