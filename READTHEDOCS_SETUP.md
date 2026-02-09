# ReadTheDocs Setup Guide for llamatelemetry

This guide provides step-by-step instructions to set up ReadTheDocs for the llamatelemetry project.

---

## ✅ Prerequisites Completed

The following have been configured and pushed to GitHub:

- ✅ `.readthedocs.yaml` - ReadTheDocs build configuration
- ✅ `mkdocs.yml` - MkDocs configuration with Material theme
- ✅ `docs/requirements.txt` - Python dependencies for docs
- ✅ `docs/index.md` - Enhanced documentation landing page
- ✅ All documentation files properly organized
- ✅ Local build tested and verified

---

## 🚀 Quick Setup

### Option 1: Automated Verification (Recommended)

Run the setup script to verify everything is ready:

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/llamatelemetry
./scripts/setup_readthedocs.sh
```

This will:
1. Verify local documentation builds correctly
2. Check all configuration files exist
3. Verify GitHub repository status
4. Provide step-by-step ReadTheDocs instructions

### Option 2: Manual Setup

Follow the manual steps below.

---

## 📋 Manual Setup Steps

### Step 1: Delete Old Project (if exists)

If you have the "Muhammad Waqas" project:

1. Go to: https://app.readthedocs.org/projects/muhammad-waqas/
2. Click **Admin** → **Settings**
3. Scroll to bottom → **Delete project**

### Step 2: Import llamatelemetry Repository

1. **Go to ReadTheDocs Dashboard:**
   ```
   https://readthedocs.org/dashboard/
   ```

2. **Click "Import a Project"**

3. **Connect GitHub (if needed):**
   - Click "Connect to GitHub"
   - Authorize ReadTheDocs
   - Grant access to `llamatelemetry` organization/repository

4. **Import the Repository:**
   - Find `llamatelemetry/llamatelemetry` in the list
   - Click the **"+"** button next to it

5. **Configure Project:**
   ```
   Project name: llamatelemetry
   Project slug: llamatelemetry
   Repository URL: https://github.com/llamatelemetry/llamatelemetry.git
   Default branch: main
   Default version: latest
   Documentation type: MkDocs
   ```

6. **Click "Finish"**

### Step 3: Verify Integration

1. **Go to Admin → Integrations:**
   ```
   https://readthedocs.org/dashboard/llamatelemetry/integrations/
   ```

2. **Should see:**
   - ✅ "GitHub incoming webhook" - Active
   - ✅ "GitHub App" - Connected

3. **If webhook missing:**
   - Click "Add integration"
   - Select "GitHub incoming webhook"
   - Click "Create"

### Step 4: Configure Advanced Settings

1. **Go to Admin → Advanced Settings:**
   ```
   https://readthedocs.org/dashboard/llamatelemetry/advanced/
   ```

2. **Set:**
   - ✅ Default branch: `main`
   - ✅ Default version: `latest`
   - ✅ Documentation type: `mkdocs`
   - ✅ Python interpreter: `CPython 3.x`
   - ✅ Install project in virtualenv: ✓

3. **Click "Save"**

### Step 5: Activate Versions

1. **Go to Versions:**
   ```
   https://readthedocs.org/dashboard/llamatelemetry/versions/
   ```

2. **Activate:**
   - ✅ `latest` (from main branch) - Set as default
   - ✅ `stable` (optional, for tagged releases)

### Step 6: Build Documentation

1. **Go to Builds:**
   ```
   https://readthedocs.org/projects/llamatelemetry/builds/
   ```

2. **Click "Build Version: latest"**

3. **Wait for build (1-2 minutes)**

4. **Expected result:** ✅ **Passed** (green checkmark)

### Step 7: Verify Live Documentation

Open your documentation:
```
https://llamatelemetry.readthedocs.io/en/latest/
```

You should see:
- ✅ llamatelemetry Documentation landing page
- ✅ Material theme with dark/light mode toggle
- ✅ Navigation tabs (Getting Started, Core Documentation, etc.)
- ✅ Search functionality
- ✅ All pages load correctly

---

## 🔧 Troubleshooting

### Build Failed

**Check build logs:**
1. Go to Builds → Click failed build
2. Review error messages

**Common solutions:**
- Configuration already fixed ✅
- Local build verified ✅
- All dependencies in `docs/requirements.txt` ✅

### Webhook Not Created

**Manually add webhook:**

1. Go to GitHub:
   ```
   https://github.com/llamatelemetry/llamatelemetry/settings/hooks
   ```

2. Click "Add webhook"

3. Configure:
   ```
   Payload URL: https://readthedocs.org/api/v2/webhook/llamatelemetry/32549266/
   Content type: application/json
   Events: Just the push event
   ```

4. Click "Add webhook"

### Integration Type Error

This occurs when:
- Project not properly linked to GitHub repository
- Solution: Delete old project and create new one (see Step 1)

---

## 📦 Configuration Files

### `.readthedocs.yaml`
```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.11"
python:
  install:
    - requirements: docs/requirements.txt
mkdocs:
  configuration: mkdocs.yml
```

### `docs/requirements.txt`
```
mkdocs>=1.5.3
mkdocs-material>=9.5.0
pymdown-extensions>=10.7
mkdocs-material-extensions>=1.3
markdown>=3.5
```

### `mkdocs.yml`
- Material theme with dark/light mode
- Organized navigation structure
- Code highlighting and copy buttons
- Search with suggestions
- Responsive design

---

## 🎯 Expected Result

After successful setup:

**Documentation URL:** https://llamatelemetry.readthedocs.io/

**Features:**
- ✅ Material Design theme
- ✅ Dark/light mode toggle
- ✅ Organized navigation with tabs
- ✅ Full-text search
- ✅ Mobile responsive
- ✅ Code syntax highlighting
- ✅ Auto-builds on every push to main

**Badge for README:**
```markdown
[![Documentation Status](https://readthedocs.org/projects/llamatelemetry/badge/?version=latest)](https://llamatelemetry.readthedocs.io/en/latest/?badge=latest)
```

---

## 📞 Support

If you encounter issues:

1. **Check build logs** on ReadTheDocs
2. **Verify local build** works: `python3.11 -m mkdocs build`
3. **Review** this guide and `docs/TROUBLESHOOTING.md`
4. **Check** GitHub repository settings and webhooks

---

## ✅ Verification Checklist

Before considering setup complete:

- [ ] ReadTheDocs project created with slug `llamatelemetry`
- [ ] GitHub integration active (webhook + app)
- [ ] Latest version built successfully
- [ ] Documentation accessible at `llamatelemetry.readthedocs.io`
- [ ] Navigation works correctly
- [ ] Search functionality works
- [ ] All pages render properly
- [ ] Dark/light mode toggle works
- [ ] Code blocks have copy buttons
- [ ] Mobile responsive design works

---

**Last Updated:** 2026-02-09
**Status:** Configuration ready, awaiting ReadTheDocs project creation
