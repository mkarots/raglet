# Release Process

This document describes how to release a new version of `raglet` to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **Trusted Publishing** (Recommended): Set up trusted publishing in PyPI for automated releases
3. **GitHub Access**: Ability to create and push tags
4. **Local Setup**: Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Initial Setup

### Setting Up Trusted Publishing (Recommended)

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI project name**: `raglet`
   - **Owner**: Your GitHub username/organization
   - **Repository name**: `raglet` (or your repo name)
   - **Workflow filename**: `.github/workflows/release.yml`
   - **Environment name**: (leave empty for default)
4. Click "Add"
5. The workflow will automatically use trusted publishing - no API tokens needed!

### Alternative: Using API Tokens

If you prefer API tokens:

1. Go to https://pypi.org/manage/account/token/
2. Create an API token with "Upload packages" scope
3. Add it as a GitHub secret: `PYPI_API_TOKEN`
4. Update `.github/workflows/release.yml` to use the token instead of trusted publishing

## Release Steps

### Option 1: Git Tag Release (Recommended)

This is the simplest and most automated method:

1. **Update Version**:
   ```bash
   # Edit raglet/__init__.py
   # Change: __version__ = "0.1.0"
   ```

2. **Commit Version Change**:
   ```bash
   git add raglet/__init__.py
   git commit -m "Bump version to 0.1.0"
   git push origin main  # or your default branch
   ```

3. **Create and Push Tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. **Automated Publishing**:
   - The `.github/workflows/release.yml` workflow will automatically:
     - Extract version from tag (`v0.1.0` → `0.1.0`)
     - Verify version matches `raglet.__version__`
     - Build the package (wheel and sdist)
     - Validate with `twine check`
     - Publish to PyPI
   - Monitor progress: GitHub → Actions → Release workflow

### Option 2: GitHub Release

1. **Update Version**:
   - Update `__version__` in `raglet/__init__.py`
   - Commit and push to your default branch

2. **Create GitHub Release**:
   - Go to GitHub → Releases → Draft a new release
   - Create a new tag: `v0.1.0` (must start with `v`)
   - Select target branch (usually `main`)
   - Write release notes
   - Click "Publish release"

3. **Automated Publishing**:
   - The workflow triggers when the tag is pushed
   - Same automated process as Option 1

### Option 3: Manual Release

For testing or when automation isn't available:

1. **Update Version**:
   ```bash
   # Edit raglet/__init__.py
   __version__ = "0.1.0"
   ```

2. **Build Package**:
   ```bash
   make build
   # Creates dist/raglet-0.1.0-py3-none-any.whl
   # and dist/raglet-0.1.0.tar.gz
   ```

3. **Check Package**:
   ```bash
   uv run twine check dist/*
   # Should show: Checking dist/raglet-0.1.0-py3-none-any.whl: PASSED
   ```

4. **Test on TestPyPI** (Recommended first):
   ```bash
   # Create TestPyPI account at https://test.pypi.org/
   # Create API token at https://test.pypi.org/manage/account/token/
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=your-testpypi-token
   make publish-test
   ```

5. **Verify TestPyPI Installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ raglet
   ```

6. **Publish to PyPI**:
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=your-pypi-token
   make publish
   ```

7. **Create Git Tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., `1.2.3`)
- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.1` → `0.2.0`: New feature
- `0.2.0` → `1.0.0`: First stable release
- `1.0.0` → `1.0.1`: Bug fix
- `1.0.1` → `2.0.0`: Breaking change

## Pre-release Checklist

- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make type-check`)
- [ ] Formatting check passes (`make format-check`)
- [ ] Documentation is up to date (README.md)
- [ ] Version number updated in `raglet/__init__.py`
- [ ] Version matches semantic versioning
- [ ] All changes committed and pushed
- [ ] Release notes prepared (for GitHub release)

## Post-release

- [ ] Verify package on PyPI: https://pypi.org/project/raglet/
- [ ] Check package metadata (description, classifiers, etc.)
- [ ] Test installation: `pip install raglet`
- [ ] Verify version: `python -c "import raglet; print(raglet.__version__)"`
- [ ] Test basic functionality
- [ ] Update any downstream dependencies
- [ ] Announce release (if applicable)

## Troubleshooting

### Workflow Fails: "Version mismatch"

**Error**: Package version does not match tag version

**Solution**: Ensure `raglet/__init__.py` has the correct version before pushing the tag:
```bash
# Check current version
python -c "import raglet; print(raglet.__version__)"
# Should match your tag (without 'v' prefix)
```

### Workflow Fails: "Permission denied"

**Error**: Cannot publish to PyPI

**Solution**: 
- Verify trusted publishing is set up correctly in PyPI
- Check GitHub Actions has `id-token: write` permission
- Ensure repository name matches PyPI trusted publishing config

### Build Fails: "No module named 'build'"

**Solution**: Install dev dependencies:
```bash
make install-dev
```

### Tag Already Exists

**Error**: `fatal: tag 'v0.1.0' already exists`

**Solution**: Delete and recreate:
```bash
git tag -d v0.1.0           # Delete local tag
git push origin :refs/tags/v0.1.0  # Delete remote tag
git tag v0.1.0             # Create new tag
git push origin v0.1.0     # Push new tag
```

### TestPyPI vs PyPI

- **TestPyPI**: Use for testing the release process
- **PyPI**: Production releases
- Packages uploaded to TestPyPI are separate from PyPI
- You can upload the same version to both

## Quick Reference

```bash
# Full release process (recommended)
vim raglet/__init__.py  # Update version
git add raglet/__init__.py
git commit -m "Bump version to X.Y.Z"
git push origin main
git tag vX.Y.Z
git push origin vX.Y.Z

# Check workflow status
# GitHub → Actions → Release workflow

# Verify release
pip install raglet
python -c "import raglet; print(raglet.__version__)"
```
