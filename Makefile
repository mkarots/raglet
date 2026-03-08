.PHONY: help install install-dev test test-unit test-integration test-e2e test-performance lint format format-check type-check coverage coverage-ci clean build dist publish publish-test ci venv check-uv benchmark benchmark-sweep docker-build docker-push docker-release smoke smoke-dev


# Check if uv is available (checks at runtime, works even if uv was installed after Makefile was parsed)
check-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi

help:
	@echo "Available targets:"
	@echo "  venv            - Create virtual environment (.venv)"
	@echo "  install          - Install package and dependencies"
	@echo "  install-dev     - Install package with dev dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-debug      - Run tests with output visible (-s -v flags)"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e        - Run end-to-end tests only"
	@echo "  test-performance - Run storage format performance tests"
	@echo "  benchmark       - Run embedding throughput benchmark"
	@echo "  benchmark-sweep - Run all benchmarks with parameter sweeps (YAML config)"
	@echo "  lint            - Run linters (ruff)"
	@echo "  format          - Format code (black)"
	@echo "  format-check    - Check formatting without modifying"
	@echo "  type-check      - Run type checker (mypy)"
	@echo "  coverage        - Generate coverage report"
	@echo "  coverage-ci     - Generate coverage report for CI"
	@echo "  clean           - Clean build artifacts"
	@echo "  build           - Build package"
	@echo "  dist            - Create distribution"
	@echo "  publish-test    - Publish to TestPyPI (requires twine)"
	@echo "  publish         - Publish to PyPI (requires twine)"
	@echo "  docker-build    - Build Docker image (local)"
	@echo "  docker-push      - Push Docker image to Docker Hub (requires docker login)"
	@echo "  docker-release  - Build and push Docker image with version tag"
	@echo "  smoke           - Run smoke tests against PyPI package (Docker)"
	@echo "  smoke-dev       - Run smoke tests against local dev build (Docker)"
	@echo "  ci              - Run full CI pipeline"
	@echo ""
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv: $$(command -v uv)"; \
	else \
		echo "⚠️  uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	fi

# Create virtual environment if it doesn't exist
# uv run automatically uses .venv if it exists, or creates one if needed
venv: check-uv
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	else \
		echo "Virtual environment already exists at .venv"; \
	fi

# Install package (production dependencies only)
# uv pip install will use .venv if it exists, or create one automatically
install: venv check-uv
	uv pip install -e .

# Install package with dev dependencies
# Stamps __version__ with a PEP 440 local version derived from git describe.
# e.g. "0.2.0+6.g7a2c883.dirty".  The stamped __init__.py is left in place
# (editable installs read it at runtime).  `make clean` or `git checkout`
# restores the release version.
#
# Conversion: git describe "v0.2.0-6-g7a2c883-dirty"
#   → strip leading v       → "0.2.0-6-g7a2c883-dirty"
#   → first hyphen becomes + → "0.2.0+6-g7a2c883-dirty"
#   → remaining hyphens → .  → "0.2.0+6.g7a2c883.dirty"  (PEP 440 compliant)
DEV_VERSION := $(shell git describe --tags --always --dirty 2>/dev/null | sed 's/^v//; s/-/+/; s/-/./g')

install-dev: venv check-uv
	@if [ -n "$(DEV_VERSION)" ]; then \
		sed -i '' 's/^__version__ = .*/__version__ = "$(DEV_VERSION)"/' raglet/__init__.py; \
		echo "Stamped version: $(DEV_VERSION)"; \
	fi
	uv pip install -e ".[dev,all]"

# All commands below use 'uv run' which automatically:
# - Uses .venv if it exists
# - Creates .venv if it doesn't exist
# - Installs dependencies on first run
# No manual venv activation needed!

test: check-uv
	uv run pytest

test-debug: check-uv
	@echo "Running tests with output visible (use -s flag)..."
	uv run pytest -s -v

test-unit: check-uv
	uv run pytest -m unit

test-integration: check-uv
	uv run pytest -m integration

test-integration-debug: check-uv
	uv run pytest -m integration -s -v

test-e2e: check-uv
	uv run pytest -m e2e

test-performance-local: check-uv
	@echo "Running storage performance tests..."
	uv run python benchmarks/storage-performance/run.py \
		--sizes 0.1 1.0 10.0 \
		--formats sqlite directory

benchmark: check-uv
	@echo "Running embedding throughput benchmark..."
	uv run python benchmarks/embedding-throughput/run.py

# Sweep: run benchmarks with parameter grids from YAML config
# Usage: make benchmark-sweep
#        make benchmark-sweep SWEEP_CONFIG=benchmarks/sweep-quick.yaml
#        make benchmark-sweep SWEEP_ARGS="--only latency --dry-run"
SWEEP_CONFIG ?= benchmarks/sweep.yaml
benchmark-sweep: check-uv
	uv run python benchmarks/sweep.py --config $(SWEEP_CONFIG) $(SWEEP_ARGS)

lint: check-uv
	uv run ruff check raglet tests

format: check-uv
	uv run black raglet tests
	uv run ruff check --fix raglet tests

format-check: check-uv
	uv run black --check raglet tests
	uv run ruff check raglet tests

type-check: check-uv
	uv run mypy raglet

coverage: check-uv
	uv run pytest --cov=raglet --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

coverage-ci: check-uv
	uv run pytest --cov=raglet --cov-report=xml --cov-report=term-missing

clean:
	@git checkout -- raglet/__init__.py 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean check-uv
	uv build

dist: build
	@echo "Distribution created in dist/"

publish-test: build check-uv
	@echo "Publishing to TestPyPI..."
	uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish: build check-uv
	@echo "Publishing to PyPI..."
	uv run twine upload dist/*

ci: lint format-check type-check coverage-ci
	@echo "CI pipeline completed successfully"

# Docker commands
DOCKER_IMAGE := mkarots/raglet
VERSION := $(shell grep '^version =' pyproject.toml | head -1 | sed 's/version = "\([^"]*\)".*/\1/')

docker-build: check-uv
	@echo "Building Docker image: $(DOCKER_IMAGE):$(VERSION)"
	docker build -t $(DOCKER_IMAGE):$(VERSION) .
	docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_IMAGE):latest
	@echo "✓ Build complete: $(DOCKER_IMAGE):$(VERSION) and $(DOCKER_IMAGE):latest"

docker-push: docker-build
	@echo "Pushing Docker image to Docker Hub..."
	@if ! docker info >/dev/null 2>&1; then \
		echo "Error: Docker daemon not running or not logged in"; \
		echo "Run: docker login"; \
		exit 1; \
	fi
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest
	@echo "✓ Pushed $(DOCKER_IMAGE):$(VERSION) and $(DOCKER_IMAGE):latest"

docker-release: docker-push
	@echo "✓ Docker release complete: $(DOCKER_IMAGE):$(VERSION)"

# Smoke tests
# Pass flags via SMOKE_ARGS, e.g.:  make smoke SMOKE_ARGS="--verbose"
smoke:
	@echo "Building smoke test image (installs from PyPI)..."
	docker build -t raglet-smoke ./smoke
	@echo "Running smoke tests..."
	docker run --rm raglet-smoke $(SMOKE_ARGS)

smoke-dev: build
	@echo "Building smoke test image (local dev build)..."
	docker build -t raglet-smoke-dev -f smoke/Dockerfile.dev .
	@echo "Running smoke tests against local build..."
	docker run --rm raglet-smoke-dev $(SMOKE_ARGS)
