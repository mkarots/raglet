.PHONY: help install install-dev test test-unit test-integration test-e2e lint format format-check type-check coverage coverage-ci clean build dist publish publish-test ci venv check-uv


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
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e        - Run end-to-end tests only"
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
# uv pip install will use .venv if it exists, or create one automatically
install-dev: venv check-uv
	uv pip install -e ".[dev,all]"

# All commands below use 'uv run' which automatically:
# - Uses .venv if it exists
# - Creates .venv if it doesn't exist
# - Installs dependencies on first run
# No manual venv activation needed!

test: check-uv
	uv run pytest

test-unit: check-uv
	uv run pytest -m unit

test-integration: check-uv
	uv run pytest -m integration

test-e2e: check-uv
	uv run pytest -m e2e

lint: check-uv
	uv run ruff check tinyrag tests

format: check-uv
	uv run black tinyrag tests
	uv run ruff check --fix tinyrag tests

format-check: check-uv
	uv run black --check tinyrag tests
	uv run ruff check tinyrag tests

type-check: check-uv
	uv run mypy tinyrag

coverage: check-uv
	uv run pytest --cov=tinyrag --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

coverage-ci: check-uv
	uv run pytest --cov=tinyrag --cov-report=xml --cov-report=term-missing

clean:
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

ci: lint type-check test
	@echo "CI pipeline completed successfully"
