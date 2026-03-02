.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check coverage clean build dist ci

# Detect uv command
UV := $(shell command -v uv 2>/dev/null || echo "")


venv:
	uv venv

activate: venv
	. ./.venv/bin/activate

help:
	@echo "Available targets:"
	@echo "  install          - Install package and dependencies"
	@echo "  install-dev     - Install package with dev dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e        - Run end-to-end tests only"
	@echo "  lint            - Run linters (ruff)"
	@echo "  format          - Format code (black)"
	@echo "  type-check      - Run type checker (mypy)"
	@echo "  coverage        - Generate coverage report"
	@echo "  clean           - Clean build artifacts"
	@echo "  build           - Build package"
	@echo "  dist            - Create distribution"
	@echo "  ci              - Run full CI pipeline"
	@echo ""
	@if [ -z "$(UV)" ]; then \
		echo "⚠️  uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	else \
		echo "Using uv: $(UV)"; \
	fi

install:
	make activate
	$(UV) pip install -e .

install-dev:
	make activate
	$(UV) pip install -e ".[dev,all]"

test:
	$(UV) run pytest

test-unit:
	$(UV) run pytest -m unit

test-integration:
	$(UV) run pytest -m integration

test-e2e:
	$(UV) run pytest -m e2e

lint:
	$(UV) run ruff check tinyrag tests

format:
	$(UV) run black tinyrag tests
	$(UV) run ruff check --fix tinyrag tests

type-check:
	$(UV) run mypy tinyrag

coverage:
	$(UV) run pytest --cov=tinyrag --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

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

build: clean
	$(UV) build

dist: build
	@echo "Distribution created in dist/"

ci: lint type-check test
	@echo "CI pipeline completed successfully"
