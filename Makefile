.PHONY: help install install-dev test test-fast test-slow test-cov lint format typecheck clean build docs serve-docs run-checks all

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in production mode
	pip install -e .

install-dev:  ## Install the package with all development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run all tests
	python -m pytest tests/ -v

test-fast:  ## Run fast tests only (exclude slow and integration tests)
	python -m pytest tests/ -v -m "not slow and not integration"

test-slow:  ## Run slow tests only
	python -m pytest tests/ -v -m "slow"

test-cov:  ## Run tests with coverage report
	python -m pytest tests/ -v --cov=tutor_bench --cov-report=term-missing --cov-report=html

lint:  ## Run linting checks with ruff
	python -m ruff check tutor_bench tests scripts

format:  ## Format code with ruff
	python -m ruff check --fix tutor_bench tests scripts
	python -m ruff format tutor_bench tests scripts

format-check:  ## Check code formatting without modifying files
	python -m ruff check tutor_bench tests scripts
	python -m ruff format --check tutor_bench tests scripts

typecheck:  ## Run type checking with mypy
	python -m mypy tutor_bench

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m pip install --upgrade build
	python -m build

docs:  ## Build documentation with Sphinx
	cd docs && make html

serve-docs:  ## Serve documentation locally with auto-reload
	sphinx-autobuild docs/source docs/build/html --port 8000

run-checks:  ## Run all code quality checks (format, lint, typecheck, test)
	@echo "Running ruff..."
	python -m ruff check --fix tutor_bench tests scripts
	python -m ruff format tutor_bench tests scripts
	@echo "Running mypy..."
	python -m mypy tutor_bench
	@echo "Running tests..."
	CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v -m "not slow"
	@echo "All checks passed!"

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

all: clean install-dev run-checks  ## Clean, install dev dependencies, and run all checks
