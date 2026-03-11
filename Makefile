.PHONY: help install install-dev test test-fast test-slow test-cov lint format format-check typecheck clean build run-checks pre-commit all

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in production mode
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run all tests
	python -m pytest -q tests

test-fast:  ## Run fast tests only (exclude slow and integration tests)
	python -m pytest -q tests -m "not slow and not integration and not gpu"

test-slow:  ## Run slow tests only
	python -m pytest -q tests -m "slow"

test-cov:  ## Run tests with coverage report
	python -m pytest tests/ -v --cov=tutor_bench --cov-report=term-missing --cov-report=html

lint:  ## Run lint checks with ruff
	python -m ruff check tutor_bench tests

format:  ## Format code with ruff
	python -m ruff check --fix tutor_bench tests
	python -m ruff format tutor_bench tests

format-check:  ## Check code formatting without modifying files
	python -m ruff check tutor_bench tests
	python -m ruff format --check tutor_bench tests

typecheck:  ## Run type checking with pyright
	python -m pyright tutor_bench

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

run-checks:  ## Run code quality checks and tests
	@echo "Running ruff lint..."
	python -m ruff check tutor_bench tests
	@echo "Running ruff format check..."
	python -m ruff format --check tutor_bench tests
	@echo "Running pyright..."
	python -m pyright tutor_bench
	@echo "Running tests..."
	python -m pytest -q tests -m "not slow and not integration and not gpu"
	@echo "All checks passed!"

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

all: clean install-dev run-checks  ## Clean, install dependencies, and run all checks
