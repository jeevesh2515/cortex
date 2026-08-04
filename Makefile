.PHONY: help install dev check lint format typecheck test cov clean

PY := .venv/bin/python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	uv pip install -e "."

dev: ## Install with dev and optional extras
	uv pip install -e ".[dev,all]"

lint: ## Run ruff
	$(PY) -m ruff check src tests

format: ## Format with ruff
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Run mypy in strict mode
	$(PY) -m mypy

test: ## Run the test suite
	$(PY) -m pytest tests/

cov: ## Run tests with coverage
	$(PY) -m pytest tests/ --cov=cortex --cov-report=term-missing

check: lint typecheck test ## Run every quality gate

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
