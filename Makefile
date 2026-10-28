# Root make targets for the Pycliques monorepo
.PHONY: help test research-test research-test-slow html pdf clean lint lint-fix typecheck
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run tests with pytest against the workspace env
	uv run pytest --doctest-modules

research-test: ## Run fast research regression tests (includes order 7)
	uv run pytest research/test_homotopy_research.py -m "not slow"

research-test-slow: ## Run the scheduled order-8 research regression test
	uv run pytest research/test_homotopy_research.py -m slow

lint: ## Run ruff linter and formatter check
	uv run ruff check packages/
	uv run ruff format --check packages/

lint-fix: ## Auto-fix ruff lint and formatting issues
	uv run ruff check --fix packages/
	uv run ruff format packages/

typecheck: ## Run mypy type checking
	uv run mypy packages/

coverage: ## Run tests with coverage report
	uv run pytest --cov=packages

coverage-html: ## Run tests with HTML coverage report
	uv run pytest --cov=packages --cov-report=html

html: ## Build HTML documentation via docs/Makefile
	uv run make -C docs html

pdf: ## Build PDF documentation via docs/Makefile
	uv run make -C docs latexpdf

clean: ## Clean documentation build artifacts
	uv run make -C docs clean
