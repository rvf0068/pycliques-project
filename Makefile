# Root make targets for the Pycliques monorepo
.PHONY: help test html pdf clean
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run tests with pytest against the workspace env
	uv run pytest

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
