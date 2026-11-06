# VN-Rank task runner. Every target runs inside the uv-managed environment.
CONFIG ?= configs/default.yaml
UV := uv run

.PHONY: help setup check lint format typecheck test validate-config \
        download features train backtest report

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Install pinned dependencies and git hooks
	uv sync --locked
	$(UV) pre-commit install

check: lint typecheck test  ## Lint, type-check, and test (must pass before every Phase Report)

lint:  ## Ruff lint and format check
	$(UV) ruff check .
	$(UV) ruff format --check .

format:  ## Apply ruff formatting and safe fixes
	$(UV) ruff check --fix .
	$(UV) ruff format .

typecheck:  ## mypy --strict on src/
	$(UV) mypy

test:  ## Run the test suite
	$(UV) pytest

validate-config:  ## Load and validate CONFIG
	$(UV) vn-rank --config $(CONFIG) validate-config

# Pipeline targets are implemented in later phases (AGENT.md section 7).
download features train backtest report:
	@echo "'make $@' is not implemented yet; it is built in a later phase." >&2
	@exit 1
