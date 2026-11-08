# VN-Rank task runner. Every target runs inside the uv-managed environment.
CONFIG ?= configs/default.yaml
UV := uv run

.PHONY: help setup check lint format typecheck test validate-config \
        download validate-data features train backtest ablation lockbox update \
        exploratory extension delisting

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

download:  ## Download or resume the raw vnstock snapshot (needs a Community API key)
	$(UV) vn-rank --config $(CONFIG) download

validate-data:  ## Data-quality report and tickers-per-year figure
	$(UV) vn-rank --config $(CONFIG) validate-data

features:  ## Build the development panel (prices before lockbox_start) in data/processed
	$(UV) vn-rank --config $(CONFIG) features

train:  ## Tune, fit, and score all models on walk-forward folds (reports/2_walk_forward/signal_results.md)
	$(UV) vn-rank --config $(CONFIG) train

backtest:  ## Backtest walk-forward predictions net of costs (reports/2_walk_forward/backtest_results.md)
	$(UV) vn-rank --config $(CONFIG) backtest

ablation:  ## Feature-group ablation, liquidity and cost sensitivity, SHAP
	$(UV) vn-rank --config $(CONFIG) ablation

lockbox:  ## One-time lockbox evaluation (refuses if reports/4_lockbox/lockbox_run.lock exists)
	$(UV) vn-rank --config $(CONFIG) lockbox --confirm-lockbox

update:  ## Score the newest session with the frozen model; track realized Rank IC
	$(UV) vn-rank --config $(CONFIG) update

exploratory:  ## Post-lockbox low-volatility comparison (exploratory, changes no result)
	$(UV) vn-rank --config $(CONFIG) exploratory

extension:  ## Pre-registered exploratory extension: literature features and XGBoost
	$(UV) vn-rank --config $(CONFIG) extension

delisting:  ## Exploratory: Rank IC with labels lost to delisting put back
	$(UV) vn-rank --config $(CONFIG) delisting
