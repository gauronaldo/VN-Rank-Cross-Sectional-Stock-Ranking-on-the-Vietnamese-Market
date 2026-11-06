# VN-Rank: cross-sectional stock ranking on the Vietnamese market

> **Research question:** can price–volume features rank Vietnamese stocks by 5-day forward
> return, net of trading costs?

VN-Rank produces a daily **ranking** of Vietnamese stocks by expected relative performance
over the next 5 trading days. It does not forecast prices or call market direction. Success
is measured by **Rank IC** and by whether a simple long-only portfolio built from the
ranking survives trading costs.

**Status:** Phase 0 (scaffold and protocol) is complete and approved (2026-11-06).
Phase 1 (data) is next. There are no results yet.

## Project status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffold, typed config, pre-registered protocol, lockbox guard design | **Done** |
| 1 | Data: vnstock download, storage, validation, data-quality report | Next |
| 2 | Universe, features, labels | Not started |
| 3 | Walk-forward splits, baselines, Elastic Net, LightGBM, Rank IC | Not started |
| 4 | MLP (multi-seed) | Not started |
| 5 | Backtest net of costs, performance, bootstrap CIs | Not started |
| 6 | Ablation, cost sensitivity, SHAP | Not started |
| 7 | Lockbox (run once) and daily update command | Not started |
| 8 | Showcase polish | Not started |

### Completed in Phase 0

- **Tooling.** `uv`-managed environment with exact dependency pins and a lockfile
  (`requires-python >=3.11`, developed on 3.13). It runs `ruff` (lint and format),
  `mypy --strict` on `src/`, `pytest`, and local `pre-commit` hooks, including one that
  refuses to commit `data/` or `mlruns/`. `make check` runs all three checks.
- **Package skeleton.** The package matches the repository map in `AGENT.md`; each module
  documents the phase that implements it.
- **Configuration.** Every tunable value lives in `configs/default.yaml`. It is parsed
  strictly into frozen dataclasses (`src/vn_rank/config.py`,
  `src/vn_rank/utils/dataclass_parsing.py`): unknown keys, missing keys, and wrong types are
  errors. Semantic checks include `walk_forward_end < lockbox_start`. Values still to be
  decided are `null`, and code that needs them must refuse to run while they are unset.
  `vn-rank validate-config` loads and checks the file.
- **Pre-registered protocol (locked).** See [`docs/protocol.md`](docs/protocol.md).
  - The primary model is LightGBM, chosen before any result is seen.
  - **C1:** the lower bound of the 95% block-bootstrap CI of the walk-forward mean Rank IC
    is above 0.
  - **C2:** the lockbox mean Rank IC is above 0.
  - **Leakage tripwire:** a mean Rank IC above 0.10, or any single fold above 0.20, stops
    work until leakage is investigated.
- **Lockbox guard design.** See [`docs/design_notes.md`](docs/design_notes.md). It consists
  of a date guard on signal and label-exit dates, a development loader that truncates data
  before the lockbox, and a `--confirm-lockbox` command with a committed lock file.
- **Tests.** 38 tests cover config parsing, validation, round-trips, and the CLI.

### Working agreements

- Commits go straight to `main`. Experimental work goes on a branch named after its
  topic, for example `experiment/turnover-band`.
- Market data is downloaded on the author's own machine, because `vnstock` generation 5
  installs from the vendor's package index (the PyPI project is quarantined).
- Decisions deferred to later phases: liquidity threshold (Phase 2); brokerage fee,
  slippage, position cap, and top-N (Phase 5).

## Protocol summary

The full pre-registered protocol is in [`docs/protocol.md`](docs/protocol.md).

- **Data:** `vnstock` Community tier, daily OHLCV adjusted for corporate actions (source
  `vci`), about 8 years from 2018-11.
- **Universe:** point-in-time. A stock needs 252 days of history, a 60-day median traded
  value above a threshold, and a trade on the day.
- **Label:** cross-sectional rank of the 5-day forward return. Entry is at the close after
  the signal.
- **Features:** 15 hand-implemented price–volume features, each ranked within its date.
- **Validation:** expanding walk-forward over 2022-01 → 2025-09 with 6-month test blocks,
  purge, and a 10-day embargo. The **lockbox** (2025-10 → latest) is evaluated once.
- **Models:** momentum and reversal baselines, Elastic Net, LightGBM (primary), and an MLP.
- **Metrics:** "IC" means **Rank IC**, the per-date Spearman correlation between predicted
  scores and realized forward returns. Rank IC is primary. Portfolio Sharpe is secondary and
  always reported with a confidence interval.

## Results

_To be added after Phases 3–7._

## Limitations

_To be completed. Known so far:_

- survivorship bias, because the symbol list only contains currently listed stocks;
- adjusted prices, which make historical traded value approximate;
- short history, which gives wide confidence intervals on Sharpe.

## Reproduction

```bash
make setup             # install pinned dependencies (uv) and git hooks
make check             # lint + type-check + tests
make validate-config   # load and validate configs/default.yaml
```

On Windows without `make`, run the underlying commands directly:

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run vn-rank validate-config
```

Pipeline targets (`download`, `features`, `train`, `backtest`, `report`) are added in later
phases.

## Repository map

```
configs/default.yaml   every tunable value
src/vn_rank/           package: data, features, validation, models, backtest, evaluation
tests/                 mirrors src/
docs/                  protocol.md (pre-registered), design_notes.md
notebooks/             EDA and result plots only
reports/               figures and tables used in this README
```

Raw market data is never committed; see the vnstock licence.
