# VN-Rank: cross-sectional stock ranking on the Vietnamese market

> **Research question:** can price–volume features rank Vietnamese stocks by 5-day forward
> return, net of trading costs?

VN-Rank produces a daily **ranking** of Vietnamese stocks by expected relative performance
over the next 5 trading days. It does not forecast prices or call market direction. Success
is measured by **Rank IC** and by whether a simple long-only portfolio built from the
ranking survives trading costs.

**Status:** Phase 0 (scaffold and protocol). No results yet.

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
