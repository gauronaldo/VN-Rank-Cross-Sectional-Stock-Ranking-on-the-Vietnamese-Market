# Reports

Every table and figure in the project, one folder per stage, in the order the project ran.
Each file is written by the command named in its heading; the `.csv` files hold the same
numbers as the `.md` tables at full precision.

![Summary](summary.png)

[`summary.png`](summary.png): the figure the main README opens with (Rank IC by model and
period, and the walk-forward cost story).

## Headline tables

The walk-forward results behind the main README, from `2_walk_forward/` and
`6_exploratory/exploratory_low_volatility.md`.

**Signal (walk-forward, 925 test dates).**

| Model | Mean Rank IC [95% CI] | Folds > 0 | LightGBM minus model [95% CI] |
|---|---|---|---|
| Momentum (20 d) | -0.001 [-0.023, 0.026] | 4 / 8 | 0.059 [0.034, 0.080] |
| Reversal (5 d) | 0.005 [-0.015, 0.023] | 5 / 8 | 0.054 [0.027, 0.084] |
| Elastic Net | 0.040 [0.011, 0.069] | 6 / 8 | 0.018 [-0.001, 0.040] |
| **LightGBM** (primary) | **0.058 [0.042, 0.076]** | **8 / 8** | — |
| MLP | 0.057 [0.040, 0.075] | 8 / 8 | 0.002 [-0.006, 0.010] |
| Low-volatility rule* | 0.055 [0.015, 0.088] | 7 / 8 | 0.003 [-0.020, 0.034] |

**Portfolio (walk-forward, % per year, net of costs unless stated).**

| Strategy | Turnover | Cost | Excess vs equal weight, before costs | Excess vs equal weight [95% CI] | Beta | Alpha after beta [95% CI] |
|---|---|---|---|---|---|---|
| LightGBM top decile | 33x | 19.9 | +12.3 | -6.4 [-19.1, +7.8] | 0.83 | -7.5 [-17.3, +4.5] |
| LightGBM, turnover band* | 25x | 14.8 | +12.2 | -1.5 [-14.2, +12.9] | 0.82 | -2.7 [-12.5, +9.4] |
| Low-volatility rule* | 5x | 3.3 | +8.0 | +5.8 [-18.5, +27.1] | 0.35 | +1.8 [-7.9, +10.2] |

\*Added after seeing results and labelled as such: the turnover band (keep holdings while
in the top 20%) after the walk-forward backtest, and the low-volatility rule after the
lockbox. Neither changes a result fixed in advance.

**Also tested:** dropping any one feature group changes Rank IC by at most 0.005 (not
significant); Rank IC falls from 0.070 to 0.045 as the liquidity threshold rises from 0.5
to 5 billion VND; net excess return breaks even near 0.15% fee plus slippage per side.

## Metric definitions

- **Rank IC:** on one day, the Spearman correlation between model scores and realized 5-day
  returns across the stocks; a day with identical scores counts as 0. IC IR (mean / std of
  the daily series) is per day, not annualized.
- **Intervals:** 95% moving-block bootstrap (20-day blocks, 10,000 resamples) of daily
  series; paired for differences. Newey-West t-statistics use lag 4 for overlapping labels.
- **Returns:** daily net returns, annualized as mean × 252; Sharpe uses a 0% risk-free
  rate. Turnover is one-way per rebalance, annualized. Alpha and beta come from regressing
  daily returns on the equal-weight benchmark.

## [1. Data](1_data/) (`vn-rank validate-data`)

| File | What it shows |
|---|---|
| [data_quality.md](1_data/data_quality.md) | Snapshot coverage, excluded symbols, flags per data-quality check, and the largest price-band breaches |
| [tickers_per_year.png](1_data/tickers_per_year.png) | Listed and delisted symbols with data in each year |

## [2. Walk-forward](2_walk_forward/) (`vn-rank train`, `vn-rank backtest`)

The main development results: 8 test blocks, 2022-01 to 2025-09.

| File | What it shows |
|---|---|
| [signal_results.md](2_walk_forward/signal_results.md) | Mean Rank IC, IC IR, Newey-West t, and quintile returns per model, pooled and per fold |
| [backtest_results.md](2_walk_forward/backtest_results.md) | Top-decile portfolios net of costs: returns, Sharpe, turnover, costs, intervals |
| [backtest_results_yearly.csv](2_walk_forward/backtest_results_yearly.csv) | The same portfolios year by year |
| [equity_curves.png](2_walk_forward/equity_curves.png) | Cumulative net returns of each model against both benchmarks |

## [3. Ablation](3_ablation/) (`vn-rank ablation`)

| File | What it shows |
|---|---|
| [ablation_results.md](3_ablation/ablation_results.md) | LightGBM without each feature group, at other liquidity thresholds, and under other costs |
| [ablation_results_costs.csv](3_ablation/ablation_results_costs.csv) | The brokerage fee by slippage grid behind the cost tables |
| [shap_summary.png](3_ablation/shap_summary.png) | Which features drive the LightGBM scores (SHAP) |

## [4. Lockbox](4_lockbox/) (`vn-rank lockbox`, run once)

The held-out year, 2025-10 to 2026-10, evaluated exactly once at the end.

| File | What it shows |
|---|---|
| [lockbox_results.md](4_lockbox/lockbox_results.md) | The success criteria fixed in advance, Rank IC, and portfolios on the lockbox, with the frozen hyperparameters |
| [lockbox_equity.png](4_lockbox/lockbox_equity.png) | Cumulative net returns on the lockbox |
| [lockbox_run.lock](4_lockbox/lockbox_run.lock) | Proof of the single run: time, git commit, and config hash; its presence blocks a second run |

## [5. Monitoring](5_monitoring/) (`vn-rank update`)

| File | What it shows |
|---|---|
| [realized_ic.png](5_monitoring/realized_ic.png) | Realized Rank IC of the frozen model on new sessions, against the walk-forward and lockbox means |

## [6. Exploratory](6_exploratory/) (after the lockbox; summarized in [docs/exploratory.md](../docs/exploratory.md))

| File | What it shows |
|---|---|
| [exploratory_low_volatility.md](6_exploratory/exploratory_low_volatility.md) | The models against a one-line low-volatility rule, with beta-adjusted alpha |
| [extension_features_xgboost.md](6_exploratory/extension_features_xgboost.md) | Four literature features and XGBoost, tested under decision rules committed before the code (code removed; tag `exploratory-extension`) |
| [delisting_sensitivity.md](6_exploratory/delisting_sensitivity.md) | Rank IC and portfolios with the 26 test labels lost to delistings put back (`vn-rank delisting`) |
