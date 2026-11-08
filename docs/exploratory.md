# Exploratory analyses (after the lockbox)

Everything on this page was run **after** the lockbox evaluation. None of it changes a
result fixed in [`protocol.md`](protocol.md); it is reported so that readers can see what
else was tried. Numbers are walk-forward (2022-01 to 2025-09) unless stated. Full tables are
in [`reports/6_exploratory/`](../reports/6_exploratory/).

## 1. A one-line low-volatility rule

**Question:** is the model more than the low-volatility effect?
**Rule:** rank stocks by minus their 60-day volatility (`lowvol_vol60`, the textbook
version). A second rule on the 14-day ATR was chosen after seeing the test blocks, so it is
optimistic and not used for claims.

| | Mean Rank IC [95% CI] | Net excess vs equal weight, % a year [95% CI] | Alpha after beta, % a year [95% CI] | Beta |
|---|---|---|---|---|
| LightGBM (primary) | 0.058 [0.042, 0.076] | -6.4 [-19.1, +7.8] | -7.5 [-17.3, +4.5] | 0.83 |
| Low-volatility rule | 0.055 [0.015, 0.088] | +5.8 [-18.5, +27.1] | +1.8 [-7.9, +10.2] | 0.35 |

LightGBM minus the rule: +0.003 [-0.020, +0.034], no difference. The model is steadier
(positive in 8 of 8 folds against 7 of 8), not stronger. The rule trades 5 times a year
instead of 33, so its costs are small, but its alpha interval still includes 0.

The rule was also scored on the lockbox (Rank IC 0.091) **after** the primary models had
been evaluated there. That number is descriptive only: the lockbox can no longer test a
rule chosen after it was opened, and it is now spent.
[Report](../reports/6_exploratory/exploratory_low_volatility.md).

## 2. Four literature features and XGBoost

Three tests were fixed in [`protocol.md`](protocol.md) Section 12 and committed before any
code, then run once on the walk-forward blocks. Adding four features from the anomaly
literature (60-day beta, idiosyncratic volatility, the largest daily return of 20 days, and
60-day skewness) raises LightGBM's Rank IC from 0.058 to 0.068 (+0.010 [+0.003, +0.017]),
but the features were chosen after volatility was known to drive the signal, so the gain
may be optimistic, and the extended model is still not significantly better than the
low-volatility rule (+0.013 [-0.014, +0.048]) and still loses to equal weight after costs.
Swapping LightGBM for XGBoost makes no difference (+0.000 [-0.004, +0.004]). A one-line idiosyncratic-
volatility rule, reported descriptively, had alpha after beta +6.7% [-1.6, +14.4] a year.
The code was removed for scope; the git tag `exploratory-extension` marks the last commit
that has it, and the [report](../reports/6_exploratory/extension_features_xgboost.md) is kept.

## 3. Labels lost to delisting

A label is missing when the stock stops trading before the exit session, so on the 925
test dates 26 universe rows (5 symbols, 0.009%) drop out of the evaluation. Putting them
back, at their last close or at -100%, changes mean Rank IC by at most 0.0003 for every
model, even when the learned models are assumed to rank them first (their fold models were
not saved).

The backtest has the same gap, and there it is a **small look-ahead**: it chooses only among
stocks whose label exists, so it never buys a stock about to stop trading. Made to buy the
4 such rows on rebalance dates, LightGBM's net excess return over equal weight moves from
-6.4% to **-6.6%** a year if they are carried at their last close (each lost at worst 1% to 25%) and
to **-9.4%** if they are a total loss, a bound. The conclusion does not change. Symbols the
source cannot serve (25) and older delistings are not covered.
[Report](../reports/6_exploratory/delisting_sensitivity.md).
