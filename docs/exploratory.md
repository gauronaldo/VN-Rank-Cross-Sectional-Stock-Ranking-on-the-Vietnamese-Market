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

Fixed in [`protocol.md`](protocol.md) Section 12 and committed before any code was written, then run
once on the walk-forward blocks. The four features are rolling 60-day market beta,
idiosyncratic volatility (`ivol_60`), the largest daily return of 20 days (`max_20`), and
60-day skewness.

| Test (paired mean Rank IC, A minus B) | Difference [95% CI] | Result |
|---|---|---|
| H1: LightGBM 19 features minus LightGBM 15 | +0.010 [+0.003, +0.017] | features help (IC 0.058 to 0.068) |
| H2: XGBoost 15 minus LightGBM 15 | +0.000 [-0.004, +0.004] | the library does not matter |
| H3: LightGBM 19 minus the low-volatility rule | +0.013 [-0.014, +0.048] | still no significant edge over the rule |

Descriptive only: a one-line idiosyncratic-volatility rule reaches Rank IC 0.071, with net
excess +10.2% a year over equal weight and alpha after beta +6.7% [-1.6, +14.4] (beta
0.44, turnover 6 times a year). The extended LightGBM still loses to equal weight after
costs (-3.0% a year). The features were chosen after earlier analyses had shown that
volatility drives the signal, so H1 may be slightly optimistic.
[Report](../reports/6_exploratory/extension_features_xgboost.md).

## 3. Labels lost to delisting

**Concern:** a label is missing when the stock stops trading before the exit session
(protocol Section 3). A stock delisted during the 5-session hold then drops out of the
evaluation, and a model is never penalized for ranking it high. This could flatter Rank IC.

**Check:** on the 925 test dates only **26** universe rows lose their label this way (5
symbols, 20 dates, 0.009% of the 303,736 scored rows). Putting them back with the return to
their last close, or with a -100% return, changes mean Rank IC by **at most 0.0003** for
every model. For the learned models this already assumes the worst case (the lost stock
gets the highest score on its date), because the fold models were not saved. Nor can the
main portfolio hold a stock that stops trading: it buys only stocks whose 5-session label
exists and sells them at that label's exit.
[Report](../reports/6_exploratory/delisting_sensitivity.md).
