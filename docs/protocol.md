# VN-Rank pre-registered research protocol

**Status: LOCKED, approved by the author on 2026-11-06.** Any change must be added to the
[Changelog](#11-changelog) with the date and the reason. Items marked **[proposed]** were
choices the spec left open; they became binding when the protocol was approved.

## 1. Research question

> Can price–volume features rank Vietnamese stocks by 5-day forward return, net of trading
> costs?

The model outputs a daily **ranking** of stocks by expected relative performance. It does not
forecast price levels or market direction. Accuracy, RMSE, and MAPE are not used.

## 2. Data

- Source: `vnstock` (generation 5), Community tier only, price source `vci` only (never
  mixed with `kbs`). Daily OHLCV adjusted for corporate actions, in thousand VND; VNINDEX
  for the trading calendar and as a benchmark.
- History: as far back as the tier allows (about 8 years, from about 2018-11).
- The symbol list contains currently listed stocks and **239 delisted symbols** (exchange
  code `DELISTED`), whose history is downloadable (verified 2026-11-07). Delisted symbols are
  included, which reduces **survivorship bias**; it is not removed, because symbols delisted
  before the list was compiled may be missing. The equal-weight benchmark of the same
  universe remains the primary comparison.
- The data is a snapshot. Download date, source, and vnstock version are recorded in
  metadata. Raw data is never committed.

## 3. Universe (point-in-time)

A stock is eligible on trading date `t` if all of the following hold, using only information
up to and including `t`:

1. at least `min_history_days` = 252 trading days of history up to `t`;
2. median traded value (`close × 1000 × volume`, in VND) over the previous
   `traded_value_window` = 60 trading days ≥ `min_traded_value_vnd` (decided in Phase 2);
3. volume on `t` is greater than zero.

Exchanges: HOSE, HNX, UPCoM, and delisted symbols (see the changelog); the liquidity filter
removes most thinly traded names. The exchange is the one reported today, not the
historical one. Traded value is approximate in early years because prices are adjusted.

## 4. Label

- Signal at the close of `t`; entry at the close of `t+1`; exit at the close of `t+1+h`,
  with `h` = 5 trading days.
- `forward_return = close[t+1+h] / close[t+1] - 1`.
- Training target: cross-sectional rank of `forward_return` within each date, scaled to
  [0, 1] **[proposed: rank rather than z-score]**.
- Portfolios rebalance every `h` trading days, so holding periods do not overlap.

## 5. Features

Fifteen features, hand-implemented, each using data up to and including `t` and converted to
a cross-sectional rank within each date:

| Group | Features | Count |
|---|---|---|
| Momentum / reversal | return over 1, 5, 20, 60, 120 days | 5 |
| Volatility | std of daily returns (20, 60 days); ATR(14) / close | 3 |
| Technical | RSI(14); MACD(12, 26, 9) histogram / close; Bollinger %B (20, 2) | 3 |
| Volume / liquidity | log median traded value (20d); volume / mean volume (20d); Amihud illiquidity (20d) | 3 |
| Price position | close / rolling 252-day high | 1 |

The spec says "about 18"; the listed table contains 15, and only these 15 are built.

## 6. Validation

| Period | Signal dates | Use |
|---|---|---|
| Feature warm-up | 2018-11 → 2019-10 | 252-day history requirement |
| Initial training | 2019-11-01 → 2021-12-31 | start of the expanding window |
| Walk-forward (development) | 2022-01-01 → 2025-09-30 | all design decisions |
| **Lockbox** | 2025-10-01 → latest matured label | evaluated **once**, at the end |

- **Walk-forward folds.** Test blocks of 6 calendar months: 2022H1, 2022H2, …, 2025H1, plus a
  final 3-month block 2025-07 → 2025-09 (8 blocks). Each fold trains on all eligible samples
  before its test block (expanding window).
- **Purge.** A training sample with signal date `t` uses prices up to `t+1+h`. It is dropped
  if `t+1+h` falls on or after the first date of the test block.
- **Embargo.** A further `embargo_days` = 10 trading days of training samples are dropped
  before the purged train end, so the last training signal date is at least
  `1 + h + 10` = 16 trading days before the test block.
- **Lockbox boundary [proposed].** No price from the lockbox period is used during
  development: walk-forward test samples whose exit date `t+1+h` falls on or after
  `lockbox_start` are dropped (the last few signal dates of September 2025).
- **Tuning.** Optuna runs inside each training window only. The validation set is the last
  6 months of that window, separated from inner training by the same purge and embargo.
  Test blocks are never used for any choice.
- **Lockbox guard.** Every training, tuning, and CV entry point checks that its data ends
  before `lockbox_start` (see `docs/design_notes.md`, "Lockbox guard").

## 7. Models

All models use the same folds, features, and metrics.

1. Baselines (no learning): 20-day momentum rank; 5-day reversal rank; equal-weight
   universe (portfolio benchmark only).
2. Elastic Net (scikit-learn).
3. **LightGBM, the pre-registered primary model.** The success criteria in Section 9 refer
   to LightGBM only, so the headline result is not chosen after seeing which model wins.
4. MLP (PyTorch, 2–3 layers, dropout, weight decay, early stopping on the inner validation
   window).

Stochastic models (MLP, and LightGBM if bagging or feature subsampling is used) run with
5 seeds; results are reported as mean ± standard deviation across seeds. The number of
configurations tried (from MLflow) is reported.

## 8. Metrics

**"IC" always means Rank IC**: the Spearman correlation, on one date, between predicted
scores and realized forward returns across eligible stocks.

Signal level (primary), computed on every trading date of the test blocks:

- mean Rank IC; IC IR = mean / standard deviation of daily Rank IC (per day, not annualized);
- t-statistic of mean Rank IC with Newey-West (HAC) standard errors, lag `h - 1` = 4,
  because 5-day labels on consecutive dates overlap;
- **95% confidence interval for mean Rank IC [proposed]:** moving-block bootstrap of the
  daily Rank IC series over all walk-forward test dates, block length 20 trading days,
  10,000 resamples, percentile interval;
- mean forward return per score quintile and the top-minus-bottom spread;
- signal autocorrelation: Spearman correlation of scores between consecutive rebalance dates.

Portfolio level (secondary), long-only top decile, equal weight, rebalanced every `h` days,
net of brokerage fee, 0.1% selling tax, and slippage (defaults approved in Phase 5):

- **Return convention [proposed]:** daily net portfolio returns; annualized return =
  mean daily return × 252; annualized volatility = std of daily returns × √252;
  Sharpe = annualized return / annualized volatility, with a risk-free rate of 0;
- maximum drawdown of the daily equity curve; turnover = ½ Σ|w_new − w_old| per rebalance,
  annualized × 252 / h; per-year breakdown;
- block-bootstrap confidence interval for Sharpe (same method as for Rank IC);
- benchmarks: equal-weight same-universe portfolio (primary), VNINDEX, momentum baseline.

## 9. Pre-registered success criteria [proposed]

Expressed in Rank IC; Sharpe is never a success criterion.

| ID | Criterion | Period |
|---|---|---|
| **C1** (primary) | The 95% CI lower bound of LightGBM's mean Rank IC is above 0. | walk-forward |
| **C2** (confirmation) | LightGBM's mean Rank IC is above 0 (same sign as C1). No significance is required, because 12 months is too short. | lockbox |

- **Supported:** C1 and C2 both hold. **Not supported:** C1 fails. **Inconclusive:** C1 holds
  and C2 fails. Each outcome is reported as it is.
- **Reported but not success-defining:** LightGBM minus each baseline in mean Rank IC (with a
  paired block-bootstrap CI); net-of-cost excess return of the top decile over the
  equal-weight benchmark, and Sharpe, both with CIs; turnover; MLP versus LightGBM.
- **Leakage tripwire:** if any model's walk-forward mean Rank IC exceeds 0.10, or any single
  fold exceeds 0.20, work stops and leakage is investigated before any result is reported.
  For context only (different market, label, and costs, so not a target): public benchmarks
  on Chinese equities report Rank IC of about 0.02–0.05.

## 10. Known limitations

- Survivorship bias: delisted stocks are probably missing from the symbol list.
- Adjusted prices make historical traded value approximate.
- Short history: about 3.75 years of walk-forward data. The standard error of annualized
  Sharpe is about 0.6 on walk-forward and about 1.2 on the 12-month lockbox, so Sharpe is
  secondary and always reported with its interval.
- Price-limit and cost rules are simplified and verified against current rules only.

## 11. Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-11-06 | Draft created. | Phase 0. |
| 2026-11-06 | Approved and locked, including all [proposed] items. | End of Phase 0. |
| 2026-11-07 | Include the 239 `DELISTED` symbols in the data and the universe; the price-limit check uses the widest band (UPCoM, ±15%) for them because their original exchange is unknown. | Phase 1 probe showed their history is available; including them reduces survivorship bias. Approved by the author. |
