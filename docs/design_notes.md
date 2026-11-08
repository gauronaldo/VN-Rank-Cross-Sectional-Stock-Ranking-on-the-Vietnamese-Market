# Design notes

One short rationale per major choice. Completed in Phase 8; entries are added as decisions
are made.

## Tooling (Phase 0)

**Environment.** `uv` manages the virtual environment and the lockfile (`uv.lock`), and
`make setup` runs `uv sync --locked`. The project declares `requires-python >=3.11` and is
developed on Python 3.13. Dependencies are pinned exactly in `pyproject.toml`. The pins
are the newest versions that still support Python 3.11, which holds `numpy` at 2.4.x,
`scipy` at 1.17.x, and `shap` at 0.51.x.

**PyTorch wheels.** The default Linux `torch` wheel from PyPI bundles CUDA libraries, and
the virtual environment is about 6 GB. The project only needs the CPU build. The PyTorch CPU
index was not reachable from the build environment, so the default wheel is used for now.

**Continuous integration.** `.github/workflows/check.yml` (outside the original repository
map, added after Phase 8) runs ruff, mypy, and pytest on every push. It uses a Windows runner
because the locked PyTorch wheel is CPU-only there, while on Linux it pulls about 3 GB of
CUDA libraries, and it syncs with `--no-group data`, so mypy ignores the missing vnstock
import. A fresh clone from GitHub was checked the same way before CI was added.

**Type stubs.** `mypy --strict` runs on `src/`. PyYAML and pandas ship no complete inline
types, so the dev group adds `types-PyYAML` and `pandas-stubs`. They are type-checking aids
only and never imported at runtime; added with the author's approval at the end of Phase 0.

## Configuration (Phase 0)

**Strict, typed, immutable.** `configs/default.yaml` maps onto frozen dataclasses in
`vn_rank/config.py`. Parsing rejects unknown keys, missing keys, and wrong types; for
example, `True` is not accepted as an `int`, and a quoted string is not accepted as a date.
Semantic checks run afterwards. A typo in a config key therefore fails loudly instead of
silently falling back to a default.

**Undecided values are `null`.** Values the user decides in a later phase, such as
`min_traded_value_vnd` (Phase 2) and the brokerage fee, slippage, and position cap
(Phase 5), are `null`. Code that needs them must refuse to run while they are `null`. No
provisional number is written into the config.

**Units in names.** The traded-value threshold is in VND (`min_traded_value_vnd`), even
though prices are in thousand VND, so the conversion lives in exactly one place in code.

**Module split.** The generic YAML-to-dataclass parser lives in
`vn_rank/utils/dataclass_parsing.py`, which keeps `config.py` focused on the schema and its
constraints. This file is not in the original repository map; the author
approved the addition at the end of Phase 0.

## Lockbox guard (designed in Phase 0, implemented in Phases 3 and 7)

1. **Configuration.** `validation.lockbox_start` (2025-10-01) is the only source of the
   date. Config validation requires `walk_forward_end < lockbox_start`.
2. **Date guard.** `validation/splits.py` exposes
   `ensure_before_lockbox(signal_dates, exit_dates, lockbox_start)`. It raises
   `LockboxViolationError` if any signal date **or any label exit date** is on or after
   `lockbox_start`. It uses an explicit `raise`, not `assert`, because `python -O` strips
   assertions. It is called on the data that enters every fold split, every model `fit`,
   and every tuning objective. It is unit-tested in Phase 3.
3. **Development data loader.** Development commands load prices truncated before
   `lockbox_start`, so lockbox prices are never in memory during development. This is
   defence in depth on top of the guard.
4. **Lockbox command (Phase 7).** `vn-rank lockbox --confirm-lockbox`:
   - refuses to run without the flag;
   - refuses if `reports/4_lockbox/lockbox_run.lock` exists;
   - refuses on a dirty git working tree, so that the recorded commit identifies the exact
     code.
   - writes the lock file **before** computing any result, so a crash part-way still counts
     as the one run. The lock file records the UTC timestamp, git commit, SHA-256 hash of the
     canonical config, and the last matured label date.

   The lock file is committed to git, because session containers are ephemeral.
5. **After the run.** No change to models, features, or parameters is allowed. Any later
   experiment is labelled exploratory in the README.

## Data source (Phase 1)

**Installation.** The PyPI project `vnstock` has been quarantined since 2026-09-24, and
generation 5 is distributed from the vendor index `https://vnstocks.com/api/packages`.
`pyproject.toml` pins `vnstock==5.2.2` in a separate `data` dependency group, resolved from
that index with `explicit = true`, so no other package can come from it. The group is a
default group; on a machine that cannot reach the index, run `uv sync --no-group data`
(everything except downloading still works, because only `vnstock_client.py` imports
vnstock, and it does so lazily).

**Account and limits.** Community tier (free API key, stored by `vnstock register` in the
user profile, never in the repository). `vnstock.usage()` reports 60 requests/minute,
3,600/hour, 10,000/day, and 100,000/month. The Guest tier (10/minute, 300/day,
1,000/month) cannot list symbols at all (`EntitlementError`), so it is not usable here.
`vnstock status` reports that 5.2.2 is licensed until 2027-11-07; results must be
reproducible from the stored snapshot, not from a re-download.

**Verified against real outputs (2026-11-07, Windows 11, Python 3.11.9, vnstock 5.2.2).**

- `Reference().equity.list_by_exchange(source="vci")` returns 1,755 rows with columns
  `symbol, exchange, organ_name, organ_short_name, icb_code_lv2` (pandas `string`).
  Exchange codes: `UPCOM` 809, `HOSE` 408, `HNX` 299, and **`DELISTED` 239**. The
  exchange is today's, not the historical one.
- **Delisted symbols have downloadable history** (for example VTL until 2025-10-09, VTS
  until 2026-08-19). They are included; see the protocol changelog.
- `Market().equity(symbol).ohlcv(start, end, interval="1D", *, limit, source, count_back,
  length, floating=2)` returns `time, open, high, low, close, volume` on a `RangeIndex`:
  `time` is `datetime64[ns, Asia/Ho_Chi_Minh]` stamped 07:00, prices are `float64` in
  thousand VND, `volume` is `int64`. There is no symbol column. `end` is inclusive.
- **Rounding.** `floating=2` (the default) rounds adjusted prices to two decimals, which
  loses precision for low-priced stocks; the downloader passes `floating=None`.
- **History cap.** Community keeps the last 2,920 calendar days and records
  `cap_rows_dropped` and `tier_cap_history_days` in `df.attrs`. Listed stocks therefore
  start on 2018-11-08; the downloader stores these attrs per symbol in `manifest.json`.
- **VNINDEX** defaults to source `kbs` and to only **1,000 rows** (from 2022-11) with
  either source. Passing `source="vci"` and `limit` (config `max_rows_per_request`) returns
  the whole requested window (2,000 rows from 2018-11-01).
- Some listed symbols are unknown to `vci` (`InputError`, for example BLU and BPW); they
  are recorded as `failed` in the manifest and reported.
- Errors: every library error subclasses `VnstockError`, which carries `retryable`,
  `code`, and `message_en`. The adapter retries when `retryable` is true or on
  `RateLimitedError`, skips the symbol on `InputError`, `ProviderError`, or `SchemaError`,
  and stops on anything else (authentication, entitlement, expired release).
- Batch calls (`equity([...])`) issue one request per symbol and drop the symbol column,
  so symbols are downloaded one at a time.

**Snapshot design.** Raw data is stored in a canonical schema (`date` as a naive date,
`symbol`, OHLC, `volume`) under `data/raw/`. `metadata.json` fixes the source, exchanges,
start, and end on the first run; a resumed run with different settings is refused, so a
snapshot is never a mix of two download dates. The end date is yesterday in market time,
so only completed sessions are stored.

**Module split.** The vnstock adapter lives in `data/vnstock_client.py`, separate from the
orchestration in `data/download.py`, because together they exceeded the 250-line module
limit and they have different responsibilities (talking to the library versus resumable
bookkeeping). This file is not in the original repository map.

## Data treatment (decided at the end of Phase 1)

The rules are listed in `docs/protocol.md`, Section 2. The reasons:

- **No-trade sessions (35% of rows).** The source emits a row for every session, even
  without a trade: `open` is a stale earlier price, and `high`, `low`, and `close` are the
  reference price. They explain 65,640 of the 67,391 OHLC-inconsistency flags. Keeping the
  rows makes every window count sessions, so a 20-day return means 20 sessions for liquid
  and illiquid stocks alike. Flattening the bar removes ranges that never traded, which
  would otherwise inflate ATR. The universe already requires volume > 0 on day `t`. One
  limitation remains: an exit day without a trade is valued at the reference price.
- **OHLC repair instead of deletion.** Of the remaining inconsistencies, most are small
  rounding effects of price adjustment, and `close` is valid in every case. Clipping
  `high` and `low` keeps the close and the session calendar intact. After an illustrative
  liquidity filter (1 billion VND) only 45 such rows and no zero-price rows remain.
- **Band breaches are kept.** 18,496 rows exceed today's exchange band. Large caps have
  none; the rest are mostly new listings, sessions after a suspension, and symbols that
  moved exchange (the band is taken from today's exchange). Capping returns would distort
  labels and features. After the same illustrative filter, 0.12% of rows are flagged. The
  backtest price-limit constraint (Phase 5) inherits the today's-exchange limitation.
- **Excluded symbols.** 25 delisted symbols unknown to `vci` and 10 empty ones; they are
  listed in `reports/1_data/data_quality.md`.

## Features, universe, and labels (Phase 2)

**Wide frames on the market calendar.** Prices are pivoted to one frame per field
(VNINDEX sessions x symbols). Every indicator is then a column-wise `shift` (positive
lags), `rolling`, or `ewm`, which is causal by construction, and every cross-sectional
step is row-wise. A session without a row after a symbol's first row is a no-trade bar
(protocol rule 5), so windows and the horizon count market sessions. The forward fill
is not bounded by the symbol's last row: on day `t` a real-time system cannot know that
a stock will never trade again. The only forward-looking frame is `alive` (sessions up
to the last real row), used to make a label missing when the exit session falls after a
delisting.

**Indicator conventions.** Returns are simple returns. Volatility is the sample standard
deviation (`ddof = 1`) of daily returns. RSI and ATR use Wilder smoothing as an EMA with
`alpha = 1 / n` seeded with the first value rather than an `n`-day average; the two agree
long before the 252-session history requirement. MACD uses EMAs with `alpha = 2 / (span
+ 1)`; the histogram and ATR are divided by `close` so they are comparable across price
levels. Bollinger %B uses the population standard deviation (`ddof = 0`), the usual
convention. Amihud is the mean of `|return| / traded value` over the sessions in the
window that traded, per billion VND. The 252-session high uses `high`, not `close`.

**Ranks.** Each feature is ranked within its date among universe members only, scaled to
`(rank - 1) / (n - 1)` with average ties (0.5 when a date has one stock). Missing raw
values stay missing in the rank. The target is the same rank of the forward return
among members with a known label.

**Development panel.** `vn-rank features` loads prices strictly before `lockbox_start`
(design of the lockbox guard, point 3), so the last labels end on 2025-09-30 and signal
dates whose exit falls in the lockbox have a missing target. The panel is written to
`data/processed/panel.parquet` with `schema.json` (one description per column, dtypes,
feature groups, and the snapshot metadata).

**Testing.** The truncation test rebuilds the panel on data cut at six dates (including
a gap in a symbol's rows, a delisting, and the last session) and requires identical
features, ranks, eligibility, and row sets up to the cut. It was checked by injecting
leaks (tomorrow's close in a return, a centred rolling window, next-session liquidity in
the universe); each one fails the test.

## Validation, models, and tuning (Phase 3)

**Splits.** Test blocks are calendar half-years (the last one is 2025-07 to 2025-09).
A training row is purged when its `exit_date` is on or after the block start, and the
last `embargo_days` purged sessions are then dropped, so with `h = 5` the last training
signal date is 17 sessions before the first test date (the protocol's "at least 16").
Test rows whose label exits in the lockbox are dropped, which ends the last block on
2025-09-22. Dates with fewer than `min_stocks_per_date` eligible stocks are removed
before any split. The purge-alignment test checks every fold for label-window overlap
and the exact session gap.

**Lockbox guard (implemented).** `ensure_before_lockbox` treats a missing exit date as
a violation, because such a label cannot be shown to end before the lockbox. It runs in
`development_samples`, in `split_fold` on both sets, and in `RankingModel.fit` on every
frame, which covers every tuning trial.

**Tuning.** Optuna's seeded TPE sampler maximizes the mean daily Rank IC on the inner
validation window (the last 6 months of the training window, purged and embargoed like
a test block). A model with constant scores has no defined Rank IC and scores 0. The
final model is refit on the whole training window with the best parameters; LightGBM
reuses the number of rounds that early stopping found on the inner validation set
(early stopping uses L2 loss, which is cheap to evaluate every round, while the trial is
judged by Rank IC). Test blocks are only scored, never used for a choice.

**Constant scores.** A date on which a model gives every stock the same score has a
Rank IC of 0 and no quantiles. It happened once: on the 2023H2 fold every Elastic Net
trial with a non-zero fit had a negative validation Rank IC (about -0.03), so tuning
chose a penalty that zeroes every coefficient. Counting those dates as 0 rather than
dropping them keeps an abstaining model from looking better than it is.

**Seeds.** Baselines and Elastic Net (cyclic coordinate descent) are deterministic and
run once. LightGBM uses feature and bagging subsampling, so it is tuned once with
`project.seed` and refit with each of the 5 `models.seeds`; results are the mean and
standard deviation across those seeds. `deterministic = true` and `force_row_wise`
make a LightGBM run repeatable for a given seed on the same machine and thread count.

**Inputs.** All models use the 15 feature ranks. Elastic Net centres them at 0.5 and
fills the rare missing rank (`rank_log_tv_20`, 0.018% of rows) with 0.5, the median
rank; LightGBM keeps NaN and routes it natively. Approved by the author.

**Experiment tracking.** MLflow 3.18 refuses its file-store backend, so runs go to a
SQLite database `mlruns/mlflow.db` with artifacts in `mlruns/artifacts` (both
gitignored). Each `train` run is a parent run (config hash and git commit) with one
nested run per model and fold, holding the chosen parameters, validation and test mean
Rank IC, and the Optuna trial table. The parent run records the total number of
configurations tried.

**Module additions.** `validation/walk_forward.py` holds the fold loop, the results
table, and the tripwire check, which keeps `cli.py` thin (approved by the author).
MLflow logging lives in `utils/logging.py`. scikit-learn ships no type information, so
`pyproject.toml` tells mypy to ignore its missing imports.

## MLP (Phase 4)

The network has 2 or 3 hidden layers of equal width with ReLU and dropout, trained with
AdamW (weight decay) on mini-batches of 4,096 rows to the L2 loss on the rank target. Its
inputs are the centred feature ranks with the same 0.5 fill as Elastic Net. It is tuned
with the same Optuna budget as LightGBM (30 trials per fold) over width (16 to 128),
depth (2 or 3), dropout (0 to 0.5), weight decay (1e-6 to 1e-2), and learning rate (1e-4
to 1e-2); these ranges were chosen by the agent under the author's standing approval.
Early stopping watches the inner-validation L2 loss with a patience of 5 epochs (at most
30), and the refit trains for the best epoch count. Initialization, dropout, and batch
order all come from the seed, so a fit is deterministic on one machine; it is refit with
the 5 seeds like LightGBM.

## Backtest (Phase 5)

**Execution.** Scores at the close of `t` are traded at the close of `t + 1`, every
`h = 5` sessions, which is exactly the label's entry and exit. The top decile (about 32
stocks) is held at equal weight; drifted weights earn close-to-close returns and cash
earns nothing. Costs are charged on the traded value on the trade day, and the portfolio
is sold on the last day (that cost counts too).

**Constraints.** A stock that closed within 0.5 percentage points of its limit-up price
cannot be bought, one at limit-down cannot be sold, and a position cannot grow beyond
10% of the stock's 20-day average traded value (known before the trade) at a 10 billion
VND portfolio. When blocked sales would push weights above 1, the purchases are scaled
down, so the book is never levered.

**Costs.** Brokerage 0.15% and slippage 0.10% per side, plus the 0.1% selling tax: about
0.6% per round trip. Large Vietnamese brokers' effective fee was about 0.16-0.17% in
2025 (range 0.1-0.3%), and the universe trades at least 1 billion VND a day, so 0.10%
slippage is moderate. Phase 6 varies both.

**Benchmarks and aggregation.** The equal-weight portfolio of every scored stock uses the
same schedule, costs, and constraints, so the excess return isolates the ranking.
VNINDEX is a buy-and-hold reference without costs. A stochastic model's portfolio is the
average of its 5 seed portfolios, and its Sharpe interval comes from that average series.

**Intervals.** All intervals are 95% moving-block bootstrap percentile intervals (20-day
blocks, 10,000 resamples, seeded) on daily series: mean Rank IC (criterion C1), paired
Rank IC differences, Sharpe, and daily excess return.

**Module additions.** `backtest/runner.py` builds schedules, runs strategies and
benchmarks, and renders the report; `pipeline.py` holds the I/O of the `train` and
`backtest` commands so `cli.py` stays a dispatcher; `utils/markdown.py` renders tables
without a new dependency.

## Ablation and sensitivity (Phase 6)

Each feature group is dropped in turn and LightGBM is rerun with the unchanged protocol
(same folds, tuning budget, seeds). The liquidity threshold is varied by rebuilding the
panel at 0.5, 2, and 5 billion VND, as pre-registered; these runs never replace the
primary result. The cost grid re-trades the primary predictions, and the equal-weight
benchmark pays the same costs. The SHAP summary explains the last fold's model, rebuilt
with the same tuning (deterministic) and seed 0, on 5,000 of its test rows.

## Lockbox and monitoring (Phase 7)

**Final model.** For the lockbox the training window is all development data: labels
that exit before 2025-10-01, embargoed 10 more sessions, tuned on its last 6 months
exactly as in walk-forward. Lockbox samples are signal dates from 2025-10-01 whose
label has matured in the snapshot. Every model is evaluated; C2 refers to LightGBM.

**Run once.** The command needs `--confirm-lockbox`, refuses if the lock file exists or
the git tree is dirty, and writes the lock file before computing anything. Before the
real run, the whole train, lockbox, and update chain was exercised on a synthetic
snapshot in the test suite.

**Update.** The frozen LightGBM boosters (5 seeds) score the newest session; the buy list
and every score go to `data/live/`, and realized Rank IC is tracked once labels mature.
Fetching new data means a complete fresh snapshot (`update --fetch`, into a dated
folder), because adjusted prices change with corporate actions and cannot be appended to
an older snapshot.

## Exploratory low-volatility comparison (after the lockbox)

A review of the finished results found two design gaps. The pre-registered baselines
(momentum, reversal) carry no signal in this market, so beating them says little, while
the low-volatility effect, well known before the project, explains most of the signal.
And excess return over equal weight ignores beta, which flatters a low-beta portfolio.
`vn-rank exploratory` therefore scores two low-volatility rules on exactly LightGBM's
rows and reports the paired Rank IC difference and the net alpha after beta to the
equal-weight benchmark (moving-block bootstrap interval, both series resampled with the
same blocks). The rules are module constants, not config keys, so the config hash in the
lockbox lock file stays valid. The finding: the textbook rule matches LightGBM's mean
Rank IC, LightGBM is steadier, and no strategy has a positive beta-adjusted alpha after
costs. It is labelled exploratory everywhere.

## Exploratory extension: literature features and XGBoost (2026-11-08)

Registered in protocol Section 12 (commit `2c12260`) before any code; implemented in
`ff3d4ff` and run once. The four features come from the anomaly literature on volatility
and lottery-like stocks: rolling beta and residual (idiosyncratic) volatility against
VNINDEX over 60 sessions, the largest daily return over 20, and skewness over 60. Two
candidates were dropped up front because both would leak future information: turnover
needs point-in-time shares outstanding, which the snapshot lacks, and the nominal price
level of adjusted prices depends on later corporate actions. XGBoost mirrors the LightGBM
setup so that only the library differs. Its new windows and search space are module
constants (`extension.py`, `models/xgb.py`) so the config hash in the lockbox lock file
stays valid; `xgboost==3.2.0` was added with the author's approval.

Result: H1 holds (the features add about 0.010 Rank IC, and XGBoost replicates the gain);
H2 shows no library effect; H3 does not hold (the extended model is not significantly
better than the 60-day volatility rule). One caveat on H1: the features were chosen after
the earlier analysis showed that volatility drives the signal, so they are informed by
the same data, even though the tests were fixed before the run. The idiosyncratic
volatility rule had the best net portfolio, but it is one of several descriptive
comparisons and carries no claim; only future data (`vn-rank update`) can confirm it.

## Core rationale (Phase 8)

**Why a 5-day horizon.** One trading week is long enough for price–volume effects
(short-term reversal, volatility, liquidity) to show up above the daily noise and the
T+2 settlement cycle, and short enough to give about 900 non-overlapping-ish walk-forward
signal dates from under 4 years of data. The cost results show the price of this choice:
weekly rebalancing of a fast-changing ranking costs about 20% a year at 0.6% per round
trip, so a longer horizon or a turnover rule is the natural next experiment.

**Why rank targets.** The question is about ordering stocks within a day, not predicting
returns. Ranking the forward return per date removes the market's common move (which
dominates raw returns, especially in 2022) and the heavy tails of single-stock returns,
so an L2 loss on the rank target optimizes something close to Rank IC and no model is
dominated by a few extreme days. Features are ranked the same way, so every model sees
inputs on the same scale in every period.

**Why purge and embargo.** A label uses prices up to `t + 6`, so with daily signals the
training samples just before a test block overlap its first labels. Purging on the exit
date removes that overlap exactly; the 10-session embargo adds a margin for serial
correlation in features and returns. Without it, the last training labels would share
returns with the first test labels and inflate test Rank IC.

**Why Rank IC first and Sharpe second.** Rank IC is measured on about 300 stocks every
day, so 925 test dates give a tight interval (0.042 to 0.076 for LightGBM). A portfolio
Sharpe ratio is one number per path, its standard error over 3.75 years is about 0.6,
and it mixes the signal with costs, constraints, and the market regime. The results show
why both matter: the Rank IC is clearly positive, while the portfolio after costs is not
distinguishable from the benchmark.
