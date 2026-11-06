# AGENT.md — VN-Rank

Instructions for the AI coding agent that builds this repository end-to-end.
Read this file completely before writing any code. Re-read the relevant section at the start of every phase.

---

## 1. Mission

Build **VN-Rank** (cross-sectional stock ranking on the Vietnamese market, net of trading costs), a portfolio-grade research project that will be shown to quantitative-finance recruiters (target role: Data Science Intern at a quant trading firm).

**Research question**

> Can price–volume features rank Vietnamese stocks by 5-day forward return, net of trading costs?

**What this project is and is not.** The model outputs a **daily ranking of stocks by expected relative performance**. It does **not** forecast price levels, and it does **not** call market direction. Success is measured by how well the predicted ranking matches the realized ranking (Rank IC), and by whether a simple portfolio built from the ranking survives trading costs. Accuracy, RMSE, and MAPE are never headline metrics.

The deliverable is a **project showcase repository**: clean structure, readable code, reproducible results, honest reporting. A negative result that is rigorously evaluated is acceptable. A positive result that contains leakage is a failure.

**Language rules**

- Talk to the user in **Vietnamese**.
- Write all code, comments, docstrings, commit messages, and documentation in **English**.

---

## 2. Operating rules (non-negotiable)

### 2.1 Phase gates

The work is split into phases (Section 7). **After finishing a phase you must stop and ask the user for approval before starting the next one.** Never chain phases.

At the end of every phase, send a **Phase Report** in Vietnamese with exactly this structure:

1. **Done** — what was built (files, modules).
2. **Evidence** — test results, lint/type-check status, key numbers, plots produced.
3. **Decisions made** — every choice you made and why.
4. **Open issues / risks** — anything surprising, any deviation from this spec.
5. **Question** — "Bạn có đồng ý chuyển sang Phase N+1 không?" plus any decisions you need from the user.

Then **stop and wait**. Do not start the next phase on silence or on an ambiguous reply.

### 2.2 Ask before you decide

Ask the user (briefly, with a recommended default) before:

- Choosing any parameter that changes results materially (liquidity threshold, cost assumptions, top-N size, embargo length).
- Deviating from this spec in any way.
- Adding a dependency not listed in Section 6.
- Deleting or re-downloading data.
- Using any paid vnstock tier or any data source other than vnstock.
- Running the lockbox evaluation (Phase 7, once only).

### 2.3 Honesty rules

- Never invent numbers, API behavior, or column names. Inspect real outputs (`df.columns`, `df.dtypes`, `df.attrs`) before coding against them.
- Never tune anything on the lockbox period or on walk-forward test folds.
- Report failures and negative results as they are. Do not cherry-pick periods, folds, seeds, or metrics.
- If a result looks too good, assume leakage first and investigate before reporting it.
- State every metric definition explicitly (for example, the Sharpe annualization convention and whether a number is per-day or per-year). Never compare numbers across sources without checking their definitions.
- Keep a count of every configuration tried (MLflow does this). Report that count.

### 2.4 Working style

- Small, reviewable commits (Section 5.6). One logical change per commit.
- Run `make check` (lint + type-check + tests) before every Phase Report. Do not report a phase as done while it fails.
- Prefer deleting code to adding code. If something is not required by this spec, do not build it (Section 8).

---

## 3. Project specification

### 3.1 Data facts (verified against vnstocks.com docs, generation 5)

- Library: `vnstock` generation 5. Docs: `https://vnstocks.com/docs/vnstock/thi-truong-market` (prices) and `https://vnstocks.com/docs/vnstock/tham-chieu-reference` (symbol lists). **Re-read both pages before Phase 1** and verify against real outputs; the API changes between versions.
- Only the **Community tier** is used: daily OHLCV, symbol lists, VNINDEX.
- Daily history is capped at about **2,920 days (~8 years)** on Guest/Community, so data starts around **2018-11**.
- Stock prices are **adjusted for corporate actions**, in **thousand VND**. The `vci` and `kbs` sources adjust differently: **use one source only (`vci`) and never mix**.
- `Reference().equity.list_by_exchange()` returns **currently listed** symbols. Delisted symbols are most likely absent → **survivorship bias is unavoidable**; it must be documented and mitigated by benchmarking against an equal-weight portfolio of the same universe.
- `ohlcv()` returns columns `time, open, high, low, close, volume` with `time` in `Asia/Ho_Chi_Minh`. A single-symbol frame has **no symbol column**; add it yourself.
- Guest rate limit is about 20 requests/minute. Throttle (~3.2 s between calls), make the downloader resumable, and retry on rate-limit errors. Check the user's actual limits with `vnstock policy`.
- Data is a **snapshot**: record download date, source, and vnstock version in metadata. Never commit raw data (see the vnstock license page).

### 3.2 Universe (point-in-time, uses only past information)

A stock is eligible on date `t` if all hold:

- at least 252 trading days of history up to `t`;
- median traded value over the previous 60 days ≥ `min_traded_value` (config; default proposed to the user in Phase 2);
- it traded on day `t` (volume > 0).

Traded value is `close × volume` and is only approximate in earlier years because prices are adjusted. Document this.

### 3.3 Label

- Signal computed at close of day `t`; entry at close of `t+1`; exit at close of `t+1+h` with `h = 5`.
- `forward_return = close[t+1+h] / close[t+1] - 1`.
- Training target = **cross-sectional rank of `forward_return` within each date** (scaled to [0, 1] or z-scored).
- Rebalance every `h` trading days so holding periods do not overlap.

### 3.4 Features (about 18, hand-implemented in pandas/numpy)

| Group | Features |
|---|---|
| Momentum / reversal | returns over 1, 5, 20, 60, 120 days |
| Volatility | std of daily returns (20, 60 days), ATR(14)/close |
| Technical | RSI(14), MACD histogram / close, Bollinger %B (20, 2) |
| Volume / liquidity | log median traded value (20d), volume / mean volume (20d), Amihud illiquidity (20d) |
| Price position | close / rolling 252-day high |

Each feature is converted to a **cross-sectional rank per date**. Features use only data up to and including day `t`.

### 3.5 Validation protocol

| Period | Dates | Use |
|---|---|---|
| Feature warm-up | 2018-11 → 2019-11 | needs 252 days |
| Initial train | 2019-11 → 2021-12 | expanding window start |
| Walk-forward (dev) | 2022-01 → 2025-09, 6-month test blocks | all design decisions |
| **Lockbox** | 2025-10 → latest | evaluated **once**, at the very end |

- **Purge**: drop train samples whose label window `[t+1, t+1+h]` overlaps the test block. **Embargo**: additional `embargo_days` (default 10) after the train end.
- **Hyperparameter tuning (Optuna)** happens **inside** each training window: validation = last 6 months of that window (with purge/embargo). Never look at the test block.
- The lockbox start date lives in `configs/default.yaml`. Training, tuning, and CV code must **assert** `max(date) < lockbox_start`.

### 3.6 Models (same folds, same features, same metrics)

1. **Baselines (no learning):** 20-day momentum rank; 5-day reversal rank; equal-weight universe.
2. **Elastic Net** (scikit-learn) — explicit regularization.
3. **LightGBM** — main model; regularization via `lambda_l1`, `lambda_l2`, `min_child_samples`, `feature_fraction`, early stopping.
4. **MLP** (PyTorch, 2–3 layers, dropout, weight decay) — the neural-network comparison. Report honestly if LightGBM wins.

**Seeds:** the MLP (and LightGBM if bagging or feature subsampling is used) is stochastic. Run each with **5 seeds** (configurable) and report **mean ± standard deviation** of the metrics. Deterministic models (baselines, Elastic Net) run once.

### 3.7 Evaluation

**Naming:** throughout the repo, docs, and README, **"IC" means Rank IC**, i.e. the per-date **Spearman** correlation between predicted scores and realized forward returns. Say "Rank IC" explicitly in the README so results are comparable with other quant literature.

**Signal level (primary result):** per-date Rank IC, mean Rank IC, IC IR (mean/std), t-stat with Newey-West (lag `h-1`) because 5-day labels overlap, quintile mean returns, top-minus-bottom spread, and **signal autocorrelation** (rank correlation of scores between consecutive rebalance dates; a stability and turnover diagnostic).

**Portfolio level (backtest, long-only, no short selling in Vietnam):**

- Top decile by model score (about 30–50 stocks), equal-weight, rebalanced every `h` days.
- Costs (all in config, with sensitivity analysis): brokerage fee per side, 0.1% selling tax, slippage per side. The user approves the default values in Phase 5 after checking real fee schedules.
- Constraints: skip buys of limit-up stocks and sells of limit-down stocks (band per exchange from config: HOSE ±7%, HNX ±10%, UPCoM ±15% — verify current rules), cap position size as a fraction of 20-day average traded value.
- Benchmarks: **equal-weight same-universe portfolio** (primary), VNINDEX, momentum baseline.
- Metrics: annualized return, volatility, Sharpe, max drawdown, turnover, per-year breakdown.
- **Optional turnover-control variant (only if the user approves in Phase 5, after seeing baseline turnover):** keep a current holding while its rank stays within a wider band (for example the top 20%), and fill vacated slots from the top 10%. One parameter, easy to explain, reduces cost drag. Do not build it unless approved.
- **Block bootstrap** confidence intervals for Sharpe and mean IC.

**Known statistical limits (state them in the README):** with about 3.75 years of walk-forward data the standard error of annualized Sharpe is roughly 0.6; for the 12-month lockbox it is roughly 1.2. Therefore **IC is the primary metric; Sharpe is secondary and always reported with its interval.**

**Realistic expectations (context from prior work).** Public benchmarks of gradient-boosting and neural models on a Chinese equity universe (Microsoft Qlib) report Rank IC of only a few hundredths (roughly 0.02–0.05) and show LightGBM beating an MLP. Those numbers come from a different market, label, and cost model and must **not** be treated as targets or compared directly. They only indicate that a small positive Rank IC is a normal, meaningful result, and that large values should trigger a leakage investigation. In the JPX Tokyo ranking competition, leakage through supplemental data inflated public scores, and simple return-based models stayed competitive; keep the momentum and reversal baselines honest and simple.

### 3.8 Ablation and explanation

- Drop each of the 5 feature groups in turn and re-run LightGBM.
- Cost sensitivity (vary fee and slippage).
- One SHAP summary plot for LightGBM.

### 3.9 Update and monitoring (small)

A single `update` command: fetch the latest days, build features for the newest date, score with the saved model, write the top-N list with its date. Once labels mature (h days later), compute realized IC and plot realized vs backtest IC. No Docker, no API server, no dashboard.

---

## 4. Repository structure

Keep it minimal. Do not add folders that are not listed here.

```
vn-rank/
├── README.md                  # showcase: question, protocol, results, limitations
├── AGENT.md                   # this file
├── pyproject.toml             # deps, ruff, mypy, pytest config
├── Makefile                   # setup, check, download, features, train, backtest, report
├── .gitignore                 # data/, mlruns/, .venv/, caches
├── configs/
│   └── default.yaml           # every tunable value lives here
├── src/vn_rank/
│   ├── config.py              # typed config loading (dataclasses)
│   ├── cli.py                 # thin command-line entry point (argparse)
│   ├── data/
│   │   ├── download.py        # vnstock → Parquet, resumable, rate-limited
│   │   ├── storage.py         # Parquet/DuckDB read-write helpers
│   │   ├── validate.py        # data-quality checks
│   │   └── universe.py        # point-in-time eligibility
│   ├── features/
│   │   ├── indicators.py      # RSI, MACD, Bollinger, ATR, returns, ...
│   │   ├── builder.py         # assemble feature panel
│   │   ├── transforms.py      # cross-sectional rank
│   │   └── labels.py          # forward-return label
│   ├── validation/
│   │   └── splits.py          # walk-forward + purge + embargo
│   ├── models/
│   │   ├── base.py            # common Model interface
│   │   ├── baselines.py
│   │   ├── linear.py          # Elastic Net
│   │   ├── gbm.py             # LightGBM
│   │   ├── mlp.py             # PyTorch MLP
│   │   └── tuning.py          # Optuna (inner validation only)
│   ├── backtest/
│   │   ├── engine.py          # rebalance loop
│   │   ├── costs.py           # fees, tax, slippage
│   │   └── constraints.py     # limit-lock, liquidity cap
│   ├── evaluation/
│   │   ├── ic.py              # IC, IC IR, Newey-West t-stat
│   │   ├── performance.py     # Sharpe, drawdown, turnover
│   │   └── bootstrap.py       # block bootstrap CIs
│   ├── monitoring/
│   │   └── update.py          # daily update + realized-IC tracking
│   └── utils/
│       ├── logging.py
│       └── seeds.py
├── tests/                     # mirrors src/ layout
├── notebooks/                 # EDA and result plots only; no core logic; outputs stripped
├── docs/
│   ├── protocol.md            # pre-registered protocol (locked after Phase 0)
│   └── design_notes.md        # one-paragraph rationale per major choice
├── reports/                   # small committed figures and tables used in the README
└── data/                      # gitignored: raw/, interim/, processed/
```

---

## 5. Code standards (enterprise grade)

### 5.1 Size and structure limits

- **Functions: at most ~30 lines**, one responsibility, at most 4 parameters (group extras into a dataclass), nesting depth ≤ 3. If a function grows past the limit, split it.
- **Modules: at most ~250 lines.** Split by responsibility, not by arbitrary line count.
- **Never write one long block or script.** No top-level procedural code outside `if __name__ == "__main__"` or the CLI.
- **Separate I/O from computation.** Pure functions take DataFrames/arrays and return DataFrames/arrays; only `storage.py`, `download.py`, and `cli.py` touch disk or network.
- No global mutable state. No hidden randomness: set seeds through `utils/seeds.py` and log them.
- **No magic numbers.** Every window length, threshold, fee, date, and seed comes from `configs/default.yaml` through typed config objects.

### 5.2 Typing, docs, naming

- Python ≥ 3.11. Full type hints on every public function; `mypy --strict` passes on `src/`.
- Google-style docstrings on every public function and class: purpose, args, returns, raises, and a note on **leakage-relevant behavior** where applicable (for example "uses only data up to and including t").
- `snake_case` for functions and variables, `PascalCase` for classes, descriptive names (`purged_walk_forward_splits`, not `split2`).
- Use `pathlib`, `logging` (not `print`), and `dataclasses`. Raise specific exceptions with clear messages; never use bare `except`.

### 5.3 Tooling

- `ruff` for lint and formatting, `mypy` for types, `pytest` for tests, `pre-commit` to run them.
- `make check` runs lint, type-check, and tests; it must pass before every Phase Report.
- Pin dependency versions in `pyproject.toml`.

### 5.4 Hand-written core (do not outsource)

The following must be implemented by hand in this repo, with unit tests, so the author can explain every line:

- technical indicators (returns, RSI, MACD, Bollinger %B, ATR, Amihud, ...);
- purged walk-forward splitter with embargo;
- IC and IC IR computation;
- backtest rebalance loop, costs, and constraints.

**Forbidden libraries:** `vectorbt`, `backtrader`, `zipline`, `TA-Lib`, `pandas-ta`, `mlfinlab`, and any other ready-made backtesting or indicator library. `statsmodels` is allowed only for Newey-West/HAC standard errors.

### 5.5 Tests (required)

Beyond ordinary unit tests, three tests are mandatory:

1. **No look-ahead in features:** features computed on data truncated at day `t` equal the features at day `t` computed on the full dataset.
2. **Label and purge alignment:** no training sample has a label window overlapping its test block; embargo is respected.
3. **Backtest cost accounting:** a hand-computed example (for instance, rotating the whole portfolio once) matches the engine's cost output exactly.

Add tests for config validation, universe eligibility, Rank IC and signal autocorrelation on synthetic data with known answers, and the lockbox guard.

### 5.6 Git conventions

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- One logical change per commit; commit after each module with its tests.
- Never commit data, `mlruns/`, credentials, or API keys.

---

## 6. Dependencies (allowed list)

`vnstock`, `pandas`, `numpy`, `pyarrow`, `duckdb`, `scipy`, `statsmodels` (HAC only), `scikit-learn`, `lightgbm`, `optuna`, `mlflow`, `torch`, `shap`, `matplotlib`, `pyyaml`, `tqdm`; dev: `pytest`, `ruff`, `mypy`, `pre-commit`.

Anything else requires user approval (Section 2.2).

---

## 7. Phases

Each phase lists deliverables, acceptance criteria, and the decisions to raise with the user. **Stop at the end of each phase and wait for approval.**

### Phase 0 — Scaffold and protocol

**Deliverables**
- Repository skeleton exactly as in Section 4 (empty modules may contain only docstrings), `pyproject.toml`, `Makefile`, `.gitignore`, pre-commit config, `configs/default.yaml`, `config.py` with typed loading and validation.
- `docs/protocol.md`: research question, universe rule, label, features, validation periods, models, metrics, and **pre-registered success criteria expressed in Rank IC and its confidence interval** (not Sharpe). Keep the criteria modest and realistic (see "Realistic expectations" in Section 3.7); for example, "the lower bound of the mean Rank IC confidence interval on walk-forward folds is above zero, and the lockbox Rank IC has the same sign". Propose the criteria to the user; do not finalize them yourself.
- README skeleton (title, question, protocol summary, limitations placeholders).
- Lockbox guard design (see Section 9).

**Acceptance:** `make setup` and `make check` pass on an empty project; config round-trips through tests.

**Ask the user:** approve `docs/protocol.md` and the success criteria. After approval the protocol is **locked**; later changes must be logged in the file's changelog with a reason.

### Phase 1 — Data

**Deliverables**
- Re-read the vnstock docs and inspect real outputs; record findings in `docs/design_notes.md`.
- `download.py`: symbol list, per-symbol Parquet, resumable, throttled, retrying; VNINDEX download; metadata (download date, source, vnstock version).
- `storage.py`, `validate.py`: duplicates, OHLC consistency, zero/NaN volume, returns beyond exchange band, gaps versus the VNINDEX calendar.
- A data-quality report and a **plot of the number of tickers per year** (to expose survivorship bias).

**Acceptance:** full download completes and can resume after interruption; validation report generated; tests for validators pass.

**Ask the user:** how to treat flagged anomalies (drop, cap, or keep) and whether the observed history length is acceptable.

### Phase 2 — Universe, features, labels

**Deliverables**
- `universe.py`, `indicators.py`, `transforms.py`, `builder.py`, `labels.py` per Section 3.
- The three mandatory tests from Section 5.5 (look-ahead, purge alignment can wait for Phase 3; look-ahead test is required here).
- Processed panel written to `data/processed/` with a schema description.
- Basic EDA notebook: feature distributions, missingness, universe size over time.

**Acceptance:** look-ahead test passes; panel row counts and universe size reported.

**Ask the user:** `min_traded_value` threshold (propose a default with the reasoning and a sensitivity range).

### Phase 3 — Validation, baselines, Elastic Net, LightGBM

**Deliverables**
- `splits.py` with purge and embargo, plus the purge-alignment test.
- `baselines.py`, `linear.py`, `gbm.py`, `tuning.py` (inner-window Optuna), MLflow logging per fold.
- `ic.py` with Rank IC, IC IR, Newey-West t-stat, quintile returns, and signal autocorrelation.
- Signal-level results table for all models across walk-forward folds.

**Acceptance:** purge test passes; the lockbox assertion is enforced in code and tested; results reproducible with fixed seeds.

**Ask the user:** whether results look plausible (flag anything suspiciously high), and the Optuna search-space bounds.

### Phase 4 — MLP

**Deliverables**
- `mlp.py` (PyTorch, small, dropout, weight decay, early stopping on the inner validation window), same folds and metrics as the other models.
- Multi-seed runs (5 by default, configurable) for the MLP, reported as mean ± standard deviation, with seeds logged in MLflow.
- Comparison table: baselines vs Elastic Net vs LightGBM vs MLP.

**Acceptance:** identical evaluation protocol across models; training is deterministic given the seed.

**Ask the user:** whether to keep the MLP configuration as is.

### Phase 5 — Backtest and performance

**Deliverables**
- `costs.py`, `constraints.py`, `engine.py`, `performance.py`, `bootstrap.py`.
- Cost-accounting test from Section 5.5.
- Backtest results on walk-forward folds: equity curves, Sharpe, max drawdown, turnover, per-year breakdown, bootstrap CIs, all versus the equal-weight same-universe benchmark, VNINDEX, and the momentum baseline.

**Acceptance:** cost test passes; metrics are net of costs; benchmarks computed on the same universe.

**Ask the user:** cost defaults (brokerage fee, slippage, position cap, top-N) after they check real fee schedules; propose a sensible range. Also report the baseline turnover and signal autocorrelation, and ask whether to build the optional turnover-control variant from Section 3.7.

### Phase 6 — Ablation and explanation

**Deliverables**
- Feature-group ablation, cost sensitivity, SHAP summary plot.
- Figures and tables saved to `reports/`.

**Acceptance:** each ablation uses the unchanged protocol; results reported even when uninteresting.

**Ask the user:** which figures and tables go into the README.

### Phase 7 — Lockbox and monitoring

**Deliverables**
- **Lockbox evaluation, run exactly once**, only after explicit user approval, via the guard in Section 9. Report results without any re-tuning afterwards.
- `monitoring/update.py` and the CLI command: update data, score newest date, write top-N list, track realized IC versus backtest IC.

**Acceptance:** lockbox lock file exists and a second run is refused; update command works end-to-end on current data.

**Ask the user:** explicit confirmation **before** the lockbox run, and again how to interpret the result (including negative outcomes).

### Phase 8 — Showcase polish

**Deliverables**
- Final `README.md`: question, data and its limits, protocol, results with confidence intervals, ablation, honest limitations (survivorship bias, adjusted prices, short history, the Sharpe standard-error caveat), metric definitions (Rank IC, Sharpe annualization convention), reproduction steps, and the repository map.
- Optional (ask the user first): a short "context" table in the README listing public Rank IC benchmarks from other markets (for example Qlib on a Chinese universe), clearly labeled as **not directly comparable** (different market, label, and costs).
- `docs/design_notes.md` completed (one short rationale per choice: why 5 days, why rank targets, why purge/embargo, why IC first).
- Clean notebooks (outputs stripped), tidy commit history, `make` targets documented.
- Suggested CV bullets with real numbers filled in.

**Acceptance:** a new user can reproduce the main results following the README; `make check` passes.

**Ask the user:** final review and sign-off.

---

## 8. Explicitly out of scope

Do not build any of the following unless the user asks:

intraday or order-book data; foreign/proprietary flow (paid tier); financial statements and macro data; LSTM/Transformer/RL/sentiment models; dozens of extra indicators; GARCH, portfolio optimization, Deflated Sharpe; Docker, FastAPI, Streamlit, dashboards; short-selling or leverage simulation.

---

## 9. Leakage and lockbox safeguards

**Leakage checklist (review at every phase):**

- Features use only data ≤ `t`; verified by the truncation test.
- Labels are built from `t+1` onward; training windows are purged and embargoed.
- Cross-sectional ranks are computed per date only (never across dates).
- Hyperparameters are tuned only inside the training window.
- The universe filter uses only past information.
- No fitted preprocessing step sees test or lockbox data.

**Lockbox guard:**

- `lockbox_start` is defined in config; every training, tuning, and CV entry point asserts that its data ends before it.
- The lockbox command requires an explicit `--confirm-lockbox` flag and refuses to run if `reports/lockbox_run.lock` exists; it writes that file (with timestamp, git commit, and config hash) when it runs.
- After the lockbox run, **no model, feature, or parameter change is allowed**. Any further experiment must be reported as exploratory.

---

## 10. Definition of done

- All phases approved by the user.
- `make check` green; the three mandatory tests pass.
- README contains the full protocol, results with confidence intervals, and an honest limitations section.
- Repo structure matches Section 4; no unused folders or dead code.
- No data, secrets, or experiment artifacts committed.
- The user can explain every module and every metric without help.
