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
constraints. This file is not in the repository map in AGENT.md section 4; the author
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
   - refuses if `reports/lockbox_run.lock` exists;
   - refuses on a dirty git working tree, so that the recorded commit identifies the exact
     code.
   - writes the lock file **before** computing any result, so a crash part-way still counts
     as the one run. The lock file records the UTC timestamp, git commit, SHA-256 hash of the
     canonical config, and the last matured label date.

   The lock file is committed to git, because session containers are ephemeral.
5. **After the run.** No change to models, features, or parameters is allowed. Any later
   experiment is labelled exploratory in the README.

## Data source (Phase 1)

**Installation.** On 2026-11-06 the PyPI project `vnstock` was quarantined ("cannot be
installed or modified until a security review is complete", since 2026-09-24). Generation 5
is installed from the vendor's own index (`--extra-index-url https://vnstocks.com/api/packages`),
which the cloud build environment cannot reach. The author therefore downloads data on their
own machine. `vnstock` is not listed in `pyproject.toml` or `uv.lock`: it is installed into the
project environment with `uv pip install`, and commands that need it run with
`uv run --no-sync` so that uv does not remove it.

**Account.** Community tier (free registration, API key stored by `vnstock register`).
Published limits: Guest 20 requests/minute, Community 60 requests/minute (PyPI page). The
key is never committed or pasted anywhere.

**Verified against real outputs (probe run 2026-11-06, Windows 11, Python 3.12.3).**

- `vnstock` 5.2.1 (core 0.7.2). `vnstock status` reports that this version works until
  2027-10-27, so results must be reproducible from the stored snapshot, not from a
  re-download.
- Top-level API: `Reference`, `Market`, plus backward-compatible `Quote` and `Listing`
  (`Quote` defaults to source `KBS`, so it is not used).
- `Reference().equity.list_by_exchange(source=None)` (`None` means `vci`) returned 1,755 rows
  with columns `symbol, exchange, organ_name, organ_short_name, icb_code_lv2`, all pandas
  `string` dtype, on a `RangeIndex`. Exchange values seen include `HOSE` and `UPCOM`.
- Every frame carries provenance in `df.attrs`, for example `source='vci'`,
  `route='reference.market.symbols_by_exchange'`, `requested_at`, `request_count`, and
  `robots_txt`. The VCI host reports
  `trading: User-agent *, Allow /, Crawl-delay 10`.
- `Reference().equity.list_by_exchange()` exchange values: `UPCOM` 809, `HOSE` 408, `HNX` 299,
  and **`DELISTED` 239**. Delisted symbols are therefore listed; whether their history can
  be downloaded is checked by the third probe.
- Price history is `Market().equity(symbol).ohlcv(start, end, interval="1D", *, limit,
  source, count_back, length, floating=2)` and the same signature on
  `Market().index("VNINDEX")`; `equity` and `index` are methods, not attributes.
- `ohlcv` returns columns `time, open, high, low, close, volume` on a `RangeIndex`, with
  `time` as `datetime64[ns, Asia/Ho_Chi_Minh]` stamped 07:00 local, prices `float64` in
  thousand VND (FPT closed at 50.8 on 2026-11-06), and `volume` `int64`. There is no symbol
  column.
- **Default sources differ by asset class:** equities default to `vci`, but VNINDEX
  defaults to `kbs`. The downloader always passes `source="vci"` explicitly.
- **History cap:** FPT from 2015-01-01 returned 1,995 rows starting 2018-11-08, with a
  warning and `attrs['cap_rows_dropped'] = 962` (`tier_cap_history_days = 2920`).
- **VNINDEX returned only 1,000 rows** (from 2022-11-02) with either source; a longer
  window is checked by the third probe.
- **Rounding:** `floating=2` rounds prices to two decimals by default. Adjusted prices of
  low-priced stocks lose precision, so the downloader passes `floating=None`.
- **Batch calls do not help:** `Market().equity(["FPT", "VCB", "HPG"]).ohlcv(...)` made
  3 requests (`attrs['request_count'] = 3`) and returned one stacked frame without a
  symbol column, so symbols are downloaded one at a time.
- `vnstock.usage()` reports the Community limits as 60/minute, 3,600/hour, 10,000/day,
  and 100,000/month, with `history_days['daily'] = 2920`. A full download (about 1,760
  requests) fits within one day.
- Error classes: `RateLimitedError`, `AccessDeniedError`, and `CircuitOpenError` all
  subclass `DataSourceBlockedError`, which subclasses `NetworkError` and `VnstockError`.
  The downloader retries on `RateLimitedError` and network errors only.
- Latency: a single call took 0.2 to 8 seconds.

## Rationale to complete in Phase 8

- Why a 5-day horizon.
- Why rank targets.
- Why purge and embargo.
- Why IC first and Sharpe second.
