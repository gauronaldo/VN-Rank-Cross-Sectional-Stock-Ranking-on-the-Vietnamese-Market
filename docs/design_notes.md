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

**Type stubs.** `mypy --strict` runs on `src/`. PyYAML has no inline types. The stub package
`types-PyYAML` is not on the allowed dependency list, so `yaml` is excluded from type
checking in `pyproject.toml` until that is approved. `pandas-stubs` raises the same question
once pandas code arrives in Phase 1.

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
constraints. This file is not in the repository map in AGENT.md section 4; see the Phase 0
report.

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

## Data source (Phase 0 finding, to be resolved in Phase 1)

On 2026-11-06 the PyPI project `vnstock` was quarantined ("cannot be installed or modified
until a security review is complete", since 2026-09-24). The upstream GitHub README says
generation 5 is distributed from the vendor's own package index on `vnstocks.com`, which
this build environment cannot reach. `vnstock` is therefore not yet listed in
`pyproject.toml`. The install route is decided with the user in Phase 1.

## Rationale to complete in Phase 8

- Why a 5-day horizon.
- Why rank targets.
- Why purge and embargo.
- Why IC first and Sharpe second.
