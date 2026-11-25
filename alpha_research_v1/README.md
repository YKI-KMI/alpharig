# alpharig

**Leakage-proof automated cross-sectional equity alpha research.**

alpharig is a research framework whose first job is not to find alpha. Its
first job is to make it hard to fool ourselves. Point-in-time data, purged
date-grouped validation, fold-only preprocessing, an append-only trial ledger,
a single-use holdout and a leakage audit are built into the plumbing rather
than bolted on at the end.

> **Research only.** Nothing in this repository is investment advice. Outputs
> must not be used to place orders or manage capital without independent
> review.

## Stage status

The build is split into gated stages; see [`docs/STAGES.md`](docs/STAGES.md)
and run `alpharig stages --verbose`.

| Stage | Area | Status |
|---:|---|---|
| 0 | Ground rules and scaffolding | done |
| 1 | Data layer and synthetic lab | done |
| 2 | Signal library and feature engine | pending |
| 3 | Labels and targets | pending |
| 4 | Validation harness and leakage audit | pending |
| 5 | Models, baseline gate, trial ledger | pending |
| 6 | Blending, orthogonalisation, regimes | pending |
| 7 | Positions, costs, capacity | pending |
| 8 | Live monitoring, decay, refit | pending |
| 9 | CLI and reports | pending |
| 10 | Automation loop | pending |
| 11 | Release and definition of done | pending |

## Install

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
```

Optional real-data adapter:

```bash
python -m pip install -e ".[yfinance]"
```

## Quickstart (Stage 1)

Generate the synthetic oracle panel and inspect its ground truth:

```bash
alpharig synth --out data/synth --config configs/synth_lab.yaml
```

Override any parameter:

```bash
alpharig synth --out data/synth_small --stocks 100 --periods 250 \
  --seed 3 --rho 0.3 --signal-ic 0.05 --half-life 20
```

Run the test suite:

```bash
pytest -q
```

Show the stage plan:

```bash
alpharig stages --verbose
```

## What Stage 1 provides

- **Point-in-time store** (`alpharig.data.store.PointInTimeStore`). Every
  value at date `t` comes from the latest row with `release_ts <= t`. A bare
  date means end of day, so a release dated that day is visible. Rows whose
  release precedes their period end are rejected outright.
- **Trading calendar** (`alpharig.data.calendar.TradingCalendar`). All date
  arithmetic is trading-day arithmetic.
- **Point-in-time universe** (`alpharig.data.universe.PointInTimeUniverse`).
  Existence, trailing ADV, restrictions and price availability, all computed
  from data at or before `t`.
- **Adapters** (`alpharig.data.adapters`). CSV/Parquet and optional yfinance.
  Missing release timestamps are filled with a declared conservative lag and
  raise a critical `DataQualityWarning`; a price panel with no delisted names
  raises a survivorship warning. The same text is repeated in every report
  header.
- **Synthetic lab** (`alpharig.synth`). N names x T periods with market and
  sector factors, configurable within-date correlation, planted signal IC and
  half-life, an optional sign-flip in high-volatility regimes, realistic
  fundamental release lags, delistings, restrictions and fat tails. Ground
  truth is stored with the panel, so later stages can test whether they
  recover what was planted.
- **Effective sample size** (`alpharig.stats`). `N_eff` and overlap
  corrections are shared utilities, not report-time afterthoughts.

## Repository layout

```
alpharig/
  pyproject.toml
  README.md
  configs/                 # YAML experiment configs
  docs/STAGES.md           # gated implementation plan
  docs/stages/             # per-stage working checklists
  src/alpharig/
    data/                  # adapters, PIT store, universe, calendar, quality
    synth/                 # synthetic oracle generator
    stats/                 # effective sample size, panel statistics
    stages.py              # executable stage registry
    cli.py
  tests/
```

Later stages add `features/`, `labels/`, `validation/`, `models/`,
`selection/`, `portfolio/` and `monitor/`.

## Data-source limits (free sources)

Free and semi-free equity data cannot support an unbiased study on its own.
alpharig does not hide this:

- **Survivorship.** yfinance returns only tickers listed today and silently
  drops delisted names. Any backtest built on it overstates returns. alpharig
  warns and repeats the warning in the report header.
- **Restatements.** Fundamental values are revised. Without a vintage column,
  old results cannot be reproduced and point-in-time joins are approximate.
- **Missing release timestamps.** yfinance fundamentals have period ends but
  no release dates. alpharig fills a conservative assumed lag and marks the
  run critical: results are look-ahead biased until a real PIT source is used.
- **Revised price history.** Adjusted prices are recalculated with today's
  split/dividend factors; historical levels are not what an investor saw.
- **Missing delisting returns.** Free panels often omit the final delisting
  return, which is usually negative and large.
- **No intraday timestamps.** Same-day releases are assumed visible at the
  close of that day.

The synthetic generator exists because it is the only dataset where the right
answer is known. It is a test oracle, not evidence about real markets.

## Ground rules encoded in the codebase

- Count dates, not rows: every reported statistic carries `N_eff`.
- Point-in-time everywhere: `period_end`, `release_ts`, `vintage` on every
  fundamental row; per-field lags.
- Preprocessing is fitted inside training folds only.
- Validation is date-grouped, purged, embargoed and walk-forward.
- Every evaluated configuration is logged; reported scores are deflated for
  the number of trials.
- The holdout opens once, with a recorded hash and timestamp.
- A too-good result is a leak until the audit says otherwise.
- Net-of-cost P&L is the only headline.
- Everything is reproducible from config, seed, data snapshot hash and code
  version.
