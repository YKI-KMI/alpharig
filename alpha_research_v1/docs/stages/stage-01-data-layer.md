# Stage 1 — Data layer and synthetic lab

**Status:** done — all acceptance tests green.
**Depends on:** Stage 0 utilities (`stats/effective.py`, conventions).
**Unblocks:** every later stage (features, labels, validation, portfolio).

## Build checklist

- [x] `data/schema.py` — canonical columns, date coercion, validation.
- [x] `data/quality.py` — `DataQualityWarning`, `DataQualityReport`,
      fundamental/price/membership checks, header repetition.
- [x] `data/store.py` — versioned `as_of(field, t)`, `as_of_many`,
      `feature_frame`, snapshot hash, look-ahead rejection.
- [x] `data/calendar.py` — trading-day navigation, business-day and
      month-end constructors.
- [x] `data/universe.py` — existence, trailing ADV, restrictions, price
      availability, no look-ahead.
- [x] `data/adapters.py` — interface, CSV/Parquet adapter, in-memory adapter,
      optional yfinance adapter with survivorship/restatement warnings.
- [x] `synth/generator.py` — market + sector factors, calibrated rho, planted
      IC and half-life, sign-flip interaction, release lags, delistings,
      restrictions, fat tails, ground truth, Parquet/JSON round-trip.
- [x] `cli.py` — `synth`, `stages`, `version`.
- [x] `configs/synth_lab.yaml` — pydantic-validated example.
- [x] `stages.py` — executable stage plan.

## Acceptance tests

| Requirement | Test | Status |
|---|---|---|
| Q2 EPS: 1.10 days 0–27, 1.35 from day 28 | `test_pit_store.py::test_q2_eps_is_visible_only_from_its_release_day` | passing |
| Deleting future rows never changes a value at t | `test_pit_store.py::test_deleting_future_rows_never_changes_a_value` | passing |
| Restatement vintages respected | `test_pit_store.py::test_restatement_vintage_wins_after_its_release` | passing |
| Look-ahead timestamps rejected | `test_pit_store.py::test_store_rejects_look_ahead_and_missing_release_timestamps` | passing |
| Snapshot hash reproducible | `test_pit_store.py::test_snapshot_hash_is_stable_under_row_order` | passing |
| Configured rho recovered | `test_synth.py::test_configured_within_date_correlation_is_recovered` | passing |
| Multi-sector rho matches analytic expectation | `test_synth.py::test_multi_sector_correlation_matches_analytic_expectation` | passing |
| Planted IC recovered | `test_synth.py::test_planted_ic_is_recovered` | passing |
| Release lags realistic | `test_synth.py::test_fundamentals_have_realistic_release_lags` | passing |
| Delistings leave prices/membership/universe | `test_synth.py::test_delisted_names_leave_prices_membership_and_universe` | passing |
| Fat tails present | `test_synth.py::test_fat_tails_are_present` | passing |
| Interaction flips conditional IC | `test_synth.py::test_interaction_flips_the_conditional_ic` | passing |
| n_eff reported and consistent | `test_synth.py::test_n_eff_is_reported_and_consistent` | passing |
| Missing release timestamps warn loudly | `test_adapters.py::test_missing_release_timestamps_warn_loudly_and_are_filled` | passing |
| Survivorship warning repeated in header | `test_adapters.py::test_survivorship_bias_is_warned_and_repeated_in_header` | passing |
| yfinance path parses and warns | `test_adapters.py::test_yfinance_adapter_parses_and_warns` | passing |
| Calendar and universe PIT filters | `test_calendar_universe.py` | passing |
| CLI synth writes a panel | `test_cli.py::test_synth_command_writes_a_panel` | passing |

## Verification commands

```bash
python -m pip install -e ".[dev]"
pytest -q
alpharig synth --out data/synth --config configs/synth_lab.yaml
alpharig stages --verbose
```

## Known limits carried forward

- yfinance cannot supply delisted names or release timestamps; both are warned
  about and must be repeated in every report.
- Adjusted prices are revised retroactively; the adapter says so.
- The synthetic panel is an oracle, not evidence about real markets.
- No costs, labels or validation exist yet: do not interpret Stage 1 output as
  an alpha claim.

## Exit criteria

- Full suite green (`pytest -q`).
- `alpharig synth` produces a panel whose realised rho and IC are within
  tolerance of the configured values and prints n_eff.
- Data-quality warnings are impossible to miss at load time and in report
  headers.
