# alpharig — staged implementation plan

alpharig is built in **gated stages**. A stage starts only when the previous
stage's acceptance suite is green. Every stage adds its acceptance tests to
`tests/`, and every stage ends with a status report: what was built, which
tests passed, which checks could not run, and what is still biased.

The executable version of this plan is `src/alpharig/stages.py`; run
`alpharig stages --verbose` to print it.

## Gate protocol

1. **Build** the stage's modules behind the interfaces already fixed by
   earlier stages.
2. **Test** with the stage's acceptance tests plus the full existing suite:
   `pytest -q`.
3. **Report** the exact numbers the acceptance tests assert (rho, IC, n_eff,
   day arithmetic, capacity, …), not just "tests pass".
4. **Freeze** the interfaces other stages depend on. Changing a signal, target
   or position mapper later requires a version bump, never a silent edit.
5. **Never** relax a threshold, drop a test or loosen a gate to get green.
   A failing acceptance test is a finding, not an inconvenience.

## Status

| Stage | Slug | Status | Gate |
|---:|---|---|---|
| 0 | ground-rules | done | Shared n_eff / overlap utilities and conventions in place |
| 1 | data-layer | done | PIT join, deletion property, rho/IC recovery, adapter warnings |
| 2 | features | pending | EWMA identities, interaction arithmetic, cluster importance |
| 3 | labels | pending | Rank/binary examples, overlap n_eff, horizon recovery |
| 4 | validation-audit | pending | Exact purge/embargo arithmetic, canary and fit-before-split detection |
| 5 | models-ledger | pending | Multiplicity, best-of-N deflation, keep-linear vs promote-interaction |
| 6 | blending-regimes | pending | Blend IC/breadth, causal regimes, hard-switch refusal |
| 7 | portfolio-costs | pending | Sizing, band arithmetic, capacity break-even/peak |
| 8 | monitoring | pending | Decay half-life, alert crossing, refit cadence |
| 9 | cli-reports | pending | Full report with audit verdict on line one |
| 10 | autoresearch | pending | Budgeted loop that can say "no promotable alpha found" |
| 11 | release | pending | `pytest -q` green, limits documented, no advice |

---

## Stage 0 — Ground rules and scaffolding

**Goal.** Make the anti-self-deception rules structural, not advisory.

**Build.**
- `stats/effective.py`: `N_eff = T*N/(1+(N-1)rho)`, cap `T/rho`,
  overlap `n/h`, overconfidence `sqrt(h)`.
- Conventions: dates not rows; point-in-time joins; fold-only preprocessing;
  purged date-grouped validation; trial ledger; locked holdout; net-of-cost
  headline; research-only outputs.
- Package skeleton, YAML + pydantic configuration, fixed seeds, data snapshot
  hashes, code version.

**Gate.** Utilities tested and imported by every later stage. Reports must
carry n_eff; code review rejects any statistic without one.

---

## Stage 1 — Data layer and synthetic lab

**Goal.** A point-in-time data layer plus a test oracle with known answers.

**Build.**
- `data/adapters.py`: adapter interface (`load_prices`,
  `load_fundamentals`, `load_universe_membership(as_of)`), CSV/Parquet
  adapter, optional yfinance adapter.
- Loud `DataQualityWarning` when a source lacks release timestamps or delisted
  names; the report header repeats every warning.
- `data/store.py`: `as_of(field, t)` returns the latest vintage with
  `release_ts <= t`; rejects look-ahead timestamps.
- `data/calendar.py`: trading-date navigation.
- `data/universe.py`: existed, trailing liquidity, not restricted, priced.
- `synth/`: N names x T periods with market + sector factors and configurable
  within-date rho, planted IC and half-life, sign-flip interaction, release
  lags, delistings, restrictions, fat tails, ground truth.
- CLI `synth` and `stages`.

**Acceptance tests.**
- Q2 EPS: Q1 1.10 visible through day 27, Q2 1.35 from day 28.
- Property: deleting rows with timestamp > t never changes any value at t.
- Synthetic panel recovers configured rho and planted IC within tolerance.
- Adapter warns on missing release timestamps and missing delisted names.
- PIT store rejects `release_ts < period_end`.

**Gate.** All above green; CLI writes a panel and reports n_eff.

---

## Stage 2 — Signal library and feature engine

**Goal.** Versioned, fold-safe signals with honest interaction and importance.

**Build.**
- Decorator registry: signals receive only pre-lagged, universe-filtered
  inputs; declare lookback, release_lag, optional half_life, version.
  Code change => version bump; old versions stay reproducible.
- Versioned exception entries for irregular data.
- Transforms: per-date cross-sectional standardise, rank, winsorise bounds
  fitted in-fold, robust scale, EWMA `lambda = 1 - 0.5**(1/h)`, beta/size
  neutralisation, lag/rolling features.
- Interaction registry: only declared `(feature, conditioner)` pairs with a
  written rationale; refuse all-pairs crossing with a warning stating the
  count it would create.
- Redundancy pruning by correlation; clustered importance via
  correlation-distance hierarchical clustering and out-of-sample
  cluster-permutation MDA. Never drop a feature on solo importance.

**Acceptance tests.**
- EWMA h=3 → lambda 0.206; h=20 → 0.034; weight 0.1 → half-life 6.6.
- Interaction with beta0=0, beta1=0.9, beta3=-1.1 → effect 0.9 at z=0,
  -0.2 at z=1, sign flip at z≈0.82.
- Four correlated momentum features + two independent: solo permutation
  small, cluster permutation largest.
- 40 features, no declared pairs → 0 interactions and a warning that all-pairs
  would create 780.

**Gate.** All above green; every transformer refuses to fit outside a train
fold.

---

## Stage 3 — Labels and targets

**Goal.** Targets with declared compatibility and honest overlap accounting.

**Build.**
- Target registry: raw, excess, residual (PIT market/sector/size/value risk
  model), vol_scaled with floored lagged EWMA, rank (uniform/Gaussian),
  binary_median, ordinal_quintile. Each declares compatible position mappers;
  direction/binary + magnitude sizing raises.
- Winsorising and Huber-loss options.
- Per-signal horizon scan over 1/2/5/10/20/60 periods; store peak and
  recommended horizon with the signal.
- `label_start` / `label_end` on every label; `n_eff = n/h`, overconfidence
  `sqrt(h)`, non-overlapping sampling, uniqueness weights.

**Acceptance tests.**
- Adding a constant to one date's returns leaves that date's rank targets
  unchanged.
- Returns +8,+1,-2,+3,-6 → ranks A=1,D=2,B=3,C=4,E=5; binary-above-median
  A and D only (median +1).
- Squared-return share of +8 is 64/114 = 56%.
- 1,250 rows at h=20 → n_eff 62.5, factor 4.47; 1,500 rows, h=15, naive
  t=3.0 → ≈0.77.
- Horizon scan with planted peak at 7 days recommends 7 ± tolerance.
- Vol scaling uses only data up to t.

**Gate.** All above green; incompatible target/mapper pairs raise.

---

## Stage 4 — Validation harness and leakage audit

**Goal.** The stage that makes fooling ourselves difficult. Most important.

**Build.**
- `PurgedWalkForward`: expanding/rolling train, whole-date test folds,
  per-row `label_end`, `feature_lookback`, `embargo`; purge rows whose label
  window or lookback touches the test fold; block the embargo buffer.
- Standing diagnostics: random-row CV vs date-grouped CV gap; shuffled-label
  null (≈0); embargo 0 vs H; naive vs n_eff-corrected t.
- `alpharig audit`: canary next-period-return feature; fit-inside-fold
  assertions for every transformer; stage-graph ordering; universe PIT check;
  "too good" plausibility bounds.
- Cross-sectional null SE `1/sqrt(T)`.

**Acceptance tests.**
- 10-day label, test days 200–230: day 195 purged; embargo blocks 231–240;
  first allowed train row 241.
- 500 days, H=10, embargo 10, 5 folds → 100 days lost (20%).
- Feature lookback 60 at day 205 reaches 146–205, 6 days inside the fold;
  purged when `feature_lookback=60`.
- N=2000, T=60, rho=0.3 → n_eff 200; null IC SE 0.129; ratio ≈44.7.
- Best of 150 noise features at T=60 → max null IC ≈0.41.
- Canary detected; scaler-before-split inflated and flagged.

**Gate.** All above green; audit runs on any saved result.

---

## Stage 5 — Models, baseline gate, trial ledger

**Goal.** ML only when it beats linear by more than noise.

**Build.**
- Ridge baseline on identical features/splits. LightGBM (shallow, strongly
  regularised), optional small MLP.
- Metrics: rank IC, IC IR, top-minus-bottom spread, turnover, net-of-cost
  Sharpe, always with n_eff-corrected significance.
- Promotion gate: beat ridge by more than the IC-difference standard error
  computed on effective samples **and** win a majority of sub-periods;
  otherwise "keep linear".
- Pooled default with group features (sector, size, liquidity); grouped and
  per-asset modes print per-model n_eff and warn when small.
- Append-only trial ledger (JSONL/DuckDB): config hash, split ids, scores.
- Deflated reporting: `sigma*sqrt(2 ln N)`. Fixed search budget declared
  before the search; stop when exhausted.
- Holdout locked and excluded from loaders unless `--open-holdout` is passed
  once; record hash and timestamp.

**Acceptance tests.**
- 20 null configs at 5% → family-wise false positive ≈0.64.
- sigma=0.15: N=10 → +0.32, N=200 → +0.49.
- Linear-only synthetic → keep linear; sign-flip interaction → promote GBM or
  interaction-augmented linear.
- Holdout opens once; second attempt raises.

**Gate.** All above green; ledger rejects edits.

---

## Stage 6 — Blending, orthogonalisation, regimes

**Goal.** Combine signals without double-counting; label regimes causally.

**Build.**
- Blenders: equal weight (mandatory), ridge, GBM.
- Pre-blend per-date sequential/symmetric orthogonalisation; hierarchical
  blend by signal family. Effective breadth
  `IC_blend = IC*sqrt(n/(1+(n-1)rho))`.
- Regimes: threshold, k-means, HMM; filtered probabilities only, hysteresis.
- Report flip rate, expected duration `1/(1-p)`, implied switching cost;
  count independent episodes and refuse hard switching below the configured
  minimum (default 8), falling back to soft conditioning.

**Acceptance tests.**
- 8 signals, IC 0.025, rho 0.05 → blend IC ≈0.061, ≈5.93 effective signals;
  rho 0.2 → ≈0.046.
- Three redundant momentum clones → orthogonalised blend gives the family no
  more than its independent share vs naive ridge.
- Median-hovering series → threshold flips ≈half of days; HMM stay 0.98 →
  ≈2% flips, 50-day duration.
- 3 crisis episodes with minimum 8 → hard-switch refusal.
- Centred-window regime labels rejected by audit.

**Gate.** All above green.

---

## Stage 7 — Positions, costs, capacity

**Goal.** Net-of-cost P&L only.

**Build.**
- Mappers declared before the backtest and immutable after: rank, z-score,
  capped z (clip c), optional vol scaling; compatibility with target type.
- No-trade band: trade only if `|p_target - p_held| > kappa*c`; default trade
  to band edge, option to trade fully. Per-instrument band from the cost model.
- Turnover penalty in training, tuned on net-of-cost walk-forward.
- Cost model: spread, commission, square-root impact.
- Capacity: `alpha - c*Q**gamma`; break-even `(alpha/c)**(1/gamma)`;
  profit-max `(alpha/(c*(1+gamma)))**(1/gamma)`; 7.5% ADV position limit;
  report by liquidity bucket.

**Acceptance tests.**
- Scores 2.1, 0.3, -0.2 → raw top/middle 7; capped at 1.5 → 5.
- score 1.5, vol 0.30, capital 1 crore → 10,00,000.
- n=10 max z ≈2.85 and the cap handles it.
- Path 100,103,98,101,96,99, band 4, start 100 → zero trades; without band
  five; steady drift eventually trades.
- Holding 0.5*a*g², a=0.02, trade cost 0.40/share, gap 60 → band-edge size 40.
- alpha 0.5%, c=0.05, gamma=1.5 → break-even ≈4.64, peak ≈2.52.

**Gate.** All above green.

---

## Stage 8 — Live monitoring, decay, refit

**Goal.** Detect decay before P&L does, and triage before retraining.

**Build.**
- Decay-curve report: IC at 1, 2, 3… months after each training end; average,
  exponential fit, half-life, graceful vs abrupt collapse.
- Refit scheduler with configured cadence; cadence backtest net of cost;
  prefer ensembles from recent refits.
- Live monitor: rolling 6-month IC, `lambda = ln(IC1/IC2)/dt`,
  `half-life = ln2/lambda`, projected date to cross cost-covering IC (0.02),
  turnover, hit rate, factor exposures. Alerts in config; triage checklist
  (data fault, disliked regime, crowding) before retrain/resize/retire.
- P&L confirms, never detects. Warn about single-month noise and report months
  needed to distinguish backtest mean from a lower live mean.

**Acceptance tests.**
- IC 0.055 → 0.019 over 15 months → lambda ≈0.071, half-life ≈9.8 months.
- IC 0.06, half-life 12, alert 0.02 → crossing ≈19 months.
- Planted half-life recovered within tolerance.
- Single-month IC warning present.

**Gate.** All above green.

---

## Stage 9 — CLI and reports

**Goal.** One command per question; the audit verdict on line one.

**Commands.** `synth`, `scan-horizon`, `run <config.yaml>`, `audit <run_id>`,
`promote <run_id>`, `capacity <run_id>`, `monitor <model_id>`,
`holdout --open`, `stages`, `autoresearch`.

**Report contents.** Data-quality warnings; n_eff; row-vs-date CV gap; null
score; baseline vs candidate with noise band; trial count and deflated score;
net-of-cost results; turnover; capacity; decay curve; audit verdict.

**Gate.** Every command runs end-to-end; first line is PASS / SUSPECT / FAIL.

---

## Stage 10 — Automation loop

**Goal.** Run the whole process under a budget without moving goalposts.

**Build.** `alpharig autoresearch`: horizon scan per signal; labels at each
signal's horizon; evaluate each signal and a ridge blend under the purged
harness; try ML only where the baseline leaves structure; log every trial;
apply the gate; audit and capacity on promoted candidates; emit a ranked
report. Stop at the declared trial budget. Never loosen a gate. Emit
"no promotable alpha found" when true.

**Gate.** Budget exhaustion stops the loop; gate cannot be bypassed; report
ranks only audited candidates.

---

## Stage 11 — Release and definition of done

**Goal.** The system is trustworthy enough to say "no".

**Checks.**
- `pytest -q` green, including every acceptance test in this document.
- Synthetic matrix: finds planted signal; rejects pure noise; catches canary
  and fit-before-split; keeps linear on linear-only data; promotes ML on
  interaction data.
- README documents installation, every command, and free-data limits
  (survivorship, restatements, missing release timestamps, revised adjusted
  prices, missing delisting returns).
- Nothing in the codebase gives investment advice; every report carries the
  research-only disclaimer.

**Gate.** All checks above.
