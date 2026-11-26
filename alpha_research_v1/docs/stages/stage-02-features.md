# Stage 2 — Signal library and feature engine

**Status:** done — all acceptance tests green.
**Depends on:** Stage 1 (point-in-time store, universe, synthetic oracle).
**Unblocks:** Stage 3 (labels), Stage 4 (purged validation needs the declared
lookbacks), Stage 5 (models need a fold-safe feature matrix).

## Build checklist

- [x] `features/fold.py` — `FoldGuard` with `assert_train_frame`,
      `assert_test_frame`, `train_frame`, `test_frame`; `FoldViolationError`.
- [x] `features/inputs.py` — `SignalInputs` (truncated at `as_of`, universe
      masked), `build_signal_inputs`, `trailing_adv`, `safe_log`.
- [x] `features/transforms.py` — cross-sectional z-score and rank, in-fold
      winsoriser, robust scaler, neutraliser, lag, rolling statistic, EWMA,
      and `FeaturePipeline`. EWMA helpers: `ewma_lambda`, `ewma_decay`,
      `half_life_from_decay`, `half_life_from_lambda`.
- [x] `features/registry.py` — `SignalSpec`, `SignalRegistry`, `signal`
      decorator; source-hash version enforcement; look-ahead rejection.
- [x] `features/library.py` — eight declared starter signals
      (momentum, reversal, low volatility, earnings yield, book-to-price,
      revenue growth, log dollar volume, Amihud illiquidity).
- [x] `features/interactions.py` — `InteractionRegistry`, rationale required,
      `InteractionBuild`, `all_pairs_count`, `marginal_effect`,
      `zero_crossing_z`, `effect_table`.
- [x] `features/redundancy.py` — correlation distance, hierarchical clustering,
      medoid pruning, `pruning_report`.
- [x] `features/importance.py` — solo permutation importance and
      cluster permutation importance (MDA), with solo explicitly marked
      report-only.

## Acceptance tests

| Requirement | Test | Status |
|---|---|---|
| EWMA h=3 → λ 0.2063 | `test_features.py::test_ewma_half_life_identities` | passing |
| EWMA h=20 → λ 0.0341 | `test_features.py::test_ewma_half_life_identities` | passing |
| Smoothing weight 0.1 → half-life 6.579 | `test_features.py::test_ewma_half_life_identities` | passing |
| EWMA reproduces its own recursion, and is causal | `test_features.py::test_ewma_recursion_matches_its_own_lambda`, `::test_ewma_is_causal` | passing |
| Interaction effect 0.9 at z=0, −0.2 at z=1 | `test_interactions.py::test_interaction_effect_arithmetic` | passing |
| Sign flip at z ≈ 0.818 | `test_interactions.py::test_interaction_effect_arithmetic` | passing |
| 4 correlated + 2 independent: solo small, cluster largest | `test_selection.py::test_cluster_importance_dominates_solo_importance` | passing |
| Redundancy collapses clones to one representative | `test_selection.py::test_pruning_keeps_one_representative_per_cluster` | passing |
| 40 features, no declared pairs → 0 interactions, warning says 780 | `test_interactions.py::test_forty_features_without_declarations_build_nothing_and_warn_780` | passing |
| Every transformer refuses to fit outside a train fold | `test_features.py::test_every_transformer_refuses_to_fit_on_the_full_panel`, `::test_every_transformer_refuses_a_missing_fold` | passing |
| In-fold bounds only (outlier outside the fold is clipped) | `test_features.py::test_winsorizer_bounds_come_from_training_rows_only` | passing |
| Signals cannot see the future | `test_signals.py::test_signal_computation_is_point_in_time`, `::test_a_signal_that_invents_a_future_date_is_rejected` | passing |
| Code change without a version bump raises | `test_signals.py::test_code_change_without_a_version_bump_is_rejected` | passing |

## Numbers the acceptance tests assert

| Quantity | Value |
|---|---|
| `ewma_lambda(3)` | 0.206299 |
| `ewma_lambda(20)` | 0.034064 |
| `half_life_from_lambda(0.1)` | 6.5788 |
| Interaction effect at `z = 0`, `z = 1` | 0.900, −0.200 |
| Sign-flip conditioner value | 0.8182 |
| All-pairs columns for 40 features | 780 = C(40, 2) |
| Momentum cluster importance vs best solo clone | > 5× |

## Design decisions worth keeping

- **The fold is an argument, not a convention.** `fit` takes a `FoldGuard`;
  passing `None` or a panel that reaches into the test window raises. There is
  no "warn and subset" path.
- **An interaction is a panel, not a cell.** `InteractionBuild.panels` maps
  each declared pair to its own dates-by-tickers frame; `to_long()` stacks them
  for storage.
- **Pruning never uses importance.** Representatives are cluster medoids.
  Importance is measured afterwards, per cluster, and solo values are labelled
  report-only so they cannot be quoted as a selection criterion.
- **Signal versions are content-addressed.** Re-registering the same
  `(name, version)` with different source raises `SignalVersionError`.

## Known limits carried forward

- The eight library signals are illustrative constructions, not validated
  alphas. Nothing in Stage 2 says anything about real expected returns.
- `Neutralizer` fits one pooled coefficient vector per fold; a regime-dependent
  exposure would need either a longer window or an explicit regime split.
- `SignalInputs` materialises fundamentals date by date through the point-in-time
  store. That is correct but quadratic in the number of dates; a cached
  vectorised join is a later optimisation, not a correctness fix.
- No labels, no validation harness and no costs exist yet. Do not read a Stage 2
  score as performance.

## Exit criteria

- Full suite green (`pytest -q`).
- Every transformer refuses to fit outside a training fold.
- The interaction registry reports 780 for 40 features and builds nothing
  undeclared.
