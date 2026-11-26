"""Stage registry: the implementation plan, executable.

Each stage maps to one phase of the project brief. A stage is only marked
``done`` when its acceptance tests pass; the CLI can print the plan with
``alpharig stages``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StageStatus = Literal["pending", "in-progress", "done"]


@dataclass(frozen=True)
class Stage:
    """One gated stage of the build."""

    number: int
    slug: str
    title: str
    goal: str
    gate: str
    status: StageStatus


STAGES: tuple[Stage, ...] = (
    Stage(
        number=0,
        slug="ground-rules",
        title="Ground rules and scaffolding",
        goal="Encode the non-negotiables: effective sample size, point-in-time data, fold-only preprocessing, purged validation, trial ledger, locked holdout, auditability, reproducibility, net-of-cost P&L.",
        gate="Shared utilities exist and every later stage is written against them.",
        status="done",
    ),
    Stage(
        number=1,
        slug="data-layer",
        title="Data layer and synthetic lab",
        goal="Point-in-time store, calendar, universe, CSV/Parquet and optional yfinance adapters, loud data-quality warnings, and a synthetic oracle with known rho, IC, decay, interaction, lags, delistings and fat tails.",
        gate="Q2 EPS join, deletion property, correlation and planted-IC recovery, adapter warnings, CLI synth smoke test.",
        status="done",
    ),
    Stage(
        number=2,
        slug="features",
        title="Signal library and feature engine",
        goal="Versioned signal registry, fold-safe transforms, declared interactions only, redundancy pruning and clustered permutation importance.",
        gate="EWMA half-life identities, interaction effect arithmetic, cluster-importance test, 40-feature interaction warning.",
        status="done",
    ),
    Stage(
        number=3,
        slug="labels",
        title="Labels and targets",
        goal="Target registry with compatibility checks, horizon scan per signal, overlap accounting and non-overlapping sampling.",
        gate="Rank/binary label examples, squared-return share, n_eff and t-stat correction, horizon recovery, vol-scale causality.",
        status="pending",
    ),
    Stage(
        number=4,
        slug="validation-audit",
        title="Validation harness and leakage audit",
        goal="Purged, embargoed, date-grouped walk-forward; standing diagnostics; canary and fit-inside-fold audit.",
        gate="Exact purge/embargo day arithmetic, 20% loss test, feature-lookback purge, null IC SE, canary detection, fit-before-split flag.",
        status="pending",
    ),
    Stage(
        number=5,
        slug="models-ledger",
        title="Models, baseline gate, trial ledger",
        goal="Ridge baseline, regularised GBM, promotion gate with noise band, append-only trial ledger, deflated scores, locked holdout.",
        gate="Null multiplicity, best-of-N inflation, keep-linear vs promote-interaction, single holdout opening.",
        status="pending",
    ),
    Stage(
        number=6,
        slug="blending-regimes",
        title="Blending, orthogonalisation, regimes",
        goal="Equal/ridge/GBM blenders, causal orthogonalisation, effective breadth, causal regime labelers with hysteresis and hard-switch refusal.",
        gate="Blend IC/breadth arithmetic, redundant-clone test, flip-rate and duration tests, centred-window rejection.",
        status="pending",
    ),
    Stage(
        number=7,
        slug="portfolio-costs",
        title="Positions, costs, capacity",
        goal="Immutable score-to-position mappers, no-trade bands, turnover penalty, cost model, capacity estimator.",
        gate="Sizing ratios, vol-scaled weight, z-cap, band trade arithmetic, capacity break-even and peak.",
        status="pending",
    ),
    Stage(
        number=8,
        slug="monitoring",
        title="Live monitoring, decay, refit",
        goal="Decay curves, refit cadence backtests, live IC monitor, triage checklist before retrain/resize/retire.",
        gate="Decay lambda/half-life, alert crossing date, planted half-life recovery, noisy-IC sample-size warning.",
        status="pending",
    ),
    Stage(
        number=9,
        slug="cli-reports",
        title="CLI and reports",
        goal="End-to-end commands and reports whose first line is the audit verdict and which repeat every data-quality warning.",
        gate="Every command runs; report contains n_eff, CV gap, null score, noise band, trial count, net P&L, capacity, decay and verdict.",
        status="pending",
    ),
    Stage(
        number=10,
        slug="autoresearch",
        title="Automation loop",
        goal="Budgeted autoresearch: horizon scan, labels, baseline, ML only where structure remains, ledger, gate, audit, capacity, ranked report.",
        gate="Stops at budget; never loosens a gate; can say 'no promotable alpha found'.",
        status="pending",
    ),
    Stage(
        number=11,
        slug="release",
        title="Definition of done",
        goal="Full suite green in CI, README complete, data-source limits documented, research-only disclaimer everywhere.",
        gate="pytest -q passes; synthetic acceptance matrix passes; no investment advice in outputs.",
        status="pending",
    ),
)


def get_stage(number: int) -> Stage:
    """Return the stage with ``number``.

    Raises
    ------
    KeyError
        If no such stage exists.
    """
    for stage in STAGES:
        if stage.number == number:
            return stage
    raise KeyError(f"unknown stage {number}")


def format_plan(verbose: bool = False) -> str:
    """Render the stage plan as a plain-text table."""
    lines = [
        "alpharig implementation stages",
        "===============================",
        f"{'#':>2}  {'status':<11} {'slug':<22} title",
    ]
    for stage in STAGES:
        lines.append(
            f"{stage.number:>2}  {stage.status:<11} {stage.slug:<22} {stage.title}"
        )
        if verbose:
            lines.append(f"    goal: {stage.goal}")
            lines.append(f"    gate: {stage.gate}")
    return "\n".join(lines)
