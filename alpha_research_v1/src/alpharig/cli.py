"""alpharig command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from alpharig import RESEARCH_ONLY, __version__
from alpharig.data.quality import (
    DataQualityReport,
    check_fundamentals,
    check_membership,
    check_prices,
)
from alpharig.stages import format_plan
from alpharig.synth.generator import PanelSpec, SyntheticPanelGenerator

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Leakage-proof automated cross-sectional equity alpha research.",
)


@app.command()
def synth(
    out: Path = typer.Option(
        Path("data/synth"), "--out", "-o", help="Directory for the generated panel."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML PanelSpec; CLI flags override it."
    ),
    stocks: Optional[int] = typer.Option(None, "--stocks", help="Number of names."),
    periods: Optional[int] = typer.Option(None, "--periods", help="Number of periods."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed."),
    rho: Optional[float] = typer.Option(
        None, "--rho", help="Target mean within-date correlation."
    ),
    signal_ic: Optional[float] = typer.Option(
        None, "--signal-ic", help="Planted cross-sectional IC at lag 1."
    ),
    half_life: Optional[float] = typer.Option(
        None, "--half-life", help="Planted signal decay half-life in periods."
    ),
    interaction: bool = typer.Option(
        False, "--interaction", help="Flip the planted signal in high-vol regimes."
    ),
    freq: Optional[str] = typer.Option(None, "--freq", help="D (business days) or M (month ends)."),
) -> None:
    """Generate a synthetic panel with known ground truth."""
    payload: dict[str, object] = (
        PanelSpec.from_yaml(config).model_dump() if config is not None else {}
    )
    overrides: dict[str, object] = {
        "n_stocks": stocks,
        "n_periods": periods,
        "seed": seed,
        "rho": rho,
        "signal_ic": signal_ic,
        "signal_half_life": half_life,
        "freq": freq,
    }
    payload.update({key: value for key, value in overrides.items() if value is not None})
    if interaction:
        payload["interaction"] = True
    try:
        spec = PanelSpec(**payload)
    except ValidationError as exc:
        typer.echo(f"Invalid panel specification:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(RESEARCH_ONLY)
    typer.echo(f"alpharig {__version__} — synthetic lab")
    panel = SyntheticPanelGenerator(spec).generate()
    paths = panel.write(out)

    quality = DataQualityReport()
    quality.extend(check_prices(panel.prices, panel.membership))
    quality.extend(check_fundamentals(panel.fundamentals))
    quality.extend(check_membership(panel.membership))
    for line in quality.header_lines():
        typer.echo(line)
    quality.warn_all()

    truth = panel.truth
    n_dates = len(panel.calendar)
    typer.echo(
        f"panel: {spec.n_stocks} names x {n_dates} periods ({spec.freq}), "
        f"seed={spec.seed}, sectors={spec.n_sectors}"
    )
    typer.echo(
        f"within-date rho: target {truth.rho_target:.3f} | expected "
        f"{truth.rho_expected:.3f} | realized {truth.rho_realized:.3f}"
    )
    typer.echo(
        f"planted IC: target {truth.signal_ic_target:.3f} | realized "
        f"{truth.signal_ic_realized:.3f} (rank {truth.signal_rank_ic_realized:.3f}), "
        f"half-life {truth.signal_half_life:g}"
    )
    if spec.interaction:
        # A sign-flipping signal has a near-zero unconditional IC by design: the
        # two regimes cancel. Reporting only the pooled number would look like a
        # failed experiment when the structure is in fact exactly where it was
        # planted, so the conditional ICs are shown side by side.
        typer.echo(
            f"conditional IC (sign-flip interaction): calm {truth.ic_low_vol:+.3f} | "
            f"high-vol {truth.ic_high_vol:+.3f} | pooled {truth.signal_ic_realized:+.3f}"
        )
        typer.echo(
            "  the pooled IC is not the target for this panel: the interaction is "
            "only visible conditional on the regime, and any unconditional model "
            "should be expected to find nothing."
        )
    typer.echo(
        f"n_eff = {truth.n_eff:,.1f} effective observations "
        f"(rows = {n_dates * spec.n_stocks:,}, T={n_dates}, N={spec.n_stocks}, "
        f"rho={truth.rho_realized:.3f})"
    )
    typer.echo(
        f"delistings: {len(truth.delistings)} | restrictions: "
        f"{len(truth.restrictions)} | excess kurtosis: {truth.realized_kurtosis:.2f}"
    )
    typer.echo(f"wrote {len(paths)} files under {out}")
    if quality.critical():
        typer.echo(
            "WARNING: critical data-quality defects were found; treat every "
            "downstream result as biased.",
            err=True,
        )


@app.command()
def stages(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show goals and gates."),
) -> None:
    """Print the stage plan and its current status."""
    typer.echo(format_plan(verbose=verbose))


@app.command()
def version() -> None:
    """Print the alpharig version."""
    typer.echo(f"alpharig {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
