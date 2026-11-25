"""Synthetic oracle tests: rho, IC, point-in-time joins, delistings, tails."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpharig.data.universe import UniverseRules
from alpharig.stats.effective import effective_sample_size
from alpharig.stats.panel import mean_pairwise_correlation
from alpharig.synth.generator import PanelSpec, SyntheticPanel, SyntheticPanelGenerator


def _wide(panel: SyntheticPanel) -> pd.DataFrame:
    return panel.returns.pivot(index="date", columns="ticker", values="ret")


def test_configured_within_date_correlation_is_recovered(ic_panel, wide_returns):
    realized = mean_pairwise_correlation(wide_returns)
    assert abs(realized - ic_panel.truth.rho_target) < 0.02
    assert abs(ic_panel.truth.rho_realized - ic_panel.truth.rho_target) < 0.02


def test_multi_sector_correlation_matches_analytic_expectation(panel):
    realized = mean_pairwise_correlation(_wide(panel))
    assert abs(realized - panel.truth.rho_expected) < 0.02
    assert abs(panel.truth.rho_expected - panel.truth.rho_target) < 0.005


def test_planted_ic_is_recovered(ic_panel):
    truth = ic_panel.truth
    assert abs(truth.signal_ic_realized - truth.signal_ic_target) < 0.012
    assert abs(truth.signal_rank_ic_realized - truth.signal_ic_target) < 0.015


def test_fundamentals_have_realistic_release_lags(panel):
    lags = (panel.fundamentals["release_ts"] - panel.fundamentals["period_end"]).dt.days
    assert panel.fundamentals["release_ts"].notna().all()
    assert lags.min() >= 30
    assert lags.max() <= 90


def test_point_in_time_join_matches_latest_release(panel):
    store = panel.store()
    fundamentals = panel.fundamentals
    for date in panel.calendar.dates[::23]:
        for field in ("eps", "revenue"):
            snapshot = store.as_of(field, date)
            for _, row in snapshot.head(2).iterrows():
                candidates = fundamentals[
                    (fundamentals["ticker"] == row["ticker"])
                    & (fundamentals["field"] == field)
                    & (fundamentals["release_ts"] <= date)
                ]
                expected = candidates.sort_values(["release_ts", "vintage"]).iloc[-1]["value"]
                assert row["value"] == pytest.approx(expected)
                assert row["release_ts"] <= date


def test_delisted_names_leave_prices_membership_and_universe(panel):
    delisted = panel.truth.delistings
    assert not delisted.empty
    last_price = panel.prices.groupby("ticker")["date"].max()
    membership = panel.membership.set_index("ticker")
    universe = panel.universe(UniverseRules(min_adv=0.0))
    for _, row in delisted.iterrows():
        ticker = row["ticker"]
        assert last_price[ticker] == row["delist_date"]
        assert membership.loc[ticker, "end_date"] == row["delist_date"]
        assert ticker in universe.members(row["delist_date"])
        assert ticker not in universe.members(row["delist_date"] + pd.Timedelta(days=1))


def test_fat_tails_are_present(panel):
    assert panel.returns["ret"].kurt() > 1.5


def test_interaction_flips_the_conditional_ic():
    spec = PanelSpec(
        n_stocks=200,
        n_periods=400,
        n_sectors=1,
        sector_vol_share=0.0,
        rho=0.20,
        signal_ic=0.06,
        signal_half_life=20.0,
        signal_max_lag=40,
        interaction=True,
        regime_stay=0.95,
        regime_vol_mult=2.0,
        seed=21,
        fat_tail_df=50.0,
        delist_fraction=0.0,
    )
    truth = SyntheticPanelGenerator(spec).generate().truth
    assert truth.ic_low_vol > 0.02
    assert truth.ic_high_vol < -0.02


def test_n_eff_is_reported_and_consistent(panel):
    truth = panel.truth
    expected = effective_sample_size(
        len(panel.calendar), panel.truth.spec.n_stocks, max(truth.rho_realized, 0.0)
    )
    assert truth.n_eff == pytest.approx(expected)
    assert 0 < truth.n_eff < len(panel.calendar) * panel.truth.spec.n_stocks


def test_panel_round_trips_through_disk(tmp_path, panel):
    panel.write(tmp_path / "lab")
    loaded = SyntheticPanel.read(tmp_path / "lab")
    assert len(loaded.prices) == len(panel.prices)
    assert len(loaded.fundamentals) == len(panel.fundamentals)
    assert loaded.truth.signal_ic_target == pytest.approx(panel.truth.signal_ic_target)
    assert loaded.truth.n_eff == pytest.approx(panel.truth.n_eff)
    pd.testing.assert_frame_equal(loaded.fundamentals, panel.fundamentals)
    assert loaded.store().snapshot_hash() == panel.store().snapshot_hash()


def test_spec_rejects_unknown_fields():
    with pytest.raises(ValueError):
        PanelSpec(n_stocks=10, not_a_field=1)
