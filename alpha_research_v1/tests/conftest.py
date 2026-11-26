"""Shared fixtures for the alpharig test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from alpharig.synth.generator import PanelSpec, SyntheticPanelGenerator


@pytest.fixture(scope="session")
def panel():
    """A modest panel with delistings, restrictions, sectors and fat tails."""
    spec = PanelSpec(
        n_stocks=80,
        n_periods=260,
        n_sectors=8,
        rho=0.30,
        signal_ic=0.05,
        signal_half_life=15.0,
        signal_max_lag=40,
        seed=11,
        fat_tail_df=5.0,
        delist_fraction=0.08,
        restricted_fraction=0.08,
        beta_dispersion=0.0,
    )
    return SyntheticPanelGenerator(spec).generate()


@pytest.fixture(scope="session")
def ic_panel():
    """A single-sector panel calibrated for exact rho and IC recovery."""
    spec = PanelSpec(
        n_stocks=250,
        n_periods=300,
        n_sectors=1,
        sector_vol_share=0.0,
        rho=0.30,
        signal_ic=0.05,
        signal_half_life=20.0,
        signal_max_lag=40,
        seed=5,
        fat_tail_df=100.0,
        regime_vol_mult=1.0,
        delist_fraction=0.0,
        restricted_fraction=0.0,
        beta_dispersion=0.0,
    )
    return SyntheticPanelGenerator(spec).generate()


@pytest.fixture(scope="session")
def wide_returns(ic_panel):
    """Wide return matrix of the IC-recovery panel."""
    return ic_panel.returns.pivot(index="date", columns="ticker", values="ret")


@pytest.fixture(scope="session")
def feature_panel():
    """A small panel with enough history for the short-horizon signals."""
    spec = PanelSpec(
        n_stocks=40,
        n_periods=140,
        n_sectors=4,
        sector_vol_share=0.20,
        rho=0.20,
        signal_ic=0.05,
        signal_half_life=10.0,
        signal_max_lag=20,
        seed=17,
        fat_tail_df=8.0,
        delist_fraction=0.05,
        restricted_fraction=0.05,
        beta_dispersion=0.0,
    )
    return SyntheticPanelGenerator(spec).generate()
