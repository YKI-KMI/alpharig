"""Adapter tests: round-trips, warnings and the optional yfinance path."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from alpharig.data.adapters import FileAdapter, FrameAdapter, YFinanceAdapter
from alpharig.data.quality import DataQualityWarning
from alpharig.data.store import PointInTimeStore


def _quality_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [item for item in caught if issubclass(item.category, DataQualityWarning)]


def test_file_adapter_round_trip_is_clean(tmp_path, panel):
    paths = panel.write(tmp_path / "lab")
    adapter = FileAdapter(
        prices_path=paths["prices"],
        fundamentals_path=paths["fundamentals"],
        membership_path=paths["membership"],
        restrictions_path=paths["restrictions"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prices = adapter.load_prices()
        fundamentals = adapter.load_fundamentals()
        membership = adapter.load_universe_membership()
        restrictions = adapter.load_restrictions()
    assert not _quality_warnings(caught)
    assert len(prices) == len(panel.prices)
    assert len(fundamentals) == len(panel.fundamentals)
    assert len(membership) == len(panel.membership)
    assert len(restrictions) == len(panel.restrictions)
    assert PointInTimeStore(fundamentals).snapshot_hash() == PointInTimeStore(
        panel.fundamentals
    ).snapshot_hash()


def test_missing_release_timestamps_warn_loudly_and_are_filled(panel):
    fundamentals = panel.fundamentals.drop(columns=["release_ts"])
    adapter = FrameAdapter(
        prices=panel.prices,
        fundamentals=fundamentals,
        membership=panel.membership,
        assumed_release_lag_days=90,
    )
    with pytest.warns(DataQualityWarning, match="restatement"):
        loaded = adapter.load_fundamentals()
    assert (loaded["release_ts"] - loaded["period_end"]).dt.days.min() >= 90
    assert loaded["vintage"].notna().all()
    assert adapter.report.critical()
    assert any("restatement" in line.lower() for line in adapter.report.header_lines())


def test_strict_mode_refuses_missing_release_timestamps(panel):
    fundamentals = panel.fundamentals.drop(columns=["release_ts"])
    adapter = FrameAdapter(
        prices=panel.prices,
        fundamentals=fundamentals,
        membership=panel.membership,
        strict=True,
    )
    with pytest.raises(ValueError, match="strict=True"):
        adapter.load_fundamentals()


def test_survivorship_bias_is_warned_and_repeated_in_header(panel):
    last_dates = panel.prices.groupby("ticker")["date"].max()
    final_date = panel.prices["date"].max()
    survivors = panel.prices[panel.prices["ticker"].isin(last_dates[last_dates == final_date].index)]
    adapter = FrameAdapter(prices=survivors)
    with pytest.warns(DataQualityWarning, match="survivorship"):
        adapter.load_prices()
    header = "\n".join(adapter.report.header_lines()).lower()
    assert "survivorship" in header
    assert adapter.report.critical()


def test_membership_without_end_dates_is_survivorship_biased(panel):
    membership = panel.membership.assign(end_date=pd.NaT)
    adapter = FrameAdapter(
        prices=panel.prices, fundamentals=panel.fundamentals, membership=membership
    )
    with pytest.warns(DataQualityWarning, match="survivorship"):
        adapter.load_prices()


def test_derived_membership_uses_price_history(panel):
    adapter = FrameAdapter(prices=panel.prices)
    membership = adapter.load_universe_membership()
    assert len(membership) == panel.prices["ticker"].nunique()
    delisted = panel.truth.delistings
    if not delisted.empty:
        row = delisted.iloc[0]
        recorded = membership.loc[membership["ticker"] == row["ticker"], "end_date"].iloc[0]
        assert recorded == row["delist_date"]


class _FakeYFinance:
    @staticmethod
    def download(
        tickers,
        start=None,
        end=None,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    ):
        dates = pd.bdate_range("2021-01-04", periods=4)
        columns = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]])
        values = np.arange(16, dtype=float).reshape(4, 4) + 10.0
        return pd.DataFrame(values, index=dates, columns=columns)


def test_yfinance_adapter_parses_and_warns(monkeypatch):
    adapter = YFinanceAdapter(["AAA", "BBB"], "2021-01-01", "2021-02-01")
    monkeypatch.setattr(adapter, "_import", lambda: _FakeYFinance)
    with pytest.warns(DataQualityWarning, match="survivorship"):
        prices = adapter.load_prices()
    assert set(prices["ticker"]) == {"AAA", "BBB"}
    assert len(prices) == 8
    assert "volume" in prices.columns
    assert any("revised" in line.lower() for line in adapter.report.header_lines())
