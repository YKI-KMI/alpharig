"""Calendar and point-in-time universe tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpharig.data.calendar import TradingCalendar
from alpharig.data.universe import PointInTimeUniverse, UniverseRules


def test_calendar_navigation():
    calendar = TradingCalendar.from_business_days("2024-01-01", 10)
    dates = calendar.dates
    assert len(calendar) == 10
    assert calendar.next(dates[0]) == dates[1]
    assert calendar.prev(dates[1]) == dates[0]
    assert calendar.latest_on_or_before("2024-01-06") == dates[4]  # Saturday -> Friday
    assert calendar.earliest_on_or_after("2024-01-06") == dates[5]
    assert calendar.shift(dates[2], -2) == dates[0]
    assert list(calendar.between(dates[2], dates[5])) == list(dates[2:6])
    with pytest.raises(IndexError):
        calendar.prev(dates[0])
    with pytest.raises(IndexError):
        calendar.next(dates[-1])


def _universe_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2022-01-03", periods=30)
    tickers = ["A", "B", "C"]
    price_rows = []
    for date_index, date in enumerate(dates):
        for ticker in tickers:
            if ticker == "C" and date_index > 20:
                continue
            close = 10.0 + date_index * 0.1
            volume = 1_000.0 if ticker == "C" else 1_000_000.0
            price_rows.append(
                {"date": date, "ticker": ticker, "close": close, "volume": volume}
            )
    prices = pd.DataFrame(price_rows)
    membership = pd.DataFrame(
        {
            "ticker": tickers,
            "start_date": [dates[0], dates[0], dates[0]],
            "end_date": [pd.NaT, pd.NaT, dates[20]],
        }
    )
    restrictions = pd.DataFrame(
        {"ticker": ["A"], "start_date": [dates[5]], "end_date": [dates[8]]}
    )
    return prices, membership, restrictions, dates


def test_universe_filters_existence_liquidity_and_restrictions():
    prices, membership, restrictions, dates = _universe_fixture()
    universe = PointInTimeUniverse(
        membership,
        prices,
        restrictions,
        UniverseRules(min_adv=1e5, adv_window=5, min_observations=1),
    )
    assert set(universe.members(dates[10])) == {"A", "B"}
    assert "A" not in universe.members(dates[6])
    assert "C" not in universe.members(dates[25])
    assert "C" not in universe.members(dates[10])  # too illiquid

    with pytest.raises(ValueError):
        PointInTimeUniverse(membership, prices=None, rules=UniverseRules(min_adv=1.0))


def test_universe_has_no_lookahead():
    prices, membership, restrictions, dates = _universe_fixture()
    rules = UniverseRules(min_adv=1e5, adv_window=5, min_observations=1)
    full = PointInTimeUniverse(membership, prices, restrictions, rules)
    truncated = PointInTimeUniverse(
        membership, prices[prices["date"] <= dates[10]], restrictions, rules
    )
    assert set(full.members(dates[10])) == set(truncated.members(dates[10]))
    assert full.adv(dates[10]) is not None
    assert set(full.adv(dates[10]).index) == set(truncated.adv(dates[10]).index)


def test_universe_from_synthetic_panel_excludes_delisted_names(panel):
    universe = panel.universe(UniverseRules(min_adv=0.0))
    delisted = panel.truth.delistings
    assert not delisted.empty
    row = delisted.iloc[0]
    after = row["delist_date"] + pd.Timedelta(days=1)
    assert row["ticker"] not in universe.members(after)
    assert row["ticker"] in universe.members(row["delist_date"])
