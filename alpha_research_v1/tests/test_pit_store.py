"""Point-in-time store tests, including the Q2 EPS acceptance example."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from alpharig.data.store import PointInTimeStore


def _row(
    ticker: str,
    field: str,
    period_end: pd.Timestamp,
    release_ts: pd.Timestamp,
    value: float,
    vintage: str = "v1",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "field": field,
        "period_end": period_end,
        "release_ts": release_ts,
        "value": value,
        "vintage": vintage,
    }


def test_q2_eps_is_visible_only_from_its_release_day():
    """Q1 1.10 stays visible until Q2 1.35 is released 28 days after period end."""
    dates = pd.bdate_range("2021-01-04", periods=90)
    day_zero = dates[30]
    q1_end = day_zero - pd.Timedelta(days=63)
    q1_release = day_zero - pd.Timedelta(days=35)
    q2_release = day_zero + pd.Timedelta(days=28)
    assert q1_release in dates and q2_release in dates

    store = PointInTimeStore(
        pd.DataFrame(
            [
                _row("AAA", "eps", q1_end, q1_release, 1.10),
                _row("AAA", "eps", day_zero, q2_release, 1.35),
            ]
        )
    )
    features = store.feature_frame("eps", dates, tickers=["AAA"])

    assert features.loc[day_zero, "AAA"] == pytest.approx(1.10)
    assert features.loc[dates[dates < q2_release][-1], "AAA"] == pytest.approx(1.10)
    assert features.loc[q2_release, "AAA"] == pytest.approx(1.35)
    assert features.loc[dates[dates < q1_release]].isna().all().all()
    assert features.loc[q1_release, "AAA"] == pytest.approx(1.10)

    snapshot = store.as_of("eps", q2_release)
    assert snapshot.loc[snapshot["ticker"] == "AAA", "period_end"].iloc[0] == day_zero
    assert snapshot.loc[snapshot["ticker"] == "AAA", "release_ts"].iloc[0] == q2_release


def test_restatement_vintage_wins_after_its_release():
    dates = pd.bdate_range("2021-01-04", periods=40)
    day_zero = dates[20]
    release = day_zero + pd.Timedelta(days=28)
    restated = release + pd.Timedelta(days=10)
    store = PointInTimeStore(
        pd.DataFrame(
            [
                _row("AAA", "eps", day_zero, release, 1.35, "2021-05-01"),
                _row("AAA", "eps", day_zero, restated, 1.20, "2021-05-11"),
            ]
        )
    )
    assert store.as_of("eps", restated - pd.Timedelta(days=1))["value"].iloc[0] == pytest.approx(1.35)
    assert store.as_of("eps", restated)["value"].iloc[0] == pytest.approx(1.20)


def test_same_day_release_is_visible_and_same_day_vintage_breaks_ties():
    day = pd.Timestamp("2022-03-01")
    store = PointInTimeStore(
        pd.DataFrame(
            [
                _row("AAA", "eps", day - pd.Timedelta(days=90), day, 1.0, "v1"),
                _row("AAA", "eps", day - pd.Timedelta(days=90), day, 2.0, "v2"),
            ]
        )
    )
    assert store.as_of("eps", day)["value"].iloc[0] == pytest.approx(2.0)


def test_store_rejects_look_ahead_and_missing_release_timestamps():
    day = pd.Timestamp("2022-03-01")
    with pytest.raises(ValueError, match="released before"):
        PointInTimeStore(
            pd.DataFrame([_row("AAA", "eps", day, day - pd.Timedelta(days=1), 1.0)])
        )
    missing = pd.DataFrame(
        [{"ticker": "AAA", "field": "eps", "period_end": day, "value": 1.0}]
    )
    with pytest.raises(ValueError, match="release_ts"):
        PointInTimeStore(missing)


def test_snapshot_hash_is_stable_under_row_order():
    day = pd.Timestamp("2022-03-01")
    rows = [
        _row("AAA", "eps", day, day + pd.Timedelta(days=40), 1.0),
        _row("BBB", "eps", day, day + pd.Timedelta(days=40), 2.0),
        _row("AAA", "book_value", day, day + pd.Timedelta(days=50), 5.0),
    ]
    first = PointInTimeStore(pd.DataFrame(rows)).snapshot_hash()
    second = PointInTimeStore(pd.DataFrame(rows[::-1])).snapshot_hash()
    assert first == second
    assert first.startswith("sha256:")


@given(
    n_assets=st.integers(min_value=1, max_value=4),
    rows=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=40),
            st.integers(min_value=0, max_value=20),
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=12,
    ),
    cutoff_day=st.integers(min_value=0, max_value=70),
)
@settings(max_examples=40, deadline=None)
def test_deleting_future_rows_never_changes_a_value(n_assets, rows, cutoff_day):
    """Deleting every row released after t must not change any feature at t."""
    base = pd.Timestamp("2020-01-01")
    records = []
    for index, (period_day, lag, value) in enumerate(rows):
        records.append(
            _row(
                f"T{index % n_assets}",
                "eps",
                base + pd.Timedelta(days=period_day),
                base + pd.Timedelta(days=period_day + lag),
                value,
            )
        )
    full = pd.DataFrame(records)
    cutoff = base + pd.Timedelta(days=cutoff_day)
    truncated = full[full["release_ts"] <= cutoff]

    before = PointInTimeStore(full).as_of("eps", cutoff)
    after = PointInTimeStore(truncated).as_of("eps", cutoff)
    if before.empty and after.empty:
        assert list(before.columns) == list(after.columns)
    else:
        pd.testing.assert_frame_equal(before, after)


def test_feature_frame_is_unchanged_by_future_releases():
    base = pd.Timestamp("2020-01-01")
    dates = pd.bdate_range(base, periods=10)
    early = PointInTimeStore(
        pd.DataFrame([_row("AAA", "eps", base, base + pd.Timedelta(days=5), 1.0)])
    )
    late = PointInTimeStore(
        pd.DataFrame(
            [
                _row("AAA", "eps", base, base + pd.Timedelta(days=5), 1.0),
                _row("AAA", "eps", base, base + pd.Timedelta(days=8), 9.9),
            ]
        )
    )
    cutoff = dates[4]
    pd.testing.assert_frame_equal(
        early.feature_frame("eps", dates[dates <= cutoff]),
        late.feature_frame("eps", dates[dates <= cutoff]),
    )
