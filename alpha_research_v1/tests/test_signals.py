"""Stage 2 acceptance tests: the versioned signal registry and its library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpharig.features.inputs import SignalInputs, build_signal_inputs
from alpharig.features.registry import (
    SIGNALS,
    InsufficientHistoryError,
    SignalLookAheadError,
    SignalRegistry,
    SignalSpec,
    SignalVersionError,
)


def test_library_signals_are_registered_with_declarations():
    registry = SIGNALS
    for name in (
        "momentum_12_1",
        "short_term_reversal",
        "low_volatility",
        "earnings_yield",
        "book_to_price",
        "revenue_growth",
        "log_dollar_volume",
        "amihud_illiquidity",
    ):
        spec = registry.get(name)
        assert spec.version >= 1
        assert spec.lookback >= 0
        assert spec.family
        assert spec.description
    table = registry.describe()
    assert len(table) == len(registry)
    assert set(table["family"]) >= {"momentum", "reversal", "value", "liquidity"}


def test_registry_is_versioned_and_old_versions_survive():
    registry = SignalRegistry("test")

    def first(inputs: SignalInputs) -> pd.DataFrame:
        return inputs.close

    def second(inputs: SignalInputs) -> pd.DataFrame:
        return inputs.close * 2.0

    registry.register(SignalSpec(name="demo", version=1, func=first, lookback=0))
    registry.register(SignalSpec(name="demo", version=2, func=second, lookback=0))
    assert registry.versions("demo") == (1, 2)
    assert registry.get("demo").version == 2
    assert registry.get("demo", 1).func is first


def test_code_change_without_a_version_bump_is_rejected():
    registry = SignalRegistry("test")

    def original(inputs: SignalInputs) -> pd.DataFrame:
        return inputs.close

    def edited(inputs: SignalInputs) -> pd.DataFrame:
        return inputs.close.shift(1)

    registry.register(SignalSpec(name="demo", version=1, func=original, lookback=0))
    # Re-registering identical code is harmless (module reloads).
    registry.register(SignalSpec(name="demo", version=1, func=original, lookback=0))
    with pytest.raises(SignalVersionError):
        registry.register(SignalSpec(name="demo", version=1, func=edited, lookback=0))


def test_signal_computation_is_point_in_time(feature_panel):
    dates = feature_panel.calendar.dates
    early = build_signal_inputs(feature_panel, dates[80])
    late = build_signal_inputs(feature_panel, dates[125])

    assert early.dates.max() <= dates[80]
    early_signal = SIGNALS.compute("earnings_yield", early)
    late_signal = SIGNALS.compute("earnings_yield", late)
    assert early_signal.index.max() == dates[80]
    # Later data must never revise an earlier value.
    pd.testing.assert_frame_equal(
        early_signal, late_signal.loc[early_signal.index], check_exact=False, rtol=1e-12
    )


def test_signals_return_wide_frames_masked_to_the_universe(feature_panel):
    inputs = build_signal_inputs(feature_panel, feature_panel.calendar.dates[100])
    for name in ("short_term_reversal", "log_dollar_volume", "low_volatility"):
        frame = SIGNALS.compute(name, inputs)
        assert frame.shape == (len(inputs.dates), len(inputs.tickers))
        assert frame.index.equals(inputs.dates)
        assert frame.columns.equals(inputs.tickers)
        assert frame.notna().to_numpy().any()


def test_insufficient_history_is_an_explicit_error(feature_panel):
    dates = feature_panel.calendar.dates
    inputs = build_signal_inputs(feature_panel, dates[100])
    assert len(inputs.dates) < SIGNALS.get("momentum_12_1").lookback
    with pytest.raises(InsufficientHistoryError):
        SIGNALS.compute("momentum_12_1", inputs)


def test_declared_release_lag_shifts_the_output(feature_panel):
    registry = SignalRegistry("test")

    def raw(inputs: SignalInputs) -> pd.DataFrame:
        return inputs.close

    registry.register(SignalSpec(name="raw", version=1, func=raw, lookback=0))
    registry.register(
        SignalSpec(name="lagged", version=1, func=raw, lookback=0, release_lag=2)
    )
    inputs = build_signal_inputs(feature_panel, feature_panel.calendar.dates[60])
    plain = registry.compute("raw", inputs)
    lagged = registry.compute("lagged", inputs)
    assert lagged.iloc[:2].isna().to_numpy().all()
    # Same dates, values moved two periods older.
    assert lagged.index.equals(plain.index)
    np.testing.assert_allclose(
        lagged.iloc[2:].to_numpy(), plain.iloc[:-2].to_numpy(), equal_nan=True
    )


def test_a_signal_that_invents_a_future_date_is_rejected(feature_panel):
    registry = SignalRegistry("test")

    def cheating(inputs: SignalInputs) -> pd.DataFrame:
        frame = inputs.close.copy()
        frame.loc[inputs.as_of + pd.Timedelta(days=1)] = 1.0
        return frame

    registry.register(SignalSpec(name="cheat", version=1, func=cheating, lookback=0))
    inputs = build_signal_inputs(feature_panel, feature_panel.calendar.dates[60])
    with pytest.raises(SignalLookAheadError):
        registry.compute("cheat", inputs)


def test_inputs_reject_dates_after_as_of():
    dates = pd.bdate_range("2022-01-03", periods=5)
    frame = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    with pytest.raises(ValueError):
        SignalInputs(
            as_of=dates[2],
            close=frame,
            volume=frame,
            fundamentals={},
        )


def test_unknown_signal_raises_key_error():
    with pytest.raises(KeyError):
        SIGNALS.get("does_not_exist")
