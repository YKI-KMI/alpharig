"""Pre-lagged, universe-filtered inputs handed to a signal.

A signal never receives the raw panel. It receives a :class:`SignalInputs`
object that is already truncated at ``as_of`` and already masked to investable
names, so a signal *cannot* read the future even if its author tries: the rows
simply are not there.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpharig.data.universe import PointInTimeUniverse, UniverseRules
from alpharig.synth.generator import SyntheticPanel


@dataclass(frozen=True)
class SignalInputs:
    """Everything a signal may look at, frozen at one observation date.

    Parameters
    ----------
    as_of:
        Last date visible to the signal. Every frame is truncated to it.
    close, volume:
        Wide price and share-volume panels indexed by date with one column per
        ticker.
    fundamentals:
        Mapping from field name to a wide point-in-time fundamental panel.
    universe:
        Boolean wide panel marking investable names; ``None`` means "all names
        with data".

    Raises
    ------
    ValueError
        If the price panels disagree on shape or the index is not date-typed.
    """

    as_of: pd.Timestamp
    close: pd.DataFrame
    volume: pd.DataFrame
    fundamentals: Mapping[str, pd.DataFrame]
    universe: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        stamp = pd.Timestamp(self.as_of)
        if stamp is pd.NaT:
            raise ValueError("as_of must be a valid date")
        object.__setattr__(self, "as_of", stamp)
        close = self.close.astype(float).sort_index()
        volume = self.volume.astype(float).sort_index().reindex(
            index=close.index, columns=close.columns
        )
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)
        for field, frame in self.fundamentals.items():
            aligned = frame.astype(float).reindex(index=close.index, columns=close.columns)
            if aligned is not frame:
                fundamentals = dict(self.fundamentals)
                fundamentals[field] = aligned
                object.__setattr__(self, "fundamentals", fundamentals)
        if close.index.max() > stamp:
            raise ValueError(
                f"inputs contain dates after as_of={stamp.date()}; build them "
                "with build_signal_inputs() so the future is unavailable"
            )

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        """Visible observation dates."""
        return pd.DatetimeIndex(self.close.index)

    @property
    def tickers(self) -> pd.Index:
        """Ticker columns."""
        return pd.Index(self.close.columns)

    @property
    def fields(self) -> tuple[str, ...]:
        """Available fundamental field names."""
        return tuple(sorted(self.fundamentals))

    def field(self, name: str) -> pd.DataFrame:
        """Return a fundamental panel.

        Raises
        ------
        KeyError
            If the field is not available, listing what is.
        """
        try:
            return self.fundamentals[name]
        except KeyError as exc:
            raise KeyError(
                f"fundamental field {name!r} is not available; inputs carry "
                f"{list(self.fields)}"
            ) from exc

    def returns(self, periods: int = 1) -> pd.DataFrame:
        """Simple returns over ``periods`` dates (no forward fill)."""
        return self.close.pct_change(periods, fill_method=None)

    def dollar_volume(self) -> pd.DataFrame:
        """Close times share volume, the liquidity primitive."""
        return self.close * self.volume

    def masked(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Blank out names that are not investable that day."""
        aligned = frame.reindex(index=self.dates, columns=self.tickers).astype(float)
        if self.universe is None:
            return aligned
        mask = self.universe.reindex(index=self.dates, columns=self.tickers)
        return aligned.where(mask.fillna(False).astype(bool))


def build_signal_inputs(
    panel: SyntheticPanel,
    as_of: object,
    fields: Iterable[str] | None = None,
    rules: UniverseRules | None = None,
    apply_universe: bool = True,
) -> SignalInputs:
    """Assemble pre-lagged, universe-filtered inputs at ``as_of``.

    Prices and volumes are truncated at ``as_of``; fundamental panels come from
    the point-in-time store, so a value appears only from its release date. The
    universe mask is evaluated date by date from trailing liquidity, listing
    history and restrictions.

    Parameters
    ----------
    panel:
        Synthetic panel supplying prices, fundamentals and membership.
    as_of:
        Observation date; nothing after it is visible.
    fields:
        Fundamental fields to materialise. Defaults to every field in the store.
    rules:
        Universe thresholds; defaults match :class:`UniverseRules`.
    apply_universe:
        When false, the mask is ``None`` and every name with data is visible.
        Use only for diagnostics, never for a reported result.

    Returns
    -------
    SignalInputs
        Truncated and masked inputs.

    Raises
    ------
    ValueError
        If ``as_of`` precedes every price observation.
    """
    stamp = pd.Timestamp(as_of).normalize()
    prices = panel.prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices[prices["date"] <= stamp]
    if prices.empty:
        raise ValueError(f"no price data at or before {stamp.date()}")

    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    close = prices.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    ).sort_index()
    volume = prices.pivot_table(
        index="date", columns="ticker", values="volume", aggfunc="last"
    ).sort_index().reindex(index=close.index, columns=close.columns)

    store = panel.store()
    wanted = tuple(fields) if fields is not None else store.fields
    fundamentals = {
        name: store.feature_frame(name, dates, close.columns) for name in wanted
    }

    universe_mask: pd.DataFrame | None = None
    if apply_universe:
        universe = panel.universe(rules or UniverseRules())
        columns = close.columns
        mask = pd.DataFrame(False, index=dates, columns=columns)
        for date in dates:
            members = [ticker for ticker in universe.members(date) if ticker in columns]
            if members:
                mask.loc[date, members] = True
        universe_mask = mask

    return SignalInputs(
        as_of=stamp,
        close=close,
        volume=volume,
        fundamentals=fundamentals,
        universe=universe_mask,
    )


def trailing_adv(inputs: SignalInputs, window: int = 20) -> pd.DataFrame:
    """Trailing mean dollar volume up to ``as_of`` (causal)."""
    if window <= 0:
        raise ValueError("window must be positive")
    return inputs.dollar_volume().rolling(window, min_periods=max(1, window // 2)).mean()


def safe_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Natural log of a strictly positive, non-finite-safe frame."""
    values = frame.astype(float).where(frame > 0)
    return np.log(values)
