"""The starter signal library.

Each signal declares its lookback, family and lag so the validation stage can
purge training rows that would otherwise peek at a test fold. Signals read only
:class:`~alpharig.features.inputs.SignalInputs`; they never touch the panel,
the store or the clock.

These are deliberately plain, well-known constructions. They are here so the
pipeline has something honest to evaluate, not because they are expected to
work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpharig.features.inputs import SignalInputs
from alpharig.features.registry import signal


@signal(
    name="momentum_12_1",
    lookback=273,
    family="momentum",
    description="Twelve-month price momentum, skipping the most recent month.",
)
def momentum_12_1(inputs: SignalInputs) -> pd.DataFrame:
    """Total return over 252 periods excluding the last 21."""
    close = inputs.close
    total = close.pct_change(252, fill_method=None)
    recent = close.pct_change(21, fill_method=None)
    value = (1.0 + total) / (1.0 + recent) - 1.0
    return inputs.masked(value)


@signal(
    name="short_term_reversal",
    lookback=6,
    family="reversal",
    description="Negative five-period return: recent losers are expected to bounce.",
)
def short_term_reversal(inputs: SignalInputs) -> pd.DataFrame:
    """Minus the trailing five-period return."""
    return inputs.masked(-inputs.close.pct_change(5, fill_method=None))


@signal(
    name="low_volatility",
    lookback=61,
    family="risk",
    description="Negative trailing return volatility: low-volatility names are favoured.",
)
def low_volatility(inputs: SignalInputs) -> pd.DataFrame:
    """Minus the 60-period standard deviation of returns."""
    volatility = inputs.returns(1).rolling(60, min_periods=40).std()
    return inputs.masked(-volatility)


@signal(
    name="earnings_yield",
    lookback=1,
    family="value",
    description="Trailing earnings per share divided by price, point in time.",
)
def earnings_yield(inputs: SignalInputs) -> pd.DataFrame:
    """EPS released at or before the observation date, over price."""
    earnings = inputs.field("eps")
    return inputs.masked(earnings.div(inputs.close))


@signal(
    name="book_to_price",
    lookback=1,
    family="value",
    description="Book value per share divided by price, point in time.",
)
def book_to_price(inputs: SignalInputs) -> pd.DataFrame:
    """Released book value over price."""
    book = inputs.field("book_value")
    return inputs.masked(book.div(inputs.close))


@signal(
    name="revenue_growth",
    lookback=64,
    family="growth",
    description="Year-over-year change in released revenue.",
)
def revenue_growth(inputs: SignalInputs) -> pd.DataFrame:
    """Revenue change over roughly four quarters of trading days."""
    revenue = inputs.field("revenue")
    return inputs.masked(revenue.pct_change(63, fill_method=None))


@signal(
    name="log_dollar_volume",
    lookback=21,
    family="liquidity",
    description="Log of trailing average dollar volume; a size and liquidity proxy.",
)
def log_dollar_volume(inputs: SignalInputs) -> pd.DataFrame:
    """Log of the 20-period mean dollar volume."""
    adv = inputs.dollar_volume().rolling(20, min_periods=10).mean()
    return inputs.masked(np.log(adv.where(adv > 0)))


@signal(
    name="amihud_illiquidity",
    lookback=21,
    family="liquidity",
    description="Mean absolute return per unit of dollar volume (Amihud).",
)
def amihud_illiquidity(inputs: SignalInputs) -> pd.DataFrame:
    """Trailing mean of |return| / dollar volume."""
    turnover = inputs.dollar_volume()
    ratio = inputs.returns(1).abs().div(turnover.where(turnover > 0))
    return inputs.masked(ratio.rolling(20, min_periods=10).mean())
