"""Panel statistics computed date by date (never pooled across dates)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_pairwise_correlation(
    returns: pd.DataFrame,
    min_periods: int = 10,
) -> float:
    """Average pairwise contemporaneous correlation of a return panel.

    Contemporaneous ("within-date") correlation is a property of the asset
    time series, not of a single cross-section: a common factor that shifts
    every name by the same amount on a date adds no cross-sectional
    dispersion, yet it is exactly what makes names move together. The
    correlation is therefore computed across dates for every asset pair and
    then averaged off the diagonal.

    Parameters
    ----------
    returns:
        Wide frame indexed by date with one column per asset. Missing values
        are handled pairwise.
    min_periods:
        Minimum overlapping observations required for a pair correlation.

    Returns
    -------
    float
        Mean off-diagonal correlation, or NaN when fewer than two assets have
        enough overlapping history.
    """
    frame = returns.dropna(axis=1, how="all").astype(float)
    if frame.shape[1] < 2:
        return float("nan")
    correlation = frame.corr(min_periods=min_periods)
    values = correlation.to_numpy(dtype=float)
    n = values.shape[0]
    mask = ~np.eye(n, dtype=bool) & np.isfinite(values)
    if not mask.any():
        return float("nan")
    return float(values[mask].mean())


def cross_sectional_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    min_names: int = 5,
    method: str = "pearson",
) -> pd.Series:
    """Correlation between two panels, computed independently on each date.

    Parameters
    ----------
    left, right:
        Wide frames with a shared date index and asset columns. Only names
        present in both frames on a date are used.
    min_names:
        Minimum overlap required to report a date.
    method:
        ``"pearson"`` or ``"spearman"`` (rank correlation, i.e. rank IC when
        ``left`` is a signal and ``right`` is a forward return).

    Returns
    -------
    pd.Series
        Per-date correlation, indexed by date. Empty if no date qualifies.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    common_dates = left.index.intersection(right.index)
    out: dict[pd.Timestamp, float] = {}
    for date in common_dates:
        x = left.loc[date].astype(float)
        y = right.loc[date].astype(float)
        mask = x.notna() & y.notna()
        if int(mask.sum()) < min_names:
            continue
        xv = x[mask]
        yv = y[mask]
        if method == "spearman":
            xv = xv.rank()
            yv = yv.rank()
        if xv.std(ddof=0) == 0.0 or yv.std(ddof=0) == 0.0:
            continue
        out[pd.Timestamp(date)] = float(np.corrcoef(xv.to_numpy(), yv.to_numpy())[0, 1])
    return pd.Series(out, dtype=float).sort_index()


def cross_sectional_rank_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_names: int = 5,
) -> pd.Series:
    """Per-date Spearman rank IC between a signal and forward returns."""
    return cross_sectional_correlation(
        signal, forward_returns, min_names=min_names, method="spearman"
    )
