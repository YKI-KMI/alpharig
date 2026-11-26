"""Effective sample size and overlap corrections.

Ground rule: count dates, not rows. A cross-section of ``N`` names observed on
``T`` dates is not ``N*T`` independent observations when names move together.
For a constant within-date correlation ``rho`` the effective sample size is

    N_eff = T * N / (1 + (N - 1) * rho)

and it is capped at ``T / rho`` (the number of effectively independent
cross-sections). Every reported statistic in alpharig is expected to carry an
``N_eff`` alongside its row count.

This module is deliberately dependency-free so it can be used by reports,
tests and the validation harness without importing the data stack.
"""

from __future__ import annotations

import math


def effective_sample_size(n_periods: float, n_assets: float, rho: float) -> float:
    """Return the effective number of independent observations in a panel.

    Parameters
    ----------
    n_periods:
        Number of dates (``T``) in the sample. Must be positive.
    n_assets:
        Number of names (``N``) observed per date. Must be positive.
    rho:
        Average pairwise within-date correlation of the quantity being
        measured. Must be greater than ``-1`` and at most ``1``.

    Returns
    -------
    float
        ``T * N / (1 + (N - 1) * rho)``, capped at ``T / rho`` when
        ``rho > 0``. With ``rho <= 0`` the raw formula is returned: negative
        cross-sectional correlation increases information and the cap does
        not apply.

    Raises
    ------
    ValueError
        If counts are not positive or ``rho`` is outside ``(-1, 1]``.
    """
    if n_periods <= 0 or n_assets <= 0:
        raise ValueError("n_periods and n_assets must be positive")
    if not (rho > -1.0 and rho <= 1.0):
        raise ValueError("rho must be in (-1, 1]")
    if n_assets == 1:
        return float(n_periods)
    raw = n_periods * n_assets / (1.0 + (n_assets - 1.0) * rho)
    if rho <= 0:
        return float(raw)
    return float(min(raw, n_periods / rho))


def overlap_effective_sample_size(n_observations: float, horizon: float) -> float:
    """Effective sample size of overlapping ``horizon``-period labels.

    ``n_observations`` rows with an ``h``-period forward label contain roughly
    ``n / h`` non-overlapping observations.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_observations <= 0:
        return 0.0
    return float(n_observations) / float(horizon)


def overconfidence_factor(horizon: float) -> float:
    """Factor by which overlapping labels inflate a naive t-statistic.

    The naive standard error of a mean computed from ``h``-overlapping
    observations is too small by roughly ``sqrt(h)``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return math.sqrt(float(horizon))
