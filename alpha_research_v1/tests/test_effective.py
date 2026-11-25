"""Tests for effective sample size and overlap corrections."""

from __future__ import annotations

import math

import pytest

from alpharig.stats.effective import (
    effective_sample_size,
    overlap_effective_sample_size,
    overconfidence_factor,
)


def test_effective_sample_size_matches_formula_and_cap():
    # T=60, N=2000, rho=0.3 -> 199.77, essentially the T/rho cap of 200.
    value = effective_sample_size(60, 2000, 0.3)
    assert value == pytest.approx(60 * 2000 / (1 + 1999 * 0.3), rel=1e-12)
    assert value < 200
    assert value == pytest.approx(199.77, abs=0.05)
    assert effective_sample_size(60, 2000, 1.0) == pytest.approx(60.0)


def test_effective_sample_size_independent_and_degenerate_cases():
    assert effective_sample_size(60, 2000, 0.0) == pytest.approx(120_000.0)
    assert effective_sample_size(100, 1, 0.9) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        effective_sample_size(0, 10, 0.1)
    with pytest.raises(ValueError):
        effective_sample_size(10, 10, 1.5)


def test_overlap_accounting():
    assert overlap_effective_sample_size(1250, 20) == pytest.approx(62.5)
    assert overconfidence_factor(20) == pytest.approx(math.sqrt(20))
    assert overconfidence_factor(20) == pytest.approx(4.4721, abs=1e-4)
    with pytest.raises(ValueError):
        overlap_effective_sample_size(100, 0)
