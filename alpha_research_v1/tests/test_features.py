"""Stage 2 acceptance tests: EWMA identities and fold-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpharig.features.fold import FoldGuard, FoldViolationError
from alpharig.features.transforms import (
    CrossSectionalRank,
    CrossSectionalZScore,
    EwmaSmoother,
    FeaturePipeline,
    Lag,
    Neutralizer,
    NotFittedError,
    RobustScaler,
    RollingStat,
    Winsorizer,
    ewma_decay,
    ewma_lambda,
    half_life_from_decay,
)

TRAIN = pd.bdate_range("2020-01-01", periods=40)
TEST = pd.bdate_range("2020-06-01", periods=10)
FOLD = FoldGuard(TRAIN, TEST, fold_id=3)
NAMES = list("ABCDE")


def _panel(seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = TRAIN.append(TEST)
    return pd.DataFrame(
        rng.normal(scale=scale, size=(len(index), len(NAMES))),
        index=index,
        columns=NAMES,
    )


# ----------------------------------------------------------------------
# EWMA arithmetic
# ----------------------------------------------------------------------
def test_ewma_half_life_identities():
    # lambda = 1 - 0.5 ** (1 / h)
    assert ewma_lambda(3) == pytest.approx(0.2063, abs=1e-4)
    assert ewma_lambda(20) == pytest.approx(0.0341, abs=1e-4)
    # decay weight 0.1 -> half-life ln(0.5)/ln(0.1) = 6.58 periods
    assert half_life_from_decay(0.1) == pytest.approx(6.579, abs=1e-3)
    assert ewma_decay(20) == pytest.approx(1.0 - ewma_lambda(20))
    assert ewma_lambda(half_life_from_decay(0.1)) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        ewma_lambda(0.0)
    with pytest.raises(ValueError):
        half_life_from_decay(1.0)


def test_ewma_recursion_matches_its_own_lambda():
    data = pd.DataFrame({"A": np.arange(1.0, 31.0)}, index=pd.bdate_range("2021-01-01", periods=30))
    fold = FoldGuard(data.index[:20], data.index[20:], fold_id=0)
    smoother = EwmaSmoother(3).fit(data.iloc[:20], fold)
    smoothed = smoother.transform(data)["A"].to_numpy()

    alpha = ewma_lambda(3)
    expected = [data["A"].iloc[0]]
    for value in data["A"].to_numpy()[1:]:
        expected.append(alpha * value + (1.0 - alpha) * expected[-1])
    np.testing.assert_allclose(smoothed, expected, rtol=1e-12, atol=1e-12)


def test_ewma_is_causal():
    data = pd.DataFrame({"A": np.arange(1.0, 31.0)}, index=pd.bdate_range("2021-01-01", periods=30))
    fold = FoldGuard(data.index[:20], data.index[20:], fold_id=0)
    smoother = EwmaSmoother(5).fit(data.iloc[:20], fold)
    smoothed = smoother.transform(data)["A"]
    # Changing a later observation must not change an earlier smoothed value.
    tampered = data.copy()
    tampered.iloc[25:, 0] += 100.0
    assert smoother.transform(tampered)["A"].iloc[:25].equals(smoothed.iloc[:25])


# ----------------------------------------------------------------------
# fold discipline
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "make",
    [
        lambda: CrossSectionalZScore(),
        lambda: CrossSectionalRank(),
        lambda: Winsorizer(),
        lambda: RobustScaler(),
        lambda: Lag(1),
        lambda: RollingStat(5),
        lambda: EwmaSmoother(3),
    ],
)
def test_every_transformer_refuses_a_missing_fold(make):
    with pytest.raises(FoldViolationError):
        make().fit(_panel().loc[TRAIN], None)


@pytest.mark.parametrize(
    "make",
    [
        lambda: CrossSectionalZScore(),
        lambda: CrossSectionalRank(),
        lambda: Winsorizer(),
        lambda: RobustScaler(),
        lambda: Lag(1),
        lambda: RollingStat(5),
        lambda: EwmaSmoother(3),
    ],
)
def test_every_transformer_refuses_to_fit_on_the_full_panel(make):
    full = _panel()
    with pytest.raises(FoldViolationError):
        make().fit(full, FOLD)


def test_fold_guard_rejects_overlapping_windows():
    with pytest.raises(FoldViolationError):
        FoldGuard(TRAIN, TRAIN[-2:], fold_id=1)
    with pytest.raises(FoldViolationError):
        FoldGuard(pd.DatetimeIndex([]), TEST)


def test_pipeline_refuses_to_fit_without_a_fold():
    pipeline = FeaturePipeline([Winsorizer(), CrossSectionalZScore()])
    with pytest.raises(FoldViolationError):
        pipeline.fit(_panel(), None)
    fitted = pipeline.fit(_panel().loc[TRAIN], FOLD)
    assert [step["name"] for step in fitted.state()["steps"]] == ["winsorize", "cs_zscore"]


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        CrossSectionalZScore().transform(_panel())


# ----------------------------------------------------------------------
# in-fold parameter estimation
# ----------------------------------------------------------------------
def test_winsorizer_bounds_come_from_training_rows_only():
    train = _panel(seed=1)
    test = _panel(seed=2)
    test.iloc[0, 0] = 1_000.0  # a violent outlier that exists only out of fold

    winsor = Winsorizer(0.01, 0.99).fit(train.loc[TRAIN], FOLD)
    out = winsor.transform(test)

    bound = winsor.state()["state"]["high_bound"]
    assert out.loc[TEST[0], "A"] == pytest.approx(bound)
    assert out.max().max() <= bound + 1e-12
    # The training fit never saw the outlier, so its bound stays sane.
    assert bound < 5.0


def test_cross_sectional_zscore_and_rank_are_per_date():
    panel = _panel(seed=4)
    z = CrossSectionalZScore().fit_transform(panel.loc[TRAIN], FOLD)
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=1, ddof=0).to_numpy(), 1.0, atol=1e-12)

    ranked = CrossSectionalRank(output="centered").fit_transform(panel.loc[TRAIN], FOLD)
    assert ranked.max().max() <= 1.0
    assert ranked.min().min() >= -1.0
    assert np.allclose(ranked.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_winsorizer_leaves_values_inside_the_bounds_untouched():
    panel = _panel(seed=5)
    winsor = Winsorizer(0.05, 0.95).fit(panel.loc[TRAIN], FOLD)
    out = winsor.transform(panel.loc[TRAIN])
    low = winsor.state()["state"]["low_bound"]
    high = winsor.state()["state"]["high_bound"]
    inside = (panel.loc[TRAIN] > low) & (panel.loc[TRAIN] < high)
    np.testing.assert_allclose(
        out.where(inside).to_numpy()[inside.to_numpy()],
        panel.loc[TRAIN].to_numpy()[inside.to_numpy()],
    )
    assert out.min().min() >= low
    assert out.max().max() <= high


def test_robust_scaler_uses_in_fold_median_and_mad():
    panel = _panel(seed=6)
    scaler = RobustScaler().fit(panel.loc[TRAIN], FOLD)
    center = scaler.state()["state"]["center"]
    assert center["A"] == pytest.approx(panel.loc[TRAIN, "A"].median())
    out = scaler.transform(panel.loc[TRAIN])
    # Median centring leaves the sign structure intact and roughly unit scale.
    assert abs(out.median().median()) < 1e-9
    assert 0.3 < out.abs().median().median() < 3.0


def test_neutralizer_removes_a_declared_exposure():
    rng = np.random.default_rng(7)
    index = TRAIN.append(TEST)
    size = pd.DataFrame(
        rng.normal(size=(len(index), 6)), index=index, columns=[f"N{i}" for i in range(6)]
    )
    noise = pd.DataFrame(
        rng.normal(scale=0.1, size=size.shape), index=index, columns=size.columns
    )
    feature = 2.5 * size + noise

    neutral = Neutralizer({"log_size": size}).fit(feature.loc[TRAIN], FOLD)
    beta = neutral.state()["state"]["beta"]["log_size"]
    assert beta == pytest.approx(2.5, abs=0.1)

    residual = neutral.transform(feature.loc[TEST])
    for date in TEST:
        correlation = residual.loc[date].corr(size.loc[date])
        assert abs(correlation) < 0.35


def test_neutralizer_needs_exposures():
    with pytest.raises(ValueError):
        Neutralizer({})
