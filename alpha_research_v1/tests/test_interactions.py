"""Stage 2 acceptance tests: declared-only interactions and their arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpharig.features.interactions import (
    InteractionRegistry,
    UndeclaredInteractionError,
    all_pairs_count,
    effect_table,
    interaction_column,
    marginal_effect,
    zero_crossing_z,
)


def _feature_frame(seed: int, n_dates: int = 30, n_names: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_dates, n_names)),
        index=pd.bdate_range("2023-01-02", periods=n_dates),
        columns=[f"T{i}" for i in range(n_names)],
    )


def test_forty_features_without_declarations_build_nothing_and_warn_780():
    registry = InteractionRegistry()
    features = {f"feature_{i:02d}": _feature_frame(i) for i in range(40)}

    assert all_pairs_count(40) == 780
    build = registry.build(features)

    assert build.panels == {}
    assert build.n_interactions == 0
    assert build.declared == []
    assert build.all_pairs == 780
    assert build.suppressed == 780
    assert any("780" in warning for warning in build.warnings)
    assert any("C(40, 2)" in warning for warning in build.warnings)


def test_only_declared_pairs_are_built():
    registry = InteractionRegistry()
    registry.declare(
        "momentum_12_1",
        "volatility",
        "Momentum inverts in high-volatility regimes; the sign flip is the hypothesis.",
    )
    features = {
        "momentum_12_1": _feature_frame(1),
        "volatility": _feature_frame(2),
        "size": _feature_frame(3),
    }

    build = registry.build(features)

    assert build.names == [interaction_column("momentum_12_1", "volatility")]
    assert build.declared == ["x__momentum_12_1__volatility"]
    assert build.all_pairs == 3  # C(3, 2)
    assert build.suppressed == 2
    assert any("suppressed" in warning for warning in build.warnings)

    expected = features["momentum_12_1"] * features["volatility"]
    pd.testing.assert_frame_equal(build.panels[build.declared[0]], expected)
    long = build.to_long()
    assert list(long.columns) == ["date", "ticker", "interaction", "value"]
    assert long["interaction"].unique().tolist() == build.declared


def test_an_interaction_without_a_rationale_is_rejected():
    registry = InteractionRegistry()
    with pytest.raises(ValueError):
        registry.declare("momentum", "volatility", "   ")
    with pytest.raises(ValueError):
        registry.declare("momentum", "momentum", "self-crossing has no meaning")
    assert len(registry) == 0


def test_all_pairs_crossing_can_only_be_requested_explicitly_as_an_error():
    registry = InteractionRegistry()
    features = {"a": _feature_frame(1), "b": _feature_frame(2)}
    with pytest.raises(UndeclaredInteractionError):
        registry.build(features, require_declaration=False)


def test_interaction_effect_arithmetic():
    # Planted model: y = 0.9*f - 1.1*f*z. The effect of f is 0.9 at z=0,
    # -0.2 at z=1, and changes sign at z = 0.9/1.1 = 0.818.
    rng = np.random.default_rng(11)
    n = 800
    feature = rng.standard_normal(n)
    conditioner = rng.standard_normal(n)
    target = 0.9 * feature - 1.1 * feature * conditioner + rng.normal(scale=0.01, size=n)

    design = np.column_stack([np.ones(n), feature, feature * conditioner])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    beta1, beta3 = float(coefficients[1]), float(coefficients[2])

    assert beta1 == pytest.approx(0.9, abs=0.01)
    assert beta3 == pytest.approx(-1.1, abs=0.01)
    assert marginal_effect(beta1, beta3, 0.0) == pytest.approx(0.9, abs=0.01)
    assert marginal_effect(beta1, beta3, 1.0) == pytest.approx(-0.2, abs=0.02)
    assert zero_crossing_z(beta1, beta3) == pytest.approx(0.818, abs=0.01)
    assert marginal_effect(beta1, beta3, zero_crossing_z(beta1, beta3)) == pytest.approx(
        0.0, abs=0.01
    )
    table = effect_table(beta1, beta3)
    assert list(table["conditioner"]) == [0.0, 0.5, 1.0]
    assert table["effect"].iloc[0] > 0 > table["effect"].iloc[-1]


def test_zero_crossing_is_none_without_an_interaction():
    assert zero_crossing_z(0.9, 0.0) is None
    assert marginal_effect(0.9, 0.0, 5.0) == pytest.approx(0.9)


def test_missing_features_are_reported_not_silently_dropped():
    registry = InteractionRegistry()
    registry.declare("momentum", "volatility", "declared for the sign-flip test")
    build = registry.build({"momentum": _feature_frame(1)})
    assert build.n_interactions == 0
    assert any("missing" in warning for warning in build.warnings)
    assert build.to_long().empty
