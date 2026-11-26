"""Stage 2 acceptance tests: redundancy pruning and cluster importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from alpharig.features.importance import (
    cluster_permutation_importance,
    permutation_importance,
    r2_score_numpy,
)
from alpharig.features.redundancy import (
    correlation_clusters,
    correlation_distance,
    prune_redundant,
    pruning_report,
)

MOMENTUM = ["mom_1", "mom_2", "mom_3", "mom_4"]
INDEPENDENT = ["ind_a", "ind_b"]


def _feature_set(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Four near-identical momentum columns plus two independent features.

    The target loads 1.0 on the shared momentum factor and only 0.3 on each
    independent feature, so a cluster-level measure must dominate any single
    column while each independent column beats each momentum column solo.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n)
    data = {name: common + 0.25 * rng.standard_normal(n) for name in MOMENTUM}
    data["ind_a"] = rng.standard_normal(n)
    data["ind_b"] = rng.standard_normal(n)
    features = pd.DataFrame(data)
    target = (
        1.0 * common
        + 0.3 * features["ind_a"].to_numpy()
        + 0.3 * features["ind_b"].to_numpy()
        + 0.15 * rng.standard_normal(n)
    )
    return features, target


def test_correlation_distance_is_monotone_in_correlation():
    frame = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    distance = correlation_distance(frame)
    assert distance.loc["a", "b"] == pytest.approx(np.sqrt(0.25))
    assert distance.loc["a", "a"] == 0.0


def test_redundant_features_are_grouped_and_independent_ones_are_not():
    features, _ = _feature_set()
    report = correlation_clusters(features, threshold=0.5)

    momentum_clusters = {report.cluster_of(name) for name in MOMENTUM}
    assert len(momentum_clusters) == 1
    assert report.cluster_of("ind_a") != report.cluster_of("ind_b")
    assert report.cluster_of("ind_a") != report.cluster_of("mom_1")
    assert len(report.multi_member()) == 1
    assert len(report.labels) == len(features.columns)


def test_pruning_keeps_one_representative_per_cluster():
    features, _ = _feature_set()
    kept = prune_redundant(features, threshold=0.5)

    assert len([name for name in kept if name in MOMENTUM]) == 1
    assert set(INDEPENDENT) <= set(kept)
    assert len(kept) == 3

    report = pruning_report(features, threshold=0.5)
    assert report["kept"].sum() == len(kept)
    assert set(report.loc[report["kept"], "feature"]) == set(kept)
    assert (report.loc[~report["kept"], "reason"].str.startswith("correlated")).all()


def test_cluster_importance_dominates_solo_importance():
    features, target = _feature_set()
    model = Ridge(alpha=0.1).fit(features.to_numpy(), target)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return model.predict(np.asarray(frame, dtype=float))

    clusters = {"momentum": MOMENTUM, "independent": INDEPENDENT}
    importance = cluster_permutation_importance(
        predict, features, target, clusters, n_repeats=3, seed=7
    )

    assert importance.baseline == pytest.approx(r2_score_numpy(target, predict(features)))
    # Solo permutation of any one momentum clone is small: the other three
    # substitute for it.
    momentum_solo = importance.solo.reindex(MOMENTUM)
    assert momentum_solo.max() < 0.15
    # Shuffling the whole family destroys the information and shows up large.
    assert importance.cluster["momentum"] > 0.4
    assert importance.cluster["momentum"] > 5.0 * momentum_solo.max()
    assert importance.cluster["momentum"] > importance.cluster["independent"]
    # Independent features are individually more useful than any single clone.
    assert importance.solo[["ind_a", "ind_b"]].min() > momentum_solo.max()
    assert importance.ranking().index[0] == "momentum"

    table = importance.as_frame()
    assert list(table["cluster"])[0] == "momentum"
    assert (table["solo_is"] == "report-only").all()
    assert "largest cluster importance: momentum" in importance.summary()


def test_solo_permutation_importance_is_sorted_and_reproducible():
    features, target = _feature_set()
    model = Ridge(alpha=0.1).fit(features.to_numpy(), target)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return model.predict(np.asarray(frame, dtype=float))

    first = permutation_importance(predict, features, target, n_repeats=3, seed=1)
    second = permutation_importance(predict, features, target, n_repeats=3, seed=1)
    pd.testing.assert_series_equal(first, second)
    assert first.is_monotonic_decreasing
    assert set(first.index) == set(features.columns)


def test_cluster_importance_requires_disjoint_non_empty_clusters():
    features, target = _feature_set(n=50)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return frame.sum(axis=1).to_numpy()

    with pytest.raises(ValueError):
        cluster_permutation_importance(predict, features, target, {"empty": []})
    with pytest.raises(ValueError):
        cluster_permutation_importance(
            predict, features, target, {"a": ["mom_1"], "b": ["mom_1"]}
        )
    with pytest.raises(KeyError):
        cluster_permutation_importance(predict, features, target, {"a": ["nope"]})
