"""Cluster permutation importance.

Solo permutation importance is misleading exactly when features are correlated.
Permute one of four near-identical momentum columns and the model substitutes
the other three, so the column looks worthless while the information it carries
is essential. Permuting the whole cluster removes the information, and the
score collapse is what should be attributed to that family of features.

Both numbers are reported. Only the cluster number is allowed to influence a
decision; ``ClusterImportance.as_frame`` labels every solo column as
"report-only" so a table cannot be quoted out of context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

Metric = Callable[[np.ndarray, np.ndarray], float]


def r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, NaN when the target has no variance."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    residual = float(np.sum((truth - prediction) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def _as_frame(frame: pd.DataFrame | np.ndarray, columns: Sequence[str] | None) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.astype(float)
    array = np.asarray(frame, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    names = list(columns) if columns is not None else [f"x{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=names)


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | np.ndarray,
    features: Sequence[str] | None = None,
    n_repeats: int = 5,
    metric: Metric = r2_score_numpy,
    seed: int = 0,
) -> pd.Series:
    """Solo permutation importance for each feature.

    Parameters
    ----------
    predict:
        Callable taking a feature frame and returning predictions.
    X:
        Feature matrix.
    y:
        Target vector.
    features:
        Subset of columns to score; defaults to all.
    n_repeats:
        Shuffles per feature.
    metric:
        Higher-is-better score, R-squared by default.
    seed:
        Seed for the shuffle order.

    Returns
    -------
    pd.Series
        Mean score drop per feature, sorted descending.
    """
    frame = _as_frame(X, features)
    target = np.asarray(y, dtype=float)
    columns = list(features) if features is not None else list(frame.columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"features not present in X: {missing}")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    baseline = metric(target, np.asarray(predict(frame), dtype=float))
    rng = np.random.default_rng(seed)
    drops: dict[str, float] = {}
    for column in columns:
        values = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = metric(target, np.asarray(predict(shuffled), dtype=float))
            values.append(baseline - score)
        drops[column] = float(np.mean(values))
    return pd.Series(drops, name="importance").sort_values(ascending=False)


@dataclass(frozen=True)
class ClusterImportance:
    """Cluster-level permutation importance with solo values alongside.

    Attributes
    ----------
    baseline:
        Score with no columns shuffled.
    solo:
        Mean score drop when one column is shuffled.
    cluster:
        Mean score drop when every column of a cluster is shuffled together.
    clusters:
        The grouping that was used.
    n_repeats:
        Shuffles per cluster.
    """

    baseline: float
    solo: pd.Series
    cluster: pd.Series
    clusters: dict[str, list[str]]
    n_repeats: int

    def ranking(self) -> pd.Series:
        """Cluster importance, largest first."""
        return self.cluster.sort_values(ascending=False)

    def as_frame(self) -> pd.DataFrame:
        """Table with the cluster number and the report-only solo mean."""
        rows = []
        for label, members in self.clusters.items():
            solo_mean = float(self.solo.reindex(members).mean())
            rows.append(
                {
                    "cluster": label,
                    "importance": float(self.cluster[label]),
                    "n_features": len(members),
                    "mean_solo_importance": solo_mean,
                    "solo_is": "report-only",
                    "members": ", ".join(members),
                }
            )
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    def summary(self) -> str:
        """One-line headline naming the most important cluster."""
        if self.cluster.empty:
            return "no clusters scored"
        top = self.ranking().index[0]
        return (
            f"largest cluster importance: {top} "
            f"({self.cluster[top]:.3f} R2 drop over {len(self.clusters[top])} "
            f"features; baseline {self.baseline:.3f})"
        )


def cluster_permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | np.ndarray,
    clusters: Mapping[str, Sequence[str]],
    n_repeats: int = 5,
    metric: Metric = r2_score_numpy,
    seed: int = 0,
) -> ClusterImportance:
    """Out-of-sample-style cluster permutation (MDA) with solo context.

    Every column of a cluster is shuffled with the *same* row permutation, so
    the dependence inside the cluster is preserved and only its information
    about the target is destroyed. Shuffling the columns independently would
    destroy the cluster's internal structure too and overstate its importance.

    Parameters
    ----------
    predict:
        Callable taking a feature frame and returning predictions.
    X:
        Feature matrix.
    y:
        Target vector.
    clusters:
        Mapping from cluster label to member columns.
    n_repeats:
        Shuffles per cluster.
    metric:
        Higher-is-better score.
    seed:
        Seed for the shuffle order.

    Returns
    -------
    ClusterImportance
        Cluster and solo importances sharing one baseline.

    Raises
    ------
    ValueError
        If a cluster is empty or duplicates a column across clusters.
    """
    frame = _as_frame(X, None)
    target = np.asarray(y, dtype=float)
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    normalised: dict[str, list[str]] = {}
    seen: set[str] = set()
    for label, members in clusters.items():
        columns = [str(name) for name in members]
        if not columns:
            raise ValueError(f"cluster {label!r} is empty")
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise KeyError(f"cluster {label!r} names absent columns: {missing}")
        duplicated = seen.intersection(columns)
        if duplicated:
            raise ValueError(f"columns appear in more than one cluster: {sorted(duplicated)}")
        seen.update(columns)
        normalised[str(label)] = columns

    baseline = metric(target, np.asarray(predict(frame), dtype=float))
    rng = np.random.default_rng(seed)

    solo_drops: dict[str, float] = {}
    for column in sorted(seen):
        values = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            values.append(baseline - metric(target, np.asarray(predict(shuffled), dtype=float)))
        solo_drops[column] = float(np.mean(values))

    cluster_drops: dict[str, float] = {}
    for label, columns in normalised.items():
        values = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            order = rng.permutation(len(shuffled))
            block = shuffled[columns].to_numpy()
            shuffled[columns] = block[order]
            values.append(baseline - metric(target, np.asarray(predict(shuffled), dtype=float)))
        cluster_drops[label] = float(np.mean(values))

    return ClusterImportance(
        baseline=float(baseline),
        solo=pd.Series(solo_drops, name="solo_importance").sort_values(ascending=False),
        cluster=pd.Series(cluster_drops, name="cluster_importance").sort_values(ascending=False),
        clusters=normalised,
        n_repeats=n_repeats,
    )
