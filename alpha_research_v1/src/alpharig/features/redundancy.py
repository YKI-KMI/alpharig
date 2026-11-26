"""Redundancy pruning by correlation structure.

Forty momentum variants are one piece of information wearing forty hats. Before
any model sees the feature set, near-duplicates are grouped by correlation
distance and collapsed to one representative per group.

Pruning here is by **correlation structure only**. Importance never enters: a
feature is not dropped for scoring badly, because a weak feature can be the one
that carries the signal once another is conditioned on. Importance is measured
afterwards, at the cluster level (see :mod:`alpharig.features.importance`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class ClusterReport:
    """Result of clustering features by correlation distance.

    Attributes
    ----------
    clusters:
        Mapping from cluster label to member feature names, both sorted.
    correlations:
        The correlation matrix the clustering was computed from.
    threshold:
        Linkage distance at which the tree was cut. Distance
        ``sqrt(0.5 * (1 - rho))`` equals ``threshold`` at
        ``rho = 1 - 2 * threshold**2``; the default 0.5 corresponds to
        ``rho = 0.5``.
    """

    clusters: dict[int, list[str]]
    correlations: pd.DataFrame
    threshold: float

    @property
    def labels(self) -> pd.Series:
        """Series mapping feature name to cluster label."""
        return pd.Series(
            {
                feature: label
                for label, members in self.clusters.items()
                for feature in members
            },
            name="cluster",
        ).sort_index()

    def cluster_of(self, feature: str) -> int:
        """Cluster label containing ``feature``."""
        for label, members in self.clusters.items():
            if feature in members:
                return label
        raise KeyError(f"{feature!r} is not in any cluster")

    def multi_member(self) -> dict[int, list[str]]:
        """Only the clusters that actually contain redundancy."""
        return {k: v for k, v in self.clusters.items() if len(v) > 1}

    def summary(self) -> pd.DataFrame:
        """One row per cluster: label, size and members."""
        return pd.DataFrame(
            [
                {"cluster": label, "size": len(members), "members": ", ".join(members)}
                for label, members in sorted(self.clusters.items())
            ]
        )


def correlation_distance(correlations: pd.DataFrame) -> pd.DataFrame:
    """Correlation distance ``sqrt(0.5 * (1 - rho))``, clipped to be valid.

    Missing correlations are treated as zero (no evidence of redundancy), which
    is the conservative choice: it keeps features apart rather than merging them.
    """
    values = correlations.astype(float).fillna(0.0)
    np.fill_diagonal(values.to_numpy(), 1.0)
    values = values.clip(-1.0, 1.0)
    distance = np.sqrt(0.5 * (1.0 - values))
    np.fill_diagonal(distance.to_numpy(), 0.0)
    return distance


def correlation_clusters(
    features: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "average",
) -> ClusterReport:
    """Hierarchically cluster features by correlation distance.

    Parameters
    ----------
    features:
        Wide frame with one row per observation and one column per feature.
    threshold:
        Cut height in correlation-distance units; two features join when their
        distance is below it. 0.5 corresponds to a correlation of 0.5.
    method:
        SciPy linkage method; ``"average"`` is the default because it does not
        chain like single linkage.

    Returns
    -------
    ClusterReport
        Cluster assignment plus the correlation matrix used.

    Raises
    ------
    ValueError
        If ``features`` has no columns or the threshold is not positive.
    """
    if features.shape[1] == 0:
        raise ValueError("no features to cluster")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    names = [str(column) for column in features.columns]
    frame = features.copy()
    frame.columns = names

    if len(names) == 1:
        correlations = pd.DataFrame([[1.0]], index=names, columns=names)
        return ClusterReport({0: names}, correlations, threshold)

    correlations = frame.corr()
    distance = correlation_distance(correlations)
    condensed = squareform(distance.to_numpy(), checks=False)
    if not np.isfinite(condensed).all():  # pragma: no cover - defensive
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=1.0, neginf=0.0)
    tree = linkage(condensed, method=method)
    raw = fcluster(tree, t=threshold, criterion="distance")

    groups: dict[int, list[str]] = {}
    for name, label in zip(names, raw):
        groups.setdefault(int(label), []).append(str(name))
    ordered = sorted(groups.values(), key=lambda members: sorted(members)[0])
    clusters = {
        index: sorted(members) for index, members in enumerate(ordered)
    }
    return ClusterReport(clusters, correlations, threshold)


def medoid(report: ClusterReport, cluster: int) -> str:
    """Member of ``cluster`` closest to all the others (deterministic tie-break)."""
    members = sorted(report.clusters[cluster])
    if len(members) == 1:
        return members[0]
    distance = correlation_distance(report.correlations).loc[members, members]
    totals = distance.sum(axis=1)
    return str(min(members, key=lambda name: (float(totals[name]), name)))


def prune_redundant(
    features: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "average",
) -> list[str]:
    """Keep one representative per correlation cluster.

    Parameters
    ----------
    features:
        Wide feature frame.
    threshold:
        Correlation-distance cut, as in :func:`correlation_clusters`.
    method:
        Linkage method.

    Returns
    -------
    list[str]
        Retained feature names, in cluster order. The representative is the
        cluster medoid, so nothing is chosen for its correlation with the
        target and no feature is dropped for scoring badly.
    """
    report = correlation_clusters(features, threshold=threshold, method=method)
    return [medoid(report, cluster) for cluster in sorted(report.clusters)]


def pruning_report(
    features: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "average",
) -> pd.DataFrame:
    """Explain which features were kept, which were dropped, and why."""
    report = correlation_clusters(features, threshold=threshold, method=method)
    rows: list[dict[str, object]] = []
    for cluster in sorted(report.clusters):
        members = sorted(report.clusters[cluster])
        keep = medoid(report, cluster)
        for member in members:
            rows.append(
                {
                    "cluster": cluster,
                    "feature": member,
                    "kept": member == keep,
                    "representative": keep,
                    "cluster_size": len(members),
                    "reason": "representative"
                    if member == keep
                    else f"correlated with {keep} above threshold",
                }
            )
    return pd.DataFrame(rows)
