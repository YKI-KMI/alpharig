"""Signal library and feature engine.

Importing this package registers the starter signal library, so
``alpharig.features.SIGNALS`` is populated after a single import.
"""

from __future__ import annotations

from alpharig.features.fold import FoldGuard, FoldViolationError
from alpharig.features.importance import (
    ClusterImportance,
    cluster_permutation_importance,
    permutation_importance,
    r2_score_numpy,
)
from alpharig.features.inputs import (
    SignalInputs,
    build_signal_inputs,
    safe_log,
    trailing_adv,
)
from alpharig.features.interactions import (
    InteractionBuild,
    InteractionRegistry,
    InteractionSpec,
    UndeclaredInteractionError,
    all_pairs_count,
    effect_table,
    interaction_column,
    marginal_effect,
    zero_crossing_z,
)
from alpharig.features.redundancy import (
    ClusterReport,
    correlation_clusters,
    correlation_distance,
    prune_redundant,
    pruning_report,
)
from alpharig.features.registry import (
    SIGNALS,
    InsufficientHistoryError,
    SignalError,
    SignalLookAheadError,
    SignalRegistry,
    SignalSpec,
    SignalVersionError,
    signal,
)
from alpharig.features.transforms import (
    CrossSectionalRank,
    CrossSectionalZScore,
    EwmaSmoother,
    FeaturePipeline,
    Lag,
    Neutralizer,
    NotFittedError,
    PanelTransformer,
    RobustScaler,
    RollingStat,
    Winsorizer,
    ewma_decay,
    ewma_lambda,
    half_life_from_decay,
    half_life_from_lambda,
)
from alpharig.features import library as _library  # noqa: F401  (registers signals)

__all__ = [
    "SIGNALS",
    "ClusterImportance",
    "ClusterReport",
    "CrossSectionalRank",
    "CrossSectionalZScore",
    "EwmaSmoother",
    "FeaturePipeline",
    "FoldGuard",
    "FoldViolationError",
    "InsufficientHistoryError",
    "InteractionBuild",
    "InteractionRegistry",
    "InteractionSpec",
    "Lag",
    "Neutralizer",
    "NotFittedError",
    "PanelTransformer",
    "RobustScaler",
    "RollingStat",
    "SignalError",
    "SignalInputs",
    "SignalLookAheadError",
    "SignalRegistry",
    "SignalSpec",
    "SignalVersionError",
    "UndeclaredInteractionError",
    "Winsorizer",
    "all_pairs_count",
    "build_signal_inputs",
    "cluster_permutation_importance",
    "correlation_clusters",
    "correlation_distance",
    "effect_table",
    "ewma_decay",
    "ewma_lambda",
    "half_life_from_decay",
    "half_life_from_lambda",
    "interaction_column",
    "marginal_effect",
    "permutation_importance",
    "prune_redundant",
    "pruning_report",
    "r2_score_numpy",
    "safe_log",
    "signal",
    "trailing_adv",
    "zero_crossing_z",
]
