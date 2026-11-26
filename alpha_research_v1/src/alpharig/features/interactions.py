"""Declared interactions only.

Interactions are where feature engineering stops being cheap. Crossing ``k``
features pairwise creates ``k(k-1)/2`` columns; at 40 features that is 780
candidates searched on the same data, and the best of 780 noise columns looks
impressive. The registry therefore builds nothing that was not declared with a
written rationale, and it reports how many columns the all-pairs alternative
would have created.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd


class UndeclaredInteractionError(ValueError):
    """Raised when an interaction is requested that was never declared."""


@dataclass(frozen=True)
class InteractionSpec:
    """One declared ``feature x conditioner`` pair.

    Attributes
    ----------
    feature:
        The signal being conditioned, e.g. ``"momentum_12_1"``.
    conditioner:
        The variable that changes the feature's meaning, e.g. ``"volatility"``.
    rationale:
        Why this pair is expected to matter. Empty rationales are rejected:
        an interaction without a mechanism is a search, not a hypothesis.
    version:
        Version of the declaration.
    """

    feature: str
    conditioner: str
    rationale: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.rationale or not self.rationale.strip():
            raise ValueError(
                f"interaction {self.column} needs a written rationale; an "
                "interaction without a mechanism is just another trial"
            )
        if self.feature == self.conditioner:
            raise ValueError("an interaction needs two different variables")

    @property
    def column(self) -> str:
        """Output column name, ``x__feature__conditioner``."""
        return interaction_column(self.feature, self.conditioner)

    def describe(self) -> str:
        """Human-readable declaration."""
        return f"{self.column}: {self.rationale}"


def interaction_column(feature: str, conditioner: str) -> str:
    """Canonical name for a product feature."""
    return f"x__{feature}__{conditioner}"


def all_pairs_count(n_features: int) -> int:
    """Number of unordered pairs among ``n_features``: ``C(n, 2)``."""
    if n_features < 0:
        raise ValueError("n_features must be non-negative")
    return math.comb(n_features, 2) if n_features >= 2 else 0


@dataclass
class InteractionBuild:
    """Result of building the declared interaction columns.

    Attributes
    ----------
    frame:
        Wide panel of product features, one column per declared pair. Empty
        (zero columns) when nothing is declared.
    declared:
        Names of the pairs that were built.
    all_pairs:
        Columns the all-pairs crossing would have produced.
    suppressed:
        Columns the registry refused to build.
    warnings:
        Plain-language reasons, including the all-pairs count.
    """

    frame: pd.DataFrame
    declared: list[str] = field(default_factory=list)
    all_pairs: int = 0
    suppressed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_interactions(self) -> int:
        """Number of built interaction columns."""
        return int(self.frame.shape[1])

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly summary for a run record."""
        return {
            "declared": list(self.declared),
            "n_interactions": self.n_interactions,
            "all_pairs_would_create": self.all_pairs,
            "suppressed": self.suppressed,
            "warnings": list(self.warnings),
        }


class InteractionRegistry:
    """Registry of declared interactions.

    Nothing is built unless it was declared here with a rationale. Building from
    a feature dictionary that contains undeclared pairs emits a warning naming
    the count the all-pairs alternative would create, and returns only the
    declared columns.
    """

    def __init__(self) -> None:
        self._specs: dict[str, InteractionSpec] = {}

    def declare(
        self,
        feature: str,
        conditioner: str,
        rationale: str,
        version: int = 1,
    ) -> InteractionSpec:
        """Declare one interaction.

        Raises
        ------
        ValueError
            If the rationale is empty or both sides name the same variable.
        """
        spec = InteractionSpec(
            feature=feature, conditioner=conditioner, rationale=rationale, version=version
        )
        self._specs[spec.column] = spec
        return spec

    def declared(self) -> tuple[InteractionSpec, ...]:
        """Every declared interaction, sorted by column name."""
        return tuple(self._specs[key] for key in sorted(self._specs))

    def __len__(self) -> int:
        return len(self._specs)

    def build(
        self,
        features: Mapping[str, pd.DataFrame],
        require_declaration: bool = True,
    ) -> InteractionBuild:
        """Build the declared interaction columns.

        Parameters
        ----------
        features:
            Available feature panels keyed by name.
        require_declaration:
            When true (the default) undeclared pairs are silently *not* built
            and the omission is reported; set it to false only for diagnostics.

        Returns
        -------
        InteractionBuild
            Built columns plus the warning trail.

        Raises
        ------
        UndeclaredInteractionError
            If ``require_declaration`` is false and an undeclared pair is found.
            (Kept for the explicit diagnostic path; the default path reports
            instead of raising so a batch job is not silently widened.)
        """
        n_features = len(features)
        possible = all_pairs_count(n_features)
        warnings: list[str] = []
        built: dict[str, pd.DataFrame] = {}

        for spec in self.declared():
            missing = [
                name for name in (spec.feature, spec.conditioner)
                if name not in features
            ]
            if missing:
                warnings.append(
                    f"declared interaction {spec.column} skipped: missing "
                    f"feature(s) {missing}"
                )
                continue
            left = features[spec.feature].astype(float)
            right = features[spec.conditioner].astype(float)
            index = left.index.union(right.index)
            columns = left.columns.union(right.columns)
            product = left.reindex(index=index, columns=columns) * right.reindex(
                index=index, columns=columns
            )
            built[spec.column] = product

        suppressed = possible - len(built)
        if not self._specs and possible:
            warnings.append(
                f"{n_features} features with no declared interaction: building "
                f"all pairs would create {possible} interaction columns "
                f"(C({n_features}, 2)); refusing to cross them. Declare the "
                "pairs you can justify instead."
            )
        elif suppressed > 0:
            warnings.append(
                f"{len(built)} declared interaction(s); all-pairs crossing would "
                f"have created {possible}, so {suppressed} undeclared pair(s) "
                "were suppressed."
            )

        if require_declaration is False:
            declared_pairs = {
                (spec.feature, spec.conditioner) for spec in self.declared()
            } | {(spec.conditioner, spec.feature) for spec in self.declared()}
            undeclared = [
                (a, b)
                for i, a in enumerate(sorted(features))
                for b in sorted(features)[i + 1:]
                if (a, b) not in declared_pairs
            ]
            if undeclared:
                raise UndeclaredInteractionError(
                    f"{len(undeclared)} undeclared pair(s) present, e.g. "
                    f"{undeclared[0]}; all-pairs crossing is not allowed"
                )

        if built:
            frame = pd.DataFrame(built).sort_index()
        else:
            index = next(iter(features.values())).index if features else pd.Index([])
            frame = pd.DataFrame(index=index)

        return InteractionBuild(
            frame=frame,
            declared=list(built),
            all_pairs=possible,
            suppressed=suppressed,
            warnings=warnings,
        )


def marginal_effect(beta1: float, beta3: float, z: float) -> float:
    """Slope of ``beta0 + beta1*f + beta3*f*z`` with respect to ``f``.

    The effect of the feature at conditioner value ``z``: ``beta1 + beta3*z``.
    """
    return beta1 + beta3 * z


def zero_crossing_z(beta1: float, beta3: float) -> float | None:
    """Conditioner value at which the feature's effect changes sign.

    Solves ``beta1 + beta3*z = 0``. Returns ``None`` when ``beta3`` is zero,
    i.e. when the effect never changes sign.
    """
    if beta3 == 0:
        return None
    return -beta1 / beta3


def effect_table(
    beta1: float,
    beta3: float,
    conditioner: Sequence[float] = (0.0, 0.5, 1.0),
) -> pd.DataFrame:
    """Tabulate the marginal effect across conditioner values."""
    return pd.DataFrame(
        {
            "conditioner": list(conditioner),
            "effect": [marginal_effect(beta1, beta3, z) for z in conditioner],
        }
    )
