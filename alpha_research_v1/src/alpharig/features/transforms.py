"""Fold-safe panel transforms.

Two kinds of operations live here and they are kept deliberately separate:

* **Cross-sectional** transforms (z-score, rank, winsorise, neutralise) act
  across names on a single date. They are the ones that leak when their bounds
  are computed on the whole panel.
* **Time-series** transforms (lag, rolling, EWMA) act along dates for each
  name. They leak when they are centred or when they are applied before a
  point-in-time join.

Every transform inherits :class:`PanelTransformer`, whose ``fit`` requires a
:class:`~alpharig.features.fold.FoldGuard` and refuses a frame that reaches
outside the training window. Stateless transforms still implement ``fit`` (it
only validates the fold) so a pipeline can be fitted uniformly and audited
uniformly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from alpharig.features.fold import FoldGuard, FoldViolationError


class NotFittedError(RuntimeError):
    """Raised when a transformer is applied before being fitted."""


# ----------------------------------------------------------------------
# EWMA arithmetic
# ----------------------------------------------------------------------
def ewma_lambda(half_life: float) -> float:
    """Smoothing constant whose weight halves every ``half_life`` periods.

    ``lambda = 1 - 0.5 ** (1 / half_life)``, so an observation that is
    ``half_life`` periods old carries half the weight of the newest one.
    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 1.0 - 0.5 ** (1.0 / half_life)


def ewma_decay(half_life: float) -> float:
    """Per-period weight of an older observation: ``0.5 ** (1 / half_life)``."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 0.5 ** (1.0 / half_life)


def half_life_from_decay(decay: float) -> float:
    """Half-life implied by a per-period decay weight, ``ln(0.5) / ln(w)``.

    ``decay`` must lie strictly between 0 and 1; ``0.1`` gives 6.58 periods.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")
    return math.log(0.5) / math.log(decay)


# ----------------------------------------------------------------------
# base
# ----------------------------------------------------------------------
class PanelTransformer(ABC):
    """Base class for every transform in the feature engine.

    Subclasses implement ``_fit`` (learn parameters from training rows only)
    and ``_apply`` (transform any rows). The public methods enforce the fold
    rule.
    """

    #: Short name recorded in pipeline provenance.
    name: str = "transformer"

    def __init__(self) -> None:
        self.fitted_: bool = False
        self.fold_id_: int | None = None
        self._state: dict[str, Any] = {}

    # -- subclass hooks -------------------------------------------------
    @abstractmethod
    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        """Learn parameters from ``train`` (already restricted to the fold)."""

    @abstractmethod
    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transform."""

    # -- public API -----------------------------------------------------
    def fit(self, frame: pd.DataFrame, fold: FoldGuard | None) -> "PanelTransformer":
        """Fit on training rows only.

        Parameters
        ----------
        frame:
            Panel to learn from. Every date must belong to ``fold.train_dates``;
            use :meth:`FoldGuard.train_frame` to slice a full panel.
        fold:
            The split the training rows come from. ``None`` is rejected: a
            transform that has never seen a fold cannot be audited.

        Raises
        ------
        FoldViolationError
            If ``fold`` is ``None`` or ``frame`` extends outside the training
            window.
        """
        if fold is None:
            raise FoldViolationError(
                f"{self.name}.fit requires a FoldGuard. Fitting without a fold is "
                "exactly the leak this engine exists to prevent."
            )
        fold.assert_train_frame(frame.index, what=f"{self.name}.fit")
        if len(frame) == 0:
            raise FoldViolationError(f"{self.name}.fit received an empty frame")
        self._fit(frame, fold)
        self.fitted_ = True
        self.fold_id_ = fold.fold_id
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform to any rows (train or test)."""
        if not self.fitted_:
            raise NotFittedError(f"{self.name} must be fitted before transform()")
        return self._apply(frame)

    def fit_transform(self, frame: pd.DataFrame, fold: FoldGuard | None) -> pd.DataFrame:
        """Fit on ``frame`` and return its transformed values."""
        return self.fit(frame, fold).transform(frame)

    def state(self) -> dict[str, Any]:
        """Learned parameters, for run provenance."""
        return {
            "name": self.name,
            "fold_id": self.fold_id_,
            "state": dict(self._state),
        }

    # -- helpers --------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise NotFittedError(f"{self.name} must be fitted before transform()")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self._state})"


# ----------------------------------------------------------------------
# cross-sectional transforms
# ----------------------------------------------------------------------
class CrossSectionalZScore(PanelTransformer):
    """Per-date (x - mean) / std across names.

    Parameters
    ----------
    clip:
        Optional absolute cap on the score, applied after scaling.
    """

    name = "cs_zscore"

    def __init__(self, clip: float | None = None) -> None:
        super().__init__()
        if clip is not None and clip <= 0:
            raise ValueError("clip must be positive")
        self.clip = clip

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        self._state = {"clip": self.clip}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.astype(float)
        centered = values.sub(values.mean(axis=1), axis=0)
        scale = values.std(axis=1, ddof=0).replace(0.0, np.nan)
        scaled = centered.div(scale, axis=0).where(values.notna())
        return scaled.clip(-self.clip, self.clip) if self.clip else scaled


class CrossSectionalRank(PanelTransformer):
    """Per-date rank across names.

    Parameters
    ----------
    output:
        ``"pct"`` maps to ``(0, 1]``; ``"centered"`` maps to
        ``(0, 1] * 2 - 1``, i.e. roughly ``(-1, 1]`` and symmetric about zero.
    method:
        Pandas rank method; ties share the average rank by default.
    """

    name = "cs_rank"

    def __init__(
        self,
        output: Literal["pct", "centered"] = "pct",
        method: str = "average",
    ) -> None:
        super().__init__()
        if output not in {"pct", "centered"}:
            raise ValueError("output must be 'pct' or 'centered'")
        self.output = output
        self.method = method

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        self._state = {"output": self.output, "method": self.method}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        ranked = frame.astype(float).rank(axis=1, method=self.method, pct=True)
        if self.output == "centered":
            return ranked.mul(2.0).sub(1.0)
        return ranked


class Winsorizer(PanelTransformer):
    """Clip to quantile bounds fitted on training rows only.

    Bounds are pooled over the whole training window rather than estimated per
    date: a per-date bound is a statement about the cross-section on that date,
    and estimating it from dates the model may see is precisely the mistake
    this class prevents.
    """

    name = "winsorize"

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        super().__init__()
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("need 0 <= lower < upper <= 1")
        self.lower = lower
        self.upper = upper

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        values = train.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Winsorizer needs at least one finite value")
        low, high = np.quantile(finite, [self.lower, self.upper])
        self._state = {"lower": self.lower, "upper": self.upper,
                       "low_bound": float(low), "high_bound": float(high)}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return frame.astype(float).clip(
            self._state["low_bound"], self._state["high_bound"]
        )


class RobustScaler(PanelTransformer):
    """Per-name median / MAD scaling with parameters fitted in-fold.

    Median absolute deviation is scaled by 1.4826 so it estimates the standard
    deviation of a normal sample. When a name has zero dispersion in the
    training window the transform falls back to the sample standard deviation,
    and then to leaving the name untouched.
    """

    name = "robust_scale"

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        median = train.median(axis=0)
        mad = (train.sub(median, axis=1)).abs().median(axis=0) * 1.4826
        std = train.std(axis=0, ddof=0)
        scale = mad.where(mad > 0, std).where(lambda s: s > 0, 1.0)
        self._state = {"scale": scale.to_dict(), "center": median.to_dict()}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        center = pd.Series(self._state["center"], dtype=float)
        scale = pd.Series(self._state["scale"], dtype=float)
        scale = scale.replace(0.0, 1.0).fillna(1.0)
        return frame.astype(float).sub(center, axis=1).div(scale, axis=1)


class Neutralizer(PanelTransformer):
    """Remove cross-sectional exposure to declared risk factors.

    For each date the feature and the exposure columns are demeaned across
    names, and the residual ``y - X beta`` is kept. ``beta`` is a single pooled
    coefficient vector estimated on training dates only, so a test date cannot
    influence the neutralisation it is subjected to.

    Parameters
    ----------
    exposures:
        Mapping from exposure name (``"beta"``, ``"log_size"``, ``"sector_1"``,
        ...) to a wide frame aligned with the feature panel.

    Raises
    ------
    ValueError
        If no exposure is supplied or too few cross-sections are usable.
    """

    name = "neutralize"

    def __init__(self, exposures: Mapping[str, pd.DataFrame]) -> None:
        super().__init__()
        if not exposures:
            raise ValueError("Neutralizer needs at least one exposure")
        self.exposures = {str(k): v for k, v in exposures.items()}

    def _design(self, date: pd.Timestamp, columns: pd.Index) -> pd.DataFrame:
        parts = {}
        for key, frame in self.exposures.items():
            if date in frame.index:
                parts[key] = frame.loc[date].reindex(columns).astype(float)
            else:
                parts[key] = pd.Series(np.nan, index=columns)
        return pd.DataFrame(parts)

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        blocks: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        minimum = len(self.exposures) + 3
        for date in train.index:
            y = train.loc[date].astype(float)
            x = self._design(pd.Timestamp(date), train.columns)
            mask = y.notna() & x.notna().all(axis=1)
            if int(mask.sum()) < max(5, minimum):
                continue
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            blocks.append(xv - xv.mean(axis=0))
            targets.append(yv - yv.mean())
        if not blocks:
            raise ValueError(
                "Neutralizer found no usable cross-sections in the training window"
            )
        design = np.vstack(blocks)
        response = np.concatenate(targets)
        beta = np.linalg.lstsq(design, response, rcond=None)[0]
        self._state = {
            "exposures": list(self.exposures),
            "beta": {k: float(v) for k, v in zip(self.exposures, beta)},
            "n_observations": int(design.shape[0]),
        }

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        beta = np.array([self._state["beta"][k] for k in self.exposures], dtype=float)
        out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
        for date in frame.index:
            y = frame.loc[date].astype(float)
            x = self._design(pd.Timestamp(date), frame.columns)
            mask = y.notna() & x.notna().all(axis=1)
            if int(mask.sum()) < max(5, len(self.exposures) + 3):
                continue
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            residual = (yv - yv.mean()) - (xv - xv.mean(axis=0)) @ beta
            out.loc[date, mask[mask].index] = residual
        return out


# ----------------------------------------------------------------------
# time-series transforms
# ----------------------------------------------------------------------
class Lag(PanelTransformer):
    """Shift each name's series forward by ``periods`` (positive = older)."""

    name = "lag"

    def __init__(self, periods: int = 1) -> None:
        super().__init__()
        if periods < 0:
            raise ValueError("periods must be non-negative")
        self.periods = periods

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        self._state = {"periods": self.periods}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.astype(float).shift(self.periods)


class RollingStat(PanelTransformer):
    """Causal rolling statistic along dates, per name.

    Parameters
    ----------
    window:
        Lookback length in periods.
    stat:
        ``"mean"``, ``"std"``, ``"sum"`` or ``"min"``/``"max"``.
    min_periods:
        Minimum observations; defaults to half the window.
    """

    name = "rolling"

    def __init__(
        self,
        window: int,
        stat: Literal["mean", "std", "sum", "min", "max"] = "mean",
        min_periods: int | None = None,
    ) -> None:
        super().__init__()
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.stat = stat
        self.min_periods = min_periods if min_periods is not None else max(1, window // 2)

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        self._state = {"window": self.window, "stat": self.stat,
                       "min_periods": self.min_periods}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        rolling = frame.astype(float).rolling(self.window, min_periods=self.min_periods)
        if self.stat == "mean":
            return rolling.mean()
        if self.stat == "std":
            return rolling.std()
        if self.stat == "sum":
            return rolling.sum()
        if self.stat == "min":
            return rolling.min()
        return rolling.max()


class EwmaSmoother(PanelTransformer):
    """Exponentially weighted moving average, causal in time.

    ``lambda = 1 - 0.5 ** (1 / half_life)``. Gaps are treated as elapsed time
    (``ignore_na=False``), which is what a decaying signal actually does when a
    name stops trading and then resumes.

    Parameters
    ----------
    half_life:
        Weight half-life in periods.
    """

    name = "ewma"

    def __init__(self, half_life: float) -> None:
        super().__init__()
        self.half_life = half_life
        self.alpha = ewma_lambda(half_life)

    def _fit(self, train: pd.DataFrame, fold: FoldGuard) -> None:
        self._state = {"half_life": self.half_life, "alpha": self.alpha}

    def _apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.astype(float).ewm(alpha=self.alpha, adjust=False).mean()


# ----------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------
class FeaturePipeline:
    """A named sequence of transforms fitted fold by fold.

    Each transform is fitted on the output of the previous one, so a scale that
    follows a winsoriser sees winsorised training data, exactly as it will at
    scoring time.
    """

    def __init__(self, transforms: Sequence[PanelTransformer], name: str = "pipeline") -> None:
        if not transforms:
            raise ValueError("a pipeline needs at least one transform")
        self.transforms = list(transforms)
        self.name = name

    def fit(self, frame: pd.DataFrame, fold: FoldGuard | None) -> "FeaturePipeline":
        """Fit every transform on training rows of ``frame``."""
        if fold is None:
            raise FoldViolationError(f"{self.name}.fit requires a FoldGuard")
        current = fold.train_frame(frame)
        for transformer in self.transforms:
            transformer.fit(current, fold)
            current = transformer.transform(current)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted transform in order."""
        current = frame
        for transformer in self.transforms:
            current = transformer.transform(current)
        return current

    def fit_transform(self, frame: pd.DataFrame, fold: FoldGuard | None) -> pd.DataFrame:
        """Fit on training rows and transform the same rows."""
        if fold is None:
            raise FoldViolationError(f"{self.name}.fit requires a FoldGuard")
        return self.fit(fold.train_frame(frame), fold).transform(fold.train_frame(frame))

    def state(self) -> dict[str, Any]:
        """Provenance for every step."""
        return {"pipeline": self.name,
                "steps": [t.state() for t in self.transforms]}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = ", ".join(type(t).__name__ for t in self.transforms)
        return f"FeaturePipeline({names})"
