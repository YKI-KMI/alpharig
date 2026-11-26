"""Fold guards: the object that makes "fit on training data only" structural.

A transformer cannot know whether a caller was honest. So it does not try to
guess: :meth:`PanelTransformer.fit` demands a :class:`FoldGuard`, and the guard
rejects any frame that contains a date outside the fold's training window. A
full-panel fit raises; it does not silently subset and it does not warn.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class FoldViolationError(RuntimeError):
    """Raised when something is fitted, or evaluated, outside a training fold."""


@dataclass(frozen=True)
class FoldGuard:
    """One date-grouped train/test split, with fit-time assertions.

    Parameters
    ----------
    train_dates:
        Dates the model may learn from.
    test_dates:
        Dates held out for evaluation. Must be disjoint from ``train_dates``.
    fold_id:
        Identifier recorded on fitted transformers so a report can say which
        fold produced which parameters.

    Raises
    ------
    FoldViolationError
        If either split is empty, if they overlap, or if the splits are not
        date-indexed.
    """

    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    fold_id: int = 0

    def __post_init__(self) -> None:
        train = pd.DatetimeIndex(pd.to_datetime(self.train_dates)).unique().sort_values()
        test = pd.DatetimeIndex(pd.to_datetime(self.test_dates)).unique().sort_values()
        if len(train) == 0:
            raise FoldViolationError("a fold needs a non-empty training window")
        if len(test) == 0:
            raise FoldViolationError("a fold needs a non-empty test window")
        overlap = train.intersection(test)
        if len(overlap):
            raise FoldViolationError(
                f"train and test windows overlap on {len(overlap)} dates "
                f"(first: {overlap[0].date()}); that is a leak, not a fold"
            )
        object.__setattr__(self, "train_dates", train)
        object.__setattr__(self, "test_dates", test)

    # ------------------------------------------------------------------
    # assertions
    # ------------------------------------------------------------------
    def assert_train_frame(self, index: pd.Index, what: str = "fit") -> None:
        """Raise unless every date in ``index`` belongs to the training window.

        Parameters
        ----------
        index:
            Dates of the frame about to be fitted.
        what:
            Label used in the error message, e.g. ``"Winsorizer.fit"``.

        Raises
        ------
        FoldViolationError
            If any date is outside the training window.
        """
        dates = pd.DatetimeIndex(pd.to_datetime(index))
        outside = dates.difference(self.train_dates)
        if len(outside):
            preview = ", ".join(str(date.date()) for date in outside[:3])
            raise FoldViolationError(
                f"{what} received {len(outside)} date(s) outside the training "
                f"window of fold {self.fold_id} (first: {preview}). Fit on "
                "fold.train_frame(frame); fitting on a panel that spans the test "
                "window leaks the test period into the preprocessing."
            )

    def assert_test_frame(self, index: pd.Index, what: str = "evaluate") -> None:
        """Raise if an evaluation frame touches the training window."""
        dates = pd.DatetimeIndex(pd.to_datetime(index))
        inside = dates.intersection(self.train_dates)
        if len(inside):
            raise FoldViolationError(
                f"{what} received {len(inside)} training date(s); evaluation must "
                "happen on held-out dates only"
            )

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def train_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the rows of ``frame`` inside the training window, in order."""
        mask = frame.index.isin(self.train_dates)
        return frame.loc[mask]

    def test_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the rows of ``frame`` inside the test window, in order."""
        mask = frame.index.isin(self.test_dates)
        return frame.loc[mask]

    def describe(self) -> str:
        """One-line human summary."""
        return (
            f"fold {self.fold_id}: train {self.train_dates[0].date()}.."
            f"{self.train_dates[-1].date()} ({len(self.train_dates)} dates), "
            f"test {self.test_dates[0].date()}..{self.test_dates[-1].date()} "
            f"({len(self.test_dates)} dates)"
        )
