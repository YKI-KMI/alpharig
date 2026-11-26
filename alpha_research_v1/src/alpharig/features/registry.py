"""Versioned signal registry.

A signal is a pure function of :class:`~alpharig.features.inputs.SignalInputs`
plus a declaration of what it needs: how much history it consumes, whether it
must be lagged before use, and which family it belongs to.

Changing a registered signal's code without bumping its version raises
:class:`SignalVersionError`. That is intentional: an old result must always be
reproducible from ``(name, version)``, and silently editing a signal would make
every stored score a lie.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pandas as pd

from alpharig.features.inputs import SignalInputs


class SignalError(Exception):
    """Base class for signal registry problems."""


class SignalVersionError(SignalError):
    """Raised when a signal's code changes without a version bump."""


class SignalLookAheadError(SignalError):
    """Raised when a signal returns values beyond its observation date."""


class InsufficientHistoryError(SignalError):
    """Raised when a signal is asked for more history than the inputs hold."""


@dataclass(frozen=True)
class SignalSpec:
    """Declaration attached to one version of one signal.

    Attributes
    ----------
    name:
        Stable signal name, e.g. ``"momentum_12_1"``.
    version:
        Integer version. Any behavioural change requires a bump.
    func:
        ``SignalInputs -> wide DataFrame``.
    lookback:
        Number of trailing periods the signal consumes. Used by the purge
        logic in validation to drop training rows whose lookback window
        overlaps a test fold.
    family:
        Grouping used for hierarchical blending and cluster reports.
    description:
        One sentence a reader can act on.
    release_lag:
        Extra periods the signal must be shifted before it may be used. Zero
        means the inputs are already point-in-time.
    half_life:
        Optional decay half-life the signal was designed around.
    """

    name: str
    version: int
    func: Callable[[SignalInputs], pd.DataFrame] = field(repr=False, compare=False)
    lookback: int
    family: str = "other"
    description: str = ""
    release_lag: int = 0
    half_life: float | None = None

    @property
    def key(self) -> str:
        """``name@vN`` identifier."""
        return f"{self.name}@v{self.version}"

    def source_hash(self) -> str:
        """Hash of the function source, used to detect silent edits."""
        try:
            source = inspect.getsource(self.func)
        except (OSError, TypeError):  # pragma: no cover - interactive functions
            source = repr(self.func)
        return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


class SignalRegistry:
    """A registry of signal versions.

    Every ``(name, version)`` pair maps to exactly one implementation. Old
    versions are never removed, so a stored result can always name the version
    that produced it.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._specs: dict[tuple[str, int], SignalSpec] = {}
        self._hashes: dict[tuple[str, int], str] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, spec: SignalSpec) -> SignalSpec:
        """Register one signal version.

        Re-registering an identical implementation is a no-op, which keeps
        module reloads harmless.

        Raises
        ------
        SignalError
            If the declaration is malformed.
        SignalVersionError
            If the same ``(name, version)`` is registered with different code.
        """
        if not spec.name or spec.name.strip() != spec.name:
            raise SignalError("signal name must be non-empty and unpadded")
        if spec.version < 1:
            raise SignalError("signal version must be >= 1")
        if spec.lookback < 0:
            raise SignalError("lookback must be non-negative")
        if spec.release_lag < 0:
            raise SignalError("release_lag must be non-negative")
        digest = spec.source_hash()
        key = (spec.name, spec.version)
        if key in self._specs:
            if self._hashes[key] != digest:
                raise SignalVersionError(
                    f"signal {spec.key} changed implementation without a version "
                    "bump; bump the version so stored results stay reproducible"
                )
            return self._specs[key]
        self._specs[key] = spec
        self._hashes[key] = digest
        return spec

    def unregister(self, name: str, version: int) -> None:
        """Remove one version (test cleanup and explicit retirement only)."""
        key = (name, version)
        self._specs.pop(key, None)
        self._hashes.pop(key, None)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def versions(self, name: str) -> tuple[int, ...]:
        """Registered versions of ``name``, ascending."""
        return tuple(sorted(v for (n, v) in self._specs if n == name))

    def names(self) -> tuple[str, ...]:
        """Every registered signal name."""
        return tuple(sorted({name for (name, _) in self._specs}))

    def latest(self, name: str) -> SignalSpec:
        """Highest registered version of ``name``."""
        versions = self.versions(name)
        if not versions:
            raise KeyError(f"unknown signal {name!r}; registered: {list(self.names())}")
        return self._specs[(name, versions[-1])]

    def get(self, name: str, version: int | None = None) -> SignalSpec:
        """Look up a signal version; ``None`` means the latest."""
        if version is None:
            return self.latest(name)
        key = (name, version)
        if key not in self._specs:
            raise KeyError(
                f"unknown signal version {name!r} v{version}; available: "
                f"{list(self.versions(name))}"
            )
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[SignalSpec]:
        for key in sorted(self._specs):
            yield self._specs[key]

    def describe(self) -> pd.DataFrame:
        """One row per registered signal version."""
        rows = [
            {
                "signal": spec.name,
                "version": spec.version,
                "family": spec.family,
                "lookback": spec.lookback,
                "release_lag": spec.release_lag,
                "half_life": spec.half_life,
                "description": spec.description,
            }
            for spec in self
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def compute(
        self,
        name: str,
        inputs: SignalInputs,
        version: int | None = None,
    ) -> pd.DataFrame:
        """Evaluate a signal on point-in-time inputs.

        Parameters
        ----------
        name:
            Signal name.
        inputs:
            Inputs already truncated at the observation date.
        version:
            Version to run; defaults to the latest.

        Returns
        -------
        pd.DataFrame
            Wide frame reindexed to the inputs' dates and tickers.

        Raises
        ------
        InsufficientHistoryError
            If the inputs are shorter than the declared lookback.
        SignalLookAheadError
            If the signal returns a date the inputs never contained.
        SignalError
            If the signal does not return a ``DataFrame``.
        """
        spec = self.get(name, version)
        if spec.lookback > 0 and len(inputs.dates) <= spec.lookback:
            raise InsufficientHistoryError(
                f"{spec.key} needs more than {spec.lookback} periods of history; "
                f"inputs cover {len(inputs.dates)}"
            )
        raw = spec.func(inputs)
        if not isinstance(raw, pd.DataFrame):
            raise SignalError(f"{spec.key} must return a DataFrame, got {type(raw)!r}")
        if len(raw.index):
            latest = pd.Timestamp(pd.DatetimeIndex(pd.to_datetime(raw.index)).max())
            if latest > inputs.as_of:
                raise SignalLookAheadError(
                    f"{spec.key} returned values on {latest.date()}, beyond "
                    f"as_of={inputs.as_of.date()}"
                )
            extra = pd.DatetimeIndex(pd.to_datetime(raw.index)).difference(inputs.dates)
            if len(extra):
                raise SignalLookAheadError(
                    f"{spec.key} returned {len(extra)} date(s) not present in the "
                    "inputs; a signal may not invent observations"
                )
        aligned = raw.reindex(index=inputs.dates, columns=inputs.tickers).astype(float)
        if spec.release_lag:
            aligned = aligned.shift(spec.release_lag)
        return aligned


#: Process-wide registry used by :func:`signal`.
SIGNALS = SignalRegistry("alpharig")


def signal(
    *,
    name: str,
    lookback: int,
    version: int = 1,
    family: str = "other",
    description: str = "",
    release_lag: int = 0,
    half_life: float | None = None,
    registry: SignalRegistry | None = None,
) -> Callable[[Callable[[SignalInputs], pd.DataFrame]], Callable[[SignalInputs], pd.DataFrame]]:
    """Register a signal function.

    Parameters
    ----------
    name:
        Signal name.
    lookback:
        Trailing periods consumed.
    version:
        Version number; bump on any behavioural change.
    family:
        Signal family for blending and redundancy reports.
    description:
        One actionable sentence.
    release_lag:
        Periods to shift the output before use.
    half_life:
        Optional declared decay half-life.
    registry:
        Registry to populate; defaults to the process-wide :data:`SIGNALS`.

    Returns
    -------
    Callable
        The undecorated function, so it can still be called directly in tests.

    Examples
    --------
    >>> @signal(name="demo", lookback=2, family="reversal")
    ... def demo(inputs):
    ...     return -inputs.returns(1)
    """

    def decorator(
        func: Callable[[SignalInputs], pd.DataFrame],
    ) -> Callable[[SignalInputs], pd.DataFrame]:
        target = registry if registry is not None else SIGNALS
        target.register(
            SignalSpec(
                name=name,
                version=version,
                func=func,
                lookback=lookback,
                family=family,
                description=description,
                release_lag=release_lag,
                half_life=half_life,
            )
        )
        return func

    return decorator
