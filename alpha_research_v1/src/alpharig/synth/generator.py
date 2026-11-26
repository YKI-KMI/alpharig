"""Synthetic equity panel generator with planted, auditable structure.

The generator is the project's test oracle. It produces a point-in-time panel
whose within-date correlation, signal IC, signal decay, regime interaction,
fundamental release lags, delistings and fat tails are all known by
construction and reported in :class:`SyntheticTruth`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from alpharig.data.calendar import TradingCalendar
from alpharig.data.schema import FUNDAMENTAL_COLUMNS, MEMBERSHIP_COLUMNS, RESTRICTION_COLUMNS
from alpharig.data.store import PointInTimeStore
from alpharig.data.universe import PointInTimeUniverse, UniverseRules
from alpharig.stats.effective import effective_sample_size
from alpharig.stats.panel import (
    cross_sectional_correlation,
    mean_pairwise_correlation,
)

DEFAULT_RELEASE_LAGS: dict[str, int] = {
    "eps": 45,
    "revenue": 40,
    "book_value": 60,
    "net_income": 45,
}


class PanelSpec(BaseModel):
    """Configuration of one synthetic panel.

    The model is frozen and rejects unknown keys so a typo cannot silently
    change the experiment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_stocks: int = Field(200, ge=2, le=5000)
    n_periods: int = Field(500, ge=60, le=20000)
    start: date = date(2015, 1, 2)
    freq: Literal["D", "M"] = "D"
    seed: int = 0
    n_sectors: int = Field(10, ge=1, le=100)

    rho: float = Field(0.25, ge=0.0, lt=0.95, description="Target mean within-date correlation.")
    annual_vol: float = Field(0.30, gt=0.0, le=5.0)
    sector_vol_share: float = Field(0.35, ge=0.0, le=1.0)
    beta_dispersion: float = Field(0.0, ge=0.0, le=1.0)
    fat_tail_df: float = Field(5.0, gt=2.5, le=1000.0)

    signal_ic: float = Field(0.04, ge=0.0, lt=0.5, description="Planted cross-sectional Pearson IC at lag 1.")
    signal_half_life: float = Field(20.0, gt=0.0, le=1000.0)
    signal_max_lag: int = Field(60, ge=1, le=1000)
    interaction: bool = Field(False, description="Flip the signal sign in the high-volatility regime.")
    interaction_strength: float = Field(1.0, ge=0.0, le=5.0)
    regime_stay: float = Field(0.97, gt=0.0, lt=1.0)
    regime_vol_mult: float = Field(2.0, ge=1.0, le=10.0)

    delist_fraction: float = Field(0.05, ge=0.0, le=0.8)
    delisting_return: float = Field(-0.30, ge=-1.0, le=0.0)
    restricted_fraction: float = Field(0.05, ge=0.0, le=0.5)
    ipo_fraction: float = Field(0.0, ge=0.0, le=0.5)

    fundamental_period_days: int = Field(91, ge=30, le=365)
    release_lag_days: dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_RELEASE_LAGS)
    )
    release_lag_jitter_days: int = Field(5, ge=0, le=60)

    @field_validator("release_lag_days")
    @classmethod
    def _validate_lags(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("release_lag_days must not be empty")
        if any(lag < 0 for lag in value.values()):
            raise ValueError("release lags must be non-negative")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PanelSpec":
        """Load and validate a YAML panel specification."""
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{source} must contain a YAML mapping")
        return cls(**payload)


@dataclass
class SyntheticTruth:
    """Everything the oracle knows about the generated panel."""

    spec: PanelSpec
    rho_target: float
    rho_expected: float
    rho_realized: float
    signal_ic_target: float
    signal_ic_realized: float
    signal_rank_ic_realized: float
    signal_half_life: float
    n_eff: float
    realized_kurtosis: float
    ic_low_vol: float
    ic_high_vol: float
    release_lags: dict[str, int]
    betas: pd.Series
    sectors: pd.Series
    signal_scores: pd.DataFrame
    regimes: pd.DataFrame
    delistings: pd.DataFrame
    restrictions: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable ground truth summary (frames are stored separately)."""
        return {
            "spec": self.spec.model_dump(mode="json"),
            "rho_target": self.rho_target,
            "rho_expected": self.rho_expected,
            "rho_realized": self.rho_realized,
            "signal_ic_target": self.signal_ic_target,
            "signal_ic_realized": self.signal_ic_realized,
            "signal_rank_ic_realized": self.signal_rank_ic_realized,
            "signal_half_life": self.signal_half_life,
            "n_eff": self.n_eff,
            "realized_kurtosis": self.realized_kurtosis,
            "ic_low_vol": self.ic_low_vol,
            "ic_high_vol": self.ic_high_vol,
            "interaction": self.spec.interaction,
            "release_lags": dict(self.release_lags),
            "research_only": True,
        }


@dataclass
class SyntheticPanel:
    """A generated panel plus its point-in-time plumbing and ground truth."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    fundamentals: pd.DataFrame
    membership: pd.DataFrame
    restrictions: pd.DataFrame
    truth: SyntheticTruth
    calendar: TradingCalendar

    def store(self) -> PointInTimeStore:
        """Point-in-time fundamental store over this panel."""
        return PointInTimeStore(self.fundamentals)

    def universe(self, rules: UniverseRules | None = None) -> PointInTimeUniverse:
        """Point-in-time universe over this panel."""
        return PointInTimeUniverse(
            self.membership, self.prices, self.restrictions, rules
        )

    def price_pivot(self) -> pd.DataFrame:
        """Wide close-price matrix (dates x tickers)."""
        return (
            self.prices.pivot(index="date", columns="ticker", values="close")
            .sort_index()
        )

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Persist the panel as Parquet tables plus a JSON truth summary."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames = {
            "prices": self.prices,
            "returns": self.returns,
            "fundamentals": self.fundamentals,
            "membership": self.membership,
            "restrictions": self.restrictions,
            "signals": self.truth.signal_scores,
            "regimes": self.truth.regimes,
            "delistings": self.truth.delistings,
            "betas": pd.DataFrame(
                {
                    "ticker": self.truth.betas.index,
                    "beta": self.truth.betas.to_numpy(),
                    "sector": self.truth.sectors.to_numpy(),
                }
            ),
        }
        paths: dict[str, Path] = {}
        for name, frame in frames.items():
            path = out / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            paths[name] = path
        truth_path = out / "truth.json"
        truth_path.write_text(json.dumps(self.truth.summary(), indent=2), encoding="utf-8")
        paths["truth"] = truth_path
        return paths

    @classmethod
    def read(cls, out_dir: str | Path) -> "SyntheticPanel":
        """Load a panel previously written by :meth:`write`."""
        out = Path(out_dir)
        payload = json.loads((out / "truth.json").read_text(encoding="utf-8"))
        spec = PanelSpec(**payload["spec"])
        prices = pd.read_parquet(out / "prices.parquet")
        returns = pd.read_parquet(out / "returns.parquet")
        fundamentals = pd.read_parquet(out / "fundamentals.parquet")
        membership = pd.read_parquet(out / "membership.parquet")
        restrictions = pd.read_parquet(out / "restrictions.parquet")
        signals = pd.read_parquet(out / "signals.parquet")
        regimes = pd.read_parquet(out / "regimes.parquet")
        delistings = pd.read_parquet(out / "delistings.parquet")
        betas_frame = pd.read_parquet(out / "betas.parquet")
        betas = betas_frame.set_index("ticker")["beta"].astype(float)
        sectors = betas_frame.set_index("ticker")["sector"].astype(int)
        truth = SyntheticTruth(
            spec=spec,
            rho_target=float(payload["rho_target"]),
            rho_expected=float(payload["rho_expected"]),
            rho_realized=float(payload["rho_realized"]),
            signal_ic_target=float(payload["signal_ic_target"]),
            signal_ic_realized=float(payload["signal_ic_realized"]),
            signal_rank_ic_realized=float(payload["signal_rank_ic_realized"]),
            signal_half_life=float(payload["signal_half_life"]),
            n_eff=float(payload["n_eff"]),
            realized_kurtosis=float(payload["realized_kurtosis"]),
            ic_low_vol=float(payload["ic_low_vol"]),
            ic_high_vol=float(payload["ic_high_vol"]),
            release_lags={k: int(v) for k, v in payload["release_lags"].items()},
            betas=betas,
            sectors=sectors,
            signal_scores=signals,
            regimes=regimes,
            delistings=delistings,
            restrictions=restrictions,
        )
        return cls(
            prices=prices,
            returns=returns,
            fundamentals=fundamentals,
            membership=membership,
            restrictions=restrictions,
            truth=truth,
            calendar=TradingCalendar(prices["date"].unique()),
        )


def _standardize_cross_section(values: np.ndarray) -> np.ndarray:
    """Demean and scale each row to unit population standard deviation."""
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale == 0.0] = 1.0
    return centered / scale


class SyntheticPanelGenerator:
    """Generate a synthetic panel from a :class:`PanelSpec`.

    The return decomposition is

    ``r_t = beta_i * market_t + sector_t + cross_section_t``

    where the cross-section carries a decayed planted signal and fat-tailed
    noise. Common-factor variance is calibrated so the mean pairwise
    within-date correlation matches ``spec.rho``.
    """

    def __init__(self, spec: PanelSpec) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)

    # ------------------------------------------------------------------
    # date grid
    # ------------------------------------------------------------------
    def _date_grid(self) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        spec = self.spec
        burn_periods = spec.signal_max_lag + 1
        if spec.freq == "D":
            out = pd.bdate_range(spec.start, periods=spec.n_periods)
            burn = pd.bdate_range(end=out[0], periods=burn_periods)[:-1]
        else:
            out = pd.date_range(spec.start, periods=spec.n_periods, freq="BME")
            burn = pd.date_range(end=out[0], periods=burn_periods, freq="BME")[:-1]
        return out, burn, burn.append(out)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(self) -> SyntheticPanel:
        """Generate the panel and compute realised oracle statistics."""
        spec = self.spec
        rng = self.rng
        out_dates, _, full_dates = self._date_grid()
        n_periods_full = len(full_dates)
        n_dates = len(out_dates)
        n_stocks = spec.n_stocks
        tickers = np.array([f"S{i:04d}" for i in range(n_stocks)])
        sectors = np.arange(n_stocks) % spec.n_sectors
        rng.shuffle(sectors)
        betas = np.clip(1.0 + spec.beta_dispersion * rng.standard_normal(n_stocks), 0.1, None)

        # --- latent volatility regime (causal Markov chain) ---------------
        high_vol = np.zeros(n_periods_full, dtype=bool)
        for t in range(1, n_periods_full):
            probability = spec.regime_stay if high_vol[t - 1] else 1.0 - spec.regime_stay
            high_vol[t] = rng.random() < probability
        vol_mult = np.where(high_vol, spec.regime_vol_mult, 1.0)

        # --- variance calibration -----------------------------------------
        periods_per_year = 252 if spec.freq == "D" else 12
        total_var = (spec.annual_vol / math.sqrt(periods_per_year)) ** 2
        counts = np.bincount(sectors, minlength=spec.n_sectors).astype(float)
        p_same_sector = float(np.sum(counts * (counts - 1)) / (n_stocks * (n_stocks - 1)))
        mean_beta_product = float(
            (betas.sum() ** 2 - (betas**2).sum()) / (n_stocks * (n_stocks - 1))
        )
        denominator = (
            (1.0 - spec.sector_vol_share) * mean_beta_product
            + spec.sector_vol_share * p_same_sector
        )
        if denominator <= 0.0:
            raise ValueError("sector configuration leaves no common-factor variance")
        common_scale = spec.rho * total_var / denominator
        var_market = (1.0 - spec.sector_vol_share) * common_scale
        var_sector = spec.sector_vol_share * common_scale
        mean_beta_sq = float(np.mean(betas**2))
        cross_var = total_var - (var_market * mean_beta_sq + var_sector)
        if cross_var <= 0.0:
            raise ValueError(
                "rho and sector_vol_share leave no idiosyncratic variance; "
                "reduce rho or sector_vol_share"
            )

        # --- common factors ----------------------------------------------
        market = rng.standard_normal(n_periods_full) * math.sqrt(var_market)
        sector_shocks = (
            rng.standard_normal((spec.n_sectors, n_periods_full)) * math.sqrt(var_sector)
        )
        common = (betas[:, None] * market[None, :] + sector_shocks[sectors, :]).T

        # --- planted signal ----------------------------------------------
        max_lag = spec.signal_max_lag
        decay = 0.5 ** (1.0 / spec.signal_half_life)
        weights = decay ** np.arange(max_lag)
        scores = _standardize_cross_section(
            rng.standard_normal((n_periods_full, n_stocks))
        )
        tilt = np.zeros_like(scores)
        for lag in range(max_lag):
            tilt[lag:] += weights[lag] * scores[: n_periods_full - lag]
        weight_square_sum = float(np.sum(weights**2))
        signal_amplitude = spec.signal_ic * math.sqrt(cross_var)
        signal_variance = signal_amplitude**2 * weight_square_sum
        noise_variance = cross_var - signal_variance
        if noise_variance <= 1e-12:
            raise ValueError(
                "signal_ic and signal_half_life imply more signal variance than "
                "the panel's idiosyncratic variance; reduce signal_ic or half-life"
            )

        degrees = spec.fat_tail_df
        innovations = rng.standard_t(degrees, size=(n_periods_full, n_stocks)) * math.sqrt(
            (degrees - 2.0) / degrees
        )
        innovations = _standardize_cross_section(innovations)

        if spec.interaction:
            flip = np.where(high_vol, -spec.interaction_strength, 1.0)
        else:
            flip = np.ones(n_periods_full)
        cross_section = (
            signal_amplitude * tilt * flip[:, None]
            + math.sqrt(noise_variance) * innovations
        )
        returns_full = (common + cross_section) * vol_mult[:, None]

        # --- slice to the output window ----------------------------------
        returns = returns_full[-n_dates:].copy()
        scores_out = scores[-n_dates:]
        high_vol_out = high_vol[-n_dates:]
        vol_mult_out = vol_mult[-n_dates:]

        # --- delistings and IPOs -----------------------------------------
        end_pos = np.full(n_stocks, n_dates - 1, dtype=int)
        n_delisted = int(round(spec.delist_fraction * n_stocks))
        delisted = np.zeros(n_stocks, dtype=bool)
        if n_delisted:
            delist_index = rng.choice(n_stocks, size=n_delisted, replace=False)
            end_pos[delist_index] = rng.integers(
                max(1, int(0.35 * n_dates)), n_dates, size=n_delisted
            )
            delisted[delist_index] = True
            returns[end_pos[delist_index], delist_index] += spec.delisting_return

        start_pos = np.zeros(n_stocks, dtype=int)
        n_ipo = int(round(spec.ipo_fraction * n_stocks))
        if n_ipo:
            ipo_index = rng.choice(n_stocks, size=n_ipo, replace=False)
            start_pos[ipo_index] = rng.integers(
                1, max(2, int(0.3 * n_dates)), size=n_ipo
            )
            start_pos = np.minimum(start_pos, end_pos - 1)

        # --- prices -------------------------------------------------------
        close = np.full((n_dates, n_stocks), np.nan)
        level = np.full(n_stocks, 100.0)
        for t in range(n_dates):
            active = (t >= start_pos) & (t <= end_pos)
            level = np.where(active, level * (1.0 + returns[t]), level)
            close[t] = np.where(active, level, np.nan)
        returns[~np.isfinite(close)] = np.nan

        base_dollar_volume = np.exp(rng.normal(13.5, 0.9, n_stocks))
        dollar_volume = (
            base_dollar_volume[None, :]
            * np.exp(rng.normal(0.0, 0.35, (n_dates, n_stocks)))
            * vol_mult_out[:, None]
        )
        volume = np.where(np.isfinite(close), dollar_volume / close, np.nan)

        price_wide = pd.DataFrame(close, index=out_dates, columns=tickers)
        volume_wide = pd.DataFrame(volume, index=out_dates, columns=tickers)
        prices = (
            price_wide.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="close")
            .dropna(subset=["close"])
        )
        volumes = (
            volume_wide.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="volume")
        )
        prices = prices.merge(volumes, on=["date", "ticker"], how="left")
        prices["dollar_volume"] = prices["close"] * prices["volume"]
        prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

        returns_wide = pd.DataFrame(returns, index=out_dates, columns=tickers)
        returns_long = (
            returns_wide.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="ret")
            .dropna(subset=["ret"])
        )
        returns_long["high_vol"] = returns_long["date"].map(
            dict(zip(out_dates, high_vol_out))
        )
        returns_long["vol_mult"] = returns_long["date"].map(
            dict(zip(out_dates, vol_mult_out))
        )
        returns_long = returns_long.sort_values(["date", "ticker"]).reset_index(drop=True)

        # --- membership and restrictions ---------------------------------
        membership = pd.DataFrame(
            {
                "ticker": tickers,
                "start_date": out_dates[start_pos],
                "end_date": [
                    out_dates[end_pos[i]] if delisted[i] else pd.NaT
                    for i in range(n_stocks)
                ],
            }
        )
        restrictions = self._restrictions(
            tickers, out_dates, start_pos, end_pos, sectors
        )

        # --- fundamentals with realistic release lags ---------------------
        fundamentals = self._fundamentals(
            tickers, out_dates, start_pos, end_pos, delisted
        )

        # --- realised oracle statistics -----------------------------------
        score_wide = pd.DataFrame(scores_out, index=out_dates, columns=tickers)
        forward_returns = returns_wide.shift(-1)
        ic_series = cross_sectional_correlation(
            score_wide, forward_returns, min_names=10, method="pearson"
        )
        rank_ic_series = cross_sectional_correlation(
            score_wide, forward_returns, min_names=10, method="spearman"
        )
        regime_series = pd.Series(high_vol_out, index=out_dates)
        low_mask = ~regime_series.reindex(ic_series.index).astype(bool)
        high_mask = regime_series.reindex(ic_series.index).astype(bool)
        rho_realized = mean_pairwise_correlation(returns_wide)

        truth = SyntheticTruth(
            spec=spec,
            rho_target=spec.rho,
            rho_expected=self._expected_rho(
                betas, sectors, var_market, var_sector, cross_var
            ),
            rho_realized=rho_realized,
            signal_ic_target=spec.signal_ic,
            signal_ic_realized=float(ic_series.mean()) if len(ic_series) else float("nan"),
            signal_rank_ic_realized=(
                float(rank_ic_series.mean()) if len(rank_ic_series) else float("nan")
            ),
            signal_half_life=spec.signal_half_life,
            n_eff=effective_sample_size(
                len(out_dates),
                n_stocks,
                0.0 if not np.isfinite(rho_realized) else max(rho_realized, 0.0),
            ),
            realized_kurtosis=float(
                pd.Series(returns_wide.to_numpy().ravel()).dropna().kurt()
            ),
            ic_low_vol=float(ic_series[low_mask].mean()) if low_mask.any() else float("nan"),
            ic_high_vol=float(ic_series[high_mask].mean()) if high_mask.any() else float("nan"),
            release_lags=dict(spec.release_lag_days),
            betas=pd.Series(betas, index=tickers, name="beta"),
            sectors=pd.Series(sectors, index=tickers, name="sector"),
            signal_scores=(
                score_wide.rename_axis("date")
                .reset_index()
                .melt(id_vars="date", var_name="ticker", value_name="score")
                .dropna(subset=["score"])
            ),
            regimes=pd.DataFrame(
                {
                    "date": out_dates,
                    "high_vol": high_vol_out,
                    "vol_mult": vol_mult_out,
                }
            ),
            delistings=pd.DataFrame(
                {
                    "ticker": tickers[delisted],
                    "delist_date": [out_dates[end_pos[i]] for i in np.where(delisted)[0]],
                }
            ),
            restrictions=restrictions,
        )

        return SyntheticPanel(
            prices=prices,
            returns=returns_long,
            fundamentals=fundamentals,
            membership=membership,
            restrictions=restrictions,
            truth=truth,
            calendar=TradingCalendar(out_dates),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _expected_rho(
        betas: np.ndarray,
        sectors: np.ndarray,
        var_market: float,
        var_sector: float,
        cross_var: float,
    ) -> float:
        """Analytic mean pairwise correlation implied by the calibration."""
        covariance = np.outer(betas, betas) * var_market + (
            sectors[:, None] == sectors[None, :]
        ) * var_sector
        variance = np.diag(covariance) + cross_var
        correlation = covariance / np.sqrt(np.outer(variance, variance))
        off_diagonal = correlation[~np.eye(len(betas), dtype=bool)]
        return float(off_diagonal.mean())

    def _restrictions(
        self,
        tickers: np.ndarray,
        out_dates: pd.DatetimeIndex,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        sectors: np.ndarray,
    ) -> pd.DataFrame:
        spec = self.spec
        rng = self.rng
        n_restricted = int(round(spec.restricted_fraction * len(tickers)))
        rows: list[dict[str, Any]] = []
        if n_restricted:
            eligible = [
                i for i in range(len(tickers)) if end_pos[i] - start_pos[i] >= 15
            ]
            if eligible:
                chosen = rng.choice(
                    eligible, size=min(n_restricted, len(eligible)), replace=False
                )
                for index in np.atleast_1d(chosen):
                    length = int(rng.integers(5, 21))
                    low = int(start_pos[index])
                    high = int(end_pos[index]) - length
                    if high <= low:
                        continue
                    begin = int(rng.integers(low, high))
                    rows.append(
                        {
                            "ticker": tickers[index],
                            "start_date": out_dates[begin],
                            "end_date": out_dates[min(begin + length, len(out_dates) - 1)],
                        }
                    )
        return pd.DataFrame(rows, columns=list(RESTRICTION_COLUMNS))

    def _fundamentals(
        self,
        tickers: np.ndarray,
        out_dates: pd.DatetimeIndex,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        delisted: np.ndarray,
    ) -> pd.DataFrame:
        spec = self.spec
        rng = self.rng
        calendar = TradingCalendar(out_dates)
        n_stocks = len(tickers)
        span_days = max(
            (out_dates[-1] - out_dates[0]).days, spec.fundamental_period_days
        )
        n_quarters = int(math.ceil(span_days / spec.fundamental_period_days)) + 2
        period_ends = pd.DatetimeIndex(
            [
                out_dates[0] + pd.Timedelta(days=q * spec.fundamental_period_days)
                for q in range(n_quarters)
            ]
        )
        shares = np.exp(rng.normal(17.0, 0.5, n_stocks))
        eps_path = 0.8 + np.cumsum(rng.normal(0.02, 0.10, (n_quarters, n_stocks)), axis=0)
        revenue_path = np.exp(rng.normal(13.0, 0.6, n_stocks))[None, :] * np.exp(
            np.cumsum(rng.normal(0.01, 0.04, (n_quarters, n_stocks)), axis=0)
        )
        book_path = np.maximum(
            0.5,
            5.0 + np.cumsum(rng.normal(0.05, 0.05, (n_quarters, n_stocks)), axis=0),
        )
        net_income_path = eps_path * shares[None, :] / 1e6
        paths = {
            "eps": eps_path,
            "revenue": revenue_path,
            "book_value": book_path,
            "net_income": net_income_path,
        }
        rows: list[dict[str, Any]] = []
        for field, matrix in paths.items():
            base_lag = spec.release_lag_days.get(field, 45)
            jitter = rng.integers(
                -spec.release_lag_jitter_days,
                spec.release_lag_jitter_days + 1,
                size=(n_quarters, n_stocks),
            )
            for q in range(n_quarters):
                period_end = period_ends[q]
                for index in range(n_stocks):
                    release = calendar.earliest_on_or_after(
                        period_end + pd.Timedelta(days=int(base_lag + jitter[q, index]))
                    )
                    if release is None or release > out_dates[-1]:
                        continue
                    if release < out_dates[start_pos[index]]:
                        continue
                    if delisted[index] and release > out_dates[end_pos[index]]:
                        continue
                    rows.append(
                        {
                            "ticker": tickers[index],
                            "field": field,
                            "period_end": period_end,
                            "release_ts": release,
                            "value": float(matrix[q, index]),
                            "vintage": release.strftime("%Y-%m-%d"),
                        }
                    )
        frame = pd.DataFrame(rows, columns=list(FUNDAMENTAL_COLUMNS))
        return frame.sort_values(
            ["ticker", "field", "release_ts"], kind="mergesort"
        ).reset_index(drop=True)
