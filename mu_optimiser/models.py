from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma


LOG_FLOOR = 1e-300


@dataclass(frozen=True)
class GammaTailFit:
    """Gamma survival fit to the TS tail above a chosen mu cut."""

    shape: float
    scale: float
    loc: float
    mu: float
    n_tail_points: int
    loss: float
    covariance: np.ndarray | None = None

    def survival(self, ts_values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(ts_values, dtype=float)
        return gamma.sf(values, self.shape, loc=self.loc, scale=self.scale)

    @property
    def params(self) -> tuple[float, float, float]:
        return (self.shape, self.loc, self.scale)


def fit_gamma_survival(
    ts_grid: np.ndarray,
    survival: np.ndarray,
    mu: float,
    *,
    min_tail_points: int = 8,
    fit_loc: bool = False,
) -> GammaTailFit:
    """Fit a gamma survival curve to empirical survival values above mu."""

    ts_grid = np.asarray(ts_grid, dtype=float)
    survival = np.asarray(survival, dtype=float)
    if ts_grid.ndim != 1 or survival.ndim != 1:
        raise ValueError("ts_grid and survival must be one-dimensional")
    if ts_grid.size != survival.size:
        raise ValueError("ts_grid and survival must have the same length")
    if not np.isfinite(mu):
        raise ValueError("mu must be finite")

    mask = (
        np.isfinite(ts_grid)
        & np.isfinite(survival)
        & (ts_grid >= mu)
        & (survival > 0)
        & (survival <= 1)
    )
    x_tail = ts_grid[mask]
    y_tail = survival[mask]

    if x_tail.size < min_tail_points:
        raise ValueError(
            f"need at least {min_tail_points} tail points above mu={mu:.4g}; "
            f"got {x_tail.size}"
        )

    y_log = np.log10(np.clip(y_tail, LOG_FLOOR, 1.0))

    if fit_loc:
        p0, lower, upper = _initial_with_loc(x_tail)

        def model(x: np.ndarray, shape: float, loc: float, scale: float) -> np.ndarray:
            values = gamma.sf(x, shape, loc=loc, scale=scale)
            return np.log10(np.clip(values, LOG_FLOOR, 1.0))

        popt, covariance = curve_fit(
            model,
            x_tail,
            y_log,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )
        shape, loc, scale = (float(popt[0]), float(popt[1]), float(popt[2]))
    else:
        p0, lower, upper = _initial_without_loc(x_tail)

        def model(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
            values = gamma.sf(x, shape, loc=0.0, scale=scale)
            return np.log10(np.clip(values, LOG_FLOOR, 1.0))

        popt, covariance = curve_fit(
            model,
            x_tail,
            y_log,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )
        shape, scale = (float(popt[0]), float(popt[1]))
        loc = 0.0

    fitted_log = np.log10(np.clip(gamma.sf(x_tail, shape, loc=loc, scale=scale), LOG_FLOOR, 1.0))
    loss = float(np.mean((fitted_log - y_log) ** 2))

    return GammaTailFit(
        shape=shape,
        scale=scale,
        loc=loc,
        mu=float(mu),
        n_tail_points=int(x_tail.size),
        loss=loss,
        covariance=covariance,
    )


def _initial_without_loc(x_tail: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mean_tail = max(float(np.mean(x_tail)), 1e-3)
    shape = 2.0
    scale = max(mean_tail / shape, 1e-3)
    return [shape, scale], [1e-3, 1e-3], [500.0, 500.0]


def _initial_with_loc(x_tail: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mean_tail = max(float(np.mean(x_tail)), 1e-3)
    loc_upper = min(float(np.min(x_tail)) - 1e-6, 0.0)
    loc_start = min(0.0, loc_upper - 1e-3)
    shape = 2.0
    scale = max((mean_tail - loc_start) / shape, 1e-3)
    return [shape, loc_start, scale], [1e-3, -100.0, 1e-3], [500.0, loc_upper, 500.0]
