from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.stats import gamma

from .data import Histogram1D
from .empirical import (
    critical_ts_for_p,
    empirical_survival,
    normal_tail_probabilities,
    p_values_at,
)
from .models import GammaTailFit, fit_gamma_survival


DEFAULT_TARGET_TS = (5.0, 9.0, 12.5, 16.0)
DEFAULT_MU_BOUNDS = (0.5, 6.0)


@dataclass(frozen=True)
class MuOptimisationResult:
    """Optimisation result for one declination band."""

    mu: float
    objective_value: float
    fit: GammaTailFit
    objective: str
    method: str
    target_ts_values: tuple[float, ...]
    empirical_p_values: tuple[float, ...]
    fitted_p_values: tuple[float, ...]
    sin_dec_start: float | None = None
    sin_dec_stop: float | None = None
    source: Path | None = None
    success: bool = True
    message: str = ""

    @property
    def sin_dec_mid(self) -> float | None:
        if self.sin_dec_start is None or self.sin_dec_stop is None:
            return None
        return 0.5 * (self.sin_dec_start + self.sin_dec_stop)

    @property
    def dec_mid_deg(self) -> float | None:
        if self.sin_dec_mid is None:
            return None
        return float(np.degrees(np.arcsin(np.clip(self.sin_dec_mid, -1.0, 1.0))))

    def to_record(self) -> dict[str, object]:
        return {
            "source": "" if self.source is None else str(self.source),
            "sin_dec_start": self.sin_dec_start,
            "sin_dec_stop": self.sin_dec_stop,
            "sin_dec_mid": self.sin_dec_mid,
            "dec_mid_deg": self.dec_mid_deg,
            "mu": self.mu,
            "objective": self.objective,
            "objective_value": self.objective_value,
            "method": self.method,
            "gamma_shape": self.fit.shape,
            "gamma_loc": self.fit.loc,
            "gamma_scale": self.fit.scale,
            "tail_points": self.fit.n_tail_points,
            "fit_loss": self.fit.loss,
            "target_ts_values": list(self.target_ts_values),
            "empirical_p_values": list(self.empirical_p_values),
            "fitted_p_values": list(self.fitted_p_values),
            "success": self.success,
            "message": self.message,
        }


def optimise_mu(
    histogram: Histogram1D,
    *,
    bounds: tuple[float, float] = DEFAULT_MU_BOUNDS,
    objective: str = "log-p",
    method: str = "bounded",
    target_ts_values: Iterable[float] = DEFAULT_TARGET_TS,
    sigma_levels: Iterable[float] = (3.0, 4.0, 4.5),
    min_tail_points: int = 8,
    fit_loc: bool = False,
    random_state: int | None = 0,
) -> MuOptimisationResult:
    survival = empirical_survival(histogram.counts)
    return optimise_mu_from_arrays(
        histogram.bin_centers,
        survival,
        bounds=bounds,
        objective=objective,
        method=method,
        target_ts_values=target_ts_values,
        sigma_levels=sigma_levels,
        min_tail_points=min_tail_points,
        fit_loc=fit_loc,
        random_state=random_state,
        sin_dec_start=histogram.sin_dec_start,
        sin_dec_stop=histogram.sin_dec_stop,
        source=histogram.source,
    )


def optimise_many(
    histograms: Iterable[Histogram1D],
    **kwargs: object,
) -> list[MuOptimisationResult]:
    return [optimise_mu(histogram, **kwargs) for histogram in histograms]


def optimise_mu_from_arrays(
    ts_grid: np.ndarray,
    survival: np.ndarray,
    *,
    bounds: tuple[float, float] = DEFAULT_MU_BOUNDS,
    objective: str = "log-p",
    method: str = "bounded",
    target_ts_values: Iterable[float] = DEFAULT_TARGET_TS,
    sigma_levels: Iterable[float] = (3.0, 4.0, 4.5),
    min_tail_points: int = 8,
    fit_loc: bool = False,
    random_state: int | None = 0,
    sin_dec_start: float | None = None,
    sin_dec_stop: float | None = None,
    source: Path | None = None,
) -> MuOptimisationResult:
    ts_grid = np.asarray(ts_grid, dtype=float)
    survival = np.asarray(survival, dtype=float)
    bounds = _clean_bounds(bounds, ts_grid)
    objective = objective.lower()
    method = method.lower()
    target_ts_values = tuple(float(value) for value in target_ts_values)
    sigma_levels = tuple(float(value) for value in sigma_levels)

    if objective not in {"log-p", "critical-ts"}:
        raise ValueError("objective must be 'log-p' or 'critical-ts'")

    def score(candidate_mu: float) -> float:
        try:
            if objective == "log-p":
                return log_p_value_loss(
                    candidate_mu,
                    ts_grid,
                    survival,
                    target_ts_values=target_ts_values,
                    min_tail_points=min_tail_points,
                    fit_loc=fit_loc,
                )
            return critical_ts_loss(
                candidate_mu,
                ts_grid,
                survival,
                sigma_levels=sigma_levels,
                min_tail_points=min_tail_points,
                fit_loc=fit_loc,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            return float("inf")

    if method == "bounded":
        optimised = minimize_scalar(score, bounds=bounds, method="bounded")
        mu = float(optimised.x)
        objective_value = float(optimised.fun)
        success = bool(optimised.success and np.isfinite(objective_value))
        message = str(optimised.message)
    elif method == "differential_evolution":
        optimised = differential_evolution(
            lambda value: score(float(value[0])),
            bounds=[bounds],
            seed=random_state,
            polish=True,
        )
        mu = float(optimised.x[0])
        objective_value = float(optimised.fun)
        success = bool(optimised.success and np.isfinite(objective_value))
        message = str(optimised.message)
    else:
        raise ValueError("method must be 'bounded' or 'differential_evolution'")

    if not success:
        raise RuntimeError(f"mu optimisation failed: {message}")

    fit = fit_gamma_survival(
        ts_grid,
        survival,
        mu,
        min_tail_points=min_tail_points,
        fit_loc=fit_loc,
    )
    evaluation_ts = _evaluation_ts(
        objective=objective,
        mu=mu,
        ts_grid=ts_grid,
        survival=survival,
        target_ts_values=target_ts_values,
        sigma_levels=sigma_levels,
    )
    empirical_p = tuple(float(value) for value in p_values_at(evaluation_ts, ts_grid, survival))
    fitted_p = tuple(float(value) for value in fit.survival(np.asarray(evaluation_ts)))

    return MuOptimisationResult(
        mu=mu,
        objective_value=objective_value,
        fit=fit,
        objective=objective,
        method=method,
        target_ts_values=tuple(float(value) for value in evaluation_ts),
        empirical_p_values=empirical_p,
        fitted_p_values=fitted_p,
        sin_dec_start=sin_dec_start,
        sin_dec_stop=sin_dec_stop,
        source=source,
        success=success,
        message=message,
    )


def log_p_value_loss(
    mu: float,
    ts_grid: np.ndarray,
    survival: np.ndarray,
    *,
    target_ts_values: Iterable[float] = DEFAULT_TARGET_TS,
    min_tail_points: int = 8,
    fit_loc: bool = False,
) -> float:
    fit = fit_gamma_survival(
        ts_grid,
        survival,
        mu,
        min_tail_points=min_tail_points,
        fit_loc=fit_loc,
    )
    targets = np.asarray(tuple(float(value) for value in target_ts_values), dtype=float)
    targets = targets[(targets >= mu) & (targets <= np.max(ts_grid))]
    if targets.size == 0:
        return float("inf")

    empirical_p = p_values_at(targets, ts_grid, survival)
    fitted_p = fit.survival(targets)
    mask = (empirical_p > 0) & (fitted_p > 0) & np.isfinite(empirical_p) & np.isfinite(fitted_p)
    if np.count_nonzero(mask) == 0:
        return float("inf")

    residual = np.log10(empirical_p[mask]) - np.log10(fitted_p[mask])
    return float(np.mean(residual**2))


def critical_ts_loss(
    mu: float,
    ts_grid: np.ndarray,
    survival: np.ndarray,
    *,
    sigma_levels: Iterable[float] = (3.0, 4.0, 4.5),
    min_tail_points: int = 8,
    fit_loc: bool = False,
) -> float:
    fit = fit_gamma_survival(
        ts_grid,
        survival,
        mu,
        min_tail_points=min_tail_points,
        fit_loc=fit_loc,
    )
    p_values = normal_tail_probabilities(sigma_levels).values()
    residuals: list[float] = []
    for p_value in p_values:
        empirical_ts = critical_ts_for_p(p_value, ts_grid, survival)
        fitted_ts = float(gamma.ppf(1.0 - p_value, fit.shape, loc=fit.loc, scale=fit.scale))
        if np.isfinite(empirical_ts) and np.isfinite(fitted_ts) and empirical_ts > 0:
            residuals.append((fitted_ts - empirical_ts) / empirical_ts)
    if not residuals:
        return float("inf")
    return float(np.mean(np.square(residuals)))


def _clean_bounds(bounds: tuple[float, float], ts_grid: np.ndarray) -> tuple[float, float]:
    low, high = (float(bounds[0]), float(bounds[1]))
    low = max(low, float(np.min(ts_grid)))
    high = min(high, float(np.max(ts_grid)) - 1e-9)
    if low >= high:
        raise ValueError("mu bounds do not overlap the TS range")
    return low, high


def _evaluation_ts(
    *,
    objective: str,
    mu: float,
    ts_grid: np.ndarray,
    survival: np.ndarray,
    target_ts_values: tuple[float, ...],
    sigma_levels: tuple[float, ...],
) -> tuple[float, ...]:
    if objective == "log-p":
        values = tuple(value for value in target_ts_values if value >= mu and value <= np.max(ts_grid))
        if values:
            return values
        return (float(mu),)

    probabilities = normal_tail_probabilities(sigma_levels).values()
    return tuple(critical_ts_for_p(p_value, ts_grid, survival) for p_value in probabilities)
