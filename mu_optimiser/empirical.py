from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import norm


DEFAULT_SIGMA_LEVELS = (3.0, 4.0, 4.5)


def empirical_survival(counts: np.ndarray) -> np.ndarray:
    """Return the empirical survival curve P(TS >= t) from histogram counts."""

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("histogram contains no counts")

    return np.cumsum(counts[::-1])[::-1] / total


def p_values_at(
    ts_values: float | Iterable[float],
    ts_grid: np.ndarray,
    survival: np.ndarray,
) -> np.ndarray:
    """Interpolate empirical p-values at one or more TS values."""

    ts_grid, survival = _valid_curve(ts_grid, survival)
    values = np.atleast_1d(np.asarray(ts_values, dtype=float))
    return np.interp(values, ts_grid, survival, left=survival[0], right=survival[-1])


def critical_ts_for_p(p_value: float, ts_grid: np.ndarray, survival: np.ndarray) -> float:
    """Interpolate the TS value where the survival curve reaches a p-value."""

    if not np.isfinite(p_value) or p_value <= 0:
        raise ValueError("p_value must be positive and finite")

    ts_grid, survival = _valid_curve(ts_grid, survival)
    if p_value >= survival[0]:
        return float(ts_grid[0])
    if p_value <= survival[-1]:
        return float(ts_grid[-1])

    reversed_p = survival[::-1]
    reversed_ts = ts_grid[::-1]
    unique_p, unique_indices = np.unique(reversed_p, return_index=True)
    unique_ts = reversed_ts[unique_indices]
    return float(np.interp(p_value, unique_p, unique_ts))


def normal_tail_probabilities(
    sigma_levels: Iterable[float] = DEFAULT_SIGMA_LEVELS,
) -> dict[float, float]:
    """Return one-sided Gaussian tail probabilities for sigma levels."""

    return {float(sigma): float(norm.sf(float(sigma))) for sigma in sigma_levels}


def _valid_curve(ts_grid: np.ndarray, survival: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts_grid = np.asarray(ts_grid, dtype=float)
    survival = np.asarray(survival, dtype=float)

    if ts_grid.ndim != 1 or survival.ndim != 1:
        raise ValueError("ts_grid and survival must be one-dimensional")
    if ts_grid.size != survival.size:
        raise ValueError("ts_grid and survival must have the same length")
    if ts_grid.size < 2:
        raise ValueError("at least two points are needed")
    if np.any(np.diff(ts_grid) <= 0):
        raise ValueError("ts_grid must be strictly increasing")

    mask = np.isfinite(ts_grid) & np.isfinite(survival) & (survival >= 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("not enough valid survival points")

    return ts_grid[mask], survival[mask]
