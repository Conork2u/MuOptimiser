from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .data import Histogram1D
from .empirical import empirical_survival
from .optimise import MuOptimisationResult


def plot_tail_fit(
    histogram: Histogram1D,
    result: MuOptimisationResult,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical and fitted p-values for one declination band."""

    ts_grid = histogram.bin_centers
    survival = empirical_survival(histogram.counts)
    fit_grid = np.linspace(max(result.mu, float(np.min(ts_grid))), float(np.max(ts_grid)), 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(ts_grid, survival, where="post", color="black", label="Empirical survival")
    ax.plot(fit_grid, result.fit.survival(fit_grid), color="#c33c2f", label="Gamma survival fit")
    ax.axvline(result.mu, color="#4066a8", linestyle="--", label=f"mu = {result.mu:.3g}")

    for ts_value in result.target_ts_values:
        ax.axvline(ts_value, color="#888888", alpha=0.25, linewidth=1)

    title_parts = ["TS tail fit"]
    if result.sin_dec_mid is not None:
        title_parts.append(f"sin(dec) = {result.sin_dec_mid:.3f}")
    ax.set_title(" | ".join(title_parts))
    ax.set_xlabel("Test statistic, TS")
    ax.set_ylabel("p-value, P(TS >= t)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=180)
    return fig, ax


def plot_mu_by_sin_dec(
    results: Iterable[MuOptimisationResult],
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the optimised mu cut as a function of sin(declination)."""

    results = [result for result in results if result.sin_dec_mid is not None]
    if not results:
        raise ValueError("no results contain sin(declination) metadata")

    results = sorted(results, key=lambda result: result.sin_dec_mid)
    sin_dec = np.asarray([result.sin_dec_mid for result in results], dtype=float)
    mu_values = np.asarray([result.mu for result in results], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sin_dec, mu_values, marker="o", color="#4066a8", linewidth=1.5)
    ax.set_xlabel("sin(declination)")
    ax.set_ylabel("Optimised mu cut")
    ax.set_title("Optimised TS-tail cut by declination")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=180)
    return fig, ax
