"""Tools for optimising gamma-tail cuts in IceCube TS distributions."""

from .data import Histogram1D, load_2d_histograms, load_histogram, load_histograms
from .empirical import critical_ts_for_p, empirical_survival, p_values_at
from .models import GammaTailFit, fit_gamma_survival
from .optimise import MuOptimisationResult, optimise_many, optimise_mu

__all__ = [
    "GammaTailFit",
    "Histogram1D",
    "MuOptimisationResult",
    "critical_ts_for_p",
    "empirical_survival",
    "fit_gamma_survival",
    "load_2d_histograms",
    "load_histogram",
    "load_histograms",
    "optimise_many",
    "optimise_mu",
    "p_values_at",
]
