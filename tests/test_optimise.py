import numpy as np
from scipy.stats import gamma

from mu_optimiser.data import Histogram1D
from mu_optimiser.optimise import optimise_mu


def test_optimise_mu_returns_public_result_record():
    edges = np.linspace(0.0, 30.0, 151)
    centers = 0.5 * (edges[:-1] + edges[1:])
    survival = gamma.sf(centers, 2.2, scale=1.9)
    probabilities = np.r_[survival, 0.0]
    counts = -np.diff(probabilities) * 100000
    histogram = Histogram1D(
        counts=counts,
        bin_edges=edges,
        sin_dec_start=-0.5,
        sin_dec_stop=-0.495,
    )

    result = optimise_mu(histogram, bounds=(0.5, 5.0), target_ts_values=(5.0, 9.0, 12.5))
    record = result.to_record()

    assert 0.5 <= result.mu <= 5.0
    assert record["sin_dec_mid"] < 0
    assert record["gamma_shape"] > 0
    assert record["gamma_scale"] > 0
