import numpy as np
from scipy.stats import gamma

from mu_optimiser.models import fit_gamma_survival


def test_fit_gamma_survival_recovers_valid_tail_model():
    ts_grid = np.linspace(0.1, 25.0, 120)
    survival = gamma.sf(ts_grid, 2.4, loc=0.0, scale=1.8)

    fit = fit_gamma_survival(ts_grid, survival, mu=2.5)

    assert fit.shape > 0
    assert fit.scale > 0
    assert fit.n_tail_points > 10
    np.testing.assert_allclose(fit.survival([5.0, 10.0]), gamma.sf([5.0, 10.0], 2.4, scale=1.8), rtol=0.2)
