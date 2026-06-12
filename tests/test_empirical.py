import numpy as np

from mu_optimiser.empirical import critical_ts_for_p, empirical_survival, p_values_at


def test_empirical_survival_uses_reverse_cumulative_counts():
    survival = empirical_survival(np.array([3, 2, 1]))

    np.testing.assert_allclose(survival, np.array([1.0, 0.5, 1.0 / 6.0]))


def test_p_values_at_interpolates_survival_curve():
    ts_grid = np.array([0.0, 1.0, 2.0])
    survival = np.array([1.0, 0.5, 0.1])

    values = p_values_at([0.5, 1.5], ts_grid, survival)

    np.testing.assert_allclose(values, np.array([0.75, 0.3]))


def test_critical_ts_for_p_interpolates_decreasing_curve():
    ts_grid = np.array([0.0, 1.0, 2.0])
    survival = np.array([1.0, 0.5, 0.1])

    assert critical_ts_for_p(0.3, ts_grid, survival) == 1.5
