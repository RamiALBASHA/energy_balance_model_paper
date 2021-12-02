from numpy import linspace
from numpy.testing import assert_almost_equal

from utils import water_retention, van_genuchten_params, plots


def assert_equivalence():
    for soil_class_data in van_genuchten_params.VanGenuchtenParams:
        name, (theta_r, theta_s, alpha, n, k_sat, m) = soil_class_data.name, soil_class_data.value
        theta_range = linspace(theta_r * (1 + 1.e-6), theta_s, 100)
        psi_range = [water_retention.calc_soil_water_potential(theta=f, soil_class=name) for f in theta_range]
        theta_range_calc = [water_retention.calc_soil_water_content(psi=f, soil_class=name) for f in psi_range]
        assert_almost_equal(actual=theta_range, desired=theta_range_calc, decimal=7)

    pass


if __name__ == '__main__':
    assert_equivalence()
    plots.plot_all_water_retention_curve()
