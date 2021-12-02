import matplotlib.pyplot as plt
from numpy import linspace

from utils import water_retention
from utils.van_genuchten_params import VanGenuchtenParams


def plot_water_retention_curve(soil_class: str, psi: list = None, theta: list = None, ax: plt.Subplot = None):
    assert any([psi, theta]), "Either 'psi' or 'theta' args must be provided."
    if psi is not None:
        x = [water_retention.calc_soil_water_content(psi=f, soil_class=soil_class) for f in psi]
        y = [abs(f) for f in psi]
    else:
        x = theta
        y = [abs(water_retention.calc_soil_water_potential(theta=f, soil_class=soil_class)) for f in theta]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_yscale('log')

    ax.plot(x, y, label=soil_class)
    ax.set(
        xlabel=r'$\mathregular{\Theta\/[m^3_{water}\/m^{-3}_{soil}]}$',
        ylabel=r'$\mathregular{\Psi_{soil}\/[cm_{water}]}$')

    ax.grid()
    fig.tight_layout()
    fig.show()
    pass


def plot_all_water_retention_curve():
    fig, ax = plt.subplots()
    ax.set_yscale('log')

    for soil_class_data in VanGenuchtenParams:
        name, (theta_r, theta_s, alpha, n, k_sat, m) = soil_class_data.name, soil_class_data.value
        theta_range = list(linspace(theta_r * (1 + 1.e-6), theta_s, 100))
        plot_water_retention_curve(soil_class=name, theta=theta_range, ax=ax)

    ax.set(
        xlabel=r'$\mathregular{\Theta\/[m^3_{water}\/m^{-3}_{soil}]}$',
        ylabel=r'$\mathregular{\Psi_{soil}\/[cm_{water}]}$',
        ylim=(1, 1.e6))

    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig('water_retention_curves_of_van_genuchten.png')
    plt.close()
    pass
