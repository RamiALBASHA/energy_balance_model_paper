from pathlib import Path
from string import ascii_lowercase

from matplotlib import pyplot
from pandas import read_csv

from sim_vs_obs.common import CMAP, NORM_INCIDENT_PAR
from utils import stats
from utils.config import UNITS_MAP

EXPERIMENTS = {
    'maricopa_face': ('Maricopa FACE', 'bigleaf'),
    'maricopa_hsc': ('Maricopa HSC', 'bigleaf'),
    'grignon': ('Grignon', 'layered'),
    'braunschweig_face': ('Braunschweig FACE', 'bigleaf')}


def add_1_1_line(ax: pyplot.Subplot, lims: dict, **kwargs):
    lims = [lims[s] for s in ('min', 'max')]
    ax.plot(lims, lims, 'k--', label='1:1', **kwargs)
    pass


def plot_sim_vs_obs(path_source: Path, path_outputs: Path, is_corrected: bool = True, is_lumped: bool = True):
    s_corrected = 'corrected' if is_corrected else 'neutral'
    s_lumped = 'lumped' if is_lumped else 'sunlit-shaded'

    plot_kwargs = dict(marker='.', edgecolor='none', alpha=0.5, cmap=CMAP, norm=NORM_INCIDENT_PAR)
    kwargs = dict(fontsize=8, ha='right')
    t_unit = UNITS_MAP['temperature'][1]

    vars_to_plot = ('temperature_canopy', 'delta_temperature_canopy')
    lims = {s: {'min': 0, 'max': 0} for s in vars_to_plot}

    fig, axs = pyplot.subplots(ncols=len(EXPERIMENTS), figsize=(7.48, 4.8), nrows=2, sharex='row', sharey='row',
                               gridspec_kw=dict(wspace=0, hspace=0.5))
    for j, experiment in enumerate(EXPERIMENTS.keys()):
        dir_name = '_'.join((EXPERIMENTS[experiment][1], s_lumped))
        df = read_csv(path_source / experiment / 'outputs' / s_corrected / dir_name / 'results.csv')
        axs[0, j].set_title(EXPERIMENTS[experiment][0], fontsize=10, pad=20)
        for i, var in enumerate(vars_to_plot):
            ax = axs[i, j]
            sim = df[f'{var}_sim']
            obs = df[f'{var}_obs']
            ax.scatter(obs, sim, c=df['incident_par'], **plot_kwargs)
            ax.text(0.05, 0.875, f"({ascii_lowercase[j + i * 4]})", transform=ax.transAxes)
            ax.text(0.95, 0.25, f"R² = {stats.calc_r2(sim, obs):.2f}", transform=ax.transAxes, **kwargs)
            ax.text(0.95, 0.15, f"RMSE = {stats.calc_rmse(sim, obs):.2f}", transform=ax.transAxes, **kwargs)
            ax.text(0.95, 0.05, f"nNS = {stats.calc_nash_sutcliffe(sim, obs):.2f}", transform=ax.transAxes, **kwargs)
            lims[var]['min'] = min(lims[var]['min'], min(sim.min(), obs.min()))
            lims[var]['max'] = max(lims[var]['max'], max(sim.max(), obs.max()))
            ax.set_aspect(aspect='equal')

    for i, s, var in zip(range(len(vars_to_plot)), ('', ' depression'), vars_to_plot):
        axs[i, 0].set_ylabel(f"Simulated canopy\ntemperature{s}\n{t_unit}")
        axs[i, 1].set_xlabel(f'Observed canopy temperature{s} {t_unit}')
        axs[i, 1].xaxis.set_label_coords(1, -0.21, transform=axs[i, 1].transAxes)

    for ax_t, ax_dt, experiment in zip(axs[0, :], axs[1, :], EXPERIMENTS.values()):
        add_1_1_line(ax_t, lims=lims['temperature_canopy'], linewidth=0.5)
        add_1_1_line(ax_dt, lims=lims['delta_temperature_canopy'], linewidth=0.5)

    fig.subplots_adjust(left=0.15, right=0.985, top=0.89, bottom=0.125)

    fig.savefig(path_outputs / f'sim_vs_obs_all_experiments_{s_corrected}_{s_lumped}.png')
    pass


if __name__ == '__main__':
    path_sources = Path(__file__).parents[1] / 'sources'
    path_fig = path_sources / 'figs'
    for is_lumped in (True, False):
        plot_sim_vs_obs(
            path_source=path_sources,
            path_outputs=path_fig,
            is_corrected=True,
            is_lumped=is_lumped)
