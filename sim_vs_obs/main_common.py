from pathlib import Path

from matplotlib import pyplot
from pandas import read_csv
from string import ascii_uppercase

from sim_vs_obs.common import CMAP, NORM_INCIDENT_PAR, format_binary_colorbar
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
    text_kwargs = dict(fontsize=8)
    t_unit = UNITS_MAP['temperature'][1]

    vars_to_plot = ('temperature_canopy', 'delta_temperature_canopy')
    lims = {s: {'min': 0, 'max': 0} for s in vars_to_plot}

    im = None
    fig, axs = pyplot.subplots(ncols=len(EXPERIMENTS), figsize=(7.48, 4.8), nrows=2, sharex='row', sharey='row',
                               gridspec_kw=dict(wspace=0, hspace=0.5))
    for j, experiment in enumerate(EXPERIMENTS.keys()):
        dir_name = '_'.join((EXPERIMENTS[experiment][1], s_lumped))

        df = read_csv(path_source / experiment / 'outputs' / s_corrected / dir_name / 'results.csv')
        for i, var in enumerate(vars_to_plot):
            ax = axs[i, j]
            sim = df[f'{var}_sim']
            obs = df[f'{var}_obs']
            im = ax.scatter(obs, sim, c=df['incident_par'], **plot_kwargs)
            ax.text(0.1, 0.9, f"R² = {stats.calc_r2(sim, obs):.3f}", transform=ax.transAxes, **text_kwargs)
            ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(sim, obs):.3f}", transform=ax.transAxes, **text_kwargs)
            ax.text(0.1, 0.7, f"nNS = {stats.calc_nash_sutcliffe(sim, obs):.3f}", transform=ax.transAxes, **text_kwargs)
            lims[var]['min'] = min(lims[var]['min'], min(sim.min(), obs.min()))
            lims[var]['max'] = max(lims[var]['max'], max(sim.max(), obs.max()))

    for i, ax in enumerate(axs.flatten()):
        ax.set_aspect(aspect='equal')
        ax.text(0.83, 0.05, f"({ascii_uppercase[i]})", transform=ax.transAxes, **text_kwargs)

    for ax_t, ax_dt, experiment in zip(axs[0, :], axs[1, :], EXPERIMENTS.values()):
        ax_t.set_title(experiment[0], fontsize=11)
        ax_t.set_xlabel(' '.join(['obs', 'T', t_unit]), **text_kwargs)
        add_1_1_line(ax_t, lims=lims['temperature_canopy'], linewidth=0.5)
        ax_dt.set_xlabel(' '.join(['obs', r'$\mathregular{\Delta}$' + 'T', t_unit]), **text_kwargs)
        add_1_1_line(ax_dt, lims=lims['delta_temperature_canopy'], linewidth=0.5)

    axs[0, 0].set_ylabel(' '.join(['sim', 'T', t_unit]), **text_kwargs)
    axs[1, 0].set_ylabel(' '.join(['sim', r'$\mathregular{\Delta}$' + 'T', t_unit]), **text_kwargs)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.37, 0.05, 0.30, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    format_binary_colorbar(cbar=cbar)

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
