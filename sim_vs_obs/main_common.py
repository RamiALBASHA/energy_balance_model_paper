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


def plot_sim_vs_obs(path_source: Path, path_outputs: Path, is_corrected: bool = True, is_lumped_leaves: bool = True):
    s_corrected = 'corrected' if is_corrected else 'neutral'
    s_lumped = 'lumped' if is_lumped_leaves else 'sunlit-shaded'

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


def plot_correction_effect(path_source: Path, path_outputs: Path):
    leaf_classes = ('lumped', 'sunlit-shaded')
    fig, axs = pyplot.subplots(nrows=len(leaf_classes), figsize=(19 / 2.54, 10 / 2.54),
                               gridspec_kw=dict(hspace=0), sharex='all', sharey='all')
    width = 0.3
    d_width = width / 1.5
    kwargs = dict(width=width, edgecolor='grey', linewidth=0.5)
    x = range(len(EXPERIMENTS))

    x_ticks = None
    x_tick_labels = None

    for ax, leaf_class in zip(axs, leaf_classes):
        x_ticks = []
        x_tick_labels = []
        for i, experiment in enumerate(EXPERIMENTS.keys()):
            for sub_dir in ('neutral', 'corrected'):
                dx = - d_width if sub_dir == 'neutral' else + d_width
                df = read_csv(path_source / ('/'.join(
                    [experiment, 'outputs', sub_dir, f'{EXPERIMENTS[experiment][1]}_{leaf_class}', 'results.csv'])))
                squared_bias, nonunity_slope, lack_of_correlation = stats.calc_mean_squared_deviation_components(
                    sim=df['delta_temperature_canopy_sim'],
                    obs=df['delta_temperature_canopy_obs'])

                x_pos = x[i] + dx
                x_ticks.append(x_pos)
                x_tick_labels.append(sub_dir)
                ax.bar(x_pos, squared_bias, label='Squared bias', color='red', **kwargs)
                ax.bar(x_pos, nonunity_slope, label='Nonunity slope', bottom=squared_bias, color='white', **kwargs)
                ax.bar(x_pos, lack_of_correlation, label='Lack of correlation', bottom=squared_bias + nonunity_slope,
                       color='blue', **kwargs)

    handles, labels_ = axs[0].get_legend_handles_labels()
    labels = sorted(set(labels_))
    handles = [handles[labels_.index(s)] for s in labels]
    axs[0].legend(handles=handles, labels=labels, framealpha=0.5, fancybox=True)
    axs[0].set_ylabel(r'Mean squared deviation ($\rm {^\circ\/C}^2$)')
    axs[0].yaxis.set_label_coords(-0.065, 0, transform=axs[0].transAxes)
    axs[0].set_ylim(0, 14)
    axs[0].set_yticks(range(0, int(max(axs[0].get_ylim())), 2))

    axs[-1].set_xticks(x_ticks, minor=True)
    axs[-1].set_xticklabels(x_tick_labels, minor=True, fontsize=8)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels([v[0] for v in EXPERIMENTS.values()])
    axs[-1].tick_params(axis='x', which='major', pad=20, length=0)
    axs[-1].tick_params(axis='x', which='minor', pad=5, length=0)
    axs[-1].set_xlim(-0.55, 3.5)

    for ax, s in zip(axs, ascii_lowercase):
        ax.text(0.01, 0.9, f'({s})', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path_outputs / f'correction_effect.png')
    pyplot.close('all')
    pass


if __name__ == '__main__':
    path_sources = Path(__file__).parents[1] / 'sources'
    path_fig = path_sources / 'figs'
    plot_correction_effect(path_source=path_sources, path_outputs=path_fig)

    for is_lumped in (True, False):
        plot_sim_vs_obs(
            path_source=path_sources,
            path_outputs=path_fig,
            is_corrected=True,
            is_lumped_leaves=is_lumped)
