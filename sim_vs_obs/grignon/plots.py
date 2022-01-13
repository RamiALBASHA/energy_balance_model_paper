from copy import deepcopy
from pathlib import Path

from matplotlib import pyplot, ticker
from pandas import isna

from utils import stats

MAP_UNITS = {
    't': [r'$\mathregular{T_{leaf}}$', r'$\mathregular{[^\circ C]}$'],
    'delta_t': [r'$\mathregular{T_{leaf}-T_{air}}$', r'$\mathregular{[^\circ C]}$'],
}


def plot_dynamic(data: dict, path_figs_dir: Path):
    idate = None
    fig_d, axs_d = pyplot.subplots(nrows=2, sharex='all', sharey='all')
    for counter, datetime_obs in enumerate(data.keys()):
        treatments = list(data[datetime_obs].keys())

        actual_date = datetime_obs.date()
        if actual_date != idate and idate is not None:
            for ax_d, treatment in zip(axs_d, treatments):
                ax_d.set(ylim=(-15, 30), ylabel=r'$\mathregular{T_{leaf}\/[^\circ C]}$',
                         title=f"{treatment} (GAI={sum(data[datetime_obs][treatment]['solver'].crop.inputs.leaf_layers.values()):.2f})")

            axs_d[-1].set(xlabel='hour')
            axs_d[-1].xaxis.set_major_locator(ticker.MultipleLocator(4))
            fig_d.savefig(path_figs_dir / f'{idate}.png')
            pyplot.close(fig_d)
            fig_d, axs_d = pyplot.subplots(nrows=len(treatments), sharex='all', sharey='all')
        idate = actual_date

        fig_h, axs_h = pyplot.subplots(nrows=len(treatments), sharex='all', sharey='all')
        for ax_h, ax_d, treatment in zip(axs_h, axs_d, treatments):
            solver = data[datetime_obs][treatment]['solver']
            obs = data[datetime_obs][treatment]['obs']
            canopy_layers = [k for k in solver.crop.components_keys if k != -1]

            y_obs = []
            x_obs = []
            x_obs_avg = []
            x_sim = []
            for layer in canopy_layers:
                ax_h.set_title(f'{treatment} (GAI={sum(solver.crop.inputs.leaf_layers.values()):.2f})')
                obs_temperature = obs[obs['leaf_level'] == layer]['temperature']
                x_obs_avg.append(obs_temperature.mean())
                x_obs += obs_temperature.to_list()
                y_obs += [layer] * len(obs_temperature)
                x_sim.append(solver.crop[layer].temperature - 273.15)

            ax_h.scatter(x_obs, y_obs, marker='s', c='red', alpha=0.3)
            ax_h.scatter(x_sim, canopy_layers, marker='o', c='blue')
            ax_d.scatter([datetime_obs.hour] * len(x_obs), x_obs, marker='s', c='red', alpha=0.3)
            ax_d.scatter([datetime_obs.hour] * len(x_sim), x_sim, marker='o', c='blue')
            ax_h.scatter(x_obs_avg, canopy_layers, marker='o', edgecolor='black', c='red')

        axs_h[0].set(ylim=(0, 13), xlim=(-5, 30))
        [ax.set_ylabel('layer index') for ax in axs_h]
        axs_h[1].set_xlabel(r'$\mathregular{T_{leaf}\/[^\circ C]}$')
        axs_h[0].yaxis.set_major_locator(ticker.MultipleLocator(1))

        fig_h.suptitle(f"{datetime_obs.strftime('%Y-%m-%d %H:%M')}")
        fig_h.savefig(path_figs_dir / f'{counter}.png')
        pyplot.close(fig_h)
    pass


def plot_sim_vs_obs(data: dict, path_figs_dir: Path, relative_layer_index: int = None):
    treatments = ('extensive', 'intensive')
    vars_to_plot = ('t', 'delta_t')

    fig, axs = pyplot.subplots(nrows=len(vars_to_plot), ncols=len(treatments), sharex='row', sharey='row')
    obs_dict = {s: {k: [] for k in treatments} for s in vars_to_plot}
    sim_dict = deepcopy(obs_dict)
    for counter, datetime_obs in enumerate(data.keys()):
        for treatment in treatments:
            solver = data[datetime_obs][treatment]['solver']
            obs = data[datetime_obs][treatment]['obs'].dropna()
            layers_sim = [k for k in solver.crop.components_keys if k != -1]
            layers_obs = list(obs['leaf_level'].unique())
            layers = sorted([i for i in layers_sim if i in layers_obs])
            if relative_layer_index is not None:
                layers = (layers[relative_layer_index],)

            for layer in layers:
                t_obs = obs[obs['leaf_level'] == layer]['temperature'].mean()
                if not isna(t_obs):
                    t_sim = solver.crop[layer].temperature - 273.15
                    t_air = solver.crop.inputs.air_temperature - 273.15
                    obs_dict['t'][treatment].append(t_obs)
                    sim_dict['t'][treatment].append(t_sim)
                    obs_dict['delta_t'][treatment].append(t_obs - t_air)
                    sim_dict['delta_t'][treatment].append(t_sim - t_air)

    for ax_row, var_to_plot in zip(axs, vars_to_plot):
        for ax, treatment in zip(ax_row, treatments):
            temperature_obs = obs_dict[var_to_plot][treatment]
            temperature_sim = sim_dict[var_to_plot][treatment]

            ax.scatter(temperature_obs, temperature_sim, marker='o', alpha=0.1)
            ax.text(0.05, 0.9,
                    ''.join([r'$\mathregular{R^2=}$', f'{stats.calc_r2(temperature_obs, temperature_sim):.3f}']),
                    transform=ax.transAxes)
            ax.text(0.05, 0.8, f'RMSE={stats.calc_rmse(temperature_obs, temperature_sim):.3f} °C',
                    transform=ax.transAxes)
            lims = [sorted(temperature_obs + temperature_sim)[i] for i in (0, -1)]
            ax.plot(lims, lims, 'k--', linewidth=0.5)

            ax.set_xlabel(' '.join(['obs'] + MAP_UNITS[var_to_plot]))
    for ax, var_to_plot in zip(axs[:, 0], vars_to_plot):
        ax.set_ylabel(' '.join(['sim'] + MAP_UNITS[var_to_plot]))
    for ax, treatment in zip(axs[0, :], treatments):
        ax.set_title(treatment)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.savefig(path_figs_dir / f'sim_vs_obs{relative_layer_index}.png')
    pass