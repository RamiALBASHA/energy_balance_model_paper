from pathlib import Path

from matplotlib import pyplot, ticker


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
