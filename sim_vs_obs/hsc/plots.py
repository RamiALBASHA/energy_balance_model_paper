from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame

from sim_vs_obs.hsc.base_functions import calc_apparent_temperature
from utils import stats

MAP_UNITS = {
    't': [r'$\mathregular{T_{canopy}}$', r'$\mathregular{[^\circ C]}$'],
    'delta_t': [r'$\mathregular{T_{canopy}-T_{air}}$', r'$\mathregular{[^\circ C]}$'],
}


def compare_temperature(obs: list, sim: list, ax: plt.Subplot = None, return_ax: bool = False,
                        plot_colorbar: bool = True, write_stats: bool = False):
    if ax is None:
        fig, ax = plt.subplots()
    cmap = cm.seismic
    norm = colors.Normalize(vmin=0, vmax=len(obs))
    ax.scatter(obs, sim, c=range(len(obs)), cmap=cmap, norm=norm)
    if plot_colorbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal")
        cbar.ax.set_xlabel('hour', va="bottom")
        cbar.ax.xaxis.set_major_locator(MultipleLocator(4))

    ax.set(xlabel='obs', ylabel='sim')
    ax_lims = sorted([v for sub_list in (ax.get_xlim(), ax.get_ylim()) for v in sub_list])
    xylims = [ax_lims[i] for i in (0, -1)]
    ax.set(xlim=xylims, ylim=xylims)
    ax.plot(xylims, xylims, 'k--')

    if write_stats:
        ax.text(0.1, 0.9, f"R2 = {stats.calc_r2(sim, obs):.3f}", transform=ax.transAxes)
        ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(sim, obs):.3f}", transform=ax.transAxes)
    if return_ax:
        return ax


def plot_results(all_solvers: dict, path_figs: Path):
    all_sim_t = []
    all_obs_t = []
    all_air_t = []

    x_ls = range(24)
    for d1, v1 in all_solvers.items():
        for plot_id, plot_res in v1.items():
            fig, axs = plt.subplots(3, 3, figsize=(8, 8))

            temp_obs = plot_res['temp_obs']
            incident_diffuse_par_irradiance = []
            incident_direct_par_irradiance = []
            wind_speed = []
            vapor_pressure_deficit = []
            air_temperature = []
            soil_water_potential = []
            richardson = []
            aerodynamic_resistance = []
            soil_abs_par = []
            temp_sim = []
            for solver in plot_res['solvers']:
                incident_diffuse_par_irradiance.append(solver.crop.inputs.incident_irradiance['diffuse'])
                incident_direct_par_irradiance.append(solver.crop.inputs.incident_irradiance['direct'])
                wind_speed.append(solver.crop.inputs.wind_speed / 3600.)
                vapor_pressure_deficit.append(solver.crop.inputs.vapor_pressure_deficit)
                air_temperature.append(solver.crop.inputs.air_temperature - 273.15)
                soil_water_potential.append(solver.crop.inputs.soil_water_potential)
                richardson.append(solver.crop.state_variables.richardson_number)
                aerodynamic_resistance.append(solver.crop.state_variables.aerodynamic_resistance * 3600.)
                soil_abs_par.append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                temp_sim.append(calc_apparent_temperature(eb_solver=solver, date_obs=d1))

            all_sim_t += temp_sim
            all_obs_t += temp_obs
            all_air_t += air_temperature

            axs[0, 0].plot(x_ls, incident_diffuse_par_irradiance, label=r'$\mathregular{{PAR}_{diff}}$')
            axs[0, 0].plot(x_ls, incident_direct_par_irradiance, label=r'$\mathregular{{PAR}_{dir}}$')
            axs[0, 0].plot(x_ls, soil_abs_par, label=r'$\mathregular{{PAR}_{abs,\/soil}}$', linewidth=2)
            axs[0, 0].set_ylim((0, 500))
            axs[0, 0].legend()

            axs[0, 1].plot(x_ls, wind_speed, label='u')
            axs[0, 1].set_ylim(0, 6)
            axs[0, 1].legend()

            axs[1, 1].plot(x_ls, richardson, label='Ri')
            axs[1, 1].hlines([-0.8] * len(x_ls), min(x_ls), max(x_ls), linewidth=2, color='red')
            axs[1, 1].set_ylim((-10, 3))
            axs[1, 1].legend()

            axs[2, 1].plot(x_ls, air_temperature, label=r'$\mathregular{T_{air}}$', color='black', linestyle='--')
            axs[2, 1].plot(x_ls, temp_obs, label=r'$\mathregular{T_{can,\/obs}}$', color='orange')
            axs[2, 1].plot(x_ls, temp_sim, label=r'$\mathregular{T_{can,\/sim}}$', color='blue')
            axs[2, 1].set_ylim(-5, 60)
            axs[2, 1].legend(fontsize='x-small')

            axs[1, 0].plot(x_ls, vapor_pressure_deficit, label='VPD')
            axs[1, 0].set_ylim(0, 6)
            axs[1, 0].legend()

            axs[2, 0].plot(x_ls, soil_water_potential, label=r'$\mathregular{\Psi_{soil}}$')
            axs[2, 0].legend()

            axs[0, 2].scatter(temp_obs, temp_sim, alpha=0.5)
            axs[0, 2].plot((-5, 40), (-5, 40), 'k--')
            axs[0, 2].set(xlim=(-5, 40), ylim=(-5, 40), xlabel='obs T', ylabel='sim T')
            roughness_length_for_momentum = plot_res["solvers"][0].crop.state_variables.roughness_length_for_momentum
            canopy_height = plot_res["solvers"][0].crop.inputs.canopy_height
            zero_displacement_height = plot_res["solvers"][0].crop.state_variables.zero_displacement_height
            axs[0, 2].text(0.1, 0.8, f'z0u/h ={roughness_length_for_momentum / canopy_height:.3f}',
                           transform=axs[0, 2].transAxes)
            axs[0, 2].text(0.1, 0.6, f'd/h ={zero_displacement_height / canopy_height:.3f}',
                           transform=axs[0, 2].transAxes)

            axs[1, 2].scatter(
                [t_obs - t_air for t_obs, t_air in zip(temp_obs, air_temperature)],
                [t_sim - t_air for t_sim, t_air in zip(temp_sim, air_temperature)],
                alpha=0.5)
            axs[1, 2].plot((-15, 21), (-15, 21), 'k--')
            axs[1, 2].set(xlim=(-15, 21), ylim=(-15, 21), xlabel='obs Tcan-Tair', ylabel='sim Tcan-Tair')

            axs[1, 2].text(0.1, 0.8, f'h={plot_res["solvers"][0].crop.inputs.canopy_height}',
                           transform=axs[1, 2].transAxes)
            axs[1, 2].text(0.1, 0.6, f'LAI={sum(plot_res["solvers"][0].crop.inputs.leaf_layers.values()):.3f}',
                           transform=axs[1, 2].transAxes)

            axs[2, 2].plot(x_ls, aerodynamic_resistance, label=r'$\mathregular{r_{a,\/0}}$')
            axs[2, 2].set(ylim=(0, 360), ylabel="s m-1")
            axs[2, 2].legend()

            for ax in axs[:, :2].flatten().tolist() + [axs[2, 2]]:
                ax.xaxis.set_major_locator(MultipleLocator(3))
                ax.grid()

            fig.savefig(path_figs / f"{plot_id}_{d1.date().strftime('%Y%m%d')}.png")
            plt.close(fig)

    fig_summary, axs_summary = plt.subplots(ncols=2)
    df = DataFrame(data={'sim_t': all_sim_t, 'obs_t': all_obs_t, 'air_t': all_air_t})
    df.loc[:, 'sim_delta_t'] = df['sim_t'] - df['air_t']
    df.loc[:, 'obs_delta_t'] = df['obs_t'] - df['air_t']

    for ax, var_to_plot in zip(axs_summary, ('t', 'delta_t')):
        x = df[f'obs_{var_to_plot}'].tolist()
        y = df[f'sim_{var_to_plot}'].tolist()
        lims = [sorted(x + y)[i] for i in [0, -1]]
        ax.scatter(x, y, marker='o', alpha=0.1)
        ax.plot(lims, lims, 'k--')
        ax.text(0.1, 0.9, f"R2 = {stats.calc_r2(x, y):.3f}", transform=ax.transAxes)
        ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(x, y):.3f}", transform=ax.transAxes)
        ax.set(xlabel=' '.join(['obs'] + MAP_UNITS[var_to_plot]), ylabel=' '.join(['sim'] + MAP_UNITS[var_to_plot]))

    fig_summary.tight_layout()
    fig_summary.savefig(path_figs / 'sim_vs_obs.png')
    plt.close()
    pass
