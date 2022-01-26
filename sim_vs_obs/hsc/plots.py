from pathlib import Path

import matplotlib.pyplot as plt
from crop_energy_balance.solver import Solver
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame

from sim_vs_obs.hsc.base_functions import calc_apparent_temperature, calc_neutral_aerodynamic_resistance
from utils import stats

MAP_UNITS = {
    't': [r'$\mathregular{T_{canopy}}$', r'$\mathregular{[^\circ C]}$'],
    'delta_t': [r'$\mathregular{T_{canopy}-T_{air}}$', r'$\mathregular{[^\circ C]}$'],
}


def calc_abs_irradiance(solver: Solver):
    return sum([sum(v.values()) for k, v in solver.inputs.absorbed_irradiance.items() if k != -1])


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
            fig, axs = plt.subplots(3, 4, figsize=(12, 8))

            temp_obs = plot_res['temp_obs']
            incident_diffuse_par_irradiance = []
            incident_direct_par_irradiance = []
            wind_speed = []
            vapor_pressure_deficit = []
            air_temperature = []
            soil_water_potential = []
            richardson = []
            monin_obukhov = []
            aerodynamic_resistance = []
            neutral_aerodynamic_resistance = []
            soil_abs_par = []
            veg_abs_par = []
            psi_u = []
            psi_h = []
            emissivity_sky = []
            is_forced_aerodynamic_resistance = []
            temp_sim = []
            for solver in plot_res['solvers']:
                incident_diffuse_par_irradiance.append(solver.crop.inputs.incident_irradiance['diffuse'])
                incident_direct_par_irradiance.append(solver.crop.inputs.incident_irradiance['direct'])
                wind_speed.append(solver.crop.inputs.wind_speed / 3600.)
                vapor_pressure_deficit.append(solver.crop.inputs.vapor_pressure_deficit)
                air_temperature.append(solver.crop.inputs.air_temperature - 273.15)
                soil_water_potential.append(solver.crop.inputs.soil_water_potential)
                richardson.append(solver.crop.state_variables.richardson_number)
                monin_obukhov.append(solver.crop.state_variables.monin_obukhov_length)
                aerodynamic_resistance.append(solver.crop.state_variables.aerodynamic_resistance * 3600.)
                neutral_aerodynamic_resistance.append(calc_neutral_aerodynamic_resistance(solver=solver))
                soil_abs_par.append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                veg_abs_par.append(calc_abs_irradiance(solver=solver))
                psi_u.append(solver.crop.state_variables.stability_correction_for_momentum)
                psi_h.append(solver.crop.state_variables.stability_correction_for_heat)
                emissivity_sky.append(solver.crop.params.simulation.atmospheric_emissivity)
                is_forced_aerodynamic_resistance.append(solver.is_forced_aerodynamic_resistance)
                temp_sim.append(calc_apparent_temperature(eb_solver=solver, date_obs=d1))

            all_sim_t += temp_sim
            all_obs_t += temp_obs
            all_air_t += air_temperature

            axs[0, 0].plot(x_ls, incident_diffuse_par_irradiance, label=r'$\mathregular{{PAR}_{diff}}$')
            axs[0, 0].plot(x_ls, incident_direct_par_irradiance, label=r'$\mathregular{{PAR}_{dir}}$')
            axs[0, 0].plot(x_ls, soil_abs_par, label=r'$\mathregular{{PAR}_{abs,\/soil}}$', linewidth=2)
            axs[0, 0].plot(x_ls, veg_abs_par, label=r'$\mathregular{{PAR}_{abs,\/veg}}$', c='k', linewidth=3)
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
            axs[2, 1].scatter(x_ls, [v if v else None for v in is_forced_aerodynamic_resistance], label='forced')
            axs[2, 1].set_ylim(-5, 60)
            axs[2, 1].legend(fontsize='x-small')

            axs[1, 0].plot(x_ls, vapor_pressure_deficit, label='VPD')
            axs[1, 0].set_ylim(0, 6)
            axs[1, 0].legend()

            axs[2, 0].plot(x_ls, soil_water_potential, label=r'$\mathregular{\Psi_{soil}}$')
            axs[2, 0].legend()

            axs[0, 3].scatter(temp_obs, temp_sim, alpha=0.5)
            axs[0, 3].plot((-5, 40), (-5, 40), 'k--')
            axs[0, 3].set(xlim=(-5, 40), ylim=(-5, 40), xlabel='obs T', ylabel='sim T')
            roughness_length_for_momentum = plot_res["solvers"][0].crop.state_variables.roughness_length_for_momentum
            canopy_height = plot_res["solvers"][0].crop.inputs.canopy_height
            zero_displacement_height = plot_res["solvers"][0].crop.state_variables.zero_displacement_height
            axs[0, 3].text(0.1, 0.8, f'z0u/h ={roughness_length_for_momentum / canopy_height:.3f}',
                           transform=axs[0, 3].transAxes)
            axs[0, 3].text(0.1, 0.6, f'd/h ={zero_displacement_height / canopy_height:.3f}',
                           transform=axs[0, 3].transAxes)
            axs[0, 3].yaxis.set_label_position("right")
            axs[0, 3].yaxis.tick_right()

            axs[1, 3].scatter(
                [t_obs - t_air for t_obs, t_air in zip(temp_obs, air_temperature)],
                [t_sim - t_air for t_sim, t_air in zip(temp_sim, air_temperature)],
                alpha=0.5)
            axs[1, 3].plot((-15, 21), (-15, 21), 'k--')
            axs[1, 3].set(xlim=(-15, 21), ylim=(-15, 21), xlabel='obs Tcan-Tair', ylabel='sim Tcan-Tair')

            axs[1, 3].text(0.1, 0.8, f'h={plot_res["solvers"][0].crop.inputs.canopy_height}',
                           transform=axs[1, 3].transAxes)
            axs[1, 3].text(0.1, 0.6, f'LAI={sum(plot_res["solvers"][0].crop.inputs.leaf_layers.values()):.3f}',
                           transform=axs[1, 3].transAxes)

            axs[2, 2].plot(x_ls, aerodynamic_resistance, label=r'$\mathregular{r_{a,\/0}}$')
            axs[2, 2].plot(x_ls, neutral_aerodynamic_resistance, label=r'$\mathregular{r_{a,\/0,\/neutral}}$')
            axs[2, 2].scatter(x_ls, [v if v else None for v in is_forced_aerodynamic_resistance], label='forced')
            axs[2, 2].set(ylim=(0, 360), ylabel="s m-1")
            axs[2, 2].legend()

            axs[1, 2].plot(x_ls, psi_u, label=r'$\mathregular{\Psi_u}$')
            axs[1, 2].plot(x_ls, psi_h, label=r'$\mathregular{\Psi_v}$')
            axs[1, 2].legend()

            # axs[2, 3].scatter(richardson, monin_obukhov, marker='.', color='k')
            axs[2, 3].scatter(richardson, psi_u, label=r'$\mathregular{\Psi_u}$')
            axs[2, 3].scatter(richardson, psi_h, label=r'$\mathregular{\Psi_v}$')
            axs[2, 3].legend()

            axs[0, 2].plot(x_ls, emissivity_sky, label=r'$\mathregular{\epsilon_{sky}}$')
            axs[0, 2].set_ylim(0, 1)
            axs[0, 2].legend()

            for ax in axs[:, :3].flatten():
                ax.xaxis.set_major_locator(MultipleLocator(3))
                ax.grid()

            for ax in axs[:, 3]:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

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


def plot_errors(all_solvers: dict, path_figs: Path):
    summary = dict(
        temperature_air=[],
        temperature_obs=[],
        temperature_sim=[],

        incident_diffuse_par_irradiance=[],
        incident_direct_par_irradiance=[],
        wind_speed=[],
        vapor_pressure_deficit=[],
        soil_water_potential=[],
        richardson=[],
        monin_obukhov=[],
        aerodynamic_resistance=[],
        neutral_aerodynamic_resistance=[],
        soil_abs_par=[],
        veg_abs_par=[],
        psi_u=[],
        psi_h=[],
        hours=[],
        net_longwave_radiation=[],
        height=[],
        gai=[]
    )

    for d1, v1 in all_solvers.items():
        for plot_id, plot_res in v1.items():
            summary['temperature_obs'] += plot_res['temp_obs']

            for hour, solver in enumerate(plot_res['solvers']):
                summary['incident_diffuse_par_irradiance'].append(solver.crop.inputs.incident_irradiance['diffuse'])
                summary['incident_direct_par_irradiance'].append(solver.crop.inputs.incident_irradiance['direct'])
                summary['wind_speed'].append(solver.crop.inputs.wind_speed / 3600.)
                summary['vapor_pressure_deficit'].append(solver.crop.inputs.vapor_pressure_deficit)
                summary['temperature_air'].append(solver.crop.inputs.air_temperature - 273.15)
                summary['soil_water_potential'].append(solver.crop.inputs.soil_water_potential)
                summary['richardson'].append(solver.crop.state_variables.richardson_number)
                summary['monin_obukhov'].append(solver.crop.state_variables.monin_obukhov_length)
                summary['aerodynamic_resistance'].append(solver.crop.state_variables.aerodynamic_resistance * 3600.)
                summary['neutral_aerodynamic_resistance'].append(calc_neutral_aerodynamic_resistance(solver=solver))
                summary['soil_abs_par'].append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                summary['veg_abs_par'].append(calc_abs_irradiance(solver=solver))
                summary['psi_u'].append(solver.crop.state_variables.stability_correction_for_momentum)
                summary['psi_h'].append(solver.crop.state_variables.stability_correction_for_heat)
                summary['temperature_sim'].append(calc_apparent_temperature(eb_solver=solver, date_obs=d1))
                summary['hours'].append(hour)
                summary['net_longwave_radiation'].append(solver.crop.state_variables.net_longwave_radiation)
                summary['height'].append(solver.crop.inputs.canopy_height)
                summary['gai'].append(sum(solver.crop.inputs.leaf_layers.values()))

    summary.update({'temperature_error': [t_sim - t_obs for t_sim, t_obs in zip(summary['temperature_sim'],
                                                                                summary['temperature_obs'])]})

    n_rows = 3
    n_cols = 4
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8), sharex='all')
    for i, explanatory in enumerate(('wind_speed', 'vapor_pressure_deficit', 'temperature_air', 'soil_water_potential',
                                     'aerodynamic_resistance', 'soil_abs_par', 'veg_abs_par', 'hours',
                                     'net_longwave_radiation', 'height', 'gai')):
        ax = axs[i % n_rows, i // n_rows]
        ax.scatter(summary['temperature_error'], summary[explanatory], marker='.', alpha=0.2)
        ax.set(ylabel=explanatory)
    for ax in axs[-1, :]:
        ax.set_xlabel(r'$\mathregular{T_{sim}-T_{obs}\/[^\circ C]}$')
    fig.savefig(path_figs / 'errors.png')
    plt.close()



