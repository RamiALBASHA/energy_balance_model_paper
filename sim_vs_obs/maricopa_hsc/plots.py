from datetime import datetime
from math import radians
from pathlib import Path

import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator
from numpy import array, linspace
from pandas import DataFrame, isna

from sim_vs_obs.common import (get_canopy_abs_irradiance_from_solver, calc_apparent_temperature,
                               calc_neutral_aerodynamic_resistance, CMAP, NORM_INCIDENT_PAR, format_binary_colorbar)
from utils import stats, config

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
        ax.text(0.1, 0.9, f"R² = {stats.calc_r2(sim, obs):.3f}", transform=ax.transAxes)
        ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(sim, obs):.3f}", transform=ax.transAxes)
    if return_ax:
        return ax


def extract_results(all_solvers: dict) -> dict:
    all_sim_t = []
    all_obs_t = []
    all_air_t = []
    all_incident_par = []
    all_incident_direct_par = []
    all_incident_diffuse_par = []
    all_wind_speed = []
    all_vapor_pressure_deficit = []
    all_soil_water_potential = []
    all_richardson = []
    all_monin_obukhov = []
    all_aerodynamic_resistance = []
    all_neutral_aerodynamic_resistance = []
    all_friction_velocity = []
    all_soil_abs_par = []
    all_veg_abs_par = []
    all_psi_u = []
    all_psi_h = []
    all_net_longwave_radiation = []
    all_height = []
    all_gai = []
    all_hours = []

    for d1, v1 in all_solvers.items():
        for plot_id, plot_res in v1.items():

            temp_obs = plot_res['temp_obs']
            temp_sim = []
            air_temperature = []
            incident_par = []
            incident_diffuse_par_irradiance = []
            incident_direct_par_irradiance = []
            wind_speed = []
            vapor_pressure_deficit = []
            soil_water_potential = []
            richardson = []
            monin_obukhov = []
            aerodynamic_resistance = []
            neutral_aerodynamic_resistance = []
            friction_velocity = []
            soil_abs_par = []
            veg_abs_par = []
            psi_u = []
            psi_h = []
            net_longwave_radiation = []
            height = []
            gai = []
            hours = []

            sensor_angle_below_horizon = radians(45 if d1 < datetime(2008, 1, 2) else 30)

            for hour, solver in enumerate(plot_res['solvers']):
                incident_par.append(sum(solver.crop.inputs.incident_irradiance.values()))
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
                friction_velocity.append(solver.crop.state_variables.friction_velocity)
                soil_abs_par.append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                veg_abs_par.append(get_canopy_abs_irradiance_from_solver(solver=solver))
                psi_u.append(solver.crop.state_variables.stability_correction_for_momentum)
                psi_h.append(solver.crop.state_variables.stability_correction_for_heat)
                temp_sim.append(calc_apparent_temperature(eb_solver=solver, sensor_angle=sensor_angle_below_horizon))
                net_longwave_radiation.append(solver.crop.state_variables.net_longwave_radiation)
                height.append(solver.crop.inputs.canopy_height)
                gai.append(sum(solver.crop.inputs.leaf_layers.values()))
                hours.append(hour)

            all_sim_t += temp_sim
            all_obs_t += temp_obs
            all_air_t += air_temperature
            all_incident_par += incident_par
            all_incident_direct_par += incident_direct_par_irradiance
            all_incident_diffuse_par += incident_diffuse_par_irradiance
            all_wind_speed += wind_speed
            all_vapor_pressure_deficit += vapor_pressure_deficit
            all_soil_water_potential += soil_water_potential
            all_richardson += richardson
            all_monin_obukhov += monin_obukhov
            all_aerodynamic_resistance += aerodynamic_resistance
            all_neutral_aerodynamic_resistance += neutral_aerodynamic_resistance
            all_friction_velocity += friction_velocity
            all_soil_abs_par += soil_abs_par
            all_veg_abs_par += veg_abs_par
            all_psi_u += psi_u
            all_psi_h += psi_h
            all_net_longwave_radiation += net_longwave_radiation
            all_height += height
            all_gai += gai
            all_hours += hours

    return {
        'sim_t': all_sim_t,
        'obs_t': all_obs_t,
        'air_t': all_air_t,
        'par_inc': all_incident_par,
        'incident_direct_par_irradiance': all_incident_direct_par,
        'incident_diffuse_par_irradiance': all_incident_diffuse_par,
        'wind_speed': all_wind_speed,
        'vapor_pressure_deficit': all_vapor_pressure_deficit,
        'soil_water_potential': all_soil_water_potential,
        'richardson': all_richardson,
        'monin_obukhov': all_monin_obukhov,
        'aerodynamic_resistance': all_aerodynamic_resistance,
        'neutral_aerodynamic_resistance': all_neutral_aerodynamic_resistance,
        'friction_velocity': all_friction_velocity,
        'absorbed_par_soil': all_soil_abs_par,
        'absorbed_par_veg': all_veg_abs_par,
        'psi_u': all_psi_u,
        'psi_h': all_psi_h,
        'net_longwave_radiation': all_net_longwave_radiation,
        'height': all_height,
        'gai': all_gai,
        'hour': all_hours
    }


def plot_dynamic(all_solvers: dict, path_figs: Path):
    all_sim_t = []
    all_obs_t = []
    all_air_t = []
    all_incident_par = []

    x_ls = range(24)
    for d1, v1 in all_solvers.items():
        for plot_id, plot_res in v1.items():
            fig, axs = plt.subplots(3, 4, figsize=(12, 8))

            temp_obs = plot_res['temp_obs']
            incident_par = []
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

            sensor_angle_below_horizon = radians(45 if d1 < datetime(2008, 1, 2) else 30)

            for solver in plot_res['solvers']:
                incident_par.append(sum(solver.crop.inputs.incident_irradiance.values()))
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
                veg_abs_par.append(get_canopy_abs_irradiance_from_solver(solver=solver))
                psi_u.append(solver.crop.state_variables.stability_correction_for_momentum)
                psi_h.append(solver.crop.state_variables.stability_correction_for_heat)
                emissivity_sky.append(solver.crop.params.simulation.atmospheric_emissivity)
                is_forced_aerodynamic_resistance.append(solver.is_forced_aerodynamic_resistance)
                temp_sim.append(calc_apparent_temperature(eb_solver=solver, sensor_angle=sensor_angle_below_horizon))

            all_sim_t += temp_sim
            all_obs_t += temp_obs
            all_air_t += air_temperature
            all_incident_par += incident_par

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

    pass


def plot_summary(results: dict, path_figs: Path, is_colormap: bool = True):
    fig_summary, axs_summary = plt.subplots(ncols=2)
    df = DataFrame(data=results)
    df.loc[:, 'sim_delta_t'] = df['sim_t'] - df['air_t']
    df.loc[:, 'obs_delta_t'] = df['obs_t'] - df['air_t']

    im = None
    for ax, var_to_plot in zip(axs_summary, ('t', 'delta_t')):
        x = df[f'obs_{var_to_plot}'].tolist()
        y = df[f'sim_{var_to_plot}'].tolist()
        lims = [sorted(x + y)[i] for i in [0, -1]]
        if is_colormap:
            kwargs = dict(c=df['par_inc'].to_list(), alpha=0.5, marker='.', edgecolor='none',
                          cmap=CMAP, norm=NORM_INCIDENT_PAR)
        else:
            kwargs = dict(alpha=0.1, edgecolor='none')

        im = ax.scatter(x, y, **kwargs)
        ax.plot(lims, lims, 'k--')
        ax.text(0.1, 0.9, f"R² = {stats.calc_r2(x, y):.3f}", transform=ax.transAxes)
        ax.text(0.1, 0.8, f"RMSE = {stats.calc_rmse(x, y):.3f}", transform=ax.transAxes)
        ax.set(xlabel=' '.join(['obs'] + MAP_UNITS[var_to_plot]), ylabel=' '.join(['sim'] + MAP_UNITS[var_to_plot]))
        ax.set_aspect('equal')

    fig_summary.tight_layout()
    if is_colormap:
        fig_summary.subplots_adjust(bottom=0.25)
        cbar_ax = fig_summary.add_axes([0.37, 0.1, 0.30, 0.04])
        cbar = fig_summary.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar_ax.set_ylabel(' '.join(config.UNITS_MAP['incident_par']), va="top", ha='right', rotation=0)
        format_binary_colorbar(cbar)

    fig_summary.savefig(path_figs / 'sim_vs_obs.png')
    plt.close()

    pass


def plot_errors(all_solvers: dict, path_figs: Path, is_colormap: bool = True):
    summary = dict(
        temperature_air=[],
        temperature_obs=[],
        temperature_sim=[],

        incident_par=[],
        incident_diffuse_par_irradiance=[],
        incident_direct_par_irradiance=[],
        wind_speed=[],
        vapor_pressure_deficit=[],
        soil_water_potential=[],
        richardson=[],
        monin_obukhov=[],
        aerodynamic_resistance=[],
        neutral_aerodynamic_resistance=[],
        absorbed_par_soil=[],
        absorbed_par_veg=[],
        psi_u=[],
        psi_h=[],
        hour=[],
        net_longwave_radiation=[],
        height=[],
        gai=[],
    )

    for d1, v1 in all_solvers.items():
        for plot_id, plot_res in v1.items():
            summary['temperature_obs'] += plot_res['temp_obs']

            for hour, solver in enumerate(plot_res['solvers']):
                summary['incident_par'].append(sum(solver.crop.inputs.incident_irradiance.values()))
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
                summary['absorbed_par_soil'].append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                summary['absorbed_par_veg'].append(get_canopy_abs_irradiance_from_solver(solver=solver))
                summary['psi_u'].append(solver.crop.state_variables.stability_correction_for_momentum)
                summary['psi_h'].append(solver.crop.state_variables.stability_correction_for_heat)
                summary['temperature_sim'].append(calc_apparent_temperature(
                    eb_solver=solver, sensor_angle=radians(45 if d1 < datetime(2008, 1, 2) else 30)))
                summary['hour'].append(hour)
                summary['net_longwave_radiation'].append(solver.crop.state_variables.net_longwave_radiation)
                summary['height'].append(solver.crop.inputs.canopy_height)
                summary['gai'].append(sum(solver.crop.inputs.leaf_layers.values()))

    summary.update({'temperature_error': [t_sim - t_obs for t_sim, t_obs in zip(summary['temperature_sim'],
                                                                                summary['temperature_obs'])]})

    n_rows = 3
    n_cols = 4
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8), sharey='all')
    for i, explanatory in enumerate(('wind_speed', 'vapor_pressure_deficit', 'temperature_air', 'soil_water_potential',
                                     'aerodynamic_resistance', 'absorbed_par_soil', 'absorbed_par_veg', 'hour',
                                     'net_longwave_radiation', 'height', 'gai')):
        ax = axs[i % n_rows, i // n_rows]

        explanatory_ls, error_ls, idx = zip(
            *[(ex, er, j) for j, (ex, er) in enumerate(zip(summary[explanatory], summary['temperature_error']))
              if not any(isna([ex, er]))])

        if is_colormap:
            kwargs = dict(c=summary['incident_par'], alpha=0.5, cmap=CMAP, norm=NORM_INCIDENT_PAR)
        else:
            kwargs = dict(alpha=0.2)

        im = ax.scatter(explanatory_ls, error_ls, marker='.', edgecolor='none', **kwargs)
        ax.set(xlabel=' '.join(config.UNITS_MAP[explanatory]))

        x = array(explanatory_ls)
        x = sm.add_constant(x)
        y = array(error_ls)
        results = sm.OLS(y, x).fit()

        ax.plot(*zip(*[(i, results.params[0] + results.params[1] * i) for i in
                       linspace(min(explanatory_ls), max(explanatory_ls), 2)]), 'k--')
        p_value_slope = results.pvalues[1] / 2.
        ax.text(0.1, 0.9, '*' if p_value_slope < 0.05 else '', transform=ax.transAxes, fontweight='bold')

    axs[1, 0].set_ylabel(r'$\mathregular{T_{sim}-T_{obs}\/[^\circ C]}$', fontsize=16)
    if is_colormap:
        cbar = fig.colorbar(im, ax=axs.flatten()[-1], orientation='horizontal')
        cbar.set_label(' '.join(config.UNITS_MAP['incident_par']))
        format_binary_colorbar(cbar=cbar)

    fig.tight_layout()
    fig.savefig(path_figs / 'errors.png')
    plt.close()


def export_results(results: dict, path_csv: Path):
    df = DataFrame(data=results)
    df.rename({'sim_t': 'temperature_canopy_sim',
               'obs_t': 'temperature_canopy_obs',
               'air_t': 'temperature_air',
               'par_inc': 'incident_par'},
              axis=1, inplace=True)
    df.loc[:, 'delta_temperature_canopy_sim'] = df['temperature_canopy_sim'] - df['temperature_air']
    df.loc[:, 'delta_temperature_canopy_obs'] = df['temperature_canopy_obs'] - df['temperature_air']
    df.dropna(inplace=True)

    df.to_csv(path_csv / 'results.csv', index=False)
    pass


def export_results_cart(summary_data: dict, path_csv: Path):
    df = DataFrame(summary_data)
    df.rename({'sim_t': 'temperature_canopy_sim',
               'obs_t': 'temperature_canopy_obs',
               'air_t': 'temperature_air',
               'par_inc': 'incident_par'},
              axis=1, inplace=True)

    df.loc[:, f'error_temperature_canopy'] = df[f'temperature_canopy_sim'] - df[f'temperature_canopy_obs']

    df = df[(df['incident_par'] >= 0) & ~df['error_temperature_canopy'].isna()]
    df.to_csv(path_csv / 'results_cart.csv', index=False)
    return
