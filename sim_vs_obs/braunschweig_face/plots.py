from pathlib import Path

import statsmodels.api as sm
from matplotlib import pyplot
from numpy import array, linspace
from pandas import isna

from sim_vs_obs.braunschweig_face.config import ExpInfos
from sim_vs_obs.common import calc_apparent_temperature, get_canopy_abs_irradiance_from_solver
from utils import config
from utils.stats import calc_rmse, calc_r2


def regroup_dates(dates: list) -> dict:
    daily = set([d.date() for d in dates])
    return {d1: [d2 for d2 in dates if d2.date() == d1] for d1 in daily}


def plot_dynamic_result(sim_obs: dict, path_figs: Path):
    for trt_id in sim_obs.keys():
        for rep_id, v in sim_obs[trt_id].items():
            dates = regroup_dates(dates=list(v.keys()))
            for date_sim in dates.keys():
                fig, (ax_dynamic, ax_11) = pyplot.subplots(ncols=2)
                temperature_sim = []
                temperature_obs = []
                dates_hourly = dates[date_sim]
                for datetime_sim in dates_hourly:
                    res = v[datetime_sim]
                    temperature_obs.append(res['obs_temperature'])
                    temperature_sim.append(calc_apparent_temperature(
                        eb_solver=res['solver'],
                        sensor_angle=ExpInfos.irt_angle_below_horizon.value))
                ax_dynamic.scatter(dates_hourly, temperature_obs, label='obs', c='orange')
                ax_dynamic.plot(dates_hourly, temperature_sim, label='sim')
                ax_11.scatter(temperature_obs, temperature_sim)
                lims = [sorted([v for v in (temperature_obs + temperature_sim) if v is not None])[i] for i in (0, -1)]
                ax_11.plot(lims, lims, 'k--')
                fig.savefig(path_figs / f'{trt_id}_{rep_id}_{date_sim}.png')
                pyplot.close('all')


def plot_all_1_1(sim_obs: dict, path_figs: Path):
    fig, (ax, ax_t_diff) = pyplot.subplots(ncols=2)

    temperature_sim = []
    temperature_obs = []
    temperature_air = []
    for trt_id in sim_obs.keys():
        for rep_id in sim_obs[trt_id].keys():
            for datetime_sim, res in sim_obs[trt_id][rep_id].items():
                solver = res['solver']
                temperature_obs.append(res['obs_temperature'])
                temperature_sim.append(calc_apparent_temperature(
                    eb_solver=solver,
                    sensor_angle=ExpInfos.irt_angle_below_horizon.value))
                temperature_air.append(solver.inputs.air_temperature - 273.15)
    ax.scatter(temperature_obs, temperature_sim, alpha=0.1)
    temperature_obs, temperature_sim, temperature_air = zip(
        *[(x, y, z) for x, y, z in zip(temperature_obs, temperature_sim, temperature_air) if
          ((x is not None) and (y is not None))])
    ax.text(0.05, 0.9,
            ''.join([r'$\mathregular{R^2=}$', f'{calc_r2(temperature_obs, temperature_sim):.3f}']),
            transform=ax.transAxes)
    ax.text(0.05, 0.8, f'RMSE={calc_rmse(temperature_obs, temperature_sim):.3f} °C',
            transform=ax.transAxes)

    lims = [sorted([v for v in (temperature_obs + temperature_sim) if v is not None])[i] for i in (0, -1)]
    ax.plot(lims, lims, 'k--')

    delta_t_sim = [t_sim - t_air for t_sim, t_air in zip(temperature_sim, temperature_air)]
    delta_t_obs = [t_obs - t_air for t_obs, t_air in zip(temperature_obs, temperature_air)]
    ax_t_diff.scatter(delta_t_obs, delta_t_sim, alpha=0.1)
    lims = [sorted([v for v in (delta_t_obs + delta_t_sim) if v is not None])[i] for i in (0, -1)]
    ax_t_diff.plot(lims, lims, 'k--')
    ax_t_diff.text(0.05, 0.9,
                   ''.join([r'$\mathregular{R^2=}$', f'{calc_r2(delta_t_obs, delta_t_sim):.3f}']),
                   transform=ax_t_diff.transAxes)
    ax_t_diff.text(0.05, 0.8, f'RMSE={calc_rmse(delta_t_obs, delta_t_sim):.3f} °C',
                   transform=ax_t_diff.transAxes)

    fig.savefig(path_figs / f'all_sim_obs.png')
    pyplot.close('all')
    pass


def extract_sim_obs_data(sim_obs: dict):
    all_t_air = []
    all_t_can = {'sim': [], 'obs': []}

    all_incident_par_irradiance = []
    all_incident_diffuse_par_irradiance = []
    all_incident_direct_par_irradiance = []
    all_wind_speed = []
    all_vapor_pressure_deficit = []
    all_soil_water_potential = []
    all_richardson = []
    all_monin_obukhov = []
    all_aerodynamic_resistance = []
    all_veg_abs_par = []
    all_soil_abs_par = []
    all_psi_u = []
    all_psi_h = []
    all_hours = []
    all_net_longwave_radiation = []
    all_height = []
    all_gai = []

    for trt_id in sim_obs.keys():
        for rep_id in sim_obs[trt_id].keys():
            for datetime_sim, res in sim_obs[trt_id][rep_id].items():
                solver = res['solver']
                all_t_can['obs'].append(res['obs_temperature'])
                all_t_can['sim'].append(calc_apparent_temperature(
                    eb_solver=solver,
                    sensor_angle=ExpInfos.irt_angle_below_horizon.value))
                all_t_air.append(solver.inputs.air_temperature - 273.15)
                all_incident_par_irradiance.append(sum(solver.crop.inputs.incident_irradiance.values()))
                all_incident_diffuse_par_irradiance.append(solver.crop.inputs.incident_irradiance['diffuse'])
                all_incident_direct_par_irradiance.append(solver.crop.inputs.incident_irradiance['direct'])
                all_wind_speed.append(solver.crop.inputs.wind_speed / 3600.)
                all_vapor_pressure_deficit.append(solver.crop.inputs.vapor_pressure_deficit)
                all_soil_water_potential.append(solver.crop.inputs.soil_water_potential)
                all_richardson.append(solver.crop.state_variables.richardson_number)
                all_monin_obukhov.append(solver.crop.state_variables.monin_obukhov_length)
                all_aerodynamic_resistance.append(solver.crop.state_variables.aerodynamic_resistance)
                all_veg_abs_par.append(get_canopy_abs_irradiance_from_solver(solver))
                all_soil_abs_par.append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                all_psi_u.append(solver.crop.state_variables.stability_correction_for_momentum)
                all_psi_h.append(solver.crop.state_variables.stability_correction_for_heat)
                all_hours.append(datetime_sim.hour)
                all_net_longwave_radiation.append(solver.crop.state_variables.net_longwave_radiation)
                all_height.append(solver.crop.inputs.canopy_height)
                all_gai.append(sum(solver.crop.inputs.leaf_layers.values()))

    return dict(
        temperature_air=all_t_air,
        temperature_canopy=all_t_can,
        incident_par=all_incident_par_irradiance,
        incident_diffuse_par_irradiance=all_incident_diffuse_par_irradiance,
        incident_direct_par_irradiance=all_incident_direct_par_irradiance,
        wind_speed=all_wind_speed,
        vapor_pressure_deficit=all_vapor_pressure_deficit,
        soil_water_potential=all_soil_water_potential,
        richardson=all_richardson,
        monin_obukhov=all_monin_obukhov,
        aerodynamic_resistance=all_aerodynamic_resistance,
        absorbed_par_veg=all_veg_abs_par,
        absorbed_par_soil=all_soil_abs_par,
        psi_u=all_psi_u,
        psi_h=all_psi_h,
        hour=all_hours,
        net_longwave_radiation=all_net_longwave_radiation,
        height=all_height,
        gai=all_gai)


def plot_error(sim_obs: dict, path_figs: Path):
    res = extract_sim_obs_data(sim_obs=sim_obs)

    n_rows = 3
    n_cols = 4
    fig, axs = pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8), sharey='all')

    idx, sim, obs = zip(*[(i, i_sim, i_obs) for i, (i_sim, i_obs) in
                          enumerate(zip(res['temperature_canopy']['sim'], res['temperature_canopy']['obs']))
                          if not any(isna([i_sim, i_obs]))])

    error = [i_sim - i_obs for i_sim, i_obs in zip(sim, obs)]

    for i_explanatory, explanatory in enumerate(
            ('wind_speed', 'vapor_pressure_deficit', 'temperature_air', 'soil_water_potential',
             'aerodynamic_resistance', 'absorbed_par_soil', 'absorbed_par_veg', 'hour',
             'net_longwave_radiation', 'gai')):
        ax = axs[i_explanatory % n_rows, i_explanatory // n_rows]
        explanatory_ls = [res[explanatory][i] for i in idx]
        ax.scatter(explanatory_ls, error, marker='.', edgecolor=None, alpha=0.2)
        ax.set(xlabel=' '.join(config.UNITS_MAP[explanatory]))

        x = array(explanatory_ls)
        x = sm.add_constant(x)
        y = array(error)
        results = sm.OLS(y, x).fit()

        ax.plot(*zip(*[(i, results.params[0] + results.params[1] * i) for i in
                       linspace(min(explanatory_ls), max(explanatory_ls), 2)]), 'k--')
        p_value_slope = results.pvalues[1] / 2.
        ax.text(0.1, 0.9, '*' if p_value_slope < 0.05 else '', transform=ax.transAxes, fontweight='bold')

    title = ' '.join(config.UNITS_MAP['temperature_canopy'])
    axs[1, 0].set_ylabel(' '.join((r'$\mathregular{\epsilon}$', title)), fontsize=16)
    fig.tight_layout()
    fig.savefig(path_figs / f'errors_temperature_canopy.png')
    pyplot.close()

    pass
