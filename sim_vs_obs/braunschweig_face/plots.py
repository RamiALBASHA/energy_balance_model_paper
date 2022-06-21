from pathlib import Path

import statsmodels.api as sm
from crop_energy_balance.formalisms.weather import calc_vapor_pressure_deficit
from matplotlib import pyplot
from numpy import array, linspace
from pandas import isna, DataFrame

from sim_vs_obs.braunschweig_face.base_functions import read_weather
from sim_vs_obs.braunschweig_face.config import ExpInfos
from sim_vs_obs.common import (calc_apparent_temperature, calc_neutral_aerodynamic_resistance,
                               get_canopy_abs_irradiance_from_solver, NORM_INCIDENT_PAR, CMAP, format_binary_colorbar)
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


def plot_all_1_1(sim_obs: dict, path_figs: Path, add_color_map: bool = True):
    fig, (ax, ax_t_diff) = pyplot.subplots(ncols=2)

    temperature_sim = []
    temperature_obs = []
    temperature_air = []
    incident_irradiance = []
    for trt_id in sim_obs.keys():
        for rep_id in sim_obs[trt_id].keys():
            for datetime_sim, res in sim_obs[trt_id][rep_id].items():
                solver = res['solver']
                temperature_obs.append(res['obs_temperature'])
                temperature_sim.append(calc_apparent_temperature(
                    eb_solver=solver,
                    sensor_angle=ExpInfos.irt_angle_below_horizon.value))
                temperature_air.append(solver.inputs.air_temperature - 273.15)
                incident_irradiance.append(sum(solver.crop.inputs.incident_irradiance.values()))
    if add_color_map:
        ax.scatter(temperature_obs, temperature_sim, marker='.', edgecolor='none', alpha=0.5,
                   c=incident_irradiance, cmap=CMAP)
    else:
        ax.scatter(temperature_obs, temperature_sim, alpha=0.5, edgecolor='none')

    temperature_obs, temperature_sim, temperature_air, incident_irradiance = zip(
        *[(x, y, z, r) for x, y, z, r in zip(temperature_obs, temperature_sim, temperature_air, incident_irradiance) if
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
    if add_color_map:
        im = ax_t_diff.scatter(delta_t_obs, delta_t_sim, marker='.', edgecolor='none', alpha=0.5,
                               c=incident_irradiance, norm=NORM_INCIDENT_PAR, cmap=CMAP)
    else:
        ax_t_diff.scatter(delta_t_obs, delta_t_sim, alpha=0.5)
    lims = [sorted([v for v in (delta_t_obs + delta_t_sim) if v is not None])[i] for i in (0, -1)]
    ax_t_diff.plot(lims, lims, 'k--')
    ax_t_diff.text(0.05, 0.9,
                   ''.join([r'$\mathregular{R^2=}$', f'{calc_r2(delta_t_obs, delta_t_sim):.3f}']),
                   transform=ax_t_diff.transAxes)
    ax_t_diff.text(0.05, 0.8, f'RMSE={calc_rmse(delta_t_obs, delta_t_sim):.3f} °C',
                   transform=ax_t_diff.transAxes)

    if add_color_map:
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.37, 0.1, 0.30, 0.04])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=' '.join(config.UNITS_MAP['incident_par']))
        format_binary_colorbar(cbar=cbar)

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
    all_neutral_aerodynamic_resistance = []
    all_friction_velocity = []
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
                all_neutral_aerodynamic_resistance.append(calc_neutral_aerodynamic_resistance(solver=solver))
                all_friction_velocity.append(solver.crop.state_variables.friction_velocity)
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
        neutral_aerodynamic_resistance=all_neutral_aerodynamic_resistance,
        friction_velocity=all_friction_velocity,
        absorbed_par_veg=all_veg_abs_par,
        absorbed_par_soil=all_soil_abs_par,
        psi_u=all_psi_u,
        psi_h=all_psi_h,
        hour=all_hours,
        net_longwave_radiation=all_net_longwave_radiation,
        height=all_height,
        gai=all_gai)


def plot_error(summary_data: dict, path_figs: Path, add_colormap: bool = True):
    n_rows = 3
    n_cols = 4
    fig, axs = pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8), sharey='all')
    im = None
    idx, sim, obs = zip(*[(i, i_sim, i_obs) for i, (i_sim, i_obs) in
                          enumerate(
                              zip(summary_data['temperature_canopy']['sim'], summary_data['temperature_canopy']['obs']))
                          if not any(isna([i_sim, i_obs]))])

    error = [i_sim - i_obs for i_sim, i_obs in zip(sim, obs)]
    c = [summary_data['incident_par'][i] for i in idx]
    explanatory_vars = ('wind_speed', 'vapor_pressure_deficit', 'temperature_air', 'soil_water_potential',
                        'aerodynamic_resistance', 'absorbed_par_soil', 'absorbed_par_veg', 'hour',
                        'net_longwave_radiation', 'gai')
    for i_explanatory, explanatory in enumerate(explanatory_vars):
        ax = axs[i_explanatory % n_rows, i_explanatory // n_rows]
        explanatory_ls = [summary_data[explanatory][i] for i in idx]
        if add_colormap:
            im = ax.scatter(explanatory_ls, error, marker='.', edgecolor='none', alpha=0.5, c=c, norm=NORM_INCIDENT_PAR,
                            cmap=CMAP)
        else:
            ax.scatter(explanatory_ls, error, marker='.', edgecolor='none', alpha=0.5)

        ax.set(xlabel=' '.join(config.UNITS_MAP[explanatory]))

        x = array(explanatory_ls)
        x = sm.add_constant(x)
        y = array(error)
        results = sm.OLS(y, x).fit()

        ax.plot(*zip(*[(i, results.params[0] + results.params[1] * i) for i in
                       linspace(min(explanatory_ls), max(explanatory_ls), 2)]), 'k--')
        p_value_slope = results.pvalues[1] / 2.
        ax.text(0.1, 0.9, '*' if p_value_slope < 0.05 else '', transform=ax.transAxes, fontweight='bold')

        if add_colormap and (i_explanatory == len(explanatory_vars) - 1):
            cbar = fig.colorbar(im, ax=axs.flatten()[-1], label=' '.join(config.UNITS_MAP['incident_par']),
                                orientation='horizontal')
            format_binary_colorbar(cbar=cbar)

    title = ' '.join(config.UNITS_MAP['temperature_canopy'])
    axs[1, 0].set_ylabel(' '.join((r'$\mathregular{\epsilon}$', title)), fontsize=16)
    fig.tight_layout()
    fig.savefig(path_figs / f'errors_temperature_canopy.png')
    pyplot.close('all')

    pass


def export_results(summary_data: dict, path_csv: Path):
    df = DataFrame(data={
        'temperature_canopy_sim': summary_data['temperature_canopy']['sim'],
        'temperature_canopy_obs': summary_data['temperature_canopy']['obs'],
        'temperature_air': summary_data['temperature_air'],
        'incident_par': summary_data['incident_par']})
    df.loc[:, 'delta_temperature_canopy_sim'] = df['temperature_canopy_sim'] - df['temperature_air']
    df.loc[:, 'delta_temperature_canopy_obs'] = df['temperature_canopy_obs'] - df['temperature_air']
    df.dropna(inplace=True)
    df.to_csv(path_csv / 'results.csv', index=False)

    pass


def export_results_cart(summary_data: dict, path_csv: Path):
    res = {k: v for k, v in summary_data.items() if k != 'temperature_canopy'}
    df = DataFrame(res)
    for s in ('sim', 'obs'):
        df.loc[:, f'temperature_canopy_{s}'] = summary_data['temperature_canopy'][s]
    df.loc[:, f'error_temperature_canopy'] = df[f'temperature_canopy_sim'] - df[f'temperature_canopy_obs']

    df.loc[:, 'incident_par'] = df[['incident_direct_par_irradiance', 'incident_diffuse_par_irradiance']].sum(axis=1)
    df = df[(df['incident_par'] >= 0) & ~df['error_temperature_canopy'].isna()]
    df.to_csv(path_csv / 'results_cart.csv', index=False)
    return df


def export_weather_summary(path_csv: Path):
    df = DataFrame(
        {s: [None] for s in
         ('year', 'air_temperature_avg', 'global_radiation_cum', 'vapor_pressure_deficit_avg', 'rainfall_cum')})
    df.set_index('year', inplace=True)
    for year in ExpInfos.years.value:
        weather_df = read_weather(year=year)
        weather_df.loc[:, 'vapor_pressure_deficit'] = weather_df.apply(
            lambda x: calc_vapor_pressure_deficit(
                temperature_air=x['TEMP'],
                temperature_leaf=x['TEMP'],
                relative_humidity=x['RH']),
            axis=1)
        weather_df.loc[:, 'SRAD'] = weather_df.apply(lambda x: max(0, x['SRAD']), axis=1)
        df.loc[int(year), ['air_temperature_avg', 'vapor_pressure_deficit_avg']] = \
            weather_df.groupby(weather_df.index.date).mean().mean()[
                ['TEMP', 'vapor_pressure_deficit']].to_list()
        df.loc[int(year), ['global_radiation_cum', 'rainfall_cum']] = \
            weather_df.groupby(weather_df.index.date).sum().mean()[
                ['SRAD', 'RAIN']].to_list()

    df.dropna(inplace=True)
    df.loc['avg', :] = df.mean()
    df.to_csv(path_csv / 'weather_summary.csv', sep=';', decimal='.')
    pass
