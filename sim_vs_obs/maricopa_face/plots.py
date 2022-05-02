from copy import deepcopy
from pathlib import Path

import statsmodels.api as sm
from matplotlib import pyplot
from numpy import array, linspace
from pandas import DataFrame, isna

from sim_vs_obs.common import get_canopy_abs_irradiance_from_solver, calc_apparent_temperature
from sim_vs_obs.maricopa_face import base_functions
from sim_vs_obs.maricopa_face.config import SensorInfos, PathInfos
from utils import stats, config


def add_1_1_line(ax):
    lims = [sorted(list(ax.get_xlim()) + list(ax.get_ylim()))[idx] for idx in (0, -1)]
    ax.plot(lims, lims, 'k--', label='1:1')
    return ax


def get_dates(t_ls: list) -> list:
    return sorted(list(set(t.date() for t in t_ls)))


def calc_diff(sim_obs: dict, idx: int) -> float:
    obs = sim_obs['obs'][idx]
    obs = sum(obs) / len(obs) if isinstance(obs, list) else obs
    return sim_obs['sim'][idx] - obs if obs is not None else None


def plot_comparison_energy_balance(sim_obs: dict, figure_dir: Path):
    counter = 0
    for trt_id, trt_obs in sim_obs.items():
        for date_obs in get_dates(trt_obs.keys()):

            datetime_obs_ls = sorted(v for v in trt_obs.keys() if v.date() == date_obs)
            hours = [t.hour for t in datetime_obs_ls]
            nb_hours = len(hours)

            pattern_list = [None] * nb_hours
            pattern_dict = {k: pattern_list.copy() for k in ('sim', 'obs')}

            par_inc = pattern_list.copy()
            par_abs_veg = pattern_list.copy()
            par_abs_sol = pattern_list.copy()
            vpd = pattern_list.copy()
            wind = pattern_list.copy()
            ra = pattern_list.copy()
            psi_soil = pattern_list.copy()
            t_air = pattern_list.copy()
            t_can = deepcopy(pattern_dict)
            t_soil = deepcopy(pattern_dict)
            net_radiation = deepcopy(pattern_dict)
            sensible_heat = deepcopy(pattern_dict)
            latent_heat = deepcopy(pattern_dict)
            soil_heat = deepcopy(pattern_dict)

            t_sunlit = deepcopy(pattern_dict)
            t_shaded = deepcopy(pattern_dict)
            t_soil2 = deepcopy(pattern_dict)

            gai = sum(trt_obs[datetime_obs_ls[0]]['solver'].crop.inputs.leaf_layers.values())

            for i, dt_obs in enumerate(datetime_obs_ls):
                solver, obs, obs2 = [trt_obs[dt_obs][s] for s in ('solver', 'obs_energy_balance', 'obs_sunlit_shaded')]

                par_inc[i] = sum(solver.crop.inputs.incident_irradiance.values())
                par_abs_veg[i] = get_canopy_abs_irradiance_from_solver(solver)
                par_abs_sol[i] = solver.crop.inputs.absorbed_irradiance[-1]['lumped']
                vpd[i] = solver.crop.inputs.vapor_pressure_deficit
                wind[i] = solver.crop.inputs.wind_speed / 3600.
                ra[i] = solver.crop.state_variables.aerodynamic_resistance * 3600.
                psi_soil[i] = solver.crop.inputs.soil_water_potential
                t_air[i] = solver.crop.inputs.air_temperature - 273.15

                t_can['sim'][i] = calc_apparent_temperature(solver, SensorInfos.irt_angle_below_horizon.value)
                t_soil['sim'][i] = solver.crop[-1].temperature - 273.15
                net_radiation['sim'][i] = solver.crop.state_variables.net_radiation
                sensible_heat['sim'][i] = solver.crop.state_variables.sensible_heat_flux
                latent_heat['sim'][i] = solver.crop.state_variables.total_penman_monteith_evaporative_energy
                soil_heat['sim'][i] = solver.crop[-1].heat_flux

                max_layer_index = max(solver.crop.keys())
                if solver.crop.leaves_category == 'sunlit-shaded':
                    t_sunlit['sim'][i] = solver.crop[max_layer_index]['sunlit'].temperature - 273.15
                    t_shaded['sim'][i] = solver.crop[max_layer_index]['shaded'].temperature - 273.15
                    t_soil2['sim'][i] = solver.crop[-1].temperature - 273.15

                if obs is not None:
                    i_latent_heat = [i_l for i_l in obs['L'] if i_l >= 0]
                    if len(i_latent_heat) > 0:
                        latent_heat['obs'][i] = i_latent_heat
                        t_can['obs'][i] = obs['CT']
                        t_soil['obs'][i] = obs['ST.1']
                        net_radiation['obs'][i] = obs['Rn']
                        sensible_heat['obs'][i] = obs['H']
                        soil_heat['obs'][i] = obs['G']

                if obs2 is not None:
                    t_sunlit['obs'][i] = obs2['sunlit']
                    t_shaded['obs'][i] = obs2['shaded']
                    t_soil2['obs'][i] = obs2['soil']

            plot_daily_dynamic(f'{trt_id}_{counter}', date_obs, trt_id, gai, hours, par_inc, par_abs_veg, par_abs_sol,
                               vpd, t_air, psi_soil, wind, ra, t_can, t_soil, net_radiation, latent_heat, sensible_heat,
                               soil_heat, t_sunlit, t_shaded, t_soil2, figure_dir)
            counter += 1

    pass


def extract_sim_obs_data(sim_obs: dict):
    all_t_air = []
    all_t_can = {'sim': [], 'obs': []}
    all_t_soil = {'sim': [], 'obs': []}
    all_net_radiation = {'sim': [], 'obs': []}
    all_sensible_heat = {'sim': [], 'obs': []}
    all_latent_heat = {'sim': [], 'obs': []}
    all_soil_heat = {'sim': [], 'obs': []}

    all_t_sunlit = {'sim': [], 'obs': []}
    all_t_shaded = {'sim': [], 'obs': []}

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

    for trt_id, trt_obs in sim_obs.items():
        for date_obs in get_dates(trt_obs.keys()):

            datetime_obs_ls = sorted(v for v in trt_obs.keys() if v.date() == date_obs)
            hours = [t.hour for t in datetime_obs_ls]
            nb_hours = len(hours)

            pattern_list = [None] * nb_hours
            pattern_dict = {k: pattern_list.copy() for k in ('sim', 'obs')}

            par_inc = pattern_list.copy()
            par_inc_direct = pattern_list.copy()
            par_inc_diffuse = pattern_list.copy()
            par_abs_veg = pattern_list.copy()
            par_abs_sol = pattern_list.copy()
            vpd = pattern_list.copy()
            wind = pattern_list.copy()
            ra = pattern_list.copy()
            psi_soil = pattern_list.copy()
            t_air = pattern_list.copy()
            richardson = pattern_list.copy()
            monin_obukhov = pattern_list.copy()
            psi_u = pattern_list.copy()
            psi_h = pattern_list.copy()
            net_longwave_radiation = pattern_list.copy()
            heights = pattern_list.copy()
            gai = pattern_list.copy()

            t_can = deepcopy(pattern_dict)
            t_soil = deepcopy(pattern_dict)
            net_radiation = deepcopy(pattern_dict)
            sensible_heat = deepcopy(pattern_dict)
            latent_heat = deepcopy(pattern_dict)
            soil_heat = deepcopy(pattern_dict)

            t_sunlit = deepcopy(pattern_dict)
            t_shaded = deepcopy(pattern_dict)

            for i, dt_obs in enumerate(datetime_obs_ls):
                solver, obs, obs2 = [trt_obs[dt_obs][s] for s in ('solver', 'obs_energy_balance', 'obs_sunlit_shaded')]

                par_inc[i] = sum(solver.crop.inputs.incident_irradiance.values())
                par_inc_direct[i] = solver.crop.inputs.incident_irradiance['direct']
                par_inc_diffuse[i] = solver.crop.inputs.incident_irradiance['diffuse']
                par_abs_veg[i] = get_canopy_abs_irradiance_from_solver(solver)
                par_abs_sol[i] = solver.crop.inputs.absorbed_irradiance[-1]['lumped']
                vpd[i] = solver.crop.inputs.vapor_pressure_deficit
                wind[i] = solver.crop.inputs.wind_speed / 3600.
                ra[i] = solver.crop.state_variables.aerodynamic_resistance * 3600.
                psi_soil[i] = solver.crop.inputs.soil_water_potential
                t_air[i] = solver.crop.inputs.air_temperature - 273.15
                richardson[i] = solver.crop.state_variables.richardson_number
                monin_obukhov[i] = solver.crop.state_variables.monin_obukhov_length
                psi_u[i] = solver.crop.state_variables.stability_correction_for_momentum
                psi_h[i] = solver.crop.state_variables.stability_correction_for_heat
                net_longwave_radiation[i] = solver.crop.state_variables.net_longwave_radiation
                heights[i] = solver.crop.inputs.canopy_height
                gai[i] = sum(solver.crop.inputs.leaf_layers.values())

                t_can['sim'][i] = calc_apparent_temperature(solver, SensorInfos.irt_angle_below_horizon.value)
                t_soil['sim'][i] = solver.crop[-1].temperature - 273.15
                net_radiation['sim'][i] = solver.crop.state_variables.net_radiation
                sensible_heat['sim'][i] = solver.crop.state_variables.sensible_heat_flux
                latent_heat['sim'][i] = solver.crop.state_variables.total_penman_monteith_evaporative_energy
                soil_heat['sim'][i] = solver.crop[-1].heat_flux

                max_layer_index = max(solver.crop.keys())
                if solver.crop.leaves_category == 'sunlit-shaded':
                    t_sunlit['sim'][i] = solver.crop[max_layer_index]['sunlit'].temperature - 273.15
                    t_shaded['sim'][i] = solver.crop[max_layer_index]['shaded'].temperature - 273.15

                if obs is not None:
                    i_latent_heat = [i_l for i_l in obs['L'] if i_l >= 0]
                    if len(i_latent_heat) > 0:
                        latent_heat['obs'][i] = i_latent_heat
                        t_can['obs'][i] = obs['CT']
                        t_soil['obs'][i] = obs['ST.1']
                        net_radiation['obs'][i] = obs['Rn']
                        sensible_heat['obs'][i] = obs['H']
                        soil_heat['obs'][i] = obs['G']

                if obs2 is not None:
                    t_sunlit['obs'][i] = obs2['sunlit']
                    t_shaded['obs'][i] = obs2['shaded']

            all_incident_par_irradiance += par_inc
            all_incident_diffuse_par_irradiance += par_inc_direct
            all_incident_direct_par_irradiance += par_inc_diffuse
            all_wind_speed += wind
            all_vapor_pressure_deficit += vpd
            all_soil_water_potential += psi_soil
            all_richardson += richardson
            all_monin_obukhov += monin_obukhov
            all_aerodynamic_resistance += ra
            all_veg_abs_par += par_abs_veg
            all_soil_abs_par += par_abs_sol
            all_psi_u += psi_u
            all_psi_h += psi_h
            all_hours += hours
            all_net_longwave_radiation += net_longwave_radiation
            all_height += heights
            all_gai += gai

            all_t_can['sim'] += t_can['sim']
            all_t_soil['sim'] += t_soil['sim']
            all_net_radiation['sim'] += net_radiation['sim']
            all_sensible_heat['sim'] += sensible_heat['sim']
            all_latent_heat['sim'] += latent_heat['sim']
            all_soil_heat['sim'] += soil_heat['sim']
            all_t_sunlit['sim'] += t_sunlit['sim']
            all_t_shaded['sim'] += t_shaded['sim']

            all_t_can['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in t_can['obs']]
            all_t_soil['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in t_soil['obs']]
            all_net_radiation['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in net_radiation['obs']]
            all_sensible_heat['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in sensible_heat['obs']]
            all_latent_heat['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in latent_heat['obs']]
            all_soil_heat['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in soil_heat['obs']]
            all_t_sunlit['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in t_sunlit['obs']]
            all_t_shaded['obs'] += [sum(v) / len(v) if isinstance(v, list) else v for v in t_shaded['obs']]

            all_t_air += t_air

    return dict(
        temperature_air=all_t_air,
        temperature_canopy=all_t_can,
        temperature_soil=all_t_soil,
        net_radiation=all_net_radiation,
        sensible_heat_flux=all_sensible_heat,
        latent_heat_flux=all_latent_heat,
        soil_heat_flux=all_soil_heat,
        temperature_sunlit=all_t_sunlit,
        temperature_shaded=all_t_shaded,
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
        gai=all_gai,
    )


def plot_daily_dynamic(counter, date_obs, trt_id, gai, hours, par_inc, par_abs_veg, par_abs_sol, vpd, t_air, psi_soil,
                       wind, ra, t_can, t_soil, net_radiation, latent_heat, sensible_heat, soil_heat, t_sunlit,
                       t_shaded, t_soil2, figure_dir):
    props = {'marker': 'o', 'color': 'b', 'alpha': 0.1}
    fig, axs = pyplot.subplots(nrows=3, ncols=8, figsize=(18, 8))

    fig.suptitle(f'{date_obs} | TRNO:{trt_id} | GAI={gai: .2f}')

    axs[0, 0].plot(hours, par_inc, label=r'$\mathregular{{PAR}_{inc}}$')
    axs[0, 0].plot(hours, par_abs_veg, label=r'$\mathregular{{PAR}_{abs,\/veg}}$', linewidth=2)
    axs[0, 0].plot(hours, par_abs_sol, label=r'$\mathregular{{PAR}_{abs,\/sol}}$')
    axs[0, 0].legend()

    axs[1, 0].plot(hours, vpd, label=r'VPD')
    axs[1, 0].legend()

    axs[2, 0].plot(hours, t_air, label=r'$\mathregular{T_{air}}$')
    axs[2, 0].legend()

    axs[0, 1].plot(hours, psi_soil, label=r'$\mathregular{{\Psi}_{soil}}$')
    axs[0, 1].legend()

    axs[1, 1].plot(hours, wind, label='u')
    axs[1, 1].legend()

    axs[2, 1].plot(hours, ra, label='ra')
    axs[2, 1].legend()

    axs[0, 2].plot(hours, t_can['sim'], label=r'$\mathregular{T_{can,\/sim}}$')
    for h in hours:
        if t_can['obs'][h] is not None:
            for hobs in t_can['obs'][h]:
                axs[0, 2].scatter(h, hobs, label=r'$\mathregular{T_{can,\/obs}}$', **props)
                axs[0, 3].scatter(hobs, t_can['sim'][h], label=r'$\mathregular{T_{can}}$', **props)
    axs[0, 3] = add_1_1_line(ax=axs[0, 3])
    axs[0, 2].legend(*[v[:2] for v in axs[0, 2].get_legend_handles_labels()])
    axs[0, 3].legend(*[v[:2] for v in axs[0, 3].get_legend_handles_labels()])

    axs[1, 2].plot(hours, t_soil['sim'], label=r'$\mathregular{T_{soil,\/sim}}$')
    for h in hours:
        if t_soil['obs'][h] is not None:
            for hobs in t_soil['obs'][h]:
                axs[1, 2].scatter(h, hobs, label=r'$\mathregular{T_{soil,\/obs}}$', **props)
                axs[1, 3].scatter(hobs, t_soil['sim'][h], label=r'$\mathregular{T_{soil}}$', **props)
    axs[1, 3] = add_1_1_line(ax=axs[1, 3])
    axs[1, 2].legend(*[v[:2] for v in axs[1, 2].get_legend_handles_labels()])
    axs[1, 3].legend(*[v[:2] for v in axs[1, 3].get_legend_handles_labels()])

    axs[2, 2].plot(hours, t_soil2['sim'], label=r'$\mathregular{T_{soil,\/sim}}$')
    for h in hours:
        if t_soil2['obs'][h] is not None:
            axs[2, 2].scatter(h, t_soil2['obs'][h], label=r'$\mathregular{T_{soil,\/obs}}$', **props)
            axs[2, 3].scatter(t_soil2['obs'][h], t_soil2['sim'][h], label=r'$\mathregular{T_{soil}}$', **props)
    axs[2, 3] = add_1_1_line(ax=axs[2, 3])
    axs[2, 2].legend(*[v[:2] for v in axs[2, 2].get_legend_handles_labels()])
    axs[2, 3].legend(*[v[:2] for v in axs[2, 3].get_legend_handles_labels()])

    axs[0, 4].plot(hours, net_radiation['sim'], label=r'$\mathregular{{Rn}_{sim}}$')
    for h in hours:
        if net_radiation['obs'][h] is not None:
            for hobs in net_radiation['obs'][h]:
                axs[0, 4].plot(h, hobs, label=r'$\mathregular{{Rn}_{obs}}$', **props)
                axs[0, 5].scatter(hobs, net_radiation['sim'][h], label='Rn', **props)
    axs[0, 5] = add_1_1_line(ax=axs[0, 5])
    axs[0, 4].legend(*[v[:2] for v in axs[0, 4].get_legend_handles_labels()])
    axs[0, 5].legend(*[v[:2] for v in axs[0, 5].get_legend_handles_labels()])

    axs[1, 4].plot(hours, latent_heat['sim'], label=r'$\mathregular{{\lambda E}_{sim}}$')
    for h in hours:
        if latent_heat['obs'][h] is not None:
            for hobs in latent_heat['obs'][h]:
                axs[1, 4].scatter(h, hobs, label=r'$\mathregular{{\lambda E}_{obs}}$', **props)
                axs[1, 5].scatter(hobs, latent_heat['sim'][h], label=r'$\mathregular{\lambda E}$', **props)
    axs[1, 5] = add_1_1_line(ax=axs[1, 5])
    axs[1, 4].legend(*[v[:2] for v in axs[1, 4].get_legend_handles_labels()])
    axs[1, 5].legend(*[v[:2] for v in axs[1, 5].get_legend_handles_labels()])

    axs[2, 4].plot(hours, sensible_heat['sim'], label=r'$\mathregular{{H}_{sim}}$')
    for h in hours:
        if sensible_heat['obs'][h] is not None:
            for hobs in sensible_heat['obs'][h]:
                axs[2, 4].scatter(h, hobs, label=r'$\mathregular{{H}_{obs}}$', **props)
                axs[2, 5].scatter(hobs, sensible_heat['sim'][h], label=r'$\mathregular{H}$', **props)
    axs[2, 5] = add_1_1_line(ax=axs[2, 5])
    axs[2, 4].legend(*[v[:2] for v in axs[2, 4].get_legend_handles_labels()])
    axs[2, 5].legend(*[v[:2] for v in axs[2, 5].get_legend_handles_labels()])

    axs[0, 6].plot(hours, soil_heat['sim'], label=r'$\mathregular{G_{sim}}$')
    for h in hours:
        if soil_heat['obs'][h] is not None:
            for hobs in soil_heat['obs'][h]:
                axs[0, 6].scatter(h, hobs, label=r'$\mathregular{G_{obs}}$', **props)
                axs[0, 7].scatter(hobs, soil_heat['sim'][h], label='G', **props)
    axs[0, 7] = add_1_1_line(ax=axs[0, 7])
    axs[0, 6].legend(*[v[:2] for v in axs[0, 6].get_legend_handles_labels()])
    axs[0, 7].legend(*[v[:2] for v in axs[0, 7].get_legend_handles_labels()])

    axs[1, 6].plot(hours, t_sunlit['sim'], label=r'$\mathregular{T_{sunlit\/sim}}$', color='orange')
    axs[1, 6].plot(hours, t_shaded['sim'], label=r'$\mathregular{T_{shaded\/sim}}$', color='brown')
    for h in hours:
        if t_sunlit['obs'][h] is not None:
            axs[1, 6].scatter(h, t_sunlit['obs'][h], label=r'$\mathregular{T_{sunlit,\/obs}}$',
                              color='orange', marker='o')
            axs[1, 7].scatter(t_sunlit['obs'][h], t_sunlit['sim'][h], label=r'$\mathregular{T_{sunlit}}$',
                              color='orange', marker='o')
        if t_shaded['obs'][h] is not None:
            axs[1, 6].scatter(h, t_shaded['obs'][h], label=r'$\mathregular{T_{shaded,\/obs}}$',
                              color='brown', marker='o')
            axs[1, 7].scatter(t_shaded['obs'][h], t_shaded['sim'][h], label=r'$\mathregular{T_{shaded}}$',
                              color='brown', marker='o')
    axs[1, 7] = add_1_1_line(ax=axs[1, 7])
    axs[1, 6].legend(*[v[:2] for v in axs[1, 6].get_legend_handles_labels()])
    axs[1, 7].legend(*[v[:2] for v in axs[1, 7].get_legend_handles_labels()])

    fig.savefig(figure_dir / f'{counter}.png')
    pyplot.close('all')
    pass


def plot_sim_vs_obs(res_all: dict, res_wet: dict, res_dry: dict, figure_dir: Path, alpha: float = 0.1,
                    fig_name_suffix: str = ''):
    n_cols = len(res_all.keys())
    fig_size = [6.4, 4.8] if n_cols < 4 else [16, 4]
    fig, axs = pyplot.subplots(ncols=len(res_all.keys()), figsize=fig_size)
    for res, c in (res_wet, 'blue'), (res_dry, 'red'):
        for ax, (k, v) in zip(axs, res.items()):
            obs_ls, sim_ls = v['obs'], v['sim']
            # obs_ls, sim_ls = zip(*[(obs, sim) for obs, sim in zip(obs_ls, sim_ls) if not any(isna([obs, sim]))])
            ax.scatter(obs_ls, sim_ls, alpha=alpha)

    for ax, (k, v) in zip(axs, res_all.items()):
        obs_ls, sim_ls = v['obs'], v['sim']
        obs_ls, sim_ls = zip(*[(obs, sim) for obs, sim in zip(obs_ls, sim_ls) if not any(isna([obs, sim]))])
        ax.text(0.1, 0.9, f'RMSE={stats.calc_rmse(obs_ls, sim_ls):.3f}', transform=ax.transAxes)
        ax.text(0.1, 0.8, f'R²={stats.calc_r2(obs_ls, sim_ls):.3f}', transform=ax.transAxes)
        ax.set_title(k)
        add_1_1_line(ax)

    fig.tight_layout()
    fig.savefig(figure_dir / f'all_sim_vs_obs_{fig_name_suffix}.png')
    pyplot.close('all')

    pass


def plot_delta_temperature(temperature_air: list, temperature_canopy_sim: list, temperature_canopy_obs: list,
                           figure_dir: Path):
    fig, ax = pyplot.subplots()
    t_air, t_sim, t_obs = zip(*[(it_air, it_sim, it_obs) for it_air, it_sim, it_obs in
                                zip(temperature_air, temperature_canopy_sim, temperature_canopy_obs) if
                                not any(isna([it_air, it_sim, it_obs]))])
    delta_t_sim = [it_sim - it_air for it_sim, it_air in zip(t_sim, t_air)]
    delta_t_obs = [it_obs - it_air for it_obs, it_air in zip(t_obs, t_air)]

    ax.scatter(delta_t_obs, delta_t_sim, alpha=0.1)
    ax.text(0.1, 0.9, f'RMSE={stats.calc_rmse(delta_t_obs, delta_t_sim):.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.8, f'R²={stats.calc_r2(delta_t_obs, delta_t_sim):.3f}', transform=ax.transAxes)
    add_1_1_line(ax)
    ax.set(xlabel=r'$\mathregular{T_{obs}-T_{air}\/[^\circ C]}$',
           ylabel=r'$\mathregular{T_{sim}-T_{air}\/[^\circ C]}$')

    fig.tight_layout()
    fig.savefig(figure_dir / f'all_sim_vs_obs_delta_temperature.png')
    pyplot.close('all')

    pass


def plot_irradiance(shoot_obj: dict, obs_df: DataFrame):
    fig, axs = pyplot.subplots(nrows=2, ncols=2)

    obs_faparc, sim_faparc = zip(*[(obs_df.loc[idx, 'fAPARc'], base_functions.calc_sim_fapar(shoot_obj[idx]))
                                   for idx in obs_df.index])

    axs[0, 0].scatter(obs_faparc, sim_faparc, marker='.', alpha=0.2, label='fAPARc')
    axs[0, 0].plot([0, 1], [0, 1], linestyle='--', color='grey', label='1:1')
    axs[0, 0].set(xlabel='obs', ylabel='sim')

    for ax in (axs[1, 0], axs[0, 1]):
        ax.scatter(obs_df['gai'], obs_df['fAPARc'], marker='.', alpha=0.2, color='r', label='obs')

    for ax in (axs[1, 0], axs[1, 1]):
        ax.scatter([sum(shoot.inputs.leaf_layers.values()) for shoot in shoot_obj.values()], sim_faparc,
                   marker='.', alpha=0.2, color='b', label='sim')

    for ax in (axs[1, 0], axs[0, 1], axs[1, 1]):
        ax.set(ylabel='fAPARc', xlabel='GAI')

    for ax in axs.flatten():
        ax.legend()

    fig.savefig(PathInfos.source_figs.value / 'irradiance.png')
    pyplot.close(fig)
    pass


def plot_errors(res: dict, figure_dir: Path):
    n_rows = 3
    n_cols = 4

    for k in ('temperature_canopy', 'temperature_soil', 'net_radiation', 'sensible_heat_flux', 'latent_heat_flux',
              'soil_heat_flux', 'temperature_sunlit', 'temperature_shaded'):

        fig, axs = pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8), sharey='all')

        idx, sim, obs = zip(*[(i, i_sim, i_obs) for i, (i_sim, i_obs) in enumerate(zip(res[k]['sim'], res[k]['obs']))
                              if not any(isna([i_sim, i_obs]))])
        error = [i_sim - i_obs for i_sim, i_obs in zip(sim, obs)]

        for i_explanatory, explanatory in enumerate(
                ('wind_speed', 'vapor_pressure_deficit', 'temperature_air', 'soil_water_potential',
                 'aerodynamic_resistance', 'absorbed_par_soil', 'absorbed_par_veg', 'hour',
                 'net_longwave_radiation', 'height', 'gai')):
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

        title = ' '.join(config.UNITS_MAP[k])
        axs[1, 0].set_ylabel(' '.join((r'$\mathregular{\epsilon}$', title)), fontsize=16)
        fig.tight_layout()
        fig.savefig(figure_dir / f'errors_{k}.png')
        pyplot.close()

    pass
