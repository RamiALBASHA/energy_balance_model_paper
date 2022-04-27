from pathlib import Path

from matplotlib import pyplot

from sim_vs_obs.braunschweig_face.config import ExpInfos
from sim_vs_obs.common import calc_apparent_temperature
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
