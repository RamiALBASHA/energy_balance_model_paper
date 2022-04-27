from copy import deepcopy
from pathlib import Path

import statsmodels.api as sm
from matplotlib import pyplot, ticker, colors
from numpy import array, linspace
from pandas import isna

from sim_vs_obs.common import get_canopy_abs_irradiance_from_solver
from utils import stats, config

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
    irradiance_dict = deepcopy(obs_dict)
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
                    inc_par = sum(solver.crop.inputs.incident_irradiance.values())
                    irradiance_dict['t'][treatment].append(inc_par)
                    irradiance_dict['delta_t'][treatment].append(inc_par)

    for ax_row, var_to_plot in zip(axs, vars_to_plot):
        for ax, treatment in zip(ax_row, treatments):
            temperature_obs = obs_dict[var_to_plot][treatment]
            temperature_sim = sim_dict[var_to_plot][treatment]
            c = irradiance_dict[var_to_plot][treatment]
            norm = colors.Normalize(0, vmax=500)
            im = ax.scatter(temperature_obs, temperature_sim, marker='.', alpha=0.5,
                            edgecolor="none", c=c, cmap='hot', norm=norm)
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

    fig.subplots_adjust(bottom=0.25)
    cbar_ax = fig.add_axes([0.37, 0.1, 0.30, 0.04])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_ylabel(' '.join(config.UNITS_MAP['incident_par']), va="top", ha='right', rotation=0)

    fig.savefig(path_figs_dir / f"sim_vs_obs_{'all' if relative_layer_index is None else relative_layer_index}.png")
    pass


def plot_errors(data: dict, path_figs_dir: Path):
    common = dict(
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
        absorbed_par_soil=[],
        absorbed_par_veg=[],
        psi_u=[],
        psi_h=[],
        hour=[],
        net_longwave_radiation=[],
        height=[],
        gai=[],
    )

    treatments = ('extensive', 'intensive')

    res = {k: deepcopy(common) for k in treatments}

    for datetime_obs in data.keys():
        for trt in treatments:
            solver = data[datetime_obs][trt]['solver']
            obs = data[datetime_obs][trt]['obs']
            canopy_layers = [k for k in solver.crop.components_keys if k != -1]

            for i, layer in enumerate(canopy_layers):
                res[trt]['temperature_obs'].append(obs[obs['leaf_level'] == layer]['temperature'].mean())
                res[trt]['temperature_air'].append(solver.crop.inputs.air_temperature - 273.15)
                res[trt]['temperature_sim'].append(solver.crop[layer].temperature - 273.15)
                res[trt]['incident_diffuse_par_irradiance'].append(solver.crop.inputs.incident_irradiance['diffuse'])
                res[trt]['incident_direct_par_irradiance'].append(solver.crop.inputs.incident_irradiance['direct'])
                res[trt]['wind_speed'].append(solver.crop.inputs.wind_speed / 3600.)
                res[trt]['vapor_pressure_deficit'].append(solver.crop.inputs.vapor_pressure_deficit)
                res[trt]['soil_water_potential'].append(solver.crop.inputs.soil_water_potential)
                res[trt]['richardson'].append(solver.crop.state_variables.richardson_number)
                res[trt]['monin_obukhov'].append(solver.crop.state_variables.monin_obukhov_length)
                res[trt]['aerodynamic_resistance'].append(solver.crop.state_variables.aerodynamic_resistance * 3600.)
                res[trt]['absorbed_par_soil'].append(solver.crop.inputs.absorbed_irradiance[-1]['lumped'])
                res[trt]['absorbed_par_veg'].append(get_canopy_abs_irradiance_from_solver(solver=solver))
                res[trt]['psi_u'].append(solver.crop.state_variables.stability_correction_for_momentum)
                res[trt]['psi_h'].append(solver.crop.state_variables.stability_correction_for_heat)
                res[trt]['hour'].append(datetime_obs.hour)
                res[trt]['net_longwave_radiation'].append(solver.crop.state_variables.net_longwave_radiation)
                res[trt]['height'].append(solver.crop.inputs.canopy_height)
                res[trt]['gai'].append(sum(solver.crop.inputs.leaf_layers.values()))

            res[trt].update({'temperature_error': [t_sim - t_obs for t_sim, t_obs in zip(res[trt]['temperature_sim'],
                                                                                         res[trt]['temperature_obs'])]})

    norm = colors.Normalize(0, vmax=500)
    n_rows = 3
    n_cols = 4

    for trt in treatments:
        par_inc = [par_dir + par_diff for par_dir, par_diff in
                   zip(res[trt]['incident_direct_par_irradiance'], res[trt]['incident_diffuse_par_irradiance'])]
        fig, axs = pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8), sharey='all')
        explanatory_vars = ('wind_speed', 'vapor_pressure_deficit', 'temperature_air', 'soil_water_potential',
                            'aerodynamic_resistance', 'absorbed_par_soil', 'absorbed_par_veg', 'hour',
                            'net_longwave_radiation', 'height', 'gai')
        for i, explanatory in enumerate(explanatory_vars):
            ax = axs[i % n_rows, i // n_rows]

            explanatory_ls, error_ls, c = zip(
                *[(ex, er, c_i) for ex, er, c_i in zip(res[trt][explanatory], res[trt]['temperature_error'], par_inc)
                  if not any(isna([ex, er]))])
            im = ax.scatter(explanatory_ls, error_ls, marker='.', alpha=0.5, edgecolor='none', c=c, cmap='hot',
                            norm=norm)

            ax.set(xlabel=' '.join(config.UNITS_MAP[explanatory]))

            x = array(explanatory_ls)
            x = sm.add_constant(x)
            y = array(error_ls)
            results = sm.OLS(y, x).fit()

            ax.plot(*zip(*[(i, results.params[0] + results.params[1] * i) for i in
                           linspace(min(explanatory_ls), max(explanatory_ls), 2)]), 'k--')
            p_value_slope = results.pvalues[1] / 2.
            ax.text(0.1, 0.9, '*' if p_value_slope < 0.05 else '', transform=ax.transAxes, fontweight='bold')

            if i == len(explanatory_vars) - 1:
                colorbar_ax = axs.flatten()[-1]
                fig.colorbar(im, ax=colorbar_ax, orientation='horizontal',
                             label=' '.join(config.UNITS_MAP['incident_par']))

        axs[1, 0].set_ylabel(r'$\mathregular{T_{sim}-T_{obs}\/[^\circ C]}$', fontsize=16)
        fig.tight_layout()
        fig.savefig(path_figs_dir / f'errors_{trt}.png')
        pyplot.close()

    pass
