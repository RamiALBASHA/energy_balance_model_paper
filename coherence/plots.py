from pathlib import Path

import pandas as pd
from crop_energy_balance import crop as eb_canopy
from matplotlib import pyplot as plt

UNITS_MAP = {
    'net_radiation': (r'$\mathregular{R_n}$', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'sensible_heat_flux': ('H', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'total_penman_monteith_evaporative_energy': (r'$\mathregular{\lambda E}$', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'soil_heat_flux': ('G', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'energy_balance': ('balance', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'richardson_number': ('Ri', '[-]'),
    'stability_correction_for_momentum': (r'$\mathregular{\Phi_m}$', '[m]'),
    'stability_correction_for_heat': (r'$\mathregular{\Phi_h}$', '[m]'),
    'monin_obukhov_length': ('L', '[m]')
}


def plot_irradiance_dynamic_comparison(incident_irradiance: pd.Series,
                                       all_cases_absorbed_irradiance: dict,
                                       figure_path: Path):
    cases = all_cases_absorbed_irradiance.keys()

    fig, axes = plt.subplots(ncols=len(cases), sharex='all', sharey='all', figsize=(15, 5))
    for i, case in enumerate(cases):
        plot_irradiance_dynamics(
            ax=axes[i],
            incident_par_irradiance=incident_irradiance,
            simulation_case=case,
            all_cases_data=all_cases_absorbed_irradiance)

    for i, ax in enumerate(axes):
        ax.grid()
        ax.legend()
        ax.set(xlabel='hour')
        if i == 0:
            ax.set_ylabel(r'$\mathregular{W_{PAR} \cdot m^{-2}_{ground}}$')

    fig.tight_layout()
    fig.savefig(figure_path / 'coherence_irradiance.png')
    plt.close()


def plot_temperature_dynamic_comparison(temperature_air: pd.Series,
                                        all_cases_temperature: dict,
                                        figure_path: Path):
    cases = all_cases_temperature.keys()

    fig, axes = plt.subplots(ncols=len(cases), sharex='all', sharey='all', figsize=(15, 5))
    for i, case in enumerate(cases):
        plot_temperature_dynamics(
            ax=axes[i],
            temperature_air=temperature_air,
            simulation_case=case,
            all_cases_data=all_cases_temperature)

    for i, ax in enumerate(axes):
        ax.legend()
        ax.grid()
        ax.set(xlabel='hour', ylim=(-10, 50)),
        if i == 0:
            ax.set_ylabel(r'$\mathregular{temperature\/[^\circ C]}$')

    fig.tight_layout()
    fig.savefig(figure_path / 'coherence_temperature.png')
    plt.close()


def plot_leaf_profile(vegetative_layers: {int, dict}, figure_path: Path):
    fig, axs = plt.subplots(ncols=4, figsize=(15, 5))
    for i, (k, v) in enumerate(vegetative_layers.items()):
        layer_indices = list(v.keys())
        axs[i].plot(list(v.values()), layer_indices, 'o-')
        axs[i].set(title=k, xlabel=r'$\mathregular{m^2_{leaf} \cdot m^{-2}_{ground}}$', ylabel='layer index [-]',
                   yticks=layer_indices)
    fig.tight_layout()
    fig.savefig(figure_path / 'coherence_layers.png')
    plt.close()


def plot_irradiance_dynamics(ax: plt.axis,
                             incident_par_irradiance: pd.Series,
                             simulation_case: str,
                             all_cases_data: dict):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    _, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    ax.set_title(simulation_case.replace('_', ' '))
    ax.plot(range(24), incident_par_irradiance, label='incident', color='k', linestyle='--', linewidth=2)

    for component_index in component_indexes:
        abs_irradiance = summary_data[component_index]
        if component_index == -1:
            ax.plot(range(24), abs_irradiance, label=f'soil', color='k', linewidth=2)
        else:
            if leaf_class == 'lumped':
                ax.plot(range(24), summary_data[component_index], label=f'abs {component_index}')
            else:
                ax.plot(range(24), summary_data[component_index]['sunlit'], label=f'abs sunlit {component_index}')
                ax.plot(range(24), summary_data[component_index]['shaded'], label=f'abs shaded {component_index}')
    pass


def plot_temperature_dynamics(ax: plt.axis,
                              temperature_air: pd.Series,
                              simulation_case: str,
                              all_cases_data: dict):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    _, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    ax.set_title(simulation_case.replace('_', ' '))

    hours = range(24)
    ax.plot(hours, temperature_air, label='air', color='k', linestyle='--', linewidth=2)

    for component_index in component_indexes:
        component_temperature = summary_data[component_index]
        if component_index == -1:
            ax.plot(hours, [(v - 273.15 if v > 273.15 else None) for v in component_temperature],
                    label=f'soil', color='k', linewidth=2)
        else:
            if leaf_class == 'lumped':
                ax.plot(hours, [(v - 273.15 if v > 273.15 else None) for v in summary_data[component_index]],
                        label=f'{component_index}')
            else:
                ax.plot(hours,
                        [(v - 273.15 if v > 273.15 else None) for v in summary_data[component_index]['sunlit']],
                        label=f'sunlit {component_index}')
                ax.plot(hours,
                        [(v - 273.15 if v > 273.15 else None) for v in summary_data[component_index]['shaded']],
                        label=f'shaded {component_index}')

    pass


def plot_temperature_one_hour_comparison(hour: int,
                                         hourly_weather: pd.DataFrame,
                                         all_cases_absorbed_irradiance: dict,
                                         all_cases_temperature: dict,
                                         figure_path: Path):
    assert all_cases_temperature.keys() == all_cases_absorbed_irradiance.keys()

    cases = all_cases_temperature.keys()

    fig, axes = plt.subplots(nrows=2, ncols=len(cases), sharex='row', sharey='all', figsize=(15, 10))
    for i, case in enumerate(cases):
        plot_temperature_at_one_hour(
            ax=axes[1, i],
            hour=hour,
            temperature_air=hourly_weather.loc[:, 'air_temperature'],
            simulation_case=case,
            all_cases_data=all_cases_temperature)
        plot_irradiance_at_one_hour(
            ax=axes[0, i],
            hour=hour,
            incident_direct=hourly_weather.loc[:, 'incident_direct_irradiance'],
            incident_diffuse=hourly_weather.loc[:, 'incident_diffuse_irradiance'],
            simulation_case=case,
            all_cases_data=all_cases_absorbed_irradiance)

    for i, ax in enumerate(axes.flatten()):
        ax.legend()

    for ax in axes[:, 0]:
        ax.set_ylabel('Component index [-]')

    fig.tight_layout()
    fig.savefig(figure_path / 'coherence_temperature_at_one_hour.png')
    plt.close()


def plot_temperature_at_one_hour(ax: plt.axis,
                                 hour: int,
                                 temperature_air: pd.Series,
                                 simulation_case: str,
                                 all_cases_data: dict):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    _, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    ax.axvline(temperature_air[hour], label='air', color='k', linestyle='--', linewidth=2)

    if leaf_class == 'lumped':
        y, x = zip(*[(i, summary_data[i][hour]) for i in component_indexes if i != -1])
        ax.plot([(v - 273.15 if v > 273.15 else None) for v in x], y, 'o-', label='lumped')
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])

        ax.plot([(v - 273.15 if v > 273.15 else None) for v in x_sun], y, 'o-', color='y', label='sunlit')
        ax.plot([(v - 273.15 if v > 273.15 else None) for v in x_sh], y, 'o-', color='brown', label='shaded')

    ax.set_xlabel(r'$\mathregular{[^\circ C]}$')
    return


def plot_irradiance_at_one_hour(ax: plt.axis,
                                hour: int,
                                incident_direct: pd.Series,
                                incident_diffuse: pd.Series,
                                simulation_case: str,
                                all_cases_data: dict):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    _, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    ax.set_title(simulation_case.replace('_', ' '))
    ax.axvline(incident_direct[hour], label='incident direct', color='y', linestyle='--', linewidth=2)
    ax.axvline(incident_diffuse[hour], label='incident diffuse', color='red', linestyle='--', linewidth=2)

    if leaf_class == 'lumped':
        y, x = zip(*[(i, summary_data[i][hour]) for i in component_indexes if i != -1])
        ax.plot(x, y, 'o-', label='lumped')
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])

        ax.plot(x_sun, y, 'o-', color='y', label='sunlit')
        ax.plot(x_sh, y, 'o-', color='brown', label='shaded')

    ax.set_xlabel(r'$\mathregular{[W \cdot m^{-2}_{ground}]}$')
    return


def plot_canopy_variable(
        all_cases_solver: eb_canopy,
        variable_to_plot: str,
        figure_path: Path,
        axes: plt.axis = None,
        return_axes: bool = False):
    hours = range(24)
    cases = all_cases_solver.keys()

    if axes is None:
        _, axes = plt.subplots(ncols=len(cases), sharex='all', sharey='all', figsize=(15, 5))

    for i, case in enumerate(cases):
        ax = axes[i]
        ax.plot(hours, [getattr(all_cases_solver[case][h].crop.state_variables, variable_to_plot) for h in hours])
    if return_axes:
        return axes
    else:
        [ax.set_xlabel('hours') for ax in axes[:]]
        plt.suptitle(variable_to_plot)
        plt.savefig(figure_path / f'coherence_{variable_to_plot}.png')
        plt.close()


def plot_energy_balance_components(
        h_solver: list,
        variable_to_plot: str,
        ax: plt.Subplot,
        figure_path: Path,
        return_ax: bool = False):
    hours = range(24)

    if variable_to_plot == 'soil_heat_flux':
        ax.plot(hours, [getattr(h_solver[h].crop[-1], 'heat_flux') for h in hours],
                label=UNITS_MAP[variable_to_plot][0])
    elif variable_to_plot == 'energy_balance':
        ax.plot(hours, [h_solver[h].energy_balance for h in hours], 'k--', label=UNITS_MAP[variable_to_plot][0])
    else:
        ax.plot(hours, [getattr(h_solver[h].crop.state_variables, variable_to_plot) for h in hours],
                label=UNITS_MAP[variable_to_plot][0])

    if return_ax:
        return ax
    else:
        ax.set_xlabel('hours')
        fig = ax.get_figure()
        fig.suptitle(variable_to_plot)
        fig.savefig(figure_path / f'coherence_{variable_to_plot}.png')
        plt.close()


def plot_energy_balance(solvers: dict, figure_path: Path):
    eb_components = [
        'net_radiation', 'sensible_heat_flux', 'total_penman_monteith_evaporative_energy', 'soil_heat_flux',
        'energy_balance']
    models = solvers.keys()
    fig, axes = plt.subplots(ncols=len(models), sharex='all', sharey='all', figsize=(15, 5))
    for model, ax in zip(models, axes):
        for eb_component in eb_components:
            ax = plot_energy_balance_components(h_solver=solvers[model], variable_to_plot=eb_component, ax=ax,
                                                figure_path=figure_path, return_ax=True)
        ax.grid()
        ax.legend()
    axes[0].set_ylabel(r'$\mathregular{Energy\/[W\/m^{-2}_{ground}]}$')
    fig.savefig(figure_path / 'coherence_energy_balance.png')
    pass


def plot_stability_terms(solvers: dict, figs_path: Path):
    # hourly plots
    terms = ['sensible_heat_flux', 'monin_obukhov_length', 'stability_correction_for_momentum',
             'stability_correction_for_heat']
    models = solvers.keys()
    fig, axes = plt.subplots(ncols=len(models), nrows=len(terms), sharex='all', sharey='row', figsize=(15, 5))

    for i, term in enumerate(terms):
        plot_canopy_variable(all_cases_solver=solvers, variable_to_plot=term, axes=axes[i, :], figure_path=figs_path,
                             return_axes=True)
        axes[i, 0].set_ylabel(' '.join(UNITS_MAP[term]))

    [ax.set_xlabel('hour') for ax in axes[-1, :]]
    fig.savefig(figs_path / 'coherence_stability_terms.png')
    plt.close('all')


def plot_universal_functions(solvers, figure_path: Path, measurement_height: float = 2):
    # universal functions plots
    x = []
    phi_h = []
    phi_m = []
    for m_solvers in solvers.values():
        for h_solver in m_solvers:
            state = h_solver.crop.state_variables
            x.append((measurement_height - state.zero_displacement_height) / state.monin_obukhov_length)
            phi_h.append(state.stability_correction_for_heat)
            phi_m.append(state.stability_correction_for_momentum)
    _, axes = plt.subplots(ncols=2, sharex='all', sharey='all')
    axes[0].scatter(x, phi_h)
    axes[0].set_ylabel(' '.join(UNITS_MAP['stability_correction_for_heat']))
    axes[1].scatter(x, phi_m)
    axes[1].set_ylabel(' '.join(UNITS_MAP['stability_correction_for_momentum']))
    for ax in axes:
        ax.set_xlabel(r'$\mathregular{\frac{z_m-d}{L}\/[m]}$')
        ax.grid()
    plt.savefig(figure_path / 'coherence_universal_functions.png')
    plt.close('all')


def get_summary_data(simulation_case: str,
                     all_cases_data: dict) -> dict:
    data_dynamic = all_cases_data[simulation_case]
    _, leaf_class = simulation_case.split('_')
    component_indexes = data_dynamic[0].keys()

    if leaf_class == 'lumped':
        summary_data = {k: [] for k in component_indexes}

        for hour in range(24):
            for component_index in component_indexes:
                summary_data[component_index].append(data_dynamic[hour][component_index]['lumped'])
    else:
        summary_data = {k: {'sunlit': [], 'shaded': []} for k in component_indexes if k != -1}
        summary_data.update({-1: []})

        for hour in range(24):
            for component_index in component_indexes:
                if component_index != -1:
                    summary_data[component_index]['sunlit'].append(
                        data_dynamic[hour][component_index]['sunlit'])
                    summary_data[component_index]['shaded'].append(
                        data_dynamic[hour][component_index]['shaded'])
                else:
                    summary_data[component_index].append(data_dynamic[hour][component_index]['lumped'])
    return summary_data
