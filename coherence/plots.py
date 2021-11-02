from pathlib import Path
from string import ascii_lowercase

import pandas as pd
from crop_energy_balance import crop as eb_canopy
from crop_irradiance.uniform_crops import shoot as irradiance_canopy
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
    'monin_obukhov_length': ('L', '[m]'),
    'available_energy': ('A', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'penman_monteith_evaporative_energy': (r'$\mathregular{\lambda E}$', r'$\mathregular{[W\/m^{-2}_{ground}]}$'),
    'boundary_resistance': (r'$\mathregular{r_a}$', r'$\mathregular{[h\/m^{-1}]}$'),
    'source_temperature': (r'$\mathregular{T_m}$', r'$\mathregular{[^\circ C]}$'),
    'PAR_direct': r'$\mathregular{R_{inc,\/PAR,\/direct}}$',
    'PAR_diffuse': r'$\mathregular{R_{inc,\/PAR,\/diffuse}}$',
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
    fig.savefig(str(figure_path / 'coherence_irradiance.png'))
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
    fig.savefig(str(figure_path / 'coherence_temperature.png'))
    plt.close()


def plot_leaf_profile(vegetative_layers: {int, dict}, figure_path: Path):
    fig, axs = plt.subplots(ncols=4, figsize=(15, 5))
    for i, (k, v) in enumerate(vegetative_layers.items()):
        layer_indices = list(v.keys())
        axs[i].plot(list(v.values()), layer_indices, 'o-')
        axs[i].set(title=handle_sim_name(k),
                   xlabel=r'$\mathregular{m^2_{leaf} \cdot m^{-2}_{ground}}$', ylabel='layer index [-]',
                   yticks=layer_indices)
    fig.tight_layout()
    fig.savefig(str(figure_path / 'coherence_layers.png'))
    plt.close()


def plot_irradiance_dynamics(ax: plt.axis,
                             incident_par_irradiance: pd.Series,
                             simulation_case: str,
                             all_cases_data: dict):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    _, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    ax.set_title(handle_sim_name(simulation_case))
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

    ax.set_title(handle_sim_name(simulation_case))

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
                                         all_cases_absorbed_irradiance: (dict, irradiance_canopy),
                                         all_cases_temperature: dict,
                                         figure_path: Path):
    assert all_cases_temperature.keys() == all_cases_absorbed_irradiance[0].keys()

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
        ax.text(0.075, 0.95, f'({ascii_lowercase[i]})', transform=ax.transAxes)

    for ax in axes[:, 0]:
        ax.set_ylabel('Component index [-]')

    fig.tight_layout()
    fig.savefig(str(figure_path / 'coherence_temperature_at_one_hour.png'))
    plt.close()


def plot_temperature_one_hour_comparison2(hour: int,
                                          hourly_weather: pd.DataFrame,
                                          all_cases_absorbed_irradiance: (dict, irradiance_canopy),
                                          all_cases_temperature: dict,
                                          figure_path: Path):
    assert all_cases_temperature.keys() == all_cases_absorbed_irradiance[0].keys()

    cases = all_cases_temperature.keys()
    component_indices = all_cases_temperature['layered_lumped'][hour].keys()

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='row', sharey='all', figsize=(8, 7))
    for case in cases:
        i = 0 if 'bigleaf' in case else 1
        is_lumped = 'lumped' in case
        plot_temperature_at_one_hour(
            ax=axes[0, i],
            hour=hour,
            temperature_air=hourly_weather.loc[:, 'air_temperature'],
            simulation_case=case,
            all_cases_data=all_cases_temperature,
            plot_air_temperature=is_lumped,
            plot_soil_temperature=is_lumped)
        plot_irradiance_at_one_hour(
            ax=axes[1, i],
            hour=hour,
            incident_direct=hourly_weather.loc[:, 'incident_direct_irradiance'],
            incident_diffuse=hourly_weather.loc[:, 'incident_diffuse_irradiance'],
            simulation_case=case,
            all_cases_data=all_cases_absorbed_irradiance,
            plot_incident=is_lumped,
            plot_soil=is_lumped,
            set_title=False)

    for i, ax in enumerate(axes.flatten()):
        ax.text(0.025, 0.025, f'({ascii_lowercase[i]})', transform=ax.transAxes)
    for ax in axes[:, 0]:
        ax.set_ylabel('Component index [-]')
        ax.legend()
    for ax in axes[0, :]:
        ax.set_title(ax.get_title().split(' ')[0])

    axes[0, 0].set_yticks(list(component_indices))

    fig.tight_layout()
    fig.savefig(str(figure_path / f'coherence_temperature_at_{hour}h.png'))
    plt.close()


def plot_temperature_at_one_hour(ax: plt.axis,
                                 hour: int,
                                 temperature_air: pd.Series,
                                 simulation_case: str,
                                 all_cases_data: dict,
                                 plot_air_temperature: bool = True,
                                 plot_soil_temperature: bool = True,
                                 set_title: bool = True):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    canopy_class, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    style = 'o-' if max(component_indexes) > 0 else 'o'
    y = None

    if leaf_class == 'lumped':
        y, x = zip(*[(i, summary_data[i][hour]) for i in component_indexes if i != -1])
        ax.plot([(v - 273.15 if v > 273.15 else None) for v in x], y, style, label='lumped')
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])

        ax.plot([(v - 273.15 if v > 273.15 else None) for v in x_sun], y, style, color='y', label='sunlit')
        ax.plot([(v - 273.15 if v > 273.15 else None) for v in x_sh], y, style, color='darkgreen', label='shaded')

    if plot_soil_temperature:
        ax.plot(summary_data[-1][hour] - 273.15, -1, 's', color='brown', label='soil')

    ax.set_xlabel(r'$\mathregular{[^\circ C]}$')

    if plot_air_temperature:
        y_text = max(y)
        ax.scatter(temperature_air[hour], y_text + 1, alpha=0)
        ax.annotate(r'$\mathregular{T_{a}}$', xy=(temperature_air[hour], y_text + 0.25),
                    xytext=(temperature_air[hour], y_text + 0.75),
                    arrowprops=dict(arrowstyle="->"), ha='center')

    if set_title:
        ax.set_title(handle_sim_name(canopy_class))
    return


def plot_irradiance_at_one_hour(ax: plt.axis,
                                hour: int,
                                incident_direct: pd.Series,
                                incident_diffuse: pd.Series,
                                simulation_case: str,
                                all_cases_data: dict,
                                plot_incident: bool = True,
                                plot_soil: bool = False,
                                set_title: bool = True):
    summary_data = get_summary_data(simulation_case, all_cases_data[0])
    canopy_class, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    if set_title:
        ax.set_title(handle_sim_name(simulation_case))

    if leaf_class == 'lumped':
        y, x = zip(*[(i, summary_data[i][hour]) for i in component_indexes if i != -1])
        ax.plot(x, y, 'o-', label='lumped')
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])

        ax.plot(x_sun, y, 'o-', color='y', label='sunlit')
        ax.plot(x_sh, y, 'o-', color='darkgreen', label='shaded')

        if canopy_class == 'bigleaf':
            ax.text(0.35 * (incident_direct[hour] + incident_diffuse[hour]), 1.5,
                    (r'$\mathregular{\phi_{shaded}}$' +
                     f'={round(all_cases_data[1][simulation_case][hour][0].shaded_fraction, 2)}'))
        else:
            for layer in all_cases_data[1][simulation_case][hour].keys():
                ax.text(0.35 * (incident_direct[hour] + incident_diffuse[hour]), layer, (
                        r'$\mathregular{\phi_{shaded}}$' + f'={round(all_cases_data[1][simulation_case][hour][layer].shaded_fraction, 2)}'))

    if plot_soil:
        ax.plot(summary_data[-1][hour], -1, 's', color='brown', label=f'{leaf_class} soil')

    if plot_incident:
        y_text = max(y)
        for s, (v, ha) in ({'PAR_direct': (incident_direct[hour], 'right'),
                            'PAR_diffuse': (incident_diffuse[hour], 'left')}).items():
            ax.scatter(v, y_text + 1, alpha=0)
            ax.annotate('', xy=(v, y_text + 0.25), xytext=(v, y_text + 0.75), arrowprops=dict(arrowstyle="->"))
            ax.text(v, y_text + 0.75, UNITS_MAP[s], ha=ha)

    ax.set_xlabel(r'$\mathregular{[W_{PAR} \cdot m^{-2}_{ground}]}$')
    return


def plot_canopy_variable(
        all_cases_solver: eb_canopy,
        variable_to_plot: str,
        figure_path: Path,
        axes: plt.axis = None,
        return_axes: bool = False):
    hours = range(24)
    cases = all_cases_solver.keys()
    conv = - 273.15 if 'temperature' in variable_to_plot else 0
    if axes is None:
        _, axes = plt.subplots(ncols=len(cases), sharex='all', sharey='all', figsize=(15, 5))

    for i, case in enumerate(cases):
        ax = axes[i]
        y_ls = []
        for h in hours:
            y = getattr(all_cases_solver[case][h].crop.state_variables, variable_to_plot)
            y_ls.append(y + conv if y is not None else y)
        ax.plot(hours, y_ls)
    if return_axes:
        return axes
    else:
        [ax.set_xlabel('hours') for ax in axes[:]]
        axes[0].set_ylabel(f'{UNITS_MAP[variable_to_plot][0]} {UNITS_MAP[variable_to_plot][1]}')
        plt.suptitle(variable_to_plot)
        plt.savefig(str(figure_path / f'coherence_{variable_to_plot}.png'))
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
        fig.savefig(str(figure_path / f'coherence_{variable_to_plot}.png'))
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
            ax.set_title(handle_sim_name(model))
        ax.grid()
    axes[0].legend()
    axes[0].set_ylabel(r'$\mathregular{Energy\/[W\/m^{-2}_{ground}]}$')
    fig.savefig(str(figure_path / 'coherence_energy_balance.png'))
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
    plt.savefig(str(figure_path / 'coherence_universal_functions.png'))
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


def handle_sim_name(sim_name: str) -> str:
    return sim_name.replace('_', ' ').lower().replace(
        'bigleaf', 'BigLeaf').replace(
        'sunlit-shaded', 'Sunlit-Shaded').replace(
        'lumped', 'Lumped').replace(
        'layered', 'Layered')


def plot_properties_profile(solver_data: dict, hours: list, component_props: [str], figure_path: Path):
    layers_idx = [k for k in solver_data[hours[0]].crop.keys() if k != -1]
    n_rows = len(hours)
    fig, axs = plt.subplots(nrows=n_rows, ncols=len(component_props), sharey='all')
    for i, hour in enumerate(hours):
        crop = solver_data[hour].crop
        for j, prop in enumerate(component_props):
            ax = axs[i, j]
            if '-' in prop:
                prop1, prop2 = prop.split('-')
                prop_ls1 = [getattr(crop[layer]['sunlit'], prop1) for layer in layers_idx]
                prop_ls2 = [getattr(crop[layer]['sunlit'], prop2) for layer in layers_idx]
                y = [(y1 - y2) for y1, y2 in zip(prop_ls1, prop_ls2)]
                xlabel = f'{UNITS_MAP[prop1][0]} - {UNITS_MAP[prop2][0]} {UNITS_MAP[prop2][1]}'
            else:
                y = [getattr(crop[layer]['sunlit'], prop) for layer in layers_idx]
                xlabel = f'{UNITS_MAP[prop][0]} {UNITS_MAP[prop][1]}'
            ax.plot(y, layers_idx)
            if j == 0:
                ax.set_ylabel('Component index [-]')
                ax.text(0.7, 0.1, f'(hour: {hour})', transform=ax.transAxes)
            if i == n_rows - 1:
                ax.set_xlabel(xlabel)

    fig.tight_layout()
    plt.savefig(str(figure_path / 'coherence_sunlit_props.png'))
    plt.close('all')
    return fig


def compare_energy_balance_terms(s1, s2):
    fig, ax = plt.subplots()
    for s in (s1, s2):
        y, x_ls = zip(*list(s['temperature']['layered_lumped'][12].items()))
        ax.plot([x['lumped'] for x in x_ls], y, label='0.1 m')
