from math import degrees, prod
from pathlib import Path
from string import ascii_lowercase

import pandas as pd
from crop_energy_balance import crop as eb_canopy
from crop_irradiance.uniform_crops import shoot as irradiance_canopy
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from numpy import median, mean

from utils.config import UNITS_MAP


def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total


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
            ax.set_ylabel(r'$\mathregular{W_{PAR} \/ m^{-2}_{ground}}$')

    fig.tight_layout()
    fig.savefig(str(figure_path / 'irradiance.png'))
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
    fig.savefig(str(figure_path / 'temperature.png'))
    plt.close()


def plot_dynamic_comparison(solvers: dict,
                            figure_path: Path,
                            variable_to_plot: str,
                            **kwargs):
    fig, ax = plt.subplots()
    plot_canopy_variable(all_cases_solver=solvers, variable_to_plot=variable_to_plot, y_cumsum=False,
                         axes=[ax, ax, ax, ax], figure_path=figure_path, return_axes=True)
    var_name = 'temperature_source' if variable_to_plot == 'source_temperature' else variable_to_plot
    ax.set(xlabel='hour', ylabel=' '.join(UNITS_MAP[var_name]), **kwargs)
    ax.legend()

    fig.tight_layout()
    fig.savefig(figure_path / f'{variable_to_plot}_comparison.png')
    plt.close('all')


def plot_leaf_profile(vegetative_layers: {int, dict}, figure_path: Path):
    fig, axs = plt.subplots(ncols=4, figsize=(15, 5))
    for i, (k, v) in enumerate(vegetative_layers.items()):
        layer_indices = list(v.keys())
        axs[i].plot(list(v.values()), layer_indices, 'o-')
        axs[i].set(title=handle_sim_name(k),
                   xlabel=r'$\mathregular{m^2_{leaf} \/ m^{-2}_{ground}}$', ylabel='layer index [-]',
                   yticks=layer_indices)
    fig.tight_layout()
    fig.savefig(str(figure_path / 'layers.png'))
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
                              all_cases_data: dict,
                              is_set_title: bool = True,
                              is_filled: bool = False):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    canopy_class, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    if is_set_title:
        ax.set_title(handle_sim_name(simulation_case))

    c_map = {'lumped': 'blue', 'sunlit': 'orange', 'shaded': 'DarkGreen'}
    hours = range(24)
    ax.plot(hours, temperature_air, label=r"T$_{\rm a}$", color='k', linestyle='--')
    ax.plot(hours, [v - 273.15 for v in summary_data[-1]], label=r"T$_{\rm soil}$", color='k')

    component_indexes = [cidx for cidx in component_indexes if cidx != -1]
    if canopy_class == 'bigleaf':
        component_temperature = summary_data[component_indexes[0]]
        if leaf_class == 'lumped':
            ax.plot(hours, [v - 273.15 for v in component_temperature],
                    label=r"T$\rm _{{{}}}$".format("s, lumped"), c=c_map['lumped'])
        else:
            ax.plot(hours, [v - 273.15 for v in component_temperature['sunlit']],
                    label=r"T$\rm _{{{}}}$".format("s, sunlit"), c=c_map['sunlit'])
            ax.plot(hours, [v - 273.15 for v in component_temperature['shaded']],
                    label=r"T$\rm _{{{}}}$".format("s, shaded"), c=c_map['shaded'])
    else:
        if not is_filled:
            for component_index in component_indexes:
                component_temperature = summary_data[component_index]
                if leaf_class == 'lumped':
                    ax.plot(hours, [v - 273.15 for v in component_temperature],
                            label=r"T$\rm _{{{}}}$".format(f's, {component_index}'), c=c_map['lumped'])
                else:
                    ax.plot(hours,
                            [v - 273.15 for v in component_temperature['sunlit']],
                            label=r"T$\rm _{{{}}}$".format(f"s, {component_index}, sunlit"), c=c_map['sunlit'])
                    ax.plot(hours,
                            [v - 273.15 for v in component_temperature['shaded']],
                            label=r"T$\rm _{{{}}}$".format(f"s, {component_index}, shaded"), c=c_map['shaded'])
        else:
            if leaf_class == 'lumped':
                t_max, t_min = zip(*[(max(ls) - 273.15, min(ls) - 273.15) for ls in
                                     zip(*[summary_data[i] for i in component_indexes])])
                ax.fill_between(hours, t_min, t_max, color=c_map['lumped'], label=r"T$\rm _{s,\/lumped}$")
            else:
                for s in ('sunlit', 'shaded'):
                    t_max, t_min = zip(*[(max(ls) - 273.15, min(ls) - 273.15) for ls in
                                         zip(*[summary_data[i][s] for i in component_indexes])])
                    ax.fill_between(hours, t_min, t_max, color=c_map[s], label=r"T$\rm _{{{}}}$".format(f"s, {s}"))

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
    fig.savefig(str(figure_path / 'temperature_at_one_hour.png'))
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
        ax.text(0.925, 0.025, f'({ascii_lowercase[i]})', transform=ax.transAxes)
    for ax in axes[:, 0]:
        ax.set_ylabel('Component index [-]')
    for ax in axes[0, :]:
        ax.set_title(ax.get_title().split(' ')[0])

    axes[0, 0].set_yticks(list(component_indices))
    axes[0, 0].legend()

    fig.suptitle(f"Solar inclination =  {round(degrees(hourly_weather['solar_declination'][hour]), 2)} °")
    fig.tight_layout()
    fig.savefig(str(figure_path / f'temperature_at_{hour}h.png'))
    plt.close()


def plot_radiation_temperature_profiles(hours: list,
                                        hourly_weather: pd.DataFrame,
                                        all_cases_absorbed_irradiance: (dict, irradiance_canopy),
                                        all_cases_temperature_ww: dict,
                                        all_cases_temperature_wd: dict,
                                        figure_path: Path):
    assert all_cases_temperature_ww.keys() == all_cases_absorbed_irradiance[0].keys()

    cases = all_cases_temperature_ww.keys()
    component_indices = all_cases_temperature_ww['layered_lumped'][hours[0]].keys()
    component_indices = [comp_index for comp_index in component_indices if comp_index != -1]
    columns = ['Absorbed_PAR', 'Shaded_fraction', 'Temperature', 'Temperature']

    fig, axes = plt.subplots(ncols=len(columns), nrows=len(hours), sharex='col', sharey='all',
                             figsize=(19 / 2.54, 24 / 2.54))
    for i_hour, hour in enumerate(hours):
        for i_case, case in enumerate(cases):
            plot_incident = i_case == 0

            plot_irradiance_at_one_hour_bis(
                ax=axes[i_hour, 0],
                hour=hour,
                incident_direct=hourly_weather.loc[:, 'incident_direct_irradiance'],
                incident_diffuse=hourly_weather.loc[:, 'incident_diffuse_irradiance'],
                simulation_case=case,
                all_cases_data=all_cases_absorbed_irradiance[0],
                forced_component_indices=component_indices,
                plot_incident=plot_incident,
                plot_soil=plot_incident,
                set_title=False)
            if 'sunlit' in case:
                plot_shaded_fraction(
                    hour=hour,
                    simulation_case=case,
                    ax=axes[i_hour, 1],
                    all_cases_data=all_cases_absorbed_irradiance[1],
                    forced_component_indices=component_indices)
            for i_data, temperature_data in enumerate((all_cases_temperature_ww, all_cases_temperature_wd)):
                plot_temperature_at_one_hour_bis(
                    ax=axes[i_hour, i_data + 2],
                    hour=hour,
                    temperature_air=hourly_weather.loc[:, 'air_temperature'],
                    simulation_case=case,
                    all_cases_data=temperature_data,
                    forced_component_indices=component_indices if 'bigleaf' in case else None,
                    plot_air_temperature=plot_incident,
                    plot_soil_temperature=plot_incident)

    axes[0, 0].set_ylim((-0.25, max(component_indices) + 2))
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_yticks(component_indices + [0])
    axes[2, 0].set_ylabel('Canopy layer index (-)', rotation=90, ha='center')

    axes[-1, 1].xaxis.set_major_locator(MultipleLocator(0.5))
    axes[-1, 1].set_xlim(-0.1, 1.2)

    xlim_temperature = [ax.get_xlim() for ax in axes[-1, 2:]]
    xlim_temperature = [item for sublist in xlim_temperature for item in sublist]
    for ax in axes[-1, 2:]:
        ax.set_xlim(min(xlim_temperature), max(xlim_temperature))

    for i, ax in enumerate(axes.flatten()):
        ax.text(0.05, 0.875, f'({ascii_lowercase[i]})', transform=ax.transAxes)

    for i_hour, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.text(0.7, 0.875, f'{hours[i_hour]:02d}:00', fontsize=9, ha='left', transform=ax.transAxes)
    for ax, col in zip(axes[-1, :3], columns[:3]):
        ax.set_xlabel('\n'.join([col.replace('_', ' '), UNITS_MAP[col.lower()][-1]]))
    axes[-1, 2].xaxis.set_label_coords(1.05, -0.165, transform=axes[-1, 2].transAxes)

    for ax, water_status in zip(axes[0, 2:4], ('Well Watered', 'Water Deficit')):
        ax.set_title(water_status, fontsize=8)

    axes[0, 1].legend(loc='lower left', fontsize=7)

    h_irradiance, l_irradiance = axes[0, 0].get_legend_handles_labels()
    labels_irradiance = ['incident_par_direct', 'incident_par_diffuse',
                         'Bigleaf Lumped', 'Bigleaf Sunlit', 'Bigleaf Shaded',
                         'Layered Lumped', 'Layered Sunlit', 'Layered Shaded',
                         'soil']
    h_irradiance = [h_irradiance[l_irradiance.index(l)] for l in labels_irradiance]
    h_temperature, l_temperature = axes[0, 2].get_legend_handles_labels()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, wspace=0, hspace=0)

    axes[-1, 0].legend(
        handles=[h_temperature[l_temperature.index('temperature_air')]] + h_irradiance + (
            [plt.Line2D([0], [0], color="w")]),
        labels=['Air temperature', 'Incident direct PAR', 'Incident diffuse PAR',
                'Bigleaf Lumped', 'Bigleaf Sunlit', 'Bigleaf Shaded',
                'Layered Lumped', 'Layered Sunlit', 'Layered Shaded',
                'soil', ''],
        bbox_to_anchor=(2.65, -0.6, 1, .102), ncol=4, prop={'size': 8})

    fig.savefig(str(figure_path / f'temperature_profiles.png'), dpi=600)
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
        ax.plot([v - 273.15 for v in x], y, style, label='lumped')
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])

        ax.plot([v - 273.15 for v in x_sun], y, style, color='y', label='sunlit')
        ax.plot([v - 273.15 for v in x_sh], y, style, color='darkgreen', label='shaded')

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


def plot_temperature_at_one_hour_bis(ax: plt.axis,
                                     hour: int,
                                     temperature_air: pd.Series,
                                     simulation_case: str,
                                     all_cases_data: dict,
                                     forced_component_indices: list = None,
                                     plot_air_temperature: bool = True,
                                     plot_soil_temperature: bool = True):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    canopy_class, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    if canopy_class == 'bigleaf':
        kwargs = {'linestyle': '--', 'linewidth': 1}
    else:
        kwargs = {'marker': '.', 'linestyle': 'None'}

    if leaf_class == 'lumped':
        y, x = zip(*[(i, summary_data[i][hour]) for i in component_indexes if i != -1])
        if forced_component_indices:
            y = forced_component_indices
            x = [x[0]] * len(forced_component_indices)
        ax.plot([v - 273.15 for v in x], y, label=f'{canopy_class.capitalize()} Lumped', color='blue', **kwargs)
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])
        if forced_component_indices:
            y = forced_component_indices
            x_sun = [x_sun[0]] * len(forced_component_indices)
            x_sh = [x_sh[0]] * len(forced_component_indices)

        ax.plot([v - 273.15 for v in x_sun], y, color='orange', label=f'{canopy_class.capitalize()} Sunlit', **kwargs)
        ax.plot([v - 273.15 for v in x_sh], y, color='darkgreen', label=f'{canopy_class.capitalize()} Shaded', **kwargs)

    if plot_soil_temperature:
        ax.plot(summary_data[-1][hour] - 273.15, 0, 's', color='brown', label='soil')

    if plot_air_temperature:
        y_text = max(y)
        ax.scatter(temperature_air[hour], y_text + 0.5, marker='$\u2193$', label='temperature_air', color='r')
        # ax.annotate('', xy=(temperature_air[hour], y_text + 0.15),
        #             xytext=(temperature_air[hour], y_text + 1.0),
        #             arrowprops=dict(arrowstyle="->"), ha='center')

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
        ax.plot(summary_data[-1][hour], -1, 's', color='brown', label='soil')

    if plot_incident:
        y_text = max(y)
        for s, (v, c) in ({'PAR_direct': (incident_direct[hour], 'yellow'),
                           'PAR_diffuse': (incident_diffuse[hour], 'darkgreen')}).items():
            ax.scatter(v, y_text + 1, alpha=0)
            ax.annotate('', xy=(v, y_text + 0.25), xytext=(v, y_text + 0.75), arrowprops=dict(arrowstyle="->", color=c))

    ax.set_xlabel(r'$\mathregular{[W_{PAR} \/ m^{-2}_{ground}]}$')
    return


def plot_irradiance_at_one_hour_bis(ax: plt.axis,
                                    hour: int,
                                    incident_direct: pd.Series,
                                    incident_diffuse: pd.Series,
                                    simulation_case: str,
                                    all_cases_data: dict,
                                    forced_component_indices: list = None,
                                    plot_incident: bool = True,
                                    plot_soil: bool = False,
                                    set_title: bool = True):
    summary_data = get_summary_data(simulation_case, all_cases_data)
    canopy_class, leaf_class = simulation_case.split('_')
    component_indexes = summary_data.keys()

    is_bigleaf = canopy_class == 'bigleaf'

    if is_bigleaf:
        kwargs = {'linestyle': '--', 'linewidth': 1}
    else:
        kwargs = {'marker': '.', 'linestyle': 'None'}

    if set_title:
        ax.set_title(handle_sim_name(simulation_case))

    if leaf_class == 'lumped':
        y, x = zip(*[(i, summary_data[i][hour]) for i in component_indexes if i != -1])
        if forced_component_indices and is_bigleaf:
            y = forced_component_indices
            x = [x[0]] * len(forced_component_indices)
        ax.plot(x, y, color='blue', label=f'{canopy_class.capitalize()} Lumped', **kwargs)
    else:
        y, x_sun, x_sh = zip(
            *[(i, summary_data[i]['sunlit'][hour], summary_data[i]['shaded'][hour]) for i in component_indexes if
              i != -1])
        if forced_component_indices and is_bigleaf:
            y = forced_component_indices
            x_sun = [x_sun[0]] * len(forced_component_indices)
            x_sh = [x_sh[0]] * len(forced_component_indices)
        ax.plot(x_sun, y, color='orange', label=f'{canopy_class.capitalize()} Sunlit', **kwargs)
        ax.plot(x_sh, y, color='darkgreen', label=f'{canopy_class.capitalize()} Shaded', **kwargs)

    if plot_soil:
        ax.plot(summary_data[-1][hour], 0, 's', color='brown', label='soil')

    if plot_incident:
        y_text = max(y)
        for s, (v, c) in ({'incident_par_direct': (incident_direct[hour], 'orange'),
                           'incident_par_diffuse': (incident_diffuse[hour], 'darkgreen')}).items():
            ax.scatter(v, y_text + 0.5, marker='$\u2193$', label=s, color=c)
            # ax.annotate('', xy=(v, y_text + 0.15), xytext=(v, y_text + 1.0), arrowprops=dict(arrowstyle="->", color=c))

    return


def plot_shaded_fraction(hour: int, simulation_case: str, ax: plt.axis, all_cases_data: dict,
                         forced_component_indices: list = None):
    canopy_class, leaf_class = simulation_case.split('_')
    if canopy_class == 'bigleaf':
        ax.vlines(all_cases_data[simulation_case][hour][0].shaded_fraction,
                  min(forced_component_indices), max(forced_component_indices),
                  colors='k', linestyles='--', linewidth=1, label='Bigleaf')
    else:
        layers = list(all_cases_data[simulation_case][hour].keys())
        ax.plot([all_cases_data[simulation_case][hour][i].shaded_fraction for i in layers], layers, 'k.',
                label='Layered')
    pass


def plot_canopy_variable(
        all_cases_solver: eb_canopy,
        variable_to_plot: str,
        figure_path: Path,
        y_cumsum: bool = False,
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
        ax.plot(hours, list(cumsum(y_ls)) if y_cumsum else y_ls, label=handle_sim_name(case))
    if return_axes:
        return axes
    else:
        [ax.set_xlabel('hours') for ax in axes[:]]
        axes[0].set_ylabel(f'{UNITS_MAP[variable_to_plot][0]} {UNITS_MAP[variable_to_plot][1]}')
        plt.suptitle(variable_to_plot)
        plt.savefig(str(figure_path / f'{variable_to_plot}.png'))
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
        ax.plot(hours, [h_solver[h].energy_balance for h in hours], 'k:', label=UNITS_MAP[variable_to_plot][0])
    else:
        ax.plot(hours, [getattr(h_solver[h].crop.state_variables, variable_to_plot) for h in hours],
                label=UNITS_MAP[variable_to_plot][0])

    if return_ax:
        return ax
    else:
        ax.set_xlabel('hours')
        fig = ax.get_figure()
        fig.suptitle(variable_to_plot)
        fig.savefig(str(figure_path / f'{variable_to_plot}.png'))
        plt.close()


def plot_energy_balance(solvers: dict, figure_path: Path, plot_iteration_nb: bool = False):
    eb_components = [
        'net_radiation', 'sensible_heat_flux', 'total_penman_monteith_evaporative_energy', 'soil_heat_flux',
        'energy_balance']
    models = solvers.keys()

    if plot_iteration_nb:
        n_rows = 2
        kwargs = {'gridspec_kw': {'height_ratios': [4, 1]}}
    else:
        n_rows = 1
        kwargs = {}
    fig, axes = plt.subplots(ncols=len(models), nrows=n_rows, sharex='all', sharey='row', figsize=(15, 5), **kwargs)
    for model, ax in zip(models, axes[0, :] if plot_iteration_nb else axes):
        for eb_component in eb_components:
            ax = plot_energy_balance_components(h_solver=solvers[model], variable_to_plot=eb_component, ax=ax,
                                                figure_path=figure_path, return_ax=True)
            ax.set_title(handle_sim_name(model))
        ax.grid()
    if plot_iteration_nb:
        for model, ax_it in zip(models, axes[1, :]):
            ax_it.plot([solvers[model][h].iterations_number for h in range(24)], 'k-')
            ax_it.grid()
        axes[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1, 0].set_ylim((0, 1.2 * max(axes[1, 0].get_ylim())))
        axes[1, 0].set_ylabel('iteration\nnumber')
        ax = axes[0, 0]
    else:
        ax = axes[0]
    ax.legend()
    ax.set(ylim=(-275, 1050), ylabel=r'$\mathregular{[W\/m^{-2}_{ground}]}$')

    fig.tight_layout()
    fig.savefig(str(figure_path / 'energy_balance.png'))
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
    fig.savefig(figs_path / 'stability_terms.png')
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
    plt.savefig(str(figure_path / 'universal_functions.png'))
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


def plot_properties_profile(solver_data: dict, hours: list, component_props: [str], multiply_by: [float], xlabels: [],
                            figure_path: Path, add_muliplicaiton_ax: bool = True):
    layers_idx = [k for k in solver_data[hours[0]].crop.keys() if k != -1]
    n_rows = len(hours)
    ncols = len(component_props)
    if add_muliplicaiton_ax:
        ncols += 1
        y_ls = []

    fig, axs = plt.subplots(nrows=n_rows, ncols=ncols, sharey='all', figsize=(7.48, 3))
    for i, hour in enumerate(hours):
        crop = solver_data[hour].crop
        for j, prop in enumerate(component_props):
            ax = axs[i, j] if n_rows > 1 else axs[j]
            if '-' in prop:
                prop1, prop2 = prop.split('-')
                prop_ls1 = [getattr(crop[layer]['sunlit'], prop1) for layer in layers_idx]
                prop_ls2 = [getattr(crop[layer]['sunlit'], prop2) for layer in layers_idx]
                y = [(y1 - y2) for y1, y2 in zip(prop_ls1, prop_ls2)]
                xlabel = f'{UNITS_MAP[prop1][0]} - {UNITS_MAP[prop2][0]} {UNITS_MAP[prop2][1]}'
            else:
                y = [getattr(crop[layer]['sunlit'], prop) for layer in layers_idx]
                xlabel = f'{UNITS_MAP[prop][0]} {UNITS_MAP[prop][1]}'
            y = [v * multiply_by[j] for v in y]
            ax.plot(y, layers_idx)
            if j == 0:
                ax.set_ylabel('Canopy layer index [-]')
                if n_rows > 1:
                    ax.text(0.7, 0.1, f'(hour: {hour})', transform=ax.transAxes)
            if i == n_rows - 1:
                ax.set_xlabel(xlabel if xlabels[j] is None else xlabels[j])
            if add_muliplicaiton_ax:
                y_ls.append(y)

        if add_muliplicaiton_ax:
            ax_multi = axs[i, -1] if n_rows > 1 else axs[-1]
            ax_multi.plot([prod(v) for v in zip(*y_ls)], layers_idx)
            ax_multi.set_xlabel(xlabels[-1])

    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylim(-1 + axs[0].get_ylim()[0], axs[0].get_ylim()[1] + 0.5)
    for i, ax in enumerate(axs):
        ax.text(0.05, 0.9, f'({ascii_lowercase[i]})', transform=ax.transAxes)

    fig.tight_layout()
    plt.savefig(str(figure_path / 'sunlit_props.png'))
    plt.close('all')
    return fig


def compare_energy_balance_terms(s1, s2):
    fig, ax = plt.subplots()
    for s in (s1, s2):
        y, x_ls = zip(*list(s['temperature']['layered_lumped'][12].items()))
        ax.plot([x['lumped'] for x in x_ls], y, label='0.1 m')


def compare_sunlit_shaded_temperatures(temperature_data: list, figure_path: Path, **kwargs):
    fig, ax = plt.subplots()
    x, temp = zip(*temperature_data)
    component_keys = [k for k in temperature_data[0][1].keys() if k != -1]
    upper_layer = []
    lower_layer = []
    for item in temp:
        upper_layer.append(item[max(component_keys)]['sunlit'] - item[max(component_keys)]['shaded'])
        lower_layer.append(item[min(component_keys)]['sunlit'] - item[min(component_keys)]['shaded'])
    for label, y in {'upper_layer': upper_layer, 'lower_layer': lower_layer}.items():
        ax.plot(x, y, label=label)
        ax.set(ylabel=r'$\mathregular{T_{sunlit}-T_{shaded}}$' + f" {UNITS_MAP['source_temperature'][1]}", **kwargs)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close('all')


def plot_surface_conductance_profile(surface_conductance: dict, figure_path: Path = None, ax: plt.Subplot = None,
                                     is_return_ax: bool = False):
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(9 / 2.54, 9 / 2.54))

    c = {'lumped': 'blue', 'sunlit': 'orange', 'shaded': 'DarkGreen', 'sunlit+shaded': 'red'}
    ax.invert_yaxis()
    ax.clear()
    for k, v in surface_conductance.items():
        ax.plot(*zip(*v), label=k, c=c[k])
    x_shaded, _ = zip(*surface_conductance['sunlit'])
    x_sunlit, y = zip(*surface_conductance['shaded'])
    ax.plot([xsun + xshade for xsun, xshade in zip(x_sunlit, x_shaded)], y, label='sunlit+shaded')
    ax.set(xlabel=f"Surface conductance {UNITS_MAP['surface_conductance'][1]}",
           ylabel=f'Cumulative leaf area index {UNITS_MAP["LAI"][1]}')
    handles, labels_ = ax.get_legend_handles_labels()
    labels = ('lumped', 'sunlit', 'shaded', 'sunlit+shaded')
    handles = [handles[labels_.index(s)] for s in labels]
    ax.legend(handles=handles, labels=labels, framealpha=0, handlelength=1)

    if is_return_ax:
        return ax
    else:
        fig.tight_layout()
        fig.savefig(figure_path / 'effect_surface_conductance.png')
        plt.close('all')


def examine_soil_saturation_effect(temperature: list, latent_heat: list, figure_path: Path):
    fig, ax_temperature = plt.subplots()
    ax_latent_heat = ax_temperature.twinx()
    ax_temperature.plot(*zip(*temperature), 'r-', label=UNITS_MAP['source_temperature'][0])
    ax_temperature.set_ylabel(' '.join(UNITS_MAP['source_temperature']))

    ax_latent_heat.plot(*zip(*latent_heat), 'b-', label=UNITS_MAP['total_penman_monteith_evaporative_energy'][0])
    ax_latent_heat.set_ylabel(' '.join(UNITS_MAP['total_penman_monteith_evaporative_energy']))

    h1, l1 = ax_temperature.get_legend_handles_labels()
    h2, l2 = ax_latent_heat.get_legend_handles_labels()
    ax_temperature.legend(h1 + h2, l1 + l2, loc='right')
    ax_temperature.set_xlabel(' '.join(['soil saturation ratio', r'$\mathregular{\frac{\Theta}{\Theta{sat}}}$', '[-]']))

    fig.tight_layout()
    fig.savefig(figure_path / 'effect_soil_saturation_ratio.png')
    plt.close('all')
    pass


def examine_shift_effect(lumped_temperature_ls: list, figure_path: Path):
    cm1 = colors.LinearSegmentedColormap.from_list("RedToBlue", ["r", "b"])
    cnorm = colors.Normalize(vmin=0, vmax=1)
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    component_indices = list(reversed(lumped_temperature_ls[0][1].keys()))
    fig, ax = plt.subplots(figsize=(9 / 2.54, 9 / 2.54))
    for saturation_rate, temp_profile in lumped_temperature_ls:
        ax.plot([temp_profile[k]['lumped'] - 273.15 for k in component_indices], component_indices, 'o-',
                color=cpick.to_rgba(saturation_rate), label=f'{saturation_rate:.2f}')

    ax.set_yticks(component_indices)
    ax.set_yticklabels([v + 1 for v in component_indices])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel=f"Surface temperature {UNITS_MAP['temperature'][1]}",
           ylabel='Canopy layer index (-)')
    ax.legend(title="Actual to saturated\nsoil water content (-)",
              framealpha=0, title_fontsize=8, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.98)
    fig.savefig(figure_path / 'effect_shift.png')
    plt.close('all')
    pass


def examine_lumped_to_sunlit_shaded_resistance_ratio(resistance_data: dict, diffuse_ratio: list,
                                                     figure_path: Path = None, ax: plt.Subplot = None,
                                                     is_return_ax: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9 / 2.54, 9 / 2.54))
    else:
        fig = ax.get_figure()

    for s, v in resistance_data.items():
        ax.plot(diffuse_ratio,
                [v_l / v_ss for (v_l, v_ss) in zip(v['lumped'], v['sunlit-shaded'])], label=s)
    ax.legend(title='\n'.join(["Total leaf area index", r'$\mathregular{(m^2_{leaf}\/m^{-2}_{ground})}$']),
              framealpha=0, title_fontsize=8, fontsize=8)

    ax.set(xlabel='Diffuse to total irradiance ratio (-)',
           ylabel='Lumped to Sunlit-Shaded canopy\nsurface resistance ratio (-)')
    if is_return_ax:
        return ax
    else:
        fig.tight_layout()
        fig.savefig(figure_path / 'effect_surface_resistance.png')

    pass


def evaluate_execution_time(time_data: dict, figure_path: Path):
    cases = list(time_data.keys())
    models = list(time_data[cases[0]].keys())
    run_times = range(len(time_data[cases[0]][models[0]][0]))
    steps = range(len(time_data[cases[0]][models[0]]))

    fig, axs = plt.subplots(nrows=len(cases), ncols=len(models), sharex='all', sharey='all')
    for i, case in enumerate(cases):
        row_axs = axs[i, :]
        for ax, model in zip(row_axs, models):
            ax.grid()
            ax.set_yscale('log')
            med = []
            avg = []
            for run_time in run_times:
                y = [time_data[case][model][step][run_time] for step in steps]
                ax.plot(steps, y, 'lightsteelblue', alpha=0.25)
            for v in time_data[case][model]:
                med.append(median(v))
                avg.append(mean(v))
            ax.plot(*zip(*enumerate(med)), label='median')
            ax.plot(*zip(*enumerate(avg)), label='mean')
    for j, model in enumerate(models):
        axs[0, j].set_title(handle_sim_name(model).replace(' ', '\n'))
        axs[-1, j].set_xlabel('hour')
        axs[-1, j].xaxis.set_major_locator(MultipleLocator(6))

    [ax.set(ylabel=f'{case}\nexecution time [s]') for case, ax in zip(cases, axs[:, 0])]
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figure_path / 'execution_time.png')
    pass


def plot_mixed(data: tuple, figs_path: Path):
    layers, irradiance, irradiance_object, temperature, solver_group, execution_time = data
    models = irradiance.keys()
    eb_components = [
        'net_radiation', 'sensible_heat_flux', 'total_penman_monteith_evaporative_energy', 'soil_heat_flux',
        'energy_balance']
    fig, axs = plt.subplots(ncols=len(models), nrows=3, sharex='all', sharey='row', figsize=(7.48, 4.8),
                            gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [4, 4, 1.5]})
    for i_model, model in enumerate(models):
        for eb_component in eb_components:
            axs[0, i_model] = plot_energy_balance_components(
                h_solver=solver_group[model],
                variable_to_plot=eb_component,
                ax=axs[0, i_model],
                figure_path=Path(),
                return_ax=True)
        plot_temperature_dynamics(
            ax=axs[1, i_model],
            temperature_air=[solver_group[model][h].crop.inputs.air_temperature - 273.15 for h in range(24)],
            simulation_case=model,
            all_cases_data=temperature,
            is_set_title=False,
            is_filled=True)
        axs[-1, i_model].plot([solver_group[model][h].iterations_number for h in range(24)], 'k-')

    for i, ax in enumerate(axs[:2, :].flatten()):
        ax.text(0.05, 0.875, f'({ascii_lowercase[i]})', transform=ax.transAxes)
    for i, ax in enumerate(axs[-1, :]):
        ax.text(0.05, 0.7, f'({ascii_lowercase[i + 8]})', transform=ax.transAxes)
    for model, ax in zip(models, axs[0, :]):
        ax.set_title('\n'.join(handle_sim_name(model).split(' ')))
    axs[0, 0].set_ylabel('\n'.join(['Energy balance term', UNITS_MAP['energy_balance'][1]]), fontsize=10)
    axs[1, 0].set_ylabel('\n'.join(['Temperature', UNITS_MAP['temperature'][1]]), fontsize=10)
    axs[1, 0].set_ylim([axs[1, 0].get_ylim()[i] + y for i, y in zip([0, -1], [-10, 0])])
    axs[2, 0].set_ylabel('Number of\niterations', fontsize=10)
    axs[2, 1].set_xlabel('Hour of the day')
    axs[2, 1].xaxis.set_label_coords(1, -0.5, transform=axs[2, 1].transAxes)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    l0 = ['balance'] + [l for l in labels if l != 'balance']
    h0 = [handles[labels.index(s)] for s in l0]
    axs[0, 0].legend(handles=h0, labels=l0, loc='center left', fontsize=8, handlelength=0.75, ncol=1, framealpha=0)

    for ax in axs[1, :]:
        ax.legend(loc='lower right', fontsize=8, handlelength=1.5, ncol=1, framealpha=0)

    for ax in axs[1:, 0]:
        ax.yaxis.set_label_coords(-0.325, 0.5, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(figs_path / 'mixed.png')
    plt.close("all")
    pass


def plot_resistance2(data: tuple, figs_path: Path):
    fig, (ax_cond, ax_res) = plt.subplots(ncols=2, figsize=(7.48, 3.5))

    ax_cond = plot_surface_conductance_profile(surface_conductance=data[0], ax=ax_cond, is_return_ax=True)
    ax_res = examine_lumped_to_sunlit_shaded_resistance_ratio(resistance_data=data[1][0], diffuse_ratio=data[1][1],
                                                              ax=ax_res, is_return_ax=True)
    ax_cond.invert_yaxis()
    ax_cond.legend(framealpha=0, title_fontsize=8, fontsize=8)

    for i, ax in enumerate((ax_cond, ax_res)):
        ax.text(0.85, 0.65, f'({ascii_lowercase[i]})', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(figs_path / 'effect_surface_resistance2.png')

    pass


def plot_energy_balance_terms_for_ww_and_wd(data: dict, figs_path: Path):
    models = ['bigleaf_lumped', 'bigleaf_sunlit-shaded', 'layered_lumped', 'layered_sunlit-shaded']
    eb_components = [
        'net_radiation', 'sensible_heat_flux', 'total_penman_monteith_evaporative_energy', 'soil_heat_flux',
        'energy_balance']
    fig, axs = plt.subplots(ncols=4, nrows=2, sharex='all', sharey='all', figsize=(7.48, 4.8),
                            gridspec_kw={'wspace': 0, 'hspace': 0})
    for i_water, water_status in enumerate(('ww', 'wd')):
        layers, irradiance, irradiance_object, temperature, solver_group, execution_time = data[water_status]
        for i_model, model in enumerate(models):
            for eb_component in eb_components:
                axs[i_water, i_model] = plot_energy_balance_components(
                    h_solver=solver_group[model],
                    variable_to_plot=eb_component,
                    ax=axs[i_water, i_model],
                    figure_path=Path(),
                    return_ax=True)

    for i, ax in enumerate(axs.flatten()):
        ax.text(0.05, 0.875, f'({ascii_lowercase[i]})', transform=ax.transAxes)
    for model, ax in zip(models, axs[0, :]):
        ax.set_title('\n'.join(handle_sim_name(model).split(' ')))
    axs[0, 0].set_ylabel('\n'.join(['Energy balance term', UNITS_MAP['energy_balance'][1]]), fontsize=10)
    axs[1, 1].set_xlabel('Hour of the day')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    l0 = ['balance'] + [l for l in labels if l != 'balance']
    h0 = [handles[labels.index(s)] for s in l0]
    axs[0, 0].legend(handles=h0, labels=l0, loc='center left', fontsize=8, handlelength=0.75, ncol=1, framealpha=0)

    axs[0, 0].yaxis.set_label_coords(-0.3, 0, transform=axs[0, 0].transAxes)
    axs[1, 1].xaxis.set_label_coords(1, -0.2, transform=axs[1, 1].transAxes)

    fig.tight_layout()
    fig.savefig(figs_path / 'mixed_ww_ws.png')
    plt.close("all")

    pass
