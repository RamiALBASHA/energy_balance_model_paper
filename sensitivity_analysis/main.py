import re
from json import load, dump
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
from SALib.analyze import fast
from SALib.sample import fast_sampler
from crop_energy_balance.solver import Solver
from matplotlib import colors
from numpy import array, arange

from coherence.sim import calc_absorbed_irradiance, get_energy_balance_inputs_and_params
from sources.demo import get_grignon_weather_data
from utils.config import UNITS_MAP
from utils.van_genuchten_params import VanGenuchtenParams
from utils.water_retention import calc_soil_water_potential

MAP_PARAMS = {
    'd_0': r'$\mathregular{D_o}$',
    'psi_half_aperture': r'$\mathregular{\psi_{50}}$',
    'steepness': r'$\mathregular{\beta}$',
    'soil_aerodynamic_resistance_shape_parameter': r'$\mathregular{\alpha_w}$',
    'soil_roughness_length_for_momentum': r'$\mathregular{z_{0,\/u}}}$',
    'leaf_characteristic_length': r'$\mathregular{w}$',
    'leaf_boundary_layer_shape_parameter': r'$\mathregular{\alpha}$',
    'wind_speed_extinction_coef': r'$\mathregular{k_u}$',
    'maximum_stomatal_conductance': r'$\mathregular{g_{s,\/max}}$',
    'residual_stomatal_conductance': r'$\mathregular{g_{s,\/res}}$',
    'diffuse_extinction_coef': r'$\mathregular{k_{diffuse}}$',
    'leaf_scattering_coefficient': r'$\mathregular{\sigma_s}$',
    'absorbed_par_50': r'$\mathregular{R_{PAR,\/50}}$',
    'soil_resistance_to_vapor_shape_parameter_1': r'$\mathregular{a_s}$',
    'soil_resistance_to_vapor_shape_parameter_2': r'$\mathregular{b_s}$',
    'grignon_high_rad_high_vpd.csv': 'HH',
    'grignon_high_rad_low_vpd.csv': 'HL',
    'grignon_low_rad_high_vpd.csv': 'LH',
    'grignon_low_rad_low_vpd.csv': 'LL',
    'source_temperature': r'$\mathregular{T_m}$',
    'drag_coefficient': r'$\mathregular{C_d}$',
    'ratio_heat_to_momentum_canopy_roughness_lengths': r'$\mathregular{\xi}$',
    'richardon_threshold_free_convection': r'$\mathregular{{Ri}_{free}}$',
    'free_convection_shape_parameter': r'$\mathregular{\eta}$'
}


def set_bounds(nominal_mean: float, interval_percentage: float) -> list:
    return sorted(nominal_mean * (1 + p / 100.) for p in (-interval_percentage, interval_percentage))


def eb_wrapper(leaves_category: str, inputs: dict, params: dict) -> Solver:
    solver = Solver(leaves_category=leaves_category,
                    inputs_dict=inputs,
                    params_dict=params)
    solver.run(is_stability_considered=True)
    return solver


def get_name_bound(param_fields: dict) -> tuple:
    interval_percent = 20
    names, bounds = [], []
    for k, v in param_fields.items():
        if k != 'stomatal_sensibility':
            names.append(k)
            bounds.append(set_bounds(nominal_mean=v, interval_percentage=interval_percent))
        else:
            for model, param_dict in v.items():
                for param_name, param_value in param_dict.items():
                    names.append('-'.join([k, model, param_name]))
                    bounds.append(set_bounds(nominal_mean=param_value, interval_percentage=interval_percent))
    return names, bounds


def sample(config_path: Path) -> (dict, array):
    with open(config_path, mode='r') as conf_file:
        param_fields = load(conf_file)
    names, bounds = get_name_bound(param_fields=param_fields)
    sa_problem = {
        'num_vars': len(names),
        'names': list(names),
        'bounds': list(bounds)
    }
    return sa_problem, fast_sampler.sample(problem=sa_problem, N=2 ** 10)


def evaluate(leaves_category: str, inputs: dict, params: dict, names: list, scenarios: array, output_variables: list):
    res = {k: [] for k in output_variables}
    for i, scenario in enumerate(scenarios):
        for param_name, param_value in zip(names, scenario):
            if 'stomatal_sensibility' in param_name:
                param_keys = param_name.split('-')
                params[param_keys[0]][param_keys[1]][param_keys[2]] = param_value
            else:
                params.update({param_name: param_value})
        solver = eb_wrapper(leaves_category=leaves_category, inputs=inputs, params=params)
        for k in output_variables:
            res[k].append(eval(f'solver.{k}'))
        print(i)
    return {k: array(v) for k, v in res.items()}


def analyze(sa_problem: dict, outputs: array) -> dict:
    return {k: fast.analyze(sa_problem, v, M=4, num_resamples=100, conf_level=0.95, print_to_console=False, seed=None)
            for k, v in outputs.items()}


def plot_barh(sa_dict: {str, dict}, path_fig: Path, model: str, range_values: bool = False,
              parameter_groups: dict = None,
              shift_bars: bool = False, suptitle: str = None):
    fig_kwargs = {'sharex': True, 'sharey': True} if not range_values else {}

    fig, axs = plt.subplots(ncols=len(sa_dict.keys()), **fig_kwargs)
    bar_height = 0.4

    for ax, (k, sa) in zip(axs, sa_dict.items()):
        plot_single(ax, bar_height, k, sa, shift_bars=shift_bars, parameter_groups=parameter_groups)
    for ax in axs[1:]:
        ax.get_yaxis().set_visible(False)

    axs[0].legend()
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path_fig / f'{model}_{suptitle}.png')
    plt.close()
    pass


def plot_single(ax, bar_height, k, sa, range_by_col=None, parameter_groups=None, shift_bars=True):
    sa_df = sa.to_df()
    if range_by_col is not None:
        sa_df.sort_values(by=range_by_col, inplace=True)
    if parameter_groups is not None:
        p_order = [item for sublist in parameter_groups.values() for item in sublist]
        sa_df = sa_df.loc[reversed(p_order), :]
    y_pos = arange(len(sa['names']))
    ax.clear()
    if shift_bars:
        for effect, shift in (('S1', bar_height / 2), ('ST', -bar_height / 2)):
            ax.barh(y_pos + shift, sa_df[effect].values, height=bar_height, label=effect,
                    xerr=sa_df[f'{effect}_conf'].values,
                    error_kw=dict(ecolor='gray', lw=0.5, capsize=2, capthick=0.5))
    else:
        ax.barh(y_pos, sa_df['S1'].values, height=bar_height, label='S1')
        ax.barh(y_pos, sa_df['ST'].values, height=bar_height, label='ST', left=sa_df['S1'].values)

    ax_title = handle_var_name(k)
    ax.set(yticks=y_pos + 0.2, yticklabels=[MAP_PARAMS[s] for s in sa_df.index], title=ax_title)
    pass


def handle_var_name(k, model: str = None, is_symbol: bool = True):
    is_layered = 'layered' in model if model is not None else None

    if 'state_variables' in k:
        if is_symbol:
            try:
                ax_title = UNITS_MAP[k.split('.')[-1]][0]
            except KeyError:
                ax_title = MAP_PARAMS[k.split('.')[-1]]
        else:
            ax_title = 'Latent heat flux'

    else:
        leaf_layer = re.search('\[(.+?)\]', k).group(1)
        try:
            leaf_category = re.search("'(.+?)'", k).group(1)
        except AttributeError:
            leaf_category = 'lumped'
        if is_symbol:
            ax_title = r'$\mathregular{T_{s,\/%s,\/%s}}$' % (leaf_category, 'l' if leaf_layer == 0 else 'u')
        else:
            ax_title = f'{leaf_category.capitalize()} leaf temperature'
            if is_layered:
                ax_title = '\n'.join([ax_title, '(lower layer)' if int(leaf_layer) == 0 else '(upper layer)'])
    return ax_title


def plot_heatmap(sa_dict: dict, model: str, fig_path: Path, parameter_groups: dict = None, name_info: str = ''):
    if parameter_groups is not None:
        params_order = [item for sublist in parameter_groups.values() for item in sublist]
    else:
        params_order = None

    s1, st, infos = build_heatmap_arrays(sa_dict=sa_dict, params_order=params_order)
    for s, data in (('s1', s1), ('st', st)):
        heatmap(data=data, parameter_groups=parameter_groups, infos=infos,
                path_fig=fig_path / f'{name_info}_{model}_{s}.png')
    pass


def heatmap(data: array, infos: dict, parameter_groups: dict, model: str = None, water_status: str = None,
            ax: plt.Subplot = None, path_fig: Path = None, is_symbol: bool = True,
            is_colorbar: bool = False, is_text_header: bool = True, is_text_groups: bool = True,
            is_return_ax: bool = False, is_return_mappable: bool = False):
    names_output_vars = infos['names_output_vars']

    number_environments = len(infos['names_environment'])
    number_output_variables = len(names_output_vars)

    col_labels = infos['names_environment'] * number_output_variables
    row_labels = infos['names_params']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
    else:
        fig = ax.get_figure()

    norm = colors.Normalize(0, vmax=1)
    im = ax.imshow(data, norm=norm, cmap='Oranges', aspect='auto')

    if is_colorbar:
        ax.figure.colorbar(im, ax=ax, orientation="horizontal")
        #    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    if is_text_header:
        ax.set(xticks=arange(data.shape[1]), xticklabels=col_labels)
    if is_text_groups:
        ax.set(yticks=arange(data.shape[0]), yticklabels=[MAP_PARAMS[s.split('-')[-1]] for s in row_labels])

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=8)
    for j_out, output_var in enumerate(names_output_vars):
        ax.vlines(j_out * number_environments - 0.5, *ax.get_ylim(), color='k', linewidth=1)
        if is_text_header:
            ax.text(j_out * number_environments + number_environments / 2 - 1.0, -5,
                    handle_var_name(output_var, model=model, is_symbol=is_symbol),
                    fontdict=dict(size=8, ha='center' if is_symbol else 'left', rotation=90))
    if parameter_groups is not None:
        len_label_max = max([len(MAP_PARAMS[s.split('-')[-1]]) for s in row_labels])
        group_row_index = -0.5
        for group_name, params in parameter_groups.items():
            group_row_index += len(params) / 2
            ax.hlines(row_labels.index(params[-1]) + 0.5, *ax.get_xlim(), color='k', linewidth=1)
            if is_text_groups:
                x_pos = max([len(MAP_PARAMS[s.split('-')[-1]]) for s in params]) / len_label_max * -6 - 0.5
                ax.annotate(group_name, xytext=(-7.9, group_row_index), xy=(x_pos, group_row_index),
                            arrowprops=dict(arrowstyle=f'-[, widthB={len(params) / 2 - 0.5}', lw=0.5),
                            annotation_clip=False,
                            fontsize=8, ha='right', va='center')
                ax.text(-3.45, 0.5, water_status, transform=ax.transAxes,
                        fontdict={'size': 8, 'ha': 'center', 'va': 'center', 'rotation': 90})
            group_row_index += len(params) / 2

    if path_fig is not None:
        fig.savefig(path_fig)

    if is_return_ax:
        if is_return_mappable:
            return ax, im
        else:
            return ax
    pass


def build_heatmap_arrays(sa_dict: dict, params_order: list = None) -> (array, array, dict):
    names_environment = list(sa_dict.keys())
    names_output_vars = list(sa_dict[names_environment[0]].keys())
    names_params = sa_dict[names_environment[0]][names_output_vars[0]]['names']
    number_environments = len(names_environment)
    number_output_vars = len(names_output_vars)
    number_parameters = len(names_params)
    s1_array = array([[None] * number_output_vars * number_environments] * number_parameters)
    st_array = s1_array.copy()
    for index_row, param in enumerate(names_params):
        for j_out, output_var in enumerate(names_output_vars):
            for j_env, env_name in enumerate(names_environment):
                index_col = j_out * number_environments + j_env
                s1_array[index_row, index_col] = sa_dict[env_name][output_var]['S1'][index_row]
                st_array[index_row, index_col] = sa_dict[env_name][output_var]['ST'][index_row]

    if params_order is not None:
        index_row_ordered = [names_params.index(s) for s in params_order]
        s1_array = s1_array[index_row_ordered, :]
        st_array = st_array[index_row_ordered, :]

    infos = {
        'names_environment': names_environment,
        'names_output_vars': names_output_vars,
        'names_params': names_params if params_order is None else params_order}

    return s1_array.astype(float), st_array.astype(float), infos


def set_output_variables(is_bigleaf: bool, is_lumped: bool, layers: dict) -> list:
    # res = ['crop.state_variables.total_penman_monteith_evaporative_energy',
    #        'crop.state_variables.source_temperature']
    res = ['crop.state_variables.total_penman_monteith_evaporative_energy']

    layer_indices = [0] if is_bigleaf else [max(layers.keys()), min(layers.keys())]
    leaf_classes = [''] if is_lumped else ["['sunlit']", "['shaded']"]
    for layer_index in layer_indices:
        for leaf_class in leaf_classes:
            res.append(f"crop[{layer_index}]{leaf_class}.temperature")

    return res


def run_sensitivity_analysis(veg_layers: dict, canopy_type: str, leaf_type: str, base_sa_inputs: dict,
                             base_sa_params: dict, sa_problem: dict, param_scenarios: array, parameter_groups: dict,
                             saturation_ratio: float, is_plot_heatmap: bool = True, is_return: bool = True):
    soil_class = 'Silt'
    _, theta_sat, *_ = getattr(VanGenuchtenParams, soil_class).value

    if canopy_type == 'bigleaf':
        veg_layers = {0: sum(veg_layers.values())}

    weather_scenarios = (('grignon_high_rad_high_vpd.csv', 14),
                         ('grignon_low_rad_high_vpd.csv', 14),
                         ('grignon_high_rad_low_vpd.csv', 11),
                         ('grignon_low_rad_low_vpd.csv', 14))

    sa_dict = {}
    for weather_scenario, hour in weather_scenarios:
        weather_data = get_grignon_weather_data(weather_scenario).loc[hour, :]
        absorbed_irradiance, _ = calc_absorbed_irradiance(
            leaf_layers=veg_layers,
            is_bigleaf=canopy_type == 'bigleaf',
            is_lumped=leaf_type == 'lumped',
            incident_direct_par_irradiance=weather_data['incident_direct_irradiance'],
            incident_diffuse_par_irradiance=weather_data['incident_diffuse_irradiance'],
            solar_inclination_angle=weather_data['solar_declination'])

        base_sa_inputs, _ = get_energy_balance_inputs_and_params(
            vegetative_layers=veg_layers,
            absorbed_par_irradiance=absorbed_irradiance,
            actual_weather_data=weather_data,
            raw_inputs=base_sa_inputs,
            json_params=base_sa_params)

        base_sa_inputs.update({
            "soil_saturation_ratio": saturation_ratio,
            "soil_water_potential": calc_soil_water_potential(
                theta=saturation_ratio * theta_sat, soil_class=soil_class) * 1.e-4})

        outputs = evaluate(
            leaves_category=leaf_type,
            inputs=base_sa_inputs,
            params=base_sa_params,
            names=sa_problem['names'],
            scenarios=param_scenarios,
            output_variables=set_output_variables(
                is_bigleaf=canopy_type == 'bigleaf',
                is_lumped=leaf_type == 'lumped',
                layers=veg_layers))
        sa_result = analyze(sa_problem=sa_problem, outputs=outputs)
        sa_dict.update({MAP_PARAMS[weather_scenario]: sa_result})

        # plot_barh(sa_dict=sa_result, shift_bars=False, model=f'{canopy_type}_{leaf_type}',
        #           parameter_groups=parameter_groups, suptitle=weather_scenario.split('.')[0], path_fig=path_figs)
    if is_plot_heatmap:
        plot_heatmap(sa_dict=sa_dict, fig_path=path_figs, model=f'{canopy_type}_{leaf_type}',
                     parameter_groups=parameter_groups, name_info=f'soil_sat_ratio_{saturation_ratio}')
    if is_return:
        return sa_dict


def plot_grouped_heatmap(sa_data: dict, path_fig: Path, parameter_groups: dict = None):
    params_order = [item for sublist in parameter_groups.values() for item in sublist] if parameter_groups else None
    models = list(sa_data.keys())
    water_conditions = sa_data[models[0]].keys()
    for is_first_order in (True, False):
        im = None
        plt.close('all')
        fig, axs = plt.subplots(nrows=len(water_conditions), ncols=len(models), figsize=(19 / 2.54, 22 / 2.54),
                                gridspec_kw=dict(hspace=0.025, wspace=0.025, width_ratios=[2, 3, 3, 5]))
        for i_model, model in enumerate(models):
            for i_soil_status, soil_status in enumerate(water_conditions):
                ax = axs[i_soil_status, i_model]
                sa_dict = sa_data[model][soil_status]
                s1, st, infos = build_heatmap_arrays(sa_dict=sa_dict, params_order=params_order)
                ax, im = heatmap(data=s1 if is_first_order else st,
                                 infos=infos,
                                 parameter_groups=parameter_groups,
                                 ax=ax,
                                 model=model,
                                 water_status=soil_status,
                                 is_symbol=False,
                                 is_text_header=i_soil_status == 0,
                                 is_text_groups=i_model == 0,
                                 is_return_ax=True,
                                 is_return_mappable=True)
        for ax in axs[:, 1:].flatten():
            ax.yaxis.set_visible(False)
        for ax in axs[1:, :].flatten():
            ax.xaxis.set_visible(False)
        for ax in axs[0, :]:
            ax.tick_params(axis='x', rotation=90)
        for ax, model in zip(axs[0, :], models):
            ax.annotate(text='\n'.join([s.capitalize().replace('-s', '-S') for s in model.split('_')]),
                        xytext=(0.5, 2.15), xy=(0.5, 2.),
                        xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle=f'-[, widthB={len(ax.get_xticklabels()) / 2 - 2}', lw=0.5),
                        annotation_clip=False,
                        fontsize=8, ha='center', va='top', fontweight='bold')

        for ax, s in zip(axs[:, 0], ascii_lowercase):
            ax.text(-3.25, 0.9, f'({s})', transform=ax.transAxes,
                    fontdict={'size': 10, 'ha': 'center', 'va': 'center', 'weight': 'bold'})
        cbar_ax = fig.add_axes([0.05, 0.9, 0.15, 0.01])
        cbar = fig.colorbar(im, cax=cbar_ax, label='First order effect (S1)' if is_first_order else 'Total effect (ST)',
                            orientation='horizontal')

        fig.subplots_adjust(left=0.35, right=0.99, bottom=0.01, top=0.70)
        fig.savefig(path_fig / f'sensitivity_analysis_{"s1" if is_first_order else "st"}.png')
        plt.close("all")

    pass


if __name__ == '__main__':
    path_root = Path(__file__).parent
    path_sources = path_root.parent / 'sources/sensitivity_analysis'
    path_figs = path_sources / 'figs'
    path_figs.mkdir(parents=True, exist_ok=True)

    with open(path_sources / 'base_inputs.json', mode='r') as f:
        base_inputs = load(f)
    with open(path_sources / 'base_params.json', mode='r') as f:
        base_params = load(f)

    param_groups = {
        'Leaf surface resistance': ['residual_stomatal_conductance',
                                    'maximum_stomatal_conductance',
                                    'stomatal_sensibility-leuning-d_0',
                                    'stomatal_sensibility-misson-psi_half_aperture',
                                    'stomatal_sensibility-misson-steepness',
                                    'absorbed_par_50'],
        'Leaf boundary-layer resistance': ['leaf_boundary_layer_shape_parameter',
                                           'leaf_characteristic_length',
                                           'wind_speed_extinction_coef'],
        'Soil aerodynamic resistance': ['soil_aerodynamic_resistance_shape_parameter',
                                        'soil_roughness_length_for_momentum'],
        'Soil surface resistance': ['soil_resistance_to_vapor_shape_parameter_1',
                                    'soil_resistance_to_vapor_shape_parameter_2'],
        'Canopy aerodynamic resistance': ['drag_coefficient',
                                          'ratio_heat_to_momentum_canopy_roughness_lengths',
                                          'richardon_threshold_free_convection',
                                          'free_convection_shape_parameter']
    }

    problem, param_values = sample(config_path=path_sources / 'param_fields.json')

    leaf_layers = {str(d): 1 for d in range(4)}
    saturation_ratios = {'Well-watered': 1, 'Mild water deficit': 0.3, 'Severe water deficit': 0.1}

    sa_result_all = {}
    for category_canopy, category_leaf in ((('bigleaf', 'lumped'),
                                            ('bigleaf', 'sunlit-shaded'),
                                            ('layered', 'lumped'),
                                            ('layered', 'sunlit-shaded'))):
        model_representation = f'{category_canopy}_{category_leaf}'
        sa_result_all.update({model_representation: {}})
        for soil_water_status, soil_saturation_ratio in saturation_ratios.items():
            run_result = run_sensitivity_analysis(
                veg_layers=leaf_layers,
                canopy_type=category_canopy,
                leaf_type=category_leaf,
                base_sa_inputs=base_inputs,
                base_sa_params=base_params,
                sa_problem=problem,
                param_scenarios=param_values,
                parameter_groups=param_groups,
                saturation_ratio=soil_saturation_ratio,
                is_plot_heatmap=False,
                is_return=True)

            sa_result_all[model_representation].update({soil_water_status: run_result})

    with open(path_figs / 'sensitivity_analysis_summary.json', mode='w') as f:
        dump(sa_result_all, f)
    plot_grouped_heatmap(sa_data=sa_result_all, path_fig=path_figs, parameter_groups=param_groups)
